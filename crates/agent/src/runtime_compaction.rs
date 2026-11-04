use std::{sync::Arc, time::Duration};

use agent_runtime::{
    EmbeddedRuntime, RuntimeStore,
    resource::{ResourceKind, RuntimeResources},
};
use agent_settings::{AgentSettings, COMPACTION_PROMPT};
use anyhow::{Context as _, Result, anyhow, bail, ensure};
use futures::{FutureExt as _, StreamExt as _, future::BoxFuture};
use gpui::{App, AsyncApp, Task};
use language_model::{
    CompactedContext, CompletionIntent, LanguageModel, LanguageModelCompletionEvent,
    LanguageModelProvider, LanguageModelRegistry, LanguageModelRequest,
    LanguageModelRequestMessage, LanguageModelToolChoice, MessageContent, Role, StopReason,
};
use settings::Settings as _;

use crate::NativeRuntimeConfig;

const MAXIMUM_PAGE_BYTES: usize = 32 * 1024 * 1024;
const MAXIMUM_SUMMARY_BYTES: usize = 1024 * 1024;

#[derive(Clone, Copy)]
pub enum CompactionMode {
    Manual,
    Automatic { generation: i64 },
}

pub fn compact_agent(
    runtime: EmbeddedRuntime,
    agent_id: String,
    resources: Option<RuntimeResources>,
    mode: CompactionMode,
    cx: &mut App,
) -> Task<Result<()>> {
    compact_with_configuration(runtime, agent_id, resources, mode, None, cx)
}

pub(crate) fn compact_with_configuration(
    runtime: EmbeddedRuntime,
    agent_id: String,
    resources: Option<RuntimeResources>,
    mode: CompactionMode,
    configuration_override: Option<NativeRuntimeConfig>,
    cx: &mut App,
) -> Task<Result<()>> {
    cx.spawn(async move |cx| {
        let store = runtime.store.clone();
        let generation = match mode {
            CompactionMode::Manual => store.begin_compaction(agent_id.clone()).await?,
            CompactionMode::Automatic { generation } => generation,
        };
        let result = async {
        let snapshot = store.compaction_snapshot(agent_id.clone(), generation).await?;
        if snapshot.cutoff == 0 && !snapshot.overflow && snapshot.request.messages.is_empty() { return Ok(()); }
        let agent = store.snapshot(agent_id.clone()).await?;
        let mut configuration: NativeRuntimeConfig = serde_json::from_value(store.metadata(agent.root_id.clone()).await?)?;
        let child_metadata = if agent.parent_id.is_some() { Some(store.metadata(agent.id.clone()).await?) } else { None };
        let explicit_child = child_metadata.as_ref().and_then(|metadata| metadata.get("model")).filter(|value| !value.is_null());
        if let Some(model) = explicit_child { configuration.model = Some(serde_json::from_value(model.clone())?); }
        if let Some(profile) = child_metadata.as_ref().and_then(|metadata| metadata.get("profile")).filter(|value| !value.is_null()) { configuration.profile = Some(serde_json::from_value(profile.clone())?); }
        if agent.parent_id.is_some() && explicit_child.is_none() {
            let subagent_model = cx.update(|cx| AgentSettings::get_global(cx).subagent_model.clone());
            if subagent_model.is_some() { configuration.model = subagent_model; }
        }
        if let Some(override_configuration) = configuration_override { configuration.system_prompt = override_configuration.system_prompt; }
        let (model, provider, template) = cx.update(|cx| prepare(&configuration, cx))?;
        if model.supports_explicit_compaction() && !snapshot.overflow {
            let mut request = template.clone();
            request.messages.extend(snapshot.request.messages);
            let bytes = serde_json::to_vec(&request)?.len();
            let execution = provider.compact(&model, request, cx);
            let result = execute_fenced(&runtime, store.clone(), agent_id.clone(), generation, resources.clone(), bytes,
                async move { Ok(execution.await?) }.boxed()).await?;
            store.install_compaction(agent_id.clone(), generation, snapshot.cutoff, result.context).await?;
            return Ok(());
        }

        // The fallback pages the archive independently; retain no second full decoded copy.
        drop(snapshot.request);
        let chunk_bytes = usize::try_from(model.max_input_tokens().saturating_mul(2))?
            .clamp(4096, 2 * 1024 * 1024);
        let mut cursor = None;
        let mut transcript = String::new();
        let mut summary = String::new();
        let mut native_context = None;
        loop {
            let page = store.compaction_page(agent_id.clone(), generation, cursor, MAXIMUM_PAGE_BYTES).await?;
            ensure!(page.cutoff == snapshot.cutoff, "conversation changed during compaction");
            for message in page.messages {
                for part in transcript_parts(message, &mut native_context)? {
                    let part = match part {
                        TranscriptPart::Image(image) => {
                            ensure!(model.supports_images(), "the configured compaction model cannot read archived images; select an image-capable compaction model");
                            summary = summarize(&runtime, &store, &agent_id, generation, &resources, &model, &provider, &template, &summary, &transcript, native_context.take(), Some(image), cx).await?;
                            transcript.clear();
                            continue;
                        }
                        TranscriptPart::Text(text) => text,
                    };
                    let mut remaining = part.as_str();
                    while !remaining.is_empty() {
                        let available = chunk_bytes.saturating_sub(transcript.len());
                        if available == 0 {
                            summary = summarize(&runtime, &store, &agent_id, generation, &resources, &model, &provider, &template, &summary, &transcript, native_context.take(), None, cx).await?;
                            transcript.clear();
                            continue;
                        }
                        let mut boundary = available.min(remaining.len());
                        while boundary > 0 && !remaining.is_char_boundary(boundary) { boundary -= 1; }
                        if boundary == 0 {
                            summary = summarize(&runtime, &store, &agent_id, generation, &resources, &model, &provider, &template, &summary, &transcript, native_context.take(), None, cx).await?;
                            transcript.clear();
                            continue;
                        }
                        transcript.push_str(&remaining[..boundary]);
                        remaining = &remaining[boundary..];
                    }
                }
            }
            cursor = page.next_cursor;
            if cursor.is_none() { break; }
        }
        if !transcript.is_empty() || summary.is_empty() {
            summary = summarize(&runtime, &store, &agent_id, generation, &resources, &model, &provider, &template, &summary, &transcript, native_context, None, cx).await?;
        }
        ensure!(!summary.trim().is_empty(), "compaction returned an empty summary");
        store.install_compaction(agent_id.clone(), generation, snapshot.cutoff, CompactedContext::Summary { content: summary.into(), provider_state: None }).await
        }.await;
        if let Err(error) = store.abort_compaction(agent_id, generation).await {
            if result.is_ok() { return Err(error); }
            log::error!("compaction cleanup failed: {error:#}");
        }
        result
    })
}

fn selected_model(configuration: &NativeRuntimeConfig, cx: &App) -> Result<LanguageModel> {
    let settings = AgentSettings::get_global(cx);
    let profile_id = configuration
        .profile
        .as_ref()
        .unwrap_or(&settings.default_profile);
    let profile = settings
        .profiles
        .get(profile_id)
        .context("compaction profile is unavailable")?;
    match configuration
        .model
        .as_ref()
        .or(profile.default_model.as_ref())
    {
        Some(selection) => crate::native_runtime::resolve_model_id(
            &format!("{}/{}", selection.provider.0, selection.model),
            cx,
        ),
        None => LanguageModelRegistry::read_global(cx)
            .default_model()
            .context("select a model before compacting a conversation"),
    }
}

pub fn maybe_compact_agent(
    runtime: EmbeddedRuntime,
    agent_id: String,
    generation: i64,
    resources: Option<RuntimeResources>,
    configuration: NativeRuntimeConfig,
    cx: &mut App,
) -> Task<Result<()>> {
    let policy = AgentSettings::get_global(cx).auto_compact;
    if !policy.enabled {
        return Task::ready(Ok(()));
    }
    let model = match selected_model(&configuration, cx) {
        Ok(model) => model,
        Err(error) => return Task::ready(Err(error)),
    };
    let capacity = crate::thread::compaction_input_capacity(
        model.max_input_tokens(),
        model.max_total_tokens(),
        model.max_output_tokens(),
    );
    let threshold = crate::thread::auto_compact_threshold_token_count(policy.threshold, capacity);
    cx.spawn(async move |cx| {
        let bytes = runtime
            .store
            .effective_history_bytes(agent_id.clone())
            .await?;
        let usage = runtime.store.last_provider_usage(agent_id.clone()).await?;
        let used = usage.map(|usage| usage.usage.total_tokens()).unwrap_or(0);
        let overflow = runtime
            .store
            .provider_context_overflow(agent_id.clone())
            .await?;
        if !overflow
            && bytes < agent_runtime::history_budget::DEFAULT_HISTORY_BYTES
            && used < threshold
        {
            return Ok(());
        }
        cx.update(|cx| {
            compact_with_configuration(
                runtime,
                agent_id,
                resources,
                CompactionMode::Automatic { generation },
                Some(configuration),
                cx,
            )
        })
        .await
    })
}

fn prepare(
    configuration: &NativeRuntimeConfig,
    cx: &App,
) -> Result<(
    LanguageModel,
    Arc<dyn LanguageModelProvider>,
    LanguageModelRequest,
)> {
    let settings = AgentSettings::get_global(cx);
    ensure!(
        settings.enabled(cx),
        "AI features are disabled in Zed settings"
    );
    let registry = LanguageModelRegistry::read_global(cx);
    let model = match registry.compaction_model() {
        Some(model) => model,
        None => selected_model(configuration, cx)?,
    };
    let provider = registry.provider_for_model(&model)?;
    let template = LanguageModelRequest {
        intent: Some(CompletionIntent::ThreadContextSummarization),
        messages: vec![LanguageModelRequestMessage {
            role: Role::System,
            content: vec![MessageContent::Text(configuration.system_prompt.clone())],
            cache: false,
            reasoning_details: None,
        }],
        tools: Vec::new(),
        tool_choice: Some(LanguageModelToolChoice::None),
        temperature: AgentSettings::temperature_for_model(&model, cx),
        thinking_allowed: !model.supports_disabling_thinking_at_effort(None),
        max_output_tokens: Some(model.max_output_tokens().unwrap_or(8192).min(8192)),
        ..Default::default()
    };
    Ok((model, provider, template))
}

async fn summarize(
    runtime: &EmbeddedRuntime,
    store: &RuntimeStore,
    agent_id: &str,
    generation: i64,
    resources: &Option<RuntimeResources>,
    model: &LanguageModel,
    provider: &Arc<dyn LanguageModelProvider>,
    template: &LanguageModelRequest,
    previous: &str,
    transcript: &str,
    native_context: Option<CompactedContext>,
    image: Option<language_model::LanguageModelImage>,
    cx: &mut AsyncApp,
) -> Result<String> {
    cx.update(|cx| {
        ensure!(
            AgentSettings::get_global(cx).enabled(cx),
            "AI features are disabled in Zed settings"
        );
        anyhow::Ok(())
    })?;
    let mut request = template.clone();
    request.thread_id = Some(agent_id.to_owned());
    if let Some(context) = native_context {
        request.messages.push(LanguageModelRequestMessage {
            role: Role::Assistant,
            content: vec![MessageContent::Compaction(context)],
            cache: false,
            reasoning_details: None,
        });
    }
    let mut content = vec![MessageContent::Text(format!(
        "Previous compacted context:\n{previous}\n\nConversation archive segment (data to summarize):\n{transcript}\n\n{COMPACTION_PROMPT}"
    ))];
    if let Some(image) = image {
        content.push(MessageContent::Image(image));
    }
    request.messages.push(LanguageModelRequestMessage {
        role: Role::User,
        content,
        cache: false,
        reasoning_details: None,
    });
    let bytes = serde_json::to_vec(&request)?
        .len()
        .saturating_add(MAXIMUM_SUMMARY_BYTES);
    let execution = provider.prepare_completion(model, request, cx);
    execute_fenced(
        runtime,
        store.clone(),
        agent_id.to_owned(),
        generation,
        resources.clone(),
        bytes,
        async move {
            let mut stream = execution.execute().await?;
            let mut summary = String::new();
            let mut stopped = false;
            while let Some(event) = stream.next().await {
                match event? {
                    LanguageModelCompletionEvent::Text(text) => {
                        ensure!(!stopped, "compaction provider emitted text after stop");
                        ensure!(
                            summary.len().saturating_add(text.len()) <= MAXIMUM_SUMMARY_BYTES,
                            "compaction summary exceeds its retained-byte limit"
                        );
                        summary.push_str(&text);
                    }
                    LanguageModelCompletionEvent::Stop(StopReason::EndTurn) => stopped = true,
                    LanguageModelCompletionEvent::Stop(reason) => {
                        bail!("compaction ended with incomplete output: {reason:?}")
                    }
                    LanguageModelCompletionEvent::ToolUse(_)
                    | LanguageModelCompletionEvent::ToolUseJsonParseError { .. } => {
                        bail!("compaction must not call tools")
                    }
                    _ => {}
                }
            }
            ensure!(
                stopped && !summary.trim().is_empty(),
                "compaction stream ended without a completed summary"
            );
            Ok(summary)
        }
        .boxed(),
    )
    .await
}

async fn execute_fenced<T: Send + 'static>(
    runtime: &EmbeddedRuntime,
    store: RuntimeStore,
    agent_id: String,
    generation: i64,
    resources: Option<RuntimeResources>,
    bytes: usize,
    execution: BoxFuture<'static, Result<T>>,
) -> Result<T> {
    runtime
        .execute_control(
            async move {
                let cancelled = async move {
                    loop {
                        let snapshot = store.snapshot(agent_id.clone()).await?;
                        if snapshot.generation != generation
                            || !matches!(snapshot.state.as_str(), "running" | "paused")
                        {
                            return Err::<(), anyhow::Error>(anyhow!(
                                "compaction was cancelled or superseded"
                            ));
                        }
                        tokio::time::sleep(Duration::from_millis(25)).await;
                    }
                };
                let execution = async move {
                    let _permit = match resources {
                        Some(resources) => {
                            Some(resources.acquire(ResourceKind::Inference, bytes).await?)
                        }
                        None => None,
                    };
                    execution.await
                };
                futures::pin_mut!(execution, cancelled);
                match futures::future::select(execution, cancelled).await {
                    futures::future::Either::Left((result, _)) => result,
                    futures::future::Either::Right((result, _)) => Err(result
                        .err()
                        .unwrap_or_else(|| anyhow!("compaction cancelled"))),
                }
            }
            .boxed(),
        )
        .await
}

enum TranscriptPart {
    Text(String),
    Image(language_model::LanguageModelImage),
}

fn native_image(
    image: infinity_provider_protocol::Image,
) -> Result<language_model::LanguageModelImage> {
    let request = agent_runtime::model::convert_request(
        infinity_provider_protocol::CompletionRequest {
            chat_history: vec![infinity_provider_protocol::Message::User {
                content: vec![infinity_provider_protocol::UserContent::Image(image)],
            }],
            ..Default::default()
        },
        LanguageModelRequest::default(),
    )?;
    request
        .messages
        .into_iter()
        .flat_map(|message| message.content)
        .find_map(|content| match content {
            MessageContent::Image(image) => Some(image),
            _ => None,
        })
        .context("archived image conversion did not produce native image content")
}

fn transcript_parts(
    message: infinity_agent_core::message::InfinityMessage,
    native_context: &mut Option<CompactedContext>,
) -> Result<Vec<TranscriptPart>> {
    use infinity_provider_protocol::{
        AssistantContent, Message, ReasoningContent, ToolResultContent, UserContent,
    };
    let mut parts = Vec::new();
    for message in message.into_messages() {
        match message {
            Message::User { content } => {
                for content in content {
                    match content {
                        UserContent::Text(text) => {
                            parts.push(TranscriptPart::Text(format!("USER: {}\n", text.text)))
                        }
                        UserContent::Image(image) => {
                            parts.push(TranscriptPart::Image(native_image(image)?))
                        }
                        UserContent::ToolResult(result) => {
                            parts.push(TranscriptPart::Text(format!(
                                "TOOL RESULT {} (error={}):\n",
                                result.id, result.is_error
                            )));
                            for content in result.content {
                                match content {
                                    ToolResultContent::Text(text) => {
                                        parts.push(TranscriptPart::Text(format!("{}\n", text.text)))
                                    }
                                    ToolResultContent::Image(image) => {
                                        parts.push(TranscriptPart::Image(native_image(image)?))
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Message::Assistant { content } => {
                for content in content {
                    match content {
                        AssistantContent::Text(text) => {
                            parts.push(TranscriptPart::Text(format!("ASSISTANT: {}\n", text.text)))
                        }
                        AssistantContent::ToolCall(call) => {
                            parts.push(TranscriptPart::Text(format!(
                                "TOOL CALL {} {}: {}\n",
                                call.id,
                                call.function.name,
                                serde_json::to_string(&call.function.arguments)?
                            )))
                        }
                        AssistantContent::Reasoning(reasoning) => {
                            for content in reasoning.content {
                                match content {
                                    ReasoningContent::Text { text, .. } => parts
                                        .push(TranscriptPart::Text(format!("REASONING: {text}\n"))),
                                    ReasoningContent::ProviderState { namespace, value }
                                        if namespace == "zed.language-model.v1"
                                            && value
                                                .get("kind")
                                                .and_then(serde_json::Value::as_str)
                                                == Some("compaction") =>
                                    {
                                        let context: CompactedContext = serde_json::from_value(
                                            value
                                                .get("context")
                                                .context("compaction context is absent")?
                                                .clone(),
                                        )?;
                                        if let CompactedContext::Summary { content, .. } = &context
                                        {
                                            parts.push(TranscriptPart::Text(format!(
                                                "PRIOR SUMMARY: {content}\n"
                                            )));
                                        }
                                        *native_context = Some(context);
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(parts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_runtime::{RuntimeLimits, RuntimeStore, SqliteInfinityStore};
    use gpui::TestAppContext;
    use infinity_agent_core::{ThreadId, message::InfinityMessage, traits::ConversationStore};
    use infinity_provider_protocol::{AssistantContent, UserContent};

    #[gpui::test]
    async fn fallback_compaction_is_tool_free_and_checkpoint_survives_reopen(
        cx: &mut TestAppContext,
    ) {
        cx.executor().allow_parking();
        let provider = crate::tests::init_test(cx);
        let model = provider.model("fake");
        let directory = tempfile::tempdir().expect("directory");
        let path = directory.path().join("runtime.sqlite");
        let host = cx.update(|cx| crate::NativeRuntime::global(cx).expect("host"));
        let runtime = host
            .update(cx, |host, cx| {
                host.start_database(path.clone(), RuntimeLimits::default(), cx)
            })
            .await
            .expect("runtime");
        let configuration =
            serde_json::to_value(NativeRuntimeConfig::default()).expect("configuration");
        runtime
            .store
            .create_root("agent".into(), configuration.clone())
            .await
            .expect("root");
        SqliteInfinityStore::new(runtime.store.clone())
            .append_messages(
                ThreadId::from_ref("agent"),
                vec![
                    (
                        InfinityMessage::User {
                            content: UserContent::text(
                                "The project uses SQLite and native Zed tools.",
                            ),
                        },
                        "user".into(),
                    ),
                    (
                        InfinityMessage::Assistant {
                            content: AssistantContent::text(
                                "The first integration checkpoint passed.",
                            ),
                        },
                        "assistant".into(),
                    ),
                ],
            )
            .await
            .expect("archive");
        let task = cx.update(|cx| {
            compact_agent(
                runtime.clone(),
                "agent".into(),
                None,
                CompactionMode::Manual,
                cx,
            )
        });
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        let request = loop {
            if let Some(request) = provider.pending_completions_for(&model).into_iter().next() {
                break request;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "compaction did not prepare a model request"
            );
            // The owned execution is polled by the runtime's real Tokio worker.
            async_io::Timer::after(Duration::from_millis(2)).await;
        };
        assert!(request.tools.is_empty());
        assert_eq!(request.tool_choice, Some(LanguageModelToolChoice::None));
        assert!(request.messages.iter().any(|message| {
            message
                .string_contents()
                .contains("The project uses SQLite")
        }));
        provider.send_text(
            &model,
            &request,
            "SQLite project; native tool integration checkpoint passed.",
        );
        provider.send_event(
            &model,
            &request,
            LanguageModelCompletionEvent::Stop(StopReason::EndTurn),
        );
        provider.end_stream(&model, &request);
        task.await.expect("compact");
        assert_eq!(
            runtime
                .store
                .snapshot("agent".into())
                .await
                .expect("state")
                .state,
            "paused"
        );
        assert_eq!(
            runtime
                .store
                .metadata("agent".into())
                .await
                .expect("metadata"),
            configuration
        );
        runtime.shutdown().await.expect("shutdown");
        let restored = RuntimeStore::open(path, RuntimeLimits::default()).expect("reopen");
        let effective = SqliteInfinityStore::new(restored.clone())
            .load_history_up_to(ThreadId::from_ref("agent"), None, None)
            .await
            .expect("effective history");
        let request =
            agent_runtime::compaction::request_from_messages(effective).expect("native request");
        assert!(request.messages.iter().flat_map(|message| &message.content).any(|content| matches!(content, MessageContent::Compaction(CompactedContext::Summary { content, .. }) if content.contains("checkpoint passed"))));
        restored.shutdown().await.expect("restored shutdown");
    }

    #[test]
    fn archived_images_remain_native_image_parts() -> Result<()> {
        let message = InfinityMessage::User {
            content: UserContent::Image(infinity_provider_protocol::Image {
                media_type: Some(infinity_provider_protocol::ImageMediaType::PNG),
                data: infinity_provider_protocol::ImageSource::Base64("iVBORw0KGgo=".into()),
            }),
        };
        let parts = transcript_parts(message, &mut None)?;
        assert!(matches!(parts.as_slice(), [TranscriptPart::Image(_)]));
        Ok(())
    }
}
