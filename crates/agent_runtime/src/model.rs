use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::Arc,
};

use anyhow::{Result, anyhow, ensure};
use async_trait::async_trait;
use base64::Engine as _;
use futures::{StreamExt as _, channel::oneshot};
use infinity_provider_protocol::{
    AssistantContent, CompletionError, CompletionRequest, ErrorClass, FinalResponse, Image,
    ImageMediaType, ImageSource, Message, ModelProvider, ModelStream, Reasoning, ReasoningContent,
    StreamChunk, ToolCall, ToolResultContent, Usage, UserContent,
};
pub use infinity_provider_protocol::{ModelEntry, ToolDefinition};
use language_model_core::{
    LanguageModelCompletionError, LanguageModelCompletionEvent, LanguageModelCompletionStream,
    LanguageModelImage, LanguageModelRequest, LanguageModelRequestMessage,
    LanguageModelRequestTool, LanguageModelToolResult, LanguageModelToolResultContent,
    LanguageModelToolUse, LanguageModelToolUseInput, MessageContent,
    PreparedLanguageModelCompletion, Role, StopReason, TokenUsage,
};
use serde::{Deserialize, Serialize};

const MAX_OUTPUT_BYTES: usize = 8 * 1024 * 1024;
const MAX_TOOL_CALLS: usize = 256;
const NATIVE_STATE_NAMESPACE: &str = "zed.language-model.v1";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum NativeProviderState {
    ToolUse {
        tool: LanguageModelToolUse,
    },
    ReasoningDetails {
        details: serde_json::Value,
    },
    Compaction {
        context: language_model_core::CompactedContext,
    },
    Stop {
        reason: StopReason,
    },
}

fn native_state(state: NativeProviderState) -> Result<StreamChunk> {
    Ok(StreamChunk::Reasoning(Reasoning {
        id: None,
        content: vec![ReasoningContent::ProviderState {
            namespace: NATIVE_STATE_NAMESPACE.into(),
            value: serde_json::to_value(state)?,
        }],
    }))
}

pub fn compacted_context_message(
    context: language_model_core::CompactedContext,
) -> Result<infinity_agent_core::message::InfinityMessage> {
    let StreamChunk::Reasoning(reasoning) =
        native_state(NativeProviderState::Compaction { context })?
    else {
        anyhow::bail!("native compaction did not produce a provider-state message");
    };
    Ok(infinity_agent_core::message::InfinityMessage::Assistant {
        content: AssistantContent::Reasoning(reasoning),
    })
}

pub(crate) fn compacted_context_from_message(
    message: &infinity_agent_core::message::InfinityMessage,
) -> Result<Option<language_model_core::CompactedContext>> {
    let infinity_agent_core::message::InfinityMessage::Assistant {
        content: AssistantContent::Reasoning(reasoning),
    } = message
    else {
        return Ok(None);
    };
    for content in &reasoning.content {
        if let ReasoningContent::ProviderState { namespace, value } = content {
            if namespace == NATIVE_STATE_NAMESPACE {
                if let NativeProviderState::Compaction { context } =
                    serde_json::from_value(value.clone())?
                {
                    return Ok(Some(context));
                }
            }
        }
    }
    Ok(None)
}

pub struct ModelExecutionRequest {
    pub model_id: String,
    pub request: LanguageModelRequest,
    response: oneshot::Sender<
        std::result::Result<PreparedLanguageModelCompletion, LanguageModelCompletionError>,
    >,
}

impl ModelExecutionRequest {
    pub fn is_cancelled(&self) -> bool {
        self.response.is_canceled()
    }

    pub fn respond(
        self,
        result: std::result::Result<PreparedLanguageModelCompletion, LanguageModelCompletionError>,
    ) {
        if self.response.send(result).is_err() {
            log::debug!("model execution requester detached");
        }
    }

    pub fn prepare_with(
        self,
        prepare: impl FnOnce(
            LanguageModelRequest,
        ) -> std::result::Result<
            PreparedLanguageModelCompletion,
            LanguageModelCompletionError,
        >,
    ) {
        let result = prepare(self.request);
        if self.response.send(result).is_err() {
            log::debug!("model execution requester detached");
        }
    }
}

#[derive(Clone)]
pub struct ModelExecutionSender(async_channel::Sender<ModelExecutionRequest>);

pub fn model_execution_channel(
    capacity: usize,
) -> Result<(
    ModelExecutionSender,
    async_channel::Receiver<ModelExecutionRequest>,
)> {
    ensure!(
        capacity > 0,
        "model preparation queue capacity must be positive"
    );
    let (sender, receiver) = async_channel::bounded(capacity);
    Ok((ModelExecutionSender(sender), receiver))
}

impl ModelExecutionSender {
    async fn prepare(
        &self,
        model_id: String,
        request: LanguageModelRequest,
    ) -> std::result::Result<PreparedLanguageModelCompletion, LanguageModelCompletionError> {
        let (response, receiver) = oneshot::channel();
        self.0
            .send(ModelExecutionRequest {
                model_id,
                request,
                response,
            })
            .await
            .map_err(|_| anyhow!("native model host has stopped"))?;
        let closed = self.0.closed();
        futures::pin_mut!(closed);
        match futures::future::select(receiver, closed).await {
            futures::future::Either::Left((result, _)) => {
                result.map_err(|_| anyhow!("native model preparation was interrupted"))?
            }
            futures::future::Either::Right(_) => {
                Err(anyhow!("native model host has stopped").into())
            }
        }
    }
}

pub type ModelEventObserver = Arc<dyn Fn(&LanguageModelCompletionEvent) + Send + Sync>;

pub struct ZedModelProvider {
    entry: ModelEntry,
    template: LanguageModelRequest,
    executor: ModelExecutionSender,
    observer: Option<ModelEventObserver>,
    ledger: Option<crate::provider_state::ProviderUsageLedger>,
    resources: Option<crate::resource::RuntimeResources>,
}

impl ZedModelProvider {
    pub fn new(
        entry: ModelEntry,
        template: LanguageModelRequest,
        executor: ModelExecutionSender,
    ) -> Self {
        Self {
            entry,
            template,
            executor,
            observer: None,
            ledger: None,
            resources: None,
        }
    }

    pub fn with_resources(mut self, resources: crate::resource::RuntimeResources) -> Self {
        self.resources = Some(resources);
        self
    }

    fn completion_error(&self, error: LanguageModelCompletionError) -> CompletionError {
        if let (Some(resources), Some(delay)) = (&self.resources, error.retry_delay(1)) {
            if let Err(error) = resources.defer_provider(&self.entry.model_id, delay) {
                log::error!("cannot record provider backoff: {error:#}");
            }
        }
        map_error(error)
    }

    pub fn with_state(mut self, store: crate::RuntimeStore, agent_id: String) -> Self {
        self.ledger = Some(crate::provider_state::ProviderUsageLedger::new(
            store,
            agent_id,
            self.entry.model_id.clone(),
        ));
        self
    }

    pub fn with_observer(mut self, observer: ModelEventObserver) -> Self {
        self.observer = Some(observer);
        self
    }
}

#[async_trait]
impl ModelProvider for ZedModelProvider {
    fn supports_parallel_tool_calls(&self) -> bool {
        true
    }
    fn preserves_unsigned_reasoning(&self) -> bool {
        true
    }

    async fn list_models(
        &self,
    ) -> std::result::Result<Vec<ModelEntry>, Box<dyn std::error::Error + Send + Sync>> {
        Ok(vec![self.entry.clone()])
    }

    async fn invoke_model(
        &self,
        model_id: &str,
        request: CompletionRequest,
    ) -> std::result::Result<ModelStream, CompletionError> {
        if model_id != self.entry.model_id {
            return Err(fatal(
                "the configured model changed; resolve it before another slice",
            ));
        }
        let permit = if let Some(resources) = &self.resources {
            resources
                .wait_for_provider(model_id)
                .await
                .map_err(|error| fatal(error.to_string()))?;
            let bytes = serialized_size(&request)
                .map_err(|error| fatal(error.to_string()))?
                .checked_mul(2)
                .and_then(|bytes| bytes.checked_add(MAX_OUTPUT_BYTES))
                .ok_or_else(|| fatal("model request byte count overflow"))?;
            Some(
                resources
                    .acquire(crate::resource::ResourceKind::Inference, bytes)
                    .await
                    .map_err(|error| fatal(error.to_string()))?,
            )
        } else {
            None
        };
        let request = convert_request(request, self.template.clone())
            .map_err(|error| fatal(error.to_string()))?;
        let attempt = match &self.ledger {
            Some(ledger) => Some(
                ledger
                    .begin_attempt()
                    .await
                    .map_err(|error| fatal(error.to_string()))?,
            ),
            None => None,
        };
        let execution = self.executor.prepare(model_id.to_owned(), request).await;
        if let (Err(error), Some(attempt)) = (&execution, &attempt) {
            attempt
                .record_provider_error(error)
                .await
                .map_err(|failure| {
                    fatal(format!(
                        "provider failure checkpoint failed: {failure:#}; {error}"
                    ))
                })?;
        }
        let execution = execution.map_err(|error| self.completion_error(error))?;
        let events = execution.execute().await;
        if let (Err(error), Some(attempt)) = (&events, &attempt) {
            attempt
                .record_provider_error(error)
                .await
                .map_err(|failure| {
                    fatal(format!(
                        "provider failure checkpoint failed: {failure:#}; {error}"
                    ))
                })?;
        }
        let events = events.map_err(|error| self.completion_error(error))?;
        let stream = convert_stream_with_ledger(
            events,
            self.observer.clone(),
            attempt,
            self.resources
                .clone()
                .map(|resources| (resources, self.entry.model_id.clone())),
        );
        Ok(
            futures::stream::unfold((stream, permit), |(mut stream, permit)| async move {
                stream.next().await.map(|chunk| (chunk, (stream, permit)))
            })
            .boxed(),
        )
    }
}

fn serialized_size(value: &impl Serialize) -> Result<usize> {
    struct Counter(usize);
    impl std::io::Write for Counter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0 = self
                .0
                .checked_add(bytes.len())
                .ok_or_else(|| std::io::Error::other("serialized byte count overflow"))?;
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut counter = Counter(0);
    serde_json::to_writer(&mut counter, value)?;
    Ok(counter.0)
}

fn fatal(message: impl Into<String>) -> CompletionError {
    CompletionError::provider(ErrorClass::Fatal, message)
}

fn map_error(error: LanguageModelCompletionError) -> CompletionError {
    use language_model_core::ProviderErrorCategory;
    let class = match &error {
        LanguageModelCompletionError::ProviderRejection {
            category: ProviderErrorCategory::PromptTooLarge { .. },
            ..
        } => ErrorClass::ContextOverflow,
        LanguageModelCompletionError::ProviderRejection {
            category: ProviderErrorCategory::RateLimit,
            ..
        } => ErrorClass::Throttled,
        LanguageModelCompletionError::ApiReadResponseError { .. }
        | LanguageModelCompletionError::HttpSend { .. }
        | LanguageModelCompletionError::StreamEndedUnexpectedly { .. } => ErrorClass::Transient,
        _ => ErrorClass::Fatal,
    };
    CompletionError::provider(class, error.to_string())
}

fn convert_image(image: Image) -> Result<LanguageModelImage> {
    ensure!(
        matches!(image.media_type, None | Some(ImageMediaType::PNG)),
        "the native bridge currently requires PNG image content"
    );
    match image.data {
        ImageSource::Base64(source) => {
            let decoded = base64::engine::general_purpose::STANDARD.decode(&source)?;
            ensure!(
                decoded.starts_with(b"\x89PNG\r\n\x1a\n"),
                "image content must be a base64-encoded PNG"
            );
            Ok(LanguageModelImage {
                source: source.into(),
            })
        }
        ImageSource::Url(_) => Err(anyhow!(
            "image URLs must be resolved by an authorized Zed tool before inference"
        )),
    }
}

pub fn convert_request(
    request: CompletionRequest,
    mut template: LanguageModelRequest,
) -> Result<LanguageModelRequest> {
    ensure!(
        request.additional_params.is_none(),
        "provider parameters must come from Zed's configured model settings"
    );
    template.messages.clear();
    if let Some(preamble) = request.preamble {
        template.messages.push(LanguageModelRequestMessage {
            role: Role::System,
            content: vec![MessageContent::Text(preamble)],
            cache: false,
            reasoning_details: None,
        });
    }
    let mut native_tools = HashMap::<String, LanguageModelToolUse>::new();
    for message in &request.chat_history {
        if let Message::Assistant { content } = message {
            for content in content {
                if let AssistantContent::Reasoning(reasoning) = content {
                    for block in &reasoning.content {
                        if let ReasoningContent::ProviderState { namespace, value } = block {
                            ensure!(
                                namespace == NATIVE_STATE_NAMESPACE,
                                "unknown provider-state namespace"
                            );
                            if let NativeProviderState::ToolUse { tool } =
                                serde_json::from_value(value.clone())?
                            {
                                native_tools.insert(tool.id.to_string(), tool);
                            }
                        }
                    }
                }
            }
        }
    }
    let configured_tools = template
        .tools
        .drain(..)
        .map(|tool| (tool.name.clone(), tool))
        .collect::<HashMap<_, _>>();
    let mut tools = HashMap::<String, String>::new();
    for message in request.chat_history {
        let mut reasoning_details = None;
        let (role, content) = match message {
            Message::Assistant { content } => {
                let mut converted = Vec::new();
                for item in content {
                    match item {
                        AssistantContent::Text(text) => {
                            converted.push(MessageContent::Text(text.text))
                        }
                        AssistantContent::ToolCall(call) => {
                            ensure!(
                                call.call_id.as_ref().is_none_or(|id| id == &call.id),
                                "secondary provider call IDs require a protocol extension"
                            );
                            tools.insert(call.id.clone(), call.function.name.clone());
                            let tool = if let Some(tool) = native_tools.remove(&call.id) {
                                let arguments = match &tool.input {
                                    LanguageModelToolUseInput::Json(value) => value.clone(),
                                    LanguageModelToolUseInput::Text(value) => {
                                        serde_json::Value::String(value.clone())
                                    }
                                };
                                ensure!(
                                    tool.name.as_ref() == call.function.name
                                        && arguments == call.function.arguments,
                                    "native tool state does not match its durable call"
                                );
                                tool
                            } else {
                                LanguageModelToolUse {
                                    id: call.id.into(),
                                    name: call.function.name.into(),
                                    raw_input: serde_json::to_string(&call.function.arguments)?,
                                    input: LanguageModelToolUseInput::Json(call.function.arguments),
                                    is_input_complete: true,
                                    thought_signature: None,
                                }
                            };
                            converted.push(MessageContent::ToolUse(tool));
                        }
                        AssistantContent::Reasoning(reasoning) => {
                            ensure!(
                                reasoning.id.is_none(),
                                "provider reasoning IDs require a protocol extension"
                            );
                            for item in reasoning.content {
                                match item {
                                    ReasoningContent::Text { text, signature } => {
                                        converted.push(MessageContent::Thinking { text, signature })
                                    }
                                    ReasoningContent::Redacted { data } => {
                                        converted.push(MessageContent::RedactedThinking(data))
                                    }
                                    ReasoningContent::ProviderState { namespace, value } => {
                                        ensure!(
                                            namespace == NATIVE_STATE_NAMESPACE,
                                            "unknown provider-state namespace"
                                        );
                                        match serde_json::from_value::<NativeProviderState>(value)?
                                        {
                                            NativeProviderState::ToolUse { .. }
                                            | NativeProviderState::Stop { .. } => {}
                                            NativeProviderState::ReasoningDetails { details } => {
                                                reasoning_details = Some(Arc::new(details))
                                            }
                                            NativeProviderState::Compaction { context } => {
                                                converted.push(MessageContent::Compaction(context))
                                            }
                                        }
                                    }
                                    _ => {
                                        return Err(anyhow!(
                                            "foreign opaque reasoning requires its native provider state"
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
                (Role::Assistant, converted)
            }
            Message::User { content } => {
                let mut converted = Vec::new();
                for item in content {
                    match item {
                        UserContent::Text(text) => converted.push(MessageContent::Text(text.text)),
                        UserContent::Image(image) => {
                            converted.push(MessageContent::Image(convert_image(image)?))
                        }
                        UserContent::ToolResult(result) => {
                            ensure!(
                                result.call_id.as_ref().is_none_or(|id| id == &result.id),
                                "secondary tool-result IDs require a protocol extension"
                            );
                            let name = tools
                                .get(&result.id)
                                .ok_or_else(|| {
                                    anyhow!("tool result has no corresponding call: {}", result.id)
                                })?
                                .clone();
                            let content = result
                                .content
                                .into_iter()
                                .map(|item| match item {
                                    ToolResultContent::Text(text) => {
                                        Ok(LanguageModelToolResultContent::Text(text.text.into()))
                                    }
                                    ToolResultContent::Image(image) => {
                                        Ok(LanguageModelToolResultContent::Image(convert_image(
                                            image,
                                        )?))
                                    }
                                })
                                .collect::<Result<Vec<_>>>()?;
                            converted.push(MessageContent::ToolResult(LanguageModelToolResult {
                                tool_use_id: result.id.into(),
                                tool_name: name.into(),
                                is_error: result.is_error,
                                content,
                                output: result.output,
                            }));
                        }
                    }
                }
                (Role::User, converted)
            }
        };
        let contains_compaction = content
            .iter()
            .any(|content| matches!(content, MessageContent::Compaction(_)));
        let previous = template.messages.last_mut().filter(|previous| {
            previous.role == role
                && role == Role::Assistant
                && !contains_compaction
                && !previous
                    .content
                    .iter()
                    .any(|content| matches!(content, MessageContent::Compaction(_)))
        });
        if let Some(previous) = previous {
            previous.content.extend(content);
            if reasoning_details.is_some() {
                previous.reasoning_details = reasoning_details;
            }
        } else if !content.is_empty() || reasoning_details.is_some() {
            template.messages.push(LanguageModelRequestMessage {
                role,
                content,
                cache: false,
                reasoning_details,
            });
        }
    }
    template.tools = request
        .tools
        .into_iter()
        .map(|tool| {
            configured_tools
                .get(&tool.name)
                .cloned()
                .unwrap_or_else(|| {
                    LanguageModelRequestTool::function(
                        tool.name,
                        tool.description,
                        tool.parameters,
                        false,
                    )
                })
        })
        .collect();
    if let Some(index) = template.messages.iter().rposition(|message| {
        message
            .content
            .iter()
            .any(|content| matches!(content, MessageContent::Compaction(_)))
    }) {
        let compacted = template.messages.split_off(index);
        template
            .messages
            .retain(|message| message.role == Role::System);
        template.messages.extend(compacted);
    }
    template.max_output_tokens = match (template.max_output_tokens, request.max_tokens) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    };
    Ok(template)
}

struct StreamState {
    events: LanguageModelCompletionStream,
    pending: VecDeque<StreamChunk>,
    thinking: Option<(String, Option<String>)>,
    partial_tools: HashSet<String>,
    partial_tool_bytes: usize,
    complete_tools: HashMap<String, ToolCall>,
    usage: TokenUsage,
    reasoning_details: Option<serde_json::Value>,
    reasoning_detail_bytes: usize,
    ledger: Option<crate::provider_state::ProviderUsageAttempt>,
    resources: Option<(crate::resource::RuntimeResources, String)>,
    stopped: bool,
    ended: bool,
    output_bytes: usize,
    observer: Option<ModelEventObserver>,
}

impl StreamState {
    fn flush_thinking(&mut self) {
        if let Some((text, signature)) = self.thinking.take() {
            self.pending
                .push_back(StreamChunk::Reasoning(Reasoning::new_with_signature(
                    text, signature,
                )));
        }
    }

    fn accept(&mut self, event: LanguageModelCompletionEvent) -> Result<()> {
        if let Some(observer) = &self.observer {
            observer(&event);
        }
        ensure!(
            !self.stopped
                || matches!(
                    event,
                    LanguageModelCompletionEvent::UsageUpdate(_)
                        | LanguageModelCompletionEvent::ReasoningDetails(_)
                        | LanguageModelCompletionEvent::Compaction(_)
                ),
            "provider emitted content after the response stop"
        );
        if matches!(
            event,
            LanguageModelCompletionEvent::Text(_)
                | LanguageModelCompletionEvent::RedactedThinking { .. }
                | LanguageModelCompletionEvent::ToolUse(_)
                | LanguageModelCompletionEvent::Stop(_)
                | LanguageModelCompletionEvent::StartMessage { .. }
        ) {
            self.flush_thinking();
        }
        match event {
            LanguageModelCompletionEvent::Text(text) => {
                self.output_bytes = self.output_bytes.saturating_add(text.len());
                self.pending.push_back(StreamChunk::Text(text));
            }
            LanguageModelCompletionEvent::Thinking { text, signature } => {
                self.output_bytes = self
                    .output_bytes
                    .saturating_add(text.len())
                    .saturating_add(signature.as_ref().map_or(0, String::len));
                let block = self.thinking.get_or_insert_with(|| (String::new(), None));
                block.0.push_str(&text);
                if signature.is_some() {
                    block.1 = signature;
                }
                self.pending
                    .push_back(StreamChunk::ReasoningDelta { id: None, text });
            }
            LanguageModelCompletionEvent::RedactedThinking { data } => {
                self.output_bytes = self.output_bytes.saturating_add(data.len());
                self.pending.push_back(StreamChunk::Reasoning(Reasoning {
                    id: None,
                    content: vec![ReasoningContent::Redacted { data }],
                }));
            }
            LanguageModelCompletionEvent::ToolUse(tool) => {
                ensure!(
                    tool.raw_input.len() <= MAX_OUTPUT_BYTES,
                    "tool output exceeds the runtime byte limit"
                );
                let id = tool.id.to_string();
                ensure!(
                    id.len().saturating_add(tool.name.len()) <= MAX_OUTPUT_BYTES,
                    "tool identity exceeds the runtime byte limit"
                );
                if !tool.is_input_complete {
                    ensure!(
                        !self.complete_tools.contains_key(&id),
                        "provider reopened a completed tool call"
                    );
                    let id_bytes = id.len();
                    if self.partial_tools.insert(id) {
                        self.partial_tool_bytes = self.partial_tool_bytes.saturating_add(id_bytes);
                    }
                    ensure!(
                        self.partial_tool_bytes.saturating_add(self.output_bytes)
                            <= MAX_OUTPUT_BYTES,
                        "partial tool identities exceed the runtime byte limit"
                    );
                    ensure!(
                        self.partial_tools.len()
                            + self.complete_tools.len()
                            + self.partial_tools.len()
                            <= MAX_TOOL_CALLS,
                        "too many tool calls in one response"
                    );
                    return Ok(());
                }
                let arguments = match &tool.input {
                    LanguageModelToolUseInput::Json(arguments) => arguments.clone(),
                    LanguageModelToolUseInput::Text(input) => {
                        serde_json::Value::String(input.clone())
                    }
                };
                if self.partial_tools.remove(&id) {
                    self.partial_tool_bytes = self.partial_tool_bytes.saturating_sub(id.len());
                }
                let call = ToolCall::new(id.clone(), tool.name.to_string(), arguments);
                if let Some(previous) = self.complete_tools.get(&id) {
                    ensure!(
                        previous == &call,
                        "provider reused a tool-call ID with different content"
                    );
                } else {
                    self.output_bytes = self
                        .output_bytes
                        .saturating_add(id.len())
                        .saturating_add(call.function.name.len())
                        .saturating_add(serde_json::to_vec(&call.function.arguments)?.len());
                    self.complete_tools.insert(id, call.clone());
                    ensure!(
                        self.complete_tools.len() + self.partial_tools.len() <= MAX_TOOL_CALLS,
                        "too many tool calls in one response"
                    );
                    self.output_bytes = self
                        .output_bytes
                        .saturating_add(serde_json::to_vec(&tool)?.len());
                    self.pending
                        .push_back(native_state(NativeProviderState::ToolUse { tool })?);
                    self.pending.push_back(StreamChunk::ToolCall(call));
                }
            }
            LanguageModelCompletionEvent::UsageUpdate(usage) => self.usage = usage,
            LanguageModelCompletionEvent::Stop(
                reason @ (StopReason::EndTurn | StopReason::ToolUse | StopReason::Refusal),
            ) => {
                self.stopped = true;
                self.pending
                    .push_back(native_state(NativeProviderState::Stop { reason })?);
            }
            LanguageModelCompletionEvent::Stop(reason) => {
                return Err(anyhow!(
                    "model response stopped with {reason:?}; an incomplete response cannot be checkpointed"
                ));
            }
            LanguageModelCompletionEvent::ToolUseJsonParseError {
                json_parse_error, ..
            } => return Err(anyhow!("invalid tool arguments: {json_parse_error}")),
            LanguageModelCompletionEvent::ReasoningDetails(details) => {
                let bytes = serde_json::to_vec(&details)?.len();
                ensure!(
                    bytes <= MAX_OUTPUT_BYTES,
                    "provider reasoning details exceed the runtime byte limit"
                );
                self.reasoning_detail_bytes = bytes;
                self.reasoning_details = Some(details);
            }
            LanguageModelCompletionEvent::Compaction(
                language_model_core::CompactionUpdate::Finished(context),
            ) => {
                self.output_bytes = self
                    .output_bytes
                    .saturating_add(serde_json::to_vec(&context)?.len());
                self.pending
                    .push_back(native_state(NativeProviderState::Compaction { context })?);
            }
            LanguageModelCompletionEvent::Compaction(_) => {}
            LanguageModelCompletionEvent::Queued { .. }
            | LanguageModelCompletionEvent::Started
            | LanguageModelCompletionEvent::StartMessage { .. } => {}
        }
        ensure!(
            self.output_bytes
                .saturating_add(self.partial_tool_bytes)
                .saturating_add(self.reasoning_detail_bytes)
                <= MAX_OUTPUT_BYTES,
            "model output exceeds the runtime byte limit"
        );
        Ok(())
    }
}

pub fn convert_stream(
    events: LanguageModelCompletionStream,
    observer: Option<ModelEventObserver>,
) -> ModelStream {
    convert_stream_with_ledger(events, observer, None, None)
}

fn convert_stream_with_ledger(
    events: LanguageModelCompletionStream,
    observer: Option<ModelEventObserver>,
    ledger: Option<crate::provider_state::ProviderUsageAttempt>,
    resources: Option<(crate::resource::RuntimeResources, String)>,
) -> ModelStream {
    let state = StreamState {
        events,
        pending: VecDeque::new(),
        thinking: None,
        partial_tools: HashSet::new(),
        partial_tool_bytes: 0,
        complete_tools: HashMap::new(),
        usage: TokenUsage::default(),
        reasoning_details: None,
        reasoning_detail_bytes: 0,
        ledger,
        resources,
        stopped: false,
        ended: false,
        output_bytes: 0,
        observer,
    };
    futures::stream::unfold(state, |mut state| async move {
        loop {
            if let Some(chunk) = state.pending.pop_front() {
                return Some((Ok(chunk), state));
            }
            if state.ended {
                return None;
            }
            match state.events.next().await {
                Some(Ok(event)) => {
                    if let LanguageModelCompletionEvent::UsageUpdate(usage) = &event {
                        if let Some(ledger) = &state.ledger {
                            if let Err(error) = ledger.record(*usage, false).await {
                                state.ended = true;
                                return Some((
                                    Err(fatal(format!("usage checkpoint failed: {error:#}"))),
                                    state,
                                ));
                            }
                        }
                    }
                    if let Err(error) = state.accept(event) {
                        state.pending.clear();
                        state.ended = true;
                        return Some((Err(fatal(error.to_string())), state));
                    }
                }
                Some(Err(error)) => {
                    if let Some(attempt) = &state.ledger {
                        if let Err(failure) = attempt.record_provider_error(&error).await {
                            state.ended = true;
                            state.pending.clear();
                            return Some((
                                Err(fatal(format!(
                                    "provider failure checkpoint failed: {failure:#}; {error}"
                                ))),
                                state,
                            ));
                        }
                    }
                    if let (Some((resources, model_id)), Some(delay)) =
                        (&state.resources, error.retry_delay(1))
                    {
                        if let Err(error) = resources.defer_provider(model_id, delay) {
                            log::error!("cannot record provider backoff: {error:#}");
                        }
                    }
                    state.ended = true;
                    state.pending.clear();
                    return Some((Err(map_error(error)), state));
                }
                None => {
                    state.ended = true;
                    if !state.stopped || !state.partial_tools.is_empty() {
                        return Some((
                            Err(CompletionError::provider(
                                ErrorClass::Transient,
                                "provider stream ended without a complete response",
                            )),
                            state,
                        ));
                    }
                    state.flush_thinking();
                    if let Some(details) = state.reasoning_details.take() {
                        match native_state(NativeProviderState::ReasoningDetails { details }) {
                            Ok(chunk) => state.pending.push_back(chunk),
                            Err(error) => return Some((Err(fatal(error.to_string())), state)),
                        }
                    }
                    let usage = state.usage;
                    if let Some(ledger) = &state.ledger {
                        if let Err(error) = ledger.record(usage, true).await {
                            state.pending.clear();
                            return Some((
                                Err(fatal(format!("usage checkpoint failed: {error:#}"))),
                                state,
                            ));
                        }
                    }
                    state.pending.push_back(StreamChunk::Final(FinalResponse {
                        usage: Some(Usage {
                            input_tokens: usage
                                .input_tokens
                                .saturating_add(usage.cache_creation_input_tokens)
                                .saturating_add(usage.cache_read_input_tokens),
                            output_tokens: usage.output_tokens,
                            total_tokens: usage
                                .input_tokens
                                .saturating_add(usage.output_tokens)
                                .saturating_add(usage.cache_creation_input_tokens)
                                .saturating_add(usage.cache_read_input_tokens),
                            cached_input_tokens: usage.cache_read_input_tokens,
                        }),
                    }));
                }
            }
        }
    })
    .boxed()
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Context as _;
    use futures::{FutureExt as _, TryStreamExt as _};
    use infinity_provider_protocol::ToolResult;
    use serde_json::json;

    fn tool(complete: bool) -> LanguageModelToolUse {
        LanguageModelToolUse {
            id: "call".into(),
            name: "read_file".into(),
            raw_input: r#"{"path":"src/main.rs"}"#.into(),
            input: LanguageModelToolUseInput::Json(json!({"path": "src/main.rs"})),
            is_input_complete: complete,
            thought_signature: None,
        }
    }

    fn stream(events: Vec<LanguageModelCompletionEvent>) -> ModelStream {
        convert_stream(
            futures::stream::iter(events.into_iter().map(Ok)).boxed(),
            None,
        )
    }

    #[test]
    fn history_conversion_preserves_options_and_correlates_tool_results() -> Result<()> {
        let call = ToolCall::new("call", "read_file", json!({"path": "src/main.rs"}));
        let request = CompletionRequest {
            preamble: Some("project rules".into()),
            chat_history: vec![
                Message::user("inspect"),
                Message::Assistant {
                    content: vec![
                        AssistantContent::Reasoning(Reasoning::new_with_signature(
                            "thought",
                            Some("signature".into()),
                        )),
                        AssistantContent::ToolCall(call),
                    ],
                },
                Message::User {
                    content: vec![UserContent::ToolResult(ToolResult {
                        id: "call".into(),
                        call_id: None,
                        content: vec![ToolResultContent::text("file contents")],

                        ..Default::default()
                    })],
                },
            ],
            tools: vec![ToolDefinition {
                name: "read_file".into(),
                description: "read".into(),
                parameters: json!({"type": "object"}),
            }],
            max_tokens: Some(400),
            ..Default::default()
        };
        let converted = convert_request(
            request,
            LanguageModelRequest {
                thread_id: Some("thread".into()),
                temperature: Some(0.25),
                thinking_allowed: true,
                thinking_effort: Some("high".into()),
                max_output_tokens: Some(200),
                ..Default::default()
            },
        )?;
        assert_eq!(converted.messages.len(), 4);
        assert_eq!(
            converted.messages.first().context("system")?.role,
            Role::System
        );
        let assistant = converted.messages.get(2).context("assistant")?;
        assert!(
            matches!(assistant.content.first(), Some(MessageContent::Thinking { signature: Some(signature), .. }) if signature == "signature")
        );
        let MessageContent::ToolResult(result) = converted
            .messages
            .last()
            .context("result")?
            .content
            .first()
            .context("content")?
        else {
            anyhow::bail!("expected tool result")
        };
        assert_eq!(result.tool_name.as_ref(), "read_file");
        assert_eq!(result.text_contents(), "file contents");
        assert_eq!(converted.tools.first().context("tool")?.name, "read_file");
        assert_eq!(converted.thread_id.as_deref(), Some("thread"));
        assert_eq!(converted.temperature, Some(0.25));
        assert!(converted.thinking_allowed);
        assert_eq!(converted.thinking_effort.as_deref(), Some("high"));
        assert_eq!(converted.max_output_tokens, Some(200));
        Ok(())
    }

    #[test]
    fn unrepresentable_history_is_rejected_before_provider_preparation() {
        let request = CompletionRequest {
            additional_params: Some(json!({"provider": "override"})),
            ..Default::default()
        };
        assert!(convert_request(request, LanguageModelRequest::default()).is_err());
        let request = CompletionRequest {
            chat_history: vec![Message::User {
                content: vec![UserContent::ToolResult(ToolResult {
                    id: "unknown".into(),
                    call_id: None,
                    content: vec![],

                    ..Default::default()
                })],
            }],
            ..Default::default()
        };
        assert!(convert_request(request, LanguageModelRequest::default()).is_err());
        for image in [
            Image {
                data: ImageSource::Url("https://example.com/image.png".into()),
                media_type: Some(ImageMediaType::PNG),
            },
            Image {
                data: ImageSource::Base64("image".into()),
                media_type: Some(ImageMediaType::JPEG),
            },
        ] {
            assert!(convert_image(image).is_err());
        }
    }

    #[tokio::test]
    async fn reasoning_signatures_survive_usage_events_and_final_usage_follows_stop() -> Result<()>
    {
        let events = vec![
            LanguageModelCompletionEvent::Thinking {
                text: "thought".into(),
                signature: None,
            },
            LanguageModelCompletionEvent::UsageUpdate(TokenUsage::default()),
            LanguageModelCompletionEvent::Thinking {
                text: String::new(),
                signature: Some("signed".into()),
            },
            LanguageModelCompletionEvent::Text("answer".into()),
            LanguageModelCompletionEvent::Stop(StopReason::EndTurn),
            LanguageModelCompletionEvent::UsageUpdate(TokenUsage {
                input_tokens: 10,
                output_tokens: 2,
                cache_creation_input_tokens: 3,
                cache_read_input_tokens: 7,
            }),
        ];
        let observed = Arc::new(std::sync::Mutex::new(Vec::new()));
        let observer: ModelEventObserver = {
            let observed = observed.clone();
            Arc::new(move |event| observed.lock().expect("observer").push(event.clone()))
        };
        let chunks = convert_stream(
            futures::stream::iter(events.into_iter().map(Ok)).boxed(),
            Some(observer),
        )
        .try_collect::<Vec<_>>()
        .await?;
        let reasoning = chunks
            .iter()
            .find_map(|chunk| match chunk {
                StreamChunk::Reasoning(reasoning) => Some(reasoning),
                _ => None,
            })
            .context("reasoning")?;
        assert_eq!(
            reasoning,
            &Reasoning::new_with_signature("thought", Some("signed".into()))
        );
        let Some(StreamChunk::Final(response)) = chunks.last() else {
            anyhow::bail!("expected final")
        };
        assert_eq!(
            response.usage,
            Some(Usage {
                input_tokens: 20,
                output_tokens: 2,
                total_tokens: 22,
                cached_input_tokens: 7
            })
        );
        assert_eq!(
            observed
                .lock()
                .map_err(|_| anyhow!("observer poisoned"))?
                .len(),
            6
        );
        Ok(())
    }

    #[tokio::test]
    async fn partial_or_invalid_responses_never_emit_a_final_checkpoint() -> Result<()> {
        let cases = vec![
            vec![LanguageModelCompletionEvent::Text("partial".into())],
            vec![
                LanguageModelCompletionEvent::ToolUse(tool(false)),
                LanguageModelCompletionEvent::Stop(StopReason::ToolUse),
            ],
            vec![LanguageModelCompletionEvent::ToolUseJsonParseError {
                id: "call".into(),
                tool_name: "read_file".into(),
                raw_input: "{".into(),
                json_parse_error: "unterminated".into(),
            }],
            vec![LanguageModelCompletionEvent::ReasoningDetails(
                json!({"id": "opaque"}),
            )],
            vec![LanguageModelCompletionEvent::Stop(StopReason::MaxTokens)],
            vec![
                LanguageModelCompletionEvent::Stop(StopReason::EndTurn),
                LanguageModelCompletionEvent::Text("late".into()),
            ],
        ];
        for events in cases {
            let chunks = stream(events).collect::<Vec<_>>().await;
            assert!(chunks.iter().any(Result::is_err));
            assert!(
                !chunks
                    .iter()
                    .any(|chunk| matches!(chunk, Ok(StreamChunk::Final(_))))
            );
        }
        Ok(())
    }

    #[tokio::test]
    async fn response_limits_bound_text_and_unfinished_tool_identities() {
        let oversized_text = stream(vec![LanguageModelCompletionEvent::Text(
            "x".repeat(MAX_OUTPUT_BYTES + 1),
        )])
        .collect::<Vec<_>>()
        .await;
        assert!(oversized_text.iter().any(Result::is_err));
        assert!(
            !oversized_text
                .iter()
                .any(|chunk| matches!(chunk, Ok(StreamChunk::Text(_))))
        );
        let unfinished_tools = (0..=MAX_TOOL_CALLS)
            .map(|index| {
                let mut tool = tool(false);
                tool.id = format!("call-{index}").into();
                LanguageModelCompletionEvent::ToolUse(tool)
            })
            .collect();
        let chunks = stream(unfinished_tools).collect::<Vec<_>>().await;
        assert!(chunks.iter().any(Result::is_err));
        assert!(!chunks.iter().any(|chunk| matches!(
            chunk,
            Ok(StreamChunk::ToolCall(_)) | Ok(StreamChunk::Final(_))
        )));
    }

    #[tokio::test]
    async fn tool_calls_wait_for_complete_arguments_and_are_emitted_once() -> Result<()> {
        let chunks = stream(vec![
            LanguageModelCompletionEvent::ToolUse(tool(false)),
            LanguageModelCompletionEvent::ToolUse(tool(true)),
            LanguageModelCompletionEvent::ToolUse(tool(true)),
            LanguageModelCompletionEvent::Stop(StopReason::ToolUse),
        ])
        .try_collect::<Vec<_>>()
        .await?;
        let calls = chunks
            .iter()
            .filter_map(|chunk| match chunk {
                StreamChunk::ToolCall(call) => Some(call),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls.first().context("call")?.function.name, "read_file");
        assert_eq!(
            calls.first().context("call")?.function.arguments,
            json!({"path": "src/main.rs"})
        );
        Ok(())
    }

    #[tokio::test]
    async fn execution_is_polled_on_the_worker_and_cancellation_drops_the_original_stream()
    -> Result<()> {
        let (executor, requests) = model_execution_channel(1)?;
        let (events, receiver) = futures::channel::mpsc::unbounded();
        let (polled, execution_thread) = oneshot::channel();
        let preparer = std::thread::spawn(move || {
            let request = futures::executor::block_on(requests.recv()).expect("request");
            let host_thread = std::thread::current().id();
            request.respond(Ok(PreparedLanguageModelCompletion::new(
                async move {
                    if polled.send(std::thread::current().id()).is_err() {
                        return Err(anyhow!("worker detached").into());
                    }
                    Ok(receiver.boxed())
                }
                .boxed(),
            )));
            host_thread
        });
        let resources =
            crate::resource::RuntimeResources::new(crate::resource::ResourceLimits::default())?;
        let provider = ZedModelProvider::new(
            ModelEntry {
                model_id: "fake/model".into(),
                display_name: "Fake".into(),
                context_window: 1000,
                max_output_tokens: None,
                supports_image_input: false,
            },
            LanguageModelRequest::default(),
            executor,
        )
        .with_resources(resources.clone());
        let mut response = provider
            .invoke_model("fake/model", CompletionRequest::default())
            .await?;
        assert_eq!(execution_thread.await?, std::thread::current().id());
        assert_ne!(
            preparer.join().map_err(|_| anyhow!("preparer panicked"))?,
            std::thread::current().id()
        );
        assert!(response.next().now_or_never().is_none());
        assert_eq!(resources.snapshot()?.active_inference, 1);
        assert!(resources.snapshot()?.retained_bytes >= MAX_OUTPUT_BYTES);
        drop(response);
        assert_eq!(resources.snapshot()?.active_inference, 0);
        assert_eq!(resources.snapshot()?.retained_bytes, 0);
        assert!(
            events.is_closed(),
            "cancellation must release the actual provider stream"
        );
        Ok(())
    }

    #[tokio::test]
    async fn preparation_queue_is_bounded_and_reports_host_shutdown() -> Result<()> {
        assert!(model_execution_channel(0).is_err());
        let (executor, requests) = model_execution_channel(1)?;
        let first = executor.prepare("fake/model".into(), LanguageModelRequest::default());
        futures::pin_mut!(first);
        assert!(first.as_mut().now_or_never().is_none());
        let second = executor.prepare("fake/model".into(), LanguageModelRequest::default());
        futures::pin_mut!(second);
        assert!(second.as_mut().now_or_never().is_none());
        assert_eq!(requests.len(), 1);
        requests.close();
        assert!(second.await.is_err());
        drop(requests);
        assert!(first.await.is_err());
        Ok(())
    }
    #[tokio::test]
    async fn native_tool_state_and_reasoning_details_round_trip_without_prompt_encoding()
    -> Result<()> {
        let mut original = tool(true);
        original.thought_signature = Some("provider-signature".into());
        original.input = LanguageModelToolUseInput::Text("*** Begin Patch\n*** End Patch".into());
        original.raw_input = "*** Begin Patch\n*** End Patch".into();
        let details = json!([{ "id": "reasoning-id", "encrypted_content": "opaque" }]);
        let chunks = stream(vec![
            LanguageModelCompletionEvent::ToolUse(original.clone()),
            LanguageModelCompletionEvent::ReasoningDetails(details.clone()),
            LanguageModelCompletionEvent::Stop(StopReason::ToolUse),
        ])
        .try_collect::<Vec<_>>()
        .await?;
        let content = chunks
            .into_iter()
            .filter_map(|chunk| match chunk {
                StreamChunk::Reasoning(reasoning) => Some(AssistantContent::Reasoning(reasoning)),
                StreamChunk::ToolCall(call) => Some(AssistantContent::ToolCall(call)),
                _ => None,
            })
            .collect();
        let request = convert_request(
            CompletionRequest {
                chat_history: vec![Message::Assistant { content }],
                ..Default::default()
            },
            LanguageModelRequest::default(),
        )?;
        let message = request.messages.first().context("assistant message")?;
        assert_eq!(message.reasoning_details.as_deref(), Some(&details));
        let restored = message
            .content
            .iter()
            .find_map(|content| match content {
                MessageContent::ToolUse(tool) => Some(tool),
                _ => None,
            })
            .context("tool")?;
        assert_eq!(restored, &original);
        assert!(!message.string_contents().contains("opaque"));
        Ok(())
    }

    #[tokio::test]
    async fn provider_compaction_replaces_prior_context_and_retains_opaque_state() -> Result<()> {
        use language_model_core::{CompactedContext, CompactionUpdate};
        let context = CompactedContext::Summary {
            content: "summary".into(),
            provider_state: None,
        };
        let chunks = stream(vec![
            LanguageModelCompletionEvent::Compaction(CompactionUpdate::Finished(context.clone())),
            LanguageModelCompletionEvent::Text("new answer".into()),
            LanguageModelCompletionEvent::Stop(StopReason::EndTurn),
        ])
        .try_collect::<Vec<_>>()
        .await?;
        let mut history = vec![Message::user("old context")];
        for chunk in chunks {
            match chunk {
                StreamChunk::Reasoning(reasoning) => history.push(Message::Assistant {
                    content: vec![AssistantContent::Reasoning(reasoning)],
                }),
                StreamChunk::Text(text) => history.push(Message::assistant(text)),
                _ => {}
            }
        }
        let request = convert_request(
            CompletionRequest {
                preamble: Some("system rules".into()),
                chat_history: history,
                ..Default::default()
            },
            LanguageModelRequest::default(),
        )?;
        assert!(
            !request
                .messages
                .iter()
                .any(|message| message.string_contents().contains("old context"))
        );
        assert!(request.messages.iter().any(|message| {
            message
                .content
                .contains(&MessageContent::Compaction(context.clone()))
        }));
        assert!(
            request
                .messages
                .iter()
                .any(|message| message.string_contents().contains("system rules"))
        );
        Ok(())
    }
    #[tokio::test]
    async fn provider_retry_after_delays_other_admitted_completions() -> Result<()> {
        use crate::resource::{ResourceLimits, RuntimeResources};
        use std::time::Duration;
        let resources = RuntimeResources::new(ResourceLimits::default())?;
        let (executor, _) = model_execution_channel(1)?;
        let provider = ZedModelProvider::new(
            ModelEntry {
                model_id: "fake/model".into(),
                display_name: "Fake".into(),
                context_window: 1000,
                max_output_tokens: None,
                supports_image_input: false,
            },
            LanguageModelRequest::default(),
            executor,
        )
        .with_resources(resources.clone());
        let error = LanguageModelCompletionError::ProviderRejection {
            provider: language_model_core::LanguageModelProviderName("Fake".into()),
            status: None,
            code: None,
            message: "rate limited".into(),
            retry_after: Some(Duration::from_millis(20)),
            category: language_model_core::ProviderErrorCategory::RateLimit,
        };
        assert_eq!(
            provider.completion_error(error).class(),
            ErrorClass::Throttled
        );
        let ready = resources.wait_for_provider("fake/model");
        futures::pin_mut!(ready);
        assert!(ready.as_mut().now_or_never().is_none());
        ready.await?;
        Ok(())
    }
}
