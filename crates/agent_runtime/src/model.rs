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

const MAX_OUTPUT_BYTES: usize = 8 * 1024 * 1024;
const MAX_TOOL_CALLS: usize = 256;

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
        }
    }

    pub fn with_observer(mut self, observer: ModelEventObserver) -> Self {
        self.observer = Some(observer);
        self
    }
}

#[async_trait]
impl ModelProvider for ZedModelProvider {
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
        let request = convert_request(request, self.template.clone())
            .map_err(|error| fatal(error.to_string()))?;
        let execution = self
            .executor
            .prepare(model_id.to_owned(), request)
            .await
            .map_err(map_error)?;
        let events = execution.execute().await.map_err(map_error)?;
        Ok(convert_stream(events, self.observer.clone()))
    }
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
    let mut tools = HashMap::<String, String>::new();
    for message in request.chat_history {
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
                            converted.push(MessageContent::ToolUse(LanguageModelToolUse {
                                id: call.id.into(),
                                name: call.function.name.into(),
                                raw_input: serde_json::to_string(&call.function.arguments)?,
                                input: LanguageModelToolUseInput::Json(call.function.arguments),
                                is_input_complete: true,
                                thought_signature: None,
                            }));
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
                                    _ => {
                                        return Err(anyhow!(
                                            "opaque provider reasoning requires a protocol extension"
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
                                is_error: false,
                                content,
                                output: None,
                            }));
                        }
                    }
                }
                (Role::User, converted)
            }
        };
        template.messages.push(LanguageModelRequestMessage {
            role,
            content,
            cache: false,
            reasoning_details: None,
        });
    }
    template.tools = request
        .tools
        .into_iter()
        .map(|tool| {
            LanguageModelRequestTool::function(tool.name, tool.description, tool.parameters, false)
        })
        .collect();
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
            !self.stopped || matches!(event, LanguageModelCompletionEvent::UsageUpdate(_)),
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
                    tool.thought_signature.is_none(),
                    "tool thought signatures require a protocol extension"
                );
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
                        self.partial_tools.len() + self.complete_tools.len() <= MAX_TOOL_CALLS,
                        "too many tool calls in one response"
                    );
                    return Ok(());
                }
                let arguments = match tool.input {
                    LanguageModelToolUseInput::Json(arguments) => arguments,
                    LanguageModelToolUseInput::Text(_) => {
                        return Err(anyhow!(
                            "custom text tool calls require a protocol extension"
                        ));
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
                        self.complete_tools.len() <= MAX_TOOL_CALLS,
                        "too many tool calls in one response"
                    );
                    self.pending.push_back(StreamChunk::ToolCall(call));
                }
            }
            LanguageModelCompletionEvent::UsageUpdate(usage) => self.usage = usage,
            LanguageModelCompletionEvent::Stop(StopReason::EndTurn | StopReason::ToolUse) => {
                self.stopped = true
            }
            LanguageModelCompletionEvent::Stop(reason) => {
                return Err(anyhow!(
                    "model response stopped with {reason:?}; an incomplete response cannot be checkpointed"
                ));
            }
            LanguageModelCompletionEvent::ToolUseJsonParseError {
                json_parse_error, ..
            } => return Err(anyhow!("invalid tool arguments: {json_parse_error}")),
            LanguageModelCompletionEvent::ReasoningDetails(_) => {
                return Err(anyhow!(
                    "provider reasoning details require a protocol extension"
                ));
            }
            LanguageModelCompletionEvent::Compaction(_) => {
                return Err(anyhow!("provider compaction requires a protocol extension"));
            }
            LanguageModelCompletionEvent::Queued { .. }
            | LanguageModelCompletionEvent::Started
            | LanguageModelCompletionEvent::StartMessage { .. } => {}
        }
        ensure!(
            self.output_bytes.saturating_add(self.partial_tool_bytes) <= MAX_OUTPUT_BYTES,
            "model output exceeds the runtime byte limit"
        );
        Ok(())
    }
}

pub fn convert_stream(
    events: LanguageModelCompletionStream,
    observer: Option<ModelEventObserver>,
) -> ModelStream {
    let state = StreamState {
        events,
        pending: VecDeque::new(),
        thinking: None,
        partial_tools: HashSet::new(),
        partial_tool_bytes: 0,
        complete_tools: HashMap::new(),
        usage: TokenUsage::default(),
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
                    if let Err(error) = state.accept(event) {
                        state.pending.clear();
                        state.ended = true;
                        return Some((Err(fatal(error.to_string())), state));
                    }
                }
                Some(Err(error)) => {
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
                    let usage = state.usage;
                    state.pending.push_back(StreamChunk::Final(FinalResponse {
                        usage: Some(Usage {
                            input_tokens: usage
                                .input_tokens
                                .saturating_add(usage.cache_creation_input_tokens)
                                .saturating_add(usage.cache_read_input_tokens),
                            output_tokens: usage.output_tokens,
                            total_tokens: usage.total_tokens(),
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
            vec![LanguageModelCompletionEvent::Stop(StopReason::Refusal)],
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
        );
        let mut response = provider
            .invoke_model("fake/model", CompletionRequest::default())
            .await?;
        assert_eq!(execution_thread.await?, std::thread::current().id());
        assert_ne!(
            preparer.join().map_err(|_| anyhow!("preparer panicked"))?,
            std::thread::current().id()
        );
        assert!(response.next().now_or_never().is_none());
        drop(response);
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
}
