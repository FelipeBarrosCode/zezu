pub mod auth_methods;
pub mod commands;
pub mod compaction;
pub mod config_options;
mod connection;
pub mod content;
mod diff;
pub mod elicitation;
mod mention;
pub mod notices;
pub mod prompt_capabilities;
pub mod subagents;
mod submission;
mod terminal;
pub use ::terminal::HeadlessTerminal;
use action_log::{ActionLog, ActionLogTelemetry};
use agent_client_protocol::{schema::MaybeUndefined, v1::schema as acp_v1, v2::schema as acp_v2};
use agent_settings::AgentSettings;
use anyhow::{Context as _, Result, anyhow};
use collections::{HashSet, IndexMap};
pub use connection::*;
pub use diff::*;
use feature_flags::{AcpBetaFeatureFlag, FeatureFlagAppExt as _};
#[cfg(any(test, feature = "test-support"))]
use futures::future::BoxFuture;
use futures::{FutureExt, channel::oneshot};
use gpui::{
    ActivityGuard, AppContext, AsyncApp, Context, Entity, EventEmitter, SharedString, Subscription,
    Task, WeakEntity,
};
use itertools::Itertools;
use language::language_settings::FormatOnSave;
use language::{
    Anchor, Buffer, BufferEditSource, BufferSnapshot, LanguageRegistry, Point, ToPoint, text_diff,
};
use markdown::{Markdown, MarkdownOptions};
pub use mention::*;
use project::lsp_store::{FormatTrigger, LspFormatTarget};
use project::{AgentLocation, Project};
use serde::{Deserialize, Serialize};
use serde_json::to_string_pretty;
use settings::{Settings, SettingsStore};
use std::borrow::Cow;
use std::collections::HashMap;
use std::error::Error;
use std::fmt::{Formatter, Write};
use std::ops::Range;
use std::process::ExitStatus;
use std::rc::Rc;
use std::time::{Duration, Instant};
use std::{fmt::Display, mem, path::PathBuf, sync::Arc};
pub use submission::*;
use task::{Shell, ShellBuilder};
pub use terminal::*;
use text::Bias;
use ui::App;
use util::markdown::{MarkdownCodeBlock, MarkdownEscaped};
use util::path_list::PathList;
use util::{ResultExt, get_default_system_shell_preferring_bash, paths::PathStyle};
use uuid::Uuid;

pub(crate) enum MetadataUpdate {
    V1(MaybeUndefined<acp_v2::Meta>),
    V2(MaybeUndefined<acp_v2::Meta>),
}

impl MetadataUpdate {
    pub(crate) fn apply(self, stored: &mut Option<acp_v2::Meta>) -> bool {
        if matches!(
            &self,
            Self::V1(MaybeUndefined::Undefined) | Self::V2(MaybeUndefined::Undefined)
        ) {
            return false;
        }
        let previous = stored.clone();
        match self {
            Self::V1(update) => {
                *stored = update.take();
            }
            Self::V2(update) => {
                let mut metadata = stored.take().map(MaybeUndefined::Value).unwrap_or_default();
                acp_v2::apply_meta_update(&mut metadata, update);
                *stored = metadata.take();
                // Empty entity metadata and absent metadata have the same meaning in v2.
                return stored.as_ref().filter(|metadata| !metadata.is_empty())
                    != previous.as_ref().filter(|metadata| !metadata.is_empty());
            }
        }
        *stored != previous
    }
}

/// Returned when the model stops because it exhausted its output token budget.
#[derive(Debug)]
pub struct MaxOutputTokensError;

impl std::fmt::Display for MaxOutputTokensError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "output token limit reached")
    }
}

impl std::error::Error for MaxOutputTokensError {}

/// Legacy ACP metadata key used before tool calls had a dedicated name field.
pub const TOOL_NAME_META_KEY: &str = "tool_name";

/// Extracts a tool name from the legacy ACP metadata field.
pub fn tool_name_from_meta(meta: &Option<acp_v2::Meta>) -> Option<SharedString> {
    meta.as_ref()
        .and_then(|meta| meta.get(TOOL_NAME_META_KEY))
        .and_then(|value| value.as_str())
        .map(|name| SharedString::from(name.to_owned()))
}

/// Creates ACP metadata containing the legacy tool-name field.
pub fn meta_with_tool_name(tool_name: &str) -> acp_v2::Meta {
    acp_v2::Meta::from_iter([(TOOL_NAME_META_KEY.into(), tool_name.into())])
}

/// Key used in ACP `AvailableCommand` meta to record which source produced a
/// slash command, so the completion popup can group commands by category.
pub const COMMAND_CATEGORY_META_KEY: &str = "command_category";

/// The source category of a slash command, used to group commands in the
/// completion popup. Only the native Zed agent annotates its commands; commands
/// from external ACP agents carry no category and are grouped on their own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommandCategory {
    /// Built-in Zed agent commands (e.g. `/compact`).
    Native,
    /// Commands sourced from MCP server prompts.
    Mcp,
}

impl CommandCategory {
    fn as_str(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::Mcp => "mcp",
        }
    }

    fn from_str(value: &str) -> Option<Self> {
        match value {
            "native" => Some(Self::Native),
            "mcp" => Some(Self::Mcp),
            _ => None,
        }
    }
}

pub fn meta_with_command_category(category: CommandCategory) -> acp_v2::Meta {
    acp_v2::Meta::from_iter([(COMMAND_CATEGORY_META_KEY.into(), category.as_str().into())])
}

pub fn command_category_from_meta(meta: Option<&acp_v2::Meta>) -> Option<CommandCategory> {
    meta.and_then(|m| m.get(COMMAND_CATEGORY_META_KEY))
        .and_then(|v| v.as_str())
        .and_then(CommandCategory::from_str)
}

/// Key used in ACP ToolCall meta to store the session id and message indexes
pub const SUBAGENT_SESSION_INFO_META_KEY: &str = "subagent_session_info";

pub const SANDBOX_AUTHORIZATION_META_KEY: &str = "sandbox_authorization";

/// Stable `PermissionOption` ids for the sandbox-escalation approval prompt.
///
/// These are shared across the option construction (in the agent), the outcome
/// dispatch, and the UI so the distinct grant lifetimes stay in sync. Note
/// that `AllowThread` and `AllowAlways` both use
/// `PermissionOptionKind::AllowAlways`; the id is what distinguishes them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SandboxPermission {
    AllowOnce,
    AllowThread,
    AllowAlways,
    Deny,
}

impl SandboxPermission {
    pub fn as_id(self) -> &'static str {
        match self {
            Self::AllowOnce => "allow",
            Self::AllowThread => "allow_thread",
            Self::AllowAlways => "allow_always",
            Self::Deny => "deny",
        }
    }

    pub fn from_id(id: &str) -> Option<Self> {
        match id {
            "allow" => Some(Self::AllowOnce),
            "allow_thread" => Some(Self::AllowThread),
            "allow_always" => Some(Self::AllowAlways),
            "deny" => Some(Self::Deny),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct SandboxAuthorizationDetails {
    #[serde(default)]
    pub command: Option<String>,
    /// Specific hosts the command requested network access to, in canonical
    /// form (`github.com`, `*.npmjs.org`). Empty when no specific hosts were
    /// requested (see `network_all_hosts`).
    #[serde(default)]
    pub network_hosts: Vec<String>,
    /// Whether the command requested access to any host ("arbitrary network
    /// access"). The `network` alias deserializes the field this replaced —
    /// a plain bool meaning "network access" — so details persisted by older
    /// builds still render the network request.
    #[serde(default, alias = "network")]
    pub network_all_hosts: bool,

    #[serde(default)]
    pub allow_fs_write_all: bool,
    #[serde(default)]
    pub unsandboxed: bool,
    #[serde(default)]
    pub write_paths: Vec<settings::GrantedWritePath>,
    /// Windows/WSL only: the command will write to a path on a Windows drive
    /// (DrvFs), whose sandbox-integrity guarantees are weaker. Drives the
    /// weaker-guarantee warning banner in the approval prompt.
    #[serde(default)]
    pub warn_windows_fs: bool,
    /// The agent-provided justification for requesting these permissions,
    /// shown to the user (attributed to the agent) in the approval prompt.
    #[serde(default)]
    pub reason: String,
}

pub fn meta_with_sandbox_authorization(details: SandboxAuthorizationDetails) -> acp_v2::Meta {
    acp_v2::Meta::from_iter([(
        SANDBOX_AUTHORIZATION_META_KEY.into(),
        serde_json::to_value(details).unwrap_or_default(),
    )])
}

pub fn sandbox_authorization_details_from_meta(
    meta: &Option<acp_v2::Meta>,
) -> Option<SandboxAuthorizationDetails> {
    meta.as_ref()
        .and_then(|m| m.get(SANDBOX_AUTHORIZATION_META_KEY))
        .and_then(|v| serde_json::from_value(v.clone()).ok())
}

pub const SANDBOX_FALLBACK_AUTHORIZATION_META_KEY: &str = "sandbox_fallback_authorization";

/// Stable `PermissionOption` id for the "Retry" choice in the sandbox
/// *fallback* prompt (shown when the OS sandbox can't be created on this
/// system). The remaining choices reuse the [`SandboxPermission`] ids.
pub const SANDBOX_FALLBACK_RETRY_OPTION_ID: &str = "retry";

/// Details shown when the OS sandbox could not be created for a command and
/// the user is asked whether to run it without a sandbox. Distinct from
/// [`SandboxAuthorizationDetails`] (a model-requested *escalation*): here the
/// sandbox itself failed, so the prompt explains why and offers a retry.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct SandboxFallbackAuthorizationDetails {
    #[serde(default)]
    pub command: Option<String>,
    /// Human-readable reason the OS sandbox could not be created (for example,
    /// "bwrap not found on PATH"), shown to the user so they can decide
    /// whether to run the command without a sandbox.
    #[serde(default)]
    pub reason: String,
    /// Slug of the sandboxing docs section that best explains how to fix this
    /// failure (see [`crate::LinuxWslSandboxError::docs_section`]), rendered as a
    /// "Learn more" link. `None` when the cause is unknown.
    #[serde(default)]
    pub docs_section: Option<String>,
}

pub fn meta_with_sandbox_fallback_authorization(
    details: SandboxFallbackAuthorizationDetails,
) -> acp_v2::Meta {
    acp_v2::Meta::from_iter([(
        SANDBOX_FALLBACK_AUTHORIZATION_META_KEY.into(),
        serde_json::to_value(details).unwrap_or_default(),
    )])
}

pub fn sandbox_fallback_authorization_details_from_meta(
    meta: &Option<acp_v2::Meta>,
) -> Option<SandboxFallbackAuthorizationDetails> {
    meta.as_ref()
        .and_then(|m| m.get(SANDBOX_FALLBACK_AUTHORIZATION_META_KEY))
        .and_then(|v| serde_json::from_value(v.clone()).ok())
}

/// Meta key recording why the OS sandbox was not applied to a terminal tool
/// call, even though sandboxing was active for the thread. The value is a
/// serialized [`SandboxNotAppliedReason`]. Surfaced as a warning in the UI and
/// used to explain the situation to both the user and the agent.
pub const SANDBOX_NOT_APPLIED_META_KEY: &str = "sandbox_not_applied";

pub fn meta_with_sandbox_not_applied(reason: &SandboxNotAppliedReason) -> acp_v2::Meta {
    acp_v2::Meta::from_iter([(
        SANDBOX_NOT_APPLIED_META_KEY.into(),
        serde_json::to_value(reason).unwrap_or_default(),
    )])
}

pub fn sandbox_not_applied_from_meta(
    meta: &Option<acp_v2::Meta>,
) -> Option<SandboxNotAppliedReason> {
    meta.as_ref()
        .and_then(|m| m.get(SANDBOX_NOT_APPLIED_META_KEY))
        .and_then(|v| serde_json::from_value(v.clone()).ok())
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SubagentSessionInfo {
    /// The session id of the subagent sessiont that was spawned
    pub session_id: acp_v2::SessionId,
    /// The index of the message of the start of the "turn" run by this tool call
    pub message_start_index: usize,
    /// The index of the output of the message that the subagent has returned
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message_end_index: Option<usize>,
}

/// Helper to extract subagent session id from ACP meta
pub fn subagent_session_info_from_meta(meta: &Option<acp_v2::Meta>) -> Option<SubagentSessionInfo> {
    meta.as_ref()
        .and_then(|m| m.get(SUBAGENT_SESSION_INFO_META_KEY))
        .and_then(|v| serde_json::from_value(v.clone()).ok())
}

#[derive(Debug)]
pub struct UserMessage {
    pub identity: MessageIdentity,
    pub meta: Option<acp_v2::Meta>,
    pub client_id: Option<ClientUserMessageId>,
    pub is_optimistic: bool,
    pub content: MessageContent,
    pub indented: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MessageIdentity {
    /// Legacy IDs delimit adjacent chunks; they do not identify a global upsert target.
    Legacy(Option<acp_v1::MessageId>),
    /// Identifies an upsert target within its message kind, retaining its transcript position.
    Keyed(acp_v2::MessageId),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MessageKind {
    User,
    Assistant,
    Thought,
    Session,
}

impl UserMessage {
    fn to_markdown(&self, cx: &App) -> String {
        let mut markdown = String::new();
        writeln!(markdown, "## User").unwrap();
        writeln!(markdown).unwrap();
        writeln!(markdown, "{}", self.content.to_markdown(cx)).unwrap();
        writeln!(markdown).unwrap();
        markdown
    }
}

/// A message between this session and another, such as a parent and its subagent. The
/// session that sent it is the outgoing view; the session that received it is the incoming
/// view. Either participant may be unknown.
#[derive(Debug)]
pub struct SessionMessage {
    pub identity: MessageIdentity,
    pub sender_session_id: Option<acp_v2::SessionId>,
    pub recipient_session_id: Option<acp_v2::SessionId>,
    /// On an incoming view, the ID of the outgoing view in the sender's transcript.
    pub sender_message_id: Option<acp_v2::MessageId>,
    /// On an outgoing view, the ID of the incoming view in the recipient's transcript.
    pub recipient_message_id: Option<acp_v2::MessageId>,
    pub meta: Option<acp_v2::Meta>,
    pub content: MessageContent,
}

/// Which side of a session message a transcript shows, and the other participant when it
/// is known.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SessionMessageSide<'a> {
    Outgoing {
        recipient: Option<&'a acp_v2::SessionId>,
    },
    Incoming {
        sender: Option<&'a acp_v2::SessionId>,
    },
    Unknown,
}

impl SessionMessage {
    /// The side of this message that the transcript of `own_session_id` shows. That session
    /// is always one participant, so a single known participant that is another session
    /// also tells the direction.
    pub fn side<'a>(&'a self, own_session_id: &acp_v2::SessionId) -> SessionMessageSide<'a> {
        let sender = self.sender_session_id.as_ref();
        let recipient = self.recipient_session_id.as_ref();
        if sender == Some(own_session_id) {
            SessionMessageSide::Outgoing { recipient }
        } else if recipient == Some(own_session_id) || sender.is_some() {
            SessionMessageSide::Incoming { sender }
        } else if recipient.is_some() {
            SessionMessageSide::Outgoing { recipient }
        } else {
            SessionMessageSide::Unknown
        }
    }

    fn to_markdown(&self, cx: &App) -> String {
        let heading = match (&self.sender_session_id, &self.recipient_session_id) {
            (Some(sender), Some(recipient)) => {
                format!("## Message from `{sender}` to `{recipient}`")
            }
            (Some(sender), None) => format!("## Message from `{sender}`"),
            (None, Some(recipient)) => format!("## Message to `{recipient}`"),
            (None, None) => "## Session message".to_string(),
        };
        format!("{heading}\n\n{}\n\n", self.content.to_markdown(cx))
    }
}

#[derive(Debug, PartialEq)]
pub struct AssistantMessage {
    pub chunks: Vec<AssistantMessageChunk>,
    pub indented: bool,
    pub is_subagent_output: bool,
}

impl AssistantMessage {
    pub fn to_markdown(&self, cx: &App) -> String {
        format!(
            "## Assistant\n\n{}\n\n",
            self.chunks
                .iter()
                .map(|chunk| chunk.to_markdown(cx))
                .join("\n\n")
        )
    }
}

#[derive(Debug, PartialEq)]
pub enum AssistantMessageChunk {
    Message {
        identity: MessageIdentity,
        meta: Option<acp_v2::Meta>,
        block: MessageContent,
    },
    Thought {
        identity: MessageIdentity,
        meta: Option<acp_v2::Meta>,
        block: MessageContent,
    },
}

impl AssistantMessageChunk {
    pub fn from_str(
        chunk: &str,
        language_registry: &Arc<LanguageRegistry>,
        path_style: PathStyle,
        cx: &mut App,
    ) -> Self {
        Self::Message {
            identity: MessageIdentity::Legacy(None),
            meta: None,
            block: MessageContent::new(chunk.into(), language_registry, path_style, cx),
        }
    }

    fn identity(&self) -> &MessageIdentity {
        match self {
            Self::Message { identity, .. } | Self::Thought { identity, .. } => identity,
        }
    }

    fn kind(&self) -> MessageKind {
        match self {
            Self::Message { .. } => MessageKind::Assistant,
            Self::Thought { .. } => MessageKind::Thought,
        }
    }

    fn to_markdown(&self, cx: &App) -> String {
        match self {
            Self::Message { block, .. } => block.to_markdown(cx),
            Self::Thought { block, .. } => {
                format!("<thinking>\n{}\n</thinking>", block.to_markdown(cx))
            }
        }
    }
}

fn can_merge_message_chunks(
    existing: Option<&acp_v1::MessageId>,
    incoming: Option<&acp_v1::MessageId>,
) -> bool {
    match (existing, incoming) {
        (Some(existing), Some(incoming)) => existing == incoming,
        _ => true,
    }
}

#[derive(Clone, Copy)]
enum MessageLocation {
    User {
        entry_index: usize,
    },
    Assistant {
        entry_index: usize,
        chunk_index: usize,
    },
    Session {
        entry_index: usize,
    },
}

impl MessageLocation {
    fn entry_index(self) -> usize {
        match self {
            Self::User { entry_index }
            | Self::Assistant { entry_index, .. }
            | Self::Session { entry_index } => entry_index,
        }
    }

    fn fields_mut(
        self,
        entries: &mut [AgentThreadEntry],
    ) -> Option<(&mut MessageContent, &mut Option<acp_v2::Meta>)> {
        match (self, entries.get_mut(self.entry_index())?) {
            (Self::User { .. }, AgentThreadEntry::UserMessage(message)) => {
                Some((&mut message.content, &mut message.meta))
            }
            (Self::Assistant { chunk_index, .. }, AgentThreadEntry::AssistantMessage(message)) => {
                match message.chunks.get_mut(chunk_index)? {
                    AssistantMessageChunk::Message { block, meta, .. }
                    | AssistantMessageChunk::Thought { block, meta, .. } => Some((block, meta)),
                }
            }
            (Self::Session { .. }, AgentThreadEntry::SessionMessage(message)) => {
                Some((&mut message.content, &mut message.meta))
            }
            _ => None,
        }
    }

    fn is_streaming_target(self, target: &StreamingTextTarget) -> bool {
        matches!(
            self,
            Self::Assistant { entry_index, chunk_index }
                if entry_index == target.entry_index && chunk_index == target.chunk_index
        )
    }
}

#[derive(Debug)]
pub enum AgentThreadEntry {
    UserMessage(UserMessage),
    AssistantMessage(AssistantMessage),
    SessionMessage(SessionMessage),
    ToolCall(ToolCall),
    Elicitation(ElicitationEntryId),
    ContextCompaction(ContextCompaction),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ElicitationEntryId(pub Arc<str>);

#[derive(Debug)]
pub struct Elicitation {
    pub id: ElicitationEntryId,
    pub request: acp_v2::CreateElicitationRequest,
    pub status: ElicitationStatus,
}

#[derive(Debug)]
pub enum ElicitationStatus {
    Pending {
        respond_tx: oneshot::Sender<acp_v2::CreateElicitationResponse>,
    },
    Accepted,
    Declined,
    Canceled,
    Completed,
}

enum ElicitationChange {
    Responded,
    Updated,
}

#[derive(Clone, Debug)]
pub enum ElicitationStoreEvent {
    ElicitationRequested(ElicitationEntryId),
    /// The request left `Pending`; this does not imply delivery to its response waiter.
    ElicitationResponded(ElicitationEntryId),
    ElicitationUpdated(ElicitationEntryId),
}

#[derive(Default)]
pub struct ElicitationStore {
    elicitations: Vec<Elicitation>,
}

impl EventEmitter<ElicitationStoreEvent> for ElicitationStore {}

impl ElicitationStore {
    pub fn elicitations(&self) -> &[Elicitation] {
        &self.elicitations
    }

    fn validate_request(request: &acp_v2::CreateElicitationRequest) -> Result<(), acp_v2::Error> {
        match &request.mode {
            acp_v2::ElicitationMode::Form(mode) => validate_form_schema(&mode.requested_schema)?,
            acp_v2::ElicitationMode::Url(mode) => {
                let url = url::Url::parse(&mode.url)
                    .map_err(|_| acp_v2::Error::invalid_params().data("invalid elicitation URL"))?;
                if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
                    return Err(acp_v2::Error::invalid_params()
                        .data("elicitation URL must use HTTP or HTTPS and include a host"));
                }
            }
            _ => {
                return Err(acp_v2::Error::invalid_params().data("unsupported elicitation mode"));
            }
        }

        Ok(())
    }

    fn insert_pending_elicitation(
        &mut self,
        request: acp_v2::CreateElicitationRequest,
    ) -> (
        ElicitationEntryId,
        oneshot::Receiver<acp_v2::CreateElicitationResponse>,
    ) {
        let (respond_tx, response_rx) = oneshot::channel();
        let id = ElicitationEntryId(Uuid::new_v4().to_string().into());
        self.elicitations.push(Elicitation {
            id: id.clone(),
            request,
            status: ElicitationStatus::Pending { respond_tx },
        });
        (id, response_rx)
    }

    fn response_task(
        response_rx: oneshot::Receiver<acp_v2::CreateElicitationResponse>,
        cx: &App,
    ) -> Task<acp_v2::CreateElicitationResponse> {
        cx.foreground_executor().spawn(async move {
            response_rx.await.unwrap_or_else(|oneshot::Canceled| {
                acp_v2::CreateElicitationResponse::new(acp_v2::ElicitationAction::Cancel)
            })
        })
    }

    fn emit_change(id: ElicitationEntryId, change: ElicitationChange, cx: &mut Context<Self>) {
        cx.emit(ElicitationStoreEvent::ElicitationUpdated(id.clone()));
        if matches!(change, ElicitationChange::Responded) {
            cx.emit(ElicitationStoreEvent::ElicitationResponded(id));
        }
    }

    fn respond_to_elicitation_entry(
        elicitation: &mut Elicitation,
        response: acp_v2::CreateElicitationResponse,
    ) -> bool {
        if !matches!(elicitation.status, ElicitationStatus::Pending { .. }) {
            return false;
        }
        let ElicitationStatus::Pending { respond_tx } = mem::replace(
            &mut elicitation.status,
            elicitation_status_for_response(&response),
        ) else {
            return false;
        };
        if respond_tx.send(response).is_err() {
            log::debug!("Elicitation waiter closed before its response was delivered");
        }
        true
    }

    fn complete_url_elicitation_entry(elicitation: &mut Elicitation) -> bool {
        let previous_status = mem::replace(&mut elicitation.status, ElicitationStatus::Completed);
        match previous_status {
            ElicitationStatus::Accepted => true,
            previous_status @ (ElicitationStatus::Pending { .. }
            | ElicitationStatus::Declined
            | ElicitationStatus::Canceled
            | ElicitationStatus::Completed) => {
                elicitation.status = previous_status;
                false
            }
        }
    }

    fn cancel_elicitation_entry(elicitation: &mut Elicitation) -> Option<ElicitationChange> {
        match mem::replace(&mut elicitation.status, ElicitationStatus::Canceled) {
            ElicitationStatus::Pending { respond_tx } => {
                if respond_tx
                    .send(acp_v2::CreateElicitationResponse::new(
                        acp_v2::ElicitationAction::Cancel,
                    ))
                    .is_err()
                {
                    log::debug!("Elicitation waiter closed before cancellation was delivered");
                }
                Some(ElicitationChange::Responded)
            }
            ElicitationStatus::Accepted
                if matches!(&elicitation.request.mode, acp_v2::ElicitationMode::Url(_)) =>
            {
                Some(ElicitationChange::Updated)
            }
            previous_status => {
                elicitation.status = previous_status;
                None
            }
        }
    }

    fn respond_to_elicitation_by_id(
        &mut self,
        id: &ElicitationEntryId,
        response: acp_v2::CreateElicitationResponse,
    ) -> bool {
        let Some((_, elicitation)) = self.elicitation_mut(id) else {
            return false;
        };
        Self::respond_to_elicitation_entry(elicitation, response)
    }

    fn complete_url_elicitation_by_id(&mut self, id: &ElicitationEntryId) -> bool {
        let Some((_, elicitation)) = self.elicitation_mut(id) else {
            return false;
        };
        Self::complete_url_elicitation_entry(elicitation)
    }

    fn cancel_elicitation_by_id(&mut self, id: &ElicitationEntryId) -> Option<ElicitationChange> {
        let (_, elicitation) = self.elicitation_mut(id)?;
        Self::cancel_elicitation_entry(elicitation)
    }

    pub fn request_elicitation(
        &mut self,
        request: acp_v2::CreateElicitationRequest,
        cx: &mut Context<Self>,
    ) -> Result<Task<acp_v2::CreateElicitationResponse>, acp_v2::Error> {
        self.request_elicitation_with_id(request, cx)
            .map(|(_, task)| task)
    }

    pub fn request_elicitation_with_id(
        &mut self,
        request: acp_v2::CreateElicitationRequest,
        cx: &mut Context<Self>,
    ) -> Result<(ElicitationEntryId, Task<acp_v2::CreateElicitationResponse>), acp_v2::Error> {
        Self::validate_request(&request)?;
        let (id, response_rx) = self.insert_pending_elicitation(request);
        cx.emit(ElicitationStoreEvent::ElicitationRequested(id.clone()));
        cx.notify();

        let task = Self::response_task(response_rx, cx);
        Ok((id, task))
    }

    pub fn respond_to_elicitation(
        &mut self,
        id: &ElicitationEntryId,
        response: acp_v2::CreateElicitationResponse,
        cx: &mut Context<Self>,
    ) {
        if !self.respond_to_elicitation_by_id(id, response) {
            return;
        }

        Self::emit_change(id.clone(), ElicitationChange::Responded, cx);
        cx.notify();
    }

    pub fn complete_url_elicitation(
        &mut self,
        elicitation_id: &acp_v2::ElicitationId,
        cx: &mut Context<Self>,
    ) {
        let Some(entry_id) = self.entry_id_for_url_elicitation(elicitation_id) else {
            return;
        };
        if !self.complete_url_elicitation_by_id(&entry_id) {
            return;
        }

        cx.emit(ElicitationStoreEvent::ElicitationUpdated(entry_id));
        cx.notify();
    }

    pub fn cancel_elicitation(&mut self, id: &ElicitationEntryId, cx: &mut Context<Self>) {
        let Some(change) = self.cancel_elicitation_by_id(id) else {
            return;
        };

        Self::emit_change(id.clone(), change, cx);
        cx.notify();
    }

    pub fn cancel_all(&mut self, cx: &mut Context<Self>) {
        for (id, change) in self.cancel_pending(|_| true) {
            Self::emit_change(id, change, cx);
        }
        cx.notify();
    }

    pub fn clear(&mut self, cx: &mut Context<Self>) {
        let changes = self.cancel_pending(|_| true);
        self.elicitations.clear();
        for (id, change) in changes {
            Self::emit_change(id, change, cx);
        }
        cx.notify();
    }

    pub fn clear_resolved(&mut self, cx: &mut Context<Self>) -> Vec<ElicitationEntryId> {
        let mut cleared_ids = Vec::new();
        self.elicitations.retain(|elicitation| {
            let keep = matches!(
                (&elicitation.status, &elicitation.request.mode),
                (ElicitationStatus::Pending { .. }, _)
                    | (ElicitationStatus::Accepted, acp_v2::ElicitationMode::Url(_))
            );
            if !keep {
                cleared_ids.push(elicitation.id.clone());
            }
            keep
        });

        if !cleared_ids.is_empty() {
            for id in &cleared_ids {
                cx.emit(ElicitationStoreEvent::ElicitationUpdated(id.clone()));
            }
            cx.notify();
        }

        cleared_ids
    }

    pub fn cancel_request(&mut self, request_id: &acp_v2::RequestId, cx: &mut Context<Self>) {
        self.cancel_request_entries(request_id, true, cx);
    }

    pub fn finish_request(&mut self, request_id: &acp_v2::RequestId, cx: &mut Context<Self>) {
        self.cancel_request_entries(request_id, false, cx);
    }

    fn cancel_request_entries(
        &mut self,
        request_id: &acp_v2::RequestId,
        include_accepted_urls: bool,
        cx: &mut Context<Self>,
    ) {
        let changes = self.cancel_pending(|elicitation| {
            matches!(
                elicitation.request.scope(),
                acp_v2::ElicitationScope::Request(scope) if &scope.request_id == request_id
            ) && (include_accepted_urls
                || matches!(elicitation.status, ElicitationStatus::Pending { .. }))
        });
        for (id, change) in changes {
            Self::emit_change(id, change, cx);
        }
        cx.notify();
    }

    pub fn elicitation(&self, id: &ElicitationEntryId) -> Option<(usize, &Elicitation)> {
        self.elicitations
            .iter()
            .enumerate()
            .rev()
            .find_map(|(index, elicitation)| {
                (&elicitation.id == id).then_some((index, elicitation))
            })
    }

    fn entry_id_for_url_elicitation(
        &self,
        elicitation_id: &acp_v2::ElicitationId,
    ) -> Option<ElicitationEntryId> {
        self.elicitations.iter().rev().find_map(|elicitation| {
            if let acp_v2::ElicitationMode::Url(mode) = &elicitation.request.mode
                && &mode.elicitation_id == elicitation_id
            {
                Some(elicitation.id.clone())
            } else {
                None
            }
        })
    }

    fn elicitation_mut(&mut self, id: &ElicitationEntryId) -> Option<(usize, &mut Elicitation)> {
        self.elicitations
            .iter_mut()
            .enumerate()
            .rev()
            .find_map(|(index, elicitation)| {
                (&elicitation.id == id).then_some((index, elicitation))
            })
    }

    fn cancel_pending(
        &mut self,
        mut should_cancel: impl FnMut(&Elicitation) -> bool,
    ) -> Vec<(ElicitationEntryId, ElicitationChange)> {
        let mut changes = Vec::new();
        for elicitation in &mut self.elicitations {
            if should_cancel(elicitation)
                && let Some(change) = Self::cancel_elicitation_entry(elicitation)
            {
                changes.push((elicitation.id.clone(), change));
            }
        }
        changes
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextCompactionId(pub Arc<str>);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContextCompactionStatus {
    InProgress,
    Completed,
    Failed,
    Canceled,
    Other(Arc<str>),
}

impl From<acp_v2::CompactionStatus> for ContextCompactionStatus {
    fn from(status: acp_v2::CompactionStatus) -> Self {
        match status {
            acp_v2::CompactionStatus::InProgress => Self::InProgress,
            acp_v2::CompactionStatus::Completed => Self::Completed,
            acp_v2::CompactionStatus::Failed => Self::Failed,
            acp_v2::CompactionStatus::Cancelled => Self::Canceled,
            acp_v2::CompactionStatus::Other(status) => Self::Other(status.into()),
            _ => Self::Other("unknown".into()),
        }
    }
}

/// A point in the thread where the conversation history was compacted to free
/// up room in the model's context window. The summary can be expanded to inspect
/// what the model retained.
#[derive(Debug)]
pub struct ContextCompaction {
    pub id: ContextCompactionId,
    pub status: ContextCompactionStatus,
    pub error: Option<Entity<Markdown>>,
    pub summary: MessageContent,
    pub meta: Option<acp_v2::Meta>,
}

impl ContextCompaction {
    pub fn is_in_progress(&self) -> bool {
        self.status == ContextCompactionStatus::InProgress
    }

    fn apply_update(
        &mut self,
        update: acp_v2::CompactionUpdate,
        metadata: MetadataUpdate,
        language_registry: &Arc<LanguageRegistry>,
        cx: &mut App,
    ) {
        self.status = update.status.into();
        match update.summary {
            MaybeUndefined::Undefined => {}
            MaybeUndefined::Null => self.summary = MessageContent::default(),
            MaybeUndefined::Value(blocks) => {
                self.summary = MessageContent::default();
                for block in blocks {
                    self.append_summary(block, language_registry, cx);
                }
            }
        }
        match update.error {
            MaybeUndefined::Undefined => {}
            MaybeUndefined::Null => self.error = None,
            MaybeUndefined::Value(error) => {
                if let Some(markdown) = &self.error {
                    markdown.update(cx, |markdown, cx| markdown.reset(error.into(), cx));
                } else {
                    self.error = Some(cx.new(|cx| Markdown::new_text(error.into(), cx)));
                }
            }
        }
        metadata.apply(&mut self.meta);
    }

    fn append_summary(
        &mut self,
        content: acp_v2::ContentBlock,
        language_registry: &Arc<LanguageRegistry>,
        cx: &mut App,
    ) {
        self.summary
            .append_compaction_summary(content, language_registry, cx);
    }
}

#[derive(Debug)]
pub struct ContextCompactionUpdate {
    pub id: ContextCompactionId,
    pub summary_delta: String,
    pub status: Option<ContextCompactionStatus>,
}

impl AgentThreadEntry {
    pub fn is_indented(&self) -> bool {
        match self {
            Self::UserMessage(message) => message.indented,
            Self::AssistantMessage(message) => message.indented,
            Self::SessionMessage(_) => false,
            Self::ToolCall(_) => false,
            Self::Elicitation(_) => false,
            Self::ContextCompaction(_) => false,
        }
    }

    pub fn has_pending_markdown_render(&self, cx: &App) -> bool {
        let has_pending_render =
            |markdown: &Entity<Markdown>| markdown.read(cx).has_pending_render();
        match self {
            Self::UserMessage(message) => message.content.markdowns().any(has_pending_render),
            Self::AssistantMessage(message) => message.chunks.iter().any(|chunk| {
                let (AssistantMessageChunk::Message { block, .. }
                | AssistantMessageChunk::Thought { block, .. }) = chunk;
                block.markdowns().any(has_pending_render)
            }),
            Self::SessionMessage(message) => message.content.markdowns().any(has_pending_render),
            Self::ToolCall(call) => {
                has_pending_render(&call.label)
                    || call
                        .raw_input_markdown
                        .as_ref()
                        .is_some_and(has_pending_render)
                    || call
                        .content()
                        .iter()
                        .filter_map(ToolCallContent::markdown)
                        .any(has_pending_render)
            }
            Self::ContextCompaction(compaction) => {
                compaction.error.as_ref().is_some_and(has_pending_render)
                    || compaction.summary.markdowns().any(has_pending_render)
            }
            Self::Elicitation(_) => false,
        }
    }

    pub fn to_markdown(&self, cx: &App) -> String {
        match self {
            Self::UserMessage(message) => message.to_markdown(cx),
            Self::AssistantMessage(message) => message.to_markdown(cx),
            Self::SessionMessage(message) => message.to_markdown(cx),
            Self::ToolCall(tool_call) => tool_call.to_markdown(cx),
            Self::Elicitation(_) => "## Input Requested\n\n".to_string(),
            Self::ContextCompaction(compaction) => {
                let status = match &compaction.status {
                    ContextCompactionStatus::InProgress => "In Progress",
                    ContextCompactionStatus::Completed => "Completed",
                    ContextCompactionStatus::Failed => "Failed",
                    ContextCompactionStatus::Canceled => "Canceled",
                    ContextCompactionStatus::Other(status) => status,
                };
                let mut markdown =
                    format!("## Context Compaction ({})\n\n", MarkdownEscaped(status));
                for block in compaction.summary.blocks() {
                    markdown.push_str(block.to_markdown(cx));
                    markdown.push_str("\n\n");
                }
                if let Some(error) = &compaction.error {
                    markdown.push_str("**Error:** ");
                    markdown.push_str(&MarkdownEscaped(error.read(cx).source()).to_string());
                    markdown.push_str("\n\n");
                }
                markdown
            }
        }
    }

    pub fn user_message(&self) -> Option<&UserMessage> {
        if let AgentThreadEntry::UserMessage(message) = self {
            Some(message)
        } else {
            None
        }
    }

    pub fn diffs(&self) -> impl Iterator<Item = &Entity<Diff>> {
        if let AgentThreadEntry::ToolCall(call) = self {
            itertools::Either::Left(call.diffs())
        } else {
            itertools::Either::Right(std::iter::empty())
        }
    }

    pub fn terminals(&self) -> impl Iterator<Item = &Entity<Terminal>> {
        if let AgentThreadEntry::ToolCall(call) = self {
            itertools::Either::Left(call.terminals())
        } else {
            itertools::Either::Right(std::iter::empty())
        }
    }

    pub fn location(&self, ix: usize) -> Option<(ToolCallLocation, AgentLocation)> {
        if let AgentThreadEntry::ToolCall(ToolCall {
            locations,
            resolved_locations,
            ..
        }) = self
        {
            Some((
                locations.get(ix)?.clone(),
                resolved_locations.get(ix)?.clone()?,
            ))
        } else {
            None
        }
    }
}

/// Native tools can supply relative paths, unlike v2 protocol locations.
/// Keep the raw path here; resolved editor locations are a separate projection.
#[derive(Clone, Debug, Eq)]
pub struct ToolCallLocation {
    pub path: PathBuf,
    pub line: Option<u32>,
    pub meta: Option<acp_v2::Meta>,
}

impl PartialEq for ToolCallLocation {
    fn eq(&self, other: &Self) -> bool {
        // Path equality compares components, hiding changes to the raw spelling.
        self.path.as_os_str() == other.path.as_os_str()
            && self.line == other.line
            && self.meta == other.meta
    }
}

impl From<acp_v1::ToolCallLocation> for ToolCallLocation {
    fn from(location: acp_v1::ToolCallLocation) -> Self {
        Self {
            path: location.path,
            line: location.line,
            meta: location.meta,
        }
    }
}

impl From<acp_v2::ToolCallLocation> for ToolCallLocation {
    fn from(location: acp_v2::ToolCallLocation) -> Self {
        Self {
            path: location.path.0,
            line: location.line.map(std::num::NonZeroU32::get),
            meta: location.meta,
        }
    }
}

#[derive(Debug)]
pub struct ToolCall {
    pub id: acp_v2::ToolCallId,
    pub label: Entity<Markdown>,
    title: Option<SharedString>,
    pub name: Option<SharedString>,
    pub reported_kind: Option<acp_v2::ToolKind>,
    pub reported_status: Option<acp_v2::ToolCallStatus>,
    pub meta: Option<acp_v2::Meta>,
    structured_content: Vec<ToolCallContent>,
    local_status: Option<ToolCallStatus>,
    authorization: Option<PermissionRequestId>,
    pub locations: Vec<ToolCallLocation>,
    pub resolved_locations: Vec<Option<AgentLocation>>,
    pub raw_input: Option<serde_json::Value>,
    pub raw_input_markdown: Option<Entity<Markdown>>,
    pub raw_output: Option<serde_json::Value>,
    raw_output_content: Option<Box<ToolCallContent>>,
    /// Whether the agent accepts `tool_call/cancel` for this tool call.
    advertises_cancel: bool,
    pub tool_name: Option<SharedString>,
    pub subagent_session_info: Option<SubagentSessionInfo>,
    pub sandbox_authorization_details: Option<SandboxAuthorizationDetails>,
    pub sandbox_fallback_authorization_details: Option<SandboxFallbackAuthorizationDetails>,
    /// Why this terminal command ran without the OS sandbox even though
    /// sandboxing was active (see [`SANDBOX_NOT_APPLIED_META_KEY`]). `None` when
    /// the command was sandboxed normally (or sandboxing was off).
    pub sandbox_not_applied: Option<SandboxNotAppliedReason>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PermissionRequestId(Uuid);

#[derive(Debug)]
pub struct PermissionRequest {
    pub id: PermissionRequestId,
    data: PermissionRequestData,
}

#[derive(Debug)]
enum PermissionRequestData {
    LegacyTool {
        tool_call_id: acp_v2::ToolCallId,
        options: PermissionOptions,
        kind: AuthorizationKind,
        respond_tx: oneshot::Sender<RequestPermissionOutcome>,
    },
    Generic {
        request: acp_v2::RequestPermissionRequest,
        respond_tx: oneshot::Sender<acp_v2::RequestPermissionOutcome>,
    },
}

impl PermissionRequest {
    pub fn legacy_tool_call_id(&self) -> Option<&acp_v2::ToolCallId> {
        match &self.data {
            PermissionRequestData::LegacyTool { tool_call_id, .. } => Some(tool_call_id),
            PermissionRequestData::Generic { .. } => None,
        }
    }

    pub fn legacy_options(&self) -> Option<&PermissionOptions> {
        match &self.data {
            PermissionRequestData::LegacyTool { options, .. } => Some(options),
            PermissionRequestData::Generic { .. } => None,
        }
    }

    pub fn legacy_kind(&self) -> Option<AuthorizationKind> {
        match &self.data {
            PermissionRequestData::LegacyTool { kind, .. } => Some(*kind),
            PermissionRequestData::Generic { .. } => None,
        }
    }

    pub fn generic_request(&self) -> Option<&acp_v2::RequestPermissionRequest> {
        match &self.data {
            PermissionRequestData::LegacyTool { .. } => None,
            PermissionRequestData::Generic { request, .. } => Some(request),
        }
    }
}

struct ToolCallPatch {
    title: MaybeUndefined<String>,
    name: MaybeUndefined<String>,
    kind: MaybeUndefined<acp_v2::ToolKind>,
    status: MaybeUndefined<acp_v2::ToolCallStatus>,
    content: MaybeUndefined<ToolContentPatch>,
    locations: MaybeUndefined<Vec<ToolCallLocation>>,
    raw_input: MaybeUndefined<serde_json::Value>,
    raw_output: MaybeUndefined<serde_json::Value>,
    capabilities: MaybeUndefined<acp_v2::ToolCallCapabilities>,
    meta: ToolMetadataPatch,
}

enum ToolContentPatch {
    V1(Vec<acp_v1::ToolCallContent>),
    V2(Vec<acp_v2::ToolCallContent>),
}

#[derive(Clone, Copy)]
struct ToolTerminalResolver<'a> {
    terminals: &'a HashMap<acp_v2::TerminalId, Entity<Terminal>>,
    client_managed_only: bool,
}

impl<'a> ToolTerminalResolver<'a> {
    fn registered(terminals: &'a HashMap<acp_v2::TerminalId, Entity<Terminal>>) -> Self {
        Self {
            terminals,
            client_managed_only: false,
        }
    }

    fn resolve(self, id: &acp_v2::TerminalId, cx: &App) -> Result<Entity<Terminal>> {
        let terminal = self
            .terminals
            .get(id)
            .cloned()
            .ok_or_else(|| anyhow!("Terminal with id `{id}` not found"))?;
        if self.client_managed_only {
            // Process ownership survives completion and does not depend on
            // whether the renderer still has an active PTY.
            anyhow::ensure!(
                terminal.read(cx).is_process_backed(),
                "Client-managed tool content cannot reference an agent-owned display terminal"
            );
        }
        Ok(terminal)
    }
}

impl ToolContentPatch {
    fn prepare(
        self,
        terminals: ToolTerminalResolver<'_>,
        cx: &App,
    ) -> Result<Vec<PreparedToolCallContent>> {
        match self {
            Self::V1(content) => PreparedToolCallContent::prepare(content, terminals, cx),
            Self::V2(content) => PreparedToolCallContent::prepare_v2(content, terminals, cx),
        }
    }
}

enum ToolMetadataPatch {
    Legacy(Option<acp_v1::Meta>),
    Protocol(MaybeUndefined<acp_v2::Meta>),
}

fn legacy_tool_field<T>(value: Option<T>) -> MaybeUndefined<T> {
    value.map(MaybeUndefined::Value).unwrap_or_default()
}

fn tool_kind_from_v1(kind: acp_v1::ToolKind) -> Option<acp_v2::ToolKind> {
    Some(match kind {
        acp_v1::ToolKind::Read => acp_v2::ToolKind::Read,
        acp_v1::ToolKind::Edit => acp_v2::ToolKind::Edit,
        acp_v1::ToolKind::Delete => acp_v2::ToolKind::Delete,
        acp_v1::ToolKind::Move => acp_v2::ToolKind::Move,
        acp_v1::ToolKind::Search => acp_v2::ToolKind::Search,
        acp_v1::ToolKind::Execute => acp_v2::ToolKind::Execute,
        acp_v1::ToolKind::Think => acp_v2::ToolKind::Think,
        acp_v1::ToolKind::Fetch => acp_v2::ToolKind::Fetch,
        acp_v1::ToolKind::SwitchMode => acp_v2::ToolKind::SwitchMode,
        acp_v1::ToolKind::Other => acp_v2::ToolKind::Other,
        _ => return None,
    })
}

fn tool_status_from_v1(status: acp_v1::ToolCallStatus) -> Option<acp_v2::ToolCallStatus> {
    Some(match status {
        acp_v1::ToolCallStatus::Pending => acp_v2::ToolCallStatus::Pending,
        acp_v1::ToolCallStatus::InProgress => acp_v2::ToolCallStatus::InProgress,
        acp_v1::ToolCallStatus::Completed => acp_v2::ToolCallStatus::Completed,
        acp_v1::ToolCallStatus::Failed => acp_v2::ToolCallStatus::Failed,
        _ => return None,
    })
}

impl ToolCallPatch {
    fn legacy(fields: acp_v1::ToolCallUpdateFields, meta: Option<acp_v1::Meta>) -> Self {
        Self {
            title: legacy_tool_field(fields.title),
            name: legacy_tool_field(fields.name),
            kind: legacy_tool_field(fields.kind.and_then(tool_kind_from_v1)),
            status: legacy_tool_field(fields.status.and_then(tool_status_from_v1)),
            content: legacy_tool_field(fields.content.map(ToolContentPatch::V1)),
            locations: legacy_tool_field(
                fields
                    .locations
                    .map(|locations| locations.into_iter().map(Into::into).collect()),
            ),
            raw_input: legacy_tool_field(fields.raw_input),
            raw_output: legacy_tool_field(fields.raw_output),
            capabilities: MaybeUndefined::Undefined,
            meta: ToolMetadataPatch::Legacy(meta),
        }
    }

    fn protocol(update: acp_v2::ToolCallUpdate) -> Self {
        Self {
            title: update.title,
            name: update.name,
            kind: update.kind,
            status: update.status,
            content: update.content.map_value(ToolContentPatch::V2),
            locations: update
                .locations
                .map_value(|locations| locations.into_iter().map(Into::into).collect()),
            raw_input: update.raw_input,
            raw_output: update.raw_output,
            capabilities: update.capabilities,
            meta: ToolMetadataPatch::Protocol(update.meta),
        }
    }
}

impl ToolCall {
    fn from_acp(
        tool_call: acp_v1::ToolCall,
        status: Option<ToolCallStatus>,
        language_registry: Arc<LanguageRegistry>,
        terminals: &HashMap<acp_v2::TerminalId, Entity<Terminal>>,
        cx: &mut App,
    ) -> Result<Self> {
        let update = acp_v1::ToolCallUpdate::from(tool_call);
        let mut call = Self::from_patch(
            acp_v2::ToolCallId::new(update.tool_call_id.0),
            ToolCallPatch::legacy(update.fields, update.meta),
            language_registry,
            ToolTerminalResolver::registered(terminals),
            cx,
        )?;
        if let Some(status) = status {
            call.set_legacy_status(status);
        }
        Ok(call)
    }

    fn from_patch(
        id: acp_v2::ToolCallId,
        patch: ToolCallPatch,
        language_registry: Arc<LanguageRegistry>,
        terminals: ToolTerminalResolver<'_>,
        cx: &mut App,
    ) -> Result<Self> {
        let content = patch
            .content
            .take()
            .map(|content| content.prepare(terminals, cx))
            .transpose()?
            .unwrap_or_default()
            .into_iter()
            .map(|item| ToolCallContent::from_prepared(item, &language_registry, cx))
            .collect();
        let title = patch.title.take().map(SharedString::from);
        let name = patch.name.take().map(SharedString::from);
        let kind = patch.kind.take();
        let raw_input = patch.raw_input.take();
        let raw_input_markdown = raw_input
            .as_ref()
            .and_then(|input| markdown_for_raw_output(input, &language_registry, cx));
        let meta = match patch.meta {
            ToolMetadataPatch::Legacy(meta) => meta,
            ToolMetadataPatch::Protocol(meta) => {
                let mut stored = None;
                MetadataUpdate::V2(meta).apply(&mut stored);
                stored
            }
        };
        let tool_name = name.clone().or_else(|| tool_name_from_meta(&meta));
        let subagent_session_info = subagent_session_info_from_meta(&meta);
        let sandbox_authorization_details = sandbox_authorization_details_from_meta(&meta);
        let sandbox_fallback_authorization_details =
            sandbox_fallback_authorization_details_from_meta(&meta);
        let sandbox_not_applied = sandbox_not_applied_from_meta(&meta);
        let label = Self::new_label(
            title.as_ref().filter(|title| !title.trim().is_empty()),
            tool_name.as_ref(),
            kind.as_ref().unwrap_or(&acp_v2::ToolKind::Other),
            language_registry.clone(),
            cx,
        );

        let mut result = Self {
            id,
            label,
            title,
            name,
            reported_kind: kind,
            reported_status: patch.status.take(),
            meta,
            structured_content: content,
            locations: patch.locations.take().unwrap_or_default(),
            resolved_locations: Vec::default(),
            local_status: None,
            authorization: None,
            raw_input,
            raw_input_markdown,
            raw_output: patch.raw_output.take(),
            raw_output_content: None,
            advertises_cancel: patch
                .capabilities
                .as_opt_ref()
                .flatten()
                .is_some_and(|capabilities| capabilities.cancel.is_some()),
            tool_name,
            subagent_session_info,
            sandbox_authorization_details,
            sandbox_fallback_authorization_details,
            sandbox_not_applied,
        };
        result.update_raw_output_content(&language_registry, cx);
        Ok(result)
    }

    pub fn kind(&self) -> &acp_v2::ToolKind {
        self.reported_kind
            .as_ref()
            .unwrap_or(&acp_v2::ToolKind::Other)
    }

    pub fn status(&self) -> ToolCallStatus {
        if self.authorization.is_some() {
            ToolCallStatus::WaitingForConfirmation
        } else {
            self.underlying_status()
        }
    }

    pub fn authorization_id(&self) -> Option<PermissionRequestId> {
        self.authorization
    }

    /// Whether Zed may stop this tool call with `tool_call/cancel`: the agent advertises it
    /// and the tool call hasn't finished.
    pub fn can_cancel(&self) -> bool {
        self.advertises_cancel
            && matches!(
                self.underlying_status(),
                ToolCallStatus::Pending | ToolCallStatus::InProgress | ToolCallStatus::Paused
            )
    }

    fn underlying_status(&self) -> ToolCallStatus {
        self.local_status
            .or_else(|| ToolCallStatus::from_reported(self.reported_status.as_ref()))
            .unwrap_or(ToolCallStatus::Pending)
    }

    fn permission_status(&self) -> Option<acp_v2::ToolCallStatus> {
        self.local_status
            .or_else(|| ToolCallStatus::from_reported(self.reported_status.as_ref()))
            .and_then(|status| status.as_acp_status())
    }

    fn set_legacy_status(&mut self, status: ToolCallStatus) {
        self.local_status = (Some(status)
            != ToolCallStatus::from_reported(self.reported_status.as_ref()))
        .then_some(status);
    }

    fn set_local_status(&mut self, status: ToolCallStatus) {
        // An explicit decision survives a later clear of reported information,
        // even if the two statuses happen to agree right now.
        self.local_status = Some(status);
    }

    fn effective_title(&self) -> Option<&SharedString> {
        self.title.as_ref().filter(|title| !title.trim().is_empty())
    }

    fn label_text(
        title: Option<&SharedString>,
        tool_name: Option<&SharedString>,
        kind: &acp_v2::ToolKind,
    ) -> SharedString {
        let Some(title) = title else {
            return tool_name
                .filter(|name| !name.trim().is_empty())
                .cloned()
                .unwrap_or_else(|| "Tool call".into());
        };

        if kind == &acp_v2::ToolKind::Execute {
            title.clone()
        } else if kind == &acp_v2::ToolKind::Edit {
            MarkdownEscaped(title).to_string().into()
        } else if let Some((first_line, _)) = title.split_once('\n') {
            (first_line.to_owned() + "…").into()
        } else {
            title.clone()
        }
    }

    fn new_label(
        title: Option<&SharedString>,
        tool_name: Option<&SharedString>,
        kind: &acp_v2::ToolKind,
        language_registry: Arc<LanguageRegistry>,
        cx: &mut App,
    ) -> Entity<Markdown> {
        let text = Self::label_text(title, tool_name, kind);
        cx.new(|cx| {
            if title.is_none() || kind == &acp_v2::ToolKind::Execute {
                Markdown::new_text(text, cx)
            } else {
                Markdown::new(text, Some(language_registry), None, cx)
            }
        })
    }

    fn apply_patch(
        &mut self,
        patch: ToolCallPatch,
        language_registry: Arc<LanguageRegistry>,
        terminals: ToolTerminalResolver<'_>,
        cx: &mut App,
    ) -> Result<()> {
        let legacy_terminal_labels = matches!(&patch.meta, ToolMetadataPatch::Legacy(_));
        let ToolCallPatch {
            kind,
            status,
            title,
            name,
            content,
            locations,
            raw_input,
            raw_output,
            capabilities,
            meta,
        } = patch;
        if let Some(capabilities) = capabilities.as_opt_ref() {
            self.advertises_cancel =
                capabilities.is_some_and(|capabilities| capabilities.cancel.is_some());
        }
        let output_changed = !content.is_undefined() || !raw_output.is_undefined();
        // Completion remains authoritative even when output cannot be displayed;
        // otherwise a finished tool could leave its permission request waiting.
        self.apply_reported_status(status);
        // Prepare all content before changing presentation fields or shared
        // display entities; cloning their handles would not isolate mutations.
        let content = match content {
            MaybeUndefined::Undefined => None,
            MaybeUndefined::Null => Some(Vec::new()),
            MaybeUndefined::Value(content) => Some(content.prepare(terminals, cx)?),
        };
        let was_plain_text =
            self.effective_title().is_none() || self.kind() == &acp_v2::ToolKind::Execute;
        let mut label_changed = !title.is_undefined() || !kind.is_undefined();
        if !kind.is_undefined() {
            self.reported_kind = kind.take();
        }
        let name_changed = !name.is_undefined();
        if name_changed {
            self.name = name.take().map(SharedString::from);
        }
        let old_tool_name = self.tool_name.clone();
        match meta {
            ToolMetadataPatch::Legacy(meta) => {
                if name_changed {
                    self.tool_name = self.name.clone();
                } else if self.tool_name.is_none() {
                    self.tool_name = tool_name_from_meta(&meta);
                }
                // Legacy extension hints are sticky; omitted or malformed hints
                // have never cleared values received in an earlier v1 update.
                if let Some(value) = subagent_session_info_from_meta(&meta) {
                    self.subagent_session_info = Some(value);
                }
                if let Some(value) = sandbox_authorization_details_from_meta(&meta) {
                    self.sandbox_authorization_details = Some(value);
                }
                if let Some(value) = sandbox_fallback_authorization_details_from_meta(&meta) {
                    self.sandbox_fallback_authorization_details = Some(value);
                }
                if let Some(value) = sandbox_not_applied_from_meta(&meta) {
                    self.sandbox_not_applied = Some(value);
                }
                if meta.is_some() {
                    self.meta = meta;
                }
            }
            ToolMetadataPatch::Protocol(meta) => {
                let cleared = matches!(&meta, MaybeUndefined::Null);
                let meta_changed = MetadataUpdate::V2(meta).apply(&mut self.meta) || cleared;
                if meta_changed {
                    self.subagent_session_info = subagent_session_info_from_meta(&self.meta);
                    self.sandbox_authorization_details =
                        sandbox_authorization_details_from_meta(&self.meta);
                    self.sandbox_fallback_authorization_details =
                        sandbox_fallback_authorization_details_from_meta(&self.meta);
                    self.sandbox_not_applied = sandbox_not_applied_from_meta(&self.meta);
                }
                if name_changed || meta_changed {
                    self.tool_name = self
                        .name
                        .clone()
                        .or_else(|| tool_name_from_meta(&self.meta));
                }
            }
        }
        label_changed |= self.tool_name != old_tool_name;

        if !title.is_undefined() {
            self.title = title.take().map(SharedString::from);
            if self.kind() == &acp_v2::ToolKind::Execute
                && let Some(title) = self.effective_title()
            {
                // A missing tool title must not overwrite an actual terminal command.
                for terminal in self.terminals() {
                    if legacy_terminal_labels || terminal.read(cx).is_process_backed() {
                        terminal.update(cx, |terminal, cx| {
                            terminal.update_command_label(title, cx);
                        });
                    }
                }
            }
        }
        if label_changed {
            let is_plain_text =
                self.effective_title().is_none() || self.kind() == &acp_v2::ToolKind::Execute;
            if was_plain_text != is_plain_text {
                self.label = Self::new_label(
                    self.effective_title(),
                    self.tool_name.as_ref(),
                    self.kind(),
                    language_registry.clone(),
                    cx,
                );
            } else {
                let text =
                    Self::label_text(self.effective_title(), self.tool_name.as_ref(), self.kind());
                if self.label.read(cx).source() != &text {
                    self.label.update(cx, |label, cx| label.replace(text, cx));
                }
            }
        }

        if let Some(content) = content {
            let new_content_len = content.len();
            let mut content = content.into_iter();

            for (old, new) in self.structured_content.iter_mut().zip(content.by_ref()) {
                old.update_from_prepared(new, &language_registry, cx);
            }
            for new in content {
                self.structured_content.push(ToolCallContent::from_prepared(
                    new,
                    &language_registry,
                    cx,
                ));
            }
            self.structured_content.truncate(new_content_len);
        }

        if !locations.is_undefined() {
            let locations = locations.take().unwrap_or_default();
            if self.locations != locations {
                self.locations = locations;
                self.resolved_locations.clear();
            }
        }

        if !raw_input.is_undefined() {
            let raw_input = raw_input.take();
            if self.raw_input != raw_input {
                match (
                    self.raw_input_markdown.as_ref(),
                    raw_input.as_ref().and_then(raw_output_text),
                ) {
                    (Some(markdown), Some(text)) => update_markdown_in_place(markdown, &text, cx),
                    (_, Some(text)) => {
                        self.raw_input_markdown = Some(cx.new(|cx| {
                            Markdown::new_deferred(text, Some(language_registry.clone()), cx)
                        }));
                    }
                    (_, None) => self.raw_input_markdown = None,
                }
                self.raw_input = raw_input;
            }
        }

        if !raw_output.is_undefined() {
            self.raw_output = raw_output.take();
        }
        if output_changed {
            self.update_raw_output_content(&language_registry, cx);
        }
        Ok(())
    }

    fn append_content(
        &mut self,
        content: PreparedToolCallContent,
        language_registry: &Arc<LanguageRegistry>,
        cx: &mut App,
    ) {
        // Agents may stream tool text one chunk at a time, like message text, so adjacent
        // plain text continues the previous block instead of starting a new one.
        if let PreparedToolCallContent::ContentBlock(acp_v2::Content {
            content: acp_v2::ContentBlock::Text(text),
            meta: None,
            ..
        }) = &content
            && let Some(ToolCallContent::ContentBlock { block, meta: None }) =
                self.structured_content.last_mut()
            && block.append_plain_text(text, cx)
        {
            self.update_raw_output_content(language_registry, cx);
            return;
        }
        self.structured_content.push(ToolCallContent::from_prepared(
            content,
            language_registry,
            cx,
        ));
        self.update_raw_output_content(language_registry, cx);
    }

    fn update_raw_output_content(
        &mut self,
        language_registry: &Arc<LanguageRegistry>,
        cx: &mut App,
    ) {
        if !self.structured_content.is_empty() {
            self.raw_output_content = None;
            return;
        }
        let Some(text) = self.raw_output.as_ref().and_then(raw_output_text) else {
            self.raw_output_content = None;
            return;
        };
        if let Some(markdown) = self
            .raw_output_content
            .as_deref()
            .and_then(ToolCallContent::markdown)
        {
            update_markdown_in_place(markdown, &text, cx);
        } else {
            let markdown =
                cx.new(|cx| Markdown::new_deferred(text, Some(language_registry.clone()), cx));
            self.raw_output_content = Some(Box::new(ToolCallContent::ContentBlock {
                block: ContentBlock::from_markdown(markdown),
                meta: None,
            }));
        }
    }

    pub fn content(&self) -> &[ToolCallContent] {
        if self.structured_content.is_empty() {
            self.raw_output_content
                .as_deref()
                .map(std::slice::from_ref)
                .unwrap_or_default()
        } else {
            &self.structured_content
        }
    }

    fn apply_reported_status(&mut self, status: MaybeUndefined<acp_v2::ToolCallStatus>) {
        match status {
            MaybeUndefined::Undefined => {}
            MaybeUndefined::Null => {
                // Clearing reported information does not resolve a local request
                // or undo a user's local decision.
                self.reported_status = None;
            }
            MaybeUndefined::Value(status) => {
                if matches!(
                    status,
                    acp_v2::ToolCallStatus::Completed
                        | acp_v2::ToolCallStatus::Failed
                        | acp_v2::ToolCallStatus::Cancelled
                ) {
                    self.authorization = None;
                }
                self.reported_status = Some(status);
                self.local_status = None;
            }
        }
    }

    pub fn diffs(&self) -> impl Iterator<Item = &Entity<Diff>> {
        self.structured_content
            .iter()
            .filter_map(|content| match content {
                ToolCallContent::Diff(diff) | ToolCallContent::LegacyDiff { diff, .. } => {
                    Some(diff)
                }
                _ => None,
            })
    }

    pub fn terminals(&self) -> impl Iterator<Item = &Entity<Terminal>> {
        self.structured_content
            .iter()
            .filter_map(|content| match content {
                ToolCallContent::Terminal { terminal, .. } => Some(terminal),
                _ => None,
            })
    }

    pub fn is_subagent(&self) -> bool {
        self.tool_name.as_ref().is_some_and(|s| s == "spawn_agent")
            || self.subagent_session_info.is_some()
    }

    pub fn to_markdown(&self, cx: &App) -> String {
        let label = self.label.read(cx).source();
        let label = if self.effective_title().is_none() {
            MarkdownEscaped(label).to_string()
        } else {
            label.to_string()
        };
        let mut markdown = format!("**Tool Call: {}**\nStatus: {}\n\n", label, self.status());
        for content in self.content() {
            markdown.push_str(content.to_markdown(cx).as_str());
            markdown.push_str("\n\n");
        }
        markdown
    }

    async fn resolve_location(
        location: ToolCallLocation,
        project: WeakEntity<Project>,
        cx: &mut AsyncApp,
    ) -> Option<ResolvedLocation> {
        let buffer = project
            .update(cx, |project, cx| {
                if let Some(path) = project.project_path_for_absolute_path(&location.path, cx) {
                    Some(project.open_buffer(path, cx))
                } else if project.path_style(cx).is_absolute(&location.path) {
                    Some(project.open_local_buffer(&location.path, cx))
                } else {
                    None
                }
            })
            .ok()??;
        let buffer = buffer.await.log_err()?;
        let position = buffer.update(cx, |buffer, _| {
            let snapshot = buffer.snapshot();
            if let Some(line) = location.line {
                let row = line.saturating_sub(1);
                let column = snapshot.indent_size_for_line(row).len;
                let point = snapshot.clip_point(Point::new(row, column), Bias::Left);
                snapshot.anchor_before(point)
            } else {
                Anchor::min_for_buffer(snapshot.remote_id())
            }
        });

        Some(ResolvedLocation { buffer, position })
    }

    fn resolve_locations(
        &self,
        project: Entity<Project>,
        cx: &mut App,
    ) -> Task<Vec<Option<ResolvedLocation>>> {
        let locations = self.locations.clone();
        project.update(cx, |_, cx| {
            cx.spawn(async move |project, cx| {
                let mut new_locations = Vec::new();
                for location in locations {
                    new_locations.push(Self::resolve_location(location, project.clone(), cx).await);
                }
                new_locations
            })
        })
    }
}

// Separate so we can hold a strong reference to the buffer
// for saving on the thread
#[derive(Clone, Debug, PartialEq, Eq)]
struct ResolvedLocation {
    buffer: Entity<Buffer>,
    position: Anchor,
}

impl From<&ResolvedLocation> for AgentLocation {
    fn from(value: &ResolvedLocation) -> Self {
        Self {
            buffer: value.buffer.downgrade(),
            position: value.position,
        }
    }
}

#[derive(Debug, Clone)]
pub enum SelectedPermissionParams {
    Terminal { patterns: Vec<String> },
}

#[derive(Debug, Clone)]
pub struct SelectedPermissionOutcome {
    pub option_id: acp_v2::PermissionOptionId,
    pub option_kind: acp_v2::PermissionOptionKind,
    pub params: Option<SelectedPermissionParams>,
}

impl SelectedPermissionOutcome {
    pub fn new(
        option_id: acp_v2::PermissionOptionId,
        option_kind: acp_v2::PermissionOptionKind,
    ) -> Self {
        Self {
            option_id,
            option_kind,
            params: None,
        }
    }

    pub fn params(mut self, params: Option<SelectedPermissionParams>) -> Self {
        self.params = params;
        self
    }
}

#[derive(Clone, Debug)]
pub enum RequestPermissionOutcome {
    Cancelled,
    InterruptedByFollowUp,
    Selected(SelectedPermissionOutcome),
}

/// What a `WaitingForConfirmation` prompt represents semantically.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthorizationKind {
    /// The user is granting or denying permission for the tool call to
    /// proceed. The selected `PermissionOptionKind` determines whether the
    /// tool call transitions to `InProgress` (allow) or `Rejected` (reject).
    /// This is the default for tool authorization prompts.
    PermissionGrant,
    /// The user is choosing between actions for the tool to take next
    /// (for example, "Save" vs "Discard" before editing a dirty buffer).
    /// The tool call always transitions to `InProgress` regardless of the
    /// selected `PermissionOptionKind`; the caller interprets the chosen
    /// `option_id` to decide what to do.
    ActionChoice,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolCallStatus {
    /// The tool call hasn't started running yet, but we start showing it to
    /// the user.
    Pending,
    /// The tool call is waiting for confirmation from the user.
    WaitingForConfirmation,
    /// The tool call is currently running.
    InProgress,
    /// The tool call has started but is not running now, until something resumes it.
    Paused,
    /// The tool call completed successfully.
    Completed,
    /// The tool call failed.
    Failed,
    /// The user rejected the tool call.
    Rejected,
    /// The user canceled generation so the tool call was canceled.
    Canceled,
}

impl From<acp_v1::ToolCallStatus> for ToolCallStatus {
    fn from(status: acp_v1::ToolCallStatus) -> Self {
        Self::from_reported(tool_status_from_v1(status).as_ref()).unwrap_or(Self::Pending)
    }
}

/// Whether Zed can render a form property. Unknown property types and multi-select items
/// can't be rendered.
pub fn is_renderable_elicitation_property(property: &acp_v2::ElicitationPropertySchema) -> bool {
    match property {
        acp_v2::ElicitationPropertySchema::String(_)
        | acp_v2::ElicitationPropertySchema::Number(_)
        | acp_v2::ElicitationPropertySchema::Integer(_)
        | acp_v2::ElicitationPropertySchema::Boolean(_) => true,
        acp_v2::ElicitationPropertySchema::Array(schema) => matches!(
            schema.items,
            acp_v2::MultiSelectItems::String(_) | acp_v2::MultiSelectItems::Titled(_)
        ),
        _ => false,
    }
}

/// A form Zed can't render is rejected with `-32602` rather than answered with `decline` or
/// `cancel`, which would report a choice the user never made. Optional properties Zed can't
/// render are left out of the form instead.
fn validate_form_schema(schema: &acp_v2::ElicitationSchema) -> Result<(), acp_v2::Error> {
    let unrenderable = |reason: String| Err(acp_v2::Error::invalid_params().data(reason));
    for name in schema.required.iter().flatten() {
        match schema.properties.get(name) {
            None => return unrenderable(format!("required form property `{name}` is not defined")),
            Some(property) if !is_renderable_elicitation_property(property) => {
                return unrenderable(format!(
                    "required form property `{name}` has an unsupported type"
                ));
            }
            Some(_) => {}
        }
    }
    for (name, property) in &schema.properties {
        if let acp_v2::ElicitationPropertySchema::String(property) = property
            && property.enum_values.is_some()
            && property.one_of.is_some()
        {
            return unrenderable(format!(
                "form property `{name}` sets both `enum` and `oneOf`"
            ));
        }
    }
    Ok(())
}

/// Lets v1 handlers answer an unknown terminal with `-32002`, as the protocol's error page asks.
/// The data mirrors the `sessionId` that the error page specifies for unknown sessions.
fn terminal_not_found(terminal_id: &acp_v2::TerminalId) -> anyhow::Error {
    acp_v1::Error::from(acp_v1::ErrorCode::ResourceNotFound)
        .data(serde_json::json!({ "terminalId": terminal_id.0.as_ref() }))
        .into()
}

impl ToolCallStatus {
    fn from_reported(status: Option<&acp_v2::ToolCallStatus>) -> Option<Self> {
        match status? {
            acp_v2::ToolCallStatus::Pending => Some(Self::Pending),
            acp_v2::ToolCallStatus::InProgress => Some(Self::InProgress),
            acp_v2::ToolCallStatus::Paused => Some(Self::Paused),
            acp_v2::ToolCallStatus::Completed => Some(Self::Completed),
            acp_v2::ToolCallStatus::Failed => Some(Self::Failed),
            acp_v2::ToolCallStatus::Cancelled => Some(Self::Canceled),
            // The protocol says to treat an unrecognized status as still in progress.
            _ => Some(Self::InProgress),
        }
    }

    fn as_acp_status(&self) -> Option<acp_v2::ToolCallStatus> {
        match self {
            ToolCallStatus::Pending => Some(acp_v2::ToolCallStatus::Pending),
            ToolCallStatus::InProgress => Some(acp_v2::ToolCallStatus::InProgress),
            ToolCallStatus::Paused => Some(acp_v2::ToolCallStatus::Paused),
            ToolCallStatus::Completed => Some(acp_v2::ToolCallStatus::Completed),
            ToolCallStatus::Failed => Some(acp_v2::ToolCallStatus::Failed),
            // A new authorization can retry a locally canceled tool; the
            // incoming request then supplies its continuation status.
            ToolCallStatus::Canceled => None,
            ToolCallStatus::WaitingForConfirmation | ToolCallStatus::Rejected => None,
        }
    }

    fn status_after_permission_grant(status: acp_v2::ToolCallStatus) -> ToolCallStatus {
        match Self::from_reported(Some(&status)).unwrap_or(Self::Pending) {
            ToolCallStatus::Pending => ToolCallStatus::InProgress,
            status => status,
        }
    }
}

impl Display for ToolCallStatus {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                ToolCallStatus::Pending => "Pending",
                ToolCallStatus::WaitingForConfirmation => "Waiting for confirmation",
                ToolCallStatus::InProgress => "In Progress",
                ToolCallStatus::Paused => "Paused",
                ToolCallStatus::Completed => "Completed",
                ToolCallStatus::Failed => "Failed",
                ToolCallStatus::Rejected => "Rejected",
                ToolCallStatus::Canceled => "Canceled",
            }
        )
    }
}

fn elicitation_status_for_response(
    response: &acp_v2::CreateElicitationResponse,
) -> ElicitationStatus {
    match &response.action {
        acp_v2::ElicitationAction::Accept(_) => ElicitationStatus::Accepted,
        acp_v2::ElicitationAction::Decline => ElicitationStatus::Declined,
        acp_v2::ElicitationAction::Cancel => ElicitationStatus::Canceled,
        _ => ElicitationStatus::Canceled,
    }
}

#[derive(Debug)]
pub struct MessageContent {
    source_blocks: Vec<acp_v2::ContentBlock>,
    source_version: MessageContentVersion,
    blocks: Vec<RenderedMessageBlock>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MessageContentVersion(u64);

impl MessageContentVersion {
    fn next() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT_VERSION: AtomicU64 = AtomicU64::new(0);
        Self(NEXT_VERSION.fetch_add(1, Ordering::Relaxed))
    }
}

impl Default for MessageContent {
    fn default() -> Self {
        Self {
            source_blocks: Vec::new(),
            source_version: MessageContentVersion::next(),
            blocks: Vec::new(),
        }
    }
}

impl PartialEq for MessageContent {
    fn eq(&self, other: &Self) -> bool {
        self.source_blocks == other.source_blocks && self.blocks == other.blocks
    }
}

#[derive(Debug, PartialEq)]
struct RenderedMessageBlock {
    render: RenderBlock,
    source_index: Option<usize>,
}

enum DesiredMessageBlock {
    Markdown(String),
    Source(usize),
}

impl MessageContent {
    pub fn new(
        block: acp_v2::ContentBlock,
        language_registry: &Arc<LanguageRegistry>,
        path_style: PathStyle,
        cx: &mut App,
    ) -> Self {
        let mut content = Self::default();
        content.append(block, language_registry, path_style, cx);
        content
    }

    /// Original content blocks, including text not yet revealed by streaming.
    pub fn source_blocks(&self) -> &[acp_v2::ContentBlock] {
        &self.source_blocks
    }

    /// Lets readers retain local UI state without caching another copy of the source.
    /// Also distinguishes different content owners; independent of protocol message IDs.
    pub fn source_version(&self) -> MessageContentVersion {
        self.source_version
    }

    fn shrink_source_capacity(&mut self) {
        self.source_blocks.shrink_to_fit();
    }

    pub fn blocks(&self) -> impl ExactSizeIterator<Item = ContentBlockView<'_>> {
        self.blocks.iter().map(|block| ContentBlockView {
            render: &block.render,
            source: block
                .source_index
                .and_then(|index| self.source_blocks.get(index)),
        })
    }

    pub fn markdowns(&self) -> impl Iterator<Item = &Entity<Markdown>> {
        self.blocks().filter_map(|block| block.markdown())
    }

    pub fn visible_content(&self, cx: &App) -> bool {
        self.blocks().any(|block| block.visible_content(cx))
    }

    pub fn to_markdown(&self, cx: &App) -> String {
        self.blocks()
            .map(|block| block.to_markdown(cx))
            .filter(|text| !text.is_empty())
            .join("\n\n")
    }

    fn trailing_text(&self) -> Option<&Entity<Markdown>> {
        match &self.blocks.last()?.render {
            RenderBlock::Markdown { markdown } => Some(markdown),
            _ => None,
        }
    }

    pub fn append(
        &mut self,
        block: acp_v2::ContentBlock,
        language_registry: &Arc<LanguageRegistry>,
        path_style: PathStyle,
        cx: &mut App,
    ) {
        self.append_rendered(&block, language_registry, path_style, cx);
        self.source_blocks.push(block);
        self.source_version = MessageContentVersion::next();
    }

    fn append_compaction_summary(
        &mut self,
        block: acp_v2::ContentBlock,
        language_registry: &Arc<LanguageRegistry>,
        cx: &mut App,
    ) {
        // Compaction links remain separate blocks rather than joining streamed message text.
        if let acp_v2::ContentBlock::Text(text) = &block {
            if let Some(markdown) = self.trailing_text() {
                markdown.update(cx, |markdown, cx| markdown.append(&text.text, cx));
            } else {
                self.blocks.push(RenderedMessageBlock {
                    render: RenderBlock::Markdown {
                        markdown: ContentBlock::create_markdown(
                            text.text.clone(),
                            language_registry,
                            cx,
                        ),
                    },
                    source_index: None,
                });
            }
        } else {
            self.blocks.push(RenderedMessageBlock {
                render: ContentBlock::render_from_source(&block, language_registry, cx),
                source_index: Some(self.source_blocks.len()),
            });
        }
        self.source_blocks.push(block);
        self.source_version = MessageContentVersion::next();
    }

    fn append_rendered(
        &mut self,
        block: &acp_v2::ContentBlock,
        language_registry: &Arc<LanguageRegistry>,
        path_style: PathStyle,
        cx: &mut App,
    ) {
        if let Some(text) = Self::inline_text(block, path_style, false, self.blocks.is_empty()) {
            self.append_text(&text, language_registry, cx);
        } else {
            let render = ContentBlock::render_from_source(block, language_registry, cx);
            let source_index = match render {
                RenderBlock::EmbeddedResource { .. }
                | RenderBlock::Unsupported { .. }
                | RenderBlock::Image { .. } => Some(self.source_blocks.len()),
                _ => None,
            };
            self.blocks.push(RenderedMessageBlock {
                render,
                source_index,
            });
        }
    }

    fn inline_text(
        block: &acp_v2::ContentBlock,
        path_style: PathStyle,
        prompt: bool,
        leading: bool,
    ) -> Option<Cow<'_, str>> {
        match block {
            acp_v2::ContentBlock::Text(text) => Some(Cow::Borrowed(&text.text)),
            acp_v2::ContentBlock::ResourceLink(resource) => {
                let mut text = ContentBlock::resource_link_md(&resource.uri, path_style);
                if leading {
                    // A leading link separates the next streamed text chunk.
                    text.push('\n');
                }
                Some(Cow::Owned(text))
            }
            acp_v2::ContentBlock::Resource(resource)
                if prompt
                    && matches!(
                        resource.resource,
                        acp_v2::EmbeddedResourceResource::TextResourceContents(_)
                    ) =>
            {
                Some(Cow::Owned(ContentBlock::embedded_resource_string_contents(
                    resource, path_style,
                )))
            }
            _ => None,
        }
    }

    fn append_text(&mut self, text: &str, language_registry: &Arc<LanguageRegistry>, cx: &mut App) {
        if text.is_empty() {
            return;
        }
        if let Some(markdown) = self.trailing_text() {
            markdown.update(cx, |markdown, cx| markdown.append(text, cx));
        } else {
            self.blocks.push(RenderedMessageBlock {
                render: RenderBlock::Markdown {
                    markdown: ContentBlock::create_markdown(text.to_owned(), language_registry, cx),
                },
                source_index: None,
            });
        }
    }

    fn append_deferred_text(&mut self, text: acp_v2::TextContent) -> usize {
        let index = self.source_blocks.len();
        self.source_blocks.push(acp_v2::ContentBlock::Text(text));
        self.source_version = MessageContentVersion::next();
        index
    }

    fn append_prompt(
        &mut self,
        block: acp_v2::ContentBlock,
        language_registry: &Arc<LanguageRegistry>,
        path_style: PathStyle,
        cx: &mut App,
    ) {
        if let Some(text) = Self::inline_text(&block, path_style, true, self.blocks.is_empty()) {
            self.append_text(&text, language_registry, cx);
        } else {
            self.append_rendered(&block, language_registry, path_style, cx);
        }
        self.source_blocks.push(block);
        self.source_version = MessageContentVersion::next();
    }

    fn replace(
        &mut self,
        blocks: Vec<acp_v2::ContentBlock>,
        language_registry: &Arc<LanguageRegistry>,
        path_style: PathStyle,
        cx: &mut App,
    ) {
        self.replace_blocks(blocks, language_registry, path_style, false, cx);
    }

    fn replace_prompt(
        &mut self,
        blocks: Vec<acp_v2::ContentBlock>,
        language_registry: &Arc<LanguageRegistry>,
        path_style: PathStyle,
        cx: &mut App,
    ) {
        self.replace_blocks(blocks, language_registry, path_style, true, cx);
    }

    fn replace_blocks(
        &mut self,
        blocks: Vec<acp_v2::ContentBlock>,
        language_registry: &Arc<LanguageRegistry>,
        path_style: PathStyle,
        prompt: bool,
        cx: &mut App,
    ) {
        if self.source_blocks != blocks {
            self.source_version = MessageContentVersion::next();
        }
        let mut desired = Vec::<DesiredMessageBlock>::new();
        for (source_index, block) in blocks.iter().enumerate() {
            if let Some(text) = Self::inline_text(block, path_style, prompt, desired.is_empty()) {
                if text.is_empty() {
                    continue;
                }
                if let Some(DesiredMessageBlock::Markdown(previous)) = desired.last_mut() {
                    previous.push_str(&text);
                } else {
                    desired.push(DesiredMessageBlock::Markdown(text.into_owned()));
                }
            } else {
                desired.push(DesiredMessageBlock::Source(source_index));
            }
        }

        let previous_blocks = std::mem::take(&mut self.blocks);
        let previous_sources = std::mem::replace(&mut self.source_blocks, blocks);
        self.blocks = desired
            .into_iter()
            .enumerate()
            .map(|(render_index, desired)| {
                let previous = previous_blocks.get(render_index);
                match desired {
                    DesiredMessageBlock::Markdown(text) => {
                        let markdown = match previous.map(|block| &block.render) {
                            Some(RenderBlock::Markdown { markdown }) => {
                                update_markdown_in_place(markdown, &text, cx);
                                markdown.clone()
                            }
                            _ => ContentBlock::create_markdown(text, language_registry, cx),
                        };
                        RenderedMessageBlock {
                            render: RenderBlock::Markdown { markdown },
                            source_index: None,
                        }
                    }
                    DesiredMessageBlock::Source(source_index) => {
                        let source = &self.source_blocks[source_index];
                        let previous_source = previous
                            .and_then(|block| block.source_index)
                            .and_then(|index| previous_sources.get(index));
                        let render = previous
                            .and_then(|block| {
                                Self::reuse_render(&block.render, previous_source, source, cx)
                            })
                            .unwrap_or_else(|| {
                                ContentBlock::render_from_source(source, language_registry, cx)
                            });
                        RenderedMessageBlock {
                            render,
                            source_index: Some(source_index),
                        }
                    }
                }
            })
            .collect();
    }

    fn reuse_render(
        previous: &RenderBlock,
        previous_source: Option<&acp_v2::ContentBlock>,
        source: &acp_v2::ContentBlock,
        cx: &mut App,
    ) -> Option<RenderBlock> {
        if previous_source == Some(source) {
            return Some(previous.clone());
        }
        match (previous, source) {
            (RenderBlock::Image { .. }, _)
                if Self::image_data(previous_source?) == Self::image_data(source)
                    && Self::image_data(source).is_some() =>
            {
                Some(previous.clone())
            }
            (
                RenderBlock::EmbeddedResource {
                    markdown: Some(markdown),
                },
                acp_v2::ContentBlock::Resource(resource),
            ) if matches!(
                &resource.resource,
                acp_v2::EmbeddedResourceResource::TextResourceContents(_)
            ) =>
            {
                let acp_v2::EmbeddedResourceResource::TextResourceContents(text) =
                    &resource.resource
                else {
                    return None;
                };
                update_markdown_in_place(markdown, &ContentBlock::text_resource_markdown(text), cx);
                Some(previous.clone())
            }
            (
                RenderBlock::EmbeddedResource { markdown: None },
                acp_v2::ContentBlock::Resource(resource),
            ) if matches!(
                &resource.resource,
                acp_v2::EmbeddedResourceResource::BlobResourceContents(_)
            ) && Self::image_data(source).is_none() =>
            {
                Some(previous.clone())
            }
            (RenderBlock::Unsupported { .. }, acp_v2::ContentBlock::Image(_))
                if matches!(previous_source, Some(acp_v2::ContentBlock::Image(_)))
                    && Self::image_data(previous_source?) == Self::image_data(source) =>
            {
                Some(previous.clone())
            }
            (RenderBlock::Unsupported { .. }, acp_v2::ContentBlock::Audio(_))
                if matches!(previous_source, Some(acp_v2::ContentBlock::Audio(_))) =>
            {
                Some(previous.clone())
            }
            (RenderBlock::Unsupported { .. }, acp_v2::ContentBlock::Other(_))
                if matches!(previous_source, Some(acp_v2::ContentBlock::Other(_))) =>
            {
                Some(previous.clone())
            }
            _ => None,
        }
    }

    fn image_data(block: &acp_v2::ContentBlock) -> Option<(&str, &str)> {
        match block {
            acp_v2::ContentBlock::Image(image) => Some((&image.data, image.mime_type.as_ref())),
            acp_v2::ContentBlock::Resource(resource) => match &resource.resource {
                acp_v2::EmbeddedResourceResource::BlobResourceContents(blob) => {
                    Some((&blob.blob, blob.mime_type.as_ref()?.as_ref()))
                }
                _ => None,
            },
            _ => None,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct ContentBlock {
    render: RenderBlock,
    source: Option<acp_v2::ContentBlock>,
}

#[derive(Debug, PartialEq, Clone)]
enum RenderBlock {
    Markdown {
        markdown: Entity<Markdown>,
    },
    EmbeddedResource {
        markdown: Option<Entity<Markdown>>,
    },
    ResourceLink,
    Image {
        image: Arc<gpui::Image>,
        dimensions: Option<gpui::Size<u32>>,
    },
    Unsupported {
        markdown: Entity<Markdown>,
    },
}

#[derive(Clone, Copy)]
pub struct ContentBlockView<'a> {
    render: &'a RenderBlock,
    source: Option<&'a acp_v2::ContentBlock>,
}

impl ContentBlock {
    pub fn from_markdown(markdown: Entity<Markdown>) -> Self {
        Self {
            render: RenderBlock::Markdown { markdown },
            source: None,
        }
    }

    #[cfg(test)]
    fn plain_markdown(&self) -> Option<&Entity<Markdown>> {
        match &self.render {
            RenderBlock::Markdown { markdown } => Some(markdown),
            _ => None,
        }
    }

    #[cfg(test)]
    fn unsupported_content(&self) -> Option<&acp_v2::ContentBlock> {
        self.as_view().unsupported_content()
    }

    pub fn new_output(
        block: acp_v2::ContentBlock,
        language_registry: &Arc<LanguageRegistry>,
        cx: &mut App,
    ) -> Self {
        match block {
            acp_v2::ContentBlock::Text(text) => {
                Self::create_markdown_block(text.text, language_registry, cx)
            }
            block => {
                let render = Self::render_from_source(&block, language_registry, cx);
                let source = match render {
                    RenderBlock::EmbeddedResource { .. }
                    | RenderBlock::ResourceLink
                    | RenderBlock::Unsupported { .. } => Some(block),
                    RenderBlock::Markdown { .. } | RenderBlock::Image { .. } => None,
                };
                Self { render, source }
            }
        }
    }

    fn new_tool_content(
        block: acp_v2::ContentBlock,
        language_registry: &Arc<LanguageRegistry>,
        cx: &mut App,
    ) -> Self {
        let render = Self::render_from_source(&block, language_registry, cx);
        Self {
            render,
            source: Some(block),
        }
    }

    fn update_tool_content(
        &mut self,
        block: acp_v2::ContentBlock,
        language_registry: &Arc<LanguageRegistry>,
        cx: &mut App,
    ) {
        let reused = match (self.source.as_ref(), &block) {
            (Some(acp_v2::ContentBlock::Text(old)), acp_v2::ContentBlock::Text(new))
                if matches!(self.render, RenderBlock::Markdown { .. }) =>
            {
                if old.text != new.text {
                    self.update_text_in_place(&block, cx);
                }
                true
            }
            (Some(acp_v2::ContentBlock::Image(old)), acp_v2::ContentBlock::Image(new)) => {
                old.data == new.data && old.mime_type == new.mime_type
            }
            (Some(acp_v2::ContentBlock::Resource(old)), acp_v2::ContentBlock::Resource(new)) => {
                match (&old.resource, &new.resource) {
                    (
                        acp_v2::EmbeddedResourceResource::TextResourceContents(_),
                        acp_v2::EmbeddedResourceResource::TextResourceContents(new),
                    ) => {
                        if let RenderBlock::EmbeddedResource {
                            markdown: Some(markdown),
                        } = &self.render
                        {
                            update_markdown_in_place(
                                markdown,
                                &Self::text_resource_markdown(new),
                                cx,
                            );
                            true
                        } else {
                            false
                        }
                    }
                    (
                        acp_v2::EmbeddedResourceResource::BlobResourceContents(old),
                        acp_v2::EmbeddedResourceResource::BlobResourceContents(new),
                    ) => old.blob == new.blob && old.mime_type == new.mime_type,
                    _ => false,
                }
            }
            (Some(acp_v2::ContentBlock::Audio(_)), acp_v2::ContentBlock::Audio(_))
            | (Some(acp_v2::ContentBlock::Other(_)), acp_v2::ContentBlock::Other(_)) => true,
            (Some(previous), _) => previous == &block,
            _ => false,
        };
        if !reused {
            self.render = Self::render_from_source(&block, language_registry, cx);
        }
        self.source = Some(block);
    }

    fn render_from_source(
        block: &acp_v2::ContentBlock,
        language_registry: &Arc<LanguageRegistry>,
        cx: &mut App,
    ) -> RenderBlock {
        match block {
            acp_v2::ContentBlock::Resource(resource) => {
                if let Some((image, dimensions)) = Self::decode_embedded_resource_image(resource) {
                    return RenderBlock::Image { image, dimensions };
                }
                if matches!(
                    &resource.resource,
                    acp_v2::EmbeddedResourceResource::TextResourceContents(_)
                        | acp_v2::EmbeddedResourceResource::BlobResourceContents(_)
                ) {
                    let markdown =
                        Self::embedded_resource_markdown(resource, language_registry, cx);
                    RenderBlock::EmbeddedResource { markdown }
                } else {
                    Self::unsupported(block, language_registry, cx)
                }
            }
            acp_v2::ContentBlock::Image(image) => {
                if let Some((image, dimensions)) = Self::decode_image(image) {
                    RenderBlock::Image { image, dimensions }
                } else {
                    RenderBlock::Unsupported {
                        markdown: Self::create_markdown(
                            "Image content could not be displayed.".into(),
                            language_registry,
                            cx,
                        ),
                    }
                }
            }
            acp_v2::ContentBlock::ResourceLink(_) => RenderBlock::ResourceLink,
            acp_v2::ContentBlock::Text(text) => RenderBlock::Markdown {
                markdown: Self::create_markdown(text.text.clone(), language_registry, cx),
            },
            acp_v2::ContentBlock::Other(_) => Self::unsupported(block, language_registry, cx),
            _ => Self::unsupported(block, language_registry, cx),
        }
    }

    fn unsupported(
        content: &acp_v2::ContentBlock,
        language_registry: &Arc<LanguageRegistry>,
        cx: &mut App,
    ) -> RenderBlock {
        let description = match content {
            acp_v2::ContentBlock::Audio(_) => "Audio content is not supported.",
            acp_v2::ContentBlock::Other(_) => "Unknown content type is not supported.",
            _ => "This content is not supported.",
        };
        RenderBlock::Unsupported {
            markdown: Self::create_markdown(description.into(), language_registry, cx),
        }
    }

    /// Updates a Markdown block in place from a streaming text `block`, reusing
    /// the existing `Markdown` entity rather than recreating it. Appends only the
    /// new suffix when the update is a continuation (the common streaming case),
    /// otherwise re-sets the source. Returns `false` when an in-place update isn't
    /// applicable, so the caller can fall back to replacing the block wholesale.
    ///
    /// Recreating the entity on every streamed snapshot causes the rendered
    /// element to tear down and rebuild, which flickers badly.
    /// Appends text without annotations or metadata to a block holding the same kind of text.
    fn append_plain_text(&mut self, text: &acp_v2::TextContent, cx: &mut App) -> bool {
        let is_plain =
            |text: &acp_v2::TextContent| text.annotations.is_none() && text.meta.is_none();
        let (RenderBlock::Markdown { markdown }, Some(acp_v2::ContentBlock::Text(source))) =
            (&self.render, &mut self.source)
        else {
            return false;
        };
        if !is_plain(text) || !is_plain(source) {
            return false;
        }
        source.text.push_str(&text.text);
        markdown.update(cx, |markdown, cx| markdown.append(&text.text, cx));
        true
    }

    pub fn update_text_in_place(&mut self, block: &acp_v2::ContentBlock, cx: &mut App) -> bool {
        let RenderBlock::Markdown { markdown } = &self.render else {
            return false;
        };
        let acp_v2::ContentBlock::Text(text_content) = block else {
            return false;
        };
        update_markdown_in_place(markdown, &text_content.text, cx);
        true
    }

    fn decode_image(
        image_content: &acp_v2::ImageContent,
    ) -> Option<(Arc<gpui::Image>, Option<gpui::Size<u32>>)> {
        Self::decode_image_data(&image_content.data, image_content.mime_type.as_ref())
    }

    fn decode_embedded_resource_image(
        resource: &acp_v2::EmbeddedResource,
    ) -> Option<(Arc<gpui::Image>, Option<gpui::Size<u32>>)> {
        let acp_v2::EmbeddedResourceResource::BlobResourceContents(blob) = &resource.resource
        else {
            return None;
        };
        let mime_type = blob.mime_type.as_ref()?.as_ref();
        Self::decode_image_data(&blob.blob, mime_type)
    }

    fn decode_image_data(
        data: &str,
        mime_type: &str,
    ) -> Option<(Arc<gpui::Image>, Option<gpui::Size<u32>>)> {
        use base64::Engine as _;

        let bytes = base64::engine::general_purpose::STANDARD
            .decode(data.as_bytes())
            .ok()?;
        let format = gpui::ImageFormat::from_mime_type(mime_type)?;
        let dimensions = Self::image_dimensions(&bytes, format);
        Some((Arc::new(gpui::Image::from_bytes(format, bytes)), dimensions))
    }

    fn image_dimensions(bytes: &[u8], format: gpui::ImageFormat) -> Option<gpui::Size<u32>> {
        let format = match format {
            gpui::ImageFormat::Png => image::ImageFormat::Png,
            gpui::ImageFormat::Jpeg => image::ImageFormat::Jpeg,
            gpui::ImageFormat::Webp => image::ImageFormat::WebP,
            gpui::ImageFormat::Gif => image::ImageFormat::Gif,
            gpui::ImageFormat::Svg => return None,
            gpui::ImageFormat::Bmp => image::ImageFormat::Bmp,
            gpui::ImageFormat::Tiff => image::ImageFormat::Tiff,
            gpui::ImageFormat::Ico => image::ImageFormat::Ico,
            gpui::ImageFormat::Pnm => image::ImageFormat::Pnm,
        };

        image::ImageReader::with_format(std::io::Cursor::new(bytes), format)
            .into_dimensions()
            .ok()
            .map(|(width, height)| gpui::Size { width, height })
    }

    fn create_markdown_block(
        content: String,
        language_registry: &Arc<LanguageRegistry>,
        cx: &mut App,
    ) -> ContentBlock {
        ContentBlock {
            render: RenderBlock::Markdown {
                markdown: Self::create_markdown(content, language_registry, cx),
            },
            source: None,
        }
    }

    fn create_markdown(
        content: String,
        language_registry: &Arc<LanguageRegistry>,
        cx: &mut App,
    ) -> Entity<Markdown> {
        cx.new(|cx| {
            Markdown::new_with_options(
                content.into(),
                Some(language_registry.clone()),
                None,
                MarkdownOptions {
                    render_mermaid_diagrams: true,
                    render_metadata_blocks: true,
                    defer_parsing: true,
                    ..Default::default()
                },
                cx,
            )
        })
    }

    fn embedded_resource_markdown(
        resource: &acp_v2::EmbeddedResource,
        language_registry: &Arc<LanguageRegistry>,
        cx: &mut App,
    ) -> Option<Entity<Markdown>> {
        match &resource.resource {
            acp_v2::EmbeddedResourceResource::TextResourceContents(text) => Some(
                Self::create_markdown(Self::text_resource_markdown(text), language_registry, cx),
            ),
            acp_v2::EmbeddedResourceResource::BlobResourceContents(_) => None,
            _ => None,
        }
    }

    fn text_resource_markdown(resource: &acp_v2::TextResourceContents) -> String {
        match text_resource_render_mode(resource.mime_type.as_ref().map(AsRef::as_ref)) {
            TextResourceRenderMode::Markdown => resource.text.clone(),
            TextResourceRenderMode::CodeBlock(language) => {
                Self::fenced_code_block(&resource.text, language)
            }
        }
    }

    pub fn as_view(&self) -> ContentBlockView<'_> {
        ContentBlockView {
            render: &self.render,
            source: self.source.as_ref(),
        }
    }

    pub fn text_content<'a>(&'a self, cx: &'a App) -> Option<&'a str> {
        self.as_view().text_content(cx)
    }

    pub fn visible_content(&self, cx: &App) -> bool {
        self.as_view().visible_content(cx)
    }

    pub fn to_markdown<'a>(&'a self, cx: &'a App) -> &'a str {
        self.as_view().to_markdown(cx)
    }

    pub fn markdown(&self) -> Option<&Entity<Markdown>> {
        self.as_view().markdown()
    }

    pub fn image(&self) -> Option<(&Arc<gpui::Image>, Option<gpui::Size<u32>>)> {
        self.as_view().image()
    }

    fn resource_link_md(uri: &str, path_style: PathStyle) -> String {
        if let Some(uri) = MentionUri::parse(uri, path_style).log_err() {
            uri.as_link().to_string()
        } else {
            uri.to_string()
        }
    }
}

impl<'a> ContentBlockView<'a> {
    #[cfg(test)]
    fn unsupported_content(&self) -> Option<&'a acp_v2::ContentBlock> {
        match self.render {
            RenderBlock::Unsupported { .. } => self.source,
            _ => None,
        }
    }

    pub fn text_content(&self, cx: &'a App) -> Option<&'a str> {
        match &self.render {
            RenderBlock::Markdown { markdown } | RenderBlock::Unsupported { markdown } => {
                Some(markdown.read(cx).source())
            }
            RenderBlock::EmbeddedResource { .. } => {
                self.embedded_resource()
                    .and_then(|(resource, _)| match &resource.resource {
                        acp_v2::EmbeddedResourceResource::TextResourceContents(text) => {
                            Some(text.text.as_str())
                        }
                        _ => None,
                    })
            }
            RenderBlock::ResourceLink | RenderBlock::Image { .. } => None,
        }
    }
}

impl ContentBlock {
    fn fenced_code_block(text: &str, language: Option<&str>) -> String {
        let fence_len = text
            .as_bytes()
            .chunk_by(|left, right| left == right)
            .filter(|chunk| chunk.first() == Some(&b'`'))
            .map(|chunk| chunk.len() + 1)
            .max()
            .unwrap_or(3)
            .max(3);
        let fence = "`".repeat(fence_len);

        let mut markdown = String::new();
        markdown.push_str(&fence);
        if let Some(language) = language {
            markdown.push_str(language);
        }
        markdown.push('\n');
        markdown.push_str(text);
        if !text.ends_with('\n') {
            markdown.push('\n');
        }
        markdown.push_str(&fence);
        markdown
    }

    fn embedded_resource_string_contents(
        resource: &acp_v2::EmbeddedResource,
        path_style: PathStyle,
    ) -> String {
        match &resource.resource {
            acp_v2::EmbeddedResourceResource::TextResourceContents(text) => {
                Self::resource_link_md(&text.uri, path_style)
            }
            acp_v2::EmbeddedResourceResource::BlobResourceContents(blob) => {
                Self::resource_link_md(&blob.uri, path_style)
            }
            _ => String::new(),
        }
    }

    fn embedded_resource_text(resource: &acp_v2::EmbeddedResource) -> &str {
        match &resource.resource {
            acp_v2::EmbeddedResourceResource::TextResourceContents(text) => &text.text,
            acp_v2::EmbeddedResourceResource::BlobResourceContents(blob) => &blob.uri,
            _ => "",
        }
    }

    fn embedded_resource_label(resource: &acp_v2::EmbeddedResource) -> &str {
        match &resource.resource {
            acp_v2::EmbeddedResourceResource::TextResourceContents(text) => &text.uri,
            acp_v2::EmbeddedResourceResource::BlobResourceContents(blob) => &blob.uri,
            _ => "",
        }
    }
}

impl<'a> ContentBlockView<'a> {
    pub fn embedded_resource(
        &self,
    ) -> Option<(&'a acp_v2::EmbeddedResource, Option<&'a Entity<Markdown>>)> {
        match &self.render {
            RenderBlock::EmbeddedResource { markdown } => match self.source {
                Some(acp_v2::ContentBlock::Resource(resource)) => {
                    Some((resource, markdown.as_ref()))
                }
                _ => None,
            },
            _ => None,
        }
    }

    pub fn visible_content(&self, cx: &App) -> bool {
        match &self.render {
            RenderBlock::Markdown { markdown } | RenderBlock::Unsupported { markdown } => {
                !markdown.read(cx).source().trim().is_empty()
            }
            RenderBlock::EmbeddedResource { markdown, .. } => match markdown {
                Some(markdown) => !markdown.read(cx).source().trim().is_empty(),
                None => self.embedded_resource().is_some_and(|(resource, _)| {
                    !ContentBlock::embedded_resource_text(resource)
                        .trim()
                        .is_empty()
                }),
            },
            RenderBlock::ResourceLink | RenderBlock::Image { .. } => true,
        }
    }

    pub fn to_markdown(&self, cx: &'a App) -> &'a str {
        match &self.render {
            RenderBlock::Markdown { markdown } | RenderBlock::Unsupported { markdown } => {
                markdown.read(cx).source()
            }
            RenderBlock::EmbeddedResource { markdown, .. } => {
                if let Some(markdown) = markdown {
                    markdown.read(cx).source()
                } else {
                    self.embedded_resource().map_or("", |(resource, _)| {
                        ContentBlock::embedded_resource_label(resource)
                    })
                }
            }
            RenderBlock::ResourceLink => match self.source {
                Some(acp_v2::ContentBlock::ResourceLink(resource_link)) => &resource_link.uri,
                _ => "",
            },
            RenderBlock::Image { .. } => "`Image`",
        }
    }

    pub fn markdown(&self) -> Option<&'a Entity<Markdown>> {
        match &self.render {
            RenderBlock::Markdown { markdown } | RenderBlock::Unsupported { markdown } => {
                Some(markdown)
            }
            RenderBlock::EmbeddedResource { markdown, .. } => markdown.as_ref(),
            RenderBlock::ResourceLink | RenderBlock::Image { .. } => None,
        }
    }

    pub fn resource_link(&self) -> Option<&'a acp_v2::ResourceLink> {
        match &self.render {
            RenderBlock::ResourceLink => match self.source {
                Some(acp_v2::ContentBlock::ResourceLink(resource_link)) => Some(resource_link),
                _ => None,
            },
            _ => None,
        }
    }

    pub fn image(&self) -> Option<(&'a Arc<gpui::Image>, Option<gpui::Size<u32>>)> {
        match &self.render {
            RenderBlock::Image { image, dimensions } => Some((image, *dimensions)),
            _ => None,
        }
    }
}

enum TextResourceRenderMode {
    Markdown,
    CodeBlock(Option<&'static str>),
}

fn text_resource_render_mode(mime_type: Option<&str>) -> TextResourceRenderMode {
    let Some(mime_type) = mime_type else {
        return TextResourceRenderMode::CodeBlock(None);
    };
    let Ok(mime) = mime_type.parse::<mime::Mime>() else {
        return TextResourceRenderMode::CodeBlock(None);
    };

    let type_ = mime.type_().as_str();
    let subtype = mime.subtype().as_str();
    let suffix = mime.suffix().map(|suffix| suffix.as_str());

    if matches!(
        (type_, subtype),
        ("text", "markdown") | ("text", "x-markdown")
    ) {
        return TextResourceRenderMode::Markdown;
    }

    let language = match (type_, subtype, suffix) {
        (_, "json", _) | (_, _, Some("json")) => Some("json"),
        (_, "xml", _) | (_, _, Some("xml")) => Some("xml"),
        ("text", "html", _) => Some("html"),
        ("text", "css", _) => Some("css"),
        ("text", "csv", _) => Some("csv"),
        ("text", "tab-separated-values", _) => Some("tsv"),
        ("text", "javascript", _) | ("application", "javascript", _) => Some("javascript"),
        ("application", "x-javascript", _) => Some("javascript"),
        ("text", "typescript", _) | ("application", "typescript", _) => Some("typescript"),
        ("text", "x-shellscript", _) | ("application", "x-shellscript", _) => Some("sh"),
        ("application", "x-sh", _) => Some("sh"),
        ("text", "x-python", _) => Some("python"),
        ("text", "x-rust", _) => Some("rust"),
        ("text", "x-go", _) => Some("go"),
        ("text", "x-ruby", _) => Some("ruby"),
        ("text", "x-c", _) => Some("c"),
        // `mime` parses `text/x-c++` as subtype `x-c+` with an empty suffix.
        ("text", "x-c+", Some("")) => Some("cpp"),
        ("text", "plain", _) => None,
        ("text", _, _) => None,
        ("application", "graphql", _) => Some("graphql"),
        ("application", "toml", _) => Some("toml"),
        ("application", "yaml", _) | ("application", "x-yaml", _) => Some("yaml"),
        (_, _, Some("yaml" | "yml")) => Some("yaml"),
        _ => return TextResourceRenderMode::CodeBlock(None),
    };

    TextResourceRenderMode::CodeBlock(language)
}

#[derive(Debug)]
pub enum ToolCallContent {
    ContentBlock {
        block: ContentBlock,
        meta: Option<acp_v2::Meta>,
    },
    Diff(Entity<Diff>),
    LegacyDiff {
        source: acp_v1::Diff,
        diff: Entity<Diff>,
    },
    Terminal {
        terminal: Entity<Terminal>,
        meta: Option<acp_v2::Meta>,
    },
    DiffPatch {
        source: acp_v2::Diff,
        render: DiffPatch,
    },
    Other {
        source: acp_v2::ToolCallContent,
        markdown: Entity<Markdown>,
    },
}

enum PreparedToolCallContent {
    ContentBlock(acp_v2::Content),
    LegacyDiff(acp_v1::Diff),
    Terminal {
        terminal: Entity<Terminal>,
        meta: Option<acp_v2::Meta>,
    },
    DiffPatch(acp_v2::Diff),
    Other(acp_v2::ToolCallContent),
}

impl PreparedToolCallContent {
    fn prepare(
        content: Vec<acp_v1::ToolCallContent>,
        terminals: ToolTerminalResolver<'_>,
        cx: &App,
    ) -> Result<Vec<Self>> {
        let mut prepared = Vec::with_capacity(content.len());
        for content in content {
            let content = match content {
                acp_v1::ToolCallContent::Content(acp_v1::Content { content, meta, .. }) => {
                    Self::ContentBlock(acp_v2::Content::new(content::from_v1(content)?).meta(meta))
                }
                acp_v1::ToolCallContent::Diff(diff) => Self::LegacyDiff(diff),
                acp_v1::ToolCallContent::Terminal(acp_v1::Terminal {
                    terminal_id, meta, ..
                }) => Self::Terminal {
                    terminal: terminals.resolve(&acp_v2::TerminalId::new(terminal_id.0), cx)?,
                    meta,
                },
                _ => continue,
            };
            prepared.push(content);
        }
        Ok(prepared)
    }

    fn prepare_v2(
        content: Vec<acp_v2::ToolCallContent>,
        terminals: ToolTerminalResolver<'_>,
        cx: &App,
    ) -> Result<Vec<Self>> {
        content
            .into_iter()
            .map(|content| Self::from_v2(content, terminals, cx))
            .collect()
    }

    fn from_v2(
        content: acp_v2::ToolCallContent,
        terminals: ToolTerminalResolver<'_>,
        cx: &App,
    ) -> Result<Self> {
        match content {
            acp_v2::ToolCallContent::Content(content) => Ok(Self::ContentBlock(*content)),
            acp_v2::ToolCallContent::Diff(diff) => Ok(Self::DiffPatch(diff)),
            acp_v2::ToolCallContent::Terminal(terminal) => Ok(Self::Terminal {
                terminal: terminals.resolve(&terminal.terminal_id, cx)?,
                meta: terminal.meta,
            }),
            other => Ok(Self::Other(other)),
        }
    }
}

impl ToolCallContent {
    fn patch_preview(diff: &acp_v2::Diff) -> String {
        match diff.patch.as_ref() {
            Some(patch) => MarkdownCodeBlock {
                tag: if patch.format == acp_v2::DiffPatchFormat::GitPatch {
                    "diff"
                } else {
                    ""
                },
                text: &patch.text,
            }
            .to_string(),
            None => "Diff patch preview unavailable.".to_string(),
        }
    }

    fn from_prepared(
        content: PreparedToolCallContent,
        language_registry: &Arc<LanguageRegistry>,
        cx: &mut App,
    ) -> Self {
        match content {
            PreparedToolCallContent::ContentBlock(content) => Self::ContentBlock {
                block: ContentBlock::new_tool_content(content.content, language_registry, cx),
                meta: content.meta,
            },
            PreparedToolCallContent::LegacyDiff(source) => {
                let diff = cx.new(|_| {
                    Diff::finalized(
                        source.path.to_string_lossy().into_owned(),
                        source.old_text.clone(),
                        source.new_text.clone(),
                        language_registry.clone(),
                    )
                });
                Self::LegacyDiff { source, diff }
            }
            PreparedToolCallContent::Terminal { terminal, meta } => {
                Self::Terminal { terminal, meta }
            }
            PreparedToolCallContent::DiffPatch(source) => Self::DiffPatch {
                render: DiffPatch::new(&source, language_registry, cx),
                source,
            },
            PreparedToolCallContent::Other(source) => Self::Other {
                source,
                markdown: ContentBlock::create_markdown(
                    "Unsupported tool call content.".into(),
                    language_registry,
                    cx,
                ),
            },
        }
    }

    fn update_from_prepared(
        &mut self,
        new: PreparedToolCallContent,
        language_registry: &Arc<LanguageRegistry>,
        cx: &mut App,
    ) {
        match (&mut *self, new) {
            (
                Self::ContentBlock { block, meta },
                PreparedToolCallContent::ContentBlock(content),
            ) => {
                block.update_tool_content(content.content, language_registry, cx);
                *meta = content.meta;
            }
            (Self::LegacyDiff { source, .. }, PreparedToolCallContent::LegacyDiff(new_source))
                if source.path == new_source.path
                    && source.old_text == new_source.old_text
                    && source.new_text == new_source.new_text =>
            {
                *source = new_source;
            }
            (
                Self::Terminal { terminal, meta },
                PreparedToolCallContent::Terminal {
                    terminal: new_terminal,
                    meta: new_meta,
                },
            ) if *terminal == new_terminal => *meta = new_meta,
            (
                Self::DiffPatch { source, render },
                PreparedToolCallContent::DiffPatch(new_source),
            ) => {
                if source.patch != new_source.patch
                    || source.changes.len() != new_source.changes.len()
                    || source
                        .changes
                        .iter()
                        .zip(&new_source.changes)
                        .any(|(old, new)| {
                            old.operation != new.operation || old.file_type != new.file_type
                        })
                {
                    *render = DiffPatch::new(&new_source, language_registry, cx);
                }
                *source = new_source;
            }
            (Self::Other { source, .. }, PreparedToolCallContent::Other(new_source)) => {
                *source = new_source;
            }
            (_, new) => *self = Self::from_prepared(new, language_registry, cx),
        }
    }

    pub fn to_markdown(&self, cx: &App) -> String {
        match self {
            Self::ContentBlock { block, .. } => block.to_markdown(cx).to_owned(),
            Self::Diff(diff) | Self::LegacyDiff { diff, .. } => diff.read(cx).to_markdown(cx),
            Self::Terminal { terminal, .. } => terminal.read(cx).to_markdown(cx),
            Self::DiffPatch { source, .. } => format!(
                "{}\n\n{}",
                source.changes.iter().map(diff_change_label).join("\n"),
                Self::patch_preview(source)
            ),
            Self::Other { markdown, .. } => markdown.read(cx).source().to_string(),
        }
    }

    pub fn markdown(&self) -> Option<&Entity<Markdown>> {
        match self {
            Self::ContentBlock { block, .. } => block.markdown(),
            Self::DiffPatch { render, .. } => render.fallback.as_ref(),
            Self::Other { markdown, .. } => Some(markdown),
            Self::Diff(_) | Self::LegacyDiff { .. } | Self::Terminal { .. } => None,
        }
    }

    pub fn image(&self) -> Option<(&Arc<gpui::Image>, Option<gpui::Size<u32>>)> {
        match self {
            Self::ContentBlock { block, .. } => block.image(),
            _ => None,
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum ToolCallUpdate {
    V2(acp_v2::ToolCallUpdate),
    V1(acp_v1::ToolCallUpdate),
    UpdateDiff(ToolCallUpdateDiff),
    UpdateTerminal(ToolCallUpdateTerminal),
}

impl From<acp_v2::ToolCallUpdate> for ToolCallUpdate {
    fn from(update: acp_v2::ToolCallUpdate) -> Self {
        Self::V2(update)
    }
}

impl From<acp_v1::ToolCallUpdate> for ToolCallUpdate {
    fn from(update: acp_v1::ToolCallUpdate) -> Self {
        Self::V1(update)
    }
}

impl From<ToolCallUpdateDiff> for ToolCallUpdate {
    fn from(diff: ToolCallUpdateDiff) -> Self {
        Self::UpdateDiff(diff)
    }
}

#[derive(Debug, PartialEq)]
pub struct ToolCallUpdateDiff {
    pub id: acp_v2::ToolCallId,
    pub diff: Entity<Diff>,
}

impl From<ToolCallUpdateTerminal> for ToolCallUpdate {
    fn from(terminal: ToolCallUpdateTerminal) -> Self {
        Self::UpdateTerminal(terminal)
    }
}

#[derive(Debug, PartialEq)]
pub struct ToolCallUpdateTerminal {
    pub id: acp_v2::ToolCallId,
    pub terminal: Entity<Terminal>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum PlanIdentity {
    Legacy,
    Keyed(acp_v2::PlanId),
}

#[derive(Debug, Default)]
pub struct Plan {
    pub entries: Vec<PlanEntry>,
    pub meta: Option<acp_v2::Meta>,
    pub update_meta: Option<acp_v2::Meta>,
}

#[derive(Debug)]
pub struct PlanStats<'a> {
    pub in_progress_entry: Option<&'a PlanEntry>,
    pub pending: u32,
    pub completed: u32,
    pub cancelled: u32,
}

impl Plan {
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn stats(&self) -> PlanStats<'_> {
        let mut stats = PlanStats {
            in_progress_entry: None,
            pending: 0,
            completed: 0,
            cancelled: 0,
        };

        for entry in &self.entries {
            match &entry.source.status {
                acp_v2::PlanEntryStatus::InProgress => {
                    stats.in_progress_entry = stats.in_progress_entry.or(Some(entry));
                    stats.pending += 1;
                }
                acp_v2::PlanEntryStatus::Completed => {
                    stats.completed += 1;
                }
                acp_v2::PlanEntryStatus::Cancelled => {
                    stats.cancelled += 1;
                }
                _ => stats.pending += 1,
            }
        }

        stats
    }

    fn replace(
        &mut self,
        entries: Vec<acp_v2::PlanEntry>,
        meta: Option<acp_v2::Meta>,
        update_meta: Option<acp_v2::Meta>,
        cx: &mut App,
    ) -> bool {
        let changed = self.entries.len() != entries.len()
            || self.entries.iter().zip(&entries).any(|(old, new)| {
                old.source.content != new.content
                    || old.source.priority != new.priority
                    || old.source.status != new.status
            });
        let new_length = entries.len();
        let mut entries = entries.into_iter();
        for (old, new) in self.entries.iter_mut().zip(entries.by_ref()) {
            if old.source.content != new.content {
                update_markdown_in_place(&old.content, &new.content, cx);
            }
            old.source = new;
        }
        self.entries
            .extend(entries.map(|source| PlanEntry::new(source, cx)));
        self.entries.truncate(new_length);
        self.meta = meta;
        self.update_meta = update_meta;
        changed
    }
}

#[derive(Debug)]
pub struct PlanEntry {
    pub source: acp_v2::PlanEntry,
    pub content: Entity<Markdown>,
}

impl PlanEntry {
    fn new(source: acp_v2::PlanEntry, cx: &mut App) -> Self {
        Self {
            content: cx.new(|cx| Markdown::new(source.content.clone().into(), None, None, cx)),
            source,
        }
    }
}

fn plan_entry_from_v1(entry: acp_v1::PlanEntry) -> Result<acp_v2::PlanEntry> {
    let priority = match entry.priority {
        acp_v1::PlanEntryPriority::High => acp_v2::PlanEntryPriority::High,
        acp_v1::PlanEntryPriority::Medium => acp_v2::PlanEntryPriority::Medium,
        acp_v1::PlanEntryPriority::Low => acp_v2::PlanEntryPriority::Low,
        // V2 preserves future wire names through its Other variant.
        other => serde_json::from_value(serde_json::to_value(other)?)?,
    };
    let status = match entry.status {
        acp_v1::PlanEntryStatus::Pending => acp_v2::PlanEntryStatus::Pending,
        acp_v1::PlanEntryStatus::InProgress => acp_v2::PlanEntryStatus::InProgress,
        acp_v1::PlanEntryStatus::Completed => acp_v2::PlanEntryStatus::Completed,
        other => serde_json::from_value(serde_json::to_value(other)?)?,
    };
    Ok(acp_v2::PlanEntry::new(entry.content, priority, status).meta(entry.meta))
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenUsage {
    pub max_tokens: u64,
    pub used_tokens: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub max_output_tokens: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct SessionCost {
    pub amount: f64,
    pub currency: SharedString,
    pub meta: Option<acp_v2::Meta>,
}

pub const TOKEN_USAGE_WARNING_THRESHOLD: f32 = 0.8;

impl TokenUsage {
    pub fn ratio(&self) -> TokenUsageRatio {
        #[cfg(debug_assertions)]
        let warning_threshold: f32 = std::env::var("ZED_THREAD_WARNING_THRESHOLD")
            .unwrap_or(TOKEN_USAGE_WARNING_THRESHOLD.to_string())
            .parse()
            .unwrap();
        #[cfg(not(debug_assertions))]
        let warning_threshold: f32 = TOKEN_USAGE_WARNING_THRESHOLD;

        // When the maximum is unknown because there is no selected model,
        // avoid showing the token limit warning.
        if self.max_tokens == 0 {
            TokenUsageRatio::Normal
        } else if self.used_tokens >= self.max_tokens {
            TokenUsageRatio::Exceeded
        } else if self.used_tokens as f32 / self.max_tokens as f32 >= warning_threshold {
            TokenUsageRatio::Warning
        } else {
            TokenUsageRatio::Normal
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum TokenUsageRatio {
    Normal,
    Warning,
    Exceeded,
}

#[derive(Debug, Clone)]
pub struct RetryStatus {
    pub last_error: SharedString,
    pub attempt: usize,
    pub max_attempts: usize,
    pub started_at: Instant,
    pub duration: Duration,
    pub meta: Option<acp_v2::Meta>,
}

pub const REFUSAL_FALLBACK_MODEL_META_KEY: &str = "refusal_fallback_model";

pub fn meta_with_refusal_fallback(model_name: &str) -> acp_v2::Meta {
    acp_v2::Meta::from_iter([(REFUSAL_FALLBACK_MODEL_META_KEY.into(), model_name.into())])
}

pub fn refusal_fallback_model_from_meta(meta: &Option<acp_v2::Meta>) -> Option<SharedString> {
    meta.as_ref()
        .and_then(|m| m.get(REFUSAL_FALLBACK_MODEL_META_KEY))
        .and_then(|v| v.as_str())
        .map(|s| SharedString::from(s.to_owned()))
}

struct RunningTurn {
    id: u32,
    first_entry_index: usize,
    send_task: Task<()>,
}

fn stop_reason_from_v1(reason: &acp_v1::StopReason) -> Option<acp_v2::StopReason> {
    Some(match reason {
        acp_v1::StopReason::EndTurn => acp_v2::StopReason::EndTurn,
        acp_v1::StopReason::MaxTokens => acp_v2::StopReason::MaxTokens,
        acp_v1::StopReason::MaxTurnRequests => acp_v2::StopReason::MaxTurnRequests,
        acp_v1::StopReason::Refusal => acp_v2::StopReason::Refusal,
        acp_v1::StopReason::Cancelled => acp_v2::StopReason::Cancelled,
        _ => return None,
    })
}

pub struct AcpThread {
    session_info: AgentSessionInfo,
    parent_session_id: Option<acp_v2::SessionId>,
    /// Child sessions this session owns, such as subagents, in registration order.
    subagents: IndexMap<acp_v2::SessionId, Entity<AcpThread>>,
    /// Set when this session is an ACP subagent announced by its parent.
    subagent_association: Option<subagents::SubagentAssociation>,
    provisional_title: Option<SharedString>,
    entries: Vec<AgentThreadEntry>,
    // Notices stay with the live session, but never enter conversation history or exports.
    notices: Vec<(usize, acp_v2::Notice)>,
    next_notice_id: usize,
    elicitations: ElicitationStore,
    permission_requests: IndexMap<PermissionRequestId, PermissionRequest>,
    plans: HashMap<PlanIdentity, Plan>,
    visible_plan: Option<PlanIdentity>,
    project: Entity<Project>,
    action_log: Entity<ActionLog>,
    shared_buffers: HashMap<Entity<Buffer>, BufferSnapshot>,
    turn_id: u32,
    running_turn: Option<RunningTurn>,
    connection: Rc<dyn AgentConnection>,
    submissions: SessionSubmissions,
    activity: SessionActivity,
    received_session_state: bool,
    /// Each idle state a subagent reported, with how many entries it had then. The subagent
    /// reports one state for all its work, so this is what shows the outcome of an earlier
    /// message to it after it has received later ones.
    subagent_outcomes: Vec<(usize, acp_v2::StateUpdate)>,
    reported_usage: Option<ReportedTokenUsage>,
    token_usage: Option<TokenUsage>,
    cost: Option<SessionCost>,
    prompt_capabilities: acp_v2::PromptCapabilities,
    available_commands: Vec<acp_v2::AvailableCommand>,
    _observe_prompt_capabilities: Task<anyhow::Result<()>>,
    _idle_sleep_subscriptions: Vec<Subscription>,
    terminals: HashMap<acp_v2::TerminalId, Entity<Terminal>>,
    pending_terminal_output: HashMap<acp_v2::TerminalId, Vec<Vec<u8>>>,
    pending_terminal_exit: HashMap<acp_v2::TerminalId, acp_v1::TerminalExitStatus>,
    had_error: bool,
    local_execution_error: Option<Arc<anyhow::Error>>,
    /// The user's unsent prompt text, persisted so it can be restored when reloading the thread.
    draft_prompt: Option<Vec<acp_v2::ContentBlock>>,
    /// Lets observers detect draft changes without comparing prompts.
    draft_prompt_revision: usize,
    /// The initial scroll position for the thread view, set during session registration.
    ui_scroll_position: Option<gpui::ListOffset>,
    /// A cursor over retained source, rather than a second text store, lets the UI
    /// reveal text gradually without changing the authoritative message.
    streaming_text_buffer: Option<StreamingTextBuffer>,
    idle_sleep_prevention: IdleSleepPrevention,
}

enum IdleSleepPrevention {
    Inactive,
    Acquiring { _task: Task<()> },
    Active { _guard: ActivityGuard },
    Failed,
}

#[derive(Default)]
struct TextCursor {
    source_index: usize,
    byte_offset: usize,
    pending_bytes: usize,
}

impl TextCursor {
    fn take(&mut self, sources: &[acp_v2::ContentBlock], max_bytes: usize) -> Option<String> {
        let max_bytes = max_bytes.min(self.pending_bytes);
        let mut revealed = String::with_capacity(max_bytes);
        while revealed.len() < max_bytes {
            let acp_v2::ContentBlock::Text(text) = sources.get(self.source_index)? else {
                return None;
            };
            let pending = text.text.get(self.byte_offset..)?;
            let boundary = pending.ceil_char_boundary(max_bytes - revealed.len());
            self.pending_bytes = self.pending_bytes.checked_sub(boundary)?;
            revealed.push_str(&pending[..boundary]);
            self.byte_offset += boundary;
            if self.byte_offset == text.text.len() {
                self.source_index += 1;
                self.byte_offset = 0;
            }
        }
        Some(revealed)
    }
}

#[derive(PartialEq)]
struct StreamingTextTarget {
    entry_index: usize,
    chunk_index: usize,
    markdown: Entity<Markdown>,
}

impl StreamingTextTarget {
    fn content<'a>(&self, entries: &'a [AgentThreadEntry]) -> Option<&'a MessageContent> {
        let AgentThreadEntry::AssistantMessage(message) = entries.get(self.entry_index)? else {
            return None;
        };
        let (AssistantMessageChunk::Message { block, .. }
        | AssistantMessageChunk::Thought { block, .. }) = message.chunks.get(self.chunk_index)?;
        // A removed entry's position can be reused before the next reveal tick.
        (block.trailing_text()? == &self.markdown).then_some(block)
    }
}

struct StreamingTextBuffer {
    cursor: TextCursor,
    /// The number of bytes to reveal per timer turn.
    bytes_to_reveal_per_tick: usize,
    target: StreamingTextTarget,
    _reveal_task: Task<()>,
}

impl StreamingTextBuffer {
    /// The number of milliseconds between each timer tick, controlling how quickly
    /// text is revealed.
    const TASK_UPDATE_MS: u64 = 16;
    /// The time in milliseconds to reveal the entire pending text.
    const REVEAL_TARGET: f32 = 200.0;

    fn reveal(&mut self, entries: &[AgentThreadEntry], max_bytes: usize, cx: &mut App) -> bool {
        let Some(content) = self.target.content(entries) else {
            return false;
        };
        let Some(revealed) = self.cursor.take(content.source_blocks(), max_bytes) else {
            log::error!("Invalid source cursor for streaming agent text");
            return false;
        };
        if !revealed.is_empty() {
            self.target
                .markdown
                .update(cx, |markdown, cx| markdown.append(&revealed, cx));
        }
        true
    }
}

impl From<&AcpThread> for ActionLogTelemetry {
    fn from(value: &AcpThread) -> Self {
        Self {
            agent_telemetry_id: value.connection().telemetry_id(),
            session_id: value.session_id().0.clone(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ReportedTokenUsage {
    activity_generation: u64,
    input_tokens: u64,
    output_tokens: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputTokensUpdate {
    Unchanged,
    /// Preserve additive presentation for legacy/native usage notifications.
    LegacyAdd {
        activity_generation: u64,
        output_tokens: u64,
    },
    Snapshot {
        activity_generation: u64,
        output_tokens: u64,
    },
}

#[derive(Debug)]
pub enum AcpThreadEvent {
    StatusChanged,
    /// The parent's title, description, or capabilities for this subagent changed. Views that
    /// read them from the parent's side, such as its subagent card, observe this event.
    SubagentAssociationUpdated,
    /// Idle confirmed after unconfirmed or unknown activity, without completing
    /// foreground work.
    ActivityConfirmedIdle,
    SubmissionUpdated(SubmissionId),
    PromptUpdated,
    NewEntry,
    TitleUpdated,
    NoticesUpdated,
    TokenUsageUpdated(OutputTokensUpdate),
    EntryUpdated(usize),
    EntriesRemoved(Range<usize>),
    ToolAuthorizationRequested(PermissionRequestId),
    ToolAuthorizationReceived(PermissionRequestId),
    ElicitationRequested(ElicitationEntryId),
    /// The request left `Pending`; this does not imply delivery to its response waiter.
    ElicitationResponded(ElicitationEntryId),
    Retry(RetryStatus),
    /// A child session was registered; read it with [`AcpThread::subagent`].
    SubagentSpawned(acp_v2::SessionId),
    Stopped {
        activity_generation: u64,
        activity_duration: Option<Duration>,
        stop_reason: Option<acp_v2::StopReason>,
    },
    Error,
    ConnectionLost,
    LoadError(LoadError),
    PromptCapabilitiesUpdated,
    Refusal,
    AvailableCommandsUpdated(Vec<acp_v2::AvailableCommand>),
    ModeUpdated(acp_v2::SessionConfigValueId),
    ConfigOptionsUpdated(Vec<acp_v2::SessionConfigOption>),
    WorkingDirectoriesUpdated,
}

impl EventEmitter<AcpThreadEvent> for AcpThread {}

#[derive(Debug, Clone)]
pub enum TerminalProviderEvent {
    Created {
        terminal_id: acp_v1::TerminalId,
        label: String,
        cwd: Option<PathBuf>,
        output_byte_limit: Option<u64>,
        terminal: Entity<::terminal::Terminal>,
    },
    Output {
        terminal_id: acp_v1::TerminalId,
        data: Vec<u8>,
    },
    TitleChanged {
        terminal_id: acp_v1::TerminalId,
        title: String,
    },
    Exit {
        terminal_id: acp_v1::TerminalId,
        status: acp_v1::TerminalExitStatus,
    },
}

#[derive(Debug, Clone)]
pub enum TerminalProviderCommand {
    WriteInput {
        terminal_id: acp_v2::TerminalId,
        bytes: Vec<u8>,
    },
    Resize {
        terminal_id: acp_v2::TerminalId,
        cols: u16,
        rows: u16,
    },
    Close {
        terminal_id: acp_v2::TerminalId,
    },
}

#[derive(PartialEq, Eq, Debug)]
pub enum ThreadStatus {
    Unconfirmed,
    Idle,
    Generating,
}

#[derive(Debug, Clone)]
pub enum LoadError {
    Unsupported {
        command: SharedString,
        current_version: SharedString,
        minimum_version: SharedString,
    },
    FailedToInstall(SharedString),
    Exited {
        status: ExitStatus,
        stderr: Option<SharedString>,
    },
    Other(SharedString),
}

impl Display for LoadError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::Unsupported {
                command: path,
                current_version,
                minimum_version,
            } => {
                write!(
                    f,
                    "version {current_version} from {path} is not supported (need at least {minimum_version})"
                )
            }
            LoadError::FailedToInstall(msg) => write!(f, "Failed to install: {msg}"),
            LoadError::Exited { status, .. } => write!(f, "Server exited with status {status}"),
            LoadError::Other(msg) => write!(f, "{msg}"),
        }
    }
}

impl Error for LoadError {}

impl AcpThread {
    pub fn new(
        parent_session_id: Option<acp_v2::SessionId>,
        title: Option<SharedString>,
        work_dirs: Option<PathList>,
        connection: Rc<dyn AgentConnection>,
        project: Entity<Project>,
        action_log: Entity<ActionLog>,
        session_id: acp_v2::SessionId,
        mut prompt_capabilities_rx: watch::Receiver<acp_v2::PromptCapabilities>,
        cx: &mut Context<Self>,
    ) -> Self {
        let prompt_capabilities = prompt_capabilities_rx.borrow().clone();
        let task = cx.spawn::<_, anyhow::Result<()>>(async move |this, cx| {
            loop {
                let caps = prompt_capabilities_rx.recv().await?;
                this.update(cx, |this, cx| {
                    this.prompt_capabilities = caps;
                    cx.emit(AcpThreadEvent::PromptCapabilitiesUpdated);
                })?;
            }
        });
        let idle_sleep_settings_subscription = cx.observe_global::<SettingsStore>(|this, cx| {
            this.update_idle_sleep_prevention(cx);
        });
        let idle_sleep_event_subscription =
            cx.subscribe_self(|this, event: &AcpThreadEvent, cx| match event {
                AcpThreadEvent::StatusChanged
                | AcpThreadEvent::EntriesRemoved(_)
                | AcpThreadEvent::ToolAuthorizationRequested(_)
                | AcpThreadEvent::ToolAuthorizationReceived(_)
                | AcpThreadEvent::ElicitationRequested(_)
                | AcpThreadEvent::ElicitationResponded(_) => {
                    if !this.uses_reported_activity() {
                        this.sync_local_action_state(cx);
                    }
                    this.update_idle_sleep_prevention(cx);
                }
                AcpThreadEvent::PromptUpdated
                | AcpThreadEvent::ActivityConfirmedIdle
                | AcpThreadEvent::SubmissionUpdated(_)
                | AcpThreadEvent::NewEntry
                | AcpThreadEvent::TitleUpdated
                | AcpThreadEvent::NoticesUpdated
                | AcpThreadEvent::TokenUsageUpdated(_)
                | AcpThreadEvent::EntryUpdated(_)
                | AcpThreadEvent::Retry(_)
                | AcpThreadEvent::SubagentSpawned(_)
                | AcpThreadEvent::SubagentAssociationUpdated
                | AcpThreadEvent::Stopped { .. }
                | AcpThreadEvent::Error
                | AcpThreadEvent::ConnectionLost
                | AcpThreadEvent::LoadError(_)
                | AcpThreadEvent::PromptCapabilitiesUpdated
                | AcpThreadEvent::Refusal
                | AcpThreadEvent::AvailableCommandsUpdated(_)
                | AcpThreadEvent::ModeUpdated(_)
                | AcpThreadEvent::ConfigOptionsUpdated(_)
                | AcpThreadEvent::WorkingDirectoriesUpdated => {}
            });

        let receipt_submissions = connection.receipt_submissions(&session_id, cx);
        Self {
            parent_session_id,
            session_info: AgentSessionInfo {
                session_id,
                work_dirs,
                title,
                updated_at: None,
                created_at: None,
                meta: None,
            },
            action_log,
            shared_buffers: Default::default(),
            entries: Default::default(),
            notices: Vec::new(),
            next_notice_id: 0,
            elicitations: ElicitationStore::default(),
            permission_requests: IndexMap::default(),
            subagents: IndexMap::default(),
            subagent_association: None,
            plans: HashMap::default(),
            visible_plan: None,
            provisional_title: None,
            project,
            running_turn: None,
            turn_id: 0,
            submissions: SessionSubmissions::new(
                receipt_submissions.map_or(SubmissionTransport::Completion, |transport| {
                    SubmissionTransport::Receipt(transport)
                }),
            ),
            activity: SessionActivity::default(),
            received_session_state: false,
            subagent_outcomes: Vec::new(),
            reported_usage: None,
            connection,
            token_usage: None,
            cost: None,
            prompt_capabilities,
            available_commands: Vec::new(),
            _observe_prompt_capabilities: task,
            _idle_sleep_subscriptions: vec![
                idle_sleep_settings_subscription,
                idle_sleep_event_subscription,
            ],
            terminals: HashMap::default(),
            pending_terminal_output: HashMap::default(),
            pending_terminal_exit: HashMap::default(),
            had_error: false,
            local_execution_error: None,
            draft_prompt: None,
            draft_prompt_revision: 0,
            ui_scroll_position: None,
            streaming_text_buffer: None,
            idle_sleep_prevention: IdleSleepPrevention::Inactive,
        }
    }

    /// Marks an ACP subagent session, whose work its parent agent owns. Zed cannot
    /// submit prompts to it, and its foreground state is unconfirmed until the agent
    /// reports one.
    pub fn agent_owned(mut self) -> Self {
        self.submissions = SessionSubmissions::new(SubmissionTransport::AgentOwned);
        self.activity.mark_unconfirmed();
        self.subagent_association = Some(subagents::SubagentAssociation::default());
        self
    }

    /// Marks a subagent session whose parent runs in Zed, such as a native subagent. Unlike
    /// `agent_owned`, Zed still submits its prompts.
    pub fn set_subagent_association(&mut self, association: subagents::SubagentAssociation) {
        self.subagent_association = Some(association);
    }

    pub fn subagent_association(&self) -> Option<&subagents::SubagentAssociation> {
        self.subagent_association.as_ref()
    }

    pub fn update_subagent_association(
        &mut self,
        update: &acp_v2::SubagentUpdate,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        let association = self
            .subagent_association
            .as_mut()
            .context("Session is not an ACP subagent")?;
        association.apply(update);
        cx.emit(AcpThreadEvent::SubagentAssociationUpdated);
        cx.notify();
        Ok(())
    }

    /// Ends a history replay into this session, after a successful load or resume. The
    /// subagents RFD treats the work states and cancel grants that the replay reported for
    /// subagents as history, outcomes included: each subagent's work state is unconfirmed,
    /// and it cannot be cancelled, until the agent reports current ones after the response.
    /// Titles and transcripts stay.
    pub fn invalidate_replayed_subagent_state(&mut self, cx: &mut Context<Self>) {
        for subagent in self.subagents.values() {
            subagent.update(cx, |subagent, cx| {
                if let Some(association) = subagent.subagent_association.as_mut() {
                    association.can_cancel = false;
                    cx.notify();
                }
                subagent.clear_session_state(cx).log_err();
                subagent.invalidate_replayed_subagent_state(cx);
            });
        }
    }

    pub fn parent_session_id(&self) -> Option<&acp_v2::SessionId> {
        self.parent_session_id.as_ref()
    }

    pub fn prompt_capabilities(&self) -> &acp_v2::PromptCapabilities {
        &self.prompt_capabilities
    }

    pub fn available_commands(&self) -> &[acp_v2::AvailableCommand] {
        &self.available_commands
    }

    pub fn update_available_commands(
        &mut self,
        commands: Vec<acp_v2::AvailableCommand>,
        cx: &mut Context<Self>,
    ) {
        self.available_commands = commands.clone();
        cx.emit(AcpThreadEvent::AvailableCommandsUpdated(commands));
    }

    pub fn is_draft_thread(&self) -> bool {
        self.entries().is_empty()
    }

    pub fn draft_prompt(&self) -> Option<&[acp_v2::ContentBlock]> {
        self.draft_prompt.as_deref()
    }

    pub fn draft_prompt_revision(&self) -> usize {
        self.draft_prompt_revision
    }

    pub fn set_draft_prompt(
        &mut self,
        prompt: Option<Vec<acp_v2::ContentBlock>>,
        cx: &mut Context<Self>,
    ) {
        cx.emit(AcpThreadEvent::PromptUpdated);
        self.draft_prompt = prompt;
        self.draft_prompt_revision += 1;
    }

    pub fn ui_scroll_position(&self) -> Option<gpui::ListOffset> {
        self.ui_scroll_position
    }

    pub fn set_ui_scroll_position(&mut self, position: Option<gpui::ListOffset>) {
        self.ui_scroll_position = position;
    }

    pub fn connection(&self) -> &Rc<dyn AgentConnection> {
        &self.connection
    }

    pub fn action_log(&self) -> &Entity<ActionLog> {
        &self.action_log
    }

    pub fn project(&self) -> &Entity<Project> {
        &self.project
    }

    pub fn title(&self) -> Option<SharedString> {
        self.session_info
            .title
            .clone()
            .or_else(|| self.provisional_title.clone())
    }

    pub fn has_provisional_title(&self) -> bool {
        self.provisional_title.is_some()
    }

    pub fn entries(&self) -> &[AgentThreadEntry] {
        &self.entries
    }

    pub fn notices(&self) -> &[(usize, acp_v2::Notice)] {
        &self.notices
    }

    pub fn push_notice(&mut self, notice: acp_v2::Notice, cx: &mut Context<Self>) {
        let notice_id = self.next_notice_id;
        self.next_notice_id += 1;
        self.notices.push((notice_id, notice));
        cx.emit(AcpThreadEvent::NoticesUpdated);
        cx.notify();
    }

    pub fn dismiss_notice(&mut self, notice_id: usize, cx: &mut Context<Self>) {
        let previous_count = self.notices.len();
        self.notices.retain(|(id, _)| *id != notice_id);
        if self.notices.len() != previous_count {
            cx.emit(AcpThreadEvent::NoticesUpdated);
            cx.notify();
        }
    }

    /// Form elicitations stop rendering after accept, so the associated tool
    /// call is the scroll target for the user's answer.
    pub fn is_user_authored_scroll_target(&self, entry: &AgentThreadEntry) -> bool {
        match entry {
            AgentThreadEntry::UserMessage(_) => true,
            AgentThreadEntry::ToolCall(call) => self.tool_call_has_accepted_user_answer(&call.id),
            _ => false,
        }
    }

    fn tool_call_has_accepted_user_answer(&self, tool_call_id: &acp_v2::ToolCallId) -> bool {
        self.elicitations.elicitations().iter().any(|elicitation| {
            matches!(elicitation.status, ElicitationStatus::Accepted)
                // Accepting a URL elicitation only consents to opening a link,
                // so it isn't an answer.
                && matches!(elicitation.request.mode, acp_v2::ElicitationMode::Form(_))
                && matches!(
                    elicitation.request.scope(),
                    acp_v2::ElicitationScope::Session(scope)
                        if scope.tool_call_id.as_ref() == Some(tool_call_id)
                )
        })
    }

    pub fn is_compacting(&self) -> bool {
        self.entries.iter().rev().any(|entry| {
            matches!(
                entry,
                AgentThreadEntry::ContextCompaction(compaction) if compaction.is_in_progress()
            )
        })
    }

    pub fn invalidate_mermaid_caches(&self, cx: &mut App) {
        for entry in &self.entries {
            let chunks = match entry {
                AgentThreadEntry::AssistantMessage(message) => &message.chunks,
                _ => continue,
            };
            for chunk in chunks {
                let block = match chunk {
                    AssistantMessageChunk::Message { block, .. } => block,
                    AssistantMessageChunk::Thought { block, .. } => block,
                };
                for markdown in block.markdowns() {
                    markdown.update(cx, |markdown, cx| {
                        markdown.invalidate_mermaid_cache(cx);
                    });
                }
            }
        }
    }

    pub fn session_id(&self) -> &acp_v2::SessionId {
        &self.session_info.session_id
    }

    pub fn session_info(&self) -> &AgentSessionInfo {
        &self.session_info
    }

    pub fn supports_truncate(&self, cx: &App) -> bool {
        self.connection.truncate(self.session_id(), cx).is_some()
    }

    /// Gates editing user messages and rewinding the thread to them.
    pub fn can_rewind_to(&self, client_id: Option<&ClientUserMessageId>, cx: &App) -> bool {
        client_id.is_some() && self.parent_session_id.is_none() && self.supports_truncate(cx)
    }

    pub fn work_dirs(&self) -> Option<&PathList> {
        self.session_info.work_dirs.as_ref()
    }

    pub fn set_work_dirs(&mut self, work_dirs: PathList, cx: &mut Context<Self>) {
        self.session_info.work_dirs = Some(work_dirs.clone());
        cx.emit(AcpThreadEvent::WorkingDirectoriesUpdated);
        for subagent in self.subagents.values() {
            subagent.update(cx, |subagent, cx| {
                subagent.inherit_subagent_work_dirs(Some(&work_dirs), cx)
            });
        }
    }

    /// Updates an ACP subagent's working directories from what its parent reported,
    /// inheriting each unreported field from the parent's working directories.
    pub fn inherit_subagent_work_dirs(
        &mut self,
        parent_work_dirs: Option<&PathList>,
        cx: &mut Context<Self>,
    ) {
        if !matches!(
            self.submissions.transport(),
            SubmissionTransport::AgentOwned
        ) {
            return;
        }
        let Some(work_dirs) = self
            .subagent_association
            .as_ref()
            .and_then(|association| association.effective_work_dirs(parent_work_dirs))
        else {
            return;
        };
        // `PathList` equality ignores order, but the first path is the cwd.
        if self
            .work_dirs()
            .is_some_and(|current| current.ordered_paths().eq(work_dirs.ordered_paths()))
        {
            return;
        }
        self.set_work_dirs(work_dirs, cx);
    }

    pub fn status(&self) -> ThreadStatus {
        match self.foreground_activity() {
            ForegroundActivity::Unconfirmed | ForegroundActivity::Unknown => {
                ThreadStatus::Unconfirmed
            }
            ForegroundActivity::Idle => ThreadStatus::Idle,
            ForegroundActivity::Running | ForegroundActivity::RequiresAction => {
                ThreadStatus::Generating
            }
        }
    }

    /// Whether the agent reports this session's foreground state, rather than Zed
    /// deriving it from its own prompt turns.
    pub fn uses_reported_activity(&self) -> bool {
        !matches!(
            self.submissions.transport(),
            SubmissionTransport::Completion
        )
    }

    pub fn foreground_state(&self) -> Option<&acp_v2::StateUpdate> {
        self.activity.state()
    }

    /// The work the agent reported as running independently of foreground work in its latest
    /// state snapshot. `None` means the agent hasn't reported it, not that nothing is running.
    pub fn background_work(&self) -> Option<&[acp_v2::BackgroundWork]> {
        self.foreground_state()?.background()
    }

    pub fn mark_activity_unconfirmed(&mut self, cx: &mut Context<Self>) {
        if self.uses_reported_activity()
            && !self.received_session_state
            && self.activity.generation() == 0
        {
            self.activity.mark_unconfirmed();
            cx.emit(AcpThreadEvent::StatusChanged);
            cx.notify();
        }
    }

    /// Distinguishes a received snapshot from the default Idle used by new sessions.
    pub fn has_received_session_state(&self) -> bool {
        self.received_session_state
    }

    /// Restores the outcomes a subagent reported before it was saved, each with the entry
    /// count when it was reported, after its entries have been replayed. The last one is
    /// also its current state, unless entries followed it, as when Zed quit mid-run.
    pub fn restore_subagent_outcomes(
        &mut self,
        outcomes: Vec<(usize, acp_v2::StateUpdate)>,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        anyhow::ensure!(
            self.subagent_association.is_some(),
            "Session is not a subagent"
        );
        let current = outcomes
            .last()
            .filter(|(entry_count, _)| *entry_count == self.entries.len())
            .map(|(_, state)| state.clone());
        self.subagent_outcomes = outcomes;
        match current {
            Some(state) => self.update_session_state(state, cx),
            None => {
                cx.notify();
                Ok(())
            }
        }
    }

    /// The first idle state this subagent reported after the entry at `entry_ix` arrived.
    pub fn subagent_outcome_after(&self, entry_ix: usize) -> Option<&acp_v2::StateUpdate> {
        self.subagent_outcomes
            .iter()
            .find(|(entry_count, _)| *entry_count > entry_ix)
            .map(|(_, state)| state)
    }

    pub fn foreground_activity(&self) -> ForegroundActivity {
        self.activity.phase()
    }

    pub fn activity_generation(&self) -> u64 {
        self.activity.generation()
    }

    pub fn reported_activity_output_tokens(&self) -> Option<u64> {
        self.reported_usage
            .filter(|usage| usage.activity_generation == self.activity.generation())
            .map(|usage| usage.output_tokens)
    }

    pub fn activity_started_at(&self) -> Option<Instant> {
        self.activity.started_at()
    }

    pub fn activity_duration(&self) -> Option<Duration> {
        self.activity.duration()
    }

    pub fn had_error(&self) -> bool {
        self.had_error
    }

    pub fn local_execution_error(&self) -> Option<&Arc<anyhow::Error>> {
        self.local_execution_error.as_ref()
    }

    pub fn report_local_execution_error(
        &mut self,
        error: Arc<anyhow::Error>,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        anyhow::ensure!(
            self.uses_reported_activity(),
            "Session does not report foreground state"
        );
        let reported_error = error
            .downcast_ref::<acp_v2::Error>()
            .cloned()
            .unwrap_or_else(|| {
                let mut reported_error = acp_v2::Error::internal_error();
                reported_error.message = error.to_string();
                reported_error
            });
        self.local_execution_error = Some(error);
        self.update_session_state(
            acp_v2::StateUpdate::Idle(acp_v2::IdleStateUpdate::new().stop_reason(
                acp_v2::StopReason::Error(acp_v2::ErrorStopReason::new().error(reported_error)),
            )),
            cx,
        )
    }

    pub fn is_waiting_for_confirmation(&self) -> bool {
        if self.uses_reported_activity() {
            return self.foreground_activity() == ForegroundActivity::RequiresAction;
        }
        self.has_pending_turn_action()
    }

    fn has_pending_turn_action(&self) -> bool {
        if self
            .permission_requests
            .values()
            .any(|request| request.generic_request().is_some())
        {
            return true;
        }
        for entry in self.entries.iter().rev() {
            match entry {
                AgentThreadEntry::UserMessage(_) => return false,
                AgentThreadEntry::ToolCall(call)
                    if call
                        .authorization_id()
                        .is_some_and(|id| self.permission_request(id).is_some()) =>
                {
                    return true;
                }
                AgentThreadEntry::Elicitation(elicitation_id)
                    if self.elicitations.elicitation(elicitation_id).is_some_and(
                        |(_, elicitation)| {
                            matches!(elicitation.status, ElicitationStatus::Pending { .. })
                        },
                    ) =>
                {
                    return true;
                }
                AgentThreadEntry::ToolCall(_)
                | AgentThreadEntry::Elicitation(_)
                | AgentThreadEntry::AssistantMessage(_)
                | AgentThreadEntry::SessionMessage(_)
                | AgentThreadEntry::ContextCompaction(_) => {}
            }
        }
        false
    }

    pub fn token_usage(&self) -> Option<&TokenUsage> {
        self.token_usage.as_ref()
    }

    pub fn submission(&self, id: SubmissionId) -> Option<&SubmissionRecord> {
        self.submissions.get(id)
    }

    pub fn latest_submission_id(&self) -> Option<SubmissionId> {
        self.submissions.latest_id()
    }

    pub fn has_unsettled_submissions(&self) -> bool {
        self.submissions.has_unsettled()
    }

    pub fn recoverable_submissions(
        &self,
    ) -> impl Iterator<Item = (SubmissionId, &SubmissionRecord)> {
        self.submissions.recoverable()
    }

    pub fn is_observation_closed(&self) -> bool {
        self.activity.observation_closed()
    }

    pub fn is_idle_for_retention(&self) -> bool {
        self.status() == ThreadStatus::Idle
            && !self.has_unsettled_submissions()
            && self.recoverable_submissions().next().is_none()
            && self.permission_requests.is_empty()
            && !self.entries.iter().any(|entry| match entry {
                AgentThreadEntry::Elicitation(id) => {
                    self.elicitations
                        .elicitation(id)
                        .is_some_and(|(_, elicitation)| {
                            matches!(elicitation.status, ElicitationStatus::Pending { .. })
                        })
                }
                _ => false,
            })
    }

    pub fn forget_submission(&mut self, id: SubmissionId, cx: &mut Context<Self>) {
        if self.submissions.forget(id) {
            cx.emit(AcpThreadEvent::SubmissionUpdated(id));
            cx.notify();
        }
    }

    /// Ends receipt and cancellation observations without declaring remote execution stopped.
    pub fn handle_connection_lost(&mut self, cx: &mut Context<Self>) {
        let was_observable = self.activity.ensure_observable().is_ok();
        self.activity.close_observation();
        let changed = self.submissions.disconnect();
        if !changed.is_empty() {
            for id in changed {
                cx.emit(AcpThreadEvent::SubmissionUpdated(id));
            }
            cx.notify();
        }
        if was_observable {
            cx.emit(AcpThreadEvent::ConnectionLost);
            cx.notify();
        }
    }

    pub fn prepare_session_close(&mut self, cx: &mut Context<Self>) {
        self.cancel_generic_permission_requests(cx);
        let pending_receipt_ids = self.submissions.pending_receipt_ids().collect::<Vec<_>>();
        if pending_receipt_ids.is_empty() {
            return;
        }
        // Prompt receipts may follow successful close. Pin only those already pending:
        // a rejected close must not extend this retention to later submissions.
        let (sender, received) = oneshot::channel();
        let mut sender = Some(sender);
        let subscription = cx.subscribe_self(move |this, event, _cx| {
            if matches!(event, AcpThreadEvent::SubmissionUpdated(_))
                && !pending_receipt_ids.iter().any(|id| {
                    this.submissions
                        .get(*id)
                        .is_some_and(|record| matches!(record.state, SubmissionState::Pending))
                })
                && let Some(sender) = sender.take()
                && sender.send(()).is_err()
            {
                log::debug!("Receipt retention observer was dropped");
            }
        });
        let thread = cx.entity();
        cx.spawn(async move |_, _| {
            let _subscription = subscription;
            let _thread = thread;
            if let Err(error) = received.await {
                log::debug!("Receipt retention ended before settlement: {error}");
            }
        })
        .detach();
    }

    pub fn handle_session_closed(&mut self, cx: &mut Context<Self>) {
        // Close confirms stopping without inventing an Idle snapshot or ending receipts.
        self.activity.finish_close();
        self.cancel_generic_permission_requests(cx);
        self.cancel_outstanding_elicitations(cx);
        cx.notify();
    }

    fn register_submission(
        &mut self,
        content: Arc<[acp_v2::ContentBlock]>,
        cx: &mut Context<Self>,
    ) -> SubmissionId {
        let id = self.submissions.register(content);
        cx.emit(AcpThreadEvent::SubmissionUpdated(id));
        cx.notify();
        id
    }

    fn observe_submission_echo(&mut self, message_id: &acp_v2::MessageId, cx: &mut Context<Self>) {
        for id in self.submissions.observe_echo(message_id) {
            cx.emit(AcpThreadEvent::SubmissionUpdated(id));
        }
        cx.notify();
    }

    fn track_submission(
        &mut self,
        id: SubmissionId,
        applied: Option<oneshot::Sender<bool>>,
        cx: &mut Context<Self>,
        complete: impl 'static
        + AsyncFnOnce(
            WeakEntity<Self>,
            &mut AsyncApp,
        ) -> Result<Option<SubmissionResponse>>,
    ) -> Submission {
        let (sender, receiver) = oneshot::channel();
        let (disconnect_sender, disconnect_receiver) = self
            .submissions
            .receipt_transport()
            .is_some()
            .then(oneshot::channel::<()>)
            .unzip();
        let task = cx.spawn(async move |this, cx| {
            let response = {
                let response = complete(this.clone(), cx);
                futures::pin_mut!(response);
                if let Some(disconnected) = disconnect_receiver {
                    match futures::future::select(response, disconnected).await {
                        futures::future::Either::Left((response, _)) => response,
                        futures::future::Either::Right(_) => Err(anyhow!(SubmissionDisconnected)),
                    }
                } else {
                    response.await
                }
            };
            let response = match response {
                Err(error) if error.is::<SubmissionCancelled>() => Ok(None),
                response => response,
            };
            let update = this.update(cx, |this, cx| {
                if this.submissions.receipt_transport().is_some()
                    && response.as_ref().err().is_some_and(|error| {
                        error.is::<SubmissionDisconnected>()
                            || error
                                .downcast_ref::<agent_client_protocol::Error>()
                                .is_some_and(|error| {
                                    agent_client_protocol::is_incoming_transport_closed(error)
                                        || agent_client_protocol::is_connection_terminated(error)
                                })
                    })
                {
                    this.handle_connection_lost(cx);
                }
                let state = match &response {
                    Ok(Some(SubmissionResponse::Accepted(receipt))) => {
                        let echoed = this.entries.iter().any(|entry| {
                            matches!(entry, AgentThreadEntry::UserMessage(message)
                                if matches!(&message.identity, MessageIdentity::Keyed(id) if id == &receipt.message_id))
                        });
                        SubmissionState::Accepted { receipt: receipt.clone(), echoed }
                    }
                    Ok(Some(SubmissionResponse::LegacyCompleted(_))) => SubmissionState::Completed,
                    Ok(None) => SubmissionState::Cancelled,
                    Err(error) => SubmissionState::Failed(format!("{error:#}").into()),
                };
                let changed = this.submissions.settle(id, state);
                if let Some(applied) = applied
                    && applied.send(response.is_ok()).is_err()
                {
                    log::debug!("Receipt transport ended before application acknowledgement");
                }
                // The thread owns settlement even when the response observer has gone away.
                if let Err(Err(error)) = sender.send(response) {
                    log::debug!("Submission observer was dropped: {error:#}");
                }
                if changed {
                    cx.emit(AcpThreadEvent::SubmissionUpdated(id));
                    cx.notify();
                }
            });
            if let Err(error) = update {
                log::debug!("Submission thread was released: {error:#}");
            }
        });
        self.submissions.track(id, task, disconnect_sender);
        Submission::new(
            id,
            async move {
                receiver
                    .await
                    .context("Submission ended before its response was delivered")?
            }
            .boxed(),
        )
    }

    fn set_foreground_state(&mut self, state: acp_v2::StateUpdate, cx: &mut Context<Self>) {
        let previous_generation = self.activity.generation();
        self.activity
            .update(state, self.entries.len().saturating_sub(1));
        if self.activity.generation() != previous_generation {
            self.had_error = false;
            self.local_execution_error = None;
        }
        cx.notify();
    }

    pub fn sync_local_action_state(&mut self, cx: &mut Context<Self>) {
        if !self.foreground_activity().is_active() {
            return;
        }
        let desired = if self.has_pending_turn_action() {
            ForegroundActivity::RequiresAction
        } else {
            ForegroundActivity::Running
        };
        if desired != self.foreground_activity() {
            self.set_foreground_state(
                if desired == ForegroundActivity::RequiresAction {
                    acp_v2::StateUpdate::RequiresAction(acp_v2::RequiresActionStateUpdate::new())
                } else {
                    acp_v2::StateUpdate::Running(acp_v2::RunningStateUpdate::new())
                },
                cx,
            );
            cx.emit(AcpThreadEvent::StatusChanged);
        }
    }

    pub fn update_session_state(
        &mut self,
        state: acp_v2::StateUpdate,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        anyhow::ensure!(
            self.uses_reported_activity(),
            "Session does not report foreground state"
        );
        let previous = self.foreground_activity();
        let was_in_progress = self.activity.in_progress();
        self.received_session_state = true;
        if self.subagent_association.is_some() && matches!(state, acp_v2::StateUpdate::Idle(_)) {
            let entry_count = self.entries.len();
            // An idle report with no entries since the previous one restates that outcome.
            match self.subagent_outcomes.last_mut() {
                Some((last_entry_count, last_state)) if *last_entry_count == entry_count => {
                    *last_state = state.clone();
                }
                _ => self.subagent_outcomes.push((entry_count, state.clone())),
            }
        }
        self.set_foreground_state(state, cx);
        if let Some(acp_v2::StateUpdate::Idle(idle)) = self.activity.state() {
            let stop_reason = idle.stop_reason.clone();
            if let Some(usage) = &idle.usage {
                let reported = ReportedTokenUsage {
                    activity_generation: self.activity.generation(),
                    input_tokens: usage.input_tokens,
                    output_tokens: usage.output_tokens,
                };
                let changed = self.reported_usage != Some(reported);
                let stored = self.token_usage.get_or_insert_with(Default::default);
                stored.input_tokens = usage.input_tokens;
                stored.output_tokens = usage.output_tokens;
                self.reported_usage = Some(reported);
                if changed {
                    cx.emit(AcpThreadEvent::TokenUsageUpdated(
                        OutputTokensUpdate::Snapshot {
                            activity_generation: reported.activity_generation,
                            output_tokens: usage.output_tokens,
                        },
                    ));
                }
            }
            if was_in_progress {
                self.flush_streaming_text(cx);
                self.shrink_message_source_capacity(self.activity.first_entry_index());
                if self.parent_session_id.is_none() {
                    self.project
                        .update(cx, |project, cx| project.set_agent_location(None, cx));
                }
                self.had_error = matches!(
                    stop_reason,
                    Some(
                        acp_v2::StopReason::MaxTokens
                            | acp_v2::StopReason::Refusal
                            | acp_v2::StopReason::Error(_)
                    )
                );
                // Reported refusal does not authorize deleting agent-owned history.
                cx.emit(AcpThreadEvent::Stopped {
                    activity_generation: self.activity.generation(),
                    activity_duration: self.activity.duration(),
                    stop_reason: stop_reason.clone(),
                });
            }
            // A subagent's work can stop because Zed cancelled an ancestor. Its requests stay
            // answerable until it reports that, and then go as if Zed had cancelled it.
            if self.subagent_association.is_some()
                && matches!(stop_reason, Some(acp_v2::StopReason::Cancelled))
            {
                self.cancel_local_actions(RequestPermissionOutcome::Cancelled, cx);
            }
        }
        if previous != self.foreground_activity() {
            cx.emit(AcpThreadEvent::StatusChanged);
        }
        if !was_in_progress
            && matches!(
                previous,
                ForegroundActivity::Unconfirmed | ForegroundActivity::Unknown
            )
            && self.foreground_activity() == ForegroundActivity::Idle
        {
            cx.emit(AcpThreadEvent::ActivityConfirmedIdle);
        }
        Ok(())
    }

    /// Forgets the reported foreground state without implying idle, as when an ACP
    /// subagent association clears its `state`. Work in progress stays in progress.
    pub fn clear_session_state(&mut self, cx: &mut Context<Self>) -> Result<()> {
        anyhow::ensure!(
            self.uses_reported_activity(),
            "Session does not report foreground state"
        );
        let previous = self.foreground_activity();
        self.activity.mark_unconfirmed();
        if previous != self.foreground_activity() {
            cx.emit(AcpThreadEvent::StatusChanged);
        }
        Ok(())
    }

    pub fn cost(&self) -> Option<&SessionCost> {
        self.cost.as_ref()
    }

    pub fn has_pending_edit_tool_calls(&self) -> bool {
        for entry in self.entries.iter().rev() {
            match entry {
                AgentThreadEntry::UserMessage(_) => return false,
                AgentThreadEntry::ToolCall(call)
                    if matches!(
                        call.status(),
                        ToolCallStatus::InProgress | ToolCallStatus::Pending
                    ) && call.diffs().next().is_some() =>
                {
                    return true;
                }
                AgentThreadEntry::ToolCall(_)
                | AgentThreadEntry::Elicitation(_)
                | AgentThreadEntry::AssistantMessage(_)
                | AgentThreadEntry::SessionMessage(_)
                | AgentThreadEntry::ContextCompaction(_) => {}
            }
        }

        false
    }

    pub fn has_in_progress_tool_calls(&self) -> bool {
        for entry in self.entries.iter().rev() {
            match entry {
                AgentThreadEntry::UserMessage(_) => return false,
                AgentThreadEntry::ToolCall(call)
                    if matches!(
                        call.status(),
                        ToolCallStatus::InProgress | ToolCallStatus::Pending
                    ) =>
                {
                    return true;
                }
                AgentThreadEntry::ToolCall(_)
                | AgentThreadEntry::Elicitation(_)
                | AgentThreadEntry::AssistantMessage(_)
                | AgentThreadEntry::SessionMessage(_)
                | AgentThreadEntry::ContextCompaction(_) => {}
            }
        }

        false
    }

    pub fn used_tools_since_last_user_message(&self) -> bool {
        for entry in self.entries.iter().rev() {
            match entry {
                AgentThreadEntry::UserMessage(..) => return false,
                AgentThreadEntry::AssistantMessage(..)
                | AgentThreadEntry::SessionMessage(_)
                | AgentThreadEntry::ContextCompaction(_)
                | AgentThreadEntry::Elicitation(_) => continue,
                AgentThreadEntry::ToolCall(..) => return true,
            }
        }

        false
    }

    pub fn handle_session_update(
        &mut self,
        update: acp_v1::SessionUpdate,
        cx: &mut Context<Self>,
    ) -> Result<(), acp_v1::Error> {
        match update {
            acp_v1::SessionUpdate::UserMessageChunk(acp_v1::ContentChunk {
                content,
                message_id,
                ..
            }) => {
                let content = content::from_v1(content).map_err(acp_v1::Error::from)?;
                // We optimistically add the full user prompt before calling `prompt`.
                // Some ACP servers echo user chunks back over updates. Skip echoed
                // chunks only when they match the local optimistic message.
                let already_in_user_message = self
                    .entries
                    .last_mut()
                    .and_then(|entry| match entry {
                        AgentThreadEntry::UserMessage(message) => Some(message),
                        _ => None,
                    })
                    .is_some_and(|message| {
                        let MessageIdentity::Legacy(protocol_id) = &mut message.identity else {
                            return false;
                        };
                        let already_in_user_message = message.is_optimistic
                            && message.content.source_blocks().contains(&content)
                            && can_merge_message_chunks(protocol_id.as_ref(), message_id.as_ref());
                        if already_in_user_message && protocol_id.is_none() {
                            *protocol_id = message_id.clone();
                        }
                        already_in_user_message
                    });
                if !already_in_user_message {
                    self.push_user_content_block_from_agent(message_id, content, cx);
                }
            }
            acp_v1::SessionUpdate::AgentMessageChunk(acp_v1::ContentChunk {
                content,
                message_id,
                ..
            }) => {
                self.push_assistant_content_block_with_message_id(
                    message_id,
                    content::from_v1(content).map_err(acp_v1::Error::from)?,
                    false,
                    false,
                    cx,
                );
            }
            acp_v1::SessionUpdate::AgentThoughtChunk(acp_v1::ContentChunk {
                content,
                message_id,
                ..
            }) => {
                self.push_assistant_content_block_with_message_id(
                    message_id,
                    content::from_v1(content).map_err(acp_v1::Error::from)?,
                    true,
                    false,
                    cx,
                );
            }
            acp_v1::SessionUpdate::ToolCall(tool_call) => {
                self.upsert_tool_call(tool_call, cx)?;
            }
            acp_v1::SessionUpdate::ToolCallUpdate(tool_call_update) => {
                self.update_tool_call(tool_call_update, cx)?;
            }
            acp_v1::SessionUpdate::CompactionUpdate(compaction_update) => {
                let mut update =
                    compaction::update_from_v1(compaction_update).map_err(acp_v1::Error::from)?;
                let metadata = MetadataUpdate::V1(mem::take(&mut update.meta));
                self.upsert_context_compaction_update_inner(update, metadata, cx);
            }
            acp_v1::SessionUpdate::CompactionSummaryChunk(summary_chunk) => {
                self.append_context_compaction_summary(
                    compaction::chunk_from_v1(summary_chunk).map_err(acp_v1::Error::from)?,
                    cx,
                );
            }
            acp_v1::SessionUpdate::Plan(plan) => {
                self.update_plan(plan, cx).map_err(acp_v1::Error::from)?;
            }
            acp_v1::SessionUpdate::Notice(notice) => {
                self.push_notice(notices::from_v1(notice).map_err(acp_v1::Error::from)?, cx);
            }
            acp_v1::SessionUpdate::SessionInfoUpdate(info_update) => {
                self.update_session_info(info_update, cx);
            }
            acp_v1::SessionUpdate::AvailableCommandsUpdate(acp_v1::AvailableCommandsUpdate {
                available_commands,
                ..
            }) => {
                self.update_available_commands(
                    commands::from_v1(available_commands).map_err(acp_v1::Error::from)?,
                    cx,
                );
            }
            acp_v1::SessionUpdate::CurrentModeUpdate(acp_v1::CurrentModeUpdate {
                current_mode_id,
                ..
            }) => cx.emit(AcpThreadEvent::ModeUpdated(
                acp_v2::SessionConfigValueId::new(current_mode_id.0),
            )),
            acp_v1::SessionUpdate::ConfigOptionUpdate(acp_v1::ConfigOptionUpdate {
                config_options,
                ..
            }) => cx.emit(AcpThreadEvent::ConfigOptionsUpdated(
                config_options::from_v1(config_options).map_err(acp_v1::Error::from)?,
            )),
            acp_v1::SessionUpdate::UsageUpdate(update) => {
                self.update_usage(
                    acp_v2::UsageUpdate::new(update.used, update.size)
                        // v1 has no way to clear the cost, so a missing cost leaves it unchanged.
                        .cost(update.cost.map_or(MaybeUndefined::Undefined, |cost| {
                            MaybeUndefined::Value(
                                acp_v2::Cost::new(cost.amount, cost.currency).meta(cost.meta),
                            )
                        }))
                        .meta(update.meta),
                    cx,
                );
            }
            acp_v1::SessionUpdate::SessionMessage(message) => {
                let content = match message.content {
                    MaybeUndefined::Undefined => MaybeUndefined::Undefined,
                    MaybeUndefined::Null => MaybeUndefined::Null,
                    MaybeUndefined::Value(blocks) => MaybeUndefined::Value(
                        blocks
                            .into_iter()
                            .map(content::from_v1)
                            .collect::<Result<Vec<_>>>()
                            .map_err(acp_v1::Error::from)?,
                    ),
                };
                self.upsert_session_message_inner(
                    acp_v2::SessionMessage::new(message.message_id.0)
                        .sender_session_id(
                            message
                                .sender_session_id
                                .map_value(|id| acp_v2::SessionId::new(id.0)),
                        )
                        .recipient_session_id(
                            message
                                .recipient_session_id
                                .map_value(|id| acp_v2::SessionId::new(id.0)),
                        )
                        .sender_message_id(
                            message
                                .sender_message_id
                                .map_value(|id| acp_v2::MessageId::new(id.0)),
                        )
                        .recipient_message_id(
                            message
                                .recipient_message_id
                                .map_value(|id| acp_v2::MessageId::new(id.0)),
                        )
                        .content(content)
                        .meta(message.meta),
                    MetadataUpdate::V1,
                    cx,
                )
                .map_err(acp_v1::Error::from)?;
            }
            acp_v1::SessionUpdate::SessionMessageChunk(chunk) => {
                self.append_session_message_chunk(
                    acp_v2::SessionMessageChunk::new(
                        chunk.message_id.0,
                        content::from_v1(chunk.content).map_err(acp_v1::Error::from)?,
                    ),
                    cx,
                )
                .map_err(acp_v1::Error::from)?;
            }
            _ => {}
        }
        Ok(())
    }

    pub fn upsert_user_message(
        &mut self,
        update: acp_v2::UserMessage,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        self.update_keyed_message(
            MessageKind::User,
            update.message_id,
            update.content,
            MetadataUpdate::V2(update.meta),
            cx,
        )
    }

    pub fn upsert_assistant_message(
        &mut self,
        update: acp_v2::AgentMessage,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        self.update_keyed_message(
            MessageKind::Assistant,
            update.message_id,
            update.content,
            MetadataUpdate::V2(update.meta),
            cx,
        )
    }

    pub fn upsert_thought(
        &mut self,
        update: acp_v2::AgentThought,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        self.update_keyed_message(
            MessageKind::Thought,
            update.message_id,
            update.content,
            MetadataUpdate::V2(update.meta),
            cx,
        )
    }

    pub fn upsert_session_message(
        &mut self,
        update: acp_v2::SessionMessage,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        self.upsert_session_message_inner(update, MetadataUpdate::V2, cx)
    }

    fn upsert_session_message_inner(
        &mut self,
        update: acp_v2::SessionMessage,
        metadata: fn(MaybeUndefined<acp_v2::Meta>) -> MetadataUpdate,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        let previous_entry_count = self.entries.len();
        let location = self.keyed_message_location(MessageKind::Session, update.message_id, cx)?;
        let Some(AgentThreadEntry::SessionMessage(message)) =
            self.entries.get_mut(location.entry_index())
        else {
            anyhow::bail!("session message disappeared during update");
        };
        // Omitted keeps the stored value, `null` clears it, and a value replaces it.
        update
            .sender_session_id
            .update_to(&mut message.sender_session_id);
        update
            .recipient_session_id
            .update_to(&mut message.recipient_session_id);
        update
            .sender_message_id
            .update_to(&mut message.sender_message_id);
        update
            .recipient_message_id
            .update_to(&mut message.recipient_message_id);
        self.replace_message_content(
            location,
            MessageKind::Session,
            update.content,
            metadata(update.meta),
            previous_entry_count,
            cx,
        )
    }

    /// The index of this transcript's view of `message`, a session message whose other view
    /// is in the transcript of `message_session_id`. The views are linked by `message`'s
    /// reference to the view here or, when it has none, by a view here that refers back to
    /// it. Views whose references name other messages are unlinked, and a referenced view
    /// that has not arrived yet is not found.
    pub fn linked_session_message(
        &self,
        message: &SessionMessage,
        message_session_id: &acp_v2::SessionId,
    ) -> Option<usize> {
        let MessageIdentity::Keyed(message_id) = &message.identity else {
            return None;
        };
        let own_session_id = self.session_id();
        let (reference, other_participant, message_is_outgoing) =
            match message.side(message_session_id) {
                SessionMessageSide::Outgoing { recipient } => {
                    (message.recipient_message_id.as_ref(), recipient, true)
                }
                SessionMessageSide::Incoming { sender } => {
                    (message.sender_message_id.as_ref(), sender, false)
                }
                SessionMessageSide::Unknown => return None,
            };
        if other_participant.is_some_and(|participant| participant != own_session_id) {
            return None;
        }
        /// A view of the same message here is on the other side. Returns that view's
        /// participant in `message_session_id`'s place, when known, and its reference back.
        fn other_view<'a>(
            view: &'a SessionMessage,
            own_session_id: &acp_v2::SessionId,
            message_is_outgoing: bool,
        ) -> Option<(Option<&'a acp_v2::SessionId>, Option<&'a acp_v2::MessageId>)> {
            match view.side(own_session_id) {
                SessionMessageSide::Incoming { sender } if message_is_outgoing => {
                    Some((sender, view.sender_message_id.as_ref()))
                }
                SessionMessageSide::Outgoing { recipient } if !message_is_outgoing => {
                    Some((recipient, view.recipient_message_id.as_ref()))
                }
                SessionMessageSide::Unknown if message_is_outgoing => {
                    Some((None, view.sender_message_id.as_ref()))
                }
                SessionMessageSide::Unknown => Some((None, view.recipient_message_id.as_ref())),
                _ => None,
            }
        }
        let mut session_messages =
            self.entries
                .iter()
                .enumerate()
                .filter_map(|(entry_ix, entry)| match entry {
                    AgentThreadEntry::SessionMessage(view) => Some((entry_ix, view)),
                    _ => None,
                });
        if let Some(reference) = reference {
            let (entry_ix, view) = session_messages.find(
                |(_, view)| matches!(&view.identity, MessageIdentity::Keyed(id) if id == reference),
            )?;
            let (participant, back_reference) =
                other_view(view, own_session_id, message_is_outgoing)?;
            (participant.is_none_or(|participant| participant == message_session_id)
                && back_reference.is_none_or(|back_reference| back_reference == message_id))
            .then_some(entry_ix)
        } else {
            // The view's reference is a message ID in `message_session_id`'s transcript, so
            // it only identifies `message` when that participant is known.
            session_messages.find_map(|(entry_ix, view)| {
                let (participant, back_reference) =
                    other_view(view, own_session_id, message_is_outgoing)?;
                (participant == Some(message_session_id) && back_reference == Some(message_id))
                    .then_some(entry_ix)
            })
        }
    }

    pub fn append_session_message_chunk(
        &mut self,
        chunk: acp_v2::SessionMessageChunk,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        // A chunk carries only content: its metadata does not patch the message's.
        let previous_entry_count = self.entries.len();
        let location = self.keyed_message_location(MessageKind::Session, chunk.message_id, cx)?;
        self.append_message_content(
            location,
            MessageKind::Session,
            chunk.content,
            previous_entry_count,
            cx,
        )
    }

    fn update_keyed_message(
        &mut self,
        kind: MessageKind,
        id: acp_v2::MessageId,
        content: MaybeUndefined<Vec<acp_v2::ContentBlock>>,
        meta: MetadataUpdate,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        let previous_entry_count = self.entries.len();
        let location = self.keyed_message_location(kind, id, cx)?;
        self.replace_message_content(location, kind, content, meta, previous_entry_count, cx)
    }

    fn replace_message_content(
        &mut self,
        location: MessageLocation,
        kind: MessageKind,
        content: MaybeUndefined<Vec<acp_v2::ContentBlock>>,
        meta: MetadataUpdate,
        previous_entry_count: usize,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        if !content.is_undefined()
            && self
                .streaming_text_buffer
                .as_ref()
                .is_some_and(|buffer| location.is_streaming_target(&buffer.target))
        {
            // Pending text belongs to the superseded snapshot, even when the
            // replacement reuses its Markdown entity.
            self.streaming_text_buffer.take();
        }

        let language_registry = self.project.read(cx).languages().clone();
        let path_style = self.project.read(cx).path_style(cx);
        let (message_content, message_meta) = location
            .fields_mut(&mut self.entries)
            .context("message disappeared during update")?;
        let replacement = match content {
            MaybeUndefined::Undefined => None,
            MaybeUndefined::Null => Some(Vec::new()),
            MaybeUndefined::Value(content) => Some(content),
        };
        if let Some(replacement) = replacement {
            if kind == MessageKind::User {
                message_content.replace_prompt(replacement, &language_registry, path_style, cx);
            } else {
                message_content.replace(replacement, &language_registry, path_style, cx);
            }
        }
        meta.apply(message_meta);
        self.emit_message_update(location, previous_entry_count, cx);
        Ok(())
    }

    pub fn append_message_chunk(
        &mut self,
        kind: MessageKind,
        chunk: acp_v2::ContentChunk,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        // Chunk-envelope metadata does not patch the message's metadata.
        let acp_v2::ContentChunk {
            message_id,
            content,
            ..
        } = chunk;
        let previous_entry_count = self.entries.len();
        let location = self.keyed_message_location(kind, message_id, cx)?;
        self.append_message_content(location, kind, content, previous_entry_count, cx)
    }

    fn append_message_content(
        &mut self,
        location: MessageLocation,
        kind: MessageKind,
        content: acp_v2::ContentBlock,
        previous_entry_count: usize,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        let content = match content {
            acp_v2::ContentBlock::Text(text) => {
                let (content, _) = location
                    .fields_mut(&mut self.entries)
                    .context("message disappeared during append")?;
                if let MessageLocation::Assistant {
                    entry_index,
                    chunk_index,
                } = location
                    && let Some(markdown) = content.trailing_text().cloned()
                {
                    let text_len = text.text.len();
                    let source_index = content.append_deferred_text(text);
                    self.buffer_streaming_text(
                        StreamingTextTarget {
                            entry_index,
                            chunk_index,
                            markdown,
                        },
                        source_index,
                        text_len,
                        cx,
                    );
                    self.emit_message_update(location, previous_entry_count, cx);
                    return Ok(());
                }
                acp_v2::ContentBlock::Text(text)
            }
            content => content,
        };

        if self
            .streaming_text_buffer
            .as_ref()
            .is_some_and(|buffer| location.is_streaming_target(&buffer.target))
        {
            self.flush_streaming_text(cx);
        }
        let language_registry = self.project.read(cx).languages().clone();
        let path_style = self.project.read(cx).path_style(cx);
        let (message_content, _) = location
            .fields_mut(&mut self.entries)
            .context("message disappeared during append")?;
        if kind == MessageKind::User {
            message_content.append_prompt(content, &language_registry, path_style, cx);
        } else {
            message_content.append(content, &language_registry, path_style, cx);
        }
        self.emit_message_update(location, previous_entry_count, cx);
        Ok(())
    }

    fn keyed_message_location(
        &mut self,
        kind: MessageKind,
        id: acp_v2::MessageId,
        cx: &mut Context<Self>,
    ) -> Result<MessageLocation> {
        // Scan the authoritative records so rewind and refusal need no secondary
        // index maintenance. Most chunks target the last record.
        let identity = MessageIdentity::Keyed(id);
        for (entry_index, entry) in self.entries.iter().enumerate().rev() {
            let found = match entry {
                AgentThreadEntry::UserMessage(message)
                    if kind == MessageKind::User && message.identity == identity =>
                {
                    Some(MessageLocation::User { entry_index })
                }
                AgentThreadEntry::SessionMessage(message)
                    if kind == MessageKind::Session && message.identity == identity =>
                {
                    Some(MessageLocation::Session { entry_index })
                }
                AgentThreadEntry::AssistantMessage(message) => message
                    .chunks
                    .iter()
                    .enumerate()
                    .rev()
                    .find(|(_, chunk)| chunk.kind() == kind && chunk.identity() == &identity)
                    .map(|(chunk_index, _)| MessageLocation::Assistant {
                        entry_index,
                        chunk_index,
                    }),
                _ => None,
            };
            if let Some(location) = found {
                return Ok(location);
            }
        }

        self.flush_streaming_text(cx);
        let entry_index = self.entries.len();
        match kind {
            MessageKind::User => {
                self.entries
                    .push(AgentThreadEntry::UserMessage(UserMessage {
                        identity,
                        meta: None,
                        client_id: None,
                        is_optimistic: false,
                        content: MessageContent::default(),
                        indented: false,
                    }));
                Ok(MessageLocation::User { entry_index })
            }
            MessageKind::Session => {
                self.entries
                    .push(AgentThreadEntry::SessionMessage(SessionMessage {
                        identity,
                        sender_session_id: None,
                        recipient_session_id: None,
                        sender_message_id: None,
                        recipient_message_id: None,
                        meta: None,
                        content: MessageContent::default(),
                    }));
                Ok(MessageLocation::Session { entry_index })
            }
            MessageKind::Assistant | MessageKind::Thought => {
                let chunk = if kind == MessageKind::Thought {
                    AssistantMessageChunk::Thought {
                        identity,
                        meta: None,
                        block: MessageContent::default(),
                    }
                } else {
                    AssistantMessageChunk::Message {
                        identity,
                        meta: None,
                        block: MessageContent::default(),
                    }
                };
                if let Some(AgentThreadEntry::AssistantMessage(message)) = self.entries.last_mut()
                    && !message.indented
                    && !message.is_subagent_output
                {
                    let chunk_index = message.chunks.len();
                    message.chunks.push(chunk);
                    Ok(MessageLocation::Assistant {
                        entry_index: entry_index - 1,
                        chunk_index,
                    })
                } else {
                    self.entries
                        .push(AgentThreadEntry::AssistantMessage(AssistantMessage {
                            chunks: vec![chunk],
                            indented: false,
                            is_subagent_output: false,
                        }));
                    Ok(MessageLocation::Assistant {
                        entry_index,
                        chunk_index: 0,
                    })
                }
            }
        }
    }

    fn emit_message_update(
        &mut self,
        location: MessageLocation,
        previous_entry_count: usize,
        cx: &mut Context<Self>,
    ) {
        if self.submissions.receipt_transport().is_some()
            && let Some(AgentThreadEntry::UserMessage(message)) =
                self.entries.get(location.entry_index())
            && let MessageIdentity::Keyed(id) = &message.identity
        {
            let id = id.clone();
            self.observe_submission_echo(&id, cx);
        }
        if self.entries.len() > previous_entry_count {
            cx.emit(AcpThreadEvent::NewEntry);
        } else {
            cx.emit(AcpThreadEvent::EntryUpdated(location.entry_index()));
        }
    }

    pub fn push_user_content_block(
        &mut self,
        client_id: Option<ClientUserMessageId>,
        chunk: acp_v2::ContentBlock,
        cx: &mut Context<Self>,
    ) {
        self.push_user_content_block_with_indent(client_id, chunk, false, cx)
    }

    pub fn push_user_content_block_with_indent(
        &mut self,
        client_id: Option<ClientUserMessageId>,
        chunk: acp_v2::ContentBlock,
        indented: bool,
        cx: &mut Context<Self>,
    ) {
        self.push_user_content_block_with_protocol_id(
            client_id.clone(),
            client_id.is_some(),
            None,
            chunk,
            indented,
            cx,
        )
    }

    fn push_user_content_block_from_agent(
        &mut self,
        id: Option<acp_v1::MessageId>,
        chunk: acp_v2::ContentBlock,
        cx: &mut Context<Self>,
    ) {
        self.push_user_content_block_with_protocol_id(None, false, id, chunk, false, cx)
    }

    fn push_user_content_block_with_protocol_id(
        &mut self,
        incoming_client_id: Option<ClientUserMessageId>,
        is_optimistic: bool,
        protocol_id: Option<acp_v1::MessageId>,
        chunk: acp_v2::ContentBlock,
        indented: bool,
        cx: &mut Context<Self>,
    ) {
        let language_registry = self.project.read(cx).languages().clone();
        let path_style = self.project.read(cx).path_style(cx);
        let entries_len = self.entries.len();
        self.flush_streaming_text(cx);

        if let Some(last_entry) = self.entries.last_mut()
            && let AgentThreadEntry::UserMessage(UserMessage {
                identity: MessageIdentity::Legacy(existing_protocol_id),
                client_id: existing_client_id,
                content,
                is_optimistic: existing_is_optimistic,
                indented: existing_indented,
                ..
            }) = last_entry
            && *existing_indented == indented
            && can_merge_message_chunks(existing_protocol_id.as_ref(), protocol_id.as_ref())
            && !(*existing_is_optimistic
                && !is_optimistic
                && existing_protocol_id.is_none()
                && protocol_id.is_some())
        {
            if let Some(incoming_client_id) = incoming_client_id {
                *existing_client_id = Some(incoming_client_id);
            }
            *existing_is_optimistic |= is_optimistic;
            if existing_protocol_id.is_none() {
                *existing_protocol_id = protocol_id;
            }
            content.append_prompt(chunk, &language_registry, path_style, cx);
            let idx = entries_len - 1;
            cx.emit(AcpThreadEvent::EntryUpdated(idx));
        } else {
            let mut content = MessageContent::default();
            content.append_prompt(chunk, &language_registry, path_style, cx);
            self.push_entry(
                AgentThreadEntry::UserMessage(UserMessage {
                    identity: MessageIdentity::Legacy(protocol_id),
                    meta: None,
                    client_id: incoming_client_id,
                    is_optimistic,
                    content,
                    indented,
                }),
                cx,
            );
        }
    }

    pub fn push_assistant_content_block(
        &mut self,
        chunk: acp_v2::ContentBlock,
        is_thought: bool,
        cx: &mut Context<Self>,
    ) {
        self.push_assistant_content_block_with_indent(chunk, is_thought, false, cx)
    }

    pub fn push_assistant_content_block_with_indent(
        &mut self,
        chunk: acp_v2::ContentBlock,
        is_thought: bool,
        indented: bool,
        cx: &mut Context<Self>,
    ) {
        self.push_assistant_content_block_with_message_id(None, chunk, is_thought, indented, cx)
    }

    fn push_assistant_content_block_with_message_id(
        &mut self,
        message_id: Option<acp_v1::MessageId>,
        chunk: acp_v2::ContentBlock,
        is_thought: bool,
        indented: bool,
        cx: &mut Context<Self>,
    ) {
        let path_style = self.project.read(cx).path_style(cx);

        // For text chunks going to an existing Markdown block, buffer for smooth
        // streaming instead of appending all at once which may feel more choppy.
        let chunk = match chunk {
            acp_v2::ContentBlock::Text(text) => {
                if let Some((content, target)) =
                    self.streaming_content_target(message_id.as_ref(), is_thought, indented)
                {
                    let text_len = text.text.len();
                    let source_index = content.append_deferred_text(text);
                    cx.emit(AcpThreadEvent::EntryUpdated(target.entry_index));
                    self.buffer_streaming_text(target, source_index, text_len, cx);
                    return;
                }
                acp_v2::ContentBlock::Text(text)
            }
            chunk => chunk,
        };

        let language_registry = self.project.read(cx).languages().clone();
        let entries_len = self.entries.len();
        self.flush_streaming_text(cx);
        if let Some(last_entry) = self.entries.last_mut()
            && let AgentThreadEntry::AssistantMessage(AssistantMessage {
                chunks,
                indented: existing_indented,
                is_subagent_output: _,
            }) = last_entry
            && *existing_indented == indented
        {
            let idx = entries_len - 1;
            cx.emit(AcpThreadEvent::EntryUpdated(idx));
            match (chunks.last_mut(), is_thought) {
                (
                    Some(AssistantMessageChunk::Message {
                        identity: MessageIdentity::Legacy(existing_id),
                        block,
                        ..
                    }),
                    false,
                )
                | (
                    Some(AssistantMessageChunk::Thought {
                        identity: MessageIdentity::Legacy(existing_id),
                        block,
                        ..
                    }),
                    true,
                ) if can_merge_message_chunks(existing_id.as_ref(), message_id.as_ref()) => {
                    if existing_id.is_none() {
                        *existing_id = message_id;
                    }
                    block.append(chunk, &language_registry, path_style, cx)
                }
                _ => {
                    let block = MessageContent::new(chunk, &language_registry, path_style, cx);
                    if is_thought {
                        chunks.push(AssistantMessageChunk::Thought {
                            identity: MessageIdentity::Legacy(message_id),
                            meta: None,
                            block,
                        })
                    } else {
                        chunks.push(AssistantMessageChunk::Message {
                            identity: MessageIdentity::Legacy(message_id),
                            meta: None,
                            block,
                        })
                    }
                }
            }
        } else {
            let block = MessageContent::new(chunk, &language_registry, path_style, cx);
            let chunk = if is_thought {
                AssistantMessageChunk::Thought {
                    identity: MessageIdentity::Legacy(message_id),
                    meta: None,
                    block,
                }
            } else {
                AssistantMessageChunk::Message {
                    identity: MessageIdentity::Legacy(message_id),
                    meta: None,
                    block,
                }
            };

            self.push_entry(
                AgentThreadEntry::AssistantMessage(AssistantMessage {
                    chunks: vec![chunk],
                    indented,
                    is_subagent_output: false,
                }),
                cx,
            );
        }
    }

    fn streaming_content_target(
        &mut self,
        message_id: Option<&acp_v1::MessageId>,
        is_thought: bool,
        indented: bool,
    ) -> Option<(&mut MessageContent, StreamingTextTarget)> {
        let (entry_index, last_entry) = self.entries.iter_mut().enumerate().next_back()?;
        if let AgentThreadEntry::AssistantMessage(AssistantMessage {
            chunks,
            indented: existing_indented,
            ..
        }) = last_entry
            && *existing_indented == indented
            && let Some((chunk_index, chunk)) = chunks.iter_mut().enumerate().next_back()
        {
            match (chunk, is_thought) {
                (
                    AssistantMessageChunk::Message {
                        identity: MessageIdentity::Legacy(existing_id),
                        block,
                        ..
                    },
                    false,
                )
                | (
                    AssistantMessageChunk::Thought {
                        identity: MessageIdentity::Legacy(existing_id),
                        block,
                        ..
                    },
                    true,
                ) if can_merge_message_chunks(existing_id.as_ref(), message_id) => {
                    let markdown = block.trailing_text()?.clone();
                    if existing_id.is_none() {
                        *existing_id = message_id.cloned();
                    }
                    Some((
                        block,
                        StreamingTextTarget {
                            entry_index,
                            chunk_index,
                            markdown,
                        },
                    ))
                }
                _ => None,
            }
        } else {
            None
        }
    }

    /// Add text to the streaming buffer. If the target changed (e.g. switching
    /// from thoughts to message text), flush the old buffer first.
    fn buffer_streaming_text(
        &mut self,
        target: StreamingTextTarget,
        source_index: usize,
        text_len: usize,
        cx: &mut Context<Self>,
    ) {
        if let Some(buffer) = &mut self.streaming_text_buffer
            && buffer.target == target
        {
            buffer.cursor.pending_bytes += text_len;
            buffer.bytes_to_reveal_per_tick = (buffer.cursor.pending_bytes as f32
                / StreamingTextBuffer::REVEAL_TARGET
                * StreamingTextBuffer::TASK_UPDATE_MS as f32)
                .ceil() as usize;
            return;
        }
        self.flush_streaming_text(cx);

        let _reveal_task = self.start_streaming_reveal(cx);
        let bytes_to_reveal = (text_len as f32 / StreamingTextBuffer::REVEAL_TARGET
            * StreamingTextBuffer::TASK_UPDATE_MS as f32)
            .ceil() as usize;
        self.streaming_text_buffer = Some(StreamingTextBuffer {
            cursor: TextCursor {
                source_index,
                byte_offset: 0,
                pending_bytes: text_len,
            },
            bytes_to_reveal_per_tick: bytes_to_reveal,
            target,
            _reveal_task,
        });
    }

    /// Flush all buffered streaming text into the Markdown entity immediately.
    fn flush_streaming_text(&mut self, cx: &mut Context<Self>) {
        if let Some(mut buffer) = self.streaming_text_buffer.take() {
            buffer.reveal(&self.entries, usize::MAX, cx);
        }
    }

    fn shrink_message_source_capacity(&mut self, first_entry_index: usize) {
        for entry in self.entries.iter_mut().skip(first_entry_index) {
            match entry {
                AgentThreadEntry::UserMessage(message) => message.content.shrink_source_capacity(),
                AgentThreadEntry::AssistantMessage(message) => {
                    for chunk in &mut message.chunks {
                        let (AssistantMessageChunk::Message { block, .. }
                        | AssistantMessageChunk::Thought { block, .. }) = chunk;
                        block.shrink_source_capacity();
                    }
                }
                _ => {}
            }
        }
    }

    /// Reveals retained source text gradually without changing its content.
    fn start_streaming_reveal(&self, cx: &mut Context<Self>) -> Task<()> {
        cx.spawn(async move |this, cx| {
            loop {
                cx.background_executor()
                    .timer(Duration::from_millis(StreamingTextBuffer::TASK_UPDATE_MS))
                    .await;

                let should_continue = this
                    .update(cx, |this, cx| {
                        let Some(buffer) = &mut this.streaming_text_buffer else {
                            return false;
                        };

                        let valid =
                            buffer.reveal(&this.entries, buffer.bytes_to_reveal_per_tick, cx);
                        if !valid {
                            this.streaming_text_buffer.take();
                        }
                        valid
                    })
                    .unwrap_or(false);

                if !should_continue {
                    break;
                }
            }
        })
    }

    fn push_entry(&mut self, entry: AgentThreadEntry, cx: &mut Context<Self>) {
        self.flush_streaming_text(cx);
        self.entries.push(entry);
        cx.emit(AcpThreadEvent::NewEntry);
    }

    fn truncate_entries(&mut self, index: usize, cx: &mut Context<Self>) {
        let removed_requests = self
            .entries
            .iter()
            .skip(index)
            .filter_map(|entry| match entry {
                AgentThreadEntry::ToolCall(call) => call.authorization_id(),
                _ => None,
            })
            .collect::<Vec<_>>();
        self.entries.truncate(index);
        for id in removed_requests {
            self.resolve_permission_request(id, RequestPermissionOutcome::Cancelled, cx);
        }
        // Unanchored prompts cannot remain valid across a transcript rollback.
        self.cancel_generic_permission_requests(cx);
    }

    pub fn push_context_compaction(
        &mut self,
        compaction: ContextCompaction,
        cx: &mut Context<Self>,
    ) {
        if let Some(ix) =
            self.entries
                .iter()
                .enumerate()
                .rev()
                .find_map(|(ix, entry)| match entry {
                    AgentThreadEntry::ContextCompaction(c) if &c.id == &compaction.id => Some(ix),
                    _ => None,
                })
        {
            self.entries[ix] = AgentThreadEntry::ContextCompaction(compaction);
            cx.emit(AcpThreadEvent::EntryUpdated(ix));
        } else {
            self.push_entry(AgentThreadEntry::ContextCompaction(compaction), cx);
        }
    }

    pub fn upsert_context_compaction_update(
        &mut self,
        mut update: acp_v2::CompactionUpdate,
        cx: &mut Context<Self>,
    ) {
        let metadata = MetadataUpdate::V2(mem::take(&mut update.meta));
        self.upsert_context_compaction_update_inner(update, metadata, cx);
    }

    fn upsert_context_compaction_update_inner(
        &mut self,
        update: acp_v2::CompactionUpdate,
        metadata: MetadataUpdate,
        cx: &mut Context<Self>,
    ) {
        let id = ContextCompactionId(update.compaction_id.0.clone());
        let language_registry = self.project.read(cx).languages().clone();

        if let Some((entry_index, compaction)) =
            self.entries
                .iter_mut()
                .enumerate()
                .rev()
                .find_map(|(entry_index, entry)| match entry {
                    AgentThreadEntry::ContextCompaction(compaction) if compaction.id == id => {
                        Some((entry_index, compaction))
                    }
                    _ => None,
                })
        {
            compaction.apply_update(update, metadata, &language_registry, cx);
            cx.emit(AcpThreadEvent::EntryUpdated(entry_index));
            return;
        }

        let mut compaction = ContextCompaction {
            id,
            status: update.status.clone().into(),
            error: None,
            summary: MessageContent::default(),
            meta: None,
        };
        compaction.apply_update(update, metadata, &language_registry, cx);
        self.push_entry(AgentThreadEntry::ContextCompaction(compaction), cx);
    }

    pub fn append_context_compaction_summary(
        &mut self,
        chunk: acp_v2::CompactionSummaryChunk,
        cx: &mut Context<Self>,
    ) {
        let language_registry = self.project.read(cx).languages().clone();
        if let Some((entry_index, compaction)) =
            self.entries
                .iter_mut()
                .enumerate()
                .rev()
                .find_map(|(entry_index, entry)| match entry {
                    AgentThreadEntry::ContextCompaction(compaction)
                        if compaction.id.0 == chunk.compaction_id.0 =>
                    {
                        Some((entry_index, compaction))
                    }
                    _ => None,
                })
        {
            // Chunk metadata is delivery-scoped, not a patch to the compaction record.
            compaction.append_summary(chunk.content, &language_registry, cx);
            cx.emit(AcpThreadEvent::EntryUpdated(entry_index));
            return;
        }
        let mut compaction = ContextCompaction {
            id: ContextCompactionId(chunk.compaction_id.0),
            status: ContextCompactionStatus::InProgress,
            error: None,
            summary: MessageContent::default(),
            meta: None,
        };
        compaction.append_summary(chunk.content, &language_registry, cx);
        self.push_entry(AgentThreadEntry::ContextCompaction(compaction), cx);
    }

    pub fn update_context_compaction(
        &mut self,
        update: ContextCompactionUpdate,
        cx: &mut Context<Self>,
    ) {
        let language_registry = self.project.read(cx).languages().clone();
        let Some((ix, compaction)) =
            self.entries
                .iter_mut()
                .enumerate()
                .rev()
                .find_map(|(ix, entry)| match entry {
                    AgentThreadEntry::ContextCompaction(c) if &c.id == &update.id => Some((ix, c)),
                    _ => None,
                })
        else {
            return;
        };

        if !update.summary_delta.is_empty() {
            compaction.append_summary(
                acp_v2::ContentBlock::Text(acp_v2::TextContent::new(update.summary_delta)),
                &language_registry,
                cx,
            );
        }

        if let Some(status) = update.status {
            compaction.status = status;
        }

        cx.emit(AcpThreadEvent::EntryUpdated(ix));
    }

    pub fn update_session_info(
        &mut self,
        update: impl Into<AgentSessionInfoUpdate>,
        cx: &mut Context<Self>,
    ) {
        let update = update.into();
        let previous_title = self.session_info.title.clone();
        let had_provisional = update.has_title() && self.provisional_title.take().is_some();
        let changed = self.session_info.apply_update(update);
        if self.session_info.title != previous_title || had_provisional {
            cx.emit(AcpThreadEvent::TitleUpdated);
        }
        if changed || had_provisional {
            cx.notify();
        }
    }

    pub fn can_set_title(&mut self, cx: &mut Context<Self>) -> bool {
        self.connection.set_title(self.session_id(), cx).is_some()
    }

    pub fn set_title(&mut self, title: SharedString, cx: &mut Context<Self>) -> Task<Result<()>> {
        let had_provisional = self.provisional_title.take().is_some();
        if self.session_info.title.as_ref() != Some(&title) {
            self.session_info.title = Some(title.clone());
            cx.emit(AcpThreadEvent::TitleUpdated);
            if let Some(set_title) = self.connection.set_title(self.session_id(), cx) {
                return set_title.run(title, cx);
            }
        } else if had_provisional {
            cx.emit(AcpThreadEvent::TitleUpdated);
        }
        Task::ready(Ok(()))
    }

    /// Sets a provisional display title without propagating back to the
    /// underlying agent connection. This is used for quick preview titles
    /// (e.g. first 20 chars of the user message) that should be shown
    /// immediately but replaced once the LLM generates a proper title via
    /// `set_title`.
    pub fn set_provisional_title(&mut self, title: SharedString, cx: &mut Context<Self>) {
        self.provisional_title = Some(title);
        cx.emit(AcpThreadEvent::TitleUpdated);
    }

    /// Takes ownership of a child session. A session ID that is already registered
    /// keeps its existing thread. Callers must not register this session or one of its
    /// ancestors, since the registry holds children strongly.
    pub fn register_subagent(&mut self, subagent: Entity<AcpThread>, cx: &mut Context<Self>) {
        let session_id = subagent.read(cx).session_id().clone();
        if self.subagents.contains_key(&session_id) {
            return;
        }
        self.subagents.insert(session_id.clone(), subagent);
        cx.emit(AcpThreadEvent::SubagentSpawned(session_id));
    }

    pub fn subagent(&self, session_id: &acp_v2::SessionId) -> Option<&Entity<AcpThread>> {
        self.subagents.get(session_id)
    }

    pub fn subagents(&self) -> impl Iterator<Item = &Entity<AcpThread>> {
        self.subagents.values()
    }

    pub fn update_token_usage(&mut self, usage: Option<TokenUsage>, cx: &mut Context<Self>) {
        let output_tokens = usage.as_ref().map_or(0, |usage| usage.output_tokens);
        self.update_token_usage_with_output_delta(usage, output_tokens, cx);
    }

    /// Apply an absolute usage snapshot without recounting previously reported output.
    pub fn update_token_usage_with_output_delta(
        &mut self,
        usage: Option<TokenUsage>,
        output_tokens: u64,
        cx: &mut Context<Self>,
    ) {
        if usage.is_none() {
            self.cost = None;
        }
        let output = usage.as_ref().map_or(OutputTokensUpdate::Unchanged, |_| {
            OutputTokensUpdate::LegacyAdd {
                activity_generation: self.activity.generation(),
                output_tokens,
            }
        });
        self.token_usage = usage;
        cx.emit(AcpThreadEvent::TokenUsageUpdated(output));
    }

    pub fn update_usage(&mut self, update: acp_v2::UsageUpdate, cx: &mut Context<Self>) {
        let usage = self.token_usage.get_or_insert_with(Default::default);
        usage.max_tokens = update.size;
        usage.used_tokens = update.used;
        match update.cost {
            MaybeUndefined::Undefined => {}
            MaybeUndefined::Null => self.cost = None,
            MaybeUndefined::Value(cost) => {
                self.cost = Some(SessionCost {
                    amount: cost.amount,
                    currency: cost.currency.into(),
                    meta: cost.meta,
                });
            }
        }
        cx.emit(AcpThreadEvent::TokenUsageUpdated(
            OutputTokensUpdate::Unchanged,
        ));
    }

    pub fn update_retry_status(&mut self, status: RetryStatus, cx: &mut Context<Self>) {
        cx.emit(AcpThreadEvent::Retry(status));
    }

    pub fn update_tool_call(
        &mut self,
        update: impl Into<ToolCallUpdate>,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        match update.into() {
            ToolCallUpdate::V1(update) => {
                let id = acp_v2::ToolCallId::new(update.tool_call_id.0);
                let Some(index) = self.tool_call_index_for_update(&id, cx)? else {
                    return Ok(());
                };
                let languages = self.project.read(cx).languages().clone();
                let Some(AgentThreadEntry::ToolCall(call)) = self.entries.get_mut(index) else {
                    anyhow::bail!("Tool call entry disappeared while updating");
                };
                let patch = ToolCallPatch::legacy(update.fields, update.meta);
                let location_updated = !patch.locations.is_undefined();
                let authorization_id = call.authorization_id();
                let result = call.apply_patch(
                    patch,
                    languages,
                    ToolTerminalResolver::registered(&self.terminals),
                    cx,
                );
                let detached_id =
                    authorization_id.filter(|id| call.authorization_id() != Some(*id));
                if let Some(id) = detached_id {
                    self.resolve_permission_request(id, RequestPermissionOutcome::Cancelled, cx);
                }
                if let Err(error) = result {
                    cx.emit(AcpThreadEvent::EntryUpdated(index));
                    return Err(error);
                }
                if location_updated {
                    self.resolve_locations(id, cx);
                }
                cx.emit(AcpThreadEvent::EntryUpdated(index));
                Ok(())
            }
            ToolCallUpdate::V2(update) => self.upsert_local_tool_call(update, cx),
            ToolCallUpdate::UpdateDiff(update) => {
                self.update_tool_call_content(update.id, ToolCallContent::Diff(update.diff), cx)
            }
            ToolCallUpdate::UpdateTerminal(update) => self.update_tool_call_content(
                update.id,
                ToolCallContent::Terminal {
                    terminal: update.terminal,
                    meta: None,
                },
                cx,
            ),
        }
    }

    fn tool_call_index_for_update(
        &mut self,
        id: &acp_v2::ToolCallId,
        cx: &mut Context<Self>,
    ) -> Result<Option<usize>> {
        if let Some(index) = self.index_for_tool_call(id) {
            return Ok(Some(index));
        }
        let languages = self.project.read(cx).languages().clone();
        let failed_tool_call = ToolCall::from_acp(
            acp_v1::ToolCall::new(acp_v1::ToolCallId::new(id.0.clone()), "Tool call not found")
                .kind(acp_v1::ToolKind::Fetch)
                .status(acp_v1::ToolCallStatus::Failed)
                .content(vec!["Tool call not found".into()]),
            Some(ToolCallStatus::Failed),
            languages,
            &self.terminals,
            cx,
        )?;
        self.push_entry(AgentThreadEntry::ToolCall(failed_tool_call), cx);
        Ok(None)
    }

    fn update_tool_call_content(
        &mut self,
        id: acp_v2::ToolCallId,
        content: ToolCallContent,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        let Some(index) = self.tool_call_index_for_update(&id, cx)? else {
            return Ok(());
        };
        let languages = self.project.read(cx).languages().clone();
        let Some(AgentThreadEntry::ToolCall(call)) = self.entries.get_mut(index) else {
            anyhow::bail!("Tool call entry disappeared while updating content");
        };
        call.structured_content.clear();
        call.structured_content.push(content);
        call.update_raw_output_content(&languages, cx);
        cx.emit(AcpThreadEvent::EntryUpdated(index));
        Ok(())
    }

    /// Updates a tool call if id matches an existing entry, otherwise inserts a new one.
    pub fn upsert_tool_call(
        &mut self,
        tool_call: acp_v1::ToolCall,
        cx: &mut Context<Self>,
    ) -> Result<(), acp_v1::Error> {
        let status = ToolCallStatus::from_reported(tool_status_from_v1(tool_call.status).as_ref());
        if let Some(status) = status {
            self.report_tool_call_completed(status);
        }
        self.upsert_tool_call_inner(tool_call.into(), status, cx)
    }

    fn report_tool_call_completed(&self, status: ToolCallStatus) {
        let agent_telemetry_id = self.connection().telemetry_id();
        let session = self.session_id();
        let parent_session_id = self.parent_session_id();
        if let ToolCallStatus::Completed | ToolCallStatus::Failed = status {
            let status = if matches!(status, ToolCallStatus::Completed) {
                "completed"
            } else {
                "failed"
            };
            telemetry::event!(
                "Agent Tool Call Completed",
                agent_telemetry_id,
                session,
                parent_session_id,
                status
            );
        }
    }

    fn upsert_tool_call_inner(
        &mut self,
        update: acp_v1::ToolCallUpdate,
        status: Option<ToolCallStatus>,
        cx: &mut Context<Self>,
    ) -> Result<(), acp_v1::Error> {
        let id = acp_v2::ToolCallId::new(update.tool_call_id.0.clone());
        let update = if self.index_for_tool_call(&id).is_none() {
            acp_v1::ToolCallUpdate::from(acp_v1::ToolCall::try_from(update)?)
        } else {
            update
        };
        self.upsert_legacy_behavior_tool_call(
            id,
            ToolCallPatch::legacy(update.fields, update.meta),
            status,
            cx,
        )
        .map_err(Into::into)
    }

    fn upsert_legacy_behavior_tool_call(
        &mut self,
        id: acp_v2::ToolCallId,
        patch: ToolCallPatch,
        status: Option<ToolCallStatus>,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        let language_registry = self.project.read(cx).languages().clone();
        if let Some(ix) = self.index_for_tool_call(&id) {
            let AgentThreadEntry::ToolCall(call) = &mut self.entries[ix] else {
                unreachable!()
            };

            let authorization_id = call.authorization_id();
            let result = call.apply_patch(
                patch,
                language_registry,
                ToolTerminalResolver::registered(&self.terminals),
                cx,
            );
            let detached_id = authorization_id.filter(|id| call.authorization_id() != Some(*id));
            if result.is_ok()
                && let Some(status) = status
            {
                call.set_legacy_status(status);
            }
            if let Some(id) = detached_id {
                self.resolve_permission_request(id, RequestPermissionOutcome::Cancelled, cx);
            }
            if let Err(error) = result {
                cx.emit(AcpThreadEvent::EntryUpdated(ix));
                return Err(error);
            }

            cx.emit(AcpThreadEvent::EntryUpdated(ix));
        } else {
            anyhow::ensure!(
                patch.title.value().is_some(),
                "title is required for a tool call"
            );
            let mut call = ToolCall::from_patch(
                id.clone(),
                patch,
                language_registry,
                ToolTerminalResolver::registered(&self.terminals),
                cx,
            )?;
            if let Some(status) = status {
                call.set_legacy_status(status);
            }
            self.push_entry(AgentThreadEntry::ToolCall(call), cx);
        };

        self.resolve_locations(id, cx);
        Ok(())
    }

    /// Wire updates may create agent-reported display terminals.
    pub fn upsert_wire_tool_call(
        &mut self,
        update: acp_v2::ToolCallUpdate,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        if let Some(content) = update.content.value() {
            for content in content {
                self.ensure_tool_content_terminal(content, cx);
            }
        }
        let id = update.tool_call_id.clone();
        let patch = ToolCallPatch::protocol(update);
        self.upsert_tool_call_patch(id, patch, false, cx)
    }

    /// In-process updates may reference registered client-managed terminals, but
    /// must not create or take ownership of agent-reported display terminals.
    pub fn upsert_local_tool_call(
        &mut self,
        update: acp_v2::ToolCallUpdate,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        let id = update.tool_call_id.clone();
        let patch = ToolCallPatch::protocol(update);
        self.upsert_tool_call_patch(id, patch, true, cx)
    }

    fn upsert_tool_call_patch(
        &mut self,
        id: acp_v2::ToolCallId,
        patch: ToolCallPatch,
        client_managed_terminals_only: bool,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        let locations_changed = !patch.locations.is_undefined();
        if let Some(status) = ToolCallStatus::from_reported(patch.status.value()) {
            self.report_tool_call_completed(status);
        }
        let languages = self.project.read(cx).languages().clone();
        let terminals = ToolTerminalResolver {
            terminals: &self.terminals,
            client_managed_only: client_managed_terminals_only,
        };
        if let Some(index) = self.index_for_tool_call(&id) {
            let AgentThreadEntry::ToolCall(call) = &mut self.entries[index] else {
                unreachable!()
            };
            let authorization_id = call.authorization_id();
            let result = call.apply_patch(patch, languages, terminals, cx);
            let detached_id = authorization_id.filter(|id| call.authorization_id() != Some(*id));
            if let Some(id) = detached_id {
                self.resolve_permission_request(id, RequestPermissionOutcome::Cancelled, cx);
            }
            cx.emit(AcpThreadEvent::EntryUpdated(index));
            result?;
        } else {
            let call = ToolCall::from_patch(id.clone(), patch, languages, terminals, cx)?;
            self.push_entry(AgentThreadEntry::ToolCall(call), cx);
        }
        if locations_changed {
            self.resolve_locations(id, cx);
        }
        Ok(())
    }

    pub fn append_tool_call_content_chunk(
        &mut self,
        chunk: acp_v2::ToolCallContentChunk,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        // Delivery metadata is not retained as history or promoted into aggregate
        // tool/content metadata, matching the message-chunk boundary.
        let acp_v2::ToolCallContentChunk {
            tool_call_id,
            content,
            ..
        } = chunk;
        self.ensure_tool_content_terminal(&content, cx);
        let content = PreparedToolCallContent::from_v2(
            content,
            ToolTerminalResolver::registered(&self.terminals),
            cx,
        )?;
        let language_registry = self.project.read(cx).languages().clone();
        let id = tool_call_id;

        if let Some((index, call)) = self.tool_call_mut(&id) {
            call.append_content(content, &language_registry, cx);
            cx.emit(AcpThreadEvent::EntryUpdated(index));
        } else {
            let mut call = ToolCall::from_patch(
                id.clone(),
                ToolCallPatch::protocol(acp_v2::ToolCallUpdate::new(id)),
                language_registry.clone(),
                ToolTerminalResolver::registered(&self.terminals),
                cx,
            )?;
            call.append_content(content, &language_registry, cx);
            self.push_entry(AgentThreadEntry::ToolCall(call), cx);
        }
        Ok(())
    }

    fn ensure_tool_content_terminal(
        &mut self,
        content: &acp_v2::ToolCallContent,
        cx: &mut Context<Self>,
    ) {
        if let acp_v2::ToolCallContent::Terminal(terminal) = content {
            self.ensure_display_terminal(terminal.terminal_id.clone(), cx);
        }
    }

    fn index_for_tool_call(&self, id: &acp_v2::ToolCallId) -> Option<usize> {
        self.entries
            .iter()
            .enumerate()
            .rev()
            .find_map(|(index, entry)| {
                if let AgentThreadEntry::ToolCall(tool_call) = entry
                    && &tool_call.id == id
                {
                    Some(index)
                } else {
                    None
                }
            })
    }

    fn tool_call_mut(&mut self, id: &acp_v2::ToolCallId) -> Option<(usize, &mut ToolCall)> {
        // The tool call we are looking for is typically the last one, or very close to the end.
        // At the moment, it doesn't seem like a hashmap would be a good fit for this use case.
        self.entries
            .iter_mut()
            .enumerate()
            .rev()
            .find_map(|(index, tool_call)| {
                if let AgentThreadEntry::ToolCall(tool_call) = tool_call
                    && &tool_call.id == id
                {
                    Some((index, tool_call))
                } else {
                    None
                }
            })
    }

    pub fn tool_call(&self, id: &acp_v2::ToolCallId) -> Option<(usize, &ToolCall)> {
        self.entries
            .iter()
            .enumerate()
            .rev()
            .find_map(|(index, tool_call)| {
                if let AgentThreadEntry::ToolCall(tool_call) = tool_call
                    && &tool_call.id == id
                {
                    Some((index, tool_call))
                } else {
                    None
                }
            })
    }

    /// The latest tool call that ran the subagent `session_id`. A resumed subagent has one
    /// per run, and the latest says how the subagent's current work went.
    pub fn tool_call_for_subagent(&self, session_id: &acp_v2::SessionId) -> Option<&ToolCall> {
        self.entries.iter().rev().find_map(|entry| match entry {
            AgentThreadEntry::ToolCall(tool_call) => {
                if let Some(subagent_session_info) = &tool_call.subagent_session_info
                    && &subagent_session_info.session_id == session_id
                {
                    Some(tool_call)
                } else {
                    None
                }
            }
            _ => None,
        })
    }

    pub fn resolve_locations(&mut self, id: acp_v2::ToolCallId, cx: &mut Context<Self>) {
        let project = self.project.clone();
        let should_update_agent_location = self.parent_session_id.is_none();
        let Some((_, tool_call)) = self.tool_call_mut(&id) else {
            return;
        };
        let expected_locations = tool_call.locations.clone();
        let task = tool_call.resolve_locations(project, cx);
        cx.spawn(async move |this, cx| {
            let resolved_locations = task.await;

            this.update(cx, |this, cx| {
                if this
                    .tool_call(&id)
                    .is_none_or(|(_, call)| call.locations != expected_locations)
                {
                    return;
                }
                let project = this.project.clone();

                for location in resolved_locations.iter().flatten() {
                    this.shared_buffers
                        .insert(location.buffer.clone(), location.buffer.read(cx).snapshot());
                }
                let Some((ix, tool_call)) = this.tool_call_mut(&id) else {
                    return;
                };

                if let Some(Some(location)) = resolved_locations.last() {
                    project.update(cx, |project, cx| {
                        let should_ignore = if let Some(agent_location) = project
                            .agent_location()
                            .filter(|agent_location| agent_location.buffer == location.buffer)
                        {
                            let snapshot = location.buffer.read(cx).snapshot();
                            let old_position = agent_location.position.to_point(&snapshot);
                            let new_position = location.position.to_point(&snapshot);

                            // ignore this so that when we get updates from the edit tool
                            // the position doesn't reset to the startof line
                            old_position.row == new_position.row
                                && old_position.column > new_position.column
                        } else {
                            false
                        };
                        if !should_ignore && should_update_agent_location {
                            project.set_agent_location(Some(location.into()), cx);
                        }
                    });
                }

                let resolved_locations = resolved_locations
                    .iter()
                    .map(|l| l.as_ref().map(|l| AgentLocation::from(l)))
                    .collect::<Vec<_>>();

                if tool_call.resolved_locations != resolved_locations {
                    tool_call.resolved_locations = resolved_locations;
                    cx.emit(AcpThreadEvent::EntryUpdated(ix));
                }
            })
        })
        .detach();
    }

    pub fn permission_request(&self, id: PermissionRequestId) -> Option<&PermissionRequest> {
        self.permission_requests.get(&id)
    }

    pub fn permission_request_for_tool(
        &self,
        tool_call_id: &acp_v2::ToolCallId,
    ) -> Option<&PermissionRequest> {
        let (_, call) = self.tool_call(tool_call_id)?;
        self.permission_request(call.authorization_id()?)
    }

    pub fn pending_permission_requests(&self) -> impl Iterator<Item = &PermissionRequest> {
        self.permission_requests.values()
    }

    pub fn request_permission(
        &mut self,
        request: acp_v2::RequestPermissionRequest,
        cx: &mut Context<Self>,
    ) -> Result<(PermissionRequestId, Task<acp_v2::RequestPermissionOutcome>)> {
        anyhow::ensure!(
            request.session_id == self.session_info.session_id,
            "Permission request belongs to a different session"
        );
        anyhow::ensure!(
            !request.options.is_empty(),
            "Permission request has no offered options"
        );
        let mut option_ids = HashSet::default();
        anyhow::ensure!(
            request
                .options
                .iter()
                .all(|option| option_ids.insert(&option.option_id)),
            "Permission request has duplicate option IDs"
        );
        // The protocol applies a tool call subject exactly like a `tool_call_update`, before
        // the prompt is shown, so the request can anchor to the tool call it describes.
        if let Some(acp_v2::RequestPermissionSubject::ToolCall(subject)) = &request.subject {
            self.upsert_wire_tool_call(subject.tool_call.clone(), cx)?;
        }

        let (respond_tx, receiver) = oneshot::channel();
        let id = PermissionRequestId(Uuid::new_v4());
        self.permission_requests.insert(
            id,
            PermissionRequest {
                id,
                data: PermissionRequestData::Generic {
                    request,
                    respond_tx,
                },
            },
        );
        cx.emit(AcpThreadEvent::ToolAuthorizationRequested(id));
        Ok((
            id,
            cx.spawn(async move |_, _| {
                receiver
                    .await
                    .unwrap_or(acp_v2::RequestPermissionOutcome::Cancelled(
                        Default::default(),
                    ))
            }),
        ))
    }

    pub fn select_permission_option(
        &mut self,
        id: PermissionRequestId,
        option_id: acp_v2::PermissionOptionId,
        cx: &mut Context<Self>,
    ) {
        let Some(request) = self
            .permission_request(id)
            .and_then(PermissionRequest::generic_request)
        else {
            return;
        };
        if !request
            .options
            .iter()
            .any(|option| option.option_id == option_id)
        {
            log::debug!("Permission choice is not an offered option");
            return;
        }
        let Some(PermissionRequest {
            data: PermissionRequestData::Generic { respond_tx, .. },
            ..
        }) = self.take_permission_request(id, cx)
        else {
            return;
        };
        if respond_tx
            .send(acp_v2::RequestPermissionOutcome::Selected(
                acp_v2::SelectedPermissionOutcome::new(option_id),
            ))
            .is_err()
        {
            log::debug!("Permission request closed before the outcome was delivered");
        }
    }

    pub fn request_tool_call_authorization(
        &mut self,
        tool_call: acp_v1::ToolCallUpdate,
        options: PermissionOptions,
        kind: AuthorizationKind,
        cx: &mut Context<Self>,
    ) -> Result<Task<RequestPermissionOutcome>> {
        self.request_tool_call_authorization_with_id(tool_call, options, kind, cx)
            .map(|(_, task)| task)
    }

    pub fn request_tool_call_authorization_with_id(
        &mut self,
        tool_call: acp_v1::ToolCallUpdate,
        options: PermissionOptions,
        kind: AuthorizationKind,
        cx: &mut Context<Self>,
    ) -> Result<(PermissionRequestId, Task<RequestPermissionOutcome>)> {
        let tool_call_id = acp_v2::ToolCallId::new(tool_call.tool_call_id.0.clone());
        let current_status = self
            .tool_call(&tool_call_id)
            .and_then(|(_, tool_call)| tool_call.permission_status())
            .or_else(|| tool_call.fields.status.and_then(tool_status_from_v1));
        let current_status = ToolCallStatus::from_reported(current_status.as_ref())
            .unwrap_or(ToolCallStatus::Pending);

        self.upsert_tool_call_inner(tool_call, Some(current_status), cx)?;
        self.install_tool_call_authorization(tool_call_id, options, kind, cx)
    }

    pub fn request_tool_call_update_authorization(
        &mut self,
        tool_call: acp_v2::ToolCallUpdate,
        options: PermissionOptions,
        kind: AuthorizationKind,
        cx: &mut Context<Self>,
    ) -> Result<Task<RequestPermissionOutcome>> {
        self.request_tool_call_update_authorization_with_id(tool_call, options, kind, cx)
            .map(|(_, task)| task)
    }

    pub fn request_tool_call_update_authorization_with_id(
        &mut self,
        tool_call: acp_v2::ToolCallUpdate,
        options: PermissionOptions,
        kind: AuthorizationKind,
        cx: &mut Context<Self>,
    ) -> Result<(PermissionRequestId, Task<RequestPermissionOutcome>)> {
        let tool_call_id = tool_call.tool_call_id.clone();
        let current_status = self
            .tool_call(&tool_call_id)
            .and_then(|(_, call)| call.permission_status())
            .or_else(|| tool_call.status.value().cloned());
        let current_status = ToolCallStatus::from_reported(current_status.as_ref())
            .unwrap_or(ToolCallStatus::Pending);
        self.upsert_local_tool_call(tool_call, cx)?;
        let (_, call) = self
            .tool_call_mut(&tool_call_id)
            .context("tool call disappeared while requesting authorization")?;
        call.set_local_status(current_status);
        self.install_tool_call_authorization(tool_call_id, options, kind, cx)
    }

    fn install_tool_call_authorization(
        &mut self,
        tool_call_id: acp_v2::ToolCallId,
        options: PermissionOptions,
        kind: AuthorizationKind,
        cx: &mut Context<Self>,
    ) -> Result<(PermissionRequestId, Task<RequestPermissionOutcome>)> {
        let (tx, rx) = oneshot::channel();
        if let Some(id) = self
            .tool_call(&tool_call_id)
            .and_then(|(_, call)| call.authorization_id())
        {
            self.resolve_permission_request(id, RequestPermissionOutcome::Cancelled, cx);
        }
        let id = PermissionRequestId(Uuid::new_v4());
        let (_, call) = self
            .tool_call_mut(&tool_call_id)
            .context("tool call disappeared while requesting authorization")?;
        // A new explicit request is installed after applying its descriptive
        // fields, preserving the legacy request's captured continuation status.
        call.authorization = Some(id);
        self.permission_requests.insert(
            id,
            PermissionRequest {
                id,
                data: PermissionRequestData::LegacyTool {
                    tool_call_id,
                    options,
                    respond_tx: tx,
                    kind,
                },
            },
        );
        cx.emit(AcpThreadEvent::ToolAuthorizationRequested(id));

        Ok((
            id,
            cx.spawn(async move |_this, _cx| {
                rx.await.unwrap_or(RequestPermissionOutcome::Cancelled)
            }),
        ))
    }

    pub fn cancel_tool_call_authorization(
        &mut self,
        id: &acp_v2::ToolCallId,
        cx: &mut Context<Self>,
    ) {
        let Some(request_id) = self
            .tool_call(id)
            .and_then(|(_, call)| call.authorization_id())
        else {
            return;
        };
        self.cancel_permission_request(request_id, cx);
    }

    pub fn cancel_permission_request(&mut self, id: PermissionRequestId, cx: &mut Context<Self>) {
        self.cancel_permission_request_with_outcome(id, RequestPermissionOutcome::Cancelled, cx);
    }

    fn cancel_permission_request_with_outcome(
        &mut self,
        id: PermissionRequestId,
        outcome: RequestPermissionOutcome,
        cx: &mut Context<Self>,
    ) {
        let Some(request) = self.permission_requests.get(&id) else {
            return;
        };
        if let Some(tool_call_id) = request.legacy_tool_call_id().cloned()
            && let Some((_, call)) = self.tool_call_mut(&tool_call_id)
            && call.authorization_id() == Some(id)
        {
            call.set_local_status(ToolCallStatus::Canceled);
        }
        self.resolve_permission_request(id, outcome, cx);
    }

    pub fn authorize_tool_call(
        &mut self,
        id: acp_v2::ToolCallId,
        outcome: SelectedPermissionOutcome,
        cx: &mut Context<Self>,
    ) {
        let Some(request_id) = self
            .tool_call(&id)
            .and_then(|(_, call)| call.authorization_id())
        else {
            return;
        };
        self.authorize_permission_request(request_id, outcome, cx);
    }

    pub fn authorize_permission_request(
        &mut self,
        id: PermissionRequestId,
        mut outcome: SelectedPermissionOutcome,
        cx: &mut Context<Self>,
    ) {
        let Some(PermissionRequest {
            data:
                PermissionRequestData::LegacyTool {
                    options,
                    tool_call_id,
                    kind,
                    ..
                },
            ..
        }) = self.permission_requests.get(&id)
        else {
            return;
        };
        let Some(option) = options.option_for_id(&outcome.option_id) else {
            log::debug!("Permission choice is not an offered option");
            return;
        };
        outcome.option_kind = option.kind.clone();
        let tool_call_id = tool_call_id.clone();
        let kind = *kind;
        let Some((_, call)) = self.tool_call_mut(&tool_call_id) else {
            return;
        };
        if call.authorization_id() != Some(id) {
            return;
        }

        let new_status = match kind {
            AuthorizationKind::ActionChoice => ToolCallStatus::InProgress,
            AuthorizationKind::PermissionGrant => {
                let current_status = call.permission_status().unwrap_or_default();
                match outcome.option_kind {
                    acp_v2::PermissionOptionKind::RejectOnce
                    | acp_v2::PermissionOptionKind::RejectAlways => ToolCallStatus::Rejected,
                    acp_v2::PermissionOptionKind::AllowOnce
                    | acp_v2::PermissionOptionKind::AllowAlways => {
                        ToolCallStatus::status_after_permission_grant(current_status)
                    }
                    _ => {
                        log::warn!(
                            "Cannot authorize a tool with an unknown permission option kind"
                        );
                        return;
                    }
                }
            }
        };
        call.set_local_status(new_status);
        self.resolve_permission_request(id, RequestPermissionOutcome::Selected(outcome), cx);
    }

    fn resolve_permission_request(
        &mut self,
        id: PermissionRequestId,
        outcome: RequestPermissionOutcome,
        cx: &mut Context<Self>,
    ) {
        let Some(request) = self.take_permission_request(id, cx) else {
            return;
        };
        let delivered = match request.data {
            PermissionRequestData::LegacyTool { respond_tx, .. } => {
                respond_tx.send(outcome).is_ok()
            }
            PermissionRequestData::Generic { respond_tx, .. } => respond_tx
                .send(acp_v2::RequestPermissionOutcome::Cancelled(
                    Default::default(),
                ))
                .is_ok(),
        };
        if !delivered {
            log::debug!("Permission request closed before the outcome was delivered");
        }
    }

    fn take_permission_request(
        &mut self,
        id: PermissionRequestId,
        cx: &mut Context<Self>,
    ) -> Option<PermissionRequest> {
        let request = self.permission_requests.shift_remove(&id)?;
        if let Some(tool_call_id) = request.legacy_tool_call_id()
            && let Some((index, call)) = self.tool_call_mut(tool_call_id)
            && call.authorization_id() == Some(id)
        {
            call.authorization = None;
            cx.emit(AcpThreadEvent::EntryUpdated(index));
        }
        cx.emit(AcpThreadEvent::ToolAuthorizationReceived(id));
        Some(request)
    }

    fn cancel_generic_permission_requests(&mut self, cx: &mut Context<Self>) {
        let ids = self
            .permission_requests
            .values()
            .filter(|request| request.generic_request().is_some())
            .map(|request| request.id)
            .collect::<Vec<_>>();
        for id in ids {
            self.cancel_permission_request(id, cx);
        }
    }

    pub fn request_elicitation(
        &mut self,
        request: acp_v2::CreateElicitationRequest,
        cx: &mut Context<Self>,
    ) -> Result<Task<acp_v2::CreateElicitationResponse>, acp_v2::Error> {
        self.request_elicitation_with_id(request, cx)
            .map(|(_, task)| task)
    }

    pub fn request_elicitation_with_id(
        &mut self,
        request: acp_v2::CreateElicitationRequest,
        cx: &mut Context<Self>,
    ) -> Result<(ElicitationEntryId, Task<acp_v2::CreateElicitationResponse>), acp_v2::Error> {
        ElicitationStore::validate_request(&request)?;

        let (id, response_rx) = self.elicitations.insert_pending_elicitation(request);
        self.push_entry(AgentThreadEntry::Elicitation(id.clone()), cx);
        cx.emit(AcpThreadEvent::ElicitationRequested(id.clone()));

        let task = ElicitationStore::response_task(response_rx, cx);
        Ok((id, task))
    }

    fn emit_elicitation_change(
        entry_index: usize,
        id: &ElicitationEntryId,
        change: ElicitationChange,
        cx: &mut Context<Self>,
    ) {
        cx.emit(AcpThreadEvent::EntryUpdated(entry_index));
        if matches!(change, ElicitationChange::Responded) {
            cx.emit(AcpThreadEvent::ElicitationResponded(id.clone()));
        }
    }

    pub fn respond_to_elicitation(
        &mut self,
        id: &ElicitationEntryId,
        response: acp_v2::CreateElicitationResponse,
        cx: &mut Context<Self>,
    ) {
        let Some(ix) = self.elicitation_entry_ix(id) else {
            return;
        };
        if !self.elicitations.respond_to_elicitation_by_id(id, response) {
            return;
        }

        Self::emit_elicitation_change(ix, id, ElicitationChange::Responded, cx);
    }

    pub fn complete_url_elicitation(
        &mut self,
        elicitation_id: &acp_v2::ElicitationId,
        cx: &mut Context<Self>,
    ) {
        let Some(entry_id) = self
            .elicitations
            .entry_id_for_url_elicitation(elicitation_id)
        else {
            return;
        };
        let Some(ix) = self.elicitation_entry_ix(&entry_id) else {
            return;
        };
        if !self.elicitations.complete_url_elicitation_by_id(&entry_id) {
            return;
        }

        cx.emit(AcpThreadEvent::EntryUpdated(ix));
    }

    pub fn cancel_elicitation(&mut self, id: &ElicitationEntryId, cx: &mut Context<Self>) {
        let Some(ix) = self.elicitation_entry_ix(id) else {
            return;
        };
        let Some(change) = self.elicitations.cancel_elicitation_by_id(id) else {
            return;
        };

        Self::emit_elicitation_change(ix, id, change, cx);
    }

    fn elicitation_entry_ix(&self, id: &ElicitationEntryId) -> Option<usize> {
        self.entries
            .iter()
            .enumerate()
            .rev()
            .find_map(|(index, entry)| {
                matches!(entry, AgentThreadEntry::Elicitation(elicitation_id) if elicitation_id == id)
                    .then_some(index)
            })
    }

    pub fn elicitation(&self, id: &ElicitationEntryId) -> Option<(usize, &Elicitation)> {
        let index = self.elicitation_entry_ix(id)?;
        let (_, elicitation) = self.elicitations.elicitation(id)?;
        Some((index, elicitation))
    }

    pub fn plan(&self) -> Option<&Plan> {
        self.plans.get(self.visible_plan.as_ref()?)
    }

    pub fn plan_by_id(&self, id: &acp_v2::PlanId) -> Option<&Plan> {
        self.plans.get(&PlanIdentity::Keyed(id.clone()))
    }

    pub fn update_plan(&mut self, request: acp_v1::Plan, cx: &mut Context<Self>) -> Result<()> {
        let entries = request
            .entries
            .into_iter()
            .map(plan_entry_from_v1)
            .collect::<Result<_>>()?;
        self.replace_plan(PlanIdentity::Legacy, entries, request.meta, None, cx);
        Ok(())
    }

    pub fn upsert_plan_items(
        &mut self,
        plan: acp_v2::PlanItems,
        update_meta: Option<acp_v2::Meta>,
        cx: &mut Context<Self>,
    ) {
        self.replace_plan(
            PlanIdentity::Keyed(plan.plan_id),
            plan.entries,
            plan.meta,
            update_meta,
            cx,
        );
    }

    fn replace_plan(
        &mut self,
        identity: PlanIdentity,
        entries: Vec<acp_v2::PlanEntry>,
        meta: Option<acp_v2::Meta>,
        update_meta: Option<acp_v2::Meta>,
        cx: &mut Context<Self>,
    ) {
        let is_new = !self.plans.contains_key(&identity);
        let changed =
            self.plans
                .entry(identity.clone())
                .or_default()
                .replace(entries, meta, update_meta, cx);
        if is_new || changed || identity == PlanIdentity::Legacy {
            self.visible_plan = Some(identity);
        }
        cx.notify();
    }

    fn clear_completed_plan_entries(&mut self, cx: &mut Context<Self>) {
        // V1's next-turn cleanup is a compatibility policy, not authority to alter keyed plans.
        if let Some(plan) = self.plans.get_mut(&PlanIdentity::Legacy) {
            plan.entries
                .retain(|entry| !matches!(entry.source.status, acp_v2::PlanEntryStatus::Completed));
        }
        cx.notify();
    }

    pub fn clear_plan(&mut self, cx: &mut Context<Self>) {
        self.visible_plan = None;
        cx.notify();
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn send_raw(
        &mut self,
        message: &str,
        cx: &mut Context<Self>,
    ) -> BoxFuture<'static, Result<Option<acp_v1::PromptResponse>>> {
        let submission = self.send(vec![message.into()], cx);
        async move {
            match submission.await? {
                Some(SubmissionResponse::LegacyCompleted(response)) => Ok(Some(response)),
                None => Ok(None),
                Some(SubmissionResponse::Accepted(_)) => {
                    Err(anyhow!("Use send to observe receipt-driven submissions"))
                }
            }
        }
        .boxed()
    }

    pub fn send(
        &mut self,
        message: Vec<acp_v2::ContentBlock>,
        cx: &mut Context<Self>,
    ) -> Submission {
        self.submit(message, true, cx)
    }

    /// Sends a prompt without displaying a user-message bubble for it.
    /// This is used for native slash commands (e.g. `/compact`) that run a turn
    /// which produces its own thread entry (like the compaction summary). The
    /// typed command isn't sent to the model as an ordinary user turn.
    pub fn send_command(
        &mut self,
        message: Vec<acp_v2::ContentBlock>,
        cx: &mut Context<Self>,
    ) -> Submission {
        self.submit(message, false, cx)
    }

    fn submit(
        &mut self,
        message: Vec<acp_v2::ContentBlock>,
        push_user_message: bool,
        cx: &mut Context<Self>,
    ) -> Submission {
        let id = self.register_submission(message.clone().into(), cx);
        match self.submissions.transport().clone() {
            SubmissionTransport::Receipt(submissions) => {
                let request = submissions.submit(message, cx);
                self.track_receipt_submission(id, request, cx)
            }
            SubmissionTransport::Completion => self.send_inner(id, message, push_user_message, cx),
            SubmissionTransport::AgentOwned => self.reject_agent_owned_submission(id, cx),
        }
    }

    fn reject_agent_owned_submission(
        &mut self,
        id: SubmissionId,
        cx: &mut Context<Self>,
    ) -> Submission {
        self.track_submission(id, None, cx, async move |_, _| {
            Err(anyhow!(
                "The agent owns this session, so it does not accept prompts"
            ))
        })
    }

    pub fn submit_receipt(
        &mut self,
        content: Vec<acp_v2::ContentBlock>,
        request: impl Into<ReceiptSubmission>,
        cx: &mut Context<Self>,
    ) -> Submission {
        let id = self.register_submission(content.into(), cx);
        self.track_receipt_submission(id, request.into(), cx)
    }

    fn track_receipt_submission(
        &mut self,
        id: SubmissionId,
        request: ReceiptSubmission,
        cx: &mut Context<Self>,
    ) -> Submission {
        let ReceiptSubmission { response, applied } = request;
        self.track_submission(id, applied, cx, async move |_, _| {
            Ok(Some(SubmissionResponse::Accepted(response.await?)))
        })
    }

    pub fn upsert_local_user_message(
        &mut self,
        id: ClientUserMessageId,
        content: Vec<acp_v2::ContentBlock>,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        let previous_entry_count = self.entries.len();
        let location = self.keyed_message_location(MessageKind::User, id.message_id(), cx)?;
        let language_registry = self.project.read(cx).languages().clone();
        let path_style = self.project.read(cx).path_style(cx);
        let Some(AgentThreadEntry::UserMessage(message)) =
            self.entries.get_mut(location.entry_index())
        else {
            return Err(anyhow!("user message disappeared during preparation"));
        };
        message.client_id = Some(id);
        message.is_optimistic = false;
        message
            .content
            .replace_prompt(content, &language_registry, path_style, cx);
        self.emit_message_update(location, previous_entry_count, cx);
        cx.notify();
        Ok(())
    }

    pub fn finish_local_turn(&mut self, reason: acp_v2::StopReason, cx: &mut Context<Self>) {
        self.flush_streaming_text(cx);
        self.cleanup_local_turn(&reason, true, cx);
    }

    pub fn validate_prompt_content(&self, content: &[acp_v2::ContentBlock]) -> Result<()> {
        self.connection.validate_prompt_content(content)
    }

    fn send_inner(
        &mut self,
        id: SubmissionId,
        message: Vec<acp_v2::ContentBlock>,
        push_user_message: bool,
        cx: &mut Context<Self>,
    ) -> Submission {
        let request = acp_v2::PromptRequest::new(self.session_id().clone(), message);
        if let Err(error) = self.validate_prompt_content(&request.prompt) {
            return self.track_submission(id, None, cx, async move |_, _| Err(error));
        }
        let language_registry = self.project.read(cx).languages().clone();
        let path_style = self.project.read(cx).path_style(cx);
        let mut block = MessageContent::default();
        for chunk in &request.prompt {
            block.append_prompt(chunk.clone(), &language_registry, path_style, cx);
        }
        let client_user_message_ids = self.connection.client_user_message_ids(cx);
        let client_id = client_user_message_ids
            .as_ref()
            .map(|client_user_message_ids| client_user_message_ids.new_id());

        self.run_turn(id, cx, async move |this, cx| {
            if push_user_message {
                this.update(cx, |this, cx| {
                    this.push_entry(
                        AgentThreadEntry::UserMessage(UserMessage {
                            identity: MessageIdentity::Legacy(None),
                            meta: None,
                            client_id: client_id.clone(),
                            is_optimistic: true,
                            content: block,
                            indented: false,
                        }),
                        cx,
                    );
                })
                .ok();
            }

            this.update(cx, |this, cx| {
                if let (Some(prompt), Some(client_id)) = (client_user_message_ids, client_id) {
                    prompt.prompt(client_id, request, cx)
                } else {
                    this.connection.prompt(request, cx)
                }
            })?
            .await
        })
    }

    pub fn can_retry(&self, cx: &App) -> bool {
        let supports_retry = match self.submissions.transport() {
            SubmissionTransport::Receipt(submissions) => submissions.supports_retry(),
            SubmissionTransport::Completion => {
                self.connection.retry(self.session_id(), cx).is_some()
            }
            SubmissionTransport::AgentOwned => false,
        };
        supports_retry
            && self
                .submissions
                .latest_id()
                .and_then(|id| self.submissions.get(id))
                .is_none_or(|submission| self.validate_prompt_content(&submission.content).is_ok())
    }

    pub fn retry(&mut self, cx: &mut Context<Self>) -> Submission {
        let content = self
            .submissions
            .latest_id()
            .and_then(|id| self.submissions.get(id))
            .map(|submission| submission.content.clone())
            .unwrap_or_default();
        let id = self.register_submission(content.clone(), cx);
        if let Err(error) = self.validate_prompt_content(&content) {
            return self.track_submission(id, None, cx, async move |_, _| Err(error));
        }
        match self.submissions.transport() {
            SubmissionTransport::Receipt(submissions) => {
                let request = submissions.retry(cx);
                return self.track_receipt_submission(id, request, cx);
            }
            SubmissionTransport::AgentOwned => {
                return self.reject_agent_owned_submission(id, cx);
            }
            SubmissionTransport::Completion => {}
        }
        self.run_turn(id, cx, async move |this, cx| {
            this.update(cx, |this, cx| {
                this.connection
                    .retry(this.session_id(), cx)
                    .map(|retry| retry.run(cx))
            })?
            .context("retrying a session is not supported")?
            .await
        })
    }

    fn run_turn(
        &mut self,
        id: SubmissionId,
        cx: &mut Context<Self>,
        f: impl 'static + AsyncFnOnce(WeakEntity<Self>, &mut AsyncApp) -> Result<acp_v1::PromptResponse>,
    ) -> Submission {
        let cancel_task = match self
            .request_cancellation(RequestPermissionOutcome::InterruptedByFollowUp, cx)
        {
            Ok(task) => task,
            Err(error) => return self.track_submission(id, None, cx, async move |_, _| Err(error)),
        };
        self.clear_completed_plan_entries(cx);
        self.had_error = false;
        self.local_execution_error = None;

        let (tx, rx) = oneshot::channel();

        self.turn_id += 1;
        let turn_id = self.turn_id;
        // A continuation can extend the trailing entry rather than create a new one.
        let first_entry_index = self.entries.len().saturating_sub(1);
        self.running_turn = Some(RunningTurn {
            id: turn_id,
            first_entry_index,
            send_task: cx.spawn(async move |this, cx| {
                let response = match cancel_task.await {
                    Ok(()) => f(this, cx).await,
                    Err(error) => Err(error),
                };
                if let Err(Err(error)) = tx.send(response) {
                    log::debug!("Legacy submission observer was released: {error:#}");
                }
            }),
        });
        self.set_foreground_state(
            acp_v2::StateUpdate::Running(acp_v2::RunningStateUpdate::new()),
            cx,
        );
        cx.emit(AcpThreadEvent::StatusChanged);

        let completion = async move |thread: WeakEntity<Self>, cx: &mut AsyncApp| {
            let response = rx.await;

            thread.update(cx, |this, cx| {
                if this.turn_id == turn_id && this.parent_session_id.is_none() {
                    this.project
                        .update(cx, |project, cx| project.set_agent_location(None, cx));
                }

                let is_same_turn = this
                    .running_turn
                    .as_ref()
                    .is_some_and(|turn| turn_id == turn.id);

                // If the user submitted a follow up message, running_turn might
                // already point to a different turn. Therefore we only want to
                // take the task if it's the same turn. We do this before the
                // dropped-tx guard below so the panel exits its generating
                // state even when the send_task is cancelled before tx.send().
                if is_same_turn {
                    this.running_turn.take();
                    this.set_foreground_state(
                        acp_v2::StateUpdate::Idle(acp_v2::IdleStateUpdate::new()),
                        cx,
                    );
                }

                if this.turn_id == turn_id {
                    this.shrink_message_source_capacity(first_entry_index);
                }

                let Ok(response) = response else {
                    if is_same_turn {
                        cx.emit(AcpThreadEvent::StatusChanged);
                    }
                    // tx dropped, just return
                    return Ok(None);
                };

                if this.turn_id != turn_id {
                    return response.map(Some);
                }

                match response {
                    Ok(r) => {
                        this.flush_streaming_text(cx);

                        if r.stop_reason == acp_v1::StopReason::MaxTokens {
                            if is_same_turn {
                                cx.emit(AcpThreadEvent::StatusChanged);
                            }
                            this.had_error = true;
                            cx.emit(AcpThreadEvent::Error);
                            log::error!("Max tokens reached. Usage: {:?}", this.token_usage);

                            let exceeded_max_output_tokens =
                                this.token_usage.as_ref().is_some_and(|u| {
                                    u.max_output_tokens
                                        .is_some_and(|max| u.output_tokens >= max)
                                });

                            if exceeded_max_output_tokens {
                                log::error!(
                                    "Max output tokens reached. Usage: {:?}",
                                    this.token_usage
                                );
                            } else {
                                log::error!("Max tokens reached. Usage: {:?}", this.token_usage);
                            }
                            this.cleanup_local_turn(
                                &acp_v2::StopReason::MaxTokens,
                                is_same_turn,
                                cx,
                            );
                            return Err(anyhow!(MaxOutputTokensError));
                        }

                        if let Some(reason) = stop_reason_from_v1(&r.stop_reason) {
                            this.cleanup_local_turn(&reason, is_same_turn, cx);
                        }
                        if r.stop_reason == acp_v1::StopReason::Refusal {
                            cx.emit(AcpThreadEvent::Refusal);
                        }

                        if cx.has_flag::<AcpBetaFeatureFlag>()
                            && let Some(response_usage) = &r.usage
                        {
                            let usage = this.token_usage.get_or_insert_with(Default::default);
                            usage.input_tokens = response_usage.input_tokens;
                            usage.output_tokens = response_usage.output_tokens;
                            cx.emit(AcpThreadEvent::TokenUsageUpdated(
                                OutputTokensUpdate::LegacyAdd {
                                    activity_generation: this.activity.generation(),
                                    output_tokens: usage.output_tokens,
                                },
                            ));
                        }

                        if is_same_turn {
                            cx.emit(AcpThreadEvent::StatusChanged);
                        }
                        let stop_reason = stop_reason_from_v1(&r.stop_reason);
                        this.set_foreground_state(
                            acp_v2::StateUpdate::Idle(
                                acp_v2::IdleStateUpdate::new().stop_reason(stop_reason.clone()),
                            ),
                            cx,
                        );
                        cx.emit(AcpThreadEvent::Stopped {
                            activity_generation: this.activity.generation(),
                            activity_duration: this.activity.duration(),
                            stop_reason,
                        });
                        Ok(Some(r))
                    }
                    Err(e) => {
                        if is_same_turn {
                            cx.emit(AcpThreadEvent::StatusChanged);
                        }
                        this.flush_streaming_text(cx);
                        if is_same_turn {
                            this.cancel_pending_turn_entries(cx);
                        }
                        this.had_error = true;
                        cx.emit(AcpThreadEvent::Error);
                        log::error!("Error in run turn: {:?}", e);
                        Err(e)
                    }
                }
            })?
        };
        self.track_submission(id, None, cx, async move |thread, cx| {
            Ok(completion(thread, cx)
                .await?
                .map(SubmissionResponse::LegacyCompleted))
        })
    }

    /// Receipt sessions wait for reported Idle without withdrawing pending prompt RPCs.
    pub fn cancel(&mut self, cx: &mut Context<Self>) -> Task<Result<()>> {
        match self.request_cancellation(RequestPermissionOutcome::Cancelled, cx) {
            Ok(task) => task,
            Err(error) => Task::ready(Err(error)),
        }
    }

    /// Asks the agent to stop one tool call, without cancelling the rest of the session's work.
    pub fn cancel_tool_call(
        &self,
        tool_call_id: &acp_v2::ToolCallId,
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        // The agent must only receive `tool_call/cancel` for a tool call that advertises it.
        let Some((_, call)) = self.tool_call(tool_call_id) else {
            return Task::ready(Err(anyhow!("Unknown tool call")));
        };
        if !call.can_cancel() {
            return Task::ready(Err(anyhow!("This tool call cannot be stopped")));
        }
        self.connection
            .cancel_tool_call(self.session_id(), tool_call_id, cx)
    }

    /// Asks the agent to stop an agent-run terminal's process. Its output stays available.
    pub fn cancel_terminal(
        &self,
        terminal_id: &acp_v2::TerminalId,
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        // The agent must only receive `terminal/cancel` for a terminal that advertises it.
        if !self
            .terminals
            .get(terminal_id)
            .is_some_and(|terminal| terminal.read(cx).can_cancel())
        {
            return Task::ready(Err(anyhow!("This terminal cannot be stopped")));
        }
        self.connection
            .cancel_terminal(self.session_id(), terminal_id, cx)
    }

    fn request_cancellation(
        &mut self,
        permission_outcome: RequestPermissionOutcome,
        cx: &mut Context<Self>,
    ) -> Result<Task<Result<()>>> {
        anyhow::ensure!(
            self.subagent_association
                .as_ref()
                .is_none_or(|association| association.can_cancel),
            "This subagent's parent does not allow cancelling it"
        );
        self.flush_streaming_text(cx);
        if self.uses_reported_activity() {
            self.activity.ensure_observable()?;
            if self.foreground_activity() != ForegroundActivity::Idle
                || self.has_unsettled_submissions()
            {
                self.connection.cancel(self.session_id(), cx)?;
            }
            if self.connection.answers_permission_requests_on_cancel() {
                if self.subagent_association.is_some() {
                    // The subagents RFD has a cancelled subagent answer all of its permission
                    // requests, including the tool-call ones a v1 subagent makes.
                    let request_ids = self.permission_requests.keys().copied().collect::<Vec<_>>();
                    for id in request_ids {
                        self.cancel_permission_request(id, cx);
                    }
                } else {
                    self.cancel_generic_permission_requests(cx);
                }
            }
            let receiver = self.activity.wait_for_idle()?;
            let Some(receiver) = receiver else {
                return Ok(Task::ready(Ok(())));
            };
            return Ok(cx.spawn(async move |_, _| {
                receiver
                    .await
                    .context("Agent connection lost before cancellation was confirmed")
            }));
        }
        if self.running_turn.is_some() {
            self.connection.cancel(self.session_id(), cx)?;
        }
        self.cancel_outstanding_elicitations(cx);

        let Some(turn) = self.running_turn.take() else {
            let request_ids = self.permission_requests.keys().copied().collect::<Vec<_>>();
            for id in request_ids {
                self.cancel_permission_request_with_outcome(id, permission_outcome.clone(), cx);
            }
            return Ok(Task::ready(Ok(())));
        };
        self.shrink_message_source_capacity(turn.first_entry_index);
        self.mark_pending_entries_as_canceled(permission_outcome, cx);
        self.set_foreground_state(
            acp_v2::StateUpdate::Idle(acp_v2::IdleStateUpdate::new()),
            cx,
        );
        cx.emit(AcpThreadEvent::StatusChanged);

        // Wait for the send task to complete
        Ok(cx.background_spawn(async move {
            turn.send_task.await;
            Ok(())
        }))
    }

    fn update_idle_sleep_prevention(&mut self, cx: &mut Context<Self>) {
        if !AgentSettings::get_global(cx).prevent_idle_sleep
            || self.foreground_activity() != ForegroundActivity::Running
        {
            self.idle_sleep_prevention = IdleSleepPrevention::Inactive;
            return;
        }

        if !matches!(self.idle_sleep_prevention, IdleSleepPrevention::Inactive) {
            return;
        }

        let acquisition = cx.prevent_idle_sleep("Agent thread in progress");
        self.idle_sleep_prevention = IdleSleepPrevention::Acquiring {
            _task: cx.spawn(async move |thread, cx| {
                let result = acquisition.await;
                thread
                    .update(cx, |thread, _| {
                        thread.idle_sleep_prevention = match result {
                            Ok(guard) => IdleSleepPrevention::Active { _guard: guard },
                            Err(error) => {
                                log::error!("Failed to prevent idle sleep: {error:#}");
                                IdleSleepPrevention::Failed
                            }
                        };
                    })
                    .log_err();
            }),
        };
    }

    fn cleanup_local_turn(
        &mut self,
        reason: &acp_v2::StopReason,
        is_same_turn: bool,
        cx: &mut Context<Self>,
    ) {
        match reason {
            acp_v2::StopReason::Cancelled
            | acp_v2::StopReason::MaxTokens
            | acp_v2::StopReason::Error(_)
                if is_same_turn =>
            {
                self.cancel_pending_turn_entries(cx);
            }
            acp_v2::StopReason::Refusal => {
                self.had_error = true;
                if is_same_turn {
                    self.cancel_generic_permission_requests(cx);
                }
                if let Some((user_message_index, _)) = self.last_user_message() {
                    // Completed tool output makes this a tool-result refusal, not a
                    // refusal of the user's prompt, so its history must be retained.
                    let has_completed_tool_output = self.entries.iter()
                        .skip(user_message_index + 1)
                        .any(|entry| matches!(entry, AgentThreadEntry::ToolCall(call)
                            if call.status() == ToolCallStatus::Completed && call.raw_output.is_some()));
                    if !has_completed_tool_output {
                        let range = user_message_index..self.entries.len();
                        if range.start < range.end {
                            self.truncate_entries(user_message_index, cx);
                            cx.emit(AcpThreadEvent::EntriesRemoved(range));
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn cancel_pending_turn_entries(&mut self, cx: &mut Context<Self>) {
        self.mark_pending_entries_as_canceled(RequestPermissionOutcome::Cancelled, cx);
        self.cancel_outstanding_elicitations(cx);
    }

    pub fn cancel_local_actions(
        &mut self,
        outcome: RequestPermissionOutcome,
        cx: &mut Context<Self>,
    ) {
        self.mark_pending_entries_as_canceled(outcome.clone(), cx);
        let request_ids = self.permission_requests.keys().copied().collect::<Vec<_>>();
        for id in request_ids {
            self.cancel_permission_request_with_outcome(id, outcome.clone(), cx);
        }
        self.cancel_outstanding_elicitations(cx);
    }

    fn mark_pending_entries_as_canceled(
        &mut self,
        permission_outcome: RequestPermissionOutcome,
        cx: &mut Context<Self>,
    ) {
        let mut canceled_requests = Vec::new();
        for (ix, entry) in self.entries.iter_mut().enumerate() {
            match entry {
                AgentThreadEntry::ToolCall(call) => {
                    let cancel = matches!(
                        call.status(),
                        ToolCallStatus::Pending
                            | ToolCallStatus::WaitingForConfirmation
                            | ToolCallStatus::InProgress
                    );
                    if cancel {
                        call.set_local_status(ToolCallStatus::Canceled);
                        if let Some(id) = call.authorization.take() {
                            canceled_requests.push(id);
                        }
                        cx.emit(AcpThreadEvent::EntryUpdated(ix));
                    }
                }
                AgentThreadEntry::ContextCompaction(compaction) => {
                    if compaction.status == ContextCompactionStatus::InProgress {
                        compaction.status = ContextCompactionStatus::Canceled;
                        cx.emit(AcpThreadEvent::EntryUpdated(ix));
                    }
                }
                _ => {}
            }
        }
        for id in canceled_requests {
            self.resolve_permission_request(id, permission_outcome.clone(), cx);
        }
        self.cancel_generic_permission_requests(cx);
    }

    pub fn cancel_outstanding_elicitations(&mut self, cx: &mut Context<Self>) {
        for ix in 0..self.entries.len() {
            let Some(AgentThreadEntry::Elicitation(elicitation_id)) = self.entries.get(ix) else {
                continue;
            };
            if let Some(change) = self.elicitations.cancel_elicitation_by_id(elicitation_id) {
                Self::emit_elicitation_change(ix, elicitation_id, change, cx);
            }
        }
    }

    /// Rewinds this thread to before the entry at `index`, removing it and all
    /// subsequent entries while rejecting any action_log changes made from that point.
    pub fn rewind(
        &mut self,
        client_id: ClientUserMessageId,
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        let Some(truncate) = self.connection.truncate(self.session_id(), cx) else {
            return Task::ready(Err(anyhow!("not supported")));
        };

        self.flush_streaming_text(cx);
        let telemetry = ActionLogTelemetry::from(&*self);
        cx.spawn(async move |this, cx| {
            cx.update(|cx| truncate.run(client_id.clone(), cx)).await?;
            this.update(cx, |this, cx| {
                this.flush_streaming_text(cx);
                if let Some((ix, _)) = this.user_message_mut(&client_id) {
                    // Collect all terminals from entries that will be removed
                    let terminals_to_remove: Vec<acp_v2::TerminalId> = this.entries[ix..]
                        .iter()
                        .flat_map(|entry| entry.terminals())
                        .filter_map(|terminal| terminal.read(cx).id().clone().into())
                        .collect();

                    let range = ix..this.entries.len();
                    this.truncate_entries(ix, cx);
                    cx.emit(AcpThreadEvent::EntriesRemoved(range));

                    // Kill and remove the terminals
                    for terminal_id in terminals_to_remove {
                        if let Some(terminal) = this.terminals.remove(&terminal_id) {
                            terminal.update(cx, |terminal, cx| {
                                terminal.kill(cx);
                            });
                        }
                    }
                }
                this.action_log().update(cx, |action_log, cx| {
                    action_log.reject_all_edits(Some(telemetry), cx)
                })
            })?
            .await;
            Ok(())
        })
    }

    fn last_user_message(&mut self) -> Option<(usize, &mut UserMessage)> {
        self.entries
            .iter_mut()
            .enumerate()
            .rev()
            .find_map(|(ix, entry)| {
                if let AgentThreadEntry::UserMessage(message) = entry {
                    Some((ix, message))
                } else {
                    None
                }
            })
    }

    fn user_message_mut(
        &mut self,
        client_id: &ClientUserMessageId,
    ) -> Option<(usize, &mut UserMessage)> {
        self.entries.iter_mut().enumerate().find_map(|(ix, entry)| {
            if let AgentThreadEntry::UserMessage(message) = entry {
                if message.client_id.as_ref() == Some(client_id) {
                    Some((ix, message))
                } else {
                    None
                }
            } else {
                None
            }
        })
    }

    pub fn read_text_file(
        &self,
        path: PathBuf,
        line: Option<u32>,
        limit: Option<u32>,
        reuse_shared_snapshot: bool,
        cx: &mut Context<Self>,
    ) -> Task<Result<String, acp_v1::Error>> {
        // Args are 1-based, move to 0-based
        let line = line.unwrap_or_default().saturating_sub(1);
        let limit = limit.unwrap_or(u32::MAX);
        let project = self.project.clone();
        let action_log = self.action_log.clone();
        let should_update_agent_location = self.parent_session_id.is_none();
        cx.spawn(async move |this, cx| {
            let load = project.update(cx, |project, cx| {
                let path = project
                    .project_path_for_absolute_path(&path, cx)
                    .ok_or_else(|| {
                        acp_v1::Error::resource_not_found(Some(path.display().to_string()))
                    })?;
                Ok::<_, acp_v1::Error>(project.open_buffer(path, cx))
            })?;

            let buffer = load.await?;

            let snapshot = if reuse_shared_snapshot {
                this.read_with(cx, |this, _| {
                    this.shared_buffers.get(&buffer.clone()).cloned()
                })
                .log_err()
                .flatten()
            } else {
                None
            };

            let snapshot = if let Some(snapshot) = snapshot {
                snapshot
            } else {
                action_log.update(cx, |action_log, cx| {
                    action_log.buffer_read(buffer.clone(), cx);
                });

                let snapshot = buffer.update(cx, |buffer, _| buffer.snapshot());
                this.update(cx, |this, _| {
                    this.shared_buffers.insert(buffer.clone(), snapshot.clone());
                })?;
                snapshot
            };

            let max_point = snapshot.max_point();
            let start_position = Point::new(line, 0);

            if start_position > max_point {
                return Err(acp_v1::Error::invalid_params().data(format!(
                    "Attempting to read beyond the end of the file, line {}:{}",
                    max_point.row + 1,
                    max_point.column
                )));
            }

            let start = snapshot.anchor_before(start_position);
            let end = snapshot.anchor_before(Point::new(line.saturating_add(limit), 0));

            if should_update_agent_location {
                project.update(cx, |project, cx| {
                    project.set_agent_location(
                        Some(AgentLocation {
                            buffer: buffer.downgrade(),
                            position: start,
                        }),
                        cx,
                    );
                });
            }

            Ok(snapshot.text_for_range(start..end).collect::<String>())
        })
    }

    pub fn write_text_file(
        &self,
        path: PathBuf,
        content: String,
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        let project = self.project.clone();
        let action_log = self.action_log.clone();
        let should_update_agent_location = self.parent_session_id.is_none();
        cx.spawn(async move |this, cx| {
            let load = project.update(cx, |project, cx| {
                let path = project
                    .project_path_for_absolute_path(&path, cx)
                    .context("invalid path")?;
                anyhow::Ok(project.open_buffer(path, cx))
            });
            let buffer = load?.await?;
            let snapshot = this.update(cx, |this, cx| {
                this.shared_buffers
                    .get(&buffer)
                    .cloned()
                    .unwrap_or_else(|| buffer.read(cx).snapshot())
            })?;
            let edits = cx
                .background_executor()
                .spawn(async move {
                    let old_text = snapshot.text();
                    text_diff(old_text.as_str(), &content)
                        .into_iter()
                        .map(|(range, replacement)| {
                            (snapshot.anchor_range_inside(range), replacement)
                        })
                        .collect::<Vec<_>>()
                })
                .await;

            if should_update_agent_location {
                project.update(cx, |project, cx| {
                    project.set_agent_location(
                        Some(AgentLocation {
                            buffer: buffer.downgrade(),
                            position: edits
                                .last()
                                .map(|(range, _)| range.end)
                                .unwrap_or(Anchor::min_for_buffer(buffer.read(cx).remote_id())),
                        }),
                        cx,
                    );
                });
            }

            let format_on_save = cx.update(|cx| {
                action_log.update(cx, |action_log, cx| {
                    action_log.buffer_read(buffer.clone(), cx);
                });

                let format_on_save = buffer.update(cx, |buffer, cx| {
                    buffer.start_transaction();
                    buffer.edit(edits, None, cx);
                    buffer.end_transaction_with_source(BufferEditSource::Agent, cx);

                    let settings =
                        language::language_settings::LanguageSettings::for_buffer(buffer, cx);

                    settings.format_on_save != FormatOnSave::Off
                });
                action_log.update(cx, |action_log, cx| {
                    action_log.buffer_edited(buffer.clone(), cx);
                });
                format_on_save
            });

            if format_on_save {
                let format_task = project.update(cx, |project, cx| {
                    project.format(
                        HashSet::from_iter([buffer.clone()]),
                        LspFormatTarget::Buffers,
                        false,
                        FormatTrigger::Save,
                        cx,
                    )
                });
                format_task.await.log_err();

                action_log.update(cx, |action_log, cx| {
                    action_log.buffer_edited(buffer.clone(), cx);
                });
            }

            project
                .update(cx, |project, cx| project.save_buffer(buffer, cx))
                .await
        })
    }

    pub fn create_terminal(
        &self,
        command: String,
        args: Vec<String>,
        extra_env: Vec<(String, String)>,
        cwd: Option<PathBuf>,
        output_byte_limit: Option<u64>,
        sandbox_wrap: Option<SandboxWrap>,
        cx: &mut Context<Self>,
    ) -> Task<Result<Entity<Terminal>>> {
        let task = create_runtime_terminal(
            self.project.clone(),
            command,
            args,
            extra_env,
            cwd,
            output_byte_limit,
            sandbox_wrap,
            cx,
        );
        cx.spawn(async move |this, cx| {
            let terminal = task.await?;
            this.update(cx, |this, cx| {
                this.terminals
                    .insert(terminal.read(cx).id().clone(), terminal.clone());
                terminal
            })
        })
    }

    pub fn kill_terminal(
        &mut self,
        terminal_id: acp_v2::TerminalId,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        self.terminals
            .get(&terminal_id)
            .ok_or_else(|| terminal_not_found(&terminal_id))?
            .update(cx, |terminal, cx| {
                terminal.kill(cx);
            });

        Ok(())
    }

    pub fn release_terminal(
        &mut self,
        terminal_id: acp_v2::TerminalId,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        self.terminals
            .remove(&terminal_id)
            .ok_or_else(|| terminal_not_found(&terminal_id))?
            .update(cx, |terminal, cx| {
                terminal.kill(cx);
            });

        Ok(())
    }

    pub fn terminal(&self, terminal_id: acp_v2::TerminalId) -> Result<Entity<Terminal>> {
        self.terminals
            .get(&terminal_id)
            .cloned()
            .ok_or_else(|| terminal_not_found(&terminal_id))
    }

    pub fn to_markdown(&self, cx: &App) -> String {
        self.entries
            .iter()
            .map(|entry| match entry {
                AgentThreadEntry::Elicitation(elicitation_id) => self
                    .elicitations
                    .elicitation(elicitation_id)
                    .map(|(_, elicitation)| {
                        format!("## Input Requested\n\n{}\n\n", elicitation.request.message)
                    })
                    .unwrap_or_else(|| entry.to_markdown(cx)),
                _ => entry.to_markdown(cx),
            })
            .collect()
    }

    pub fn emit_load_error(&mut self, error: LoadError, cx: &mut Context<Self>) {
        cx.emit(AcpThreadEvent::LoadError(error));
    }

    pub fn register_terminal_created(
        &mut self,
        terminal_id: acp_v2::TerminalId,
        command_label: String,
        working_dir: Option<PathBuf>,
        output_byte_limit: Option<u64>,
        terminal: Entity<::terminal::Terminal>,
        cx: &mut Context<Self>,
    ) -> Entity<Terminal> {
        let language_registry = self.project.read(cx).languages().clone();

        let entity = cx.new(|cx| {
            Terminal::new(
                terminal_id.clone(),
                &command_label,
                working_dir.clone(),
                output_byte_limit.map(|l| l as usize),
                terminal,
                language_registry,
                // External terminal providers manage their own sandboxing
                // (if any). We don't wrap their commands.
                None,
                cx,
            )
        });
        self.terminals.insert(terminal_id.clone(), entity.clone());
        entity
    }

    pub fn mark_as_subagent_output(&mut self, cx: &mut Context<Self>) {
        for entry in self.entries.iter_mut().rev() {
            if let AgentThreadEntry::AssistantMessage(assistant_message) = entry {
                assistant_message.is_subagent_output = true;
                cx.notify();
                return;
            }
        }
    }

    pub fn upsert_display_terminal(
        &mut self,
        terminal_id: acp_v2::TerminalId,
        patch: DisplayTerminalPatch,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        let terminal = self.ensure_display_terminal(terminal_id, cx);
        terminal.update(cx, |terminal, cx| terminal.apply_display_patch(patch, cx))
    }

    pub fn append_display_terminal_output(
        &mut self,
        terminal_id: acp_v2::TerminalId,
        data: &[u8],
        cx: &mut Context<Self>,
    ) -> Result<()> {
        let terminal = self.ensure_display_terminal(terminal_id, cx);
        terminal.update(cx, |terminal, cx| terminal.append_display_bytes(data, cx))
    }

    fn ensure_display_terminal(
        &mut self,
        terminal_id: acp_v2::TerminalId,
        cx: &mut Context<Self>,
    ) -> Entity<Terminal> {
        if let Some(terminal) = self.terminals.get(&terminal_id) {
            return terminal.clone();
        }
        let builder = ::terminal::TerminalBuilder::new_display_only(
            ::terminal::terminal_settings::CursorShape::default(),
            ::terminal::terminal_settings::AlternateScroll::On,
            None,
            0,
            cx.background_executor(),
            self.project.read(cx).path_style(cx),
        );
        let terminal = cx.new(|cx| builder.subscribe(cx));
        self.register_display_terminal(terminal_id, None, None, None, terminal, cx)
    }

    fn register_display_terminal(
        &mut self,
        terminal_id: acp_v2::TerminalId,
        command: Option<&str>,
        cwd: Option<PathBuf>,
        output_byte_limit: Option<u64>,
        terminal: Entity<::terminal::Terminal>,
        cx: &mut Context<Self>,
    ) -> Entity<Terminal> {
        let language_registry = self.project.read(cx).languages().clone();
        let entity = cx.new(|cx| {
            Terminal::new_display(
                terminal_id.clone(),
                command,
                cwd,
                output_byte_limit.map(|limit| limit as usize),
                terminal,
                language_registry,
                cx,
            )
        });
        // Provider state can change without another tool-call update.
        cx.observe(&entity, |this, terminal, cx| {
            for (index, entry) in this.entries.iter().enumerate() {
                if entry
                    .terminals()
                    .any(|entry_terminal| entry_terminal == &terminal)
                {
                    cx.emit(AcpThreadEvent::EntryUpdated(index));
                }
            }
            cx.notify();
        })
        .detach();
        self.terminals.insert(terminal_id.clone(), entity.clone());
        if let Some(chunks) = self.pending_terminal_output.remove(&terminal_id) {
            entity.update(cx, |terminal, cx| {
                for data in chunks {
                    terminal.write_display_output(&data, cx);
                }
            });
        }
        if let Some(status) = self.pending_terminal_exit.remove(&terminal_id) {
            entity.update(cx, |terminal, cx| terminal.finish_display(status, cx));
        }
        entity
    }

    pub fn on_terminal_provider_event(
        &mut self,
        event: TerminalProviderEvent,
        cx: &mut Context<Self>,
    ) {
        match event {
            TerminalProviderEvent::Created {
                terminal_id,
                label,
                cwd,
                output_byte_limit,
                terminal,
            } => {
                let terminal_id = acp_v2::TerminalId::new(terminal_id.0);
                let entity = self
                    .terminals
                    .get(&terminal_id)
                    .cloned()
                    .unwrap_or_else(|| {
                        self.register_display_terminal(
                            terminal_id,
                            Some(&label),
                            cwd.clone(),
                            output_byte_limit,
                            terminal,
                            cx,
                        )
                    });
                // A prior reference may already have a view; keep its renderer when metadata arrives.
                entity.update(cx, |terminal, cx| {
                    terminal.initialize_legacy_display(
                        &label,
                        cwd,
                        output_byte_limit.map(|limit| limit as usize),
                        cx,
                    );
                });

                cx.notify();
            }
            TerminalProviderEvent::Output { terminal_id, data } => {
                let terminal_id = acp_v2::TerminalId::new(terminal_id.0);
                if let Some(entity) = self.terminals.get(&terminal_id) {
                    entity.update(cx, |term, cx| {
                        term.write_display_output(&data, cx);
                    });
                } else {
                    self.pending_terminal_output
                        .entry(terminal_id)
                        .or_default()
                        .push(data);
                }
            }
            TerminalProviderEvent::TitleChanged { terminal_id, title } => {
                let terminal_id = acp_v2::TerminalId::new(terminal_id.0);
                if let Some(entity) = self.terminals.get(&terminal_id) {
                    entity.update(cx, |term, cx| {
                        term.inner().update(cx, |inner, cx| {
                            inner.breadcrumb_text = title;
                            cx.emit(::terminal::Event::BreadcrumbsChanged);
                        })
                    });
                }
            }
            TerminalProviderEvent::Exit {
                terminal_id,
                status,
            } => {
                let terminal_id = acp_v2::TerminalId::new(terminal_id.0);
                if let Some(entity) = self.terminals.get(&terminal_id) {
                    entity.update(cx, |term, cx| term.finish_display(status, cx));
                } else {
                    self.pending_terminal_exit
                        .entry(terminal_id)
                        .or_insert(status);
                }
            }
        }
    }
}

fn markdown_for_raw_output(
    raw_output: &serde_json::Value,
    language_registry: &Arc<LanguageRegistry>,
    cx: &mut App,
) -> Option<Entity<Markdown>> {
    let text = raw_output_text(raw_output)?;
    Some(cx.new(|cx| Markdown::new_deferred(text, Some(language_registry.clone()), cx)))
}

fn raw_output_text(raw_output: &serde_json::Value) -> Option<String> {
    match raw_output {
        serde_json::Value::Null => None,
        serde_json::Value::Bool(value) => Some(value.to_string()),
        serde_json::Value::Number(value) => Some(value.to_string()),
        serde_json::Value::String(value) => Some(value.clone()),
        value => {
            let pretty_json = to_string_pretty(value).unwrap_or_else(|_| value.to_string());
            Some(format!("```json\n{}\n```", pretty_json))
        }
    }
}

fn update_markdown_in_place(markdown: &Entity<Markdown>, text: &str, cx: &mut App) {
    markdown.update(cx, |markdown, cx| {
        match text.strip_prefix(markdown.source().as_ref()) {
            Some("") => {}
            Some(suffix) => markdown.append(suffix, cx),
            None => markdown.reset(text.to_owned().into(), cx),
        }
    });
}

#[cfg(test)]
#[path = "acp_thread_tests.rs"]
mod tests;

pub fn create_runtime_terminal(
    project: Entity<Project>,
    command: String,
    args: Vec<String>,
    extra_env: Vec<(String, String)>,
    cwd: Option<PathBuf>,
    output_byte_limit: Option<u64>,
    sandbox_wrap: Option<SandboxWrap>,
    cx: &mut App,
) -> Task<Result<Entity<Terminal>>> {
    let env = match &cwd {
        Some(dir) => project.update(cx, |project, cx| {
            project.environment().update(cx, |env, cx| {
                env.directory_environment(dir.as_path().into(), cx)
            })
        }),
        None => Task::ready(None).shared(),
    };
    let env = cx.spawn(async move |_| {
        let mut env = env.await.unwrap_or_default();

        disable_pagers_through_env(&mut env);

        for (name, value) in extra_env {
            env.insert(name, value);
        }
        env
    });

    let language_registry = project.read(cx).languages().clone();
    let is_windows = project.read(cx).path_style(cx).is_windows();
    // Headless hosts (e.g. the eval CLI) have no controlling TTY, so PTY
    // setup fails with `ENOTTY`. Run the command non-interactively and
    // without a PTY in that case.
    let headless = HeadlessTerminal::is_enabled(cx);

    let terminal_id = acp_v2::TerminalId::new(Uuid::new_v4().to_string());
    cx.spawn({
            let terminal_id = terminal_id.clone();
            async move |cx| {
                let env = env.await;
                let shell = project
                    .update(cx, |project, cx| {
                        project
                            .remote_client()
                            .and_then(|r| r.read(cx).default_system_shell())
                    })
                    .unwrap_or_else(|| get_default_system_shell_preferring_bash());

                // The sandbox owns the network proxy (for restricted-network
                // policies) and injects the child's proxy env vars, returning
                // the env to spawn with. On Windows, restricted host access is
                // rejected inside the sandbox before command preparation.
                #[cfg(target_os = "windows")]
                let (task_command, task_args, task_env, sandbox, spawn_cwd) =
                    if sandbox_wrap.is_some() {
                        let (task_command, task_args) = task::ShellBuilder::new(
                            &Shell::Program("/bin/sh".to_string()),
                            false,
                        )
                        .non_interactive()
                        .redirect_stdin_to_dev_null()
                        .build(Some(command.clone()), &args);
                        let wrap = cx.background_spawn(prepare_sandbox_wrap(
                            task_command,
                            task_args,
                            cwd.clone(),
                            sandbox_wrap,
                            env,
                        ));
                        let timeout = cx.background_executor().timer(WSL_SANDBOX_WRAP_TIMEOUT);
                        let (task_command, task_args, task_env, sandbox) = futures::select_biased! {
                            result = wrap.fuse() => result?,
                            _ = timeout.fuse() => return Err(anyhow::Error::new(
                                sandbox::SandboxError::WslUnavailable(format!(
                                    "WSL did not respond within {} seconds while preparing the sandboxed command",
                                    WSL_SANDBOX_WRAP_TIMEOUT.as_secs()
                                )),
                            )),
                        };
                        (task_command, task_args, task_env, sandbox, None)
                    } else {
                        // No sandbox wrap means we're running unsandboxed, and
                        // on Windows that deliberately changes the shell: the
                        // sandboxed path runs under WSL's Linux bash, but this
                        // fallback uses the host's `shell` against the native cwd.
                        let mut builder = ShellBuilder::new(&Shell::Program(shell), is_windows);
                        if headless {
                            builder = builder.non_interactive();
                        }
                        let (task_command, task_args) = builder
                            .redirect_stdin_to_dev_null()
                            .build(Some(command.clone()), &args);
                        (task_command, task_args, env, None, cwd.clone())
                    };

                #[cfg(not(target_os = "windows"))]
                let (task_command, task_args, task_env, sandbox, spawn_cwd) = {
                    let mut builder = ShellBuilder::new(&Shell::Program(shell), is_windows);
                    if headless || (cfg!(target_os = "linux") && sandbox_wrap.is_some()) {
                        builder = builder.non_interactive();
                    }
                    let (task_command, task_args) = builder
                        .redirect_stdin_to_dev_null()
                        .build(Some(command.clone()), &args);
                    let (task_command, task_args, task_env, sandbox) = cx
                        .background_spawn(prepare_sandbox_wrap(
                            task_command,
                            task_args,
                            cwd.clone(),
                            sandbox_wrap,
                            env,
                        ))
                        .await?;
                    (task_command, task_args, task_env, sandbox, cwd.clone())
                };
                let terminal = project
                    .update(cx, |project, cx| {
                        project.create_terminal_task(
                            task::SpawnInTerminal {
                                command: Some(task_command),
                                args: task_args,
                                cwd: spawn_cwd,
                                env: task_env,
                                ..Default::default()
                            },
                            cx,
                        )
                    })
                    .await?;

                anyhow::Ok(cx.new(|cx: &mut Context<Terminal>| {
                    cx.on_release(|terminal,cx|terminal.kill(cx)).detach();
                    Terminal::new(
                        terminal_id,
                        &format!("{} {}", command, args.join(" ")),
                        cwd,
                        output_byte_limit.map(|l| l as usize),
                        terminal,
                        language_registry,
                        sandbox,
                        cx,
                    )
                }))
            }
        })
}
