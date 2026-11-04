use std::{
    cell::RefCell,
    collections::{BTreeMap, VecDeque},
    rc::Rc,
    sync::Arc,
};

use action_log::ActionLog;
use agent_client_protocol::v2::schema as acp;
use agent_settings::{AgentProfileId, AgentSettings};
use anyhow::{Context as _, Result, anyhow, ensure};
use futures::{
    channel::oneshot,
    future::{Either, select},
};
use gpui::{App, AppContext as _, Entity, SharedString, Task};
use language::Buffer;
use language_model::LanguageModelProviderId;
use project::{Project, ProjectPath, trusted_worktrees::TrustedWorktrees};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use settings::Settings as _;

use crate::{
    AgentTool, AnyAgentTool, ToolCallEventStream, ToolInput, ToolPermissionContext, tools::*,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RuntimeBufferSnapshot {
    pub text: String,
    pub dirty: bool,
}

enum ProjectRequest {
    Read {
        path: ProjectPath,
        response: oneshot::Sender<Result<RuntimeBufferSnapshot>>,
    },
    Replace {
        path: ProjectPath,
        expected: String,
        replacement: String,
        save: bool,
        response: oneshot::Sender<Result<RuntimeBufferSnapshot>>,
    },
    Diagnostics {
        path: ProjectPath,
        response: oneshot::Sender<Result<(usize, usize)>>,
    },
}

#[derive(Clone)]
pub struct RuntimeProjectClient {
    sender: async_channel::Sender<ProjectRequest>,
    maximum_bytes: usize,
}

pub struct RuntimeProjectBridge {
    pub client: RuntimeProjectClient,
    pub task: Task<()>,
}

impl RuntimeProjectBridge {
    pub fn new(
        project: Entity<Project>,
        capacity: usize,
        maximum_bytes: usize,
        cx: &mut App,
    ) -> Result<Self> {
        ensure!(
            capacity > 0 && maximum_bytes > 0,
            "project bridge limits must be positive"
        );
        let (sender, requests) = async_channel::bounded(capacity);
        let task = cx.spawn(async move |cx| {
            while let Ok(request) = requests.recv().await {
                futures_lite::future::yield_now().await;
                match request {
                    ProjectRequest::Read { path, response } => {
                        if response.is_canceled() { continue; }
                        let result = async {
                            validate_project_path(&project, &path, false, cx)?;
                            let buffer = project.update(cx, |project, cx| project.open_buffer(path, cx)).await?;
                            buffer.read_with(cx, |buffer, _| buffer_projection(buffer, maximum_bytes))
                        }.await;
                        if response.send(result).is_err() { log::debug!("runtime buffer reader detached"); }
                    }
                    ProjectRequest::Replace { path, expected, replacement, save, response } => {
                        if response.is_canceled() { continue; }
                        let result = async {
                            validate_project_path(&project, &path, true, cx)?;
                            let buffer = project.update(cx, |project, cx| project.open_buffer(path.clone(), cx)).await?;
                            if response.is_canceled() { return Err(anyhow!("buffer edit was cancelled before application")); }
                            validate_project_path(&project, &path, true, cx)?;
                            buffer.update(cx, |buffer, cx| {
                                ensure!(buffer.as_rope().to_string() == expected, "buffer changed since the agent read it; read it again before editing");
                                buffer.set_text(replacement, cx);
                                Ok::<_, anyhow::Error>(())
                            })?;
                            if save { project.update(cx, |project, cx| project.save_buffer(buffer.clone(), cx)).await?; }
                            buffer.read_with(cx, |buffer, _| buffer_projection(buffer, maximum_bytes))
                        }.await;
                        if response.send(result).is_err() { log::debug!("runtime edit requester detached after dispatch"); }
                    }
                    ProjectRequest::Diagnostics { path, response } => {
                        if response.is_canceled() { continue; }
                        let result = validate_project_path(&project, &path, false, cx).map(|()| {
                            project.read_with(cx, |project, cx| {
                                let summary = project.diagnostic_summary_for_path(&path, cx);
                                (summary.error_count, summary.warning_count)
                            })
                        });
                        if response.send(result).is_err() { log::debug!("runtime diagnostics requester detached"); }
                    }
                }
            }
        });
        Ok(Self {
            client: RuntimeProjectClient {
                sender,
                maximum_bytes,
            },
            task,
        })
    }
}

fn validate_project_path(
    project: &Entity<Project>,
    path: &ProjectPath,
    edit: bool,
    cx: &mut gpui::AsyncApp,
) -> Result<()> {
    project.read_with(cx, |project, cx| {
        ensure!(
            AgentSettings::get_global(cx).enabled(cx),
            "AI features are disabled in Zed settings"
        );
        ensure!(
            project.worktree_for_id(path.worktree_id, cx).is_some(),
            "worktree is no longer part of this project"
        );
        if edit {
            ensure!(
                !TrustedWorktrees::has_restricted_worktrees(&project.worktree_store(), cx),
                "cannot edit a restricted workspace"
            );
        }
        Ok(())
    })
}

fn buffer_projection(buffer: &Buffer, maximum_bytes: usize) -> Result<RuntimeBufferSnapshot> {
    ensure!(
        buffer.len() <= maximum_bytes,
        "buffer exceeds the runtime projection byte limit"
    );
    Ok(RuntimeBufferSnapshot {
        text: buffer.as_rope().to_string(),
        dirty: buffer.is_dirty(),
    })
}

impl RuntimeProjectClient {
    async fn response<T>(&self, response: oneshot::Receiver<Result<T>>) -> Result<T> {
        let closed = self.sender.closed();
        futures::pin_mut!(closed);
        match select(response, closed).await {
            Either::Left((result, _)) => {
                result.context("native project operation was interrupted")?
            }
            Either::Right(_) => Err(anyhow!("native project bridge stopped")),
        }
    }

    pub async fn read_buffer(&self, path: ProjectPath) -> Result<RuntimeBufferSnapshot> {
        let (response, receiver) = oneshot::channel();
        self.sender
            .send(ProjectRequest::Read { path, response })
            .await
            .map_err(|_| anyhow!("native project bridge stopped"))?;
        self.response(receiver).await
    }

    // Only authorized native job adapters call this; exposing it as a model tool would bypass approvals.
    pub async fn replace_authorized_buffer(
        &self,
        path: ProjectPath,
        expected: String,
        replacement: String,
        save: bool,
    ) -> Result<RuntimeBufferSnapshot> {
        ensure!(
            expected.len() <= self.maximum_bytes && replacement.len() <= self.maximum_bytes,
            "buffer edit exceeds the runtime projection byte limit"
        );
        let (response, receiver) = oneshot::channel();
        self.sender
            .send(ProjectRequest::Replace {
                path,
                expected,
                replacement,
                save,
                response,
            })
            .await
            .map_err(|_| anyhow!("native project bridge stopped"))?;
        self.response(receiver).await
    }

    pub async fn diagnostics(&self, path: ProjectPath) -> Result<(usize, usize)> {
        let (response, receiver) = oneshot::channel();
        self.sender
            .send(ProjectRequest::Diagnostics { path, response })
            .await
            .map_err(|_| anyhow!("native project bridge stopped"))?;
        self.response(receiver).await
    }
}

struct CachedAction<T> {
    id: String,
    actor: String,
    bytes: usize,
    value: T,
}

struct ActionCache<T> {
    entries: VecDeque<CachedAction<T>>,
    bytes: usize,
    maximum_entries: usize,
    maximum_bytes: usize,
}

impl<T> ActionCache<T> {
    fn insert(&mut self, actor: String, value: T, bytes: usize) -> Result<String> {
        ensure!(
            bytes <= self.maximum_bytes,
            "code action exceeds the retained-byte limit"
        );
        while self.entries.len() >= self.maximum_entries
            || self.bytes.saturating_add(bytes) > self.maximum_bytes
        {
            let expired = self
                .entries
                .pop_front()
                .context("code-action cache limits are inconsistent")?;
            self.bytes = self.bytes.saturating_sub(expired.bytes);
        }
        let id = uuid::Uuid::new_v4().to_string();
        self.bytes += bytes;
        self.entries.push_back(CachedAction {
            id: id.clone(),
            actor,
            bytes,
            value,
        });
        Ok(id)
    }

    fn take(&mut self, actor: &str, id: &str) -> Result<T> {
        let index = self
            .entries
            .iter()
            .position(|entry| entry.id == id && entry.actor == actor)
            .context("code action is expired or belongs to another agent; request actions again")?;
        let entry = self
            .entries
            .remove(index)
            .context("code action disappeared")?;
        self.bytes = self.bytes.saturating_sub(entry.bytes);
        Ok(entry.value)
    }

    fn clear_agent(&mut self, actor: &str) {
        self.entries.retain(|entry| entry.actor != actor);
        self.bytes = self.entries.iter().map(|entry| entry.bytes).sum();
    }
}

struct RuntimeCodeAction {
    action: project::CodeAction,
    buffer: Entity<Buffer>,
    snapshot: language::BufferSnapshot,
}

#[derive(Clone)]
pub struct RuntimeCodeActions(Rc<RefCell<ActionCache<RuntimeCodeAction>>>);

impl RuntimeCodeActions {
    pub fn new(maximum_entries: usize, maximum_bytes: usize) -> Result<Self> {
        ensure!(
            maximum_entries > 0 && maximum_bytes > 0,
            "code action limits must be positive"
        );
        Ok(Self(Rc::new(RefCell::new(ActionCache {
            entries: VecDeque::new(),
            bytes: 0,
            maximum_entries,
            maximum_bytes,
        }))))
    }

    pub fn clear_agent(&self, actor: &str) {
        self.0.borrow_mut().clear_agent(actor);
    }
}

pub fn runtime_project_tools(
    project: Entity<Project>,
    actor: &str,
    provider: &LanguageModelProviderId,
    profile_id: &AgentProfileId,
    actions: RuntimeCodeActions,
    cx: &mut App,
) -> Result<BTreeMap<SharedString, Arc<dyn AnyAgentTool>>> {
    let action_log = cx.new(|_| ActionLog::new(project.clone()));
    runtime_project_tools_with_log(
        project, actor, provider, profile_id, actions, action_log, cx,
    )
}

pub fn runtime_project_tools_with_log(
    project: Entity<Project>,
    actor: &str,
    provider: &LanguageModelProviderId,
    profile_id: &AgentProfileId,
    actions: RuntimeCodeActions,
    action_log: Entity<ActionLog>,
    cx: &mut App,
) -> Result<BTreeMap<SharedString, Arc<dyn AnyAgentTool>>> {
    ensure!(
        !actor.is_empty() && actor.len() <= 256,
        "invalid runtime agent identity"
    );
    let settings = AgentSettings::get_global(cx);
    ensure!(
        settings.enabled(cx),
        "AI features are disabled in Zed settings"
    );
    let profile = settings
        .profiles
        .get(profile_id)
        .context("runtime agent profile is unavailable")?
        .clone();
    let restricted =
        TrustedWorktrees::has_restricted_worktrees(&project.read(cx).worktree_store(), cx);
    let languages = project.read(cx).languages().clone();
    let tools: Vec<Arc<dyn AnyAgentTool>> = vec![
        EditFileTool::new_runtime(project.clone(), action_log.clone(), languages.clone()).erase(),
        WriteFileTool::new_runtime(project.clone(), action_log.clone(), languages).erase(),
        CopyPathTool::new(project.clone()).erase(),
        CreateDirectoryTool::new(project.clone()).erase(),
        DeletePathTool::new(project.clone(), action_log.clone()).erase(),
        FetchTool::new(project.read(cx).client().http_client()).erase(),
        FindPathTool::new(project.clone()).erase(),
        GrepTool::new(project.clone()).erase(),
        ListDirectoryTool::new(project.clone()).erase(),
        MovePathTool::new(project.clone()).erase(),
        ReadFileTool::new(project.clone(), action_log, false).erase(),
        WebSearchTool.erase(),
        AskUserTool.erase(),
        DiagnosticsTool::new(project.clone()).erase(),
        FindReferencesTool::new(project.clone()).erase(),
        GoToDefinitionTool::new(project.clone()).erase(),
        RenameTool::new(project.clone()).erase(),
        RuntimeGetCodeActions {
            project: project.clone(),
            actor: actor.to_owned(),
            cache: actions.clone(),
        }
        .erase(),
        RuntimeApplyCodeAction {
            project: project.clone(),
            actor: actor.to_owned(),
            cache: actions,
        }
        .erase(),
    ];
    let mut tools: BTreeMap<SharedString, Arc<dyn AnyAgentTool>> = tools
        .into_iter()
        .filter(|tool| !restricted || tool.allow_in_restricted_mode())
        .filter(|tool| {
            tool.supports_provider(provider) && profile.is_tool_enabled(tool.name().as_ref())
        })
        .filter(|tool| crate::tools::tool_feature_flag_enabled(tool.name().as_ref(), cx))
        .map(|tool| {
            (
                crate::thread::provider_compatible_tool_name(tool.name().as_ref()).into(),
                tool,
            )
        })
        .collect();
    tools.extend(crate::runtime_terminal::runtime_terminal_tools(
        project, provider, profile_id, cx,
    )?);
    Ok(tools)
}

struct RuntimeGetCodeActions {
    project: Entity<Project>,
    actor: String,
    cache: RuntimeCodeActions,
}

impl AgentTool for RuntimeGetCodeActions {
    type Input = GetCodeActionsToolInput;
    type Output = String;
    const NAME: &'static str = "get_code_actions";
    fn kind() -> acp::ToolKind {
        acp::ToolKind::Search
    }
    fn initial_title(
        &self,
        _: Result<Self::Input, serde_json::Value>,
        _: &mut App,
    ) -> SharedString {
        "Get code actions".into()
    }
    fn run(
        self: Arc<Self>,
        input: ToolInput<Self::Input>,
        _: ToolCallEventStream,
        cx: &mut App,
    ) -> Task<Result<String, String>> {
        cx.spawn(async move |cx| {
            let input = input.recv().await.map_err(|error| error.to_string())?;
            let resolved = input.symbol.resolve(&self.project, cx).await?;
            let actions = self.project.update(cx, |project, cx| project.code_actions(&resolved.buffer, resolved.position..resolved.position, None, cx))
                .await.map_err(|error| error.to_string())?.unwrap_or_default();
            let snapshot = resolved.buffer.read_with(cx, |buffer, _| buffer.snapshot());
            self.cache.clear_agent(&self.actor);
            let mut output = Vec::new();
            for action in actions {
                let encoded = match &action.lsp_action {
                    project::LspAction::Action(action) => serde_json::to_vec(action),
                    project::LspAction::Command(command) => serde_json::to_vec(command),
                    project::LspAction::CodeLens(lens) => serde_json::to_vec(lens),
                }.map_err(|error| error.to_string())?;
                let bytes = snapshot.len().saturating_add(encoded.len());
                let title = action.lsp_action.title().to_owned();
                let id = self.cache.0.borrow_mut().insert(self.actor.clone(), RuntimeCodeAction { action, buffer: resolved.buffer.clone(), snapshot: snapshot.clone() }, bytes)
                    .map_err(|error| error.to_string())?;
                output.push((id, title));
            }
            let output = output.into_iter().filter(|(id, _)| self.cache.0.borrow().entries.iter().any(|entry| entry.id == *id))
                .map(|(id, title)| format!("{id}: {title}")).collect::<Vec<_>>();
            if output.is_empty() { Ok("No code actions available within the runtime cache budget.".into()) }
            else { Ok(format!("{}\nUse apply_code_action with the action_id shown above. IDs are scoped to this agent and expire when evicted.", output.join("\n"))) }
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
struct RuntimeApplyCodeActionInput {
    action_id: String,
}
struct RuntimeApplyCodeAction {
    project: Entity<Project>,
    actor: String,
    cache: RuntimeCodeActions,
}

impl AgentTool for RuntimeApplyCodeAction {
    type Input = RuntimeApplyCodeActionInput;
    type Output = String;
    const NAME: &'static str = "apply_code_action";
    fn kind() -> acp::ToolKind {
        acp::ToolKind::Other
    }
    fn allow_in_restricted_mode() -> bool {
        false
    }
    fn initial_title(
        &self,
        _: Result<Self::Input, serde_json::Value>,
        _: &mut App,
    ) -> SharedString {
        "Apply code action".into()
    }
    fn run(
        self: Arc<Self>,
        input: ToolInput<Self::Input>,
        event_stream: ToolCallEventStream,
        cx: &mut App,
    ) -> Task<Result<String, String>> {
        cx.spawn(async move |cx| {
            let input = input.recv().await.map_err(|error| error.to_string())?;
            let pending = self.cache.0.borrow_mut().take(&self.actor, &input.action_id).map_err(|error| error.to_string())?;
            let title = pending.action.lsp_action.title().to_owned();
            cx.update(|cx| event_stream.authorize(
                format!("Apply code action: {title}"),
                ToolPermissionContext::new(Self::NAME, vec![title.clone()]), cx
            )).await.map_err(|error| error.to_string())?;
            let unchanged = pending.buffer.read_with(cx, |buffer, _| buffer.as_text_snapshot().version() == pending.snapshot.version());
            if !unchanged { return Err("The buffer changed after these code actions were generated; request actions again.".into()); }
            let restricted = self.project.read_with(cx, |project, cx| TrustedWorktrees::has_restricted_worktrees(&project.worktree_store(), cx));
            if restricted { return Err("Cannot apply code actions to a restricted workspace.".into()); }
            let transaction = self.project.update(cx, |project, cx| project.apply_code_action(pending.buffer, pending.action, true, cx)).await.map_err(|error| error.to_string())?;
            Ok(format!("Applied '{title}' to {} buffer(s).", transaction.0.len()))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fs::FakeFs;
    use gpui::TestAppContext;
    use serde_json::json;

    #[test]
    fn code_action_ids_are_agent_scoped_consumed_once_and_bounded() -> Result<()> {
        let mut cache = ActionCache {
            entries: VecDeque::new(),
            bytes: 0,
            maximum_entries: 2,
            maximum_bytes: 8,
        };
        let first = cache.insert("agent-a".into(), "first", 4)?;
        assert!(cache.take("agent-b", &first).is_err());
        assert_eq!(cache.take("agent-a", &first)?, "first");
        assert!(cache.take("agent-a", &first).is_err());
        let expired = cache.insert("agent-a".into(), "expired", 4)?;
        cache.insert("agent-b".into(), "kept", 4)?;
        cache.insert("agent-c".into(), "new", 4)?;
        assert!(cache.take("agent-a", &expired).is_err());
        assert!(cache.insert("agent-a".into(), "oversized", 9).is_err());
        cache.clear_agent("agent-b");
        assert_eq!(cache.entries.len(), 1);
        assert_eq!(cache.bytes, 4);
        Ok(())
    }

    #[gpui::test]
    async fn project_bridge_reads_unsaved_buffers_and_refuses_stale_edits(cx: &mut TestAppContext) {
        crate::tests::init_test(cx);
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree("/project", json!({"main.rs": "disk version"}))
            .await;
        let project = Project::test(fs.clone(), [], cx).await;
        let worktree = project
            .update(cx, |project, cx| {
                project.create_worktree("/project", true, cx)
            })
            .await
            .expect("worktree");
        let path = cx.update(|cx| ProjectPath {
            worktree_id: worktree.read(cx).id(),
            path: util::rel_path::RelPath::from_unix_str("main.rs")
                .expect("relative path")
                .into(),
        });
        let buffer = project
            .update(cx, |project, cx| project.open_buffer(path.clone(), cx))
            .await
            .expect("buffer");
        buffer.update(cx, |buffer, cx| {
            buffer.set_text("unsaved version", cx);
        });
        let bridge =
            cx.update(|cx| RuntimeProjectBridge::new(project, 2, 1024, cx).expect("bridge"));
        assert_eq!(
            bridge
                .client
                .read_buffer(path.clone())
                .await
                .expect("read")
                .text,
            "unsaved version"
        );
        assert!(
            bridge
                .client
                .replace_authorized_buffer(
                    path.clone(),
                    "disk version".into(),
                    "bad edit".into(),
                    false
                )
                .await
                .is_err()
        );
        assert_eq!(
            bridge
                .client
                .replace_authorized_buffer(
                    path.clone(),
                    "unsaved version".into(),
                    "accepted edit".into(),
                    false
                )
                .await
                .expect("edit")
                .text,
            "accepted edit"
        );
        assert!(bridge.client.read_buffer(path).await.expect("read").dirty);
        drop(bridge.task);
        assert!(
            bridge
                .client
                .read_buffer(cx.update(|cx| {
                    ProjectPath {
                        worktree_id: worktree.read(cx).id(),
                        path: util::rel_path::RelPath::from_unix_str("main.rs")
                            .expect("path")
                            .into(),
                    }
                }))
                .await
                .is_err()
        );
    }
}
