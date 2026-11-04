use std::collections::BTreeMap;

use action_log::ActionLog;
use agent_runtime::{
    AgentSnapshot, Job, RuntimeStore, orchestration::TaskIdentity, workspaces::WorkspaceLease,
};
use anyhow::{Context as _, Result, ensure};
use gpui::{App, AppContext as _, Context, Entity, EntityId, Global, Task, WeakEntity};
use project::{LocalProjectFlags, Project, trusted_worktrees::TrustedWorktrees};

use crate::{runtime_mcp::RuntimeMcpCatalog, runtime_project::RuntimeCodeActions};

struct WorkspaceProject {
    lease: WorkspaceLease,
    project: Entity<Project>,
    actions: RuntimeCodeActions,
    action_log: Entity<ActionLog>,
    mcp: RuntimeMcpCatalog,
    ready: bool,
}

pub struct RuntimeWorkspaceProjects {
    original: Entity<Project>,
    original_actions: RuntimeCodeActions,
    original_action_log: Entity<ActionLog>,
    original_mcp: RuntimeMcpCatalog,
    maximum: usize,
    workspaces: BTreeMap<i64, WorkspaceProject>,
}

#[derive(Default)]
struct SharedWorkspaceProjects(BTreeMap<EntityId, WeakEntity<RuntimeWorkspaceProjects>>);
impl Global for SharedWorkspaceProjects {}

impl RuntimeWorkspaceProjects {
    pub fn new(original: Entity<Project>, maximum: usize, cx: &mut App) -> Result<Entity<Self>> {
        ensure!(
            (1..=4096).contains(&maximum),
            "workspace project limit must be 1–4096"
        );
        let key = original.entity_id();
        if let Some(existing) = cx
            .try_global::<SharedWorkspaceProjects>()
            .and_then(|shared| shared.0.get(&key))
            .and_then(WeakEntity::upgrade)
        {
            ensure!(
                existing.read(cx).maximum == maximum,
                "restart the runtime before changing its workspace project capacity"
            );
            return Ok(existing);
        }
        let original_actions = RuntimeCodeActions::new(128, 1024 * 1024)?;
        let original_mcp = RuntimeMcpCatalog::new(original.clone(), cx);
        let original_action_log = cx.new(|_| ActionLog::new(original.clone()));
        let registry = cx.new(|_| Self {
            original,
            original_actions,
            original_action_log,
            original_mcp,
            maximum,
            workspaces: BTreeMap::new(),
        });
        if cx.try_global::<SharedWorkspaceProjects>().is_none() {
            cx.set_global(SharedWorkspaceProjects::default());
        }
        let shared = cx.global_mut::<SharedWorkspaceProjects>();
        shared.0.retain(|_, registry| registry.upgrade().is_some());
        shared.0.insert(key, registry.downgrade());
        Ok(registry)
    }

    pub fn prepare(
        &mut self,
        store: RuntimeStore,
        job: &Job,
        snapshot: &AgentSnapshot,
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        let identity = TaskIdentity {
            agent_id: job.agent_id.clone(),
            generation: job.generation,
        };
        let generation = snapshot.generation;
        let tool = job
            .payload
            .get("tool")
            .and_then(|tool| tool.as_str())
            .unwrap_or_default()
            .to_owned();
        cx.spawn(async move |this, cx| {
            ensure!(
                identity.generation == generation,
                "workspace request belongs to a stale generation"
            );
            let lease = store.workspace_lease_for_task(identity.clone()).await?;
            let Some(lease) = lease else {
                this.update(cx, |registry, _| {
                    registry
                        .workspaces
                        .retain(|_, workspace| workspace.lease.agent_id != identity.agent_id)
                })?;
                ensure!(
                    !requires_workspace(&tool),
                    "acquire an isolated runtime workspace before using {tool}"
                );
                return Ok(());
            };
            let path = lease.path.clone();
            cx.background_spawn(async move {
                ensure!(
                    !std::fs::symlink_metadata(&path)?.is_symlink(),
                    "leased workspace cannot be a symlink"
                );
                ensure!(
                    path.canonicalize()? == path,
                    "leased workspace path changed since acquisition"
                );
                Ok::<_, anyhow::Error>(())
            })
            .await?;
            let creation = this.update(cx, |registry, cx| {
                registry.ensure_original_trusted(cx)?;
                if let Some(existing) = registry.workspaces.get(&lease.slot_id) {
                    if existing.lease.token == lease.token
                        && existing.lease.agent_id == lease.agent_id
                        && existing.lease.generation == lease.generation
                        && existing.ready
                    {
                        return Ok(None);
                    }
                }
                registry.workspaces.remove(&lease.slot_id);
                registry
                    .workspaces
                    .retain(|_, workspace| workspace.lease.agent_id != lease.agent_id);
                ensure!(
                    registry.workspaces.len() < registry.maximum,
                    "workspace project cache reached its configured capacity"
                );
                let original = registry.original.read(cx);
                ensure!(
                    original.is_local(),
                    "isolated runtime workspaces require a local project"
                );
                let client = original.client();
                let node = original
                    .node_runtime()
                    .context("project has no local Node runtime")?
                    .clone();
                let users = original.user_store();
                let languages = original.languages().clone();
                let fs = original.fs().clone();
                let environment = original.cli_environment(cx);
                // Pool workspaces inherit source-project trust, checked again for every routed job.
                let project = Project::local(
                    client,
                    node,
                    users,
                    languages,
                    fs,
                    environment,
                    LocalProjectFlags {
                        init_worktree_trust: false,
                        watch_global_configs: false,
                    },
                    cx,
                );
                let actions = RuntimeCodeActions::new(128, 1024 * 1024)?;
                let mcp = RuntimeMcpCatalog::new(project.clone(), cx);
                let action_log = cx.new(|_| ActionLog::new(project.clone()));
                let task = project.update(cx, |project, cx| {
                    project.find_or_create_worktree(&lease.path, true, cx)
                });
                registry.workspaces.insert(
                    lease.slot_id,
                    WorkspaceProject {
                        lease: lease.clone(),
                        project: project.clone(),
                        actions,
                        action_log,
                        mcp,
                        ready: false,
                    },
                );
                Ok::<_, anyhow::Error>(Some((project, task)))
            })??;
            if let Some((project, task)) = creation {
                task.await?;
                let current = store
                    .workspace_lease_for_task(identity)
                    .await?
                    .context("workspace was released during project initialization")?;
                ensure!(
                    current.token == lease.token && current.path == lease.path,
                    "workspace ownership changed during project initialization"
                );
                this.update(cx, |registry, cx| {
                    registry.ensure_original_trusted(cx)?;
                    let entry = registry
                        .workspaces
                        .get_mut(&lease.slot_id)
                        .context("workspace project was evicted during initialization")?;
                    ensure!(
                        entry.lease.token == lease.token && entry.project == project,
                        "workspace project changed during initialization"
                    );
                    entry.ready = true;
                    Ok::<_, anyhow::Error>(())
                })??;
            }
            Ok(())
        })
    }

    pub fn project_for(&self, job: &Job, cx: &App) -> Result<Entity<Project>> {
        if let Some(entry) = self.entry(job) {
            self.ensure_original_trusted(cx)?;
            return Ok(entry.project.clone());
        }
        let tool = job
            .payload
            .get("tool")
            .and_then(|tool| tool.as_str())
            .unwrap_or_default();
        ensure!(
            !requires_workspace(tool),
            "acquire an isolated runtime workspace before using {tool}"
        );
        Ok(self.original.clone())
    }

    pub fn code_actions_for(&self, job: &Job) -> RuntimeCodeActions {
        self.entry(job)
            .map(|entry| entry.actions.clone())
            .unwrap_or_else(|| self.original_actions.clone())
    }

    pub fn action_log_for(&self, job: &Job) -> Entity<ActionLog> {
        self.entry(job)
            .map(|entry| entry.action_log.clone())
            .unwrap_or_else(|| self.original_action_log.clone())
    }

    pub fn mcp_catalog_for(&self, job: &Job) -> RuntimeMcpCatalog {
        self.entry(job)
            .map(|entry| entry.mcp.clone())
            .unwrap_or_else(|| self.original_mcp.clone())
    }

    pub fn has_workspace(&self, job: &Job) -> bool {
        self.entry(job).is_some()
    }

    pub fn detach_agent(&mut self, agent_id: &str) {
        self.original_actions.clear_agent(agent_id);
        self.workspaces
            .retain(|_, workspace| workspace.lease.agent_id != agent_id);
    }

    pub fn resident_projects(&self) -> usize {
        self.workspaces.len()
    }

    fn entry(&self, job: &Job) -> Option<&WorkspaceProject> {
        self.workspaces.values().find(|entry| {
            entry.ready
                && entry.lease.agent_id == job.agent_id
                && entry.lease.generation == job.generation
        })
    }

    fn ensure_original_trusted(&self, cx: &App) -> Result<()> {
        ensure!(
            !TrustedWorktrees::has_restricted_worktrees(
                &self.original.read(cx).worktree_store(),
                cx
            ),
            "source project is restricted; its derived workspaces cannot execute tools"
        );
        Ok(())
    }
}

fn requires_workspace(tool: &str) -> bool {
    matches!(
        tool,
        "edit_file"
            | "write_file"
            | "copy_path"
            | "create_directory"
            | "delete_path"
            | "move_path"
            | "rename"
            | "rename_symbol"
            | "get_code_actions"
            | "apply_code_action"
            | "terminal"
            | "sandboxed_terminal"
            | "run_command"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn editing_and_process_tools_require_an_isolated_workspace() {
        for tool in [
            "edit_file",
            "write_file",
            "terminal",
            "apply_code_action",
            "delete_path",
            "rename",
        ] {
            assert!(requires_workspace(tool));
        }
        for tool in [
            "read_file",
            "grep",
            "runtime_workspace_acquire",
            "runtime_workspace_checkpoint",
            "runtime_workspace_release",
            "runtime_integrate_artifact",
        ] {
            assert!(!requires_workspace(tool));
        }
    }

    #[gpui::test]
    async fn leased_project_edits_stay_out_of_the_original_project(cx: &mut gpui::TestAppContext) {
        use agent_runtime::{JobState, RuntimeLimits, workspaces::WorkspacePool};
        use project::ProjectPath;
        cx.executor().allow_parking();
        crate::tests::init_test(cx);
        let directory = tempfile::tempdir().expect("directory");
        let repository = directory.path().join("repository");
        std::fs::create_dir(&repository).expect("repository");
        fn git(path: &std::path::Path, arguments: &[&str]) {
            let output = std::process::Command::new("git")
                .args(arguments)
                .current_dir(path)
                .output()
                .expect("git");
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        git(&repository, &["init", "--initial-branch=main"]);
        git(&repository, &["config", "user.name", "Runtime test"]);
        git(
            &repository,
            &["config", "user.email", "runtime@example.invalid"],
        );
        std::fs::write(repository.join("code.txt"), "original\n").expect("source");
        git(&repository, &["add", "code.txt"]);
        git(&repository, &["commit", "-m", "initial"]);
        let original = cx
            .spawn({
                let repository = repository.clone();
                async move |mut cx| Project::example([repository.as_path()], &mut cx).await
            })
            .await;
        let store = RuntimeStore::open(
            directory.path().join("runtime.sqlite"),
            RuntimeLimits::default(),
        )
        .expect("store");
        store
            .create_root("root".into(), serde_json::json!({}))
            .await
            .expect("root");
        let pool = WorkspacePool::open(
            store.clone(),
            repository.clone(),
            directory.path().join("pool"),
            1,
        )
        .await
        .expect("pool");
        let lease = pool
            .acquire(
                TaskIdentity {
                    agent_id: "root".into(),
                    generation: 0,
                },
                "main".into(),
            )
            .await
            .expect("acquire")
            .expect("lease");
        let registry = cx
            .update(|cx| RuntimeWorkspaceProjects::new(original.clone(), 1, cx).expect("registry"));
        let same_project_registry = cx.update(|cx| {
            RuntimeWorkspaceProjects::new(original.clone(), 1, cx).expect("shared registry")
        });
        assert_eq!(registry.entity_id(), same_project_registry.entity_id());
        let job = Job {
            id: "job".into(),
            agent_id: "root".into(),
            generation: 0,
            state: JobState::Queued,
            payload: serde_json::json!({"tool": "write_file"}),
            result: None,
        };
        let snapshot = store.snapshot("root".into()).await.expect("snapshot");
        registry
            .update(cx, |registry, cx| {
                registry.prepare(store.clone(), &job, &snapshot, cx)
            })
            .await
            .expect("prepare");
        let isolated = registry.read_with(cx, |registry, cx| {
            registry.project_for(&job, cx).expect("isolated project")
        });
        assert_ne!(isolated.entity_id(), original.entity_id());
        let path = isolated.read_with(cx, |project, cx| ProjectPath {
            worktree_id: project
                .worktrees(cx)
                .next()
                .expect("worktree")
                .read(cx)
                .id(),
            path: util::rel_path::RelPath::from_unix_str("code.txt")
                .expect("relative path")
                .into(),
        });
        let buffer = isolated
            .update(cx, |project, cx| project.open_buffer(path, cx))
            .await
            .expect("buffer");
        buffer.update(cx, |buffer, cx| buffer.set_text("isolated edit\n", cx));
        isolated
            .update(cx, |project, cx| project.save_buffer(buffer, cx))
            .await
            .expect("save");
        assert_eq!(
            std::fs::read_to_string(repository.join("code.txt")).expect("original file"),
            "original\n"
        );
        assert_eq!(
            std::fs::read_to_string(lease.path.join("code.txt")).expect("isolated file"),
            "isolated edit\n"
        );
        pool.checkpoint(lease.clone(), "checkpoint isolated edit".into())
            .await
            .expect("checkpoint");
        pool.release(lease).await.expect("release");
        registry
            .update(cx, |registry, cx| {
                registry.prepare(store.clone(), &job, &snapshot, cx)
            })
            .await
            .expect_err("editing without a lease must fail");
        assert_eq!(
            registry.read_with(cx, |registry, _| registry.resident_projects()),
            0
        );
        store.shutdown().await.expect("shutdown");
    }
}
