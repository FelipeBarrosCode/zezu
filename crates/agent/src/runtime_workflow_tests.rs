#![cfg(unix)]

use std::{
    cell::Cell,
    path::Path,
    rc::Rc,
    time::{Duration, Instant},
};

use agent_runtime::{
    Job, JobState, RuntimeLimits, RuntimeStore, SliceOutcome,
    integration::{GitIntegration, IntegrationCheck, IntegrationRequest},
    orchestration::TaskIdentity,
    workspaces::WorkspacePool,
};
use agent_settings::AgentSettings;
use anyhow::{Context as _, Result, anyhow, ensure};
use gpui::TestAppContext;
use project::Project;
use serde_json::{Value, json};
use settings::Settings as _;

use crate::{
    ThreadEvent,
    runtime_project::runtime_project_tools_with_log,
    runtime_tools::{
        RuntimeToolContext, RuntimeToolEventSink, RuntimeToolHost, RuntimeToolPreparation,
        RuntimeToolResolver,
    },
    runtime_workspaces::RuntimeWorkspaceProjects,
};

const ORIGINAL_SETTINGS: &str = "{\"tab_size\":2}\n";
const EDITED_SETTINGS: &str = "{\"tab_size\":4}\n";

#[allow(
    clippy::disallowed_methods,
    reason = "Fixture owns synchronous Git processes and never converts them to smol"
)]
fn git(repository: &Path, arguments: &[&str]) -> Result<()> {
    let output = std::process::Command::new("git")
        .args(arguments)
        .current_dir(repository)
        .output()?;
    ensure!(
        output.status.success(),
        "Git fixture failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(())
}

async fn enqueue_tool(
    store: &RuntimeStore,
    id: &str,
    tool: &str,
    call: &str,
    arguments: Value,
) -> Result<()> {
    store
        .enqueue_text_input(
            "actor".into(),
            format!("input-{id}"),
            "next workflow step".into(),
        )
        .await?;
    let lease = store.admit().await?.context("workflow lease")?;
    store
        .record_job(
            lease.clone(),
            id.into(),
            json!({"tool":tool,"tool_call_id":call,"arguments":arguments}),
        )
        .await?;
    store.finish(lease, json!({}), SliceOutcome::Waiting).await
}

async fn completed(store: &RuntimeStore, id: &str) -> Job {
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        let job = store.job(id.into()).await.expect("workflow job");
        if job.state == JobState::Completed {
            return job;
        }
        assert!(
            Instant::now() < deadline,
            "workflow job {id} did not complete: {:?}",
            job.state
        );
        // SQLite and Git workers use OS threads outside GPUI's simulated clock.
        async_io::Timer::after(Duration::from_millis(2)).await;
    }
}

#[cfg(unix)]
#[gpui::test]
async fn native_read_write_approval_checkpoint_and_integration_use_the_isolated_workspace(
    cx: &mut TestAppContext,
) {
    use agent_client_protocol::v2::schema as acp;
    cx.executor().allow_parking();
    let provider = crate::tests::init_test(cx);
    cx.update(|cx| {
        let mut settings = AgentSettings::get_global(cx).clone();
        settings.tool_permissions.default = settings::ToolPermissionMode::Confirm;
        settings.tool_permissions.tools.clear();
        AgentSettings::override_global(settings, cx);
    });
    let directory = tempfile::tempdir().expect("directory");
    let repository = directory.path().join("repository");
    std::fs::create_dir_all(repository.join(".zed")).expect("repository");
    git(&repository, &["init", "--initial-branch=main"]).expect("init");
    git(&repository, &["config", "user.name", "Runtime Test"]).expect("name");
    git(
        &repository,
        &["config", "user.email", "runtime@example.invalid"],
    )
    .expect("email");
    std::fs::write(repository.join(".zed/settings.json"), ORIGINAL_SETTINGS).expect("file");
    git(&repository, &["add", ".zed/settings.json"]).expect("add");
    git(&repository, &["commit", "-m", "initial"]).expect("commit");
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
        .create_root("actor".into(), json!({}))
        .await
        .expect("actor");
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
                agent_id: "actor".into(),
                generation: 0,
            },
            "main".into(),
        )
        .await
        .expect("acquire")
        .expect("lease");
    let registry =
        cx.update(|cx| RuntimeWorkspaceProjects::new(original.clone(), 1, cx).expect("routing"));
    let sample = Job {
        id: "sample".into(),
        agent_id: "actor".into(),
        generation: 0,
        state: JobState::Queued,
        payload: json!({"tool":"read_file"}),
        result: None,
    };
    let snapshot = store.snapshot("actor".into()).await.expect("snapshot");
    registry
        .update(cx, |registry, cx| {
            registry.prepare(store.clone(), &sample, &snapshot, cx)
        })
        .await
        .expect("prepare isolated project");
    let isolated = registry.read_with(cx, |registry, cx| {
        registry.project_for(&sample, cx).expect("isolated project")
    });
    let worktree = isolated.read_with(cx, |project, cx| {
        project.worktrees(cx).next().expect("worktree")
    });
    worktree
        .read_with(cx, |tree, _| {
            tree.as_local().expect("local worktree").scan_complete()
        })
        .await;
    assert_ne!(isolated.entity_id(), original.entity_id());
    let log = registry.read_with(cx, |registry, _| registry.action_log_for(&sample));
    let resolver: RuntimeToolResolver = {
        let registry = registry.clone();
        let original = original.clone();
        let provider_id = provider.model("fake").provider_id();
        Rc::new(move |job, _, cx| {
            let (project, actions, log) = registry.read_with(cx, |registry, cx| {
                Ok::<_, anyhow::Error>((
                    registry.project_for(job, cx)?,
                    registry.code_actions_for(job),
                    registry.action_log_for(job),
                ))
            })?;
            let profile = AgentSettings::get_global(cx).default_profile.clone();
            let tools = runtime_project_tools_with_log(
                project.clone(),
                &job.agent_id,
                &provider_id,
                &profile,
                actions,
                log,
                cx,
            )?;
            Ok(
                RuntimeToolContext::new(project, tools, provider_id.clone(), cx)
                    .with_trust_source(original.clone()),
            )
        })
    };
    let approvals = Rc::new(Cell::new(0usize));
    let sink: RuntimeToolEventSink = {
        let approvals = approvals.clone();
        let original_file = repository.join(".zed/settings.json");
        Rc::new(move |job, _, event, _| {
            if let Some(ThreadEvent::ToolCallAuthorization(authorization)) = event.take() {
                if job.id == "write-job" {
                    assert_eq!(std::fs::read_to_string(&original_file)?, ORIGINAL_SETTINGS);
                    approvals.set(approvals.get() + 1);
                }
                let option = authorization
                    .options
                    .allow_once_option_id()
                    .context("native approval must offer allow once")?;
                authorization
                    .response
                    .send(acp_thread::SelectedPermissionOutcome::new(
                        option,
                        acp::PermissionOptionKind::AllowOnce,
                    ))
                    .map_err(|_| anyhow!("approval requester detached"))?;
            }
            Ok(())
        })
    };
    let preparation: RuntimeToolPreparation = {
        let registry = registry.clone();
        let store = store.clone();
        Rc::new(move |job, snapshot, cx| {
            registry.update(cx, |registry, cx| {
                registry.prepare(store.clone(), job, snapshot, cx)
            })
        })
    };
    let host = cx.update(|cx| {
        let host =
            RuntimeToolHost::start(store.clone(), resolver, sink, 1, cx).expect("native host");
        host.update(cx, |host, _| host.set_preparation(preparation));
        host
    });
    enqueue_tool(
        &store,
        "read-job",
        "read_file",
        "read-call",
        json!({"path":"repository/.zed/settings.json"}),
    )
    .await
    .expect("read job");
    let read = completed(&store, "read-job")
        .await
        .result
        .expect("read result");
    assert_eq!(read["content"]["id"], "read-call");
    assert_eq!(read["content"]["is_error"], false, "{read}");
    assert!(
        log.read_with(cx, |log, _| log
            .file_read_time(&lease.path.join(".zed/settings.json"))
            .is_some()),
        "native ReadFile must persist its observed file version in the shared ActionLog"
    );
    enqueue_tool(
        &store,
        "write-job",
        "write_file",
        "write-call",
        json!({"path":"repository/.zed/settings.json","content":EDITED_SETTINGS}),
    )
    .await
    .expect("write job");
    let write = completed(&store, "write-job")
        .await
        .result
        .expect("write result");
    assert_eq!(write["content"]["id"], "write-call");
    assert_eq!(write["content"]["is_error"], false, "{write}");
    assert!(
        approvals.get() > 0,
        "native write must pass through its permission decision"
    );
    assert_eq!(
        log.entity_id(),
        registry.read_with(cx, |registry, _| registry
            .action_log_for(&sample)
            .entity_id())
    );
    assert_eq!(
        std::fs::read_to_string(repository.join(".zed/settings.json")).expect("original"),
        ORIGINAL_SETTINGS
    );
    assert_eq!(
        serde_json::from_str::<Value>(
            &std::fs::read_to_string(lease.path.join(".zed/settings.json")).expect("isolated")
        )
        .expect("valid isolated settings"),
        json!({"tab_size":4})
    );
    assert!(
        store
            .latest_job_event("actor".into(), "approval_decision".into())
            .await
            .expect("audit")
            .is_some()
    );
    drop(host);
    cx.update(|_| ());
    let artifact = pool
        .checkpoint(lease.clone(), "native approved write".into())
        .await
        .expect("artifact");
    pool.release(lease).await.expect("release");
    let integration = GitIntegration::open(store.clone(), repository.clone())
        .await
        .expect("integration");
    let result = integration
        .integrate(IntegrationRequest {
            id: "native-workflow".into(),
            agent_id: "actor".into(),
            generation: 0,
            source_owner: Some(TaskIdentity {
                agent_id: artifact.agent_id,
                generation: artifact.generation,
            }),
            target_branch: "main".into(),
            source_commit: artifact.commit,
            checks: vec![IntegrationCheck {
                program: "/bin/sh".into(),
                arguments: vec![
                    "-c".into(),
                    "grep -Eq '\"tab_size\"[[:space:]]*:[[:space:]]*4' .zed/settings.json".into(),
                ],
                timeout_seconds: 5,
            }],
        })
        .await
        .expect("checked integration");
    assert_eq!(result.state, "integrated");
    assert_eq!(
        serde_json::from_str::<Value>(
            &std::fs::read_to_string(repository.join(".zed/settings.json")).expect("integrated")
        )
        .expect("valid integrated settings"),
        json!({"tab_size":4})
    );
    assert!(
        store.dispatch_job("write-job".into()).await.is_err(),
        "completed write must not replay during integration"
    );
    store.shutdown().await.expect("shutdown");
}
