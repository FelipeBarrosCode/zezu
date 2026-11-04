use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use anyhow::{Context as _, Result, ensure};
use async_trait::async_trait;
use futures::{FutureExt as _, StreamExt as _};
use infinity_provider_protocol::{
    CompletionError, CompletionRequest, FinalResponse, ModelEntry, ModelProvider, ModelStream,
    StreamChunk,
};
use rusqlite::params;

use crate::{
    EmbeddedRuntime, InfinitySlice, JobRecovery, JobState, RuntimeLimits, RuntimeStore,
    SliceFactory,
};

#[test]
#[ignore = "subprocess entry point for abrupt process termination tests"]
fn crash_process_entry() {
    let Some(path) = std::env::var_os("ZEZU_CRASH_DATABASE") else {
        return;
    };
    let scenario = std::env::var("ZEZU_CRASH_SCENARIO").expect("scenario");
    futures::executor::block_on(async {
        let store =
            RuntimeStore::open(PathBuf::from(path), RuntimeLimits::default()).expect("store");
        store
            .create_root("root".into(), serde_json::json!({"retained": true}))
            .await
            .expect("root");
        store
            .enqueue_text_input(
                "root".into(),
                "input".into(),
                "work surviving the process".into(),
            )
            .await
            .expect("input");
        let lease = store.admit().await.expect("admission").expect("lease");
        store
            .record_job(
                lease,
                "external-effect".into(),
                serde_json::json!({"tool": "external"}),
            )
            .await
            .expect("job");
        if scenario == "dispatched" {
            store
                .dispatch_job("external-effect".into())
                .await
                .expect("dispatch");
        }
        // Bypass all Rust destructors: only acknowledged SQLite transactions survive.
        std::process::exit(79);
    });
}

#[test]
#[allow(
    clippy::disallowed_methods,
    reason = "The crash fixture deliberately runs a synchronous subprocess to verify abrupt OS process termination."
)]
fn abrupt_exit_preserves_intents_and_requires_verified_reconciliation() -> Result<()> {
    for scenario in ["queued", "dispatched"] {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("runtime.sqlite");
        let status = std::process::Command::new(std::env::current_exe()?)
            .args([
                "--ignored",
                "--exact",
                "recovery_tests::crash_process_entry",
            ])
            .env("ZEZU_CRASH_DATABASE", &path)
            .env("ZEZU_CRASH_SCENARIO", scenario)
            .status()?;
        ensure!(
            status.code() == Some(79),
            "crash subprocess did not reach the durable boundary: {status}"
        );
        futures::executor::block_on(async {
            let store = RuntimeStore::open(&path, RuntimeLimits::default())?;
            let snapshot = store.snapshot("root".into()).await?;
            assert_eq!(snapshot.state, "paused");
            assert!(snapshot.generation > 0);
            assert_eq!(store.metadata("root".into()).await?["retained"], true);
            let job = store.job("external-effect".into()).await?;
            assert_eq!(
                job.state,
                if scenario == "dispatched" {
                    JobState::Uncertain
                } else {
                    JobState::Queued
                }
            );
            assert!(store.admit().await?.is_none());
            assert!(store.dispatch_job("external-effect".into()).await.is_err());
            if scenario == "dispatched" {
                store
                    .reconcile_job("external-effect".into(), JobRecovery::ConfirmedNotExecuted)
                    .await?;
                assert!(store.dispatch_job("external-effect".into()).await.is_err());
            }
            store.shutdown().await
        })?;
    }
    Ok(())
}

struct FleetProvider {
    active: Arc<AtomicUsize>,
    peak: Arc<AtomicUsize>,
}

struct Active(Arc<AtomicUsize>);
impl Drop for Active {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

#[async_trait]
impl ModelProvider for FleetProvider {
    async fn list_models(
        &self,
    ) -> Result<Vec<ModelEntry>, Box<dyn std::error::Error + Send + Sync>> {
        Ok(vec![fleet_model()])
    }
    async fn invoke_model(
        &self,
        _: &str,
        _: CompletionRequest,
    ) -> Result<ModelStream, CompletionError> {
        let active = self.active.fetch_add(1, Ordering::AcqRel) + 1;
        self.peak.fetch_max(active, Ordering::AcqRel);
        let guard = Active(self.active.clone());
        Ok(futures::stream::once(async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            drop(guard);
            Ok(StreamChunk::Final(FinalResponse::default()))
        })
        .boxed())
    }
}

fn fleet_model() -> ModelEntry {
    ModelEntry {
        model_id: "fleet-fixture".into(),
        display_name: "Fleet fixture".into(),
        context_window: 4096,
        max_output_tokens: Some(32),
        supports_image_input: false,
    }
}

#[tokio::test]
#[ignore = "explicit 75,000 logical-agent runtime scenario; synthetic provider, no editor or network inference"]
async fn fleet_capacity_reports_measured_runtime_admission_and_memory() -> Result<()> {
    const AGENTS: usize = 75_000;
    const ACTIVE: usize = 16;
    const SUBMISSIONS: usize = 256;
    let started = Instant::now();
    let directory = tempfile::tempdir()?;
    let limits = RuntimeLimits {
        max_agents_per_run: AGENTS,
        max_active_agents: ACTIVE,
        ..Default::default()
    };
    let store = RuntimeStore::open(directory.path().join("fleet.sqlite"), limits)?;
    store
        .create_root("root".into(), serde_json::json!({}))
        .await?;
    store.write(|connection| {
        let transaction = connection.transaction()?;
        {
            let mut statement = transaction.prepare("INSERT INTO agents(id,root_id,parent_id,spawn_key,fresh) VALUES (?,'root','root',?,1)")?;
            for index in 1..AGENTS { statement.execute(params![format!("agent-{index:06}"),format!("spawn-{index}")])?; }
        }
        transaction.execute("UPDATE runs SET agent_count=? WHERE id='root'", [AGENTS])?;
        transaction.commit()?;
        Ok(())
    }).await?;
    let page = store.list_agents("root".into(), None, 64).await?;
    assert_eq!(page.agents.len(), 64);
    assert!(page.next_cursor.is_some());
    let active = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));
    let provider = Arc::new(FleetProvider {
        active: active.clone(),
        peak: peak.clone(),
    });
    let factory: SliceFactory = Arc::new(move |_| {
        let provider = provider.clone();
        async move {
            Ok(InfinitySlice {
                provider,
                model: fleet_model(),
                tools: Vec::new(),
                system_prompt: String::new(),
            })
        }
        .boxed()
    });
    let runtime = EmbeddedRuntime::start(store.clone(), factory)?;
    let mut notices = runtime.subscribe();
    let mut received = 0;
    for index in 1..=SUBMISSIONS {
        store
            .enqueue_text_input(
                format!("agent-{index:06}"),
                format!("input-{index}"),
                "bounded synthetic work".into(),
            )
            .await?;
    }
    while received < SUBMISSIONS {
        let notice = tokio::time::timeout(Duration::from_secs(30), notices.recv()).await??;
        ensure!(
            notice.error.is_none(),
            "fleet slice failed: {:?}",
            notice.error
        );
        received += 1;
    }
    runtime.shutdown().await?;
    assert_eq!(active.load(Ordering::Acquire), 0);
    assert!(peak.load(Ordering::Acquire) <= ACTIVE);
    assert!(peak.load(Ordering::Acquire) > 1);
    #[cfg(target_os = "linux")]
    let memory = std::fs::read_to_string("/proc/self/status")?
        .lines()
        .find(|line| line.starts_with("VmHWM:"))
        .map(str::to_owned);
    #[cfg(not(target_os = "linux"))]
    let memory: Option<String> = None;
    eprintln!(
        "ZEZU_RUNTIME_FLEET_REPORT logical_agents={AGENTS} synthetic_completed={received} configured_active={ACTIVE} measured_peak_active={} elapsed_ms={} process_peak_memory={:?}; scope=runtime_test_process, not full editor, no paid/live inference",
        peak.load(Ordering::Acquire),
        started.elapsed().as_millis(),
        memory
    );
    Ok(())
}

#[cfg(unix)]
#[tokio::test]
async fn pooled_workspace_artifact_passes_checks_integrates_and_reverts_selectively() -> Result<()>
{
    use crate::{
        integration::{GitIntegration, IntegrationCheck, IntegrationRequest},
        orchestration::TaskIdentity,
        workspaces::WorkspacePool,
    };
    #[allow(
        clippy::disallowed_methods,
        reason = "This synchronous fixture invokes std::process directly without any smol conversion."
    )]
    fn git(repository: &std::path::Path, arguments: &[&str]) -> Result<String> {
        let output = std::process::Command::new("git")
            .args(arguments)
            .current_dir(repository)
            .output()?;
        ensure!(
            output.status.success(),
            "Git fixture failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        Ok(String::from_utf8(output.stdout)?.trim_end().to_owned())
    }
    let directory = tempfile::tempdir()?;
    let repository = directory.path().join("repository");
    std::fs::create_dir(&repository)?;
    git(&repository, &["init", "--initial-branch=main"])?;
    git(&repository, &["config", "user.name", "Runtime Test"])?;
    git(
        &repository,
        &["config", "user.email", "runtime@example.invalid"],
    )?;
    std::fs::write(repository.join("code.txt"), "before\n")?;
    git(&repository, &["add", "code.txt"])?;
    git(&repository, &["commit", "-m", "initial"])?;
    let store = RuntimeStore::open(
        directory.path().join("runtime.sqlite"),
        RuntimeLimits::default(),
    )?;
    store
        .create_root("root".into(), serde_json::json!({}))
        .await?;
    let identity = TaskIdentity {
        agent_id: "root".into(),
        generation: 0,
    };
    let pool = WorkspacePool::open(
        store.clone(),
        repository.clone(),
        directory.path().join("pool"),
        1,
    )
    .await?;
    let lease = pool
        .acquire(identity, "main".into())
        .await?
        .context("workspace lease")?;
    std::fs::write(lease.path.join("code.txt"), "agent change\n")?;
    let artifact = pool
        .checkpoint(lease.clone(), "checkpoint agent change".into())
        .await?;
    pool.release(lease).await?;
    let integration = GitIntegration::open(store.clone(), repository.clone()).await?;
    let record = integration
        .integrate(IntegrationRequest {
            id: artifact.id,
            agent_id: artifact.agent_id,
            generation: artifact.generation,
            source_owner: Some(TaskIdentity {
                agent_id: "root".into(),
                generation: 0,
            }),
            target_branch: "main".into(),
            source_commit: artifact.commit,
            checks: vec![IntegrationCheck {
                program: "/bin/sh".into(),
                arguments: vec![
                    "-c".into(),
                    "test \"$(cat code.txt)\" = 'agent change'".into(),
                ],
                timeout_seconds: 5,
            }],
        })
        .await?;
    assert_eq!(record.state, "integrated");
    assert_eq!(
        std::fs::read_to_string(repository.join("code.txt"))?,
        "agent change\n"
    );
    std::fs::write(repository.join("user.txt"), "independent user change\n")?;
    git(&repository, &["add", "user.txt"])?;
    git(&repository, &["commit", "-m", "independent user change"])?;
    integration
        .revert(
            record.request.id,
            vec![IntegrationCheck {
                program: "/bin/sh".into(),
                arguments: vec!["-c".into(), "test \"$(cat code.txt)\" = before".into()],
                timeout_seconds: 5,
            }],
        )
        .await?;
    assert_eq!(
        std::fs::read_to_string(repository.join("code.txt"))?,
        "before\n"
    );
    assert_eq!(
        std::fs::read_to_string(repository.join("user.txt"))?,
        "independent user change\n"
    );
    store.shutdown().await
}
