use std::sync::Arc;

use anyhow::{Context as _, Result};
use async_trait::async_trait;
use futures::StreamExt as _;
use infinity_agent_core::{
    ThreadId,
    message::{InfinityMessage, InputMessage, InputMessageContent},
    system::EventCollector,
    traits::{ConversationStore, SpawnContext, StateStore},
};
use infinity_provider_protocol::{
    CompletionError, CompletionRequest, FinalResponse, ModelEntry, ModelProvider, ModelStream,
    StreamChunk, ToolCall, ToolDefinition, UserContent,
};
use rap_protocol::ToolCallId;
use serde_json::json;

use crate::{
    EmbeddedRuntime, InfinitySlice, JobRecovery, JobState, RuntimeLimits, RuntimeStore,
    SliceFactory, SliceOutcome, SqliteInfinityStore,
};

#[test]
fn store_operations_do_not_require_a_tokio_caller() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let store = RuntimeStore::open(
        directory.path().join("runtime.sqlite"),
        RuntimeLimits::default(),
    )?;
    futures::executor::block_on(async {
        store.create_root("root".into(), json!({})).await?;
        assert_eq!(store.snapshot("root".into()).await?.state, "waiting");
        store.shutdown().await
    })
}

#[tokio::test]
async fn concurrent_shutdown_callers_wait_for_the_same_checkpoint() -> Result<()> {
    use futures::FutureExt as _;
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("runtime.sqlite");
    let store = RuntimeStore::open(&path, RuntimeLimits::default())?;
    store
        .create_root("root".into(), json!({"configuration": "retained"}))
        .await?;
    let runtime = EmbeddedRuntime::start(store, Arc::new(|_| futures::future::pending().boxed()))?;
    let (first, second) = futures::join!(runtime.shutdown(), runtime.shutdown());
    first?;
    second?;
    let restored = RuntimeStore::open(path, RuntimeLimits::default())?;
    assert_eq!(
        restored.metadata("root".into()).await?,
        json!({"configuration": "retained"})
    );
    assert_eq!(restored.snapshot("root".into()).await?.state, "paused");
    restored.shutdown().await
}

#[tokio::test]
async fn reconciliation_requires_a_verified_outcome_before_retry() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("runtime.sqlite");
    let store = RuntimeStore::open(&path, RuntimeLimits::default())?;
    store.create_root("root".into(), json!({})).await?;
    store
        .enqueue_user_input("root".into(), "user".into(), input("root", "start")?)
        .await?;
    let lease = store.admit().await?.context("lease")?;
    store
        .record_job(lease, "job".into(), json!({"tool":"external"}))
        .await?;
    store.dispatch_job("job".into()).await?;
    store.shutdown().await?;
    let restored = RuntimeStore::open(&path, RuntimeLimits::default())?;
    restored
        .reconcile_job("job".into(), JobRecovery::ConfirmedNotExecuted)
        .await?;
    assert!(
        restored.dispatch_job("job".into()).await.is_err(),
        "reconciliation does not resume a paused run"
    );
    restored.resume("root".into()).await?;
    restored.dispatch_job("job".into()).await?;
    restored
        .complete_job("job".into(), input("root", "verified result")?)
        .await?;
    assert_eq!(restored.job("job".into()).await?.state, JobState::Completed);
    restored.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn embedded_worker_executes_without_a_view_and_shuts_down_paused() -> Result<()> {
    use futures::FutureExt as _;
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("runtime.sqlite");
    let store = RuntimeStore::open(&path, RuntimeLimits::default())?;
    store.create_root("root".into(), json!({})).await?;
    let factory: SliceFactory = Arc::new(|_| {
        async {
            Ok(InfinitySlice {
                provider: Arc::new(TestProvider {
                    chunks: vec![
                        StreamChunk::Text("background answer".into()),
                        StreamChunk::Final(FinalResponse::default()),
                    ],
                }),
                model: model(),
                tools: vec![],
                system_prompt: String::new(),
            })
        }
        .boxed()
    });
    let runtime = EmbeddedRuntime::start(store.clone(), factory)?;
    let mut notices = runtime.subscribe();
    store
        .enqueue_user_input("root".into(), "user".into(), input("root", "hello")?)
        .await?;
    let notice = tokio::time::timeout(std::time::Duration::from_secs(5), notices.recv()).await??;
    assert_eq!(notice.agent_id, "root");
    assert!(notice.error.is_none());
    let backend = SqliteInfinityStore::new(store.clone());
    assert_eq!(
        backend
            .load_history_up_to(ThreadId::from_ref("root"), None, None)
            .await?
            .len(),
        2
    );
    runtime.shutdown().await?;
    let restored = RuntimeStore::open(&path, RuntimeLimits::default())?;
    assert_eq!(restored.snapshot("root".into()).await?.state, "paused");
    restored.shutdown().await?;
    Ok(())
}

fn input(thread: &str, text: &str) -> Result<serde_json::Value> {
    Ok(serde_json::to_value(InputMessage {
        content: InputMessageContent::User(UserContent::text(text)),
        group_id: ThreadId::from(thread.to_owned()),
        metadata: None,
        synthetic: None,
        display_as: None,
        subscription: false,
    })?)
}

#[tokio::test]
async fn exclusive_owner_and_paused_restart() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("runtime.sqlite");
    let store = RuntimeStore::open(&path, RuntimeLimits::default())?;
    assert!(RuntimeStore::open(&path, RuntimeLimits::default()).is_err());
    store
        .create_root("root".into(), json!({"provider":"existing-setting"}))
        .await?;
    store
        .enqueue_user_input("root".into(), "input".into(), input("root", "hello")?)
        .await?;
    let lease = store.admit().await?.context("no lease")?;
    store
        .record_job(lease, "job".into(), json!({"tool":"external"}))
        .await?;
    store.dispatch_job("job".into()).await?;
    store.shutdown().await?;
    let restored = RuntimeStore::open(&path, RuntimeLimits::default())?;
    assert_eq!(restored.snapshot("root".into()).await?.state, "paused");
    assert_eq!(restored.job("job".into()).await?.state, JobState::Uncertain);
    assert!(restored.admit().await?.is_none());
    restored.resume("root".into()).await?;
    assert!(
        restored.admit().await?.is_none(),
        "uncertain effects must block automatic execution"
    );
    restored.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn fair_admission_limits_and_per_agent_fifo() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let store = RuntimeStore::open(
        directory.path().join("runtime.sqlite"),
        RuntimeLimits {
            max_active_agents: 1,
            max_agents_per_run: 3,
            max_inputs_per_slice: 1,
            ..RuntimeLimits::default()
        },
    )?;
    for root in ["a", "b"] {
        store.create_root(root.into(), json!({})).await?;
        store
            .enqueue_user_input(root.into(), "first".into(), input(root, "one")?)
            .await?;
        store
            .enqueue_user_input(root.into(), "second".into(), input(root, "two")?)
            .await?;
    }
    let first = store.admit().await?.context("first lease")?;
    assert_eq!(first.agent_id, "a");
    assert_eq!(first.inputs.first().context("input")?.0, "first");
    assert!(store.admit().await?.is_none());
    store
        .finish(first, json!({}), SliceOutcome::Waiting)
        .await?;
    let second = store.admit().await?.context("second lease")?;
    assert_eq!(second.agent_id, "b");
    store
        .finish(second, json!({}), SliceOutcome::Waiting)
        .await?;
    let third = store.admit().await?.context("third lease")?;
    assert_eq!(third.agent_id, "a");
    assert_eq!(third.inputs.first().context("input")?.0, "second");
    store.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn spawn_is_idempotent_and_all_descendants_share_the_cap() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let store = RuntimeStore::open(
        directory.path().join("runtime.sqlite"),
        RuntimeLimits {
            max_agents_per_run: 3,
            ..RuntimeLimits::default()
        },
    )?;
    store.create_root("root".into(), json!({})).await?;
    let child = store
        .spawn_child("root".into(), "spawn-one".into(), 0, false)
        .await?;
    assert_eq!(
        child,
        store
            .spawn_child("root".into(), "spawn-one".into(), 0, false)
            .await?
    );
    store
        .spawn_child(child, "spawn-two".into(), 0, false)
        .await?;
    assert!(
        store
            .spawn_child("root".into(), "spawn-three".into(), 0, false)
            .await
            .is_err()
    );
    store.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn cancel_fences_old_leases_and_preserves_new_user_turns() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let store = RuntimeStore::open(
        directory.path().join("runtime.sqlite"),
        RuntimeLimits::default(),
    )?;
    store.create_root("root".into(), json!({})).await?;
    let child = store
        .spawn_child("root".into(), "child".into(), 0, false)
        .await?;
    store
        .enqueue_user_input("root".into(), "first".into(), input("root", "hello")?)
        .await?;
    let lease = store.admit().await?.context("lease")?;
    store.cancel("root".into()).await?;
    assert_eq!(store.snapshot(child).await?.state, "cancelled");
    assert!(
        store
            .finish(lease.clone(), json!({}), SliceOutcome::Waiting)
            .await
            .is_err()
    );
    assert!(
        store
            .enqueue_from(lease, "root".into(), "late".into(), input("root", "late")?)
            .await
            .is_err()
    );
    store
        .enqueue_user_input(
            "root".into(),
            "new-user-turn".into(),
            input("root", "continue")?,
        )
        .await?;
    assert!(store.admit().await?.is_some());
    store.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn pause_waits_for_the_active_slice_and_keeps_wakeups() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let store = RuntimeStore::open(
        directory.path().join("runtime.sqlite"),
        RuntimeLimits::default(),
    )?;
    store.create_root("root".into(), json!({})).await?;
    store
        .enqueue_user_input("root".into(), "first".into(), input("root", "hello")?)
        .await?;
    let lease = store.admit().await?.context("lease")?;
    store.pause("root".into()).await?;
    assert_eq!(store.snapshot("root".into()).await?.state, "running");
    store
        .enqueue_from(
            lease.clone(),
            "root".into(),
            "wake".into(),
            input("root", "done")?,
        )
        .await?;
    store
        .finish(lease, json!({}), SliceOutcome::Waiting)
        .await?;
    assert_eq!(store.snapshot("root".into()).await?.state, "paused");
    store.resume("root".into()).await?;
    let resumed = store.admit().await?.context("resumed lease")?;
    assert_eq!(resumed.inputs.first().context("wake input")?.0, "wake");
    store.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn history_and_dedup_commit_together_and_children_reference_a_fixed_cutoff() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let store = RuntimeStore::open(
        directory.path().join("runtime.sqlite"),
        RuntimeLimits::default(),
    )?;
    store.create_root("root".into(), json!({})).await?;
    let backend = SqliteInfinityStore::new(store.clone());
    let root = ThreadId::from("root".to_owned());
    backend
        .append_messages(
            &root,
            vec![(
                InfinityMessage::User {
                    content: UserContent::text("inherited"),
                },
                "one".into(),
            )],
        )
        .await?;
    assert!(backend.get_processed_ids(&root).await?.contains("one"));
    let child = backend
        .spawn_thread(
            &root,
            ToolCallId::from_ref("child"),
            false,
            SpawnContext::InheritUpTo(1),
        )
        .await?;
    backend
        .append_messages(
            &root,
            vec![(
                InfinityMessage::User {
                    content: UserContent::text("later"),
                },
                "two".into(),
            )],
        )
        .await?;
    let (history, _, prefix) = backend.load_history_with_ancestors(&child).await?;
    assert_eq!(history.len(), 1);
    assert_eq!(prefix, 1);
    backend
        .save_compaction_summary(&root, "later summary", 2)
        .await?;
    assert_eq!(
        backend.load_history_with_ancestors(&child).await?.0.len(),
        1
    );
    let fresh = backend
        .spawn_thread(
            &root,
            ToolCallId::from_ref("fresh"),
            false,
            SpawnContext::Fresh,
        )
        .await?;
    assert!(
        backend
            .load_history_with_ancestors(&fresh)
            .await?
            .0
            .is_empty()
    );
    store.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn failed_history_transaction_rolls_back_prior_rows_and_dedup() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let store = RuntimeStore::open(
        directory.path().join("runtime.sqlite"),
        RuntimeLimits {
            max_input_bytes: 128,
            ..RuntimeLimits::default()
        },
    )?;
    store.create_root("root".into(), json!({})).await?;
    let backend = SqliteInfinityStore::new(store.clone());
    let root = ThreadId::from("root".to_owned());
    assert!(
        backend
            .append_messages(
                &root,
                vec![
                    (
                        InfinityMessage::User {
                            content: UserContent::text("valid")
                        },
                        "valid".into()
                    ),
                    (
                        InfinityMessage::User {
                            content: UserContent::text("x".repeat(256))
                        },
                        "large".into()
                    ),
                ]
            )
            .await
            .is_err()
    );
    assert!(
        backend
            .load_history_up_to(&root, None, None)
            .await?
            .is_empty()
    );
    assert!(backend.get_processed_ids(&root).await?.is_empty());
    store.shutdown().await?;
    Ok(())
}

struct TestProvider {
    chunks: Vec<StreamChunk>,
}

#[async_trait]
impl ModelProvider for TestProvider {
    async fn list_models(
        &self,
    ) -> Result<Vec<ModelEntry>, Box<dyn std::error::Error + Send + Sync>> {
        Ok(vec![model()])
    }
    async fn invoke_model(
        &self,
        _: &str,
        _: CompletionRequest,
    ) -> Result<ModelStream, CompletionError> {
        Ok(futures::stream::iter(self.chunks.clone().into_iter().map(Ok)).boxed())
    }
}

fn model() -> ModelEntry {
    ModelEntry {
        model_id: "test".into(),
        display_name: "Test".into(),
        context_window: 8192,
        max_output_tokens: Some(128),
        supports_image_input: false,
    }
}

#[tokio::test]
#[ignore = "explicit capacity scenario with 75,000 parked metadata records"]
async fn large_parked_fleet_uses_paged_metadata() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let store = RuntimeStore::open(
        directory.path().join("runtime.sqlite"),
        RuntimeLimits::default(),
    )?;
    store.create_root("root".into(), json!({})).await?;
    store.write(|connection| {
        let transaction=connection.transaction()?;
        {
            let mut statement=transaction.prepare("INSERT INTO agents(id,root_id,parent_id,spawn_key,parent_cutoff) VALUES (?,'root','root',?,0)")?;
            for index in 1..75_000 { statement.execute(rusqlite::params![format!("agent-{index:06}"),format!("spawn-{index}")])?; }
        }
        transaction.execute("UPDATE runs SET agent_count=75000 WHERE id='root'",[])?;
        transaction.commit()?;
        Ok(())
    }).await?;
    let started = std::time::Instant::now();
    let page = store.list_agents("root".into(), None, 64).await?;
    assert_eq!(page.agents.len(), 64);
    assert!(page.next_cursor.is_some());
    assert!(store.admit().await?.is_none());
    assert!(
        store
            .spawn_child("root".into(), "over-limit".into(), 0, false)
            .await
            .is_err()
    );
    println!(
        "75,000 parked records: a 64-row page and idle admission check in {:?}",
        started.elapsed()
    );
    store.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn automatic_delivery_cannot_cross_runs_or_change_its_target() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let store = RuntimeStore::open(
        directory.path().join("runtime.sqlite"),
        RuntimeLimits::default(),
    )?;
    store.create_root("a".into(), json!({})).await?;
    store.create_root("b".into(), json!({})).await?;
    assert!(
        store
            .enqueue_user_input("a".into(), "wrong".into(), input("b", "wrong target")?)
            .await
            .is_err()
    );
    store
        .enqueue_user_input("a".into(), "input".into(), input("a", "hello")?)
        .await?;
    let origin = store.admit().await?.context("lease")?;
    assert!(
        store
            .enqueue_from(
                origin,
                "b".into(),
                "cross-run".into(),
                input("b", "wrong run")?
            )
            .await
            .is_err()
    );
    store.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn real_infinity_slice_commits_a_completed_response() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let store = RuntimeStore::open(
        directory.path().join("runtime.sqlite"),
        RuntimeLimits::default(),
    )?;
    store.create_root("root".into(), json!({})).await?;
    store
        .enqueue_user_input("root".into(), "user".into(), input("root", "hello")?)
        .await?;
    let lease = store.admit().await?.context("lease")?;
    let slice = InfinitySlice {
        provider: Arc::new(TestProvider {
            chunks: vec![
                StreamChunk::Text("answer".into()),
                StreamChunk::Final(FinalResponse::default()),
            ],
        }),
        model: model(),
        tools: vec![],
        system_prompt: "Existing instructions".into(),
    };
    slice
        .execute(store.clone(), lease, &EventCollector::new())
        .await?;
    let backend = SqliteInfinityStore::new(store.clone());
    assert_eq!(
        backend
            .load_history_up_to(ThreadId::from_ref("root"), None, None)
            .await?
            .len(),
        2
    );
    assert_eq!(store.snapshot("root".into()).await?.state, "waiting");
    store.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn incomplete_inference_cannot_commit_partial_output() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let store = RuntimeStore::open(
        directory.path().join("runtime.sqlite"),
        RuntimeLimits::default(),
    )?;
    store.create_root("root".into(), json!({})).await?;
    store
        .enqueue_user_input("root".into(), "user".into(), input("root", "hello")?)
        .await?;
    let lease = store.admit().await?.context("lease")?;
    let slice = InfinitySlice {
        provider: Arc::new(TestProvider {
            chunks: vec![StreamChunk::Text("partial".into())],
        }),
        model: model(),
        tools: vec![],
        system_prompt: String::new(),
    };
    assert!(
        slice
            .execute(store.clone(), lease.clone(), &EventCollector::new())
            .await
            .is_err()
    );
    let backend = SqliteInfinityStore::new(store.clone());
    assert!(
        backend
            .load_history_up_to(ThreadId::from_ref("root"), None, None)
            .await?
            .is_empty()
    );
    assert!(
        backend
            .get_processed_ids(ThreadId::from_ref("root"))
            .await?
            .is_empty()
    );
    store.interrupt(lease).await?;
    assert_eq!(store.snapshot("root".into()).await?.state, "paused");
    store.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn tool_intent_is_durable_and_completion_is_idempotent() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let store = RuntimeStore::open(
        directory.path().join("runtime.sqlite"),
        RuntimeLimits::default(),
    )?;
    store.create_root("root".into(), json!({})).await?;
    store
        .enqueue_user_input("root".into(), "user".into(), input("root", "search")?)
        .await?;
    let lease = store.admit().await?.context("lease")?;
    let slice = InfinitySlice {
        provider: Arc::new(TestProvider {
            chunks: vec![
                StreamChunk::ToolCall(ToolCall::new("call", "search", json!({"query":"x"}))),
                StreamChunk::Final(FinalResponse::default()),
            ],
        }),
        model: model(),
        tools: vec![ToolDefinition {
            name: "search".into(),
            description: "Existing tool".into(),
            parameters: json!({"type":"object"}),
        }],
        system_prompt: String::new(),
    };
    slice
        .execute(store.clone(), lease, &EventCollector::new())
        .await?;
    let job_id = "root:0:call".to_owned();
    assert_eq!(store.job(job_id.clone()).await?.state, JobState::Queued);
    store.dispatch_job(job_id.clone()).await?;
    let result = input("root", "tool result")?;
    assert!(store.complete_job(job_id.clone(), result.clone()).await?);
    assert!(!store.complete_job(job_id.clone(), result).await?);
    assert!(
        store
            .complete_job(job_id, input("root", "different")?)
            .await
            .is_err()
    );
    assert_eq!(store.snapshot("root".into()).await?.state, "queued");
    store.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn pagination_does_not_load_transcripts_and_rejects_unbounded_pages() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let store = RuntimeStore::open(
        directory.path().join("runtime.sqlite"),
        RuntimeLimits::default(),
    )?;
    store.create_root("root".into(), json!({})).await?;
    for index in 0..10 {
        store
            .spawn_child("root".into(), format!("spawn-{index}"), 0, false)
            .await?;
    }
    let mut cursor = None;
    let mut seen = std::collections::HashSet::new();
    loop {
        let page = store.list_agents("root".into(), cursor, 3).await?;
        assert!(page.agents.len() <= 3);
        for agent in page.agents {
            assert!(seen.insert(agent.id));
        }
        cursor = page.next_cursor;
        if cursor.is_none() {
            break;
        }
    }
    assert_eq!(seen.len(), 11);
    assert!(store.list_agents("root".into(), None, 1000).await.is_err());
    store.shutdown().await?;
    Ok(())
}
