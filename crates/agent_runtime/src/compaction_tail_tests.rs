use super::*;
use crate::RuntimeLimits;
use infinity_agent_core::{ThreadId, traits::ConversationStore as _};
use infinity_provider_protocol::{
    AssistantContent, Reasoning, ToolCall, ToolResult, ToolResultContent, UserContent,
};
use language_model_core::MessageContent;

#[tokio::test]
async fn compaction_retains_signed_pending_tool_tail_and_does_not_replay_completed_effects()
-> Result<()> {
    let directory = tempfile::tempdir()?;
    let store = RuntimeStore::open(
        directory.path().join("runtime.sqlite"),
        RuntimeLimits::default(),
    )?;
    store
        .create_root("root".into(), serde_json::json!({}))
        .await?;
    let backend = crate::SqliteInfinityStore::new(store.clone());
    let archive = vec![
        InfinityMessage::User {
            content: UserContent::text("earlier question"),
        },
        InfinityMessage::ToolCall {
            call: ToolCall::new(
                "previous-call",
                "read_file",
                serde_json::json!({"path":"repo/old"}),
            ),
            display_as: None,
        },
        InfinityMessage::ToolResult {
            result: ToolResult {
                id: "previous-call".into(),
                content: vec![ToolResultContent::text("previous result")],
                ..Default::default()
            },
            display_segments: None,
        },
        InfinityMessage::Assistant {
            content: AssistantContent::Reasoning(Reasoning::new_with_signature(
                "current signed reasoning",
                Some("current-signature".into()),
            )),
        },
        InfinityMessage::ToolCall {
            call: ToolCall::new(
                "pending-call",
                "read_file",
                serde_json::json!({"path":"repo/current"}),
            ),
            display_as: None,
        },
    ];
    backend
        .append_messages(
            ThreadId::from_ref("root"),
            archive
                .into_iter()
                .enumerate()
                .map(|(index, message)| (message, format!("history-{index}")))
                .collect(),
        )
        .await?;
    store
        .enqueue_text_input("root".into(), "start".into(), "continue".into())
        .await?;
    let lease = store.admit().await?.context("lease")?;
    store.record_job(lease.clone(),"effect".into(),serde_json::json!({"tool":"read_file","arguments":{"path":"repo/current"},"tool_call_id":"pending-call"})).await?;
    store
        .finish(lease, serde_json::json!({}), crate::SliceOutcome::Waiting)
        .await?;
    let job = store.dispatch_job("effect".into()).await?;
    store
        .complete_tool_job(
            job,
            vec![serde_json::json!({"type":"text","text":"current result"})],
            false,
            serde_json::Value::Null,
        )
        .await?;
    let completed = store.job("effect".into()).await?;
    let result = completed.result.clone().context("completed tool result")?;
    let next = store.admit().await?.context("tool-result slice")?;
    let snapshot = store
        .compaction_snapshot("root".into(), next.generation)
        .await?;
    assert_eq!(snapshot.full_history_end, 5);
    assert_eq!(snapshot.cutoff, 3);
    let prefix_calls = snapshot
        .request
        .messages
        .iter()
        .flat_map(|message| message.content.iter())
        .filter_map(|content| match content {
            MessageContent::ToolUse(tool) => Some(tool.id.to_string()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(prefix_calls, vec!["previous-call"]);
    assert!(snapshot.request.messages.iter().flat_map(|message|&message.content).any(|content|matches!(content,MessageContent::ToolResult(result) if result.tool_use_id.to_string()=="previous-call")));
    store
        .install_compaction(
            "root".into(),
            next.generation,
            snapshot.cutoff,
            CompactedContext::Summary {
                content: "earlier conversation".into(),
                provider_state: None,
            },
        )
        .await?;
    let (mut history, coverage, _) = store
        .read(|connection| load_effective_history(connection, "root", &HistoryBudget::new(4096)?))
        .await?;
    assert_eq!(coverage, Some(3));
    history.push(InfinityMessage::User {
        content: serde_json::from_value(result.get("content").context("tool content")?.clone())?,
    });
    let request = request_from_messages(history)?;
    assert!(request.messages.iter().flat_map(|message|&message.content).any(|content|matches!(content,MessageContent::Thinking {text,signature:Some(signature)} if text=="current signed reasoning" && signature=="current-signature")));
    assert!(request.messages.iter().flat_map(|message|&message.content).any(|content|matches!(content,MessageContent::ToolUse(tool) if tool.id.to_string()=="pending-call" && tool.name.as_ref()=="read_file")));
    assert!(request.messages.iter().flat_map(|message|&message.content).any(|content|matches!(content,MessageContent::ToolResult(result) if result.tool_use_id.to_string()=="pending-call" && result.tool_name.as_ref()=="read_file")));
    assert_eq!(
        store.job("effect".into()).await?.state,
        crate::JobState::Completed
    );
    assert_eq!(store.job("effect".into()).await?.result, completed.result);
    assert!(store.dispatch_job("effect".into()).await.is_err());
    let archive_count = store
        .read(|connection| {
            Ok(
                connection.query_row("SELECT count(*) FROM history", [], |row| {
                    row.get::<_, usize>(0)
                })?,
            )
        })
        .await?;
    assert_eq!(archive_count, 5);
    store.interrupt(next).await?;
    store.shutdown().await
}
