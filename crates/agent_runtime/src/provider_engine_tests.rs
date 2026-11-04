use std::sync::Arc;

use anyhow::{Context as _, Result};
use futures::{FutureExt as _, StreamExt as _};
use infinity_agent_core::{
    ThreadId, message::InfinityMessage, system::EventCollector, traits::ConversationStore,
};
use infinity_provider_protocol::{AssistantContent, ModelEntry, ReasoningContent, ToolDefinition};
use language_model_core::{
    LanguageModelCompletionEvent, LanguageModelRequest, LanguageModelToolUse,
    LanguageModelToolUseInput, MessageContent, PreparedLanguageModelCompletion, StopReason,
};
use serde_json::json;

use crate::{
    InfinitySlice, RuntimeLimits, RuntimeStore, SqliteInfinityStore,
    model::{ZedModelProvider, model_execution_channel},
};

#[tokio::test]
async fn actual_infinity_preserves_native_metadata_and_all_parallel_tool_results() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let store = RuntimeStore::open(
        directory.path().join("runtime.sqlite"),
        RuntimeLimits::default(),
    )?;
    store.create_root("root".into(), json!({})).await?;
    store
        .enqueue_text_input("root".into(), "user".into(), "inspect".into())
        .await?;
    let lease = store.admit().await?.context("first lease")?;
    let generation = lease.generation;
    let (executor, requests) = model_execution_channel(2)?;
    let entry = ModelEntry {
        model_id: "fake/model".into(),
        display_name: "Fake".into(),
        context_window: 10000,
        max_output_tokens: None,
        supports_image_input: false,
    };
    let provider = Arc::new(
        ZedModelProvider::new(entry.clone(), LanguageModelRequest::default(), executor)
            .with_state(store.clone(), "root".into()),
    );
    let slice = InfinitySlice {
        provider,
        model: entry,
        tools: vec![
            ToolDefinition {
                name: "read_file".into(),
                description: "read".into(),
                parameters: json!({"type":"object"}),
            },
            ToolDefinition {
                name: "search".into(),
                description: "search".into(),
                parameters: json!({"type":"object"}),
            },
        ],
        system_prompt: "instructions".into(),
    };
    let details = json!([{ "id": "opaque-reasoning", "encrypted_content": "state" }]);
    let events = vec![
        LanguageModelCompletionEvent::ToolUse(LanguageModelToolUse {
            id: "first".into(),
            name: "read_file".into(),
            raw_input: "{}".into(),
            input: LanguageModelToolUseInput::Json(json!({})),
            is_input_complete: true,
            thought_signature: Some("signature-first".into()),
        }),
        LanguageModelCompletionEvent::ToolUse(LanguageModelToolUse {
            id: "second".into(),
            name: "search".into(),
            raw_input: "{}".into(),
            input: LanguageModelToolUseInput::Json(json!({})),
            is_input_complete: true,
            thought_signature: Some("signature-second".into()),
        }),
        LanguageModelCompletionEvent::ReasoningDetails(details.clone()),
        LanguageModelCompletionEvent::Stop(StopReason::ToolUse),
    ];
    let responder = async {
        let request = requests
            .recv()
            .await
            .map_err(|error| anyhow::anyhow!("model preparation interrupted: {error}"))?;
        request.respond(Ok(PreparedLanguageModelCompletion::new(
            async move { Ok(futures::stream::iter(events.into_iter().map(Ok)).boxed()) }.boxed(),
        )));
        anyhow::Ok(())
    };
    let observer = EventCollector::new();
    let (outcome, response) =
        futures::join!(slice.execute(store.clone(), lease, &observer), responder);
    outcome?;
    response?;
    let backend = SqliteInfinityStore::new(store.clone());
    let history = backend
        .load_history_up_to(ThreadId::from_ref("root"), None, None)
        .await?;
    assert_eq!(
        history
            .iter()
            .filter(|message| matches!(message, InfinityMessage::ToolCall { .. }))
            .count(),
        2
    );
    assert!(history.iter().any(|message| matches!(message, InfinityMessage::Assistant { content: AssistantContent::Reasoning(reasoning) }
        if reasoning.content.iter().any(|content| matches!(content, ReasoningContent::ProviderState { value, .. } if value.get("kind").and_then(|kind| kind.as_str()) == Some("reasoning_details"))))));
    for id in ["first", "second"] {
        let job_id = format!("root:{generation}:{id}");
        let job = store.dispatch_job(job_id).await?;
        store
            .complete_tool_job(
                job,
                vec![json!({"type":"text", "text":"result"})],
                id == "first",
                json!({"source":id}),
            )
            .await?;
        if id == "first" {
            assert!(
                store.admit().await?.is_none(),
                "a model round must wait for all sibling tool results"
            );
        }
    }
    let lease = store.admit().await?.context("result lease")?;
    assert_eq!(lease.inputs.len(), 2);
    let responder = async {
        let request = requests
            .recv()
            .await
            .map_err(|error| anyhow::anyhow!("model preparation interrupted: {error}"))?;
        assert_eq!(store.metadata("root".into()).await?, json!({}));
        let mut tool_calls = 0;
        let mut tool_results = 0;
        let mut reasoning = false;
        for message in &request.request.messages {
            if message.reasoning_details.as_deref() == Some(&details) {
                reasoning = true;
            }
            for content in &message.content {
                match content {
                    MessageContent::ToolUse(tool) => {
                        tool_calls += 1;
                        assert!(tool.thought_signature.is_some());
                    }
                    MessageContent::ToolResult(result) => {
                        tool_results += 1;
                        let id = result.tool_use_id.to_string();
                        assert_eq!(result.is_error, id == "first");
                        assert_eq!(result.output, Some(json!({"source": id})));
                    }
                    _ => {}
                }
            }
        }
        assert_eq!(tool_calls, 2);
        assert_eq!(tool_results, 2);
        assert!(reasoning);
        request.respond(Ok(PreparedLanguageModelCompletion::new(
            async move {
                Ok(futures::stream::iter([
                    Ok(LanguageModelCompletionEvent::Text("done".into())),
                    Ok(LanguageModelCompletionEvent::Stop(StopReason::EndTurn)),
                ])
                .boxed())
            }
            .boxed(),
        )));
        anyhow::Ok(())
    };
    let (outcome, response) =
        futures::join!(slice.execute(store.clone(), lease, &observer), responder);
    outcome?;
    response?;
    let records = store.provider_usage("root".into(), None, 10).await?;
    assert_eq!(records.len(), 2);
    assert!(records.iter().all(|record| record.completed));
    store.shutdown().await
}
