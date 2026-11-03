use std::{
    collections::HashSet,
    fmt,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use anyhow::{Result, ensure};
use async_trait::async_trait;
use futures::StreamExt as _;
use infinity_agent_core::{
    ThreadId,
    message::{InfinityMessage, InputMessage},
    system::{
        AgentSystemBuilder, NoDeferral, StaticModel, StepOutcome, ThreadObserver, UserChoice,
    },
    tools::{Tool, ToolContext},
    traits::{ConversationStore, InputSender, SpawnContext, StateStore},
};
use infinity_provider_protocol::{
    CompletionError, CompletionRequest, ModelEntry, ModelProvider, ModelStream, StreamChunk,
    ToolDefinition,
};
use rap_protocol::{ChoiceId, ProviderCallId, ToolCallId};
use rusqlite::{Connection, OptionalExtension as _, params};
use serde_json::Value;

use crate::{
    Lease, RuntimeStore, SliceOutcome,
    store::{check_lease, spawn_child},
};

#[derive(Debug)]
pub struct StoreError(anyhow::Error);

impl From<anyhow::Error> for StoreError {
    fn from(error: anyhow::Error) -> Self {
        Self(error)
    }
}

impl fmt::Display for StoreError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

impl std::error::Error for StoreError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.0.as_ref())
    }
}

#[derive(Clone)]
pub struct SqliteInfinityStore {
    store: RuntimeStore,
    origin: Option<Lease>,
    response_complete: Option<Arc<AtomicBool>>,
    tool_names: Arc<HashSet<String>>,
}

impl SqliteInfinityStore {
    pub fn new(store: RuntimeStore) -> Self {
        Self {
            store,
            origin: None,
            response_complete: None,
            tool_names: Arc::default(),
        }
    }

    async fn flag(&self, thread: &ThreadId<str>, column: &'static str) -> Result<bool, StoreError> {
        let thread = thread.as_str().to_owned();
        self.store
            .read(move |connection| {
                Ok(connection.query_row(
                    &format!("SELECT {column} FROM agents WHERE id=?"),
                    [&thread],
                    |row| row.get(0),
                )?)
            })
            .await
            .map_err(Into::into)
    }

    async fn mutate(
        &self,
        operation: impl FnOnce(&mut Connection) -> Result<()> + Send + 'static,
    ) -> Result<(), StoreError> {
        let origin = self.origin.clone();
        self.store
            .write(move |connection| {
                if let Some(origin) = &origin {
                    check_lease(connection, origin)?;
                }
                operation(connection)
            })
            .await
            .map_err(Into::into)
    }
}

#[async_trait]
impl ConversationStore for SqliteInfinityStore {
    type Error = StoreError;

    async fn ensure_root_thread(&self, thread: &ThreadId<str>) -> Result<(), Self::Error> {
        let id = thread.as_str().to_owned();
        // Loading a child must not accidentally create a root with the same ID.
        if self.thread_exists(thread).await? {
            return Ok(());
        }
        self.store
            .create_root(id, Value::Null)
            .await
            .map_err(Into::into)
    }

    async fn thread_exists(&self, thread: &ThreadId<str>) -> Result<bool, Self::Error> {
        let id = thread.as_str().to_owned();
        self.store
            .read(move |connection| {
                Ok(connection.query_row(
                    "SELECT EXISTS(SELECT 1 FROM agents WHERE id=?)",
                    [&id],
                    |row| row.get(0),
                )?)
            })
            .await
            .map_err(Into::into)
    }

    async fn load_history_up_to(
        &self,
        thread: &ThreadId<str>,
        start: Option<i64>,
        end: Option<i64>,
    ) -> Result<Vec<InfinityMessage>, Self::Error> {
        let id = thread.as_str().to_owned();
        self.store.read(move |connection| {
            ensure!(start.unwrap_or(0) >= 0 && end.unwrap_or(i64::MAX) >= 0, "negative history boundary");
            let mut statement = connection.prepare("SELECT payload FROM history WHERE agent_id=? AND position>=? AND position<? ORDER BY position")?;
            let payloads = statement.query_map(params![id,start.unwrap_or(0),end.unwrap_or(i64::MAX)], |row| row.get::<_,String>(0))?.collect::<rusqlite::Result<Vec<_>>>()?;
            payloads.into_iter().map(|payload| Ok(serde_json::from_str(&payload)?)).collect()
        }).await.map_err(Into::into)
    }

    async fn append_messages(
        &self,
        thread: &ThreadId<str>,
        messages: Vec<(InfinityMessage, String)>,
    ) -> Result<(), Self::Error> {
        let id = thread.as_str().to_owned();
        let complete = self.response_complete.clone();
        let origin = self.origin.clone();
        let tool_names = self.tool_names.clone();
        let maximum_bytes = self.store.limits().max_input_bytes;
        self.mutate(move |connection| {
            if let Some(complete) = complete { ensure!(complete.load(Ordering::Acquire), "incomplete model output cannot be checkpointed"); }
            let transaction = connection.transaction()?;
            let mut position: i64 = transaction.query_row("SELECT coalesce(max(position)+1,0) FROM history WHERE agent_id=?", [&id], |row| row.get(0))?;
            for (message, message_id) in messages {
                let payload = serde_json::to_string(&message)?;
                ensure!(payload.len() <= maximum_bytes, "history item exceeds byte limit");
                let existing: Option<String> = transaction.query_row("SELECT payload FROM history WHERE agent_id=? AND message_id=?", params![id,message_id], |row| row.get(0)).optional()?;
                if let Some(existing) = existing { ensure!(existing == payload, "history dedup ID reused with different content"); }
                else {
                    transaction.execute("INSERT INTO history(agent_id,position,message_id,payload) VALUES (?,?,?,?)", params![id,position,message_id,payload])?;
                    position += 1;
                }
                if let (Some(origin), InfinityMessage::ToolCall {call,..}) = (&origin, &message)
                    && tool_names.contains(&call.function.name)
                {
                    let job_id = format!("{}:{}:{}",id,origin.generation,call.id);
                    let intent = serde_json::to_string(&serde_json::json!({"tool":call.function.name,"arguments":call.function.arguments,"tool_call_id":call.id,"provider_call_id":call.call_id}))?;
                    transaction.execute("INSERT OR IGNORE INTO jobs(id,agent_id,generation,state,payload) VALUES (?,?,?,'queued',?)",params![job_id,id,origin.generation,intent])?;
                }
                // Infinity currently writes these IDs in a separate call and ignores its
                // error. Commit both here so that gap cannot lose durable deduplication.
                if matches!(message, InfinityMessage::User{..}|InfinityMessage::SubscriptionEvent{..}) {
                    transaction.execute("INSERT OR IGNORE INTO processed_inputs(agent_id,message_id) VALUES (?,?)", params![id,message_id])?;
                }
            }
            transaction.commit()?;
            Ok(())
        }).await
    }

    async fn spawn_thread(
        &self,
        parent: &ThreadId<str>,
        call: &ToolCallId<str>,
        subscription: bool,
        context: SpawnContext,
    ) -> Result<ThreadId, Self::Error> {
        let parent = parent.as_str().to_owned();
        let key = call.as_str().to_owned();
        let (cutoff, fresh) = match context {
            SpawnContext::InheritUpTo(cutoff) => (
                i64::try_from(cutoff).map_err(|error| StoreError(error.into()))?,
                false,
            ),
            SpawnContext::Fresh => (0, true),
        };
        let origin = self.origin.clone();
        let maximum = self.store.limits().max_agents_per_run;
        self.store
            .write(move |connection| {
                if let Some(origin) = &origin {
                    check_lease(connection, origin)?;
                }
                Ok(ThreadId::from(spawn_child(
                    connection,
                    &parent,
                    &key,
                    cutoff,
                    fresh,
                    subscription,
                    maximum,
                )?))
            })
            .await
            .map_err(Into::into)
    }

    async fn is_thread_closed(&self, thread: &ThreadId<str>) -> Result<bool, Self::Error> {
        self.flag(thread, "closed").await
    }
    async fn is_subscription_event_thread(
        &self,
        thread: &ThreadId<str>,
    ) -> Result<bool, Self::Error> {
        self.flag(thread, "subscription_event").await
    }
    async fn is_fresh_context_thread(&self, thread: &ThreadId<str>) -> Result<bool, Self::Error> {
        self.flag(thread, "fresh").await
    }
    async fn is_compaction_thread(&self, thread: &ThreadId<str>) -> Result<bool, Self::Error> {
        self.flag(thread, "compaction").await
    }

    async fn close_thread(&self, thread: &ThreadId<str>) -> Result<(), Self::Error> {
        let id = thread.as_str().to_owned();
        self.mutate(move |connection| {
            connection.execute("UPDATE agents SET closed=1 WHERE id=?", [&id])?;
            Ok(())
        })
        .await
    }

    async fn mark_thread_as_compaction(&self, thread: &ThreadId<str>) -> Result<(), Self::Error> {
        let id = thread.as_str().to_owned();
        self.mutate(move |connection| {
            connection.execute("UPDATE agents SET compaction=1 WHERE id=?", [&id])?;
            Ok(())
        })
        .await
    }

    async fn get_thread_parent_info(
        &self,
        thread: &ThreadId<str>,
    ) -> Result<Option<(ThreadId, ToolCallId)>, Self::Error> {
        let id = thread.as_str().to_owned();
        self.store
            .read(move |connection| {
                let pair: (Option<String>, Option<String>) = connection.query_row(
                    "SELECT parent_id,spawn_key FROM agents WHERE id=?",
                    [&id],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )?;
                Ok(pair
                    .0
                    .zip(pair.1)
                    .map(|(parent, call)| (ThreadId::from(parent), ToolCallId::from(call))))
            })
            .await
            .map_err(Into::into)
    }

    async fn get_ancestor_chain(
        &self,
        thread: &ThreadId<str>,
    ) -> Result<Vec<(ThreadId, i64)>, Self::Error> {
        let mut id = thread.as_str().to_owned();
        self.store
            .read(move |connection| {
                let mut chain = Vec::new();
                let mut visited = HashSet::new();
                while visited.insert(id.clone()) {
                    let (parent, cutoff): (Option<String>, Option<i64>) = connection.query_row(
                        "SELECT parent_id,parent_cutoff FROM agents WHERE id=?",
                        [&id],
                        |row| Ok((row.get(0)?, row.get(1)?)),
                    )?;
                    let Some(parent) = parent else {
                        chain.reverse();
                        return Ok(chain);
                    };
                    chain.push((ThreadId::from(parent.clone()), cutoff.unwrap_or(0)));
                    id = parent;
                }
                anyhow::bail!("cycle in agent history ancestry")
            })
            .await
            .map_err(Into::into)
    }

    async fn get_thread_spawn_order(
        &self,
        thread: &ThreadId<str>,
    ) -> Result<Option<i64>, Self::Error> {
        let id = thread.as_str().to_owned();
        self.store
            .read(move |connection| {
                Ok(connection.query_row(
                    "SELECT parent_cutoff FROM agents WHERE id=?",
                    [&id],
                    |row| row.get(0),
                )?)
            })
            .await
            .map_err(Into::into)
    }

    async fn save_compaction_summary(
        &self,
        thread: &ThreadId<str>,
        summary: &str,
        cutoff: i64,
    ) -> Result<(), Self::Error> {
        let id = thread.as_str().to_owned();
        let summary = summary.to_owned();
        self.mutate(move |connection| { ensure!(cutoff >= 0, "negative compaction cutoff"); connection.execute("INSERT INTO summaries(agent_id,cutoff,summary) VALUES (?,?,?) ON CONFLICT(agent_id,cutoff) DO UPDATE SET summary=excluded.summary", params![id,cutoff,summary])?; Ok(()) }).await
    }

    async fn load_latest_compaction_summary_up_to(
        &self,
        thread: &ThreadId<str>,
        cutoff: Option<i64>,
    ) -> Result<Option<(String, i64)>, Self::Error> {
        let id = thread.as_str().to_owned();
        self.store.read(move |connection| Ok(connection.query_row("SELECT summary,cutoff FROM summaries WHERE agent_id=? AND cutoff<=? ORDER BY cutoff DESC LIMIT 1", params![id,cutoff.unwrap_or(i64::MAX)], |row| Ok((row.get(0)?,row.get(1)?))).optional()?)).await.map_err(Into::into)
    }
}

#[async_trait]
impl StateStore for SqliteInfinityStore {
    type Error = StoreError;

    async fn get_processed_ids(
        &self,
        thread: &ThreadId<str>,
    ) -> Result<HashSet<String>, Self::Error> {
        let id = thread.as_str().to_owned();
        self.store
            .read(move |connection| {
                let mut statement = connection
                    .prepare("SELECT message_id FROM processed_inputs WHERE agent_id=?")?;
                Ok(statement
                    .query_map([id], |row| row.get(0))?
                    .collect::<rusqlite::Result<_>>()?)
            })
            .await
            .map_err(Into::into)
    }

    async fn add_processed_message_ids(
        &self,
        thread: &ThreadId<str>,
        ids: Vec<String>,
    ) -> Result<(), Self::Error> {
        let id = thread.as_str().to_owned();
        self.mutate(move |connection| {
            let transaction = connection.transaction()?;
            for message_id in ids {
                transaction.execute(
                    "INSERT OR IGNORE INTO processed_inputs(agent_id,message_id) VALUES (?,?)",
                    params![id, message_id],
                )?;
            }
            transaction.commit()?;
            Ok(())
        })
        .await
    }

    async fn get_metadata(&self, thread: &ThreadId<str>) -> Result<Option<Value>, Self::Error> {
        let id = thread.as_str().to_owned();
        self.store
            .read(move |connection| {
                let value: Option<String> = connection.query_row(
                    "SELECT metadata FROM agents WHERE id=?",
                    [&id],
                    |row| row.get(0),
                )?;
                Ok(value
                    .map(|value| serde_json::from_str(&value))
                    .transpose()?)
            })
            .await
            .map_err(Into::into)
    }

    async fn set_metadata(&self, thread: &ThreadId<str>, value: Value) -> Result<(), Self::Error> {
        let id = thread.as_str().to_owned();
        let value = self.store.encode(&value).map_err(StoreError::from)?;
        self.mutate(move |connection| {
            connection.execute(
                "UPDATE agents SET metadata=? WHERE id=?",
                params![value, id],
            )?;
            Ok(())
        })
        .await
    }

    async fn get_active_subscriptions(
        &self,
        thread: &ThreadId<str>,
    ) -> Result<Vec<ToolCallId>, Self::Error> {
        let id = thread.as_str().to_owned();
        self.store
            .read(move |connection| {
                let mut statement = connection.prepare(
                    "SELECT call_id FROM subscriptions WHERE agent_id=? ORDER BY call_id",
                )?;
                Ok(statement
                    .query_map([id], |row| row.get::<_, String>(0))?
                    .map(|result| result.map(ToolCallId::from))
                    .collect::<rusqlite::Result<_>>()?)
            })
            .await
            .map_err(Into::into)
    }

    async fn add_active_subscription(
        &self,
        thread: &ThreadId<str>,
        call: &ToolCallId<str>,
    ) -> Result<(), Self::Error> {
        let id = thread.as_str().to_owned();
        let call = call.as_str().to_owned();
        self.mutate(move |connection| {
            connection.execute(
                "INSERT OR IGNORE INTO subscriptions(agent_id,call_id) VALUES (?,?)",
                params![id, call],
            )?;
            Ok(())
        })
        .await
    }

    async fn remove_active_subscription(
        &self,
        thread: &ThreadId<str>,
        call: &ToolCallId<str>,
    ) -> Result<(), Self::Error> {
        let id = thread.as_str().to_owned();
        let call = call.as_str().to_owned();
        self.mutate(move |connection| {
            connection.execute(
                "DELETE FROM subscriptions WHERE agent_id=? AND call_id=?",
                params![id, call],
            )?;
            Ok(())
        })
        .await
    }

    async fn add_pending_user_choice(
        &self,
        thread: &ThreadId<str>,
        choice: UserChoice,
    ) -> Result<(), Self::Error> {
        let id = thread.as_str().to_owned();
        let choice_id = choice.id.as_str().to_owned();
        let payload = serde_json::to_string(&choice).map_err(|error| StoreError(error.into()))?;
        self.mutate(move |connection| { connection.execute("INSERT INTO choices(agent_id,choice_id,payload) VALUES (?,?,?) ON CONFLICT(agent_id,choice_id) DO UPDATE SET payload=excluded.payload", params![id,choice_id,payload])?; Ok(()) }).await
    }

    async fn remove_pending_user_choice(
        &self,
        thread: &ThreadId<str>,
        choice: &ChoiceId<str>,
    ) -> Result<(), Self::Error> {
        let id = thread.as_str().to_owned();
        let choice = choice.as_str().to_owned();
        self.mutate(move |connection| {
            connection.execute(
                "DELETE FROM choices WHERE agent_id=? AND choice_id=?",
                params![id, choice],
            )?;
            Ok(())
        })
        .await
    }

    async fn get_pending_user_choices(
        &self,
        thread: &ThreadId<str>,
    ) -> Result<Vec<UserChoice>, Self::Error> {
        let id = thread.as_str().to_owned();
        self.store
            .read(move |connection| {
                let mut statement = connection
                    .prepare("SELECT payload FROM choices WHERE agent_id=? ORDER BY choice_id")?;
                let values = statement
                    .query_map([id], |row| row.get::<_, String>(0))?
                    .collect::<rusqlite::Result<Vec<_>>>()?;
                values
                    .into_iter()
                    .map(|value| Ok(serde_json::from_str(&value)?))
                    .collect()
            })
            .await
            .map_err(Into::into)
    }

    async fn is_thread_stopped(&self, thread: &ThreadId<str>) -> Result<bool, Self::Error> {
        let id = thread.as_str().to_owned();
        self.store
            .read(move |connection| {
                Ok(connection.query_row(
                    "SELECT state='cancelled' FROM agents WHERE id=?",
                    [&id],
                    |row| row.get(0),
                )?)
            })
            .await
            .map_err(Into::into)
    }
}

#[async_trait]
impl InputSender for SqliteInfinityStore {
    type Error = StoreError;

    async fn send_to_input_queue(
        &self,
        message: InputMessage,
        dedup: &str,
    ) -> Result<(), Self::Error> {
        let origin = self.origin.clone().ok_or_else(|| {
            StoreError(anyhow::anyhow!(
                "automatic input requires an execution origin"
            ))
        })?;
        let target = message.group_id.as_str().to_owned();
        let input = serde_json::to_value(message).map_err(|error| StoreError(error.into()))?;
        self.store
            .enqueue_from(origin, target, dedup.to_owned(), input)
            .await
            .map_err(StoreError::from)?;
        Ok(())
    }
}

struct CompletedResponseProvider {
    provider: Arc<dyn ModelProvider>,
    complete: Arc<AtomicBool>,
}

#[async_trait]
impl ModelProvider for CompletedResponseProvider {
    async fn list_models(
        &self,
    ) -> Result<Vec<ModelEntry>, Box<dyn std::error::Error + Send + Sync>> {
        self.provider.list_models().await
    }

    async fn invoke_model(
        &self,
        model: &str,
        request: CompletionRequest,
    ) -> Result<ModelStream, CompletionError> {
        self.complete.store(false, Ordering::Release);
        let stream = self
            .provider
            .invoke_model(model, request)
            .await
            .map_err(|error| {
                CompletionError::provider(
                    infinity_provider_protocol::ErrorClass::Fatal,
                    error.to_string(),
                )
            })?;
        let complete = self.complete.clone();
        Ok(futures::stream::unfold(
            (stream, false, complete),
            |(mut stream, ended, complete)| async move {
                if ended {
                    return None;
                }
                let chunk = stream.next().await;
                let (chunk, ended) = match chunk {
                    Some(Ok(StreamChunk::Final(response))) => {
                        complete.store(true, Ordering::Release);
                        (Ok(StreamChunk::Final(response)), false)
                    }
                    Some(Ok(chunk)) => (Ok(chunk), false),
                    Some(Err(error)) => {
                        complete.store(false, Ordering::Release);
                        (
                            Err(CompletionError::provider(
                                infinity_provider_protocol::ErrorClass::Fatal,
                                error.to_string(),
                            )),
                            true,
                        )
                    }
                    None if !complete.load(Ordering::Acquire) => (
                        Err(CompletionError::provider(
                            infinity_provider_protocol::ErrorClass::Fatal,
                            "model stream ended without a final response",
                        )),
                        true,
                    ),
                    None => return None,
                };
                Some((chunk, (stream, ended, complete)))
            },
        )
        .boxed())
    }
}

struct QueuedTool {
    definition: ToolDefinition,
    store: RuntimeStore,
    lease: Lease,
}

#[async_trait]
impl Tool<SqliteInfinityStore> for QueuedTool {
    fn name(&self) -> &str {
        &self.definition.name
    }
    fn description(&self) -> &str {
        &self.definition.description
    }
    fn parameters(&self) -> Value {
        self.definition.parameters.clone()
    }

    async fn execute(
        &self,
        args: Value,
        id: ToolCallId,
        call_id: Option<ProviderCallId>,
        context: &ToolContext<SqliteInfinityStore>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let job_id = format!(
            "{}:{}:{}",
            context.group_id.as_str(),
            self.lease.generation,
            id.as_str()
        );
        self.store.record_job(self.lease.clone(),job_id,serde_json::json!({"tool":self.definition.name,"arguments":args,"tool_call_id":id.as_str(),"provider_call_id":call_id.map(|id| id.into_inner())})).await?;
        Ok(())
    }
}

pub struct InfinitySlice {
    pub provider: Arc<dyn ModelProvider>,
    pub model: ModelEntry,
    pub tools: Vec<ToolDefinition>,
    pub system_prompt: String,
}

impl InfinitySlice {
    pub async fn execute<O: ThreadObserver>(
        &self,
        store: RuntimeStore,
        lease: Lease,
        observer: &O,
    ) -> Result<()> {
        let complete = Arc::new(AtomicBool::new(false));
        let provider = Arc::new(CompletedResponseProvider {
            provider: self.provider.clone(),
            complete: complete.clone(),
        });
        let backend = SqliteInfinityStore {
            store: store.clone(),
            origin: Some(lease.clone()),
            response_complete: Some(complete.clone()),
            tool_names: Arc::new(self.tools.iter().map(|tool| tool.name.clone()).collect()),
        };
        let tools = self
            .tools
            .iter()
            .cloned()
            .map(|definition| {
                Box::new(QueuedTool {
                    definition,
                    store: store.clone(),
                    lease: lease.clone(),
                }) as Box<dyn Tool<SqliteInfinityStore>>
            })
            .collect::<Vec<_>>();
        // Built-in synchronous tools may loop back into inference indefinitely in one
        // step. Every tool exposed here instead records a durable asynchronous job.
        let mut system = AgentSystemBuilder::new(
            backend.clone(),
            backend.clone(),
            StaticModel::from_entry(provider, &self.model),
            backend,
        )
        .without_builtin_tools()
        .tools(tools)
        .extra_system_prompt(self.system_prompt.clone())
        .build();
        let inputs = lease
            .inputs
            .iter()
            .map(|(id, payload)| Ok((serde_json::from_value(payload.clone())?, id.clone())))
            .collect::<Result<Vec<(InputMessage, String)>>>()?;
        let outcomes = system
            .step(inputs, observer, &mut NoDeferral)
            .await
            .map_err(|error| anyhow::anyhow!("Infinity slice failed: {error}"))?;
        for (_, outcome) in outcomes {
            if matches!(outcome, StepOutcome::Completed { .. }) {
                ensure!(
                    complete.load(Ordering::Acquire),
                    "model stream ended without a complete response"
                );
            }
        }
        store
            .finish(
                lease,
                serde_json::json!({"completed_step":true}),
                SliceOutcome::Waiting,
            )
            .await
    }
}
