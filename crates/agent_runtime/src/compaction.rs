use anyhow::{Context as _, Result, ensure};
use infinity_agent_core::message::InfinityMessage;
use infinity_provider_protocol::{CompletionRequest, Message};
use language_model_core::{CompactedContext, LanguageModelRequest};
use rusqlite::{Connection, OptionalExtension as _, params};
use serde::{Deserialize, Serialize};

use crate::{
    RuntimeStore,
    history_budget::{DEFAULT_HISTORY_BYTES, HistoryBudget, load_bounded_history},
};

const MAXIMUM_CONTEXT_BYTES: usize = 16 * 1024 * 1024;

#[cfg(test)]
#[path = "compaction_tail_tests.rs"]
mod tail_tests;

pub(crate) fn record_context(
    connection: &Connection,
    agent: &str,
    cutoff: i64,
    context: &CompactedContext,
) -> Result<()> {
    let context = serde_json::to_string(context)?;
    ensure!(
        context.len() <= MAXIMUM_CONTEXT_BYTES,
        "compacted context exceeds its durable byte limit"
    );
    let coverage = safe_prefix_cutoff(connection, agent, cutoff)?;
    record_context_payload(connection, agent, coverage, &context)?;
    Ok(())
}

fn record_context_payload(
    connection: &Connection,
    agent: &str,
    cutoff: i64,
    context: &str,
) -> Result<()> {
    connection.execute("INSERT INTO runtime_compacted_contexts(agent_id,cutoff,context) VALUES (?,?,?) ON CONFLICT(agent_id,cutoff) DO UPDATE SET context=excluded.context",params![agent,cutoff,context])?;
    connection.execute("INSERT INTO runtime_compaction_heads(agent_id,cutoff) VALUES (?,?) ON CONFLICT(agent_id) DO UPDATE SET cutoff=excluded.cutoff",params![agent,cutoff])?;
    let exists: bool = connection.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE name='provider_usage')",
        [],
        |row| row.get(0),
    )?;
    let cursor: i64 = if exists {
        connection.query_row(
            "SELECT COALESCE(MAX(rowid),0) FROM provider_usage WHERE agent_id=?",
            [agent],
            |row| row.get(0),
        )?
    } else {
        0
    };
    connection.execute("INSERT INTO runtime_compaction_usage(agent_id,last_usage_rowid) VALUES (?,?) ON CONFLICT(agent_id) DO UPDATE SET last_usage_rowid=excluded.last_usage_rowid",params![agent,cursor])?;
    crate::provider_state::clear_context_overflow(connection, agent)?;
    Ok(())
}

fn safe_prefix_cutoff(connection: &Connection, agent: &str, end: i64) -> Result<i64> {
    let open:Option<i64> = connection.query_row("SELECT MIN(call.position) FROM history call WHERE call.agent_id=? AND call.position<? AND json_extract(call.payload,'$.kind')='tool_call' AND NOT EXISTS(SELECT 1 FROM history result WHERE result.agent_id=call.agent_id AND result.position>call.position AND result.position<? AND ((json_extract(result.payload,'$.kind') IN ('tool_result','subscription_event') AND json_extract(result.payload,'$.result.id')=json_extract(call.payload,'$.call.id')) OR (json_extract(result.payload,'$.kind')='user' AND json_extract(result.payload,'$.content.type')='toolresult' AND json_extract(result.payload,'$.content.id')=json_extract(call.payload,'$.call.id'))))",params![agent,end,end],|row|row.get(0))?;
    let Some(open) = open else {
        return Ok(end);
    };
    Ok(connection.query_row("SELECT COALESCE(MAX(position)+1,0) FROM history WHERE agent_id=? AND position<? AND json_extract(payload,'$.kind') IN ('user','tool_result','subscription_event')",params![agent,open],|row|row.get(0))?)
}

pub fn initialize(connection: &Connection) -> Result<()> {
    connection.execute_batch(
        "CREATE TABLE IF NOT EXISTS runtime_compaction_attempts (
        agent_id TEXT PRIMARY KEY REFERENCES agents(id), generation INTEGER NOT NULL,
        cutoff INTEGER NOT NULL, history_end INTEGER NOT NULL, lease_id TEXT, metadata TEXT, root_metadata TEXT
    ); CREATE TABLE IF NOT EXISTS runtime_compacted_contexts (
        agent_id TEXT NOT NULL REFERENCES agents(id), cutoff INTEGER NOT NULL,
        context TEXT NOT NULL, PRIMARY KEY(agent_id,cutoff)
    ); CREATE TABLE IF NOT EXISTS runtime_compaction_usage (
        agent_id TEXT PRIMARY KEY REFERENCES agents(id), last_usage_rowid INTEGER NOT NULL
    ); CREATE TABLE IF NOT EXISTS runtime_compaction_heads (
        agent_id TEXT PRIMARY KEY REFERENCES agents(id), cutoff INTEGER NOT NULL
    );",
    )?;
    Ok(())
}

pub struct CompactionSnapshot {
    pub cutoff: i64,
    pub full_history_end: i64,
    pub request: LanguageModelRequest,
    pub overflow: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CompactionCursor {
    pub offset: usize,
}

pub struct CompactionPage {
    pub cutoff: i64,
    pub messages: Vec<InfinityMessage>,
    pub next_cursor: Option<CompactionCursor>,
}

pub(crate) struct ContextCheckpoint {
    pub cutoff: i64,
    pub message: InfinityMessage,
}

pub(crate) fn latest_context(
    connection: &Connection,
    agent: &str,
    end: i64,
) -> Result<Option<ContextCheckpoint>> {
    let exists: bool = connection.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE name='runtime_compacted_contexts')",
        [],
        |row| row.get(0),
    )?;
    let has_heads: bool = connection.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE name='runtime_compaction_heads')",
        [],
        |row| row.get(0),
    )?;
    let head: Option<i64> = if has_heads {
        connection
            .query_row(
                "SELECT cutoff FROM runtime_compaction_heads WHERE agent_id=? AND cutoff<=?",
                params![agent, end],
                |row| row.get(0),
            )
            .optional()?
    } else {
        None
    };
    let native = if exists {
        if let Some(head) = head {
            connection.query_row("SELECT cutoff,context FROM runtime_compacted_contexts WHERE agent_id=? AND cutoff=?",params![agent,head],|row|Ok((row.get::<_,i64>(0)?,row.get::<_,String>(1)?))).optional()?
        } else {
            connection.query_row("SELECT cutoff,context FROM runtime_compacted_contexts WHERE agent_id=? AND cutoff<=? ORDER BY cutoff DESC LIMIT 1",params![agent,end],|row|Ok((row.get::<_,i64>(0)?,row.get::<_,String>(1)?))).optional()?
        }
    } else {
        None
    };
    let exists: bool = connection.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE name='summaries')",
        [],
        |row| row.get(0),
    )?;
    let legacy = if exists {
        connection.query_row("SELECT cutoff,summary FROM summaries WHERE agent_id=? AND cutoff<=? ORDER BY cutoff DESC LIMIT 1", params![agent,end], |row| Ok((row.get::<_,i64>(0)?,row.get::<_,String>(1)?))).optional()?
    } else {
        None
    };
    if let Some((cutoff, summary)) = legacy {
        if head.is_none() && native.as_ref().is_none_or(|native| cutoff > native.0) {
            return Ok(Some(ContextCheckpoint {
                cutoff,
                message: InfinityMessage::Assistant {
                    content: infinity_provider_protocol::AssistantContent::text(format!(
                        "[Compacted conversation summary]\n{summary}"
                    )),
                },
            }));
        }
    }
    native
        .map(|(cutoff, context)| {
            Ok(ContextCheckpoint {
                cutoff,
                message: crate::model::compacted_context_message(serde_json::from_str(&context)?)?,
            })
        })
        .transpose()
}

#[derive(Clone)]
struct Segment {
    agent: String,
    cutoff: i64,
    start: i64,
    marker: bool,
}

fn segments(connection: &Connection, agent: &str, leaf_end: Option<i64>) -> Result<Vec<Segment>> {
    let mut chain = Vec::new();
    let mut current = agent.to_owned();
    let mut end = leaf_end.unwrap_or(connection.query_row(
        "SELECT count(*) FROM history WHERE agent_id=?",
        [agent],
        |row| row.get(0),
    )?);
    let mut seen = std::collections::HashSet::new();
    loop {
        ensure!(
            seen.insert(current.clone()) && seen.len() <= 4096,
            "invalid or excessive history ancestry"
        );
        let context = latest_context(connection, &current, end)?;
        chain.push(Segment {
            agent: current.clone(),
            cutoff: end,
            start: context.as_ref().map_or(0, |context| context.cutoff),
            marker: context.is_some(),
        });
        if context.is_some() {
            break;
        }
        let (parent, cutoff, fresh): (Option<String>, Option<i64>, bool) = connection.query_row(
            "SELECT parent_id,parent_cutoff,fresh FROM agents WHERE id=?",
            [&current],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )?;
        if fresh {
            break;
        }
        let Some(parent) = parent else {
            break;
        };
        end = cutoff.context("ancestor has no immutable cutoff")?;
        current = parent;
    }
    chain.reverse();
    Ok(chain)
}

pub(crate) fn load_effective_history(
    connection: &Connection,
    agent: &str,
    budget: &HistoryBudget,
) -> Result<(Vec<InfinityMessage>, Option<i64>, usize)> {
    load_effective_history_up_to(connection, agent, None, budget)
}

fn load_effective_history_up_to(
    connection: &Connection,
    agent: &str,
    end: Option<i64>,
    budget: &HistoryBudget,
) -> Result<(Vec<InfinityMessage>, Option<i64>, usize)> {
    let chain = segments(connection, agent, end)?;
    let mut history = Vec::new();
    let mut prefix = 0;
    let mut leaf_cutoff = None;
    for segment in chain {
        if segment.agent == agent {
            prefix = history.len();
            if segment.marker {
                leaf_cutoff = Some(segment.start);
            }
        }
        history.extend(load_bounded_history(
            connection,
            &segment.agent,
            Some(segment.start),
            Some(segment.cutoff),
            budget,
        )?);
    }
    Ok((history, leaf_cutoff, prefix))
}

fn effective_bytes(connection: &Connection, agent: &str, cutoff: Option<i64>) -> Result<usize> {
    let mut bytes = 0usize;
    for segment in segments(connection, agent, cutoff)? {
        let rows:usize = connection.query_row("SELECT COALESCE(SUM(length(CAST(payload AS BLOB))+?),0) FROM history WHERE agent_id=? AND position>=? AND position<?",params![std::mem::size_of::<InfinityMessage>(),segment.agent,segment.start,segment.cutoff],|row|row.get(0))?;
        bytes = bytes.saturating_add(rows);
        if segment.marker {
            if let Some(marker) = latest_context(connection, &segment.agent, segment.cutoff)? {
                bytes = bytes
                    .saturating_add(serde_json::to_vec(&marker.message)?.len())
                    .saturating_add(std::mem::size_of::<InfinityMessage>());
            }
        }
    }
    Ok(bytes)
}

fn validate_attempt(connection: &Connection, agent: &str, generation: i64) -> Result<i64> {
    let attempt: (i64,i64,Option<String>,i64,String,Option<String>,i64) = connection.query_row("SELECT c.generation,c.cutoff,c.lease_id,a.generation,a.state,a.lease_id,c.history_end FROM runtime_compaction_attempts c JOIN agents a ON a.id=c.agent_id WHERE c.agent_id=?", [agent], |row| Ok((row.get(0)?,row.get(1)?,row.get(2)?,row.get(3)?,row.get(4)?,row.get(5)?,row.get(6)?)))?;
    ensure!(
        attempt.0 == generation
            && attempt.3 == generation
            && matches!(attempt.4.as_str(), "paused" | "running"),
        "compaction belongs to a stale or cancelled agent generation"
    );
    ensure!(attempt.2 == attempt.5, "compaction execution lease changed");
    let metadata_matches:bool=connection.query_row("SELECT a.metadata IS c.metadata AND (SELECT metadata FROM agents WHERE id=a.root_id) IS c.root_metadata FROM runtime_compaction_attempts c JOIN agents a ON a.id=c.agent_id WHERE c.agent_id=?",[agent],|row|row.get(0))?;
    ensure!(
        metadata_matches,
        "agent configuration changed during compaction"
    );
    let current: i64 = connection.query_row(
        "SELECT count(*) FROM history WHERE agent_id=?",
        [agent],
        |row| row.get(0),
    )?;
    ensure!(
        current == attempt.6,
        "history changed while compaction was running"
    );
    let busy: bool = connection.query_row("SELECT EXISTS(SELECT 1 FROM jobs WHERE agent_id=? AND state IN ('queued','dispatched','uncertain'))", [agent], |row| row.get(0))?;
    ensure!(
        !busy,
        "finish or reconcile pending tool effects before compacting this conversation"
    );
    Ok(attempt.1)
}

impl RuntimeStore {
    pub async fn effective_history_bytes(&self, agent: String) -> Result<usize> {
        self.read(move |connection| effective_bytes(connection, &agent, None))
            .await
    }

    pub async fn last_provider_usage(
        &self,
        agent: String,
    ) -> Result<Option<crate::provider_state::ProviderUsageRecord>> {
        self.read(move |connection| {
            let exists:bool=connection.query_row("SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE name='provider_usage')",[],|row|row.get(0))?;
            if !exists { return Ok(None); }
            Ok(connection.query_row("SELECT attempt_id,agent_id,model_id,input_tokens,output_tokens,cache_creation_tokens,cache_read_tokens,completed FROM provider_usage WHERE agent_id=? AND rowid>COALESCE((SELECT last_usage_rowid FROM runtime_compaction_usage WHERE agent_id=?),0) ORDER BY rowid DESC LIMIT 1",params![agent,agent],|row|Ok(crate::provider_state::ProviderUsageRecord { attempt_id:row.get(0)?,agent_id:row.get(1)?,model_id:row.get(2)?,usage:language_model_core::TokenUsage {input_tokens:row.get(3)?,output_tokens:row.get(4)?,cache_creation_input_tokens:row.get(5)?,cache_read_input_tokens:row.get(6)?},completed:row.get(7)? })).optional()?)
        }).await
    }

    pub async fn begin_compaction(&self, agent: String) -> Result<i64> {
        self.write(move |connection| {
            let transaction = connection.transaction()?;
            let changed = transaction.execute("UPDATE agents SET state='paused',pause_requested=1,generation=generation+1,lease_id=NULL WHERE id=? AND state NOT IN ('cancelled','completed','failed')", [&agent])?;
            ensure!(changed == 1, "no active conversation to compact");
            transaction.execute("UPDATE inbox SET lease_id=NULL WHERE agent_id=? AND processed=0", [&agent])?;
            transaction.execute("UPDATE jobs SET state='uncertain' WHERE agent_id=? AND state='dispatched'", [&agent])?;
            let generation: i64 = transaction.query_row("SELECT generation FROM agents WHERE id=?", [&agent], |row| row.get(0))?;
            transaction.execute("DELETE FROM runtime_compaction_attempts WHERE agent_id=?", [&agent])?;
            transaction.commit()?;
            Ok(generation)
        }).await
    }

    pub async fn compaction_snapshot(
        &self,
        agent: String,
        generation: i64,
    ) -> Result<CompactionSnapshot> {
        let (cutoff,full_history_end) = self.write({ let agent = agent.clone(); move |connection| {
            let transaction = connection.transaction()?;
            let (current,state,lease): (i64,String,Option<String>) = transaction.query_row("SELECT generation,state,lease_id FROM agents WHERE id=?", [&agent], |row| Ok((row.get(0)?,row.get(1)?,row.get(2)?)))?;
            ensure!(current == generation && matches!(state.as_str(),"paused"|"running"), "agent must be paused or hold its current execution lease to compact");
            let full_history_end: i64 = transaction.query_row("SELECT count(*) FROM history WHERE agent_id=?", [&agent], |row| row.get(0))?;
            let cutoff=safe_prefix_cutoff(&transaction,&agent,full_history_end)?;
            transaction.execute("INSERT INTO runtime_compaction_attempts(agent_id,generation,cutoff,history_end,lease_id,metadata,root_metadata) SELECT id,?,?,?,?,metadata,(SELECT metadata FROM agents root WHERE root.id=agents.root_id) FROM agents WHERE id=? ON CONFLICT(agent_id) DO UPDATE SET generation=excluded.generation,cutoff=excluded.cutoff,history_end=excluded.history_end,lease_id=excluded.lease_id,metadata=excluded.metadata,root_metadata=excluded.root_metadata", params![generation,cutoff,full_history_end,lease,agent])?;
            validate_attempt(&transaction,&agent,generation)?;
            transaction.commit()?;
            Ok((cutoff,full_history_end))
        }}).await?;
        self.read(move |connection| {
            validate_attempt(connection, &agent, generation)?;
            let bytes = effective_bytes(connection, &agent, Some(cutoff))?;
            if bytes > DEFAULT_HISTORY_BYTES {
                return Ok(CompactionSnapshot {
                    cutoff,
                    full_history_end,
                    request: LanguageModelRequest::default(),
                    overflow: true,
                });
            }
            let (history, _, _) = load_effective_history_up_to(
                connection,
                &agent,
                Some(cutoff),
                &HistoryBudget::new(DEFAULT_HISTORY_BYTES)?,
            )?;
            let request = request_from_messages(history)?;
            Ok(CompactionSnapshot {
                cutoff,
                full_history_end,
                request,
                overflow: false,
            })
        })
        .await
    }

    pub async fn compaction_page(
        &self,
        agent: String,
        generation: i64,
        after: Option<CompactionCursor>,
        maximum_bytes: usize,
    ) -> Result<CompactionPage> {
        ensure!(
            (1..=DEFAULT_HISTORY_BYTES).contains(&maximum_bytes),
            "compaction page budget must be 1–32 MiB"
        );
        self.read(move |connection| {
            let cutoff = validate_attempt(connection,&agent,generation)?;
            let after = after.map_or(0, |cursor| cursor.offset);
            let mut offset = 0usize;
            let mut bytes = 0usize;
            let mut messages = Vec::new();
            for segment in segments(connection,&agent,Some(cutoff))? {
                if segment.marker {
                    if offset >= after {
                        let marker = latest_context(connection,&segment.agent,segment.cutoff)?.context("compaction checkpoint disappeared")?.message;
                        let size = serde_json::to_vec(&marker)?.len() + std::mem::size_of::<InfinityMessage>();
                        ensure!(!messages.is_empty() || size <= maximum_bytes, "single compacted context exceeds the requested page budget");
                        if bytes.saturating_add(size) > maximum_bytes { return Ok(CompactionPage { cutoff,messages,next_cursor:Some(CompactionCursor {offset}) }); }
                        bytes += size; messages.push(marker);
                    }
                    offset += 1;
                }
                let skipped = after.saturating_sub(offset).min((segment.cutoff-segment.start).max(0) as usize);
                offset += skipped;
                let start = segment.start + skipped as i64;
                let mut statement = connection.prepare("SELECT payload,length(CAST(payload AS BLOB)) FROM history WHERE agent_id=? AND position>=? AND position<? ORDER BY position")?;
                let mut rows = statement.query(params![segment.agent,start,segment.cutoff])?;
                while let Some(row) = rows.next()? {
                    let size = row.get::<_,usize>(1)?.saturating_add(std::mem::size_of::<InfinityMessage>());
                    ensure!(!messages.is_empty() || size <= maximum_bytes, "single history item exceeds the requested page budget");
                    if bytes.saturating_add(size) > maximum_bytes { return Ok(CompactionPage { cutoff,messages,next_cursor:Some(CompactionCursor {offset}) }); }
                    messages.push(serde_json::from_str(row.get_ref(0)?.as_str()?)?);
                    bytes += size; offset += 1;
                }
            }
            ensure!(after <= offset, "compaction cursor exceeds the effective history");
            Ok(CompactionPage { cutoff,messages,next_cursor:None })
        }).await
    }

    pub async fn abort_compaction(&self, agent: String, generation: i64) -> Result<()> {
        self.write(move |connection| {
            connection.execute(
                "DELETE FROM runtime_compaction_attempts WHERE agent_id=? AND generation=?",
                params![agent, generation],
            )?;
            Ok(())
        })
        .await
    }

    pub async fn install_compaction(
        &self,
        agent: String,
        generation: i64,
        cutoff: i64,
        context: CompactedContext,
    ) -> Result<()> {
        let context = serde_json::to_string(&context)?;
        ensure!(
            context.len() <= MAXIMUM_CONTEXT_BYTES,
            "compacted context exceeds its durable byte limit"
        );
        self.write(move |connection| {
            let transaction = connection.transaction()?;
            ensure!(
                validate_attempt(&transaction, &agent, generation)? == cutoff,
                "compaction cutoff changed"
            );
            record_context_payload(&transaction, &agent, cutoff, &context)?;
            transaction.execute(
                "DELETE FROM runtime_compaction_attempts WHERE agent_id=? AND generation=?",
                params![agent, generation],
            )?;
            transaction.commit()?;
            Ok(())
        })
        .await
    }
}

pub fn request_from_messages(messages: Vec<InfinityMessage>) -> Result<LanguageModelRequest> {
    crate::model::convert_request(
        CompletionRequest {
            chat_history: messages
                .into_iter()
                .flat_map(InfinityMessage::into_messages)
                .collect::<Vec<Message>>(),
            ..Default::default()
        },
        LanguageModelRequest::default(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RuntimeLimits;
    use infinity_provider_protocol::UserContent;
    use language_model_core::MessageContent;

    async fn history(store: &RuntimeStore, agent: &str, count: usize) -> Result<()> {
        let agent = agent.to_owned();
        store.write(move |connection| {
            let transaction=connection.transaction()?;
            let start:i64=transaction.query_row("SELECT count(*) FROM history WHERE agent_id=?",[&agent],|row|row.get(0))?;
            for offset in 0..count {
                let position = start.checked_add(i64::try_from(offset)?).context("history position overflow")?;
                let message=InfinityMessage::User {content:UserContent::text(format!("history-{position}:{}","x".repeat(200)))};
                transaction.execute("INSERT INTO history(agent_id,position,message_id,payload) VALUES (?,?,?,?)",params![agent,position,format!("message-{position}"),serde_json::to_string(&message)?])?;
            }
            transaction.commit()?;Ok(())
        }).await
    }

    fn summary() -> CompactedContext {
        CompactedContext::Summary {
            content: "retained context".into(),
            provider_state: None,
        }
    }

    #[tokio::test]
    async fn compaction_preserves_archive_and_survives_paused_reopen() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("runtime.sqlite");
        let store = RuntimeStore::open(&path, RuntimeLimits::default())?;
        store
            .create_root("root".into(), serde_json::json!({}))
            .await?;
        history(&store, "root", 12).await?;
        let generation = store.begin_compaction("root".into()).await?;
        let snapshot = store.compaction_snapshot("root".into(), generation).await?;
        assert_eq!(snapshot.cutoff, 12);
        assert!(!snapshot.overflow);
        store
            .install_compaction("root".into(), generation, snapshot.cutoff, summary())
            .await?;
        assert_eq!(store.snapshot("root".into()).await?.state, "paused");
        let raw = store
            .read(|connection| {
                Ok(
                    connection.query_row("SELECT count(*) FROM history", [], |row| {
                        row.get::<_, usize>(0)
                    })?,
                )
            })
            .await?;
        assert_eq!(raw, 12);
        store.shutdown().await?;
        let reopened = RuntimeStore::open(path, RuntimeLimits::default())?;
        let (messages, cutoff, prefix) = reopened
            .read(|connection| {
                load_effective_history(connection, "root", &HistoryBudget::new(2048)?)
            })
            .await?;
        assert_eq!(messages.len(), 1);
        assert_eq!(cutoff, Some(12));
        assert_eq!(prefix, 0);
        let request = request_from_messages(messages)?;
        assert!(request.messages.iter().any(|message| {
            message
                .content
                .contains(&MessageContent::Compaction(summary()))
        }));
        assert_eq!(reopened.snapshot("root".into()).await?.state, "paused");
        reopened.shutdown().await
    }

    #[tokio::test]
    async fn stale_compaction_cannot_publish_after_cancel_or_replace_a_new_turn() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let store = RuntimeStore::open(
            directory.path().join("runtime.sqlite"),
            RuntimeLimits::default(),
        )?;
        store
            .create_root("root".into(), serde_json::json!({}))
            .await?;
        history(&store, "root", 1).await?;
        store
            .enqueue_text_input("root".into(), "input".into(), "next".into())
            .await?;
        let old = store.admit().await?.context("lease")?;
        let generation = store.begin_compaction("root".into()).await?;
        assert!(generation > old.generation);
        assert!(
            store
                .finish(old, serde_json::json!({}), crate::SliceOutcome::Waiting)
                .await
                .is_err()
        );
        let snapshot = store.compaction_snapshot("root".into(), generation).await?;
        store.cancel("root".into()).await?;
        assert!(
            store
                .install_compaction("root".into(), generation, snapshot.cutoff, summary())
                .await
                .is_err()
        );
        store.shutdown().await
    }

    #[tokio::test]
    async fn child_pages_include_fixed_ancestor_context_and_checkpoint_skips_it_within_budget()
    -> Result<()> {
        let directory = tempfile::tempdir()?;
        let store = RuntimeStore::open(
            directory.path().join("runtime.sqlite"),
            RuntimeLimits::default(),
        )?;
        store
            .create_root("root".into(), serde_json::json!({}))
            .await?;
        history(&store, "root", 8).await?;
        let child = store
            .spawn_child("root".into(), "child".into(), 8, false)
            .await?;
        history(&store, &child, 1).await?;
        let generation = store.begin_compaction(child.clone()).await?;
        let snapshot = store.compaction_snapshot(child.clone(), generation).await?;
        let mut cursor = None;
        let mut count = 0;
        loop {
            let page = store
                .compaction_page(child.clone(), generation, cursor, 1024)
                .await?;
            count += page.messages.len();
            cursor = page.next_cursor;
            if cursor.is_none() {
                break;
            }
        }
        assert_eq!(count, 9);
        store
            .install_compaction(child.clone(), generation, snapshot.cutoff, summary())
            .await?;
        history(&store, &child, 1).await?;
        let child_read = child.clone();
        let (messages, cutoff, prefix) = store
            .read(move |connection| {
                load_effective_history(connection, &child_read, &HistoryBudget::new(1024)?)
            })
            .await?;
        assert_eq!(messages.len(), 2);
        assert_eq!(cutoff, Some(1));
        assert_eq!(prefix, 0);
        store.shutdown().await
    }

    #[tokio::test]
    async fn automatic_compaction_retains_the_current_execution_lease() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let store = RuntimeStore::open(
            directory.path().join("runtime.sqlite"),
            RuntimeLimits::default(),
        )?;
        store
            .create_root("root".into(), serde_json::json!({}))
            .await?;
        history(&store, "root", 1).await?;
        store
            .enqueue_text_input("root".into(), "input".into(), "next".into())
            .await?;
        let lease = store.admit().await?.context("lease")?;
        let snapshot = store
            .compaction_snapshot("root".into(), lease.generation)
            .await?;
        store
            .install_compaction("root".into(), lease.generation, snapshot.cutoff, summary())
            .await?;
        assert_eq!(
            store.snapshot("root".into()).await?.generation,
            lease.generation
        );
        assert_eq!(store.snapshot("root".into()).await?.state, "running");
        store
            .finish(
                lease,
                serde_json::json!({"completed_step":true}),
                crate::SliceOutcome::Waiting,
            )
            .await?;
        store.shutdown().await
    }

    #[tokio::test]
    async fn completed_inline_provider_compaction_indexes_its_context_without_deleting_archive()
    -> Result<()> {
        use infinity_agent_core::{ThreadId, traits::ConversationStore as _};
        use language_model_core::{LanguageModelProviderId, ProviderCompactionState};
        let directory = tempfile::tempdir()?;
        let store = RuntimeStore::open(
            directory.path().join("runtime.sqlite"),
            RuntimeLimits::default(),
        )?;
        store
            .create_root("root".into(), serde_json::json!({}))
            .await?;
        history(&store, "root", 12).await?;
        let context = CompactedContext::ProviderState(ProviderCompactionState::new(
            LanguageModelProviderId("provider".into()),
            "native-format",
            "opaque-native-state",
        ));
        let backend = crate::SqliteInfinityStore::new(store.clone());
        backend
            .append_messages(
                ThreadId::from_ref("root"),
                vec![(
                    crate::model::compacted_context_message(context.clone())?,
                    "inline-context".into(),
                )],
            )
            .await?;
        let (messages, cutoff, _) = store
            .read(|connection| {
                load_effective_history(connection, "root", &HistoryBudget::new(2048)?)
            })
            .await?;
        assert_eq!(messages.len(), 1);
        assert_eq!(cutoff, Some(13));
        let request = request_from_messages(messages)?;
        assert!(request.messages.iter().any(|message| {
            message
                .content
                .contains(&MessageContent::Compaction(context.clone()))
        }));
        let raw = store
            .read(|connection| {
                Ok(
                    connection.query_row("SELECT count(*) FROM history", [], |row| {
                        row.get::<_, usize>(0)
                    })?,
                )
            })
            .await?;
        assert_eq!(raw, 13);
        store.shutdown().await
    }

    #[tokio::test]
    async fn a_child_keeps_its_original_parent_cutoff_after_parent_compaction() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let store = RuntimeStore::open(
            directory.path().join("runtime.sqlite"),
            RuntimeLimits::default(),
        )?;
        store
            .create_root("root".into(), serde_json::json!({}))
            .await?;
        history(&store, "root", 1).await?;
        let child = store
            .spawn_child("root".into(), "before-compaction".into(), 1, false)
            .await?;
        history(&store, "root", 2).await?;
        let generation = store.begin_compaction("root".into()).await?;
        let snapshot = store.compaction_snapshot("root".into(), generation).await?;
        store
            .install_compaction("root".into(), generation, snapshot.cutoff, summary())
            .await?;
        let (parent, _, _) = store
            .read(|connection| {
                load_effective_history(connection, "root", &HistoryBudget::new(2048)?)
            })
            .await?;
        assert!(matches!(
            parent.first(),
            Some(InfinityMessage::Assistant { .. })
        ));
        let (child, cutoff, prefix) = store
            .read(move |connection| {
                load_effective_history(connection, &child, &HistoryBudget::new(2048)?)
            })
            .await?;
        assert_eq!(child.len(), 1);
        assert!(matches!(child.first(), Some(InfinityMessage::User { .. })));
        assert_eq!(cutoff, None);
        assert_eq!(prefix, 1);
        store.shutdown().await
    }

    #[tokio::test]
    async fn a_checkpoint_consumes_old_usage_and_new_usage_is_visible() -> Result<()> {
        use crate::provider_state::ProviderUsageLedger;
        let directory = tempfile::tempdir()?;
        let store = RuntimeStore::open(
            directory.path().join("runtime.sqlite"),
            RuntimeLimits::default(),
        )?;
        store
            .create_root("root".into(), serde_json::json!({}))
            .await?;
        history(&store, "root", 1).await?;
        let ledger = ProviderUsageLedger::new(store.clone(), "root".into(), "model".into());
        ledger
            .attempt()
            .record(
                language_model_core::TokenUsage {
                    input_tokens: 50000,
                    ..Default::default()
                },
                true,
            )
            .await?;
        assert!(store.last_provider_usage("root".into()).await?.is_some());
        let generation = store.begin_compaction("root".into()).await?;
        let snapshot = store.compaction_snapshot("root".into(), generation).await?;
        store
            .install_compaction("root".into(), generation, snapshot.cutoff, summary())
            .await?;
        assert!(store.last_provider_usage("root".into()).await?.is_none());
        ledger
            .attempt()
            .record(
                language_model_core::TokenUsage {
                    input_tokens: 10,
                    ..Default::default()
                },
                true,
            )
            .await?;
        assert_eq!(
            store
                .last_provider_usage("root".into())
                .await?
                .context("new usage")?
                .usage
                .input_tokens,
            10
        );
        store.shutdown().await
    }

    #[tokio::test]
    async fn configuration_changes_invalidate_compaction_and_abort_releases_the_attempt()
    -> Result<()> {
        let directory = tempfile::tempdir()?;
        let store = RuntimeStore::open(
            directory.path().join("runtime.sqlite"),
            RuntimeLimits::default(),
        )?;
        store
            .create_root("root".into(), serde_json::json!({"model":"before"}))
            .await?;
        history(&store, "root", 1).await?;
        let generation = store.begin_compaction("root".into()).await?;
        let snapshot = store.compaction_snapshot("root".into(), generation).await?;
        store
            .write(|connection| {
                connection.execute(
                    "UPDATE agents SET metadata=? WHERE id='root'",
                    ["{\"model\":\"after\"}"],
                )?;
                Ok(())
            })
            .await?;
        assert!(
            store
                .install_compaction("root".into(), generation, snapshot.cutoff, summary())
                .await
                .is_err()
        );
        store.abort_compaction("root".into(), generation).await?;
        let count = store
            .read(|connection| {
                Ok(connection.query_row(
                    "SELECT count(*) FROM runtime_compaction_attempts",
                    [],
                    |row| row.get::<_, usize>(0),
                )?)
            })
            .await?;
        assert_eq!(count, 0);
        assert_eq!(store.snapshot("root".into()).await?.state, "paused");
        store.shutdown().await
    }
}
