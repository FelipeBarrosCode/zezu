use anyhow::{Result, ensure};
use rusqlite::{Connection, OptionalExtension as _, params};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{AgentPage, AgentSnapshot, Lease, RuntimeStore};

const MAX_TRANSCRIPT_PAGE_BYTES: usize = 4 * 1024 * 1024;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TranscriptEntry {
    pub position: i64,
    pub payload: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HistoryReference {
    pub agent_id: String,
    pub cutoff: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TranscriptPage {
    pub entries: Vec<TranscriptEntry>,
    pub next_cursor: Option<i64>,
    pub inherited: Option<HistoryReference>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct FleetSummary {
    pub total: usize,
    pub queued: usize,
    pub running: usize,
    pub waiting: usize,
    pub paused: usize,
    pub completed: usize,
    pub failed: usize,
    pub cancelled: usize,
}

pub(crate) fn initialize(connection: &Connection) -> Result<()> {
    connection.execute_batch(
        "CREATE TABLE IF NOT EXISTS runtime_errors (
        agent_id TEXT PRIMARY KEY REFERENCES agents(id),
        generation INTEGER NOT NULL,
        message TEXT NOT NULL
    );",
    )?;
    Ok(())
}

impl RuntimeStore {
    pub async fn set_metadata(&self, agent_id: String, metadata: Value) -> Result<()> {
        let metadata = serde_json::to_string(&metadata)?;
        ensure!(
            metadata.len() <= self.limits().max_input_bytes,
            "metadata exceeds the input byte limit"
        );
        self.write(move |connection| {
            let transaction=connection.transaction()?;
            let idle:bool=transaction.query_row("SELECT state!='running' AND NOT EXISTS(SELECT 1 FROM jobs WHERE agent_id=agents.id AND state IN ('queued','dispatched','uncertain')) FROM agents WHERE id=?",[&agent_id],|row|row.get(0))?;
            ensure!(idle,"wait for the active turn and tool jobs before changing agent configuration");
            transaction.execute("UPDATE agents SET metadata=? WHERE id=?",params![metadata,agent_id])?;
            transaction.commit()?;
            Ok(())
        }).await
    }

    pub async fn transcript_after(
        &self,
        agent_id: String,
        after: i64,
        limit: usize,
    ) -> Result<Vec<TranscriptEntry>> {
        ensure!(
            (1..=256).contains(&limit),
            "transcript page size must be between 1 and 256"
        );
        self.read(move |connection| {
            let mut statement=connection.prepare("SELECT position,payload FROM history WHERE agent_id=? AND position>? ORDER BY position LIMIT ?")?;
            let rows=statement.query_map(params![agent_id,after,limit],|row|Ok((row.get::<_,i64>(0)?,row.get::<_,String>(1)?)))?;
            let mut entries=Vec::new();let mut bytes=0usize;
            for row in rows {
                let (position,payload)=row?;
                if !entries.is_empty() && bytes.saturating_add(payload.len())>MAX_TRANSCRIPT_PAGE_BYTES {break;}
                bytes=bytes.saturating_add(payload.len());
                entries.push(TranscriptEntry {position,payload:serde_json::from_str(&payload)?});
            }
            Ok(entries)
        }).await
    }

    pub async fn transcript_page(
        &self,
        agent_id: String,
        before: Option<i64>,
        limit: usize,
    ) -> Result<TranscriptPage> {
        ensure!(
            (1..=256).contains(&limit),
            "transcript page size must be between 1 and 256"
        );
        ensure!(
            before.is_none_or(|cursor| cursor >= 0),
            "negative transcript cursor"
        );
        self.read(move |connection| {
            let inherited = connection.query_row(
                "SELECT parent_id,parent_cutoff,fresh FROM agents WHERE id=?", [&agent_id],
                |row| Ok((row.get::<_,Option<String>>(0)?,row.get::<_,Option<i64>>(1)?,row.get::<_,bool>(2)?)),
            )?;
            let inherited = match inherited {
                (Some(agent_id),Some(cutoff),false) => Some(HistoryReference {agent_id,cutoff}),
                _ => None,
            };
            let mut statement = connection.prepare("SELECT position,payload FROM history WHERE agent_id=? AND position<? ORDER BY position DESC LIMIT ?")?;
            let query = statement.query_map(params![agent_id,before.unwrap_or(i64::MAX),limit+1], |row| Ok((row.get::<_,i64>(0)?,row.get::<_,String>(1)?)))?;
            let mut rows=Vec::new();let mut bytes=0usize;let mut more=false;
            for row in query {
                let row=row?;
                if rows.len()==limit || (!rows.is_empty() && bytes.saturating_add(row.1.len())>MAX_TRANSCRIPT_PAGE_BYTES) {more=true;break;}
                bytes=bytes.saturating_add(row.1.len());rows.push(row);
            }
            let next_cursor = if more { rows.last().map(|row| row.0) } else { None };
            rows.reverse();
            let entries = rows.into_iter().map(|(position,payload)| Ok(TranscriptEntry {position,payload:serde_json::from_str(&payload)?})).collect::<Result<Vec<_>>>()?;
            Ok(TranscriptPage {entries,next_cursor,inherited})
        }).await
    }

    pub async fn fleet_page(
        &self,
        root_id: Option<String>,
        after: Option<String>,
        limit: usize,
    ) -> Result<AgentPage> {
        if let Some(root_id) = root_id {
            return self.list_agents(root_id, after, limit).await;
        }
        ensure!(
            (1..=256).contains(&limit),
            "fleet page size must be between 1 and 256"
        );
        self.read(move |connection| {
            let mut statement = connection.prepare("SELECT id,root_id,parent_id,state,generation FROM agents WHERE id>? ORDER BY id LIMIT ?")?;
            let mut agents = statement.query_map(params![after.unwrap_or_default(),limit+1], |row| Ok(AgentSnapshot {
                id:row.get(0)?,root_id:row.get(1)?,parent_id:row.get(2)?,state:row.get(3)?,generation:row.get(4)?,
            }))?.collect::<rusqlite::Result<Vec<_>>>()?;
            let more = agents.len()>limit;
            agents.truncate(limit);
            let next_cursor = if more {agents.last().map(|agent| agent.id.clone())} else {None};
            Ok(AgentPage {agents,next_cursor})
        }).await
    }

    pub async fn fleet_summary(&self, root_id: Option<String>) -> Result<FleetSummary> {
        self.read(move |connection| {
            let mut summary = FleetSummary::default();
            let mut statement = connection.prepare(
                "SELECT state,count(*) FROM agents WHERE (? IS NULL OR root_id=?) GROUP BY state",
            )?;
            let states = statement.query_map(params![root_id, root_id], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, usize>(1)?))
            })?;
            for row in states {
                let (state, count) = row?;
                summary.total += count;
                match state.as_str() {
                    "queued" => summary.queued = count,
                    "running" => summary.running = count,
                    "waiting" => summary.waiting = count,
                    "paused" => summary.paused = count,
                    "completed" => summary.completed = count,
                    "failed" => summary.failed = count,
                    "cancelled" => summary.cancelled = count,
                    _ => {}
                }
            }
            Ok(summary)
        })
        .await
    }

    pub async fn pending_job_count(&self, agent_id: String) -> Result<usize> {
        self.read(move |connection| Ok(connection.query_row(
            "SELECT (SELECT count(*) FROM jobs WHERE agent_id=?1 AND state IN ('queued','dispatched','uncertain')) + (SELECT count(*) FROM runtime_joins j JOIN agents a ON a.id=j.agent_id WHERE j.agent_id=?1 AND j.generation=a.generation AND j.state='pending')",[agent_id],|row| row.get(0),
        )?)).await
    }

    pub async fn agent_error(&self, agent_id: String) -> Result<Option<String>> {
        self.read(move |connection| {
            Ok(connection
                .query_row(
                    "SELECT message FROM runtime_errors WHERE agent_id=?",
                    [agent_id],
                    |row| row.get(0),
                )
                .optional()?)
        })
        .await
    }

    pub(crate) async fn record_error(&self, lease: Lease, message: String) -> Result<()> {
        let message: String = message.chars().take(8192).collect();
        self.write(move |connection| {
            crate::store::check_lease(connection,&lease)?;
            connection.execute("INSERT INTO runtime_errors(agent_id,generation,message) VALUES (?,?,?) ON CONFLICT(agent_id) DO UPDATE SET generation=excluded.generation,message=excluded.message",params![lease.agent_id,lease.generation,message])?;
            Ok(())
        }).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RuntimeLimits;
    use serde_json::json;

    #[tokio::test]
    async fn projections_are_bounded_and_errors_survive_reopen() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("runtime.sqlite");
        let store = RuntimeStore::open(&path, RuntimeLimits::default())?;
        store.create_root("root".into(), json!({})).await?;
        store
            .enqueue_text_input("root".into(), "input".into(), "work".into())
            .await?;
        let lease = store
            .admit()
            .await?
            .ok_or_else(|| anyhow::anyhow!("lease"))?;
        store
            .record_error(lease.clone(), "provider unavailable".into())
            .await?;
        store.interrupt(lease).await?;
        store
            .write(|connection| {
                for position in 0..600 {
                    connection.execute(
                        "INSERT INTO history VALUES ('root',?,?,?)",
                        params![
                            position,
                            format!("message-{position}"),
                            "{\"kind\":\"assistant\"}"
                        ],
                    )?;
                }
                Ok(())
            })
            .await?;
        let newest = store.transcript_page("root".into(), None, 100).await?;
        assert_eq!(newest.entries.len(), 100);
        assert_eq!(
            newest.entries.first().map(|entry| entry.position),
            Some(500)
        );
        assert_eq!(newest.next_cursor, Some(500));
        let older = store
            .transcript_page("root".into(), newest.next_cursor, 100)
            .await?;
        assert_eq!(older.entries.last().map(|entry| entry.position), Some(499));
        assert!(
            store
                .transcript_page("root".into(), None, 257)
                .await
                .is_err()
        );
        assert_eq!(store.fleet_summary(None).await?.paused, 1);
        store.shutdown().await?;
        let reopened = RuntimeStore::open(path, RuntimeLimits::default())?;
        assert_eq!(
            reopened.agent_error("root".into()).await?.as_deref(),
            Some("provider unavailable")
        );
        reopened.shutdown().await
    }
}
