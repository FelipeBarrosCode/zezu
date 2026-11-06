use anyhow::{Result, ensure};
use rusqlite::params;
use serde_json::{Value, json};

use crate::{Job, JobState, RuntimeStore, model::serialized_size};

const TOOL_RESULT_PREVIEW_BYTES: usize = 64 * 1024;
const TOOL_RESULT_METADATA_BYTES: usize = 512;

pub(crate) fn bound_tool_result(input: &mut Value, maximum_bytes: usize) -> Result<()> {
    let original_bytes = serialized_size(input)?;
    if original_bytes <= maximum_bytes {
        return Ok(());
    }

    let original_content = input["content"]["content"].take();
    let raw_output = input["content"]["output"].take();
    input["content"]["content"] = json!([]);
    let envelope_bytes = serialized_size(input)?;
    let preview_budget =
        maximum_bytes.min(envelope_bytes.saturating_add(TOOL_RESULT_PREVIEW_BYTES));
    let mut preview = format!(
        "Tool output truncated. The serialized result was {original_bytes} bytes and exceeded the runtime limit of {maximum_bytes} bytes. Inspect the existing terminal output or request a smaller chunk of the source if needed."
    );
    input["content"]["output"] = tool_result_metadata(raw_output)?;
    input["content"]["content"] = json!([{"type":"text","text":preview}]);
    if serialized_size(input)? > preview_budget {
        preview = "Tool output truncated.".into();
        input["content"]["content"] = json!([{"type":"text","text":preview}]);
    }
    if serialized_size(input)? > preview_budget {
        input["content"]["output"] = Value::Null;
    }
    if serialized_size(input)? > preview_budget {
        input["content"]["content"] = json!([]);
        return Ok(());
    }

    // A source byte expands to at most six bytes after JSON escaping.
    let mut remaining = (preview_budget - serialized_size(input)?) / 6;
    for block in original_content.as_array().into_iter().flatten() {
        let Some(text) = block.get("text").and_then(Value::as_str) else {
            continue;
        };
        let length = text.floor_char_boundary(remaining.saturating_sub(2));
        if length > 0 {
            preview.push_str("\n\n");
            preview.push_str(&text[..length]);
            remaining -= length + 2;
        }
        if length < text.len() {
            break;
        }
    }
    input["content"]["content"] = json!([{"type":"text","text":preview}]);
    Ok(())
}

fn tool_result_metadata(raw_output: Value) -> Result<Value> {
    if serialized_size(&raw_output)? <= TOOL_RESULT_METADATA_BYTES {
        return Ok(raw_output);
    }
    let mut metadata = json!({"truncated":true});
    for key in [
        "exit_code",
        "exit_status",
        "status",
        "signal",
        "timed_out",
        "cancelled",
    ] {
        if let Some(value) = raw_output.get(key)
            && serialized_size(value)? <= TOOL_RESULT_METADATA_BYTES
        {
            metadata[key] = value.clone();
        }
    }
    Ok(metadata)
}

pub(crate) fn initialize_connection(connection: &rusqlite::Connection) -> Result<()> {
    connection.execute_batch("CREATE TABLE IF NOT EXISTS runtime_job_events(sequence INTEGER PRIMARY KEY AUTOINCREMENT,job_id TEXT NOT NULL REFERENCES jobs(id),kind TEXT NOT NULL,payload TEXT NOT NULL); CREATE INDEX IF NOT EXISTS runtime_job_events_job ON runtime_job_events(job_id,sequence); CREATE TABLE IF NOT EXISTS runtime_resource_waits(job_id TEXT PRIMARY KEY REFERENCES jobs(id),kind TEXT NOT NULL);")?;
    connection.execute_batch("CREATE TABLE IF NOT EXISTS runtime_dispatch_clock(singleton INTEGER PRIMARY KEY CHECK(singleton=1),revision INTEGER NOT NULL); INSERT OR IGNORE INTO runtime_dispatch_clock(singleton,revision) VALUES (1,0);
      CREATE TRIGGER IF NOT EXISTS runtime_dispatch_job_insert AFTER INSERT ON jobs BEGIN UPDATE runtime_dispatch_clock SET revision=revision+1 WHERE singleton=1; END;
      CREATE TRIGGER IF NOT EXISTS runtime_dispatch_job_delete AFTER DELETE ON jobs BEGIN UPDATE runtime_dispatch_clock SET revision=revision+1 WHERE singleton=1; END;
      CREATE TRIGGER IF NOT EXISTS runtime_dispatch_job_update AFTER UPDATE OF state,generation ON jobs WHEN OLD.state IS NOT NEW.state OR OLD.generation IS NOT NEW.generation BEGIN UPDATE runtime_dispatch_clock SET revision=revision+1 WHERE singleton=1; END;
      CREATE TRIGGER IF NOT EXISTS runtime_dispatch_agent_update AFTER UPDATE OF state,pause_requested,generation ON agents WHEN OLD.state IS NOT NEW.state OR OLD.pause_requested IS NOT NEW.pause_requested OR OLD.generation IS NOT NEW.generation BEGIN UPDATE runtime_dispatch_clock SET revision=revision+1 WHERE singleton=1; END;
      CREATE TRIGGER IF NOT EXISTS runtime_dispatch_wait_insert AFTER INSERT ON runtime_resource_waits BEGIN UPDATE runtime_dispatch_clock SET revision=revision+1 WHERE singleton=1; END;
      CREATE TRIGGER IF NOT EXISTS runtime_dispatch_wait_delete AFTER DELETE ON runtime_resource_waits BEGIN UPDATE runtime_dispatch_clock SET revision=revision+1 WHERE singleton=1; END;
      CREATE TRIGGER IF NOT EXISTS runtime_dispatch_wait_update AFTER UPDATE ON runtime_resource_waits BEGIN UPDATE runtime_dispatch_clock SET revision=revision+1 WHERE singleton=1; END;")?;
    Ok(())
}

#[derive(Default)]
pub struct JobDispatchCursor {
    empty_at: Option<(i64, Vec<String>)>,
    scans: u64,
}

#[derive(Clone, Debug)]
pub struct UncertainJob {
    pub id: String,
    pub agent_id: String,
    /// The current agent generation, which can be newer than the interrupted job.
    pub generation: i64,
    pub agent_state: String,
    pub tool_name: Option<String>,
}

impl JobDispatchCursor {
    pub fn scan_count(&self) -> u64 {
        self.scans
    }

    pub async fn poll(
        &mut self,
        store: &RuntimeStore,
        limit: usize,
        mut excluded_agents: Vec<String>,
    ) -> Result<Vec<Job>> {
        ensure!(
            (1..=256).contains(&limit),
            "job page size must be between 1 and 256"
        );
        ensure!(excluded_agents.len() <= 256, "too many active tool agents");
        excluded_agents.sort_unstable();
        let revision = store.dispatch_revision().await?;
        if self
            .empty_at
            .as_ref()
            .is_some_and(|(previous, agents)| *previous == revision && agents == &excluded_agents)
        {
            return Ok(Vec::new());
        }
        let jobs = store
            .queued_jobs_excluding_agents(limit, excluded_agents.clone())
            .await?;
        self.scans = self.scans.saturating_add(1);
        // A writer may change eligibility between the clock read and the queue scan.
        self.empty_at = if jobs.is_empty() && store.dispatch_revision().await? == revision {
            Some((revision, excluded_agents))
        } else {
            None
        };
        Ok(jobs)
    }
}

impl RuntimeStore {
    /// A small recovery projection for the selected agent, without decoding tool arguments.
    pub async fn uncertain_jobs(
        &self,
        agent_id: String,
        limit: usize,
    ) -> Result<Vec<UncertainJob>> {
        ensure!(
            (1..=20).contains(&limit),
            "uncertain job page size must be between 1 and 20"
        );
        self.read(move |connection| {
            let mut statement = connection.prepare("SELECT jobs.id,jobs.agent_id,agents.generation,agents.state,substr(json_extract(jobs.payload,'$.tool'),1,128) FROM jobs JOIN agents ON agents.id=jobs.agent_id WHERE jobs.agent_id=? AND jobs.state='uncertain' ORDER BY jobs.rowid LIMIT ?")?;
            Ok(statement.query_map(params![agent_id,limit], |row| Ok(UncertainJob { id:row.get(0)?,agent_id:row.get(1)?,generation:row.get(2)?,agent_state:row.get(3)?,tool_name:row.get(4)? }))?.collect::<rusqlite::Result<Vec<_>>>()?)
        }).await
    }

    /// Record the user's acknowledgement of an unknown effect, never a verified
    /// outcome or permission to replay the original tool. Agent state stays unchanged.
    pub async fn acknowledge_uncertain_job(
        &self,
        job_id: String,
        agent_id: String,
        expected_generation: i64,
        expected_state: String,
    ) -> Result<()> {
        let maximum_bytes = self.limits().max_input_bytes;
        self.write(move |connection| {
            let transaction = connection.transaction()?;
            let (job_generation,job_state,payload,generation,state,closed):(i64,String,String,i64,String,bool) = transaction.query_row(
                "SELECT jobs.generation,jobs.state,jobs.payload,agents.generation,agents.state,agents.closed FROM jobs JOIN agents ON agents.id=jobs.agent_id WHERE jobs.id=? AND jobs.agent_id=?",params![job_id,agent_id],|row|Ok((row.get(0)?,row.get(1)?,row.get(2)?,row.get(3)?,row.get(4)?,row.get(5)?)))?;
            ensure!(job_state=="uncertain", "job is not awaiting reconciliation");
            ensure!(generation==expected_generation && state==expected_state, "agent changed; refresh the recovery action");
            ensure!(!closed && state!="running", "stop the active turn before acknowledging an uncertain tool");
            let payload:Value=serde_json::from_str(&payload)?;
            let call_id=payload.get("tool_call_id").and_then(Value::as_str).filter(|id|!id.is_empty()).ok_or_else(||anyhow::anyhow!("uncertain tool has no canonical call identity"))?;
            let warning="Tool outcome is unknown. The operation may have changed files or external state. The user acknowledged this uncertainty and chose to continue without retrying. Inspect changes before making further edits. This is an error result, not confirmation that the operation succeeded or did not execute.";
            let result=json!({"group_id":agent_id,"content":{"type":"toolresult","id":call_id,"call_id":payload.get("provider_call_id"),"content":[{"type":"text","text":warning}],"is_error":true,"output":{"outcome":"unknown","acknowledged_by_user":true,"replayed":false}},"metadata":null});
            let _:infinity_agent_core::message::InputMessage=serde_json::from_value(result.clone())?;
            let result=serde_json::to_string(&result)?;
            ensure!(result.len()<=maximum_bytes, "recovery result exceeds input byte limit");
            let message_id=format!("job:{job_id}:result");
            use rusqlite::OptionalExtension as _;
            let existing:Option<String>=transaction.query_row("SELECT payload FROM inbox WHERE agent_id=? AND message_id=?",params![agent_id,message_id],|row|row.get(0)).optional()?;
            ensure!(existing.as_ref().is_none_or(|existing|existing==&result), "conflicting recovery result already exists");
            transaction.execute("UPDATE jobs SET state='completed',result=? WHERE id=? AND state='uncertain'",params![result,job_id])?;
            // Passive delivery also preserves correlation for a cancelled conversation's
            // next explicit user turn. Acknowledgement itself never resumes it.
            transaction.execute("INSERT OR IGNORE INTO inbox(agent_id,message_id,payload) VALUES (?,?,?)",params![agent_id,message_id,result])?;
            transaction.execute("INSERT INTO runtime_job_events(job_id,kind,payload) VALUES (?,'recovery_acknowledged',?)",params![job_id,serde_json::to_string(&json!({"outcome":"unknown","acknowledged_by_user":true,"replayed":false,"job_generation":job_generation,"agent_generation":generation,"agent_state":state}))?])?;
            transaction.commit()?;
            Ok(())
        }).await
    }

    pub async fn dispatch_revision(&self) -> Result<i64> {
        self.read(|connection| {
            Ok(connection.query_row(
                "SELECT revision FROM runtime_dispatch_clock WHERE singleton=1",
                [],
                |row| row.get(0),
            )?)
        })
        .await
    }
    pub async fn latest_job_event(&self, agent_id: String, kind: String) -> Result<Option<Value>> {
        self.read(move |connection| {
            let exists:bool=connection.query_row("SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE name='runtime_job_events')",[],|row|row.get(0))?;
            if !exists { return Ok(None); }
            use rusqlite::OptionalExtension as _;
            let value:Option<String>=connection.query_row("SELECT runtime_job_events.payload FROM runtime_job_events JOIN jobs ON jobs.id=runtime_job_events.job_id WHERE jobs.agent_id=? AND runtime_job_events.kind=? ORDER BY runtime_job_events.sequence DESC LIMIT 1",params![agent_id,kind],|row|row.get(0)).optional()?;
            value.map(|value|serde_json::from_str(&value)).transpose().map_err(Into::into)
        }).await
    }

    pub async fn queued_jobs(&self, limit: usize) -> Result<Vec<Job>> {
        self.queued_jobs_excluding_agents(limit, Vec::new()).await
    }

    pub async fn queued_jobs_excluding_agents(
        &self,
        limit: usize,
        excluded_agents: Vec<String>,
    ) -> Result<Vec<Job>> {
        ensure!(excluded_agents.len() <= 256, "too many active tool agents");
        let excluded = serde_json::to_string(&excluded_agents)?;
        ensure!(
            (1..=256).contains(&limit),
            "job page size must be between 1 and 256"
        );
        self.read(move |connection| {
            let mut statement = connection.prepare("SELECT jobs.id,jobs.agent_id,jobs.generation,jobs.state,jobs.payload,jobs.result FROM jobs JOIN agents ON agents.id=jobs.agent_id WHERE jobs.state='queued' AND NOT EXISTS(SELECT 1 FROM runtime_configuration_blocked blocked WHERE blocked.agent_id=agents.id) AND jobs.generation=agents.generation AND agents.pause_requested=0 AND agents.state IN ('queued','running','waiting') AND NOT EXISTS(SELECT 1 FROM runtime_resource_waits resource_wait WHERE resource_wait.job_id=jobs.id) AND jobs.agent_id NOT IN (SELECT value FROM json_each(?)) AND NOT EXISTS(SELECT 1 FROM jobs earlier WHERE earlier.agent_id=jobs.agent_id AND earlier.state='queued' AND earlier.generation=jobs.generation AND earlier.rowid<jobs.rowid) ORDER BY jobs.rowid LIMIT ?")?;
            let rows = statement.query_map(params![excluded,limit], |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?, row.get::<_, i64>(2)?, row.get::<_, String>(4)?, row.get::<_, Option<String>>(5)?)))?;
            rows.map(|row| {
                let (id,agent_id,generation,payload,result)=row?;
                Ok(Job { id,agent_id,generation,state:JobState::Queued,payload:serde_json::from_str(&payload)?,result:result.map(|value|serde_json::from_str(&value)).transpose()? })
            }).collect()
        }).await
    }

    pub async fn record_job_event(&self, job_id: String, kind: String, value: Value) -> Result<()> {
        let payload = serde_json::to_string(&value)?;
        ensure!(
            payload.len() <= self.limits().max_input_bytes,
            "job event exceeds payload limit"
        );
        self.write(move |connection| {
            connection.execute_batch("CREATE TABLE IF NOT EXISTS runtime_job_events(sequence INTEGER PRIMARY KEY AUTOINCREMENT,job_id TEXT NOT NULL REFERENCES jobs(id),kind TEXT NOT NULL,payload TEXT NOT NULL); CREATE INDEX IF NOT EXISTS runtime_job_events_job ON runtime_job_events(job_id,sequence);")?;
            let valid:bool = connection.query_row("SELECT EXISTS(SELECT 1 FROM jobs JOIN agents ON agents.id=jobs.agent_id WHERE jobs.id=? AND jobs.state='dispatched' AND jobs.generation=agents.generation)", [&job_id], |row| row.get(0))?;
            ensure!(valid,"stale job event");
            connection.execute("INSERT INTO runtime_job_events(job_id,kind,payload) VALUES (?,?,?)",params![job_id,kind,payload])?;
            Ok(())
        }).await
    }

    pub async fn mark_job_uncertain(&self, job_id: String) -> Result<()> {
        self.write(move |connection| {
            connection.execute(
                "UPDATE jobs SET state='uncertain' WHERE id=? AND state='dispatched'",
                [job_id],
            )?;
            Ok(())
        })
        .await
    }

    pub async fn complete_tool_job(
        &self,
        job: Job,
        content: Vec<Value>,
        is_error: bool,
        raw_output: Value,
    ) -> Result<bool> {
        let call_id = job
            .payload
            .get("tool_call_id")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("job has no tool call identity"))?;
        let mut input = json!({
            "group_id":job.agent_id,
            "content":{"type":"toolresult","id":call_id,"call_id":job.payload.get("provider_call_id"),"is_error":is_error},
            "metadata":null,
        });
        input["content"]["content"] = Value::Array(content);
        input["content"]["output"] = raw_output;
        self.complete_job(job.id, input).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RuntimeLimits;

    #[tokio::test]
    async fn acknowledged_unknown_effect_preserves_identity_without_replaying_after_restart_or_stop()
    -> Result<()> {
        for cancelled in [false, true] {
            let directory = tempfile::tempdir()?;
            let database = directory.path().join("recovery.sqlite");
            let mut store = RuntimeStore::open(&database, RuntimeLimits::default())?;
            store.create_root("agent".into(), json!({})).await?;
            store
                .enqueue_text_input("agent".into(), "input".into(), "edit".into())
                .await?;
            let lease = store
                .admit()
                .await?
                .ok_or_else(|| anyhow::anyhow!("lease"))?;
            store.record_job(lease.clone(),"job".into(),json!({"tool":"write_file","tool_call_id":"logical-call","provider_call_id":"provider-call","arguments":{"path":"file.txt"}})).await?;
            store
                .finish(lease, json!({}), crate::SliceOutcome::Waiting)
                .await?;
            store.dispatch_job("job".into()).await?;
            store
                .record_job_event(
                    "job".into(),
                    "approval".into(),
                    json!({"decision":"allow_once"}),
                )
                .await?;
            if cancelled {
                store.cancel("agent".into()).await?;
            } else {
                store.shutdown().await?;
                store = RuntimeStore::open(&database, RuntimeLimits::default())?;
            }
            let snapshot = store.snapshot("agent".into()).await?;
            assert!(snapshot.generation > store.job("job".into()).await?.generation);
            let jobs = store.uncertain_jobs("agent".into(), 1).await?;
            assert_eq!(jobs.len(), 1);
            assert_eq!(jobs[0].generation, snapshot.generation);
            assert_eq!(jobs[0].tool_name.as_deref(), Some("write_file"));
            assert!(store.uncertain_jobs("agent".into(), 21).await.is_err());
            assert!(
                store
                    .acknowledge_uncertain_job(
                        "job".into(),
                        "agent".into(),
                        snapshot.generation - 1,
                        snapshot.state.clone()
                    )
                    .await
                    .is_err()
            );
            assert!(
                store
                    .acknowledge_uncertain_job(
                        "job".into(),
                        "agent".into(),
                        snapshot.generation,
                        "running".into()
                    )
                    .await
                    .is_err()
            );
            assert_eq!(store.job("job".into()).await?.state, JobState::Uncertain);
            store
                .acknowledge_uncertain_job(
                    "job".into(),
                    "agent".into(),
                    snapshot.generation,
                    snapshot.state.clone(),
                )
                .await?;
            assert_eq!(
                store.snapshot("agent".into()).await?,
                snapshot,
                "acknowledgement must not resume or resurrect an agent"
            );
            let job = store.job("job".into()).await?;
            assert_eq!(job.state, JobState::Completed);
            let result = job.result.ok_or_else(|| anyhow::anyhow!("result"))?;
            assert_eq!(result["content"]["id"], "logical-call");
            assert_eq!(result["content"]["call_id"], "provider-call");
            assert_eq!(result["content"]["is_error"], true);
            assert_eq!(result["content"]["output"]["outcome"], "unknown");
            assert_eq!(result["content"]["output"]["replayed"], false);
            assert!(store.uncertain_jobs("agent".into(), 20).await?.is_empty());
            assert!(store.queued_jobs(20).await?.is_empty());
            assert!(store.dispatch_job("job".into()).await.is_err());
            assert!(store.admit().await?.is_none());
            assert_eq!(
                store
                    .latest_job_event("agent".into(), "approval".into())
                    .await?,
                Some(json!({"decision":"allow_once"}))
            );
            let audit = store
                .latest_job_event("agent".into(), "recovery_acknowledged".into())
                .await?
                .ok_or_else(|| anyhow::anyhow!("audit"))?;
            assert_eq!(audit["outcome"], "unknown");
            assert_eq!(audit["acknowledged_by_user"], true);
            assert!(
                store
                    .acknowledge_uncertain_job(
                        "job".into(),
                        "agent".into(),
                        snapshot.generation,
                        snapshot.state
                    )
                    .await
                    .is_err()
            );
            if cancelled {
                store.cancel("agent".into()).await?;
                store
                    .enqueue_text_input(
                        "agent".into(),
                        "continue".into(),
                        "I inspected the changes; continue.".into(),
                    )
                    .await?;
            } else {
                store.resume("agent".into()).await?;
            }
            let lease = store
                .admit()
                .await?
                .ok_or_else(|| anyhow::anyhow!("explicit continuation lease"))?;
            let recovered = lease
                .inputs
                .iter()
                .filter(|(id, _)| id == "job:job:result")
                .collect::<Vec<_>>();
            assert_eq!(
                recovered.len(),
                1,
                "deliver exactly one correlated error, including after Stop"
            );
            assert_eq!(&recovered[0].1, &result);
            store
                .finish(lease, json!({}), crate::SliceOutcome::Waiting)
                .await?;
            assert!(store.queued_jobs(20).await?.is_empty());
            assert!(store.dispatch_job("job".into()).await.is_err());
            store.shutdown().await?;
        }
        Ok(())
    }

    #[tokio::test]
    async fn empty_dispatch_cursor_wakes_on_agent_job_and_resource_wait_changes() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let store = RuntimeStore::open(
            directory.path().join("clock.sqlite"),
            RuntimeLimits::default(),
        )?;
        let mut cursor = JobDispatchCursor::default();
        assert!(cursor.poll(&store, 8, vec![]).await?.is_empty());
        assert!(cursor.poll(&store, 8, vec![]).await?.is_empty());
        assert_eq!(cursor.scan_count(), 1);
        store.create_root("agent".into(), json!({})).await?;
        store
            .enqueue_text_input("agent".into(), "input".into(), "go".into())
            .await?;
        assert!(cursor.poll(&store, 8, vec![]).await?.is_empty());
        assert_eq!(cursor.scan_count(), 2);
        let lease = store
            .admit()
            .await?
            .ok_or_else(|| anyhow::anyhow!("lease"))?;
        store
            .record_job(lease.clone(), "job".into(), json!({}))
            .await?;
        store
            .finish(lease, json!({}), crate::SliceOutcome::Waiting)
            .await?;
        assert_eq!(cursor.poll(&store, 8, vec![]).await?.len(), 1);
        store
            .write(|connection| {
                connection.execute(
                    "INSERT INTO runtime_resource_waits(job_id,kind) VALUES ('job','workspace')",
                    [],
                )?;
                Ok(())
            })
            .await?;
        assert!(cursor.poll(&store, 8, vec![]).await?.is_empty());
        let scans = cursor.scan_count();
        assert!(cursor.poll(&store, 8, vec![]).await?.is_empty());
        assert_eq!(cursor.scan_count(), scans);
        store
            .write(|connection| {
                connection.execute("DELETE FROM runtime_resource_waits WHERE job_id='job'", [])?;
                Ok(())
            })
            .await?;
        assert_eq!(cursor.poll(&store, 8, vec![]).await?.len(), 1);
        assert!(
            cursor
                .poll(&store, 8, vec!["agent".into()])
                .await?
                .is_empty()
        );
        assert_eq!(
            cursor.poll(&store, 8, vec![]).await?.len(),
            1,
            "active-agent exclusions must invalidate an empty scan"
        );
        store.shutdown().await
    }

    #[tokio::test]
    #[ignore = "explicit 75,000 parked-job dispatch scan scenario"]
    async fn parked_job_fleet_scans_once_until_a_waiter_becomes_eligible() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let store = RuntimeStore::open(
            directory.path().join("clock.sqlite"),
            RuntimeLimits::default(),
        )?;
        store.create_root("agent".into(), json!({})).await?;
        store.write(|connection| {
            let transaction=connection.transaction()?;
            {
                let mut jobs=transaction.prepare("INSERT INTO jobs(id,agent_id,generation,state,payload) VALUES (?,'agent',0,'queued','{}')")?;
                let mut waits=transaction.prepare("INSERT INTO runtime_resource_waits(job_id,kind) VALUES (?,'workspace')")?;
                for index in 0..75_000 {let id=format!("job-{index:05}");jobs.execute([&id])?;waits.execute([&id])?;}
            }
            transaction.commit()?;
            Ok(())
        }).await?;
        let mut cursor = JobDispatchCursor::default();
        for _ in 0..100 {
            assert!(cursor.poll(&store, 8, vec![]).await?.is_empty());
        }
        assert_eq!(
            cursor.scan_count(),
            1,
            "idle dispatch must not repeatedly scan 75,000 blocked rows"
        );
        store
            .write(|connection| {
                connection.execute(
                    "DELETE FROM runtime_resource_waits WHERE job_id='job-00000'",
                    [],
                )?;
                Ok(())
            })
            .await?;
        assert_eq!(
            cursor
                .poll(&store, 8, vec![])
                .await?
                .first()
                .map(|job| job.id.as_str()),
            Some("job-00000")
        );
        assert_eq!(cursor.scan_count(), 2);
        store.shutdown().await
    }

    #[tokio::test]
    async fn job_pages_admit_one_job_per_agent_and_skip_active_agents() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let store = RuntimeStore::open(
            directory.path().join("jobs.sqlite"),
            RuntimeLimits::default(),
        )?;
        for agent in ["first", "second"] {
            store.create_root(agent.into(), json!({})).await?;
            store
                .enqueue_text_input(agent.into(), "input".into(), "go".into())
                .await?;
            let lease = store
                .admit()
                .await?
                .ok_or_else(|| anyhow::anyhow!("lease"))?;
            store
                .record_job(lease.clone(), format!("{agent}-1"), json!({}))
                .await?;
            store
                .record_job(lease.clone(), format!("{agent}-2"), json!({}))
                .await?;
            store
                .finish(lease, json!({}), crate::SliceOutcome::Waiting)
                .await?;
        }
        let jobs = store.queued_jobs(2).await?;
        assert_eq!(
            jobs.iter().map(|job| job.id.as_str()).collect::<Vec<_>>(),
            vec!["first-1", "second-1"]
        );
        let available = store
            .queued_jobs_excluding_agents(2, vec!["first".into()])
            .await?;
        assert_eq!(available.len(), 1);
        assert_eq!(
            available.first().map(|job| job.id.as_str()),
            Some("second-1")
        );
        store.shutdown().await
    }

    #[tokio::test]
    async fn queued_jobs_follow_pause_and_generation_and_preserve_result_identity() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let store = RuntimeStore::open(
            directory.path().join("jobs.sqlite"),
            RuntimeLimits::default(),
        )?;
        store.create_root("agent".into(), json!({})).await?;
        store
            .enqueue_text_input("agent".into(), "input".into(), "run".into())
            .await?;
        let lease = store
            .admit()
            .await?
            .ok_or_else(|| anyhow::anyhow!("lease"))?;
        store
            .record_job(
                lease.clone(),
                "job".into(),
                json!({"tool":"read_file","arguments":{},"tool_call_id":"call"}),
            )
            .await?;
        assert_eq!(store.queued_jobs(1).await?.len(), 1);
        store.pause("agent".into()).await?;
        assert!(store.queued_jobs(1).await?.is_empty());
        store
            .finish(lease, json!({}), crate::SliceOutcome::Waiting)
            .await?;
        store.resume("agent".into()).await?;
        let job = store.dispatch_job("job".into()).await?;
        store
            .record_job_event(
                job.id.clone(),
                "approval".into(),
                json!({"decision":"allow_once"}),
            )
            .await?;
        assert!(
            store
                .complete_tool_job(
                    job,
                    vec![json!({"type":"text","text":"done"})],
                    false,
                    Value::Null
                )
                .await?
        );
        let result = store
            .job("job".into())
            .await?
            .result
            .ok_or_else(|| anyhow::anyhow!("result"))?;
        assert_eq!(result["content"]["id"], "call");
        assert_eq!(
            result["content"]["content"],
            json!([{"type":"text","text":"done"}])
        );
        assert!(store.queued_jobs(257).await.is_err());
        store.shutdown().await
    }

    #[tokio::test]
    async fn oversized_tool_results_complete_with_bounded_output_and_preserve_status() -> Result<()>
    {
        for (is_error, exit_code) in [(false, 0), (true, 23)] {
            let directory = tempfile::tempdir()?;
            let limits = RuntimeLimits {
                max_input_bytes: 1024,
                ..RuntimeLimits::default()
            };
            let store = RuntimeStore::open(directory.path().join("jobs.sqlite"), limits.clone())?;
            store.create_root("agent".into(), json!({})).await?;
            store
                .enqueue_text_input("agent".into(), "input".into(), "run".into())
                .await?;
            let lease = store
                .admit()
                .await?
                .ok_or_else(|| anyhow::anyhow!("lease"))?;
            store
                .record_job(
                    lease.clone(),
                    "job".into(),
                    json!({"tool":"terminal","arguments":{},"tool_call_id":"call","provider_call_id":"provider-call"}),
                )
                .await?;
            store
                .finish(lease, json!({}), crate::SliceOutcome::Waiting)
                .await?;
            let job = store.dispatch_job("job".into()).await?;
            let output = format!(
                "Command exited with code {exit_code}.\n{}",
                "\"\0🦀".repeat(1024)
            );
            let content = vec![json!({"type":"text","text":output})];
            let raw_output = json!({"output":output,"exit_status":{"exit_code":exit_code}});
            assert!(
                store
                    .complete_tool_job(job.clone(), content.clone(), is_error, raw_output.clone())
                    .await?
            );
            assert!(
                !store
                    .complete_tool_job(job, content, is_error, raw_output)
                    .await?
            );

            let job = store.job("job".into()).await?;
            assert_eq!(job.state, JobState::Completed);
            assert!(store.uncertain_jobs("agent".into(), 1).await?.is_empty());
            let result = job.result.ok_or_else(|| anyhow::anyhow!("result"))?;
            assert!(serialized_size(&result)? <= limits.max_input_bytes);
            assert_eq!(result["content"]["id"], "call");
            assert_eq!(result["content"]["call_id"], "provider-call");
            assert_eq!(result["content"]["is_error"], is_error);
            assert_eq!(
                result["content"]["output"]["exit_status"]["exit_code"],
                exit_code
            );
            assert_eq!(result["content"]["output"]["truncated"], true);
            let preview = result["content"]["content"][0]["text"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("text preview"))?;
            assert!(preview.contains("Tool output truncated."));
            assert!(preview.contains(&format!("Command exited with code {exit_code}.")));
            assert!(preview.contains('🦀'));

            let lease = store
                .admit()
                .await?
                .ok_or_else(|| anyhow::anyhow!("tool result lease"))?;
            assert_eq!(lease.inputs, vec![("job:job:result".into(), result)]);
            store.shutdown().await?;
        }
        Ok(())
    }
}
