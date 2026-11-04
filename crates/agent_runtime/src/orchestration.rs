use anyhow::{Result, ensure};
use infinity_agent_core::message::InputMessage;
use rusqlite::{Connection, OptionalExtension as _, params};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{Lease, RuntimeStore};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct TaskIdentity {
    pub agent_id: String,
    pub generation: i64,
}

impl From<&Lease> for TaskIdentity {
    fn from(lease: &Lease) -> Self {
        Self {
            agent_id: lease.agent_id.clone(),
            generation: lease.generation,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "status", content = "result", rename_all = "snake_case")]
pub enum TaskResult {
    Success(Value),
    Failed(String),
    Cancelled,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FailurePolicy {
    Collect,
    FailFast,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TaskReport {
    pub sequence: i64,
    pub task: TaskIdentity,
    pub result: TaskResult,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JoinStatus {
    pub id: String,
    pub state: String,
    pub remaining: usize,
}

#[derive(Clone)]
pub struct Orchestrator {
    store: RuntimeStore,
}

impl Orchestrator {
    pub async fn open(store: RuntimeStore) -> Result<Self> {
        store
            .write(|connection| Self::initialize_connection(connection))
            .await?;
        Ok(Self { store })
    }

    pub(crate) fn initialize_connection(connection: &Connection) -> Result<()> {
        connection.execute_batch(
            "CREATE TABLE IF NOT EXISTS runtime_task_reports (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL REFERENCES agents(id), generation INTEGER NOT NULL,
                report_key TEXT NOT NULL, parent_id TEXT REFERENCES agents(id), result TEXT NOT NULL,
                UNIQUE(agent_id,generation), UNIQUE(agent_id,generation,report_key));
             CREATE INDEX IF NOT EXISTS runtime_reports_parent ON runtime_task_reports(parent_id,sequence);
             CREATE TABLE IF NOT EXISTS runtime_joins (
                id TEXT PRIMARY KEY, agent_id TEXT NOT NULL REFERENCES agents(id), generation INTEGER NOT NULL,
                join_key TEXT NOT NULL, policy TEXT NOT NULL CHECK(policy IN ('collect','fail_fast')),
                state TEXT NOT NULL DEFAULT 'pending' CHECK(state IN ('pending','ready','cancelled')),
                UNIQUE(agent_id,generation,join_key));
             CREATE INDEX IF NOT EXISTS runtime_joins_agent ON runtime_joins(agent_id,generation,state);
             CREATE TABLE IF NOT EXISTS runtime_join_dependencies (
                join_id TEXT NOT NULL REFERENCES runtime_joins(id), agent_id TEXT NOT NULL REFERENCES agents(id),
                generation INTEGER NOT NULL, PRIMARY KEY(join_id,agent_id));
             CREATE INDEX IF NOT EXISTS runtime_dependencies_agent ON runtime_join_dependencies(agent_id,generation,join_id);"
        )?;
        Ok(())
    }

    pub(crate) fn recover_connection(connection: &Connection) -> Result<()> {
        connection.execute("UPDATE runtime_joins SET state='cancelled' WHERE state='pending' AND agent_id IN (SELECT id FROM agents WHERE state IN ('cancelled','failed','completed'))", [])?;
        // Runtime recovery fences active generations; pending joins belong to the paused task, not its abandoned inference slice.
        connection.execute("UPDATE runtime_joins SET generation=(SELECT generation FROM agents WHERE agents.id=runtime_joins.agent_id) WHERE state='pending'", [])?;
        connection.execute("UPDATE runtime_join_dependencies SET generation=(SELECT generation FROM agents WHERE agents.id=runtime_join_dependencies.agent_id) WHERE join_id IN (SELECT id FROM runtime_joins WHERE state='pending') AND NOT EXISTS(SELECT 1 FROM runtime_task_reports r WHERE r.agent_id=runtime_join_dependencies.agent_id AND r.generation=runtime_join_dependencies.generation)", [])?;
        Ok(())
    }

    pub(crate) fn rebind_task(connection: &Connection, agent_id: &str) -> Result<()> {
        connection.execute("UPDATE runtime_joins SET state='cancelled' WHERE agent_id=? AND state='pending' AND EXISTS(SELECT 1 FROM agents a WHERE a.id=runtime_joins.agent_id AND a.state IN ('cancelled','failed','completed'))", [agent_id])?;
        connection.execute("UPDATE runtime_joins SET generation=(SELECT generation FROM agents WHERE id=?) WHERE agent_id=? AND state='pending'", params![agent_id,agent_id])?;
        connection.execute("UPDATE runtime_join_dependencies SET generation=(SELECT generation FROM agents WHERE id=?) WHERE agent_id=? AND join_id IN (SELECT id FROM runtime_joins WHERE state='pending') AND NOT EXISTS(SELECT 1 FROM runtime_task_reports r WHERE r.agent_id=runtime_join_dependencies.agent_id AND r.generation=runtime_join_dependencies.generation)", params![agent_id,agent_id])?;
        Ok(())
    }

    pub async fn spawn_child(
        &self,
        parent: TaskIdentity,
        key: String,
        cutoff: i64,
        fresh: bool,
    ) -> Result<String> {
        let maximum = self.store.limits().max_agents_per_run;
        self.store
            .write(move |connection| {
                validate_identity(connection, &parent)?;
                crate::store::spawn_child(
                    connection,
                    &parent.agent_id,
                    &key,
                    cutoff,
                    fresh,
                    false,
                    maximum,
                )
            })
            .await
    }

    pub async fn wait_for(
        &self,
        parent: TaskIdentity,
        key: String,
        mut dependencies: Vec<String>,
        policy: FailurePolicy,
    ) -> Result<JoinStatus> {
        ensure!(
            !key.is_empty() && key.len() <= 256,
            "join key must contain 1 to 256 bytes"
        );
        dependencies.sort();
        dependencies.dedup();
        ensure!(
            !dependencies.is_empty() && dependencies.len() <= 256,
            "a join requires 1 to 256 dependencies"
        );
        self.store.write(move |connection| {
            let transaction = connection.transaction()?;
            let root = validate_identity(&transaction, &parent)?;
            if let Some(id) = transaction.query_row("SELECT id FROM runtime_joins WHERE agent_id=? AND generation=? AND join_key=?", params![parent.agent_id,parent.generation,key], |row| row.get::<_,String>(0)).optional()? {
                let existing = dependency_ids(&transaction, &id)?;
                let existing_policy: String = transaction.query_row("SELECT policy FROM runtime_joins WHERE id=?", [&id], |row| row.get(0))?;
                ensure!(existing == dependencies && existing_policy == policy_name(policy), "join key reused with a different dependency set or policy");
                return join_status(&transaction, &id);
            }
            for dependency in &dependencies {
                ensure!(dependency != &parent.agent_id, "a task cannot depend on itself");
                let dependency_root: String = transaction.query_row("SELECT root_id FROM agents WHERE id=?", [dependency], |row| row.get(0))?;
                ensure!(dependency_root == root, "dependencies cannot cross runs");
                let cycle: bool = transaction.query_row(
                    "WITH RECURSIVE reachable(id) AS (SELECT ? UNION SELECT d.agent_id FROM reachable r JOIN runtime_joins j ON j.agent_id=r.id AND j.state='pending' JOIN runtime_join_dependencies d ON d.join_id=j.id) SELECT EXISTS(SELECT 1 FROM reachable WHERE id=?)",
                    params![dependency,parent.agent_id], |row| row.get(0))?;
                ensure!(!cycle, "dependency would create a cycle");
            }
            let id = uuid::Uuid::new_v4().to_string();
            transaction.execute("INSERT INTO runtime_joins(id,agent_id,generation,join_key,policy) VALUES (?,?,?,?,?)", params![id,parent.agent_id,parent.generation,key,policy_name(policy)])?;
            for dependency in dependencies {
                transaction.execute("INSERT INTO runtime_join_dependencies(join_id,agent_id,generation) SELECT ?,id,generation FROM agents WHERE id=?", params![id,dependency])?;
                record_terminal(&transaction, &dependency)?;
            }
            resolve_join(&transaction, &id)?;
            let status = join_status(&transaction, &id)?;
            transaction.commit()?;
            Ok(status)
        }).await
    }

    pub async fn report(
        &self,
        task: TaskIdentity,
        key: String,
        result: TaskResult,
    ) -> Result<bool> {
        self.report_with_job(task, key, result, None).await
    }

    pub async fn complete_idle_child(&self, agent_id: String, generation: i64) -> Result<bool> {
        self.store.write(move |connection| {
            let transaction=connection.transaction()?;
            let ready:bool=transaction.query_row("SELECT EXISTS(SELECT 1 FROM agents a WHERE a.id=? AND a.generation=? AND a.parent_id IS NOT NULL AND a.state='waiting' AND a.pause_requested=0 AND json_extract(a.checkpoint,'$.completed_step')=1 AND NOT EXISTS(SELECT 1 FROM inbox WHERE agent_id=a.id AND processed=0) AND NOT EXISTS(SELECT 1 FROM jobs WHERE agent_id=a.id AND state IN ('queued','dispatched','uncertain')) AND NOT EXISTS(SELECT 1 FROM runtime_joins WHERE agent_id=a.id AND generation=a.generation AND state='pending') AND NOT EXISTS(SELECT 1 FROM subscriptions WHERE agent_id=a.id) AND NOT EXISTS(SELECT 1 FROM choices WHERE agent_id=a.id))",params![agent_id,generation],|row|row.get(0))?;
            if !ready { return Ok(false); }
            let summary:Option<String>=transaction.query_row("SELECT substr(json_extract(payload,'$.content.text'),1,32768) FROM history WHERE agent_id=? AND json_extract(payload,'$.kind')='assistant' AND json_extract(payload,'$.content.text') IS NOT NULL ORDER BY position DESC LIMIT 1",[&agent_id],|row|row.get(0)).optional()?.flatten();
            let result=TaskResult::Success(summary.map(Value::String).unwrap_or(Value::Null));
            let inserted=transaction.execute("INSERT OR IGNORE INTO runtime_task_reports(agent_id,generation,report_key,parent_id,result) SELECT id,generation,?,parent_id,? FROM agents WHERE id=?",params![format!("idle:{generation}"),serde_json::to_string(&result)?,agent_id])?;
            ensure!(inserted==1,"idle child already has a conflicting report");
            transaction.execute("UPDATE agents SET state='completed',lease_id=NULL WHERE id=? AND generation=?",params![agent_id,generation])?;
            resolve_dependents(&transaction,&agent_id)?;
            transaction.commit()?;
            Ok(true)
        }).await
    }

    pub async fn report_dispatched_job(&self, job: crate::Job, result: TaskResult) -> Result<bool> {
        let task = TaskIdentity {
            agent_id: job.agent_id.clone(),
            generation: job.generation,
        };
        self.report_with_job(task, job.id.clone(), result, Some(job))
            .await
    }

    async fn report_with_job(
        &self,
        task: TaskIdentity,
        key: String,
        result: TaskResult,
        job: Option<crate::Job>,
    ) -> Result<bool> {
        ensure!(
            !key.is_empty() && key.len() <= 256,
            "report key must contain 1 to 256 bytes"
        );
        let encoded = serde_json::to_string(&result)?;
        ensure!(
            encoded.len() <= self.store.limits().max_input_bytes,
            "task report exceeds byte limit"
        );
        self.store.write(move |connection| {
            let transaction = connection.transaction()?;
            if let Some((existing_key, existing_result)) = transaction.query_row("SELECT report_key,result FROM runtime_task_reports WHERE agent_id=? AND generation=?", params![task.agent_id,task.generation], |row| Ok((row.get::<_,String>(0)?,row.get::<_,String>(1)?))).optional()? {
                ensure!(existing_key == key && existing_result == encoded, "conflicting duplicate task report");
                return Ok(false);
            }
            validate_identity(&transaction, &task)?;
            let active: bool = transaction.query_row("SELECT state='running' FROM agents WHERE id=?", [&task.agent_id], |row| row.get(0))?;
            ensure!(!active, "complete the active inference slice before reporting its task result");
            if let Some(job) = &job {
                let input = tool_result_input(job, serde_json::json!({"reported":true}), false)?;
                let changed = transaction.execute("UPDATE jobs SET state='completed',result=? WHERE id=? AND agent_id=? AND generation=? AND state='dispatched'", params![serde_json::to_string(&input)?,job.id,task.agent_id,task.generation])?;
                ensure!(changed == 1, "task report job is not currently dispatched");
            }
            let uncertain: bool = transaction.query_row("SELECT EXISTS(SELECT 1 FROM jobs WHERE agent_id=? AND state IN ('queued','dispatched','uncertain'))", [&task.agent_id], |row| row.get(0))?;
            ensure!(!uncertain, "finish or reconcile outstanding tool effects before reporting completion");
            transaction.execute("INSERT INTO runtime_task_reports(agent_id,generation,report_key,parent_id,result) SELECT id,generation,?,parent_id,? FROM agents WHERE id=?", params![key,encoded,task.agent_id])?;
            let state = match result { TaskResult::Success(_) => "completed", TaskResult::Failed(_) => "failed", TaskResult::Cancelled => "cancelled" };
            transaction.execute("UPDATE agents SET state=?,lease_id=NULL WHERE id=?", params![state,task.agent_id])?;
            transaction.execute("UPDATE inbox SET processed=1,lease_id=NULL WHERE agent_id=?", [&task.agent_id])?;
            transaction.execute("UPDATE runtime_joins SET state='cancelled' WHERE agent_id=? AND state='pending'", [&task.agent_id])?;
            resolve_dependents(&transaction, &task.agent_id)?;
            transaction.commit()?;
            Ok(true)
        }).await
    }

    pub async fn reports(
        &self,
        parent: String,
        after: i64,
        limit: usize,
    ) -> Result<Vec<TaskReport>> {
        ensure!(
            after >= 0 && (1..=256).contains(&limit),
            "invalid report page"
        );
        let byte_limit = self.store.limits().max_input_bytes;
        self.store.read(move |connection| {
            let mut statement = connection.prepare("SELECT sequence,agent_id,generation,result FROM (SELECT *,sum(length(CAST(result AS BLOB))) OVER (ORDER BY sequence) AS bytes,row_number() OVER (ORDER BY sequence) AS position FROM (SELECT sequence,agent_id,generation,result FROM runtime_task_reports WHERE parent_id=? AND sequence>? ORDER BY sequence LIMIT ?)) WHERE bytes<=? OR position=1 ORDER BY sequence")?;
            let rows = statement.query_map(params![parent,after,limit,byte_limit], |row| Ok((row.get::<_,i64>(0)?,row.get::<_,String>(1)?,row.get::<_,i64>(2)?,row.get::<_,String>(3)?)))?.collect::<rusqlite::Result<Vec<_>>>()?;
            rows.into_iter().map(|(sequence,agent_id,generation,result)| Ok(TaskReport { sequence, task: TaskIdentity {agent_id,generation}, result: serde_json::from_str(&result)? })).collect()
        }).await
    }

    pub async fn task_report(&self, task: TaskIdentity) -> Result<Option<TaskReport>> {
        self.store.read(move |connection| {
            let record: Option<(i64,String)> = connection.query_row("SELECT sequence,result FROM runtime_task_reports WHERE agent_id=? AND generation=?", params![task.agent_id,task.generation], |row| Ok((row.get(0)?,row.get(1)?))).optional()?;
            record.map(|(sequence,result)| Ok(TaskReport { sequence, task, result: serde_json::from_str(&result)? })).transpose()
        }).await
    }

    pub async fn reconcile_task(&self, agent_id: String) -> Result<()> {
        self.store
            .write(move |connection| {
                let transaction = connection.transaction()?;
                Self::reconcile_connection(&transaction, &agent_id)?;
                transaction.commit()?;
                Ok(())
            })
            .await
    }

    pub(crate) fn reconcile_connection(connection: &Connection, agent_id: &str) -> Result<()> {
        record_terminal(connection, agent_id)?;
        resolve_dependents(connection, agent_id)
    }
}

fn policy_name(policy: FailurePolicy) -> &'static str {
    match policy {
        FailurePolicy::Collect => "collect",
        FailurePolicy::FailFast => "fail_fast",
    }
}

fn validate_identity(connection: &Connection, task: &TaskIdentity) -> Result<String> {
    let (root, generation, state): (String, i64, String) = connection.query_row(
        "SELECT root_id,generation,state FROM agents WHERE id=?",
        [&task.agent_id],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
    )?;
    ensure!(
        generation == task.generation
            && !matches!(state.as_str(), "cancelled" | "failed" | "completed"),
        "stale or terminal task identity"
    );
    Ok(root)
}

fn dependency_ids(connection: &Connection, id: &str) -> Result<Vec<String>> {
    let mut statement = connection.prepare(
        "SELECT agent_id FROM runtime_join_dependencies WHERE join_id=? ORDER BY agent_id",
    )?;
    Ok(statement
        .query_map([id], |row| row.get(0))?
        .collect::<rusqlite::Result<_>>()?)
}

fn join_status(connection: &Connection, id: &str) -> Result<JoinStatus> {
    connection.query_row("SELECT id,state,(SELECT count(*) FROM runtime_join_dependencies d WHERE d.join_id=j.id AND NOT EXISTS(SELECT 1 FROM runtime_task_reports r WHERE r.agent_id=d.agent_id AND r.generation=d.generation)) FROM runtime_joins j WHERE id=?", [id], |row| Ok(JoinStatus {id:row.get(0)?,state:row.get(1)?,remaining:row.get(2)?})).map_err(Into::into)
}

fn record_terminal(connection: &Connection, id: &str) -> Result<()> {
    let terminal: Option<(i64,String)> = connection.query_row("SELECT generation,state FROM agents WHERE id=? AND state IN ('completed','failed','cancelled')", [id], |row| Ok((row.get(0)?,row.get(1)?))).optional()?;
    if let Some((generation, state)) = terminal {
        let result = match state.as_str() {
            "completed" => TaskResult::Success(Value::Null),
            "failed" => TaskResult::Failed("task failed before reporting a result".into()),
            _ => TaskResult::Cancelled,
        };
        connection.execute("INSERT OR IGNORE INTO runtime_task_reports(agent_id,generation,report_key,parent_id,result) SELECT id,?, ?,parent_id,? FROM agents WHERE id=?", params![generation,format!("terminal:{generation}"),serde_json::to_string(&result)?,id])?;
        connection.execute("INSERT OR IGNORE INTO runtime_task_reports(agent_id,generation,report_key,parent_id,result) SELECT a.id,d.generation,'terminal:'||d.generation,a.parent_id,? FROM agents a JOIN runtime_join_dependencies d ON d.agent_id=a.id JOIN runtime_joins j ON j.id=d.join_id AND j.state='pending' WHERE a.id=?", params![serde_json::to_string(&result)?,id])?;
    }
    Ok(())
}

fn resolve_dependents(connection: &Connection, task: &str) -> Result<()> {
    let ids = {
        let mut statement = connection.prepare("SELECT j.id FROM runtime_joins j JOIN runtime_join_dependencies d ON d.join_id=j.id WHERE d.agent_id=? AND j.state='pending'")?;
        statement
            .query_map([task], |row| row.get::<_, String>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?
    };
    for id in ids {
        resolve_join(connection, &id)?;
    }
    Ok(())
}

fn resolve_join(connection: &Connection, id: &str) -> Result<()> {
    let (agent_id, generation, policy, state): (String, i64, String, String) = connection
        .query_row(
            "SELECT agent_id,generation,policy,state FROM runtime_joins WHERE id=?",
            [id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )?;
    if state != "pending" {
        return Ok(());
    }
    let current: bool = connection.query_row("SELECT EXISTS(SELECT 1 FROM agents WHERE id=? AND generation=? AND state NOT IN ('cancelled','completed','failed'))", params![agent_id,generation], |row| row.get(0))?;
    if !current {
        connection.execute(
            "UPDATE runtime_joins SET state='cancelled' WHERE id=?",
            [id],
        )?;
        return Ok(());
    }
    let rows = {
        let mut statement = connection.prepare("SELECT d.agent_id,CASE WHEN length(CAST(r.result AS BLOB))<=65536 THEN r.result ELSE NULL END,json_extract(r.result,'$.status'),r.sequence FROM runtime_join_dependencies d LEFT JOIN runtime_task_reports r ON r.agent_id=d.agent_id AND r.generation=d.generation WHERE d.join_id=? ORDER BY d.agent_id")?;
        statement
            .query_map([id], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, Option<String>>(1)?,
                    row.get::<_, Option<String>>(2)?,
                    row.get::<_, Option<i64>>(3)?,
                ))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?
    };
    let mut results = Vec::new();
    let mut remaining = 0;
    let mut failed = false;
    let mut included_bytes = 0;
    for (task, result, status, sequence) in rows {
        if let Some(sequence) = sequence {
            failed |= status.as_deref() != Some("success");
            if let Some(result) = result.filter(|result| included_bytes + result.len() <= 131072) {
                included_bytes += result.len();
                let result: TaskResult = serde_json::from_str(&result)?;
                results.push(
                    serde_json::json!({"agent_id":task,"report_sequence":sequence,"result":result}),
                );
            } else {
                results.push(serde_json::json!({"agent_id":task,"report_sequence":sequence,"status":status,"result_available":true}));
            }
        } else {
            remaining += 1;
        }
    }
    if remaining > 0 && !(policy == "fail_fast" && failed) {
        return Ok(());
    }
    let input = InputMessage::user_text(
        agent_id.clone(),
        serde_json::to_string(
            &serde_json::json!({"runtime_join":id,"remaining":remaining,"results":results}),
        )?,
    );
    let payload = serde_json::to_string(&input)?;
    connection.execute(
        "INSERT OR IGNORE INTO inbox(agent_id,message_id,payload) VALUES (?,?,?)",
        params![agent_id, format!("join:{id}:ready"), payload],
    )?;
    connection.execute("UPDATE agents SET state=CASE WHEN pause_requested=1 THEN 'paused' WHEN state='running' THEN state ELSE 'queued' END WHERE id=? AND generation=?", params![agent_id,generation])?;
    connection.execute("UPDATE runtime_joins SET state='ready' WHERE id=?", [id])?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{RuntimeLimits, SliceOutcome};
    use anyhow::Context as _;
    use serde_json::json;

    async fn fixture() -> Result<(
        tempfile::TempDir,
        RuntimeStore,
        Orchestrator,
        String,
        String,
    )> {
        let directory = tempfile::tempdir()?;
        let store = RuntimeStore::open(
            directory.path().join("runtime.sqlite"),
            RuntimeLimits::default(),
        )?;
        let orchestrator = Orchestrator::open(store.clone()).await?;
        store.create_root("root".into(), json!({})).await?;
        store
            .enqueue_text_input("root".into(), "initial".into(), "start".into())
            .await?;
        let lease = store.admit().await?.context("root lease")?;
        let first = orchestrator
            .spawn_child(TaskIdentity::from(&lease), "first".into(), 0, true)
            .await?;
        let second = orchestrator
            .spawn_child(TaskIdentity::from(&lease), "second".into(), 0, true)
            .await?;
        store
            .finish(lease, json!({}), SliceOutcome::Waiting)
            .await?;
        Ok((directory, store, orchestrator, first, second))
    }

    #[tokio::test]
    async fn joins_wait_for_all_reports_and_wake_exactly_once() -> Result<()> {
        let (_directory, store, orchestrator, first, second) = fixture().await?;
        let parent = TaskIdentity {
            agent_id: "root".into(),
            generation: 0,
        };
        let join = orchestrator
            .wait_for(
                parent.clone(),
                "join".into(),
                vec![first.clone(), second.clone()],
                FailurePolicy::Collect,
            )
            .await?;
        assert_eq!(join.remaining, 2);
        assert!(
            orchestrator
                .report(
                    TaskIdentity {
                        agent_id: first.clone(),
                        generation: 0
                    },
                    "done".into(),
                    TaskResult::Success(json!(1))
                )
                .await?
        );
        assert!(store.admit().await?.is_none());
        assert!(
            !orchestrator
                .report(
                    TaskIdentity {
                        agent_id: first,
                        generation: 0
                    },
                    "done".into(),
                    TaskResult::Success(json!(1))
                )
                .await?
        );
        orchestrator
            .report(
                TaskIdentity {
                    agent_id: second,
                    generation: 0,
                },
                "done".into(),
                TaskResult::Success(json!(2)),
            )
            .await?;
        let lease = store.admit().await?.context("join wakeup")?;
        assert_eq!(lease.inputs.len(), 1);
        assert_eq!(lease.agent_id, "root");
        store
            .finish(lease, json!({}), SliceOutcome::Waiting)
            .await?;
        assert!(store.admit().await?.is_none());
        assert_eq!(orchestrator.reports("root".into(), 0, 1).await?.len(), 1);
        store.shutdown().await
    }

    #[tokio::test]
    async fn graph_rejects_cycles_cross_run_edges_and_stale_reports() -> Result<()> {
        let (_directory, store, orchestrator, first, second) = fixture().await?;
        orchestrator
            .wait_for(
                TaskIdentity {
                    agent_id: first.clone(),
                    generation: 0,
                },
                "join".into(),
                vec![second.clone()],
                FailurePolicy::Collect,
            )
            .await?;
        assert!(
            orchestrator
                .wait_for(
                    TaskIdentity {
                        agent_id: second,
                        generation: 0
                    },
                    "cycle".into(),
                    vec![first.clone()],
                    FailurePolicy::Collect
                )
                .await
                .is_err()
        );
        store.create_root("other".into(), json!({})).await?;
        assert!(
            orchestrator
                .wait_for(
                    TaskIdentity {
                        agent_id: first.clone(),
                        generation: 0
                    },
                    "cross".into(),
                    vec!["other".into()],
                    FailurePolicy::Collect
                )
                .await
                .is_err()
        );
        store.cancel(first.clone()).await?;
        assert!(
            orchestrator
                .report(
                    TaskIdentity {
                        agent_id: first,
                        generation: 0
                    },
                    "late".into(),
                    TaskResult::Success(Value::Null)
                )
                .await
                .is_err()
        );
        store.shutdown().await
    }

    #[tokio::test]
    async fn fail_fast_reports_failures_and_preserves_user_pause() -> Result<()> {
        let (_directory, store, orchestrator, first, second) = fixture().await?;
        orchestrator
            .wait_for(
                TaskIdentity {
                    agent_id: "root".into(),
                    generation: 0,
                },
                "join".into(),
                vec![first.clone(), second],
                FailurePolicy::FailFast,
            )
            .await?;
        store.pause("root".into()).await?;
        orchestrator
            .report(
                TaskIdentity {
                    agent_id: first,
                    generation: 0,
                },
                "failed".into(),
                TaskResult::Failed("check failed".into()),
            )
            .await?;
        assert_eq!(store.snapshot("root".into()).await?.state, "paused");
        assert!(store.admit().await?.is_none());
        store.resume("root".into()).await?;
        assert_eq!(store.admit().await?.context("resumed")?.agent_id, "root");
        store.shutdown().await
    }
}

fn tool_result_input(job: &crate::Job, result: Value, is_error: bool) -> Result<InputMessage> {
    let id = job
        .payload
        .get("tool_call_id")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("job tool call id"))?;
    let call_id = job
        .payload
        .get("provider_call_id")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let text = serde_json::to_string(&serde_json::json!({"is_error":is_error,"result":result}))?;
    Ok(InputMessage {
        content: infinity_agent_core::message::InputMessageContent::User(
            infinity_provider_protocol::UserContent::ToolResult(
                infinity_provider_protocol::ToolResult {
                    id: id.into(),
                    call_id,
                    content: vec![infinity_provider_protocol::ToolResultContent::text(text)],

                    ..Default::default()
                },
            ),
        ),
        group_id: job.agent_id.clone().into(),
        metadata: None,
        synthetic: None,
        display_as: None,
        subscription: false,
    })
}
