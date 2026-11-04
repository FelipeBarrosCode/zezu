use std::{
    fs::{File, OpenOptions},
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::Duration,
};

use anyhow::{Context as _, Result, anyhow, bail, ensure};
use rusqlite::{Connection, OptionalExtension as _, params};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::{mpsc, oneshot};
use uuid::Uuid;

use crate::RuntimeLimits;

type WriteOperation = Box<dyn FnOnce(&mut Connection) + Send>;

enum Command {
    Write(WriteOperation),
    Shutdown(oneshot::Sender<Result<()>>),
}

struct Inner {
    sender: mpsc::Sender<Command>,
    readers: Vec<mpsc::Sender<WriteOperation>>,
    next_reader: AtomicUsize,
    closed: AtomicBool,
    limits: RuntimeLimits,
}

#[derive(Clone)]
pub struct RuntimeStore(Arc<Inner>);

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AgentSnapshot {
    pub id: String,
    pub root_id: String,
    pub parent_id: Option<String>,
    pub state: String,
    pub generation: i64,
}

#[derive(Clone, Debug)]
pub struct AgentPage {
    pub agents: Vec<AgentSnapshot>,
    pub next_cursor: Option<String>,
}

#[derive(Clone, Debug)]
pub struct Lease {
    pub agent_id: String,
    pub root_id: String,
    pub generation: i64,
    pub id: String,
    pub inputs: Vec<(String, Value)>,
}

#[derive(Clone, Copy, Debug)]
pub enum SliceOutcome {
    Waiting,
    Completed,
    Failed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JobState {
    Queued,
    Dispatched,
    Completed,
    Uncertain,
    Cancelled,
}

impl JobState {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "queued" => Ok(Self::Queued),
            "dispatched" => Ok(Self::Dispatched),
            "completed" => Ok(Self::Completed),
            "uncertain" => Ok(Self::Uncertain),
            "cancelled" => Ok(Self::Cancelled),
            _ => bail!("invalid persisted job state: {value}"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Job {
    pub id: String,
    pub agent_id: String,
    pub generation: i64,
    pub state: JobState,
    pub payload: Value,
    pub result: Option<Value>,
}

pub enum JobRecovery {
    ConfirmedResult(Value),
    ConfirmedNotExecuted,
}

impl RuntimeStore {
    pub fn open(path: impl AsRef<Path>, limits: RuntimeLimits) -> Result<Self> {
        limits.validate()?;
        let path = path.as_ref();
        let owner = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(path.with_extension("sqlite.lock"))?;
        owner
            .try_lock()
            .map_err(|error| anyhow!("runtime database is already owned: {error}"))?;
        let mut connection = Connection::open(path)?;
        connection.busy_timeout(Duration::from_secs(5))?;
        connection.execute_batch(
            "PRAGMA foreign_keys=ON; PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;",
        )?;
        let version_table: bool = connection.query_row(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE name='runtime_version')",
            [],
            |row| row.get(0),
        )?;
        if !version_table {
            let transaction = connection.transaction()?;
            transaction.execute_batch(include_str!("schema.sql"))?;
            transaction.commit()?;
        }
        let version: i64 =
            connection.query_row("SELECT version FROM runtime_version", [], |row| row.get(0))?;
        ensure!(
            version == 1,
            "unsupported runtime database version: {version}"
        );
        crate::jobs::initialize_connection(&connection)?;
        crate::compaction::initialize(&connection)?;
        crate::input_batches::initialize(&connection)?;
        crate::orchestration::Orchestrator::initialize_connection(&connection)?;
        crate::projection::initialize(&connection)?;
        recover(&mut connection)?;
        crate::orchestration::Orchestrator::recover_connection(&connection)?;
        let mut readers = Vec::new();
        for _ in 0..2 {
            let reader =
                Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)?;
            reader.busy_timeout(Duration::from_secs(5))?;
            let (reader_sender, mut reader_receiver) =
                mpsc::channel::<WriteOperation>(limits.command_queue_capacity);
            std::thread::Builder::new()
                .name("agent-runtime-reader".into())
                .spawn(move || {
                    let mut reader = reader;
                    while let Some(operation) = reader_receiver.blocking_recv() {
                        operation(&mut reader);
                    }
                })?;
            readers.push(reader_sender);
        }
        let (sender, receiver) = mpsc::channel(limits.command_queue_capacity);
        std::thread::Builder::new()
            .name("agent-runtime-store".into())
            .spawn(move || writer_loop(connection, owner, receiver))?;
        Ok(Self(Arc::new(Inner {
            sender,
            readers,
            next_reader: AtomicUsize::new(0),
            closed: AtomicBool::new(false),
            limits,
        })))
    }

    pub fn limits(&self) -> &RuntimeLimits {
        &self.0.limits
    }

    pub(crate) async fn write<T: Send + 'static>(
        &self,
        operation: impl FnOnce(&mut Connection) -> Result<T> + Send + 'static,
    ) -> Result<T> {
        ensure!(
            !self.0.closed.load(Ordering::Acquire),
            "runtime store is shut down"
        );
        let (reply, response) = oneshot::channel();
        self.0
            .sender
            .send(Command::Write(Box::new(move |connection| {
                let result = operation(connection);
                if let Err(result) = reply.send(result) {
                    if let Err(error) = result {
                        log::error!("abandoned runtime operation failed: {error:#}");
                    }
                }
            })))
            .await
            .map_err(|_| anyhow!("runtime writer stopped"))?;
        response.await.context("runtime writer dropped response")?
    }

    pub(crate) async fn read<T: Send + 'static>(
        &self,
        operation: impl FnOnce(&Connection) -> Result<T> + Send + 'static,
    ) -> Result<T> {
        ensure!(
            !self.0.closed.load(Ordering::Acquire),
            "runtime store is shut down"
        );
        let index = self.0.next_reader.fetch_add(1, Ordering::Relaxed) % self.0.readers.len();
        let reader = self
            .0
            .readers
            .get(index)
            .context("missing runtime reader")?;
        let (reply, response) = oneshot::channel();
        reader
            .send(Box::new(move |connection| {
                let result = operation(connection);
                if let Err(Err(error)) = reply.send(result) {
                    log::error!("abandoned runtime read failed: {error:#}");
                }
            }))
            .await
            .map_err(|_| anyhow!("runtime reader stopped"))?;
        response.await.context("runtime reader dropped response")?
    }

    pub(crate) fn encode(&self, value: &Value) -> Result<String> {
        let payload = serde_json::to_string(value)?;
        ensure!(
            payload.len() <= self.0.limits.max_input_bytes,
            "runtime payload exceeds byte limit"
        );
        Ok(payload)
    }

    pub async fn create_root(&self, id: String, metadata: Value) -> Result<()> {
        let metadata = self.encode(&metadata)?;
        self.write(move |connection| {
            let transaction = connection.transaction()?;
            transaction.execute("INSERT OR IGNORE INTO runs(id) VALUES (?)", [&id])?;
            transaction.execute(
                "INSERT OR IGNORE INTO agents(id,root_id,metadata) VALUES (?,?,?)",
                params![id, id, metadata],
            )?;
            let parent: Option<String> =
                transaction.query_row("SELECT parent_id FROM agents WHERE id=?", [&id], |row| {
                    row.get(0)
                })?;
            ensure!(parent.is_none(), "agent ID already belongs to a child");
            transaction.commit()?;
            Ok(())
        })
        .await
    }

    pub async fn metadata(&self, agent_id: String) -> Result<Value> {
        self.read(move |connection| {
            let metadata: Option<String> = connection.query_row(
                "SELECT metadata FROM agents WHERE id=?",
                [&agent_id],
                |row| row.get(0),
            )?;
            Ok(metadata
                .map(|value| serde_json::from_str(&value))
                .transpose()?
                .unwrap_or(Value::Null))
        })
        .await
    }

    pub async fn spawn_child(
        &self,
        parent: String,
        spawn_key: String,
        cutoff: i64,
        fresh: bool,
    ) -> Result<String> {
        let maximum = self.0.limits.max_agents_per_run;
        self.write(move |connection| {
            spawn_child(
                connection, &parent, &spawn_key, cutoff, fresh, false, maximum,
            )
        })
        .await
    }

    pub async fn enqueue_user_input(
        &self,
        agent_id: String,
        message_id: String,
        payload: Value,
    ) -> Result<bool> {
        let payload = self.encode(&payload)?;
        self.write(move |connection| {
            let transaction = connection.transaction()?;
            let inserted = enqueue(&transaction, &agent_id, &message_id, &payload, true)?;
            transaction.commit()?;
            Ok(inserted)
        })
        .await
    }

    pub(crate) async fn enqueue_from(
        &self,
        origin: Lease,
        agent_id: String,
        message_id: String,
        payload: Value,
    ) -> Result<bool> {
        let payload = self.encode(&payload)?;
        self.write(move |connection| {
            let transaction = connection.transaction()?;
            check_generation(&transaction, &origin)?;
            let target_root: String = transaction.query_row(
                "SELECT root_id FROM agents WHERE id=?",
                [&agent_id],
                |row| row.get(0),
            )?;
            ensure!(
                target_root == origin.root_id,
                "automatic input cannot cross run boundaries"
            );
            let result = enqueue(&transaction, &agent_id, &message_id, &payload, false)?;
            transaction.commit()?;
            Ok(result)
        })
        .await
    }

    pub async fn admit(&self) -> Result<Option<Lease>> {
        let maximum_active = self.0.limits.max_active_agents;
        let input_limit = self.0.limits.max_inputs_per_slice;
        self.write(move |connection| {
            let transaction = connection.transaction()?;
            let active: usize = transaction.query_row("SELECT count(*) FROM agents WHERE state='running'", [], |row| row.get(0))?;
            if active >= maximum_active { return Ok(None); }
            // AdmissionCursor caches empty results. Changes to these eligibility
            // dependencies must advance the clock initialized by admission/jobs.
            let next: Option<(String,String,i64)> = transaction.query_row(
                "SELECT agents.id,root_id,generation FROM agents JOIN runs ON root_id=runs.id
                 WHERE state='queued' AND closed=0 AND pause_requested=0

                 AND NOT EXISTS(SELECT 1 FROM jobs WHERE jobs.agent_id=agents.id AND jobs.state='uncertain')
                 AND NOT EXISTS(SELECT 1 FROM jobs WHERE jobs.agent_id=agents.id AND jobs.generation=agents.generation AND jobs.state IN ('queued','dispatched'))
                 AND NOT EXISTS(SELECT 1 FROM runtime_joins j WHERE j.agent_id=agents.id AND j.generation=agents.generation AND j.state='pending')
                 AND EXISTS(SELECT 1 FROM inbox WHERE agent_id=agents.id AND processed=0)
                 ORDER BY runs.last_admitted,agents.last_admitted,agents.id LIMIT 1",
                [], |row| Ok((row.get(0)?,row.get(1)?,row.get(2)?)),
            ).optional()?;
            let Some((agent_id, root_id, generation)) = next else { return Ok(None); };
            let lease_id = Uuid::new_v4().to_string();
            let clock: i64 = transaction.query_row("SELECT coalesce(max(last_admitted),0)+1 FROM runs", [], |row| row.get(0))?;
            transaction.execute("UPDATE runs SET last_admitted=? WHERE id=?", params![clock,root_id])?;
            transaction.execute("UPDATE agents SET state='running',lease_id=?,last_admitted=? WHERE id=?", params![lease_id,clock,agent_id])?;
            let mut inputs: Vec<(String, String)> = {
                let mut statement = transaction.prepare("SELECT message_id,payload FROM inbox WHERE agent_id=? AND processed=0 ORDER BY sequence LIMIT ?")?;
                statement.query_map(params![agent_id,input_limit], |row| Ok((row.get(0)?,row.get(1)?)))?.collect::<rusqlite::Result<_>>()?
            };
            crate::input_batches::preserve_boundary(&transaction, &agent_id, &mut inputs)?;
            for (message_id, _) in &inputs {
                transaction.execute("UPDATE inbox SET lease_id=? WHERE agent_id=? AND message_id=?", params![lease_id,agent_id,message_id])?;
            }
            let inputs = inputs.into_iter().map(|(id,payload)| Ok((id,serde_json::from_str(&payload)?))).collect::<Result<_>>()?;
            transaction.commit()?;
            Ok(Some(Lease { agent_id,root_id,generation,id:lease_id,inputs }))
        }).await
    }

    pub async fn finish(
        &self,
        lease: Lease,
        checkpoint: Value,
        outcome: SliceOutcome,
    ) -> Result<()> {
        let checkpoint = self.encode(&checkpoint)?;
        self.write(move |connection| {
            let transaction = connection.transaction()?;
            check_lease(&transaction, &lease)?;
            transaction.execute("DELETE FROM runtime_errors WHERE agent_id=?", [&lease.agent_id])?;
            transaction.execute("UPDATE inbox SET processed=1,lease_id=NULL WHERE agent_id=? AND lease_id=?", params![lease.agent_id,lease.id])?;
            let pending: bool = transaction.query_row("SELECT EXISTS(SELECT 1 FROM inbox WHERE agent_id=? AND processed=0)", [&lease.agent_id], |row| row.get(0))?;
            let state = match outcome {
                SliceOutcome::Completed => "completed",
                SliceOutcome::Failed => "failed",
                SliceOutcome::Waiting if pending => "queued",
                SliceOutcome::Waiting => "waiting",
            };
            transaction.execute("UPDATE agents SET checkpoint=?,lease_id=NULL,state=CASE WHEN closed=1 THEN 'completed' WHEN pause_requested=1 AND ? IN ('queued','waiting') THEN 'paused' ELSE ? END WHERE id=?", params![checkpoint,state,state,lease.agent_id])?;
            crate::orchestration::Orchestrator::reconcile_connection(&transaction, &lease.agent_id)?;
            transaction.commit()?;
            Ok(())
        }).await
    }

    pub async fn interrupt(&self, lease: Lease) -> Result<()> {
        self.write(move |connection| {
            let transaction = connection.transaction()?;
            check_lease(&transaction, &lease)?;
            transaction.execute("UPDATE inbox SET lease_id=NULL WHERE agent_id=? AND lease_id=?", params![lease.agent_id,lease.id])?;
            transaction.execute("UPDATE agents SET state='paused',pause_requested=1,lease_id=NULL,generation=generation+1 WHERE id=?", [&lease.agent_id])?;
            crate::orchestration::Orchestrator::rebind_task(&transaction, &lease.agent_id)?;
            transaction.execute("UPDATE jobs SET state='uncertain' WHERE agent_id=? AND state='dispatched'", [&lease.agent_id])?;
            transaction.commit()?;
            Ok(())
        }).await
    }

    pub async fn pause(&self, agent_id: String) -> Result<()> {
        self.write(move |connection| {
            let changed = connection.execute(
                "WITH RECURSIVE scope(id) AS (SELECT id FROM agents WHERE id=? UNION ALL SELECT a.id FROM agents a JOIN scope s ON a.parent_id=s.id)
                 UPDATE agents SET pause_requested=1,state=CASE WHEN state IN ('queued','waiting') THEN 'paused' ELSE state END
                 WHERE id IN (SELECT id FROM scope) AND state NOT IN ('completed','failed','cancelled')", [&agent_id],
            )?;
            ensure!(changed > 0, "no active agent to pause");
            Ok(())
        }).await
    }

    pub async fn resume(&self, agent_id: String) -> Result<()> {
        self.write(move |connection| {
            let transaction = connection.transaction()?;
            transaction.execute(
                "WITH RECURSIVE scope(id) AS (SELECT id FROM agents WHERE id=? UNION ALL SELECT a.id FROM agents a JOIN scope s ON a.parent_id=s.id)
                 UPDATE agents SET pause_requested=0,state=CASE WHEN EXISTS(SELECT 1 FROM inbox WHERE agent_id=agents.id AND processed=0) THEN 'queued' ELSE 'waiting' END
                 WHERE id IN (SELECT id FROM scope) AND state='paused'", [&agent_id],
            )?;
            transaction.execute("UPDATE jobs SET generation=(SELECT generation FROM agents WHERE agents.id=jobs.agent_id) WHERE state='queued' AND agent_id IN (SELECT id FROM agents WHERE root_id=(SELECT root_id FROM agents WHERE id=?) AND pause_requested=0)", [&agent_id])?;
            transaction.commit()?;
            Ok(())
        }).await?;
        Ok(())
    }

    pub async fn cancel(&self, agent_id: String) -> Result<()> {
        self.write(move |connection| {
            let transaction = connection.transaction()?;
            transaction.execute(
                "WITH RECURSIVE scope(id) AS (SELECT id FROM agents WHERE id=? UNION ALL SELECT a.id FROM agents a JOIN scope s ON a.parent_id=s.id)
                 UPDATE agents SET state='cancelled',generation=generation+1,lease_id=NULL,pause_requested=0
                 WHERE id IN (SELECT id FROM scope) AND state NOT IN ('completed','failed','cancelled')", [&agent_id],
            )?;
            // Discard cancelled prompts, but retain canonical completed tool results
            // for a later explicit user turn, including an unknown-outcome acknowledgement.
            transaction.execute("UPDATE inbox SET processed=CASE WHEN EXISTS(SELECT 1 FROM jobs WHERE jobs.agent_id=inbox.agent_id AND jobs.state='completed' AND jobs.result=inbox.payload) THEN processed ELSE 1 END,lease_id=NULL WHERE agent_id IN (SELECT id FROM agents WHERE state='cancelled')", [])?;
            transaction.execute("UPDATE jobs SET state=CASE WHEN state='dispatched' THEN 'uncertain' ELSE 'cancelled' END WHERE state IN ('queued','dispatched') AND agent_id IN (SELECT id FROM agents WHERE state='cancelled')", [])?;
            let affected = {
                let mut statement = transaction.prepare("WITH RECURSIVE scope(id) AS (SELECT id FROM agents WHERE id=? UNION ALL SELECT a.id FROM agents a JOIN scope s ON a.parent_id=s.id) SELECT id FROM scope")?;
                statement.query_map([&agent_id], |row| row.get::<_,String>(0))?.collect::<rusqlite::Result<Vec<_>>>()?
            };
            for id in affected { crate::orchestration::Orchestrator::reconcile_connection(&transaction,&id)?; }
            transaction.commit()?;
            Ok(())
        }).await
    }

    pub async fn snapshot(&self, agent_id: String) -> Result<AgentSnapshot> {
        self.read(move |connection| {
            connection
                .query_row(
                    "SELECT id,root_id,parent_id,state,generation FROM agents WHERE id=?",
                    [&agent_id],
                    snapshot_row,
                )
                .map_err(Into::into)
        })
        .await
    }

    pub async fn list_agents(
        &self,
        root_id: String,
        after: Option<String>,
        limit: usize,
    ) -> Result<AgentPage> {
        ensure!(
            (1..=256).contains(&limit),
            "page size must be between 1 and 256"
        );
        self.read(move |connection| {
            let mut statement = connection.prepare("SELECT id,root_id,parent_id,state,generation FROM agents WHERE root_id=? AND id>? ORDER BY id LIMIT ?")?;
            let mut agents = statement.query_map(params![root_id,after.unwrap_or_default(),limit+1], snapshot_row)?.collect::<rusqlite::Result<Vec<_>>>()?;
            let has_more = agents.len() > limit;
            agents.truncate(limit);
            let next_cursor = if has_more { agents.last().map(|agent| agent.id.clone()) } else { None };
            Ok(AgentPage { agents,next_cursor })
        }).await
    }

    pub async fn record_job(&self, lease: Lease, id: String, payload: Value) -> Result<()> {
        let payload = self.encode(&payload)?;
        self.write(move |connection| {
            let transaction = connection.transaction()?;
            check_lease(&transaction, &lease)?;
            transaction.execute("INSERT OR IGNORE INTO jobs(id,agent_id,generation,state,payload) VALUES (?,?,?,'queued',?)", params![id,lease.agent_id,lease.generation,payload])?;
            let existing: (String,i64,String) = transaction.query_row("SELECT agent_id,generation,payload FROM jobs WHERE id=?", [&id], |row| Ok((row.get(0)?,row.get(1)?,row.get(2)?)))?;
            ensure!(existing == (lease.agent_id,lease.generation,payload), "job ID reused with different intent");
            transaction.commit()?;
            Ok(())
        }).await
    }

    pub async fn dispatch_job(&self, id: String) -> Result<Job> {
        self.write(move |connection| {
            let transaction = connection.transaction()?;
            let changed = transaction.execute("UPDATE jobs SET state='dispatched' WHERE id=? AND state='queued' AND EXISTS(SELECT 1 FROM agents WHERE agents.id=jobs.agent_id AND agents.generation=jobs.generation AND pause_requested=0  AND state IN ('queued','running','waiting'))", [&id])?;
            ensure!(changed == 1, "job cannot be dispatched in its current state");
            let job = load_job(&transaction, &id)?;
            transaction.commit()?;
            Ok(job)
        }).await
    }

    pub async fn job(&self, id: String) -> Result<Job> {
        self.read(move |connection| load_job(connection, &id)).await
    }

    /// Call only after the tool owner verifies the outcome or obtains the
    /// required user decision. Merely resuming an agent is not reconciliation.
    pub async fn reconcile_job(&self, id: String, recovery: JobRecovery) -> Result<()> {
        let encoded = match &recovery {
            JobRecovery::ConfirmedResult(input) => Some(self.encode(input)?),
            JobRecovery::ConfirmedNotExecuted => None,
        };
        self.write(move |connection| {
            let transaction = connection.transaction()?;
            let job = load_job(&transaction, &id)?;
            ensure!(
                job.state == JobState::Uncertain,
                "job is not awaiting reconciliation"
            );
            let (generation, state): (i64, String) = transaction.query_row(
                "SELECT generation,state FROM agents WHERE id=?",
                [&job.agent_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )?;
            match encoded {
                Some(payload) => {
                    transaction.execute(
                        "UPDATE jobs SET state='completed',result=? WHERE id=?",
                        params![payload, id],
                    )?;
                    if !matches!(state.as_str(), "cancelled" | "completed" | "failed") {
                        enqueue(
                            &transaction,
                            &job.agent_id,
                            &format!("job:{id}:result"),
                            &payload,
                            false,
                        )?;
                    }
                }
                None => {
                    ensure!(
                        !matches!(state.as_str(), "cancelled" | "completed" | "failed"),
                        "cancelled job cannot be retried"
                    );
                    transaction.execute(
                        "UPDATE jobs SET state='queued',generation=? WHERE id=?",
                        params![generation, id],
                    )?;
                }
            }
            transaction.commit()?;
            Ok(())
        })
        .await
    }

    pub async fn complete_job(&self, id: String, input: Value) -> Result<bool> {
        let payload = self.encode(&input)?;
        self.write(move |connection| {
            let transaction = connection.transaction()?;
            let job = load_job(&transaction, &id)?;
            if job.state == JobState::Completed {
                ensure!(
                    job.result.as_ref() == Some(&input),
                    "conflicting duplicate tool result"
                );
                return Ok(false);
            }
            ensure!(
                job.state == JobState::Dispatched,
                "job requires reconciliation before completion"
            );
            let (generation, state): (i64, String) = transaction.query_row(
                "SELECT generation,state FROM agents WHERE id=?",
                [&job.agent_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )?;
            ensure!(
                generation == job.generation && state != "cancelled",
                "stale tool result"
            );
            transaction.execute(
                "UPDATE jobs SET state='completed',result=? WHERE id=?",
                params![payload, id],
            )?;
            let queued = enqueue(
                &transaction,
                &job.agent_id,
                &format!("job:{id}:result"),
                &payload,
                false,
            )?;
            transaction.commit()?;
            Ok(queued)
        })
        .await
    }

    pub async fn shutdown(&self) -> Result<()> {
        ensure!(
            !self.0.closed.swap(true, Ordering::AcqRel),
            "runtime store is already shut down"
        );
        let (reply, response) = oneshot::channel();
        self.0
            .sender
            .send(Command::Shutdown(reply))
            .await
            .map_err(|_| anyhow!("runtime writer stopped"))?;
        response.await.context("runtime shutdown response lost")?
    }
}

fn writer_loop(mut connection: Connection, owner: File, mut receiver: mpsc::Receiver<Command>) {
    let mut shutdown_reply = None;
    while let Some(command) = receiver.blocking_recv() {
        match command {
            Command::Write(operation) => operation(&mut connection),
            Command::Shutdown(reply) => {
                shutdown_reply = Some(reply);
                break;
            }
        }
    }
    let result = recover(&mut connection);
    if let Err(error) = &result {
        log::error!("runtime shutdown checkpoint failed: {error:#}");
    }
    drop(connection);
    let result = result.and(owner.unlock().map_err(Into::into));
    drop(owner);
    if let Some(reply) = shutdown_reply {
        if let Err(Err(error)) = reply.send(result) {
            log::error!("abandoned shutdown failed: {error:#}");
        }
    }
}

fn recover(connection: &mut Connection) -> Result<()> {
    let transaction = connection.transaction()?;
    transaction.execute("UPDATE agents SET state='paused',pause_requested=1,lease_id=NULL,generation=generation+1 WHERE state IN ('queued','running','waiting','paused')", [])?;
    transaction.execute("UPDATE inbox SET lease_id=NULL WHERE processed=0", [])?;
    transaction.execute(
        "UPDATE jobs SET state='uncertain' WHERE state='dispatched'",
        [],
    )?;
    transaction.commit()?;
    Ok(())
}

fn snapshot_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<AgentSnapshot> {
    Ok(AgentSnapshot {
        id: row.get(0)?,
        root_id: row.get(1)?,
        parent_id: row.get(2)?,
        state: row.get(3)?,
        generation: row.get(4)?,
    })
}

pub(crate) fn check_generation(connection: &Connection, origin: &Lease) -> Result<()> {
    let valid: bool = connection.query_row(
        "SELECT EXISTS(SELECT 1 FROM agents WHERE id=? AND generation=? AND state!='cancelled')",
        params![origin.agent_id, origin.generation],
        |row| row.get(0),
    )?;
    ensure!(valid, "stale or cancelled runtime generation");
    Ok(())
}

pub(crate) fn check_lease(connection: &Connection, lease: &Lease) -> Result<()> {
    let valid: bool = connection.query_row("SELECT EXISTS(SELECT 1 FROM agents WHERE id=? AND generation=? AND lease_id=? AND state='running')", params![lease.agent_id,lease.generation,lease.id], |row| row.get(0))?;
    ensure!(valid, "stale runtime execution lease");
    Ok(())
}

pub(crate) fn spawn_child(
    connection: &mut Connection,
    parent: &str,
    key: &str,
    cutoff: i64,
    fresh: bool,
    subscription: bool,
    maximum: usize,
) -> Result<String> {
    ensure!(cutoff >= 0, "negative history cutoff");
    let transaction = connection.transaction()?;
    let parent_state: (String, String, bool) = transaction.query_row(
        "SELECT root_id,state,pause_requested FROM agents WHERE id=?",
        [parent],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
    )?;
    ensure!(
        !matches!(
            parent_state.1.as_str(),
            "cancelled" | "completed" | "failed"
        ),
        "parent is not active"
    );
    if let Some(existing) = transaction
        .query_row(
            "SELECT id FROM agents WHERE parent_id=? AND spawn_key=?",
            params![parent, key],
            |row| row.get::<_, String>(0),
        )
        .optional()?
    {
        return Ok(existing);
    }
    let count: usize = transaction.query_row(
        "SELECT agent_count FROM runs WHERE id=?",
        [&parent_state.0],
        |row| row.get(0),
    )?;
    ensure!(count < maximum, "agent limit reached for this run");
    let available: i64 = transaction.query_row(
        "SELECT count(*) FROM history WHERE agent_id=?",
        [parent],
        |row| row.get(0),
    )?;
    ensure!(
        fresh || cutoff <= available,
        "history cutoff exceeds committed parent history"
    );
    let id = Uuid::new_v4().to_string();
    transaction.execute("INSERT INTO agents(id,root_id,parent_id,spawn_key,parent_cutoff,fresh,subscription_event,state,pause_requested) VALUES (?,?,?,?,?,?,?, ?,?)", params![id,parent_state.0,parent,key,cutoff,fresh,subscription,if parent_state.2 {"paused"} else {"waiting"},parent_state.2])?;
    transaction.execute(
        "UPDATE runs SET agent_count=agent_count+1 WHERE id=?",
        [&parent_state.0],
    )?;
    transaction.commit()?;
    Ok(id)
}

pub(crate) fn enqueue(
    connection: &Connection,
    agent_id: &str,
    message_id: &str,
    payload: &str,
    user: bool,
) -> Result<bool> {
    let input: infinity_agent_core::message::InputMessage = serde_json::from_str(payload)?;
    ensure!(
        input.group_id.as_str() == agent_id,
        "input targets a different agent"
    );
    let (state, closed): (String, bool) = connection.query_row(
        "SELECT state,closed FROM agents WHERE id=?",
        [agent_id],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?;
    ensure!(!closed, "thread is closed");
    ensure!(
        user || !matches!(state.as_str(), "cancelled" | "completed" | "failed"),
        "agent cannot receive automatic input"
    );
    if let Some(existing) = connection
        .query_row(
            "SELECT payload FROM inbox WHERE agent_id=? AND message_id=?",
            params![agent_id, message_id],
            |row| row.get::<_, String>(0),
        )
        .optional()?
    {
        ensure!(
            existing == payload,
            "input dedup ID reused with different content"
        );
        return Ok(false);
    }
    let inserted = connection.execute(
        "INSERT OR IGNORE INTO inbox(agent_id,message_id,payload) VALUES (?,?,?)",
        params![agent_id, message_id, payload],
    )? == 1;
    if inserted {
        if user && matches!(state.as_str(), "completed" | "failed" | "cancelled") {
            crate::orchestration::Orchestrator::rebind_task(connection, agent_id)?;
            connection.execute(
                "UPDATE agents SET generation=generation+1 WHERE id=?",
                [agent_id],
            )?;
        }
        connection.execute("UPDATE agents SET state='queued' WHERE id=? AND state!='running' AND (state!='paused' OR ?)", params![agent_id,user])?;
        if user {
            connection.execute("UPDATE agents SET pause_requested=0 WHERE id=?", [agent_id])?;
        }
    }
    Ok(inserted)
}

fn load_job(connection: &Connection, id: &str) -> Result<Job> {
    let (agent_id, generation, state, payload, result): (
        String,
        i64,
        String,
        String,
        Option<String>,
    ) = connection.query_row(
        "SELECT agent_id,generation,state,payload,result FROM jobs WHERE id=?",
        [id],
        |row| {
            Ok((
                row.get(0)?,
                row.get(1)?,
                row.get(2)?,
                row.get(3)?,
                row.get(4)?,
            ))
        },
    )?;
    Ok(Job {
        id: id.into(),
        agent_id,
        generation,
        state: JobState::parse(&state)?,
        payload: serde_json::from_str(&payload)?,
        result: result
            .map(|result| serde_json::from_str(&result))
            .transpose()?,
    })
}
