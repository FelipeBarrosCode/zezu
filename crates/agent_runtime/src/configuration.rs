use anyhow::{Context as _, Result, ensure};
use rusqlite::{Connection, OptionalExtension as _, params};
use serde_json::Value;

use crate::RuntimeStore;

#[derive(Clone, Debug)]
pub struct ConfigurationLease {
    pub agent_id: String,
    pub token: String,
    pub generation: i64,
    pub was_paused: bool,
    pub previous_state: String,
}

pub(crate) fn initialize_connection(connection: &Connection) -> Result<()> {
    connection.execute_batch("CREATE TABLE IF NOT EXISTS runtime_configuration_locks(agent_id TEXT PRIMARY KEY REFERENCES agents(id),token TEXT NOT NULL UNIQUE,generation INTEGER NOT NULL,previous_state TEXT NOT NULL,previous_pause INTEGER NOT NULL);
      CREATE VIEW IF NOT EXISTS runtime_configuration_blocked AS WITH RECURSIVE blocked(agent_id) AS (SELECT agent_id FROM runtime_configuration_locks UNION SELECT agents.id FROM agents JOIN blocked ON agents.parent_id=blocked.agent_id) SELECT agent_id FROM blocked;
      CREATE TRIGGER IF NOT EXISTS runtime_dispatch_configuration_insert AFTER INSERT ON runtime_configuration_locks BEGIN UPDATE runtime_dispatch_clock SET revision=revision+1 WHERE singleton=1; END;
      CREATE TRIGGER IF NOT EXISTS runtime_dispatch_configuration_delete AFTER DELETE ON runtime_configuration_locks BEGIN UPDATE runtime_dispatch_clock SET revision=revision+1 WHERE singleton=1; END;")?;
    Ok(())
}

pub(crate) fn recover_connection(connection: &Connection) -> Result<()> {
    connection.execute("DELETE FROM runtime_configuration_locks", [])?;
    Ok(())
}

pub(crate) fn ensure_scope_unlocked(connection: &Connection, agent_id: &str) -> Result<()> {
    let locked:bool=connection.query_row("WITH RECURSIVE descendants(id) AS (SELECT id FROM agents WHERE id=? UNION ALL SELECT agents.id FROM agents JOIN descendants ON agents.parent_id=descendants.id), ancestors(id,parent_id) AS (SELECT id,parent_id FROM agents WHERE id=? UNION ALL SELECT agents.id,agents.parent_id FROM agents JOIN ancestors ON agents.id=ancestors.parent_id) SELECT EXISTS(SELECT 1 FROM runtime_configuration_locks WHERE agent_id IN (SELECT id FROM descendants UNION SELECT id FROM ancestors))",params![agent_id,agent_id],|row|row.get(0))?;
    ensure!(
        !locked,
        "agent configuration is changing; retry this action after it completes"
    );
    let compaction_table: bool = connection.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE name='runtime_compaction_attempts')",
        [],
        |row| row.get(0),
    )?;
    if compaction_table {
        let compacting:bool=connection.query_row("WITH RECURSIVE descendants(id) AS (SELECT id FROM agents WHERE id=? UNION ALL SELECT agents.id FROM agents JOIN descendants ON agents.parent_id=descendants.id), ancestors(id,parent_id) AS (SELECT id,parent_id FROM agents WHERE id=? UNION ALL SELECT agents.id,agents.parent_id FROM agents JOIN ancestors ON agents.id=ancestors.parent_id) SELECT EXISTS(SELECT 1 FROM runtime_compaction_attempts attempts JOIN agents ON agents.id=attempts.agent_id AND agents.generation=attempts.generation WHERE attempts.agent_id IN (SELECT id FROM descendants UNION SELECT id FROM ancestors))",params![agent_id,agent_id],|row|row.get(0))?;
        ensure!(
            !compacting,
            "agent compaction is in progress; retry after its checkpoint completes"
        );
    }
    Ok(())
}

impl RuntimeStore {
    pub async fn begin_configuration(&self, agent_id: String) -> Result<ConfigurationLease> {
        self.write(move |connection| {
            let transaction=connection.transaction()?;
            ensure_scope_unlocked(&transaction,&agent_id)?;
            let (state,generation,pause,closed):(String,i64,bool,bool)=transaction.query_row("SELECT state,generation,pause_requested,closed FROM agents WHERE id=?",[&agent_id],|row|Ok((row.get(0)?,row.get(1)?,row.get(2)?,row.get(3)?))).context("agent is unavailable")?;
            ensure!(!closed && matches!(state.as_str(),"queued"|"waiting"|"paused"|"cancelled"),"wait for the active turn to stop before changing configuration");
            let busy:bool=transaction.query_row("WITH RECURSIVE scope(id) AS (SELECT id FROM agents WHERE id=? UNION ALL SELECT agents.id FROM agents JOIN scope ON agents.parent_id=scope.id) SELECT EXISTS(SELECT 1 FROM agents WHERE id IN (SELECT id FROM scope) AND state='running') OR EXISTS(SELECT 1 FROM jobs WHERE agent_id IN (SELECT id FROM scope) AND state IN ('queued','dispatched','uncertain')) OR EXISTS(SELECT 1 FROM runtime_joins joins JOIN agents ON agents.id=joins.agent_id WHERE joins.agent_id IN (SELECT id FROM scope) AND joins.generation=agents.generation AND joins.state='pending')",[&agent_id],|row|row.get(0))?;
            ensure!(!busy,"finish or reconcile the active tasks and tool jobs before changing configuration");
            let token=uuid::Uuid::new_v4().to_string();
            transaction.execute("INSERT INTO runtime_configuration_locks(agent_id,token,generation,previous_state,previous_pause) VALUES (?,?,?,?,?)",params![agent_id,token,generation,state,pause])?;
            if state!="cancelled" {transaction.execute("UPDATE agents SET state='paused',pause_requested=1 WHERE id=?",[&agent_id])?;}
            transaction.commit()?;
            Ok(ConfigurationLease {agent_id,token,generation,was_paused:state=="paused"||pause,previous_state:state})
        }).await
    }

    pub async fn commit_configuration(
        &self,
        lease: ConfigurationLease,
        metadata: Value,
    ) -> Result<()> {
        let metadata = serde_json::to_string(&metadata)?;
        ensure!(
            metadata.len() <= self.limits().max_input_bytes,
            "configuration exceeds the runtime payload limit"
        );
        self.write(move |connection| {
            let transaction = connection.transaction()?;
            let saved =
                load_lease(&transaction, &lease)?.context("configuration lease has expired")?;
            let (generation, state): (i64, String) = transaction.query_row(
                "SELECT generation,state FROM agents WHERE id=?",
                [&lease.agent_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )?;
            ensure!(
                generation == lease.generation && generation == saved.0,
                "configuration lease is stale"
            );
            ensure!(
                state
                    == if saved.1 == "cancelled" {
                        "cancelled"
                    } else {
                        "paused"
                    },
                "agent state changed during configuration"
            );
            transaction.execute(
                "UPDATE agents SET metadata=? WHERE id=?",
                params![metadata, lease.agent_id],
            )?;
            restore_state(&transaction, &lease.agent_id, &saved.1, saved.2)?;
            transaction.execute(
                "DELETE FROM runtime_configuration_locks WHERE agent_id=? AND token=?",
                params![lease.agent_id, lease.token],
            )?;
            transaction.commit()?;
            Ok(())
        })
        .await
    }

    pub async fn abort_configuration(&self, lease: ConfigurationLease) -> Result<()> {
        self.write(move |connection| {
            let transaction = connection.transaction()?;
            if let Some(saved) = load_lease(&transaction, &lease)? {
                let (generation, state): (i64, String) = transaction.query_row(
                    "SELECT generation,state FROM agents WHERE id=?",
                    [&lease.agent_id],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )?;
                if generation == lease.generation
                    && generation == saved.0
                    && (state == "paused" || state == "cancelled" && saved.1 == "cancelled")
                {
                    restore_state(&transaction, &lease.agent_id, &saved.1, saved.2)?;
                }
                transaction.execute(
                    "DELETE FROM runtime_configuration_locks WHERE agent_id=? AND token=?",
                    params![lease.agent_id, lease.token],
                )?;
            }
            transaction.commit()?;
            Ok(())
        })
        .await
    }
}

fn load_lease(
    connection: &Connection,
    lease: &ConfigurationLease,
) -> Result<Option<(i64, String, bool)>> {
    Ok(connection.query_row("SELECT generation,previous_state,previous_pause FROM runtime_configuration_locks WHERE agent_id=? AND token=?",params![lease.agent_id,lease.token],|row|Ok((row.get(0)?,row.get(1)?,row.get(2)?))).optional()?)
}

fn restore_state(
    connection: &Connection,
    agent_id: &str,
    previous: &str,
    pause: bool,
) -> Result<()> {
    if previous == "cancelled" {
        connection.execute(
            "UPDATE agents SET state='cancelled',pause_requested=? WHERE id=?",
            params![pause, agent_id],
        )?;
    } else if previous == "paused" || pause {
        connection.execute(
            "UPDATE agents SET state='paused',pause_requested=1 WHERE id=?",
            [agent_id],
        )?;
    } else {
        connection.execute("UPDATE agents SET state=CASE WHEN EXISTS(SELECT 1 FROM inbox WHERE agent_id=agents.id AND processed=0) THEN 'queued' ELSE 'waiting' END,pause_requested=0 WHERE id=?",[agent_id])?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RuntimeLimits;
    use serde_json::json;

    #[tokio::test]
    async fn configuration_fences_admission_resume_and_user_input_until_commit() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let store = RuntimeStore::open(
            directory.path().join("configuration.sqlite"),
            RuntimeLimits::default(),
        )?;
        store
            .create_root("root".into(), json!({"model":"old"}))
            .await?;
        let child = store
            .spawn_child("root".into(), "child".into(), 0, true)
            .await?;
        store
            .enqueue_text_input(child.clone(), "input".into(), "go".into())
            .await?;
        let lease = store.begin_configuration("root".into()).await?;
        assert!(store.admit().await?.is_none());
        assert!(store.resume(child.clone()).await.is_err());
        assert!(
            store
                .enqueue_text_input("root".into(), "racing-input".into(), "go".into())
                .await
                .is_err()
        );
        assert!(store.begin_configuration(child.clone()).await.is_err());
        store
            .commit_configuration(lease, json!({"model":"new"}))
            .await?;
        assert_eq!(store.metadata("root".into()).await?, json!({"model":"new"}));
        assert_eq!(
            store.admit().await?.map(|lease| lease.agent_id),
            Some(child)
        );
        store.shutdown().await
    }

    #[tokio::test]
    async fn configuration_blocks_public_spawns_and_orchestration_until_abort() -> Result<()> {
        use crate::orchestration::{FailurePolicy, Orchestrator, TaskIdentity, TaskResult};
        let directory = tempfile::tempdir()?;
        let store = RuntimeStore::open(
            directory.path().join("configuration.sqlite"),
            RuntimeLimits::default(),
        )?;
        store.create_root("root".into(), json!({})).await?;
        let child = store
            .spawn_child("root".into(), "existing".into(), 0, true)
            .await?;
        let orchestrator = Orchestrator::open(store.clone()).await?;
        let lease = store.begin_configuration("root".into()).await?;
        assert!(store.pause("root".into()).await.is_err());
        assert!(store.pause(child.clone()).await.is_err());
        assert!(
            store
                .spawn_child("root".into(), "blocked".into(), 0, true)
                .await
                .is_err()
        );
        assert!(
            store
                .spawn_child(child.clone(), "blocked-descendant".into(), 0, true)
                .await
                .is_err()
        );
        let identity = TaskIdentity {
            agent_id: child,
            generation: 0,
        };
        assert!(
            orchestrator
                .wait_for(
                    identity.clone(),
                    "blocked-join".into(),
                    vec!["root".into()],
                    FailurePolicy::Collect
                )
                .await
                .is_err()
        );
        assert!(
            orchestrator
                .report(
                    identity,
                    "blocked-report".into(),
                    TaskResult::Success(Value::Null)
                )
                .await
                .is_err()
        );
        store.abort_configuration(lease).await?;
        assert_eq!(store.snapshot("root".into()).await?.state, "waiting");
        let spawned = store
            .spawn_child("root".into(), "blocked".into(), 0, true)
            .await?;
        assert_eq!(
            store.snapshot(spawned).await?.parent_id.as_deref(),
            Some("root")
        );
        store.shutdown().await
    }

    #[tokio::test]
    async fn aborted_or_cancelled_configuration_never_resurrects_an_agent() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("configuration.sqlite");
        let store = RuntimeStore::open(&path, RuntimeLimits::default())?;
        store
            .create_root("root".into(), json!({"model":"old"}))
            .await?;
        let lease = store.begin_configuration("root".into()).await?;
        store.cancel("root".into()).await?;
        assert!(
            store
                .commit_configuration(lease.clone(), json!({"model":"wrong"}))
                .await
                .is_err()
        );
        store.abort_configuration(lease).await?;
        assert_eq!(store.snapshot("root".into()).await?.state, "cancelled");
        let lease = store.begin_configuration("root".into()).await?;
        assert_eq!(lease.previous_state, "cancelled");
        store
            .commit_configuration(lease, json!({"model":"new"}))
            .await?;
        assert_eq!(store.snapshot("root".into()).await?.state, "cancelled");
        store
            .enqueue_text_input("root".into(), "new-turn".into(), "go".into())
            .await?;
        let abandoned = store.begin_configuration("root".into()).await?;
        store.shutdown().await?;
        let restored = RuntimeStore::open(&path, RuntimeLimits::default())?;
        assert_eq!(restored.snapshot("root".into()).await?.state, "paused");
        assert!(
            restored
                .commit_configuration(abandoned, json!({}))
                .await
                .is_err()
        );
        restored.resume("root".into()).await?;
        restored.shutdown().await
    }
}
