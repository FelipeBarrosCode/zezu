use anyhow::Result;

use crate::{Lease, RuntimeStore};

// These tables are initialized after the shared dispatch clock and its job/state
// triggers. Install the remaining admission dependencies only after all exist.
pub(crate) fn initialize_connection(connection: &rusqlite::Connection) -> Result<()> {
    connection.execute_batch(
        "CREATE TRIGGER IF NOT EXISTS runtime_dispatch_inbox_insert AFTER INSERT ON inbox BEGIN UPDATE runtime_dispatch_clock SET revision=revision+1 WHERE singleton=1; END;
         CREATE TRIGGER IF NOT EXISTS runtime_dispatch_inbox_delete AFTER DELETE ON inbox BEGIN UPDATE runtime_dispatch_clock SET revision=revision+1 WHERE singleton=1; END;
         CREATE TRIGGER IF NOT EXISTS runtime_dispatch_inbox_update AFTER UPDATE OF agent_id,processed,lease_id ON inbox WHEN OLD.agent_id IS NOT NEW.agent_id OR OLD.processed IS NOT NEW.processed OR OLD.lease_id IS NOT NEW.lease_id BEGIN UPDATE runtime_dispatch_clock SET revision=revision+1 WHERE singleton=1; END;
         CREATE TRIGGER IF NOT EXISTS runtime_dispatch_join_insert AFTER INSERT ON runtime_joins BEGIN UPDATE runtime_dispatch_clock SET revision=revision+1 WHERE singleton=1; END;
         CREATE TRIGGER IF NOT EXISTS runtime_dispatch_join_delete AFTER DELETE ON runtime_joins BEGIN UPDATE runtime_dispatch_clock SET revision=revision+1 WHERE singleton=1; END;
         CREATE TRIGGER IF NOT EXISTS runtime_dispatch_join_update AFTER UPDATE OF agent_id,generation,state ON runtime_joins WHEN OLD.agent_id IS NOT NEW.agent_id OR OLD.generation IS NOT NEW.generation OR OLD.state IS NOT NEW.state BEGIN UPDATE runtime_dispatch_clock SET revision=revision+1 WHERE singleton=1; END;
         CREATE TRIGGER IF NOT EXISTS runtime_dispatch_agent_insert AFTER INSERT ON agents BEGIN UPDATE runtime_dispatch_clock SET revision=revision+1 WHERE singleton=1; END;
         CREATE TRIGGER IF NOT EXISTS runtime_dispatch_agent_delete AFTER DELETE ON agents BEGIN UPDATE runtime_dispatch_clock SET revision=revision+1 WHERE singleton=1; END;
         CREATE TRIGGER IF NOT EXISTS runtime_dispatch_agent_identity AFTER UPDATE OF id,root_id,parent_id,closed,lease_id ON agents WHEN OLD.id IS NOT NEW.id OR OLD.root_id IS NOT NEW.root_id OR OLD.parent_id IS NOT NEW.parent_id OR OLD.closed IS NOT NEW.closed OR OLD.lease_id IS NOT NEW.lease_id BEGIN UPDATE runtime_dispatch_clock SET revision=revision+1 WHERE singleton=1; END;
         CREATE TRIGGER IF NOT EXISTS runtime_dispatch_job_agent AFTER UPDATE OF agent_id ON jobs WHEN OLD.agent_id IS NOT NEW.agent_id BEGIN UPDATE runtime_dispatch_clock SET revision=revision+1 WHERE singleton=1; END;
         CREATE TRIGGER IF NOT EXISTS runtime_dispatch_configuration_update AFTER UPDATE ON runtime_configuration_locks BEGIN UPDATE runtime_dispatch_clock SET revision=revision+1 WHERE singleton=1; END;",
    )?;
    Ok(())
}

/// Remembers an empty admission scan until durable eligibility changes.
///
/// Memory stays constant as the parked fleet grows. Active leases, inbox changes,
/// joins, configuration locks and tool jobs all advance the shared SQLite clock.
#[derive(Default)]
pub struct AdmissionCursor {
    empty_at: Option<i64>,
    scans: u64,
}

impl AdmissionCursor {
    pub fn scan_count(&self) -> u64 {
        self.scans
    }

    pub async fn poll(&mut self, store: &RuntimeStore) -> Result<Option<Lease>> {
        let revision = store.dispatch_revision().await?;
        if self.empty_at == Some(revision) {
            return Ok(None);
        }
        let lease = store.admit().await?;
        self.scans = self.scans.saturating_add(1);
        // Do not cache a negative result if a writer changed eligibility between
        // the clock read and the admission transaction. The next poll must retry.
        self.empty_at = if lease.is_none() && store.dispatch_revision().await? == revision {
            Some(revision)
        } else {
            None
        };
        Ok(lease)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{RuntimeLimits, SliceOutcome};
    use rusqlite::params;
    use serde_json::json;

    async fn blocked_fleet_scans_once(count: usize) -> Result<()> {
        let directory = tempfile::tempdir()?;
        let store = RuntimeStore::open(
            directory.path().join("admission.sqlite"),
            RuntimeLimits::default(),
        )?;
        store.create_root("fleet".into(), json!({})).await?;
        store.write(move |connection| {
            let transaction = connection.transaction()?;
            transaction.execute("UPDATE agents SET state='queued' WHERE id='fleet'", [])?;
            transaction.execute("UPDATE runs SET agent_count=? WHERE id='fleet'", [count])?;
            {
                let mut agents = transaction.prepare("INSERT INTO agents(id,root_id,parent_id,state) VALUES (?,'fleet','fleet','queued')")?;
                let mut inputs = transaction.prepare("INSERT INTO inbox(agent_id,message_id,payload) VALUES (?,'input',?)")?;
                let mut jobs = transaction.prepare("INSERT INTO jobs(id,agent_id,generation,state,payload) VALUES (?,?,0,'queued','{}')")?;
                let mut joins = transaction.prepare("INSERT INTO runtime_joins(id,agent_id,generation,join_key,policy) VALUES (?,?,0,'join','collect')")?;
                for index in 0..count {
                    let agent = if index == 0 { "fleet".to_owned() } else { format!("agent-{index:05}") };
                    if index != 0 { agents.execute([&agent])?; }
                    inputs.execute(params![agent,serde_json::to_string(&json!({"group_id":agent,"content":{"type":"text","text":"go"},"metadata":null}))?])?;
                    if index % 2 == 0 { jobs.execute(params![format!("job-{index:05}"),agent])?; }
                    else { joins.execute(params![format!("join-{index:05}"),agent])?; }
                }
            }
            transaction.commit()?;
            Ok(())
        }).await?;
        let mut cursor = AdmissionCursor::default();
        for _ in 0..100 {
            assert!(cursor.poll(&store).await?.is_none());
        }
        assert_eq!(
            cursor.scan_count(),
            1,
            "unchanged blocked agents must not be rescanned every runtime tick"
        );

        store
            .write(|connection| {
                connection.execute("UPDATE jobs SET state='completed' WHERE id='job-00000'", [])?;
                Ok(())
            })
            .await?;
        let lease = cursor
            .poll(&store)
            .await?
            .ok_or_else(|| anyhow::anyhow!("completed job did not wake admission"))?;
        assert_eq!(lease.agent_id, "fleet");
        assert_eq!(cursor.scan_count(), 2);
        store
            .finish(lease, json!({}), SliceOutcome::Completed)
            .await?;

        store
            .write(|connection| {
                connection.execute(
                    "UPDATE runtime_joins SET state='ready' WHERE id='join-00001'",
                    [],
                )?;
                Ok(())
            })
            .await?;
        let lease = cursor
            .poll(&store)
            .await?
            .ok_or_else(|| anyhow::anyhow!("resolved join did not wake admission"))?;
        assert_eq!(lease.agent_id, "agent-00001");
        assert_eq!(cursor.scan_count(), 3);
        store
            .finish(lease, json!({}), SliceOutcome::Completed)
            .await?;
        assert!(cursor.poll(&store).await?.is_none());
        for _ in 0..100 {
            assert!(cursor.poll(&store).await?.is_none());
        }
        assert_eq!(cursor.scan_count(), 4);
        store.shutdown().await
    }

    #[tokio::test]
    async fn blocked_admission_cursor_wakes_on_job_completion_and_join_resolution() -> Result<()> {
        blocked_fleet_scans_once(1_000).await
    }

    #[tokio::test]
    #[ignore = "explicit 75,000 queued-agent admission scan scenario"]
    async fn blocked_admission_fleet_scans_once_until_eligibility_changes() -> Result<()> {
        blocked_fleet_scans_once(75_000).await
    }

    #[tokio::test]
    async fn admission_cursor_wakes_when_pending_input_changes_without_a_state_change() -> Result<()>
    {
        let directory = tempfile::tempdir()?;
        let store = RuntimeStore::open(
            directory.path().join("inbox.sqlite"),
            RuntimeLimits::default(),
        )?;
        store.create_root("agent".into(), json!({})).await?;
        store
            .write(|connection| {
                connection.execute("UPDATE agents SET state='queued' WHERE id='agent'", [])?;
                Ok(())
            })
            .await?;
        let mut cursor = AdmissionCursor::default();
        assert!(cursor.poll(&store).await?.is_none());
        store
            .enqueue_text_input("agent".into(), "input".into(), "go".into())
            .await?;
        let lease = cursor
            .poll(&store)
            .await?
            .ok_or_else(|| anyhow::anyhow!("inbox insertion did not wake admission"))?;
        store
            .finish(lease, json!({}), SliceOutcome::Waiting)
            .await?;
        store
            .write(|connection| {
                connection.execute("UPDATE agents SET state='queued' WHERE id='agent'", [])?;
                Ok(())
            })
            .await?;
        assert!(cursor.poll(&store).await?.is_none());
        store
            .write(|connection| {
                connection.execute("UPDATE inbox SET processed=0 WHERE agent_id='agent'", [])?;
                Ok(())
            })
            .await?;
        assert!(cursor.poll(&store).await?.is_some());
        assert_eq!(cursor.scan_count(), 4);
        store.shutdown().await
    }

    #[tokio::test]
    async fn admission_cursor_wakes_after_close_configuration_and_capacity_blockers_clear()
    -> Result<()> {
        for (block, unblock) in [
            (
                "UPDATE agents SET closed=1 WHERE id='agent'",
                "UPDATE agents SET closed=0 WHERE id='agent'",
            ),
            (
                "INSERT INTO runtime_configuration_locks(agent_id,token,generation,previous_state,previous_pause) VALUES ('agent','config',0,'queued',0)",
                "DELETE FROM runtime_configuration_locks WHERE agent_id='agent'",
            ),
            (
                "INSERT INTO runs(id) VALUES ('busy'); INSERT INTO agents(id,root_id,state,lease_id) VALUES ('busy','busy','running','lease')",
                "UPDATE agents SET state='waiting',lease_id=NULL WHERE id='busy'",
            ),
        ] {
            let directory = tempfile::tempdir()?;
            let store = RuntimeStore::open(
                directory.path().join("blocked.sqlite"),
                RuntimeLimits {
                    max_active_agents: 1,
                    ..RuntimeLimits::default()
                },
            )?;
            store.create_root("agent".into(), json!({})).await?;
            store
                .enqueue_text_input("agent".into(), "input".into(), "go".into())
                .await?;
            store
                .write(move |connection| {
                    connection.execute_batch(block)?;
                    Ok(())
                })
                .await?;
            let mut cursor = AdmissionCursor::default();
            assert!(cursor.poll(&store).await?.is_none());
            assert!(cursor.poll(&store).await?.is_none());
            assert_eq!(cursor.scan_count(), 1);
            store
                .write(move |connection| {
                    connection.execute_batch(unblock)?;
                    Ok(())
                })
                .await?;
            let lease = cursor
                .poll(&store)
                .await?
                .ok_or_else(|| anyhow::anyhow!("admission did not wake after: {unblock}"))?;
            assert_eq!(lease.agent_id, "agent");
            assert_eq!(cursor.scan_count(), 2);
            store
                .finish(lease, json!({}), SliceOutcome::Completed)
                .await?;
            store.shutdown().await?;
        }
        Ok(())
    }
}
