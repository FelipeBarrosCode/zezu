use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use anyhow::{Result, anyhow, ensure};
use infinity_agent_core::message::InfinityMessage;
use rusqlite::{Connection, params};

pub const DEFAULT_HISTORY_BYTES: usize = 32 * 1024 * 1024;

struct Inner {
    maximum: usize,
    reserved: AtomicUsize,
}

#[derive(Clone)]
pub struct HistoryBudget(Arc<Inner>);

struct Reservation {
    budget: HistoryBudget,
    bytes: usize,
    retained: bool,
}

impl HistoryBudget {
    pub fn new(maximum: usize) -> Result<Self> {
        ensure!(maximum > 0, "history memory budget must be positive");
        Ok(Self(Arc::new(Inner {
            maximum,
            reserved: AtomicUsize::new(0),
        })))
    }

    fn reserve(&self, bytes: usize) -> Result<Reservation> {
        self.0.reserved.fetch_update(Ordering::AcqRel, Ordering::Acquire,
            |current| current.checked_add(bytes).filter(|total| *total <= self.0.maximum))
            .map_err(|_| anyhow!("conversation history exceeds the execution memory budget; compact or start a fresh branch before resuming"))?;
        Ok(Reservation {
            budget: self.clone(),
            bytes,
            retained: false,
        })
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        if !self.retained {
            self.budget
                .0
                .reserved
                .fetch_sub(self.bytes, Ordering::AcqRel);
        }
    }
}

pub(crate) fn load_bounded_history(
    connection: &Connection,
    agent_id: &str,
    start: Option<i64>,
    end: Option<i64>,
    budget: &HistoryBudget,
) -> Result<Vec<InfinityMessage>> {
    let start = start.unwrap_or(0);
    let end = end.unwrap_or(i64::MAX);
    ensure!(start >= 0 && end >= 0, "negative history boundary");
    let transaction = connection.unchecked_transaction()?;
    let marker = crate::compaction::latest_context(&transaction, agent_id, end)?
        .filter(|marker| marker.cutoff >= start);
    let start = marker.as_ref().map_or(start, |marker| marker.cutoff);
    let marker_bytes = marker
        .as_ref()
        .map(|marker| {
            serde_json::to_vec(&marker.message).map(|payload| {
                payload
                    .len()
                    .saturating_add(std::mem::size_of::<InfinityMessage>())
            })
        })
        .transpose()?
        .unwrap_or(0);
    // Both statements share a snapshot, so concurrent appends cannot evade the preflight.
    let bytes: usize = transaction.query_row(
        "SELECT COALESCE(SUM(length(CAST(payload AS BLOB))+?),0) FROM history WHERE agent_id=? AND position>=? AND position<?",
        params![std::mem::size_of::<InfinityMessage>(), agent_id, start, end], |row| row.get(0))?;
    let mut reservation = budget.reserve(bytes.saturating_add(marker_bytes))?;
    let messages = {
        let mut statement = transaction.prepare("SELECT payload FROM history WHERE agent_id=? AND position>=? AND position<? ORDER BY position")?;
        let mut rows = statement.query(params![agent_id, start, end])?;
        let mut messages = Vec::new();
        if let Some(marker) = marker {
            messages.push(marker.message);
        }
        while let Some(row) = rows.next()? {
            let payload = row.get_ref(0)?.as_str()?;
            messages.push(serde_json::from_str(payload)?);
        }
        messages
    };
    transaction.commit()?;
    // A slice's ancestor/history loads remain charged until the entire slice is dropped.
    reservation.retained = true;
    Ok(messages)
}

#[cfg(test)]
mod tests {
    use super::*;
    use infinity_provider_protocol::UserContent;

    fn database() -> Result<Connection> {
        let connection = Connection::open_in_memory()?;
        connection
            .execute_batch("CREATE TABLE history(agent_id TEXT,position INTEGER,payload TEXT)")?;
        Ok(connection)
    }

    #[test]
    fn preflight_rejects_over_budget_history_before_deserializing() -> Result<()> {
        let mut connection = database()?;
        connection.execute(
            "INSERT INTO history VALUES('agent',0,?)",
            ["not JSON".repeat(100)],
        )?;
        let budget = HistoryBudget::new(10)?;
        let error = load_bounded_history(&mut connection, "agent", None, None, &budget)
            .expect_err("budget must reject");
        assert!(
            error
                .to_string()
                .contains("compact or start a fresh branch")
        );
        assert_eq!(budget.0.reserved.load(Ordering::Acquire), 0);
        Ok(())
    }

    #[test]
    fn ancestor_loads_share_the_same_budget_and_failed_reads_release_reservations() -> Result<()> {
        let mut connection = database()?;
        let message = InfinityMessage::User {
            content: UserContent::text("history"),
        };
        let payload = serde_json::to_string(&message)?;
        connection.execute("INSERT INTO history VALUES('parent',0,?)", [&payload])?;
        connection.execute("INSERT INTO history VALUES('child',0,?)", [&payload])?;
        let bytes = payload.len() + std::mem::size_of::<InfinityMessage>();
        let budget = HistoryBudget::new(bytes)?;
        assert_eq!(
            load_bounded_history(&mut connection, "parent", None, None, &budget)?.len(),
            1
        );
        assert!(load_bounded_history(&mut connection, "child", None, None, &budget).is_err());
        let budget = HistoryBudget::new(4096)?;
        connection.execute("INSERT INTO history VALUES('broken',0,'not JSON')", [])?;
        assert!(load_bounded_history(&mut connection, "broken", None, None, &budget).is_err());
        assert_eq!(budget.0.reserved.load(Ordering::Acquire), 0);
        assert_eq!(
            load_bounded_history(&mut connection, "child", Some(0), Some(1), &budget)?.len(),
            1
        );
        Ok(())
    }
}
