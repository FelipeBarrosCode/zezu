use anyhow::{Result, ensure};
use language_model_core::TokenUsage;
use rusqlite::{OptionalExtension as _, params};
use serde::{Deserialize, Serialize};

use crate::RuntimeStore;

#[derive(Clone)]
pub struct ProviderUsageLedger {
    store: RuntimeStore,
    agent_id: String,
    model_id: String,
}

#[derive(Clone)]
pub struct ProviderUsageAttempt {
    ledger: ProviderUsageLedger,
    id: String,
    generation: Option<i64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProviderUsageRecord {
    pub attempt_id: String,
    pub agent_id: String,
    pub model_id: String,
    pub usage: TokenUsage,
    pub completed: bool,
}

impl ProviderUsageLedger {
    pub fn new(store: RuntimeStore, agent_id: String, model_id: String) -> Self {
        Self {
            store,
            agent_id,
            model_id,
        }
    }

    pub async fn begin_attempt(&self) -> Result<ProviderUsageAttempt> {
        let mut attempt = self.attempt();
        attempt.generation = Some(self.store.snapshot(self.agent_id.clone()).await?.generation);
        attempt.record(TokenUsage::default(), false).await?;
        Ok(attempt)
    }

    pub fn attempt(&self) -> ProviderUsageAttempt {
        ProviderUsageAttempt {
            ledger: self.clone(),
            id: uuid::Uuid::new_v4().to_string(),
            generation: None,
        }
    }
}

impl ProviderUsageAttempt {
    pub async fn record_provider_error(
        &self,
        error: &language_model_core::LanguageModelCompletionError,
    ) -> Result<()> {
        if !matches!(
            error,
            language_model_core::LanguageModelCompletionError::ProviderRejection {
                category: language_model_core::ProviderErrorCategory::PromptTooLarge { .. },
                ..
            }
        ) {
            return Ok(());
        }
        let generation = self
            .generation
            .ok_or_else(|| anyhow::anyhow!("provider attempt has no generation fence"))?;
        let agent_id = self.ledger.agent_id.clone();
        let model_id = self.ledger.model_id.clone();
        self.ledger.store.write(move |connection| {
            connection.execute_batch("CREATE TABLE IF NOT EXISTS runtime_provider_context_overflow(agent_id TEXT PRIMARY KEY REFERENCES agents(id), model_id TEXT NOT NULL, generation INTEGER NOT NULL)")?;
            connection.execute("INSERT INTO runtime_provider_context_overflow(agent_id,model_id,generation)
                SELECT id,?,generation FROM agents WHERE id=? AND generation=? AND state='running'
                ON CONFLICT(agent_id) DO UPDATE SET model_id=excluded.model_id,generation=excluded.generation",
                params![model_id,agent_id,generation])?;
            Ok(())
        }).await
    }

    pub async fn record(&self, usage: TokenUsage, completed: bool) -> Result<()> {
        let agent_id = self.ledger.agent_id.clone();
        let model_id = self.ledger.model_id.clone();
        let attempt_id = self.id.clone();
        self.ledger.store.write(move |connection| {
            // Usage is billable even when inference never reaches a history checkpoint.
            connection.execute_batch("CREATE TABLE IF NOT EXISTS provider_usage (
                attempt_id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL REFERENCES agents(id),
                model_id TEXT NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                cache_creation_tokens INTEGER NOT NULL,
                cache_read_tokens INTEGER NOT NULL,
                completed INTEGER NOT NULL CHECK(completed IN (0,1))
            ); CREATE INDEX IF NOT EXISTS provider_usage_agent ON provider_usage(agent_id,attempt_id);")?;
            connection.execute("INSERT INTO provider_usage VALUES(?,?,?,?,?,?,?,?)
                ON CONFLICT(attempt_id) DO UPDATE SET
                    input_tokens=excluded.input_tokens, output_tokens=excluded.output_tokens,
                    cache_creation_tokens=excluded.cache_creation_tokens,
                    cache_read_tokens=excluded.cache_read_tokens, completed=excluded.completed
                WHERE provider_usage.completed=0", params![attempt_id, agent_id, model_id,
                    usage.input_tokens, usage.output_tokens, usage.cache_creation_input_tokens,
                    usage.cache_read_input_tokens, completed])?;
            Ok(())
        }).await
    }
}

pub(crate) fn clear_context_overflow(
    connection: &rusqlite::Connection,
    agent_id: &str,
) -> Result<()> {
    let exists: bool = connection.query_row("SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='runtime_provider_context_overflow')", [], |row| row.get(0))?;
    if exists {
        connection.execute(
            "DELETE FROM runtime_provider_context_overflow WHERE agent_id=?",
            [agent_id],
        )?;
    }
    Ok(())
}

impl RuntimeStore {
    pub async fn provider_context_overflow(&self, agent_id: String) -> Result<bool> {
        self.read(move |connection| {
            let exists: bool = connection.query_row("SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='runtime_provider_context_overflow')", [], |row| row.get(0))?;
            if !exists { return Ok(false); }
            Ok(connection.query_row("SELECT EXISTS(SELECT 1 FROM runtime_provider_context_overflow WHERE agent_id=?)", [agent_id], |row| row.get(0))?)
        }).await
    }

    pub async fn provider_usage(
        &self,
        agent_id: String,
        after: Option<String>,
        limit: usize,
    ) -> Result<Vec<ProviderUsageRecord>> {
        ensure!(
            (1..=1000).contains(&limit),
            "usage page size must be between 1 and 1000"
        );
        self.read(move |connection| {
            let exists = connection.query_row("SELECT 1 FROM sqlite_master WHERE type='table' AND name='provider_usage'", [], |row| row.get::<_, i64>(0)).optional()?.is_some();
            if !exists { return Ok(Vec::new()); }
            let mut statement = connection.prepare("SELECT attempt_id,agent_id,model_id,input_tokens,output_tokens,cache_creation_tokens,cache_read_tokens,completed
                FROM provider_usage WHERE agent_id=? AND (? IS NULL OR attempt_id>?) ORDER BY attempt_id LIMIT ?")?;
            let rows = statement.query_map(params![agent_id,after,after,limit], |row| Ok(ProviderUsageRecord {
                attempt_id: row.get(0)?, agent_id: row.get(1)?, model_id: row.get(2)?,
                usage: TokenUsage { input_tokens: row.get(3)?, output_tokens: row.get(4)?,
                    cache_creation_input_tokens: row.get(5)?, cache_read_input_tokens: row.get(6)? },
                completed: row.get(7)?,
            }))?;
            Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
        }).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RuntimeLimits;

    #[tokio::test]
    async fn usage_snapshots_are_idempotent_and_keep_cache_counters_after_restart() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("runtime.sqlite");
        let store = RuntimeStore::open(&path, RuntimeLimits::default())?;
        store
            .create_root("agent".into(), serde_json::json!({}))
            .await?;
        assert!(
            store
                .provider_usage("agent".into(), None, 10)
                .await?
                .is_empty()
        );
        let attempt =
            ProviderUsageLedger::new(store.clone(), "agent".into(), "provider/model".into())
                .attempt();
        let usage = TokenUsage {
            input_tokens: 9,
            output_tokens: 4,
            cache_creation_input_tokens: 3,
            cache_read_input_tokens: 2,
        };
        attempt.record(TokenUsage::default(), false).await?;
        attempt.record(usage, true).await?;
        attempt.record(TokenUsage::default(), false).await?;
        store.shutdown().await?;
        let restored = RuntimeStore::open(path, RuntimeLimits::default())?;
        let rows = restored.provider_usage("agent".into(), None, 10).await?;
        assert_eq!(rows.len(), 1);
        let row = rows
            .first()
            .ok_or_else(|| anyhow::anyhow!("missing usage"))?;
        assert_eq!(row.usage, usage);
        assert!(row.completed);
        assert!(
            restored
                .provider_usage("agent".into(), Some(row.attempt_id.clone()), 10)
                .await?
                .is_empty()
        );
        assert!(
            restored
                .provider_usage("agent".into(), None, 0)
                .await
                .is_err()
        );
        restored.shutdown().await
    }
    #[tokio::test]
    async fn context_overflow_is_typed_generation_fenced_and_cleared_explicitly() -> Result<()> {
        use language_model_core::{
            LanguageModelCompletionError, LanguageModelProviderName, ProviderErrorCategory,
        };
        let directory = tempfile::tempdir()?;
        let store = RuntimeStore::open(
            directory.path().join("runtime.sqlite"),
            RuntimeLimits::default(),
        )?;
        store
            .create_root("agent".into(), serde_json::json!({}))
            .await?;
        assert!(!store.provider_context_overflow("agent".into()).await?);
        store
            .enqueue_text_input("agent".into(), "first".into(), "start".into())
            .await?;
        let lease = store
            .admit()
            .await?
            .ok_or_else(|| anyhow::anyhow!("missing lease"))?;
        let ledger = ProviderUsageLedger::new(store.clone(), "agent".into(), "model".into());
        let attempt = ledger.begin_attempt().await?;
        let error = |category| LanguageModelCompletionError::ProviderRejection {
            provider: LanguageModelProviderName("Fake".into()),
            status: None,
            code: None,
            message: "untrusted provider prose".into(),
            retry_after: None,
            category,
        };
        attempt
            .record_provider_error(&error(ProviderErrorCategory::InvalidRequest))
            .await?;
        assert!(!store.provider_context_overflow("agent".into()).await?);
        attempt
            .record_provider_error(&error(ProviderErrorCategory::PromptTooLarge {
                tokens: None,
            }))
            .await?;
        assert!(store.provider_context_overflow("agent".into()).await?);
        store
            .write(|connection| clear_context_overflow(connection, "agent"))
            .await?;
        assert!(!store.provider_context_overflow("agent".into()).await?);
        store.interrupt(lease).await?;
        attempt
            .record_provider_error(&error(ProviderErrorCategory::PromptTooLarge {
                tokens: None,
            }))
            .await?;
        assert!(
            !store.provider_context_overflow("agent".into()).await?,
            "stale model errors cannot mark a new generation"
        );
        store.shutdown().await
    }
}
