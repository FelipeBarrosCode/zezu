use anyhow::{Result, ensure};
use rusqlite::{Connection, params};
use serde_json::Value;

use crate::RuntimeStore;

pub(crate) fn initialize(connection: &Connection) -> Result<()> {
    connection.execute_batch("CREATE TABLE IF NOT EXISTS runtime_input_batches (
        agent_id TEXT NOT NULL REFERENCES agents(id), message_id TEXT NOT NULL, batch_id TEXT NOT NULL,
        PRIMARY KEY(agent_id,message_id));
        CREATE INDEX IF NOT EXISTS runtime_input_batches_group ON runtime_input_batches(agent_id,batch_id);")?;
    Ok(())
}

pub(crate) fn preserve_boundary(
    connection: &Connection,
    agent_id: &str,
    inputs: &mut Vec<(String, String)>,
) -> Result<()> {
    use rusqlite::OptionalExtension as _;
    let Some((last, _)) = inputs.last() else {
        return Ok(());
    };
    let batch: Option<String> = connection
        .query_row(
            "SELECT batch_id FROM runtime_input_batches WHERE agent_id=? AND message_id=?",
            params![agent_id, last],
            |row| row.get(0),
        )
        .optional()?;
    if let Some(batch) = batch {
        let expected:usize=connection.query_row("SELECT count(*) FROM runtime_input_batches b JOIN inbox i ON i.agent_id=b.agent_id AND i.message_id=b.message_id WHERE b.agent_id=? AND b.batch_id=? AND i.processed=0",params![agent_id,batch],|row|row.get(0))?;
        let mut matching = Vec::new();
        for (id, _) in inputs.iter() {
            let member:bool=connection.query_row("SELECT EXISTS(SELECT 1 FROM runtime_input_batches WHERE agent_id=? AND message_id=? AND batch_id=?)",params![agent_id,id,batch],|row|row.get(0))?;
            if member {
                matching.push(id.clone());
            }
        }
        if matching.len() != expected {
            inputs.retain(|(id, _)| !matching.contains(id));
        }
        ensure!(!inputs.is_empty(), "input batch exceeds slice capacity");
    }
    Ok(())
}

impl RuntimeStore {
    pub async fn enqueue_user_batch(
        &self,
        agent_id: String,
        inputs: Vec<(String, Value)>,
    ) -> Result<()> {
        ensure!(
            !inputs.is_empty() && inputs.len() <= self.limits().max_inputs_per_slice,
            "prompt batch exceeds slice input limit"
        );
        let mut bytes = 0usize;
        let inputs = inputs
            .into_iter()
            .map(|(id, value)| {
                let encoded = serde_json::to_string(&value)?;
                bytes = bytes.saturating_add(encoded.len());
                Ok((id, encoded))
            })
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            bytes <= self.limits().max_input_bytes,
            "prompt batch exceeds input byte limit"
        );
        self.write(move |connection| {
            let transaction = connection.transaction()?;
            let batch = uuid::Uuid::new_v4().to_string();
            let mut inserted = 0;
            for (id, payload) in &inputs {
                if crate::store::enqueue(&transaction, &agent_id, id, payload, true)? {
                    inserted += 1;
                }
                transaction.execute(
                    "INSERT OR IGNORE INTO runtime_input_batches VALUES (?,?,?)",
                    params![agent_id, id, batch],
                )?;
            }
            ensure!(
                inserted == 0 || inserted == inputs.len(),
                "prompt retry mixes old and new input IDs"
            );
            transaction.commit()?;
            Ok(())
        })
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{RuntimeLimits, SliceOutcome};
    use infinity_agent_core::message::InputMessage;
    use serde_json::json;

    #[tokio::test]
    async fn admission_never_splits_a_prompt_behind_an_older_input() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let store = RuntimeStore::open(
            directory.path().join("runtime.sqlite"),
            RuntimeLimits {
                max_inputs_per_slice: 2,
                ..Default::default()
            },
        )?;
        store.create_root("root".into(), json!({})).await?;
        store
            .enqueue_text_input("root".into(), "older".into(), "older".into())
            .await?;
        let inputs = vec![
            (
                "text".into(),
                serde_json::to_value(InputMessage::user_text("root", "text"))?,
            ),
            (
                "image".into(),
                serde_json::to_value(InputMessage::user_text("root", "image caption"))?,
            ),
        ];
        store
            .enqueue_user_batch("root".into(), inputs.clone())
            .await?;
        store.enqueue_user_batch("root".into(), inputs).await?;
        let first = store
            .admit()
            .await?
            .ok_or_else(|| anyhow::anyhow!("first lease"))?;
        assert_eq!(first.inputs.len(), 1);
        assert_eq!(
            first.inputs.first().map(|input| input.0.as_str()),
            Some("older")
        );
        store
            .finish(first, json!({}), SliceOutcome::Waiting)
            .await?;
        let second = store
            .admit()
            .await?
            .ok_or_else(|| anyhow::anyhow!("second lease"))?;
        assert_eq!(
            second
                .inputs
                .iter()
                .map(|input| input.0.as_str())
                .collect::<Vec<_>>(),
            vec!["text", "image"]
        );
        store.shutdown().await
    }
}
