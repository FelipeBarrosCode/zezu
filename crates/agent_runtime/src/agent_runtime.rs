pub mod compaction;
pub mod history_budget;
mod infinity;
mod input_batches;
mod jobs;
pub mod model;
pub mod orchestration;
pub mod projection;
#[cfg(test)]
mod provider_engine_tests;
pub mod provider_state;
pub mod resource;
mod runtime;
mod store;
#[cfg(test)]
mod tests;

pub use infinity::{InfinitySlice, SqliteInfinityStore};
pub use projection::{FleetSummary, HistoryReference, TranscriptEntry, TranscriptPage};
pub use runtime::{EmbeddedRuntime, RuntimeNotice, RuntimeNoticeReceiver, SliceFactory};
pub use store::{
    AgentPage, AgentSnapshot, Job, JobRecovery, JobState, Lease, RuntimeStore, SliceOutcome,
};

use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RuntimeLimits {
    pub max_agents_per_run: usize,
    pub max_active_agents: usize,
    pub command_queue_capacity: usize,
    pub max_input_bytes: usize,
    pub max_inputs_per_slice: usize,
}

impl Default for RuntimeLimits {
    fn default() -> Self {
        Self {
            max_agents_per_run: 75_000,
            max_active_agents: 16,
            command_queue_capacity: 256,
            max_input_bytes: 1024 * 1024,
            max_inputs_per_slice: 32,
        }
    }
}

impl RuntimeLimits {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.max_agents_per_run > 0, "agent limit must be positive");
        ensure!(self.max_active_agents > 0, "active limit must be positive");
        ensure!(
            self.command_queue_capacity > 0,
            "queue capacity must be positive"
        );
        ensure!(self.max_input_bytes > 0, "input limit must be positive");
        ensure!(
            self.max_inputs_per_slice > 0,
            "slice input limit must be positive"
        );
        Ok(())
    }
}

pub use jobs::{JobDispatchCursor, UncertainJob};
