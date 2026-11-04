use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use anyhow::{Result, anyhow, ensure};
use serde::{Deserialize, Serialize};
use tokio::sync::Notify;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ResourceLimits {
    pub max_agents: usize,
    pub max_active_agents: usize,
    pub max_tool_jobs: usize,
    pub max_workspaces: usize,
    pub max_queued_jobs: usize,
    pub max_memory_bytes: usize,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_agents: 75_000,
            max_active_agents: 16,
            max_tool_jobs: 8,
            max_workspaces: 4,
            max_queued_jobs: 256,
            max_memory_bytes: 512 * 1024 * 1024,
        }
    }
}

impl ResourceLimits {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.max_agents > 0
                && self.max_active_agents > 0
                && self.max_tool_jobs > 0
                && self.max_workspaces > 0
                && self.max_queued_jobs > 0
                && self.max_memory_bytes > 0,
            "all runtime resource limits must be positive"
        );
        ensure!(
            self.max_active_agents <= self.max_agents,
            "active agents cannot exceed the logical agent limit"
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResourceKind {
    Inference,
    Tool,
    Workspace,
    Queued,
    Memory,
}

impl ResourceKind {
    fn index(self) -> usize {
        match self {
            Self::Inference => 0,
            Self::Tool => 1,
            Self::Workspace => 2,
            Self::Queued => 3,
            Self::Memory => 4,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ResourceSnapshot {
    pub active_inference: usize,
    pub active_tools: usize,
    pub active_workspaces: usize,
    pub queued_jobs: usize,
    pub retained_bytes: usize,
}

#[derive(Default)]
struct State {
    counts: [usize; 5],
    bytes: usize,
    workspace_epoch: u64,
    provider_backoff: VecDeque<(String, Instant)>,
}

struct Inner {
    limits: ResourceLimits,
    state: Mutex<State>,
    changed: Notify,
}

#[derive(Clone)]
pub struct RuntimeResources(Arc<Inner>);

pub struct ResourcePermit {
    resources: RuntimeResources,
    kind: ResourceKind,
    bytes: usize,
}

impl RuntimeResources {
    pub fn new(limits: ResourceLimits) -> Result<Self> {
        limits.validate()?;
        Ok(Self(Arc::new(Inner {
            limits,
            state: Mutex::new(State::default()),
            changed: Notify::new(),
        })))
    }

    pub fn limits(&self) -> &ResourceLimits {
        &self.0.limits
    }

    fn reserve(&self, kind: ResourceKind, bytes: usize) -> Result<Option<ResourcePermit>> {
        let limits = &self.0.limits;
        ensure!(
            bytes <= limits.max_memory_bytes,
            "request exceeds the runtime memory budget"
        );
        let maximum = match kind {
            ResourceKind::Inference => limits.max_active_agents,
            ResourceKind::Tool => limits.max_tool_jobs,
            ResourceKind::Workspace => limits.max_workspaces,
            ResourceKind::Queued => limits.max_queued_jobs,
            ResourceKind::Memory => usize::MAX,
        };
        let mut state = self
            .0
            .state
            .lock()
            .map_err(|_| anyhow!("runtime resource state poisoned"))?;
        let index = kind.index();
        if state.counts[index] >= maximum || state.bytes > limits.max_memory_bytes - bytes {
            return Ok(None);
        }
        state.counts[index] += 1;
        state.bytes += bytes;
        Ok(Some(ResourcePermit {
            resources: self.clone(),
            kind,
            bytes,
        }))
    }

    pub fn try_reserve(&self, kind: ResourceKind, bytes: usize) -> Result<Option<ResourcePermit>> {
        self.reserve(kind, bytes)
    }

    pub fn try_acquire(&self, kind: ResourceKind, bytes: usize) -> Result<ResourcePermit> {
        self.reserve(kind, bytes)?
            .ok_or_else(|| anyhow!("runtime resource capacity is exhausted"))
    }

    pub async fn acquire(&self, kind: ResourceKind, bytes: usize) -> Result<ResourcePermit> {
        loop {
            let changed = self.0.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            if let Some(permit) = self.reserve(kind, bytes)? {
                return Ok(permit);
            }
            changed.await;
        }
    }

    pub fn workspace_epoch(&self) -> Result<u64> {
        Ok(self
            .0
            .state
            .lock()
            .map_err(|_| anyhow!("runtime resource state poisoned"))?
            .workspace_epoch)
    }

    pub fn snapshot(&self) -> Result<ResourceSnapshot> {
        let state = self
            .0
            .state
            .lock()
            .map_err(|_| anyhow!("runtime resource state poisoned"))?;
        Ok(ResourceSnapshot {
            active_inference: state.counts[0],
            active_tools: state.counts[1],
            active_workspaces: state.counts[2],
            queued_jobs: state.counts[3],
            retained_bytes: state.bytes,
        })
    }

    pub fn defer_provider(&self, model_id: &str, delay: Duration) -> Result<()> {
        let deadline = Instant::now()
            .checked_add(delay)
            .ok_or_else(|| anyhow!("provider backoff deadline overflow"))?;
        let mut state = self
            .0
            .state
            .lock()
            .map_err(|_| anyhow!("runtime resource state poisoned"))?;
        if let Some((_, current)) = state
            .provider_backoff
            .iter_mut()
            .find(|(id, _)| id == model_id)
        {
            *current = (*current).max(deadline);
        } else {
            state
                .provider_backoff
                .retain(|(_, deadline)| *deadline > Instant::now());
            if state.provider_backoff.len() >= 1024 {
                state.provider_backoff.pop_front();
            }
            state
                .provider_backoff
                .push_back((model_id.to_owned(), deadline));
        }
        Ok(())
    }

    pub async fn wait_for_provider(&self, model_id: &str) -> Result<()> {
        loop {
            let deadline = self
                .0
                .state
                .lock()
                .map_err(|_| anyhow!("runtime resource state poisoned"))?
                .provider_backoff
                .iter()
                .find(|(id, _)| id == model_id)
                .map(|(_, deadline)| *deadline);
            match deadline.and_then(|deadline| deadline.checked_duration_since(Instant::now())) {
                Some(duration) => tokio::time::sleep(duration).await,
                None => return Ok(()),
            }
        }
    }
}

impl Drop for ResourcePermit {
    fn drop(&mut self) {
        match self.resources.0.state.lock() {
            Ok(mut state) => {
                state.counts[self.kind.index()] = state.counts[self.kind.index()].saturating_sub(1);
                state.bytes = state.bytes.saturating_sub(self.bytes);
                if self.kind == ResourceKind::Workspace {
                    state.workspace_epoch = state.workspace_epoch.wrapping_add(1);
                }
            }
            Err(error) => log::error!("cannot release poisoned runtime resource state: {error}"),
        }
        self.resources.0.changed.notify_waiters();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::FutureExt as _;

    #[tokio::test]
    async fn permits_reserve_count_and_memory_together_and_release_on_cancellation() -> Result<()> {
        let resources = RuntimeResources::new(ResourceLimits {
            max_active_agents: 1,
            max_memory_bytes: 100,
            ..Default::default()
        })?;
        let inference = resources.acquire(ResourceKind::Inference, 70).await?;
        assert!(resources.try_acquire(ResourceKind::Inference, 0).is_err());
        assert!(resources.try_acquire(ResourceKind::Tool, 40).is_err());
        let waiting = resources.acquire(ResourceKind::Tool, 40);
        futures::pin_mut!(waiting);
        assert!(waiting.as_mut().now_or_never().is_none());
        assert_eq!(resources.snapshot()?.active_tools, 0);
        drop(inference);
        let tool = waiting.await?;
        assert_eq!(resources.snapshot()?.retained_bytes, 40);
        drop(tool);
        assert_eq!(resources.snapshot()?, ResourceSnapshot::default());
        assert!(resources.acquire(ResourceKind::Memory, 101).await.is_err());
        Ok(())
    }

    #[tokio::test]
    async fn independent_resource_limits_and_provider_backoff_are_enforced() -> Result<()> {
        let resources = RuntimeResources::new(ResourceLimits {
            max_workspaces: 1,
            max_tool_jobs: 1,
            max_queued_jobs: 1,
            ..Default::default()
        })?;
        let workspace = resources.try_acquire(ResourceKind::Workspace, 0)?;
        let tool = resources.try_acquire(ResourceKind::Tool, 0)?;
        let queued = resources.try_acquire(ResourceKind::Queued, 0)?;
        assert!(resources.try_acquire(ResourceKind::Workspace, 0).is_err());
        assert!(resources.try_acquire(ResourceKind::Tool, 0).is_err());
        assert!(resources.try_acquire(ResourceKind::Queued, 0).is_err());
        resources.defer_provider("provider/model", Duration::from_millis(20))?;
        let ready = resources.wait_for_provider("provider/model");
        futures::pin_mut!(ready);
        assert!(ready.as_mut().now_or_never().is_none());
        resources.wait_for_provider("other/model").await?;
        ready.await?;
        let epoch = resources.workspace_epoch()?;
        drop((tool, queued));
        assert_eq!(resources.workspace_epoch()?, epoch);
        drop(workspace);
        assert_ne!(resources.workspace_epoch()?, epoch);
        assert_eq!(resources.snapshot()?, ResourceSnapshot::default());
        Ok(())
    }
}
