use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    thread::JoinHandle,
    time::Duration,
};

use anyhow::{Context as _, Result, anyhow};
use futures::{
    FutureExt as _, StreamExt as _,
    future::{AbortHandle, Abortable, BoxFuture, LocalBoxFuture, Shared},
    stream::FuturesUnordered,
};
use infinity_agent_core::{
    ThreadId,
    system::{AgentEvent, ThreadObserver},
};
use tokio::sync::{broadcast, watch};

use crate::{AgentSnapshot, InfinitySlice, Lease, RuntimeStore};

pub type SliceFactory =
    Arc<dyn Fn(AgentSnapshot) -> BoxFuture<'static, Result<InfinitySlice>> + Send + Sync>;

#[derive(Clone, Debug)]
pub struct RuntimeNotice {
    pub agent_id: String,
    pub error: Option<String>,
}

pub type RuntimeNoticeReceiver = broadcast::Receiver<RuntimeNotice>;

struct Owner {
    stop: watch::Sender<bool>,
    controls: async_channel::Sender<BoxFuture<'static, ()>>,
    worker: Mutex<WorkerState>,
}

type Shutdown = Shared<BoxFuture<'static, Result<(), String>>>;

struct WorkerState {
    thread: Option<JoinHandle<Result<()>>>,
    shutdown: Option<Shutdown>,
}

#[derive(Clone)]
pub struct EmbeddedRuntime {
    pub store: RuntimeStore,
    owner: Arc<Owner>,
    notices: broadcast::Sender<RuntimeNotice>,
}

struct QuietObserver;

impl ThreadObserver for QuietObserver {
    type SubscribeRequest = ();
    fn on_event(&self, _: &ThreadId<str>, _: &AgentEvent) {}
}

impl EmbeddedRuntime {
    pub fn start(store: RuntimeStore, factory: SliceFactory) -> Result<Self> {
        let (stop, receiver) = watch::channel(false);
        let (controls, control_receiver) = async_channel::bounded(16);
        let (notices, _) = broadcast::channel(256);
        let worker_notices = notices.clone();
        let worker_store = store.clone();
        let (ready, started) = std::sync::mpsc::sync_channel(1);
        let worker = std::thread::Builder::new()
            .name("agent-runtime".into())
            .spawn(move || {
                let runtime = match tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    Ok(runtime) => runtime,
                    Err(error) => {
                        let message = error.to_string();
                        if ready.send(Err(message)).is_err() {
                            log::error!("runtime owner disappeared during startup");
                        }
                        return Err(error.into());
                    }
                };
                if ready.send(Ok(())).is_err() {
                    return Err(anyhow!("runtime owner disappeared during startup"));
                }
                let local = tokio::task::LocalSet::new();
                runtime.block_on(local.run_until(drive(
                    worker_store,
                    factory,
                    receiver,
                    worker_notices,
                    control_receiver,
                )))
            })?;
        started
            .recv()
            .context("runtime worker failed before startup")?
            .map_err(|error| anyhow!(error))?;
        Ok(Self {
            store,
            owner: Arc::new(Owner {
                stop,
                controls,
                worker: Mutex::new(WorkerState {
                    thread: Some(worker),
                    shutdown: None,
                }),
            }),
            notices,
        })
    }

    pub fn subscribe(&self) -> RuntimeNoticeReceiver {
        self.notices.subscribe()
    }

    pub fn request_shutdown(&self) {
        self.owner.stop.send_replace(true);
    }

    pub async fn execute_control<T: Send + 'static>(
        &self,
        operation: BoxFuture<'static, Result<T>>,
    ) -> Result<T> {
        struct Cancel(futures::future::AbortHandle);
        impl Drop for Cancel {
            fn drop(&mut self) {
                self.0.abort();
            }
        }
        let (abort, registration) = AbortHandle::new_pair();
        let _cancel = Cancel(abort);
        let (response, receiver) = futures::channel::oneshot::channel();
        let control = async move {
            let result = Abortable::new(operation, registration)
                .await
                .map_err(|_| anyhow!("runtime control was cancelled"))
                .and_then(|result| result);
            if let Err(Err(error)) = response.send(result) {
                log::debug!("detached runtime control: {error:#}");
            }
        }
        .boxed();
        self.owner
            .controls
            .send(control)
            .await
            .map_err(|_| anyhow!("runtime control host stopped"))?;
        receiver.await.context("runtime control host stopped")?
    }

    pub async fn shutdown(&self) -> Result<()> {
        self.request_shutdown();
        let shutdown = {
            let mut state = self
                .owner
                .worker
                .lock()
                .map_err(|_| anyhow!("runtime owner poisoned"))?;
            if let Some(shutdown) = &state.shutdown {
                shutdown.clone()
            } else {
                let worker = state.thread.take().context("runtime worker is missing")?;
                let (reply, response) = futures::channel::oneshot::channel();
                let join = std::thread::Builder::new()
                    .name("agent-runtime-join".into())
                    .spawn(move || {
                        let result = worker
                            .join()
                            .map_err(|_| anyhow!("runtime worker panicked"))
                            .and_then(|result| result);
                        if let Err(Err(error)) = reply.send(result) {
                            log::error!("abandoned runtime shutdown failed: {error:#}");
                        }
                    });
                let shutdown = async move {
                    join.map_err(|error| error.to_string())?;
                    response
                        .await
                        .map_err(|error| format!("runtime join task failed: {error}"))?
                        .map_err(|error| format!("{error:#}"))
                }
                .boxed()
                .shared();
                state.shutdown = Some(shutdown.clone());
                shutdown
            }
        };
        shutdown.await.map_err(|error| anyhow!(error))
    }
}

type SliceTask = LocalBoxFuture<'static, (Lease, Result<Result<()>, futures::future::Aborted>)>;

async fn drive(
    store: RuntimeStore,
    factory: SliceFactory,
    stop: watch::Receiver<bool>,
    notices: broadcast::Sender<RuntimeNotice>,
    controls: async_channel::Receiver<BoxFuture<'static, ()>>,
) -> Result<()> {
    let outcome = drive_inner(store.clone(), factory, stop, notices, controls).await;
    let shutdown = store.shutdown().await;
    outcome.and(shutdown)
}

async fn drive_inner(
    store: RuntimeStore,
    factory: SliceFactory,
    mut stop: watch::Receiver<bool>,
    notices: broadcast::Sender<RuntimeNotice>,
    controls: async_channel::Receiver<BoxFuture<'static, ()>>,
) -> Result<()> {
    let mut active = HashMap::<String, (Lease, AbortHandle)>::new();
    let mut tasks = FuturesUnordered::<SliceTask>::new();
    let mut control_tasks = FuturesUnordered::<BoxFuture<'static, ()>>::new();
    let mut tick = tokio::time::interval(Duration::from_millis(25));
    loop {
        if *stop.borrow() {
            break;
        }
        while active.len() < store.limits().max_active_agents {
            let Some(lease) = store.admit().await? else {
                break;
            };
            let snapshot = store.snapshot(lease.agent_id.clone()).await?;
            let (abort, registration) = AbortHandle::new_pair();
            active.insert(lease.agent_id.clone(), (lease.clone(), abort));
            let factory = factory.clone();
            let store = store.clone();
            tasks.push(
                async move {
                    let execution = async {
                        let slice = factory(snapshot).await?;
                        slice.execute(store, lease.clone(), &QuietObserver).await
                    };
                    let outcome = Abortable::new(execution, registration).await;
                    (lease, outcome)
                }
                .boxed_local(),
            );
        }
        tokio::select! {
            control=controls.recv(),if control_tasks.len()<4=> { if let Ok(control)=control {control_tasks.push(control);} }
            Some(())=control_tasks.next(),if !control_tasks.is_empty()=>{}
            changed=stop.changed()=> { if changed.is_err() || *stop.borrow() { break; } }
            Some((lease,outcome))=tasks.next(),if !tasks.is_empty()=> {
                active.remove(&lease.agent_id);
                let error=match outcome { Ok(Ok(()))=>{ crate::orchestration::Orchestrator::open(store.clone()).await?.complete_idle_child(lease.agent_id.clone(),lease.generation).await?; None },Ok(Err(error))=>Some(error.to_string()),Err(_)=>None };
                if let Some(error)=&error {
                    log::error!("agent {} stopped at checkpoint: {error}",lease.agent_id);
                    let current=store.snapshot(lease.agent_id.clone()).await?;
                    if current.state=="running" && current.generation==lease.generation {
                        store.record_error(lease.clone(), error.clone()).await?;
                        store.interrupt(lease.clone()).await?;
                    }
                }
                if notices.receiver_count()>0 && notices.send(RuntimeNotice {agent_id:lease.agent_id,error}).is_err() {
                    log::debug!("runtime notice subscriber detached");
                }
            }
            _=tick.tick()=> {
                for (lease,abort) in active.values() {
                    let current=store.snapshot(lease.agent_id.clone()).await?;
                    if current.generation!=lease.generation || current.state=="cancelled" { abort.abort(); }
                }
            }
        }
    }
    for (_, abort) in active.values() {
        abort.abort();
    }
    drop(tasks);
    controls.close();
    while let Ok(control) = controls.try_recv() {
        drop(control);
    }
    drop(control_tasks);
    Ok(())
}
