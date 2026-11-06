use std::{cell::RefCell, rc::Rc, sync::Arc, time::Duration};

use agent_runtime::resource::{ResourceKind, RuntimeResources};
use agent_runtime::{AgentSnapshot, Job, JobDispatchCursor, RuntimeStore, model::ToolDefinition};
use anyhow::{Context as _, Result, anyhow, ensure};
use collections::{BTreeMap, HashMap};
use futures::{FutureExt as _, StreamExt as _, channel::oneshot};
use gpui::{App, AppContext as _, AsyncApp, Context, Entity, Subscription, Task};
use language_model::{LanguageModelProviderId, LanguageModelToolResultContent};
use project::{Project, trusted_worktrees::TrustedWorktrees};
use serde_json::{Value, json};
use settings::Settings as _;

use crate::{
    AnyAgentTool, ThreadEvent, ToolCallEventStream, ToolInput, sandboxing::ThreadSandboxGrants,
};

pub type RuntimeToolResolver =
    Rc<dyn Fn(&Job, &AgentSnapshot, &mut App) -> Result<RuntimeToolContext>>;
pub type RuntimeToolEventSink =
    Rc<dyn Fn(&Job, &AgentSnapshot, &mut Option<ThreadEvent>, &mut App) -> Result<()>>;
pub type RuntimeInternalExecutor =
    Rc<dyn Fn(Job) -> futures::future::LocalBoxFuture<'static, Result<()>>>;
pub type RuntimeToolPreparation = Rc<dyn Fn(&Job, &AgentSnapshot, &mut App) -> Task<Result<()>>>;

#[derive(Clone)]
pub struct RuntimeToolContext {
    pub project: Entity<Project>,
    pub tools: BTreeMap<gpui::SharedString, Arc<dyn AnyAgentTool>>,
    pub provider_id: LanguageModelProviderId,
    grants: Rc<RefCell<ThreadSandboxGrants>>,
    trust_source: Entity<Project>,
    internal_tools: Vec<ToolDefinition>,
    internal_executor: Option<RuntimeInternalExecutor>,
}

impl RuntimeToolContext {
    pub fn new(
        project: Entity<Project>,
        tools: BTreeMap<gpui::SharedString, Arc<dyn AnyAgentTool>>,
        provider_id: LanguageModelProviderId,
        _: &App,
    ) -> Self {
        Self {
            trust_source: project.clone(),
            project,
            tools,
            provider_id,
            grants: Rc::default(),
            internal_tools: Vec::new(),
            internal_executor: None,
        }
    }

    pub fn with_trust_source(mut self, project: Entity<Project>) -> Self {
        self.trust_source = project;
        self
    }

    pub fn install_internal_tools(
        &mut self,
        definitions: Vec<ToolDefinition>,
        executor: RuntimeInternalExecutor,
    ) -> Result<()> {
        let mut names = std::collections::HashSet::new();
        for definition in &definitions {
            ensure!(
                !self.tools.contains_key(definition.name.as_str())
                    && names.insert(definition.name.clone()),
                "duplicate internal tool name: {}",
                definition.name
            );
        }
        self.internal_tools = definitions;
        self.internal_executor = Some(executor);
        Ok(())
    }

    pub fn definitions(&self, cx: &App) -> Vec<ToolDefinition> {
        let restricted = self.workspace_restriction(cx).is_some();
        let mut definitions: Vec<_> = self
            .tools
            .iter()
            .filter(|(_, tool)| {
                tool.supports_provider(&self.provider_id)
                    && (!restricted || tool.allow_in_restricted_mode())
            })
            .map(|(name, tool)| ToolDefinition {
                name: name.to_string(),
                description: tool.description().to_string(),
                parameters: tool.input_schema(),
            })
            .collect();
        definitions.extend(
            self.internal_tools
                .iter()
                .filter(|definition| !restricted || !requires_workspace_trust(&definition.name))
                .cloned(),
        );
        definitions
    }

    fn workspace_restriction(&self, cx: &App) -> Option<&'static str> {
        if TrustedWorktrees::has_restricted_worktrees(
            &self.trust_source.read(cx).worktree_store(),
            cx,
        ) {
            Some(
                "The source project is in Restricted Mode. Trust the source project in its Zezu window before retrying.",
            )
        } else if TrustedWorktrees::has_restricted_worktrees(
            &self.project.read(cx).worktree_store(),
            cx,
        ) {
            Some(
                "The isolated workspace is restricted. Reopen the source project and acquire a fresh workspace.",
            )
        } else {
            None
        }
    }
}

struct ActiveJob {
    agent_id: String,
    trust_source: Entity<Project>,
    requires_trust: bool,
    cancellation: watch::Sender<bool>,
    _task: Task<()>,
}

pub struct RuntimeToolHost {
    store: RuntimeStore,
    resolver: RuntimeToolResolver,
    sink: RuntimeToolEventSink,
    maximum: usize,
    active: HashMap<String, ActiveJob>,
    finished: async_channel::Sender<String>,
    quitting: bool,
    resources: Option<RuntimeResources>,
    preparation: Option<RuntimeToolPreparation>,
    workspace_epoch: Option<u64>,
    _driver: Task<()>,
    _quit: Subscription,
}

impl RuntimeToolHost {
    pub fn set_preparation(&mut self, preparation: RuntimeToolPreparation) {
        self.preparation = Some(preparation);
    }

    pub fn set_resources(&mut self, resources: RuntimeResources) {
        self.resources = Some(resources);
        self.workspace_epoch = None;
    }

    pub fn start(
        store: RuntimeStore,
        resolver: RuntimeToolResolver,
        sink: RuntimeToolEventSink,
        maximum: usize,
        cx: &mut App,
    ) -> Result<Entity<Self>> {
        ensure!(
            (1..=256).contains(&maximum),
            "active tool limit must be between 1 and 256"
        );
        Ok(cx.new(|cx| Self::new(store, resolver, sink, maximum, cx)))
    }

    fn new(
        store: RuntimeStore,
        resolver: RuntimeToolResolver,
        sink: RuntimeToolEventSink,
        maximum: usize,
        cx: &mut Context<Self>,
    ) -> Self {
        let (finished, completions) = async_channel::bounded(maximum);
        let driver = cx.spawn(async move |this, cx| {
            let mut dispatch_cursor=JobDispatchCursor::default();
            loop {
                let state = this.update(cx, |host, _| {
                    while let Ok(id) = completions.try_recv() {
                        host.active.remove(&id);
                    }
                    (
                        host.store.clone(),
                        host.maximum.saturating_sub(host.active.len()),
                        host.quitting,
                        host.active.iter().map(|(id,job)|(id.clone(),job.agent_id.clone())).collect::<Vec<_>>(),
                    )
                });
                let Ok((store, available, quitting, active)) = state else {
                    break;
                };
                if quitting {
                    break;
                }
                let wake=this.update(cx,|host,_| {
                    let epoch=host.resources.as_ref().map(RuntimeResources::workspace_epoch).transpose()?;
                    let changed=epoch.is_some() && epoch!=host.workspace_epoch;
                    host.workspace_epoch=epoch;
                    Ok::<_,anyhow::Error>(changed)
                });
                match wake {
                    Ok(Ok(true))=>if let Err(error)=store.notify_workspace_available().await {log::error!("could not wake workspace jobs: {error:#}");},
                    Ok(Ok(false))=>{},
                    Ok(Err(error))=> {log::error!("workspace resource state failed: {error:#}");break;},
                    Err(_)=>break,
                }
                let active_agents=active.iter().map(|(_,agent_id)|agent_id.clone()).collect();
                let enabled=cx.update(|cx|agent_settings::AgentSettings::get_global(cx).enabled(cx));
                for (id,_) in active {
                    let current = match store.job(id.clone()).await {
                        Ok(job) => store
                            .snapshot(job.agent_id)
                            .await
                            .map(|snapshot| (job.generation, snapshot)),
                        Err(error) => Err(error),
                    };
                    let revoked=this.read_with(cx,|host,cx|host.active.get(&id).is_some_and(|job|job.requires_trust && TrustedWorktrees::has_restricted_worktrees(&job.trust_source.read(cx).worktree_store(),cx))).unwrap_or(true);
                    let cancelled = !enabled || revoked || match current {
                        Ok((generation, snapshot)) => {
                            generation != snapshot.generation || snapshot.state == "cancelled"
                        }
                        Err(_) => true,
                    };
                    if cancelled {
                        if let Err(error) = this.update(cx, |host, _| {
                            if let Some(active) = host.active.get_mut(&id) {
                                if active.cancellation.send(true).is_err() {
                                    log::debug!("tool cancellation receiver detached");
                                }
                            }
                        }) {
                            log::debug!("tool host detached: {error}");
                        }
                    }
                }
                if available > 0 && enabled {
                    match dispatch_cursor.poll(&store,available,active_agents).await {
                        Ok(jobs) => {
                            for job in jobs {
                                let snapshot = match store.snapshot(job.agent_id.clone()).await {
                                    Ok(snapshot) => snapshot,
                                    Err(error) => {
                                        log::error!("tool agent lookup failed: {error:#}");
                                        continue;
                                    }
                                };
                                let preparation=match this.read_with(cx,|host,_|host.preparation.clone()) {
                                    Ok(preparation)=>preparation,
                                    Err(error)=> {log::debug!("tool host detached: {error}"); break;}
                                };
                                if let Some(preparation)=preparation {
                                    let task=cx.update(|cx|preparation(&job,&snapshot,cx));
                                    if let Err(error)=task.await {
                                        match store.dispatch_job(job.id.clone()).await {
                                            Ok(job)=>if let Err(error)=store.complete_tool_job(job,vec![json!({"type":"text","text":format!("Tool preparation failed: {error:#}")})],true,Value::Null).await {log::error!("could not persist tool preparation failure: {error:#}");},
                                            Err(error)=>log::debug!("tool preparation generation changed: {error:#}"),
                                        }
                                        continue;
                                    }
                                }
                                if let Err(error) =
                                    this.update(cx, |host, cx| host.start_job(job, snapshot, cx))
                                {
                                    log::debug!("tool host detached: {error}");
                                }
                            }
                        }
                        Err(error) => {
                            log::debug!("tool broker stopped: {error:#}");
                            break;
                        }
                    }
                }
                cx.background_executor()
                    .timer(Duration::from_millis(25))
                    .await;
            }
        });
        let quit = cx.on_app_quit(|host, _| {
            host.quitting = true;
            for active in host.active.values_mut() {
                if active.cancellation.send(true).is_err() {
                    log::debug!("tool cancellation receiver detached");
                }
            }
            let active = std::mem::take(&mut host.active);
            async move {
                futures::future::join_all(active.into_values().map(|active| active._task)).await;
            }
        });
        Self {
            store,
            resolver,
            sink,
            maximum,
            active: HashMap::default(),
            finished,
            quitting: false,
            resources: None,
            preparation: None,
            workspace_epoch: None,
            _driver: driver,
            _quit: quit,
        }
    }

    fn start_job(&mut self, job: Job, snapshot: AgentSnapshot, cx: &mut Context<Self>) {
        if self.active.contains_key(&job.id)
            || self
                .active
                .values()
                .any(|active| active.agent_id == job.agent_id)
            || self.quitting
        {
            return;
        }
        let context = match (self.resolver)(&job, &snapshot, cx) {
            Ok(context) => context,
            Err(error) => {
                log::debug!("tool context unavailable: {error:#}");
                return;
            }
        };
        let agent_id = job.agent_id.clone();
        let trust_source = context.trust_source.clone();
        let name = job
            .payload
            .get("tool")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let requires_trust = context
            .tools
            .get(name)
            .is_some_and(|tool| !tool.allow_in_restricted_mode())
            || requires_workspace_trust(name);
        let store = self.store.clone();
        let sink = self.sink.clone();
        let finished = self.finished.clone();
        let resources = self.resources.clone();
        let id = job.id.clone();
        let active_id = id.clone();
        let (cancellation, receiver) = watch::channel(false);
        let task = cx.spawn(async move |_, cx| {
            if let Err(error) = execute_job(
                store.clone(),
                job,
                snapshot,
                sink,
                context,
                resources,
                receiver,
                cx,
            )
            .await
            {
                log::error!("runtime tool {id} stopped: {error:#}");
                if let Err(error) = store.mark_job_uncertain(id.clone()).await {
                    log::debug!("tool reconciliation checkpoint unavailable: {error:#}");
                }
            }
            if finished.send(id).await.is_err() {
                log::debug!("runtime tool host detached");
            }
        });
        self.active.insert(
            active_id,
            ActiveJob {
                agent_id,
                trust_source,
                requires_trust,
                cancellation,
                _task: task,
            },
        );
    }
}

async fn execute_job(
    store: RuntimeStore,
    job: Job,
    snapshot: AgentSnapshot,
    sink: RuntimeToolEventSink,
    context: RuntimeToolContext,
    resources: Option<RuntimeResources>,
    cancellation: watch::Receiver<bool>,
    cx: &mut AsyncApp,
) -> Result<()> {
    let _permit = match resources {
        Some(resources) => Some(
            resources
                .acquire(
                    ResourceKind::Tool,
                    serde_json::to_vec(&job.payload)?
                        .len()
                        .saturating_add(store.limits().max_input_bytes),
                )
                .await?,
        ),
        None => None,
    };
    let job = store.dispatch_job(job.id).await?;
    let name = job
        .payload
        .get("tool")
        .and_then(Value::as_str)
        .context("job has no tool name")?;
    if context
        .internal_tools
        .iter()
        .any(|definition| definition.name == name)
    {
        if requires_workspace_trust(name)
            && let Some(message) = cx.update(|cx| context.workspace_restriction(cx))
        {
            store
                .complete_tool_job(
                    job,
                    vec![json!({"type":"text","text":message})],
                    true,
                    Value::Null,
                )
                .await?;
            return Ok(());
        }
        let execution = context
            .internal_executor
            .as_ref()
            .context("internal tool executor is unavailable")?(job);
        let mut cancellation = cancellation;
        let cancelled = async move {
            loop {
                if *cancellation.borrow() {
                    return;
                }
                if cancellation.changed().await.is_err() {
                    std::future::pending::<()>().await;
                }
            }
        };
        futures::pin_mut!(cancelled);
        return match futures::future::select(execution, cancelled).await {
            futures::future::Either::Left((result, _)) => result,
            futures::future::Either::Right(_) => Err(anyhow!(
                "internal tool cancelled; reconcile uncertain effects before retry"
            )),
        };
    }
    if let Some(grants) = store
        .latest_job_event(job.agent_id.clone(), "sandbox_grants".into())
        .await?
    {
        *context.grants.borrow_mut() =
            ThreadSandboxGrants::from_db(&serde_json::from_value(grants)?);
    }
    let name = job
        .payload
        .get("tool")
        .and_then(Value::as_str)
        .context("job has no tool name")?;
    let arguments = job
        .payload
        .get("arguments")
        .context("job has no arguments")?
        .clone();
    let call_id = job
        .payload
        .get("tool_call_id")
        .and_then(Value::as_str)
        .context("job has no tool call identity")?
        .to_owned();
    let prepared = cx.update(|cx| {
        let tool = context
            .tools
            .get(name)
            .context("tool is not enabled by the selected profile")?
            .clone();
        ensure!(
            tool.supports_provider(&context.provider_id),
            "tool does not support the selected provider"
        );
        if !tool.allow_in_restricted_mode()
            && let Some(message) = context.workspace_restriction(cx)
        {
            anyhow::bail!("{message}");
        }
        let (events, receiver) = ToolCallEventStream::for_runtime(
            call_id.into(),
            job.id.clone(),
            context.project.read(cx).fs().clone(),
            cancellation.clone(),
            context.grants.clone(),
        );
        let title = tool.initial_title(arguments.clone(), cx);
        events.update_fields(|update| {
            update
                .name(name)
                .title(title.to_string())
                .kind(tool.kind())
                .raw_input(arguments.clone())
                .status(agent_client_protocol::v2::schema::ToolCallStatus::InProgress)
        });
        Ok::<_, anyhow::Error>((
            tool.run(ToolInput::ready(arguments), events.clone(), cx),
            events,
            receiver,
        ))
    });
    let (task, event_stream, mut events) = match prepared {
        Ok(prepared) => prepared,
        Err(error) => {
            store
                .complete_tool_job(
                    job,
                    vec![json!({"type":"text","text":error.to_string()})],
                    true,
                    Value::Null,
                )
                .await?;
            return Ok(());
        }
    };
    let task = task.fuse();
    futures::pin_mut!(task);
    let mut pending = None;
    let output = loop {
        if let Some(event) = pending.take() {
            pending = Some(event);
            cx.update(|cx| sink(&job, &snapshot, &mut pending, cx))?;
            if pending.is_some() {
                if !matches!(
                    pending,
                    Some(ThreadEvent::ToolCallAuthorization(_) | ThreadEvent::Elicitation(_))
                ) {
                    pending = None;
                    continue;
                }
                if event_stream.was_cancelled_by_user() {
                    return Err(anyhow!("tool cancelled while waiting for approval"));
                }
                if let Some(output) = task.as_mut().now_or_never() {
                    break output;
                }
                cx.background_executor()
                    .timer(Duration::from_millis(25))
                    .await;
                continue;
            }
        }
        futures::select_biased! {
            event=events.next().fuse()=>match event {
                Some(event)=> {
                    let mut event=event?;
                    if let ThreadEvent::ToolCallAuthorization(authorization)=&mut event {
                        store.record_job_event(job.id.clone(),"approval_requested".into(),json!({"tool_call_id":authorization.tool_call.tool_call_id,"kind":format!("{:?}",authorization.kind)})).await?;
                        let (response,receiver)=oneshot::channel::<acp_thread::SelectedPermissionOutcome>();
                        let original=std::mem::replace(&mut authorization.response,response);
                        let store=store.clone(); let id=job.id.clone();
                        cx.spawn(async move |_| {
                            if let Ok(outcome)=receiver.await {
                                match store.record_job_event(id,"approval_decision".into(),json!({"option_id":outcome.option_id.0,"option_kind":format!("{:?}",outcome.option_kind)})).await {
                                    Ok(())=>if original.send(outcome).is_err() { log::debug!("tool approval requester detached"); },
                                    Err(error)=>log::error!("could not persist tool approval: {error:#}"),
                                }
                            }
                        }).detach();
                    }
                    pending=Some(event);
                },
                None=>return Err(anyhow!("tool event stream closed before its result")),
            },
            output=task=>break output,
            _=event_stream.cancelled_by_user().fuse()=> {
                let cleanup=cx.background_executor().timer(Duration::from_millis(150));
                futures::pin_mut!(cleanup);
                let cleanup_result=futures::future::select(task.as_mut(),cleanup).await;
                drop(cleanup_result);
                return Err(anyhow!("tool cancelled; reconcile any uncertain effects before retry"));
            },
        }
    };
    let (is_error, output) = match output {
        Ok(output) => (false, output),
        Err(output) => (true, output),
    };
    let content = output_content(output.llm_output, is_error);
    let grants = serde_json::to_value(context.grants.borrow().to_db())?;
    store
        .record_job_event(job.id.clone(), "sandbox_grants".into(), grants)
        .await?;
    store
        .complete_tool_job(job.clone(), content, is_error, output.raw_output.clone())
        .await?;
    let status = if is_error {
        agent_client_protocol::v2::schema::ToolCallStatus::Failed
    } else {
        agent_client_protocol::v2::schema::ToolCallStatus::Completed
    };
    let mut final_event = Some(ThreadEvent::ToolCallUpdate(
        agent_client_protocol::v2::schema::ToolCallUpdate::new(
            agent_client_protocol::v2::schema::ToolCallId::new(job.id.clone()),
        )
        .status(status)
        .raw_output(output.raw_output)
        .into(),
    ));
    if let Err(error) = cx.update(|cx| sink(&job, &snapshot, &mut final_event, cx)) {
        log::debug!("completed tool projection detached: {error:#}");
    }
    Ok(())
}

fn output_content(output: Vec<LanguageModelToolResultContent>, is_error: bool) -> Vec<Value> {
    let mut content = Vec::new();
    if is_error {
        content.push(json!({"type":"text","text":"Tool execution failed."}));
    }
    content.extend(output.into_iter().map(|part| match part {
        LanguageModelToolResultContent::Text(text) => json!({"type":"text","text":text}),
        LanguageModelToolResultContent::Image(image) => {
            json!({"type":"image","data":{"type":"base64","value":image.source},"media_type":"png"})
        }
    }));
    content
}

fn requires_workspace_trust(name: &str) -> bool {
    name.starts_with("runtime_workspace_") || matches!(name, "runtime_integrate" | "runtime_revert")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AgentTool as _;

    #[derive(Clone)]
    struct ApprovalTool(Arc<std::sync::atomic::AtomicBool>);

    impl crate::AgentTool for ApprovalTool {
        type Input = serde_json::Value;
        type Output = String;
        const NAME: &'static str = "runtime_test_approval";
        fn kind() -> agent_client_protocol::v2::schema::ToolKind {
            agent_client_protocol::v2::schema::ToolKind::Other
        }
        fn initial_title(
            &self,
            _: std::result::Result<Self::Input, Value>,
            _: &mut App,
        ) -> gpui::SharedString {
            "runtime approval".into()
        }
        fn run(
            self: Arc<Self>,
            input: ToolInput<Self::Input>,
            events: ToolCallEventStream,
            cx: &mut App,
        ) -> Task<std::result::Result<String, String>> {
            let authorization = events.authorize_always_prompt(
                "Approve durable tool",
                crate::ToolPermissionContext::new(Self::NAME, vec![]),
                cx,
            );
            cx.spawn(async move |_| {
                input.recv().await.map_err(|error| error.to_string())?;
                authorization.await.map_err(|error| error.to_string())?;
                self.0.store(true, std::sync::atomic::Ordering::SeqCst);
                Ok("approved result".into())
            })
        }
    }

    #[gpui::test]
    async fn durable_host_waits_for_attached_approval_then_commits_correlated_result(
        cx: &mut gpui::TestAppContext,
    ) {
        cx.executor().allow_parking();
        let provider = crate::tests::init_test(cx);
        let filesystem = fs::FakeFs::new(cx.executor());
        filesystem.insert_tree("/project", json!({})).await;
        let project = Project::test(filesystem, [std::path::Path::new("/project")], cx).await;
        let directory = tempfile::tempdir().expect("directory");
        let store = RuntimeStore::open(
            directory.path().join("runtime.sqlite"),
            agent_runtime::RuntimeLimits::default(),
        )
        .expect("store");
        store
            .create_root("agent".into(), json!({}))
            .await
            .expect("agent");
        store
            .enqueue_text_input("agent".into(), "input".into(), "go".into())
            .await
            .expect("input");
        let lease = store.admit().await.expect("admit").expect("lease");
        store
            .record_job(
                lease.clone(),
                "job".into(),
                json!({"tool":ApprovalTool::NAME,"arguments":{},"tool_call_id":"original-call"}),
            )
            .await
            .expect("job");
        store
            .finish(lease, json!({}), agent_runtime::SliceOutcome::Waiting)
            .await
            .expect("finish");
        let tool = ApprovalTool(Arc::new(std::sync::atomic::AtomicBool::new(false)));
        let mut tools = BTreeMap::new();
        tools.insert(
            gpui::SharedString::from(ApprovalTool::NAME),
            tool.clone().erase(),
        );
        let context = cx.update(|cx| {
            RuntimeToolContext::new(project, tools, provider.model("fake").provider_id(), cx)
        });
        let resolver: RuntimeToolResolver = Rc::new(move |_, _, _| Ok(context.clone()));
        let seen = Rc::new(std::cell::Cell::new(0));
        let sink: RuntimeToolEventSink = {
            let seen = seen.clone();
            let tool = tool.clone();
            Rc::new(move |_, _, event, _| {
                if matches!(event, Some(ThreadEvent::ToolCallAuthorization(_))) {
                    seen.set(seen.get() + 1);
                    assert!(!tool.0.load(std::sync::atomic::Ordering::SeqCst));
                    if seen.get() == 1 {
                        return Ok(());
                    }
                    if let Some(ThreadEvent::ToolCallAuthorization(authorization)) = event.take() {
                        authorization
                            .response
                            .send(acp_thread::SelectedPermissionOutcome::new(
                                agent_client_protocol::v2::schema::PermissionOptionId::new("allow"),
                                agent_client_protocol::v2::schema::PermissionOptionKind::AllowOnce,
                            ))
                            .map_err(|_| anyhow!("approval requester detached"))?;
                    }
                } else {
                    drop(event.take());
                }
                Ok(())
            })
        };
        let host = cx.update(|cx| {
            RuntimeToolHost::start(store.clone(), resolver, sink, 1, cx).expect("host")
        });
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        loop {
            if store.job("job".into()).await.expect("job").state
                == agent_runtime::JobState::Completed
            {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "tool job did not complete"
            );
            async_io::Timer::after(Duration::from_millis(2)).await;
        }
        assert!(seen.get() >= 2);
        assert!(tool.0.load(std::sync::atomic::Ordering::SeqCst));
        let result = store
            .job("job".into())
            .await
            .expect("job")
            .result
            .expect("result");
        assert_eq!(result["content"]["id"], "original-call");
        assert_eq!(
            store
                .latest_job_event("agent".into(), "approval_decision".into())
                .await
                .expect("audit")
                .expect("decision")["option_id"],
            "allow"
        );
        drop(host);
        cx.update(|_| ());
        store.shutdown().await.expect("shutdown");
    }
    #[gpui::test]
    async fn runtime_event_stream_preserves_job_identity_denial_and_cancellation(
        cx: &mut gpui::TestAppContext,
    ) {
        crate::tests::init_test(cx);
        let fs = fs::FakeFs::new(cx.background_executor.clone());
        let (mut cancellation, receiver) = watch::channel(false);
        let (stream, mut events) = ToolCallEventStream::for_runtime(
            "provider-call".into(),
            "durable-job".into(),
            fs,
            receiver,
            Rc::default(),
        );
        assert_eq!(stream.tool_use_id().to_string(), "provider-call");
        assert_eq!(stream.tool_call_id().0.as_ref(), "durable-job");
        let authorization = cx.update(|cx| {
            stream.authorize_always_prompt(
                "Authorize runtime tool",
                crate::ToolPermissionContext::new("test_tool", vec![]),
                cx,
            )
        });
        let event = events
            .next()
            .await
            .expect("authorization event")
            .expect("event");
        let ThreadEvent::ToolCallAuthorization(request) = event else {
            panic!("expected approval")
        };
        assert!(
            request
                .response
                .send(acp_thread::SelectedPermissionOutcome::new(
                    agent_client_protocol::v2::schema::PermissionOptionId::new("deny"),
                    agent_client_protocol::v2::schema::PermissionOptionKind::RejectOnce
                ))
                .is_ok()
        );
        assert!(authorization.await.is_err());
        cancellation.send(true).expect("cancel");
        stream.cancelled_by_user().await;
        assert!(stream.was_cancelled_by_user());
    }

    #[test]
    fn tool_errors_remain_visible_and_images_keep_their_media_type() {
        let result = output_content(
            vec![LanguageModelToolResultContent::Text("denied".into())],
            true,
        );
        assert_eq!(result.len(), 2);
        assert_eq!(
            result.last().and_then(|value| value.get("text")),
            Some(&json!("denied"))
        );
    }
}
