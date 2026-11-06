use std::{
    any::Any,
    cell::{Cell, RefCell},
    rc::Rc,
    sync::Arc,
    time::Duration,
};

use acp_thread::{
    AcpThread, AgentConnection, AgentModelId, AgentModelSelector, ClientUserMessageId,
};
use action_log::ActionLog;
use agent_client_protocol::{v1::schema as acp_v1, v2::schema as acp_v2};
use agent_runtime::{RuntimeStore, TranscriptEntry};
use agent_settings::{AgentSettings, language_model_to_selection};
use anyhow::{Context as _, Result, ensure};
use collections::{HashMap, HashSet};
use fs::Fs;
use futures::FutureExt as _;
use gpui::{
    App, AppContext as _, AsyncApp, Entity, Global, SharedString, Task, TaskExt as _, WeakEntity,
};
use language_model::{
    LanguageModelId, LanguageModelProviderId, LanguageModelRegistry, MessageContent, SelectedModel,
};
use project::{AgentId, Project};
use serde_json::{Value, json};
use settings::Settings as _;
use util::ResultExt as _;
use util::path_list::PathList;

use crate::{
    NativeAgentConnection, NativeAgentModelSelector, NativeRuntime, NativeRuntimeConfig,
    ThreadEvent, UserMessage, UserMessageContent,
    runtime_context::prepare_runtime_context,
    runtime_mcp::RuntimeMcpCatalog,
    runtime_orchestration::{
        RuntimeOrchestrationServices, append_runtime_instructions, prepare_runtime_orchestration,
        with_runtime_orchestration,
    },
    runtime_project::{runtime_project_tools, runtime_project_tools_with_log},
    runtime_session_controls::{RuntimeSessionConfigOptions, RuntimeSessionModes},
    runtime_tools::{RuntimeToolContext, RuntimeToolPreparation, RuntimeToolResolver},
    runtime_workspaces::RuntimeWorkspaceProjects,
};

const SESSION_PREFIX: &str = "runtime:";
const PAGE_SIZE: usize = 100;

#[derive(Clone)]
pub struct NativeRuntimeConnection {
    legacy: NativeAgentConnection,
    host: Entity<NativeRuntime>,
    fs: Arc<dyn Fs>,
    views: Rc<RefCell<HashMap<String, VisibleProjection>>>,
    configurations: Rc<RefCell<HashMap<String, NativeRuntimeConfig>>>,
    config_controls: Rc<RefCell<HashMap<String, std::rc::Weak<RuntimeSessionConfigOptions>>>>,
    updating: Rc<RefCell<HashSet<String>>>,
    skills: Rc<RefCell<HashMap<String, Arc<Vec<crate::Skill>>>>>,
}

struct VisibleProjection {
    view: WeakEntity<AcpThread>,
    position: Rc<Cell<i64>>,
    usage: Rc<RefCell<Option<(String, acp_thread::TokenUsage)>>>,
    _observer: Task<()>,
}

#[derive(Default)]
struct RuntimeServicesCache(
    HashMap<
        std::path::PathBuf,
        futures::future::Shared<Task<Result<RuntimeOrchestrationServices, Arc<anyhow::Error>>>>,
    >,
);
impl Global for RuntimeServicesCache {}

impl NativeRuntimeConnection {
    pub fn new(
        legacy: NativeAgentConnection,
        host: Entity<NativeRuntime>,
        fs: Arc<dyn Fs>,
    ) -> Self {
        Self {
            legacy,
            host,
            fs,
            views: Rc::default(),
            configurations: Rc::default(),
            config_controls: Rc::default(),
            updating: Rc::default(),
            skills: Rc::default(),
        }
    }

    pub fn legacy(&self) -> &NativeAgentConnection {
        &self.legacy
    }

    pub fn thread_model(
        &self,
        session_id: &acp_v2::SessionId,
        cx: &App,
    ) -> Option<crate::ThreadModel> {
        let configurations = self.configurations.borrow();
        let configuration = configurations.get(runtime_id(session_id)?)?;
        Some(match configured_model(configuration, cx) {
            Ok(model) => crate::ThreadModel::Ready(model),
            Err(_) => match &configuration.model {
                Some(selection) => crate::ThreadModel::Unresolved(SelectedModel {
                    provider: LanguageModelProviderId(selection.provider.0.clone().into()),
                    model: LanguageModelId(selection.model.clone().into()),
                }),
                None => crate::ThreadModel::Unset,
            },
        })
    }

    pub fn validate_prompt_model(
        &self,
        session_id: &acp_v2::SessionId,
        cx: &mut App,
    ) -> Task<Result<()>> {
        let Some(id) = runtime_id(session_id) else {
            return Task::ready(Ok(()));
        };
        if !self.configurations.borrow().contains_key(id) {
            return Task::ready(Ok(()));
        }
        let id = id.to_owned();
        let store = self.store(cx);
        cx.spawn(async move |cx| {
            let configuration = serde_json::from_value(store?.metadata(id).await?)?;
            cx.update(|cx| {
                configured_model(&configuration, cx)
                    .map(|_| ())
                    .map_err(|error| error.context(crate::NoModelConfiguredError))
            })
        })
    }

    pub fn available_skills(
        &self,
        session: &acp_v2::SessionId,
        cx: &App,
    ) -> Vec<crate::NativeAvailableSkill> {
        let Some(id) = runtime_id(session) else {
            return self.legacy.available_skills(session, cx);
        };
        self.skills
            .borrow()
            .get(id)
            .map(|skills| {
                skills
                    .iter()
                    .map(crate::NativeAvailableSkill::from)
                    .collect()
            })
            .unwrap_or_default()
    }

    fn store(&self, cx: &App) -> Result<RuntimeStore> {
        self.host
            .read(cx)
            .active()
            .map(|runtime| runtime.store)
            .context("native runtime is not started")
    }

    fn register_project(
        &self,
        id: String,
        project: Entity<Project>,
        configuration: NativeRuntimeConfig,
        cx: &mut App,
    ) -> Task<Result<(NativeRuntimeConfig, Option<String>)>> {
        let fs = self.fs.clone();
        let host = self.host.clone();
        let configurations = self.configurations.clone();
        let skills = self.skills.clone();
        let settings = AgentSettings::get_global(cx);
        let profile = configuration
            .profile
            .clone()
            .unwrap_or_else(|| settings.default_profile.clone());
        let selected = configuration
            .model
            .as_ref()
            .or_else(|| settings.profiles.get(&profile)?.default_model.as_ref());
        let model = match selected {
            Some(selection) => crate::native_runtime::resolve_model_id(
                &format!("{}/{}", selection.provider.0, selection.model),
                cx,
            )
            .ok(),
            None => LanguageModelRegistry::read_global(cx).default_model(),
        };
        let selected = selected.or_else(|| match &model {
            Some(model) => settings.default_model.as_ref().filter(|selection| {
                selection.provider.0 == model.provider_id().0.as_ref()
                    && selection.model == model.id().0.as_ref()
            }),
            None => cx
                .global::<settings::SettingsStore>()
                .raw_user_settings()?
                .content
                .agent
                .as_ref()?
                .default_model
                .as_ref(),
        });
        // A provider can populate its catalog after the conversation opens. Keep the
        // requested selection so the picker can recover without changing models implicitly.
        let pinned_model = model
            .as_ref()
            .map(|model| language_model_to_selection(model, selected))
            .or_else(|| selected.cloned());
        let provider = pinned_model
            .as_ref()
            .map(|selection| LanguageModelProviderId(selection.provider.0.clone().into()));
        if model.is_none()
            && let Some(provider) = provider.as_ref()
            && let Some(provider) = LanguageModelRegistry::read_global(cx).provider(provider)
        {
            provider.refresh_models(cx);
        }
        let registry = match RuntimeWorkspaceProjects::new(
            project.clone(),
            AgentSettings::get_global(cx)
                .runtime
                .max_workspaces
                .unwrap_or(4),
            cx,
        ) {
            Ok(registry) => registry,
            Err(error) => return Task::ready(Err(error)),
        };
        let catalog_job = agent_runtime::Job {
            id: "catalog".into(),
            agent_id: id.clone(),
            generation: 0,
            state: agent_runtime::JobState::Queued,
            payload: json!({}),
            result: None,
        };
        let tool_names = if let Some(provider) = &provider {
            let actions = registry.read(cx).code_actions_for(&catalog_job);
            let catalog = RuntimeMcpCatalog::new(project.clone(), cx);
            let mut tools = match runtime_project_tools(
                project.clone(),
                &id,
                provider,
                &profile,
                actions,
                cx,
            ) {
                Ok(tools) => tools,
                Err(error) => return Task::ready(Err(error)),
            };
            if let Err(error) = catalog.merge_into(&mut tools, &profile, provider, cx) {
                return Task::ready(Err(error));
            }
            tools.keys().cloned().collect()
        } else {
            Vec::new()
        };
        let context = prepare_runtime_context(
            project.clone(),
            fs,
            Some(profile),
            tool_names,
            model.map(|model| model.name().0.to_string()),
            cx,
        );
        let store = match self.store(cx) {
            Ok(store) => store,
            Err(error) => return Task::ready(Err(error)),
        };
        let resources = match host.read(cx).resources() {
            Some(resources) => resources,
            None => {
                return Task::ready(Err(anyhow::anyhow!("native resources are not initialized")));
            }
        };
        let repository_path = project
            .read(cx)
            .visible_worktrees(cx)
            .next()
            .map(|worktree| worktree.read(cx).abs_path().to_path_buf());
        let repository = cx.background_spawn(async move {
            let Some(path) = repository_path else {
                return Ok(None);
            };
            let output = std::process::Command::new("git")
                .arg("-C")
                .arg(path)
                .args(["rev-parse", "--show-toplevel"])
                .env_remove("GIT_DIR")
                .env_remove("GIT_WORK_TREE")
                .stdin(std::process::Stdio::null())
                .output()?;
            if !output.status.success() {
                return Ok(None);
            }
            Ok::<_, anyhow::Error>(Some(std::path::PathBuf::from(
                String::from_utf8(output.stdout)?.trim(),
            )))
        });
        let views = self.views.clone();
        let loader_project = project.clone();
        let loader_fs = self.fs.clone();
        let loader_skills = self.skills.clone();
        cx.spawn(async move |cx| {
            let prepared = context.await?;
            let available_skills = Arc::new(prepared.skills.clone());
            let mut configuration = prepared.configuration(pinned_model);
            let services = if let Some(repository) = repository.await? {
                let preparation = cx.update(|cx| {
                    if !cx.has_global::<RuntimeServicesCache>() {
                        cx.set_global(RuntimeServicesCache::default());
                    }
                    if let Some(preparation) =
                        cx.global::<RuntimeServicesCache>().0.get(&repository)
                    {
                        return preparation.clone();
                    }
                    let pool_directory = workspace_pool_directory(&repository);
                    let task = prepare_runtime_orchestration(
                        store.clone(),
                        repository.clone(),
                        pool_directory,
                        resources,
                        project.clone(),
                        cx,
                    );
                    let preparation = cx
                        .spawn(async move |_| task.await.map_err(Arc::new))
                        .shared();
                    cx.global_mut::<RuntimeServicesCache>()
                        .0
                        .insert(repository.clone(), preparation.clone());
                    preparation
                });
                let result = preparation.clone().await;
                if !result
                    .as_ref()
                    .is_ok_and(|services| services.workspace.is_some())
                {
                    cx.update(|cx| {
                        let cache = cx.global_mut::<RuntimeServicesCache>();
                        if cache
                            .0
                            .get(&repository)
                            .is_some_and(|current| current.ptr_eq(&preparation))
                        {
                            cache.0.remove(&repository);
                        }
                    });
                }
                result.map_err(|error| anyhow::anyhow!("{error:#}"))?
            } else {
                RuntimeOrchestrationServices::metadata_only(store.clone())
            };
            let workspace_error = services.workspace_error.clone();
            append_runtime_instructions(&mut configuration.system_prompt, &services);
            let profile = prepared.profile_id;
            let runtime_profile = profile.clone();
            let original_project = project.clone();
            let resolver_registry = registry.clone();
            let has_provider = provider.is_some();
            let resolver: RuntimeToolResolver = Rc::new(move |job, _snapshot, cx| {
                let requested_profile = job
                    .payload
                    .get("runtime_profile_id")
                    .and_then(Value::as_str)
                    .map(|profile| agent_settings::AgentProfileId(profile.to_owned().into()))
                    .unwrap_or_else(|| profile.clone());
                let (profile, _) = crate::Thread::profile_for_restricted_workspace(
                    requested_profile,
                    &original_project,
                    cx,
                );
                let provider = job
                    .payload
                    .get("runtime_model_id")
                    .and_then(Value::as_str)
                    .and_then(|model| model.split_once('/'))
                    .map(|(provider, _)| {
                        language_model::LanguageModelProviderId(provider.to_owned().into())
                    })
                    .or_else(|| provider.clone())
                    .context("select a model before running agent tools")?;
                let project = resolver_registry.read(cx).project_for(job, cx)?;
                let actions = resolver_registry.read(cx).code_actions_for(job);
                let action_log = resolver_registry.read(cx).action_log_for(job);
                let mut tools = runtime_project_tools_with_log(
                    project.clone(),
                    &job.agent_id,
                    &provider,
                    &profile,
                    actions.clone(),
                    action_log,
                    cx,
                )?;
                if resolver_registry.read(cx).has_workspace(job) {
                    resolver_registry
                        .read(cx)
                        .mcp_catalog_for(job)
                        .merge_into(&mut tools, &profile, &provider, cx)?;
                }
                Ok(
                    RuntimeToolContext::new(project.clone(), tools, provider, cx)
                        .with_trust_source(original_project.clone()),
                )
            });
            let sink = Rc::new(
                move |job: &agent_runtime::Job,
                      snapshot: &agent_runtime::AgentSnapshot,
                      event: &mut Option<ThreadEvent>,
                      cx: &mut App| {
                    let view = {
                        let views = views.borrow();
                        views
                            .get(&job.agent_id)
                            .or_else(|| views.get(&snapshot.root_id))
                            .map(|projection| projection.view.clone())
                    };
                    if let Some(view) = view {
                        project_runtime_tool_event(&view, event, cx)?;
                    }
                    Ok(())
                },
            );
            let loader_services = services.clone();
            let resolver = with_runtime_orchestration(resolver, runtime_profile, services, sink);
            let definitions = cx.update(|cx| {
                if !has_provider {
                    return Ok(Vec::new());
                }
                let job = agent_runtime::Job {
                    id: "catalog".into(),
                    agent_id: id.clone(),
                    generation: 0,
                    state: agent_runtime::JobState::Queued,
                    payload: json!({}),
                    result: None,
                };
                let snapshot = agent_runtime::AgentSnapshot {
                    id: id.clone(),
                    root_id: id.clone(),
                    parent_id: None,
                    state: "waiting".into(),
                    generation: 0,
                };
                resolver(&job, &snapshot, cx).map(|context| context.definitions(cx))
            })?;
            host.update(cx, |host, cx| {
                host.bind_run(id.clone(), resolver.clone(), definitions, cx)
            })?;
            let context_loader: crate::native_runtime::RuntimeContextLoader = Rc::new(move |configuration, snapshot, cx| {
                let model = match configured_model(&configuration, cx) { Ok(model) => model, Err(error) => return Task::ready(Err(error)) };
                let job = agent_runtime::Job { id: "context".into(), agent_id: snapshot.id.clone(), generation: snapshot.generation, state: agent_runtime::JobState::Queued,
                    payload: json!({"runtime_root_id":snapshot.root_id,"runtime_model_id":format!("{}/{}", model.provider_id().0, model.id().0),"runtime_profile_id":configuration.profile.as_ref().map(|profile|profile.as_str())}), result: None };
                let definitions = match resolver(&job, &snapshot, cx) { Ok(context) => context.definitions(cx), Err(error) => return Task::ready(Err(error)) };
                let prepared = prepare_runtime_context(loader_project.clone(), loader_fs.clone(), configuration.profile.clone(), definitions.into_iter().map(|definition| definition.name.into()).collect(), Some(model.name().0.to_string()), cx);
                let services = loader_services.clone();
                let skills = loader_skills.clone();
                cx.spawn(async move |_| {
                    let prepared = prepared.await?;
                    if snapshot.parent_id.is_none() { skills.borrow_mut().insert(snapshot.id, Arc::new(prepared.skills.clone())); }
                    let mut configuration = prepared.configuration(configuration.model);
                    append_runtime_instructions(&mut configuration.system_prompt, &services);
                    Ok(configuration)
                })
            });
            host.update(cx, |host, _| host.bind_run_context(id.clone(), context_loader))?;
            let preparation: RuntimeToolPreparation = Rc::new(move |job, snapshot, cx| {
                registry.update(cx, |registry, cx| {
                    registry.prepare(store.clone(), job, snapshot, cx)
                })
            });
            host.update(cx, |host, _| host.bind_run_preparation(id, preparation))?;
            configurations
                .borrow_mut()
                .insert(catalog_job.agent_id.clone(), configuration.clone());
            skills
                .borrow_mut()
                .insert(catalog_job.agent_id, available_skills);
            Ok((configuration, workspace_error))
        })
    }

    fn update_configuration(
        &self,
        id: String,
        configuration: NativeRuntimeConfig,
        cx: &mut App,
    ) -> Task<Result<()>> {
        let store = match self.store(cx) {
            Ok(store) => store,
            Err(error) => return Task::ready(Err(error)),
        };
        let project = self.views.borrow().get(&id).and_then(|projection| {
            projection
                .view
                .read_with(cx, |thread, _| thread.project().clone())
                .ok()
        });
        let Some(project) = project else {
            return Task::ready(Err(anyhow::anyhow!(
                "open the conversation before changing its configuration"
            )));
        };
        if !self.updating.borrow_mut().insert(id.clone()) {
            return Task::ready(Err(anyhow::anyhow!(
                "agent configuration is already updating"
            )));
        }
        struct Updating {
            id: String,
            updates: Rc<RefCell<HashSet<String>>>,
        }
        impl Drop for Updating {
            fn drop(&mut self) {
                self.updates.borrow_mut().remove(&self.id);
            }
        }
        let updating = Updating {
            id: id.clone(),
            updates: self.updating.clone(),
        };
        let connection = self.clone();
        let old_configuration = self
            .configurations
            .borrow()
            .get(&id)
            .cloned()
            .unwrap_or_default();
        let changed_model = (configuration.model != old_configuration.model)
            .then(|| configuration.model.clone())
            .flatten();
        let changed_profile = (configuration.profile != old_configuration.profile)
            .then(|| configuration.profile.clone())
            .flatten();
        let (response, receiver) = futures::channel::oneshot::channel();
        cx.spawn(async move |cx| {
            let _updating = updating;
            let result: Result<()> = async {
                let lease = store.begin_configuration(id.clone()).await?;
                let prepared = cx.update(|cx| connection.register_project(id.clone(), project.clone(), configuration, cx)).await;
                let result = match prepared {
                    Ok((configuration, workspace_error)) => match serde_json::to_value(configuration) {
                        Ok(metadata) => store.commit_configuration(lease.clone(), metadata).await.map(|()| workspace_error),
                        Err(error) => Err(error.into()),
                    },
                    Err(error) => Err(error),
                };
                let workspace_error = match result {
                    Ok(workspace_error) => workspace_error,
                    Err(error) => {
                        cx.update(|cx| connection.register_project(id.clone(), project, old_configuration, cx)).await
                            .context("failed to restore the previous runtime binding; restart to recover the paused agent")?;
                        store.abort_configuration(lease).await?;
                        return Err(error);
                    }
                };
                cx.update(|cx| {
                    let view = connection.views.borrow().get(&id).map(|projection| projection.view.clone());
                    if let Some(view) = view
                        && let Err(error) = view.update(cx, |thread, cx| update_workspace_notice(thread, workspace_error, cx))
                    {
                        log::debug!("workspace notice view detached: {error:#}");
                    }
                    if let Some(control) = connection.config_controls.borrow().get(&id).and_then(std::rc::Weak::upgrade) {
                        let configuration = connection.configurations.borrow().get(&id).cloned().unwrap_or_default();
                        control.replace_options(runtime_model_options(&configuration, cx).unwrap_or_default());
                    }
                    connection.legacy.0.update(cx, |agent, _| agent.models.notify_model_selection_changed());
                    if changed_model.is_some() || changed_profile.is_some() {
                        settings::update_settings_file(connection.fs.clone(), cx, move |settings, _| {
                            let agent = settings.agent.get_or_insert_default();
                            if let Some(model) = changed_model { agent.set_model(model); }
                            if let Some(profile) = changed_profile { agent.set_profile(profile.0); }
                        });
                    }
                    Ok::<_, anyhow::Error>(())
                })?;
                Ok(())
            }.await;
            if response.send(result).is_err() { log::debug!("runtime configuration observer detached"); }
        }).detach();
        cx.spawn(async move |_| {
            receiver
                .await
                .context("runtime configuration owner stopped")?
        })
    }

    fn projection(
        self: Rc<Self>,
        id: String,
        project: Entity<Project>,
        work_dirs: PathList,
        title: Option<SharedString>,
        position: i64,
        cx: &mut App,
    ) -> Entity<AcpThread> {
        let (_, capabilities) = watch::channel(
            acp_v2::PromptCapabilities::new()
                .image(acp_v2::PromptImageCapabilities::new())
                .embedded_context(acp_v2::PromptEmbeddedContextCapabilities::new()),
        );
        let action_log = cx.new(|_| ActionLog::new(project.clone()));
        let session_id = acp_v2::SessionId::new(format!("{SESSION_PREFIX}{id}"));
        let connection = self.clone() as Rc<dyn AgentConnection>;
        let thread = cx.new(|cx| {
            AcpThread::new(
                None,
                title,
                Some(work_dirs),
                connection,
                project,
                action_log,
                session_id,
                capabilities,
                cx,
            )
        });
        self.views
            .borrow_mut()
            .retain(|_, projection| projection.view.upgrade().is_some());
        let cursor = Rc::new(Cell::new(position));
        let projected_usage = Rc::new(RefCell::new(None));
        if let Some(runtime) = self.host.read(cx).active() {
            let mut notices = runtime.subscribe();
            let store = runtime.store;
            let view = thread.downgrade();
            let position = cursor.clone();
            let usage = projected_usage.clone();
            let agent_id = id.clone();
            let observer = cx.spawn(async move |cx| {
                loop {
                    match flush_projection(&store, &agent_id, &view, &position, &usage, cx).await {
                        Ok(true) => {}
                        Ok(false) => break,
                        Err(error) => {
                            if view.upgrade().is_none() {
                                break;
                            }
                            log::warn!("could not update runtime transcript: {error:#}");
                            view.update(cx, |thread, cx| {
                                projection_notice(
                                    thread,
                                    "Could not read saved messages",
                                    "Reopen the conversation to retry.",
                                    cx,
                                );
                            })
                            .log_err();
                        }
                    }
                    loop {
                        let notification = notices.recv().await;
                        if notification.is_err() && notices.is_closed() {
                            return;
                        }
                        let mut relevant =
                            notification.map_or(true, |notice| notice.agent_id == agent_id);
                        while let Ok(notice) = notices.try_recv() {
                            relevant |= notice.agent_id == agent_id;
                        }
                        if relevant {
                            break;
                        }
                        futures_lite::future::yield_now().await;
                    }
                }
            });
            self.views.borrow_mut().insert(
                id.clone(),
                VisibleProjection {
                    view: thread.downgrade(),
                    position: cursor,
                    usage: projected_usage,
                    _observer: observer,
                },
            );
        }
        self.host
            .update(cx, |host, _| host.bind_view(id, thread.downgrade()));
        thread
    }

    pub fn summary(
        &self,
        session_id: &acp_v2::SessionId,
        cx: &mut App,
    ) -> Task<Result<SharedString>> {
        let Some(id) = runtime_id(session_id).map(str::to_owned) else {
            return Task::ready(Err(anyhow::anyhow!("not a runtime session")));
        };
        let store = match self.store(cx) {
            Ok(store) => store,
            Err(error) => return Task::ready(Err(error)),
        };
        cx.spawn(async move |_| {
            let page = store.transcript_page(id, None, PAGE_SIZE).await?;
            Ok(page
                .entries
                .iter()
                .filter_map(|entry| history_text(&entry.payload).ok().flatten())
                .map(|(_, text)| text)
                .collect::<Vec<_>>()
                .join("\n\n")
                .into())
        })
    }
}

fn runtime_id(id: &acp_v2::SessionId) -> Option<&str> {
    id.0.strip_prefix(SESSION_PREFIX)
}

fn workspace_pool_directory(repository: &std::path::Path) -> std::path::PathBuf {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in repository.as_os_str().to_string_lossy().bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    paths::data_dir()
        .join("agent-workspaces")
        .join(format!("{hash:016x}"))
}

impl AgentConnection for NativeRuntimeConnection {
    fn agent_id(&self) -> AgentId {
        crate::ZED_AGENT_ID.clone()
    }
    fn telemetry_id(&self) -> SharedString {
        "zed".into()
    }
    fn auth_methods(&self) -> &[acp_v2::AuthMethod] {
        &[]
    }
    fn authenticate(&self, _: acp_v2::AuthMethodId, _: &mut App) -> Task<Result<()>> {
        Task::ready(Ok(()))
    }
    fn supports_load_session(&self) -> bool {
        true
    }
    fn should_close_sessions_on_release(&self) -> bool {
        false
    }

    fn new_session(
        self: Rc<Self>,
        project: Entity<Project>,
        work_dirs: PathList,
        cx: &mut App,
    ) -> Task<Result<Entity<AcpThread>>> {
        let store = match self.store(cx) {
            Ok(store) => store,
            Err(error) => return Task::ready(Err(error)),
        };
        let id = uuid::Uuid::new_v4().to_string();
        let context = self.register_project(
            id.clone(),
            project.clone(),
            NativeRuntimeConfig::default(),
            cx,
        );
        cx.spawn(async move |cx| {
            let (configuration, workspace_error) = context.await?;
            store
                .create_root(id.clone(), serde_json::to_value(configuration)?)
                .await?;
            Ok(cx.update(|cx| {
                let thread = self.projection(id, project, work_dirs, None, -1, cx);
                thread.update(cx, |thread, cx| {
                    update_workspace_notice(thread, workspace_error, cx)
                });
                thread
            }))
        })
    }

    fn load_session(
        self: Rc<Self>,
        session_id: acp_v2::SessionId,
        project: Entity<Project>,
        work_dirs: PathList,
        title: Option<SharedString>,
        cx: &mut App,
    ) -> Task<Result<Entity<AcpThread>>> {
        let Some(id) = runtime_id(&session_id).map(str::to_owned) else {
            return Rc::new(self.legacy.clone())
                .load_session(session_id, project, work_dirs, title, cx);
        };
        let store = match self.store(cx) {
            Ok(store) => store,
            Err(error) => return Task::ready(Err(error)),
        };
        cx.spawn(async move |cx| {
            let snapshot = store.snapshot(id.clone()).await?;
            let configuration =
                serde_json::from_value(store.metadata(snapshot.root_id.clone()).await?)?;
            let (_, workspace_error) = cx
                .update(|cx| {
                    self.register_project(snapshot.root_id, project.clone(), configuration, cx)
                })
                .await?;
            let page = store.transcript_page(id.clone(), None, PAGE_SIZE).await?;
            let position = page.entries.last().map_or(-1, |entry| entry.position);
            let thread =
                cx.update(|cx| self.projection(id, project, work_dirs, title, position, cx));
            thread.update(cx, |thread, cx| {
                project_history(thread, page.entries, true, cx);
                update_workspace_notice(thread, workspace_error, cx);
            });
            Ok(thread)
        })
    }

    fn validate_prompt_content(&self, content: &[acp_v2::ContentBlock]) -> Result<()> {
        ensure!(!content.is_empty(), "the prompt is empty");
        for block in content {
            ensure!(
                !matches!(block, acp_v2::ContentBlock::Audio(_)),
                "audio input is not supported by this runtime"
            );
            UserMessageContent::validate_content_block(block)?;
        }
        Ok(())
    }

    fn prompt(
        &self,
        params: acp_v2::PromptRequest,
        cx: &mut App,
    ) -> Task<Result<acp_v1::PromptResponse>> {
        let Some(id) = runtime_id(&params.session_id).map(str::to_owned) else {
            return self.legacy.prompt(params, cx);
        };
        if self.updating.borrow().contains(&id) {
            return Task::ready(Err(anyhow::anyhow!(
                "wait for the agent configuration update before submitting"
            )));
        }
        if let Err(error) = self.validate_prompt_content(&params.prompt) {
            return Task::ready(Err(error));
        }
        // ACP calls this method during a thread update. Access its projection
        // inside the returned task, once that update has finished.
        if params.prompt.first().is_some_and(|block| matches!(block, acp_v2::ContentBlock::Text(text) if text.text.trim_start().split_whitespace().next() == Some("/compact"))) {
            let card_id = acp_thread::ContextCompactionId(uuid::Uuid::new_v4().to_string().into());
            let view = self.views.borrow().get(&id).map(|projection| projection.view.clone());
            let host = self.host.clone();
            return cx.spawn(async move |cx| {
                if let Some(view) = &view {
                    view.update(cx, |thread, cx| {
                        thread.push_context_compaction(
                            acp_thread::ContextCompaction {
                                id: card_id.clone(),
                                status: acp_thread::ContextCompactionStatus::InProgress,
                                error: None,
                                summary: acp_thread::MessageContent::default(),
                                meta: None,
                            },
                            cx,
                        )
                    })
                    .log_err();
                }
                let task = host.update(cx, |host, cx| host.compact(id, cx));
                let result = task.await;
                if let Some(view) = view {
                    let status = if result.is_ok() { acp_thread::ContextCompactionStatus::Completed } else { acp_thread::ContextCompactionStatus::Failed };
                    let summary = match &result { Ok(()) => "Context checkpoint saved. Resume or send a message to continue.".to_owned(), Err(error) => format!("{error:#}") };
                    view.update(cx, |thread, cx| {
                        thread.update_context_compaction(acp_thread::ContextCompactionUpdate { id: card_id, summary_delta: summary, status: Some(status) }, cx);
                        if result.is_ok() { thread.update_token_usage(None, cx); }
                    }).log_err();
                }
                result?;
                Ok(acp_v1::PromptResponse::new(acp_v1::StopReason::EndTurn))
            });
        }
        let store = match self.store(cx) {
            Ok(store) => store,
            Err(error) => return Task::ready(Err(error)),
        };
        let projection = self.views.borrow().get(&id).map(|projection| {
            (
                projection.view.clone(),
                projection.position.clone(),
                projection.usage.clone(),
            )
        });
        let Some((view, position, usage)) = projection else {
            return Task::ready(Err(anyhow::anyhow!("conversation projection is detached")));
        };
        let model_check = self.validate_prompt_model(&params.session_id, cx);
        let skills = self.skills.clone();
        let fs = self.fs.clone();
        cx.spawn(async move |cx| {
            model_check.await?;
            ensure_no_uncertain_tools(&store, &id).await?;
            let (project, path_style) = view.read_with(cx, |thread, cx| {
                let project = thread.project().clone();
                let path_style = project.read(cx).path_style(cx);
                (project, path_style)
            })?;
            let mut content = params
                .prompt
                .into_iter()
                .map(|block| UserMessageContent::from_content_block(block, path_style))
                .collect::<Result<Vec<_>>>()?;
            let skill_name = content.first().and_then(|content| match content {
                UserMessageContent::Text(text) => text
                    .trim_start()
                    .strip_prefix('/')
                    .and_then(|command| command.split_whitespace().next()),
                _ => None,
            });
            let skill = skill_name.and_then(|name| {
                skills
                    .borrow()
                    .get(&id)
                    .and_then(|skills| skills.iter().find(|skill| skill.name == name).cloned())
            });
            if let Some(skill) = skill {
                let read_skill = crate::skill_body_resolver_for_project(project, fs);
                let body = match skill.embedded_body { Some(body) => body.to_owned(), None => read_skill(skill.clone(), cx).await? };
                let envelope = crate::tools::render_skill_envelope(&skill, &body);
                if let Some(UserMessageContent::Text(text)) = content.first_mut() { *text = crate::strip_slash_command_prefix(text); }
                content.insert(0, UserMessageContent::Text(envelope.clone()));
                view.update(cx, |thread, cx| thread.push_user_content_block_with_indent(None, envelope.into(), true, cx))?;
            }
            let request = UserMessage { id: ClientUserMessageId::new(), content: content.into() }.to_request();
            let submission = uuid::Uuid::new_v4().to_string();
            let inputs = request.content.into_iter().enumerate().map(|(index, content)| {
                let content = match content {
                    MessageContent::Text(text) => json!({"type":"text","text":text}),
                    MessageContent::Image(image) => json!({"type":"image","media_type":"png","data":{"type":"base64","value":image.source.to_string()}}),
                    _ => anyhow::bail!("unsupported user prompt content"),
                };
                Ok((format!("{submission}:{index}"), json!({"group_id":id,"content":content})))
            }).collect::<Result<Vec<_>>>()?;
            store.enqueue_user_batch(id.clone(), inputs).await?;
            loop {
                let state = store.snapshot(id.clone()).await?;
                ensure_no_uncertain_tools(&store, &id).await?;
                match state.state.as_str() {
                    "paused" | "failed" => anyhow::bail!("{}", store.agent_error(id.clone()).await?.unwrap_or_else(|| format!("Agent is {}; use Resume in Agent fleet", state.state))),
                    "cancelled" => return Ok(acp_v1::PromptResponse::new(acp_v1::StopReason::Cancelled)),
                    "waiting" | "completed" if store.pending_job_count(id.clone()).await? == 0 => {
                        flush_projection(&store,&id,&view,&position,&usage,cx).await?;
                        return Ok(acp_v1::PromptResponse::new(acp_v1::StopReason::EndTurn));
                    },
                    _ => {}
                }
                cx.background_executor().timer(Duration::from_millis(100)).await;
            }
        })
    }

    fn cancel(&self, session_id: &acp_v2::SessionId, cx: &mut App) -> Result<()> {
        let Some(id) = runtime_id(session_id).map(str::to_owned) else {
            return self.legacy.cancel(session_id, cx);
        };
        let store = self.store(cx)?;
        cx.spawn(async move |_| store.cancel(id).await)
            .detach_and_log_err(cx);
        Ok(())
    }

    fn model_selector(&self, session_id: &acp_v2::SessionId) -> Option<Rc<dyn AgentModelSelector>> {
        let Some(id) = runtime_id(session_id) else {
            return self.legacy.model_selector(session_id);
        };
        Some(Rc::new(RuntimeModelSelector {
            id: id.to_owned(),
            connection: self.clone(),
        }))
    }

    fn session_modes(
        &self,
        session_id: &acp_v2::SessionId,
        cx: &App,
    ) -> Option<Rc<dyn acp_thread::AgentSessionModes>> {
        let id = runtime_id(session_id)?.to_owned();
        let settings = AgentSettings::get_global(cx);
        let current = self
            .configurations
            .borrow()
            .get(&id)
            .and_then(|configuration| configuration.profile.clone())
            .unwrap_or_else(|| settings.default_profile.clone());
        let profiles = settings
            .profiles
            .iter()
            .map(|(id, profile)| (id.clone(), profile.name.to_string()))
            .collect();
        let connection = self.clone();
        let configurations = self.configurations.clone();
        let getter_id = id.clone();
        let fallback = current.clone();
        Some(Rc::new(
            RuntimeSessionModes::new(
                current,
                profiles,
                Rc::new(move |profile, cx| {
                    let mut configuration = connection
                        .configurations
                        .borrow()
                        .get(&id)
                        .cloned()
                        .unwrap_or_default();
                    if let Some(model) = AgentSettings::get_global(cx)
                        .profiles
                        .get(&profile)
                        .and_then(|profile| profile.default_model.clone())
                    {
                        configuration.model = Some(model);
                    }
                    configuration.profile = Some(profile);
                    connection.update_configuration(id.clone(), configuration, cx)
                }),
            )
            .with_current_profile(Rc::new(move || {
                configurations
                    .borrow()
                    .get(&getter_id)
                    .and_then(|configuration| configuration.profile.clone())
                    .unwrap_or_else(|| fallback.clone())
            })),
        ))
    }

    fn session_config_options(
        &self,
        session_id: &acp_v2::SessionId,
        cx: &App,
    ) -> Option<Rc<dyn acp_thread::AgentSessionConfigOptions>> {
        let id = runtime_id(session_id)?.to_owned();
        if let Some(control) = self
            .config_controls
            .borrow()
            .get(&id)
            .and_then(std::rc::Weak::upgrade)
        {
            return Some(control);
        }
        let configuration = self
            .configurations
            .borrow()
            .get(&id)
            .cloned()
            .unwrap_or_default();
        let options = runtime_model_options(&configuration, cx).unwrap_or_default();
        let connection = self.clone();
        let control_id = id.clone();
        let control = Rc::new(RuntimeSessionConfigOptions::new(
            options,
            Rc::new(move |option, value, cx| {
                let mut configuration = connection
                    .configurations
                    .borrow()
                    .get(&id)
                    .cloned()
                    .unwrap_or_default();
                let model = match configured_model(&configuration, cx) {
                    Ok(model) => model,
                    Err(error) => return Task::ready(Err(error)),
                };
                let mut selection =
                    language_model_to_selection(&model, configuration.model.as_ref());
                match (option.0.as_ref(), value) {
                    ("thinking", acp_v2::SessionConfigOptionValue::Boolean { value }) => {
                        if !value
                            && !model
                                .supports_disabling_thinking_at_effort(selection.effort.as_deref())
                        {
                            return Task::ready(Err(anyhow::anyhow!(
                                "selected model requires thinking at this effort"
                            )));
                        }
                        selection.enable_thinking = value;
                    }
                    ("effort", acp_v2::SessionConfigOptionValue::Id { value }) => {
                        selection.effort = Some(value.0.to_string())
                    }
                    ("fast", acp_v2::SessionConfigOptionValue::Boolean { value }) => {
                        selection.speed = Some(if value {
                            language_model::Speed::Fast
                        } else {
                            language_model::Speed::Standard
                        })
                    }
                    _ => {
                        return Task::ready(Err(anyhow::anyhow!(
                            "unsupported model configuration"
                        )));
                    }
                }
                configuration.model = Some(selection);
                let task = connection.update_configuration(id.clone(), configuration, cx);
                let connection = connection.clone();
                let id = id.clone();
                cx.spawn(async move |cx| {
                    task.await?;
                    cx.update(|cx| {
                        runtime_model_options(
                            &connection
                                .configurations
                                .borrow()
                                .get(&id)
                                .cloned()
                                .unwrap_or_default(),
                            cx,
                        )
                    })
                })
            }),
        ));
        self.config_controls
            .borrow_mut()
            .insert(control_id, Rc::downgrade(&control));
        Some(control)
    }

    fn into_any(self: Rc<Self>) -> Rc<dyn Any> {
        self
    }
}

struct RuntimeModelSelector {
    id: String,
    connection: NativeRuntimeConnection,
}

fn configured_model(
    configuration: &NativeRuntimeConfig,
    cx: &App,
) -> Result<language_model::LanguageModel> {
    match configuration.model.as_ref() {
        Some(selection) => crate::native_runtime::resolve_model_id(
            &format!("{}/{}", selection.provider.0, selection.model),
            cx,
        ),
        None => LanguageModelRegistry::read_global(cx)
            .default_model()
            .context("no model selected"),
    }
}

fn runtime_model_options(
    configuration: &NativeRuntimeConfig,
    cx: &App,
) -> Result<Vec<acp_v2::SessionConfigOption>> {
    let model = configured_model(configuration, cx)?;
    let selection = language_model_to_selection(&model, configuration.model.as_ref());
    let mut options = Vec::new();
    if model.supports_thinking()
        && model.supports_disabling_thinking_at_effort(selection.effort.as_deref())
    {
        options.push(acp_v2::SessionConfigOption::boolean(
            "thinking",
            "Thinking",
            selection.enable_thinking,
        ));
    }
    let efforts = model.supported_effort_levels();
    if let Some(current) = selection.effort {
        let choices = efforts
            .into_iter()
            .map(|effort| {
                acp_v2::SessionConfigSelectOption::new(
                    effort.value.to_string(),
                    effort.name.to_string(),
                )
            })
            .collect::<Vec<_>>();
        options.push(
            acp_v2::SessionConfigOption::select("effort", "Reasoning effort", current, choices)
                .category(acp_v2::SessionConfigOptionCategory::ThoughtLevel),
        );
    }
    if model.supports_fast_mode() {
        options.push(acp_v2::SessionConfigOption::boolean(
            "fast",
            "Fast mode",
            selection.speed == Some(language_model::Speed::Fast),
        ));
    }
    Ok(options)
}

impl AgentModelSelector for RuntimeModelSelector {
    fn list_models(&self, cx: &mut App) -> Task<Result<acp_thread::AgentModelList>> {
        NativeAgentModelSelector {
            session_id: acp_v2::SessionId::new(format!("{SESSION_PREFIX}{}", self.id)),
            connection: self.connection.legacy.clone(),
        }
        .list_models(cx)
    }
    fn select_model(&self, model_id: AgentModelId, cx: &mut App) -> Task<Result<()>> {
        let selection = crate::model_id_to_selection(&model_id, cx);
        if let Err(error) = crate::native_runtime::resolve_model_id(
            &format!("{}/{}", selection.provider.0, selection.model),
            cx,
        ) {
            return Task::ready(Err(error));
        }
        let mut configuration = self
            .connection
            .configurations
            .borrow()
            .get(&self.id)
            .cloned()
            .unwrap_or_default();
        configuration.model = Some(selection);
        self.connection
            .update_configuration(self.id.clone(), configuration, cx)
    }
    fn selected_model(&self, cx: &mut App) -> Task<Result<acp_thread::AgentModelInfo>> {
        let store = match self.connection.store(cx) {
            Ok(store) => store,
            Err(error) => return Task::ready(Err(error)),
        };
        let id = self.id.clone();
        let config_controls = self.connection.config_controls.clone();
        cx.spawn(async move |cx| {
            let configuration: NativeRuntimeConfig =
                serde_json::from_value(store.metadata(id.clone()).await?)?;
            cx.update(|cx| {
                if let Some(control) = config_controls
                    .borrow()
                    .get(&id)
                    .and_then(std::rc::Weak::upgrade)
                {
                    control.replace_options(
                        runtime_model_options(&configuration, cx).unwrap_or_default(),
                    );
                }
                let model = configured_model(&configuration, cx)?;
                let provider = LanguageModelRegistry::read_global(cx).provider_for_model(&model)?;
                Ok(crate::LanguageModels::map_language_model_to_info(
                    &model, &provider,
                ))
            })
        })
    }

    fn favorite_model_ids(&self, cx: &mut App) -> HashSet<AgentModelId> {
        NativeAgentModelSelector {
            session_id: acp_v2::SessionId::new(format!("{SESSION_PREFIX}{}", self.id)),
            connection: self.connection.legacy.clone(),
        }
        .favorite_model_ids(cx)
    }

    fn toggle_favorite_model(&self, id: AgentModelId, favorite: bool, cx: &App) {
        NativeAgentModelSelector {
            session_id: acp_v2::SessionId::new(format!("{SESSION_PREFIX}{}", self.id)),
            connection: self.connection.legacy.clone(),
        }
        .toggle_favorite_model(id, favorite, cx);
    }

    fn watch(&self, cx: &mut App) -> Option<watch::Receiver<()>> {
        Some(self.connection.legacy.0.read(cx).models.watch())
    }

    fn should_render_footer(&self) -> bool {
        true
    }
}

async fn ensure_no_uncertain_tools(store: &RuntimeStore, id: &str) -> Result<()> {
    ensure!(
        store.uncertain_jobs(id.to_owned(), 1).await?.is_empty(),
        "An interrupted tool has an unknown outcome. Open Agent fleet, select this agent, inspect changes, and choose Continue without retrying before resuming or sending another message."
    );
    Ok(())
}

async fn flush_projection(
    store: &RuntimeStore,
    agent_id: &str,
    view: &WeakEntity<AcpThread>,
    position: &Rc<Cell<i64>>,
    projected_usage: &Rc<RefCell<Option<(String, acp_thread::TokenUsage)>>>,
    cx: &mut AsyncApp,
) -> Result<bool> {
    if view.upgrade().is_none() {
        return Ok(false);
    }
    loop {
        let entries = store
            .transcript_after(agent_id.to_owned(), position.get(), PAGE_SIZE)
            .await?;
        if entries.is_empty() {
            break;
        }
        for entry in entries {
            // A prompt completion and its observer may have read the same page.
            if entry.position <= position.get() {
                continue;
            }
            // Let the live view handle each append before changing the entry count again.
            view.update(cx, |thread, cx| {
                project_history_entry(thread, &entry, false, cx);
                position.set(entry.position);
            })?;
        }
        // A short page can hit the byte limit before the row limit.
        futures_lite::future::yield_now().await;
    }
    let usage = match store.last_provider_usage(agent_id.to_owned()).await {
        Ok(usage) => usage,
        Err(error) => {
            log::warn!("could not read runtime usage: {error:#}");
            return Ok(view.upgrade().is_some());
        }
    };
    view.update(cx, |thread, cx| {
        let Some(record) = usage else {
            projected_usage.borrow_mut().take();
            if thread.token_usage().is_some() {
                thread.update_token_usage(None, cx);
            }
            return;
        };
        let model = match crate::native_runtime::resolve_model_id(&record.model_id, cx) {
            Ok(model) => model,
            Err(error) => {
                log::debug!("runtime usage model is unavailable: {error:#}");
                return;
            }
        };
        let usage = acp_thread::TokenUsage {
            max_tokens: model.max_token_count(),
            max_output_tokens: model.max_output_tokens(),
            used_tokens: record.usage.total_tokens(),
            input_tokens: record
                .usage
                .input_tokens
                .saturating_add(record.usage.cache_creation_input_tokens)
                .saturating_add(record.usage.cache_read_input_tokens),
            output_tokens: record.usage.output_tokens,
        };
        let mut projected = projected_usage.borrow_mut();
        let output_delta = match projected.as_ref() {
            Some((attempt, previous)) if attempt == &record.attempt_id => {
                if previous == &usage {
                    return;
                }
                usage.output_tokens.saturating_sub(previous.output_tokens)
            }
            _ => usage.output_tokens,
        };
        thread.update_token_usage_with_output_delta(Some(usage.clone()), output_delta, cx);
        *projected = Some((record.attempt_id, usage));
    })?;
    Ok(true)
}

fn history_text(payload: &Value) -> Result<Option<(bool, String)>> {
    let content = payload.get("content");
    match payload.get("kind").and_then(Value::as_str) {
        Some("user") => Ok(content
            .and_then(|content| content.get("text"))
            .and_then(Value::as_str)
            .map(|text| (false, text.to_owned()))),
        Some("assistant") => assistant_history_text(content.context("missing assistant content")?),
        _ => Ok(None),
    }
}

fn assistant_history_text(content: &Value) -> Result<Option<(bool, String)>> {
    let content_type = content
        .get("type")
        .map(|tag| tag.as_str().context("invalid assistant content type"))
        .transpose()?;
    // Infinity's AssistantContent is untagged. Older projections also accepted tagged text.
    if matches!(content_type, None | Some("text"))
        && let Some(text) = content.get("text")
    {
        let text = text.as_str().context("invalid assistant text")?;
        return Ok((!text.is_empty()).then(|| (false, text.to_owned())));
    }
    ensure!(
        matches!(content_type, None | Some("reasoning")),
        "unsupported assistant content type"
    );
    let blocks = content
        .get("content")
        .and_then(Value::as_array)
        .context("unsupported assistant content shape")?;
    let mut thinking = String::new();
    for block in blocks {
        let text = match block.get("type").and_then(Value::as_str) {
            Some("text") => block
                .pointer("/content/text")
                .or_else(|| block.get("text"))
                .and_then(Value::as_str)
                .context("invalid reasoning text")?,
            Some("summary") => block
                .get("content")
                .and_then(Value::as_str)
                .context("invalid reasoning summary")?,
            Some("provider_state" | "encrypted" | "redacted") => continue,
            None if block.get("type").is_none() => block
                .get("text")
                .and_then(Value::as_str)
                .context("unsupported reasoning block")?,
            _ => anyhow::bail!("unsupported reasoning block type"),
        };
        thinking.push_str(text);
    }
    Ok((!thinking.is_empty()).then_some((true, thinking)))
}

fn project_history(
    thread: &mut AcpThread,
    entries: Vec<TranscriptEntry>,
    replay_users: bool,
    cx: &mut gpui::Context<AcpThread>,
) {
    for entry in entries {
        project_history_entry(thread, &entry, replay_users, cx);
    }
}

fn project_history_entry(
    thread: &mut AcpThread,
    entry: &TranscriptEntry,
    replay_users: bool,
    cx: &mut gpui::Context<AcpThread>,
) {
    let result = (|| -> Result<()> {
        let payload = &entry.payload;
        match payload.get("kind").and_then(Value::as_str) {
            Some("user") if replay_users => {
                let content = history_text(payload)?
                    .map(|(_, text)| text.into())
                    .or_else(|| payload.get("content").and_then(history_content))
                    .context("unsupported user content")?;
                thread.push_user_content_block(None, content, cx);
            }
            Some("assistant") => {
                if let Some((thinking, text)) = history_text(payload)? {
                    thread.push_assistant_content_block(text.into(), thinking, cx);
                }
            }
            Some("tool_call" | "tool_result") => {
                let update = history_tool_update(payload).context("unsupported tool content")?;
                thread.update_tool_call(update, cx)?;
            }
            Some("user" | "subscription_event") => {}
            _ => anyhow::bail!("unsupported history record kind"),
        }
        Ok(())
    })();
    if let Err(error) = result {
        log::warn!(
            "could not display runtime history row {}: {error:#}",
            entry.position
        );
        projection_notice(
            thread,
            "Some saved messages could not be displayed",
            "The original messages are still saved. See the Zed log for details.",
            cx,
        );
    }
}

fn projection_notice(
    thread: &mut AcpThread,
    title: &'static str,
    description: &'static str,
    cx: &mut gpui::Context<AcpThread>,
) {
    if !thread
        .notices()
        .iter()
        .any(|(_, notice)| notice.title == title)
    {
        thread.push_notice(
            acp_v2::Notice::new(acp_v2::NoticeSeverity::Warning, title).description(description),
            cx,
        );
    }
}

fn update_workspace_notice(
    thread: &mut AcpThread,
    description: Option<String>,
    cx: &mut gpui::Context<AcpThread>,
) {
    const TITLE: &str = "Editing workspace unavailable";
    let previous = thread
        .notices()
        .iter()
        .find(|(_, notice)| notice.title == TITLE)
        .map(|(id, _)| *id);
    if let Some(id) = previous {
        thread.dismiss_notice(id, cx);
    }
    if let Some(description) = description {
        thread.push_notice(
            acp_v2::Notice::new(acp_v2::NoticeSeverity::Warning, TITLE).description(description),
            cx,
        );
    }
}

fn history_content(content: &Value) -> Option<acp_v2::ContentBlock> {
    match content.get("type")?.as_str()? {
        "text" => content
            .get("text")
            .and_then(Value::as_str)
            .map(|text| text.to_owned().into()),
        "image" => {
            let source = content.get("data")?;
            if source.get("type")?.as_str()? != "base64" {
                return None;
            }
            let data = source.get("value")?.as_str()?.to_owned();
            let media_type = match content
                .get("media_type")
                .and_then(Value::as_str)
                .unwrap_or("png")
            {
                "jpeg" => "image/jpeg",
                "gif" => "image/gif",
                "webp" => "image/webp",
                "svg" => "image/svg+xml",
                "heic" => "image/heic",
                "heif" => "image/heif",
                _ => "image/png",
            };
            Some(acp_v2::ContentBlock::Image(acp_v2::ImageContent::new(
                data, media_type,
            )))
        }
        _ => None,
    }
}

fn history_tool_update(payload: &Value) -> Option<acp_v2::ToolCallUpdate> {
    match payload.get("kind")?.as_str()? {
        "tool_call" => {
            let call = payload.get("call")?;
            let id = payload
                .get("runtime_job_id")
                .and_then(Value::as_str)
                .or_else(|| call.get("id").and_then(Value::as_str))?;
            let function = call.get("function")?;
            let name = function.get("name")?.as_str()?.to_owned();
            Some(
                acp_v2::ToolCallUpdate::new(id.to_owned())
                    .name(name.clone())
                    .title(name)
                    .raw_input(function.get("arguments")?.clone())
                    .status(acp_v2::ToolCallStatus::Pending),
            )
        }
        "tool_result" => {
            let result = payload.get("result")?;
            let id = payload
                .get("runtime_job_id")
                .and_then(Value::as_str)
                .or_else(|| result.get("id").and_then(Value::as_str))?;
            let failed = result
                .get("is_error")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let content = result
                .get("content")?
                .as_array()?
                .iter()
                .filter_map(history_content)
                .map(|content| {
                    acp_v2::ToolCallContent::Content(Box::new(acp_v2::Content::new(content)))
                })
                .collect::<Vec<_>>();
            Some(
                acp_v2::ToolCallUpdate::new(id.to_owned())
                    .status(if failed {
                        acp_v2::ToolCallStatus::Failed
                    } else {
                        acp_v2::ToolCallStatus::Completed
                    })
                    .content(content)
                    .raw_output(
                        result
                            .get("output")
                            .cloned()
                            .unwrap_or_else(|| result.clone()),
                    ),
            )
        }
        _ => None,
    }
}

pub(crate) fn project_runtime_tool_event(
    view: &WeakEntity<AcpThread>,
    event: &mut Option<ThreadEvent>,
    cx: &mut App,
) -> Result<()> {
    let Some(thread) = view.upgrade() else {
        return Ok(());
    };
    let Some(event) = event.take() else {
        return Ok(());
    };
    match event {
        ThreadEvent::ToolCallUpdate(update) => {
            thread.update(cx, |thread, cx| thread.update_tool_call(update, cx))?
        }
        ThreadEvent::ToolCallAuthorization(authorization) => {
            let task = thread.update(cx, |thread, cx| {
                thread.request_tool_call_update_authorization(
                    authorization.tool_call,
                    authorization.options,
                    authorization.kind,
                    cx,
                )
            })?;
            cx.spawn(async move |_| {
                if let acp_thread::RequestPermissionOutcome::Selected(outcome) = task.await {
                    if authorization.response.send(outcome).is_err() {
                        log::debug!("runtime approval requester detached");
                    }
                }
            })
            .detach();
        }
        ThreadEvent::ToolCallAuthorizationResolved {
            tool_call_id,
            outcome,
        } => thread.update(cx, |thread, cx| {
            thread.authorize_tool_call(tool_call_id, outcome, cx)
        }),
        ThreadEvent::Elicitation(request) => {
            let task = thread.update(cx, |thread, cx| {
                let scope = acp_v2::ElicitationSessionScope::new(thread.session_id().clone())
                    .tool_call_id(request.tool_call_id);
                thread.request_elicitation(
                    acp_v2::CreateElicitationRequest::new(
                        acp_v2::ElicitationFormMode::new(scope, request.schema),
                        request.message,
                    ),
                    cx,
                )
            })?;
            cx.spawn(async move |_| {
                if request.response.send(task.await).is_err() {
                    log::debug!("runtime elicitation requester detached");
                }
            })
            .detach();
        }
        ThreadEvent::AgentText(text) => thread.update(cx, |thread, cx| {
            thread.push_assistant_content_block(text.into(), false, cx)
        }),
        ThreadEvent::AgentThinking(text) => thread.update(cx, |thread, cx| {
            thread.push_assistant_content_block(text.into(), true, cx)
        }),
        ThreadEvent::Retry(status) => {
            thread.update(cx, |thread, cx| thread.update_retry_status(status, cx))
        }
        ThreadEvent::ContextCompaction(compaction) => thread.update(cx, |thread, cx| {
            thread.push_context_compaction(compaction, cx)
        }),
        ThreadEvent::ContextCompactionUpdate(update) => thread.update(cx, |thread, cx| {
            thread.update_context_compaction(update, cx)
        }),
        ThreadEvent::UserMessage(_) | ThreadEvent::Stop(_) => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use fs::FakeFs;
    use gpui::{TestAppContext, UpdateGlobal as _};
    use infinity_agent_core::{ThreadId, message::InfinityMessage, traits::ConversationStore as _};
    use infinity_provider_protocol::message::{
        AssistantContent, Reasoning, ReasoningContent, UserContent,
    };
    use language_model::{
        LanguageModelCompletionEvent, LanguageModelToolUse, LanguageModelToolUseInput, StopReason,
    };

    async fn wait_until(mut condition: impl FnMut() -> bool) {
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while !condition() {
            assert!(
                std::time::Instant::now() < deadline,
                "native runtime condition timed out"
            );
            // These workers run on OS threads, outside the simulated GPUI clock.
            async_io::Timer::after(Duration::from_millis(2)).await;
        }
    }

    #[gpui::test]
    async fn runtime_session_recovers_unavailable_model(cx: &mut TestAppContext) {
        cx.executor().allow_parking();
        let provider = crate::tests::init_test(cx);
        let model = provider.model("fake");
        let mut unavailable = language_model_to_selection(&model, None);
        unavailable.model = "removed-from-catalog".into();
        cx.update(|cx| {
            settings::SettingsStore::update_global(cx, |store, cx| {
                store
                    .set_user_settings(
                        &json!({"agent": {"default_model": unavailable}}).to_string(),
                        cx,
                    )
                    .result()
                    .expect("test settings should parse");
            });
            LanguageModelRegistry::global(cx).update(cx, |registry, cx| {
                registry.set_default_model(None, cx);
            });
        });
        let fs = FakeFs::new(cx.executor());
        let project = Project::test(fs.clone(), [], cx).await;
        let directory = tempfile::tempdir().expect("directory");
        let host = cx.update(|cx| NativeRuntime::global(cx).expect("host"));
        let runtime = host
            .update(cx, |host, cx| {
                host.start_database(
                    directory.path().join("runtime.sqlite"),
                    agent_runtime::RuntimeLimits::default(),
                    cx,
                )
            })
            .await
            .expect("runtime");
        let connection = cx.update(|cx| {
            let thread_store = cx.new(crate::ThreadStore::new);
            let legacy = NativeAgentConnection(crate::NativeAgent::new(
                thread_store,
                crate::Templates::new(),
                fs.clone(),
                cx,
            ));
            Rc::new(NativeRuntimeConnection::new(legacy, host, fs))
        });
        let work_dirs = PathList::new::<&std::path::Path>(&[]);
        let thread = cx
            .update(|cx| {
                connection
                    .clone()
                    .new_session(project.clone(), work_dirs.clone(), cx)
            })
            .await
            .expect("open a conversation with an unavailable model");
        let session = thread.read_with(cx, |thread, _| thread.session_id().clone());
        let id = runtime_id(&session).expect("runtime session").to_owned();
        let reopened_connection = Rc::new(NativeRuntimeConnection::new(
            connection.legacy.clone(),
            connection.host.clone(),
            connection.fs.clone(),
        ));
        let reopened = cx
            .update(|cx| {
                reopened_connection.clone().load_session(
                    session.clone(),
                    project.clone(),
                    work_dirs.clone(),
                    None,
                    cx,
                )
            })
            .await
            .expect("reopen the conversation with its saved model");
        let selector = connection.model_selector(&session).expect("model picker");
        assert!(cx.update(|cx| selector.selected_model(cx)).await.is_err());
        assert!(cx.update(|cx| selector.list_models(cx)).await.is_ok());

        let prompt = || acp_v2::PromptRequest::new(session.clone(), vec!["Hello".into()]);
        let error = cx
            .update(|cx| reopened_connection.prompt(prompt(), cx))
            .await
            .expect_err("the unavailable selection must block inference");
        assert!(error.is::<crate::NoModelConfiguredError>());
        assert_eq!(
            runtime
                .store
                .snapshot(id.clone())
                .await
                .expect("agent")
                .state,
            "waiting"
        );
        assert!(provider.pending_completions_for(&model).is_empty());

        let compaction_error = reopened
            .update(cx, |thread, cx| {
                thread.send_command(vec!["/compact".into()], cx)
            })
            .await
            .expect_err("compaction should report the unavailable model");
        assert!(format!("{compaction_error:#}").contains("removed-from-catalog"));

        cx.update(|cx| {
            selector.select_model(
                AgentModelId::from(format!("{}/{}", model.provider_id().0, model.id().0)),
                cx,
            )
        })
        .await
        .expect("select a model from the live catalog");
        let turns = [
            ("Hello", "First answer from Infinity."),
            ("Continue", "Second answer from Infinity."),
        ];
        for (prompt, answer) in turns {
            let execution = reopened.update(cx, |thread, cx| thread.send(vec![prompt.into()], cx));
            wait_until(|| !provider.pending_completions_for(&model).is_empty()).await;
            let request = provider.pending_completions_for(&model).remove(0);
            provider.send_text(&model, &request, answer);
            provider.send_event(
                &model,
                &request,
                LanguageModelCompletionEvent::Stop(StopReason::EndTurn),
            );
            provider.end_stream(&model, &request);
            execution
                .await
                .expect("completion in the recovered conversation");
        }
        let live_markdown = reopened.read_with(cx, |thread, cx| thread.to_markdown(cx));
        drop(reopened);
        drop(thread);
        let replayed = cx
            .update(|cx| {
                reopened_connection.clone().load_session(
                    session.clone(),
                    project,
                    work_dirs,
                    None,
                    cx,
                )
            })
            .await
            .expect("replay the saved conversation");
        assert_eq!(
            replayed.read_with(cx, |thread, _| thread.session_id().clone()),
            session
        );
        let replayed_markdown = replayed.read_with(cx, |thread, cx| thread.to_markdown(cx));
        for (_, answer) in turns {
            assert_eq!(live_markdown.matches(answer).count(), 1, "live transcript");
            assert_eq!(
                replayed_markdown.matches(answer).count(),
                1,
                "replayed transcript"
            );
        }
        assert!(replayed.read_with(cx, |thread, _| thread.notices().is_empty()));
    }

    #[gpui::test]
    async fn native_panel_connection_runs_tool_approval_and_followup_without_legacy_threads(
        cx: &mut TestAppContext,
    ) {
        cx.executor().allow_parking();
        let provider = crate::tests::init_test(cx);
        let model = provider.model("fake");
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            "/runtime-flow",
            json!({"project":{}, "external":{"data.txt":"approved external content"}}),
        )
        .await;
        fs.create_symlink(
            std::path::Path::new("/runtime-flow/project/link.txt"),
            std::path::PathBuf::from("../external/data.txt"),
        )
        .await
        .expect("symlink");
        let project = Project::test(
            fs.clone(),
            [std::path::Path::new("/runtime-flow/project")],
            cx,
        )
        .await;
        let directory = tempfile::tempdir().expect("directory");
        let host = cx.update(|cx| NativeRuntime::global(cx).expect("host"));
        let runtime = host
            .update(cx, |host, cx| {
                host.start_database(
                    directory.path().join("runtime.sqlite"),
                    agent_runtime::RuntimeLimits::default(),
                    cx,
                )
            })
            .await
            .expect("runtime");
        let legacy = cx.update(|cx| {
            let thread_store = cx.new(|cx| crate::ThreadStore::new(cx));
            NativeAgentConnection(crate::NativeAgent::new(
                thread_store,
                crate::Templates::new(),
                fs.clone(),
                cx,
            ))
        });
        let connection = Rc::new(NativeRuntimeConnection::new(legacy.clone(), host, fs));
        let thread = cx
            .update(|cx| {
                connection
                    .clone()
                    .new_session(project, PathList::new::<&std::path::Path>(&[]), cx)
            })
            .await
            .expect("runtime session");
        let session = thread.read_with(cx, |thread, _| thread.session_id().clone());
        assert!(runtime_id(&session).is_some());
        let output_tokens = Rc::new(Cell::new(0));
        let _usage_subscription = cx.update(|cx| {
            cx.subscribe(&thread, {
                let output_tokens = output_tokens.clone();
                move |_, event, _| {
                    if let acp_thread::AcpThreadEvent::TokenUsageUpdated(
                        acp_thread::OutputTokensUpdate::LegacyAdd {
                            output_tokens: delta,
                            ..
                        },
                    ) = event
                    {
                        output_tokens.set(output_tokens.get() + delta);
                    }
                }
            })
        });
        assert!(
            legacy
                .0
                .read_with(cx, |legacy, _| legacy.sessions.is_empty())
        );
        let execution = thread.update(cx, |thread, cx| {
            thread.send(vec!["Read the external linked file.".into()], cx)
        });
        wait_until(|| !provider.pending_completions_for(&model).is_empty()).await;
        let request = provider
            .pending_completions_for(&model)
            .into_iter()
            .next()
            .expect("model request");
        assert!(request.tools.iter().any(|tool| tool.name == "read_file"));
        provider.send_event(
            &model,
            &request,
            LanguageModelCompletionEvent::ToolUse(LanguageModelToolUse {
                id: "read-external".into(),
                name: "read_file".into(),
                raw_input: r#"{"path":"project/link.txt"}"#.into(),
                input: LanguageModelToolUseInput::Json(json!({"path":"project/link.txt"})),
                is_input_complete: true,
                thought_signature: None,
            }),
        );
        provider.send_event(
            &model,
            &request,
            LanguageModelCompletionEvent::Stop(StopReason::ToolUse),
        );
        provider.end_stream(&model, &request);
        wait_until(|| {
            thread.read_with(cx, |thread, _| {
                thread.pending_permission_requests().next().is_some()
            })
        })
        .await;
        thread.update(cx, |thread, cx| {
            let approval = thread
                .pending_permission_requests()
                .next()
                .expect("approval");
            let id = approval.id;
            let option = approval
                .legacy_options()
                .expect("native options")
                .first_option_of_kind(acp_v2::PermissionOptionKind::AllowOnce)
                .expect("allow");
            thread.authorize_permission_request(
                id,
                acp_thread::SelectedPermissionOutcome::new(
                    option.option_id.clone(),
                    option.kind.clone(),
                ),
                cx,
            );
        });
        wait_until(|| !provider.pending_completions_for(&model).is_empty()).await;
        let request = provider
            .pending_completions_for(&model)
            .into_iter()
            .next()
            .expect("followup");
        assert!(format!("{:?}", request.messages).contains("approved external content"));
        provider.send_event(
            &model,
            &request,
            LanguageModelCompletionEvent::UsageUpdate(language_model::TokenUsage {
                input_tokens: 100,
                output_tokens: 7,
                cache_creation_input_tokens: 3,
                cache_read_input_tokens: 5,
            }),
        );
        provider.send_text(&model, &request, "Read successfully.");
        provider.send_event(
            &model,
            &request,
            LanguageModelCompletionEvent::Stop(StopReason::EndTurn),
        );
        provider.end_stream(&model, &request);
        execution.await.expect("completion");
        assert!(thread.read_with(cx, |thread, cx| {
            thread.to_markdown(cx).contains("Read successfully.")
        }));
        assert_eq!(
            thread.read_with(cx, |thread, _| thread.token_usage().cloned()),
            Some(acp_thread::TokenUsage {
                max_tokens: model.max_token_count(),
                max_output_tokens: model.max_output_tokens(),
                used_tokens: 115,
                input_tokens: 108,
                output_tokens: 7,
            })
        );
        let id = runtime_id(&session).expect("runtime id").to_owned();
        let (view, position, usage) = {
            let mut projections = connection.views.borrow_mut();
            let projection = projections.get_mut(&id).expect("projection");
            // Exercise one flush without an observer racing to read a second page.
            projection._observer = Task::ready(());
            (
                projection.view.clone(),
                projection.position.clone(),
                projection.usage.clone(),
            )
        };
        let provider_state = InfinityMessage::Assistant {
            content: AssistantContent::Reasoning(Reasoning {
                id: None,
                content: vec![provider_state_block()],
            }),
        };
        let mut history = (0..PAGE_SIZE)
            .map(|index| (provider_state.clone(), format!("projection-state-{index}")))
            .collect::<Vec<_>>();
        history.push((
            InfinityMessage::Assistant {
                content: AssistantContent::ToolCall(
                    infinity_provider_protocol::message::ToolCall::new(
                        "unexpected-assistant-call",
                        "unsupported_shape",
                        json!({}),
                    ),
                ),
            },
            "projection-unsupported".into(),
        ));
        history.push((
            InfinityMessage::ToolCall {
                call: infinity_provider_protocol::message::ToolCall::new(
                    "projection-call",
                    "read_file",
                    json!({"path":"example.txt"}),
                ),
                display_as: None,
            },
            "projection-tool-call".into(),
        ));
        let last_answer = "Visible answer from the next page.";
        history.push((
            InfinityMessage::Assistant {
                content: AssistantContent::text(last_answer),
            },
            "projection-answer".into(),
        ));
        let last_position = position.get() + history.len() as i64;
        agent_runtime::SqliteInfinityStore::new(runtime.store.clone())
            .append_messages(ThreadId::from_ref(&id), history)
            .await
            .expect("persist a burst spanning transcript pages");
        let initial_entry_count = thread.read_with(cx, |thread, _| thread.entries().len());
        let live_entry_counts = Rc::new(RefCell::new(Vec::new()));
        let _entry_subscription = cx.update(|cx| {
            let counts = live_entry_counts.clone();
            cx.subscribe(&thread, move |thread, event, cx| {
                if matches!(event, acp_thread::AcpThreadEvent::NewEntry) {
                    counts.borrow_mut().push(thread.read(cx).entries().len());
                }
            })
        });
        let store = runtime.store.clone();
        cx.update(|cx| {
            cx.spawn(async move |cx| {
                flush_projection(&store, &id, &view, &position, &usage, cx).await?;
                assert_eq!(position.get(), last_position);
                flush_projection(&store, &id, &view, &position, &usage, cx).await
            })
        })
        .await
        .expect("repeat projection");
        cx.run_until_parked();
        assert_eq!(
            *live_entry_counts.borrow(),
            [initial_entry_count + 1, initial_entry_count + 2]
        );
        thread.read_with(cx, |thread, cx| {
            assert_eq!(thread.to_markdown(cx).matches(last_answer).count(), 1);
            assert_eq!(
                thread.notices().len(),
                1,
                "unsupported history should be reported once"
            );
        });
        assert_eq!(
            output_tokens.get(),
            7,
            "replayed usage must not recount output"
        );
        assert!(
            legacy
                .0
                .read_with(cx, |legacy, _| legacy.sessions.is_empty())
        );
        runtime.shutdown().await.expect("shutdown");
    }

    #[gpui::test]
    async fn child_profile_limits_the_actual_model_tool_catalog(cx: &mut TestAppContext) {
        cx.executor().allow_parking();
        let provider = crate::tests::init_test(cx);
        let model = provider.model("fake");
        let profile = agent_settings::AgentProfileId("runtime-read-only".into());
        cx.update(|cx| {
            let mut settings = AgentSettings::get_global(cx).clone();
            let mut read_only = settings
                .profiles
                .get(&settings.default_profile)
                .expect("profile")
                .clone();
            read_only.tools.clear();
            read_only.tools.insert("read_file".into(), true);
            settings.profiles.insert(profile.clone(), read_only);
            AgentSettings::override_global(settings, cx);
        });
        let fs = FakeFs::new(cx.executor());
        let project = Project::test(fs.clone(), [], cx).await;
        let directory = tempfile::tempdir().expect("directory");
        let host = cx.update(|cx| NativeRuntime::global(cx).expect("host"));
        let runtime = host
            .update(cx, |host, cx| {
                host.start_database(
                    directory.path().join("runtime.sqlite"),
                    agent_runtime::RuntimeLimits::default(),
                    cx,
                )
            })
            .await
            .expect("runtime");
        let legacy = cx.update(|cx| {
            let thread_store = cx.new(|cx| crate::ThreadStore::new(cx));
            NativeAgentConnection(crate::NativeAgent::new(
                thread_store,
                crate::Templates::new(),
                fs.clone(),
                cx,
            ))
        });
        let connection = Rc::new(NativeRuntimeConnection::new(legacy, host, fs));
        let thread = cx
            .update(|cx| {
                connection.new_session(project, PathList::new::<&std::path::Path>(&[]), cx)
            })
            .await
            .expect("session");
        let root = thread.read_with(cx, |thread, _| {
            runtime_id(thread.session_id())
                .expect("runtime id")
                .to_owned()
        });
        let child = runtime
            .store
            .spawn_child(root, "read-only".into(), 0, true)
            .await
            .expect("child");
        runtime
            .store
            .set_metadata(
                child.clone(),
                serde_json::to_value(NativeRuntimeConfig {
                    profile: Some(profile),
                    ..Default::default()
                })
                .expect("configuration"),
            )
            .await
            .expect("profile");
        runtime
            .store
            .enqueue_text_input(
                child.clone(),
                "child-input".into(),
                "Inspect without changing files.".into(),
            )
            .await
            .expect("input");
        wait_until(|| !provider.pending_completions_for(&model).is_empty()).await;
        let request = provider
            .pending_completions_for(&model)
            .into_iter()
            .next()
            .expect("request");
        assert_eq!(request.thread_id.as_deref(), Some(child.as_str()));
        assert!(request.tools.iter().any(|tool| tool.name == "read_file"));
        for forbidden in [
            "edit_file",
            "write_file",
            "terminal",
            "runtime_spawn",
            "runtime_workspace_acquire",
            "runtime_integrate",
            "runtime_revert",
        ] {
            assert!(
                !request.tools.iter().any(|tool| tool.name == forbidden),
                "read-only child advertised {forbidden}"
            );
        }
        provider.send_text(&model, &request, "Read-only inspection complete.");
        provider.send_event(
            &model,
            &request,
            LanguageModelCompletionEvent::Stop(StopReason::EndTurn),
        );
        provider.end_stream(&model, &request);
        runtime.shutdown().await.expect("shutdown");
    }

    #[test]
    fn runtime_session_namespace_keeps_existing_sessions_distinct() {
        assert_eq!(
            runtime_id(&acp_v2::SessionId::new("runtime:agent")),
            Some("agent")
        );
        assert_eq!(runtime_id(&acp_v2::SessionId::new("legacy-agent")), None);
    }

    #[test]
    fn transcript_projection_preserves_thinking_and_user_roles() {
        let user = serde_json::to_value(InfinityMessage::User {
            content: UserContent::text("request"),
        })
        .unwrap();
        let assistant = serde_json::to_value(InfinityMessage::Assistant {
            content: AssistantContent::text("answer"),
        })
        .unwrap();
        assert_eq!(
            assistant,
            json!({"kind":"assistant","content":{"text":"answer"}})
        );
        let reasoning = serde_json::to_value(InfinityMessage::Assistant {
            content: AssistantContent::Reasoning(Reasoning {
                id: None,
                content: vec![
                    ReasoningContent::Text {
                        text: "thinking ".into(),
                        signature: Some("signed".into()),
                    },
                    ReasoningContent::Summary("summary".into()),
                ],
            }),
        })
        .unwrap();
        assert_eq!(
            history_text(&user).unwrap(),
            Some((false, "request".into()))
        );
        assert_eq!(
            history_text(&assistant).unwrap(),
            Some((false, "answer".into()))
        );
        assert_eq!(
            history_text(&reasoning).unwrap(),
            Some((true, "thinking summary".into()))
        );
        assert_eq!(
            history_text(&json!({"kind":"assistant","content":{"type":"text","text":"answer"}}))
                .unwrap(),
            Some((false, "answer".into()))
        );
        assert_eq!(
            history_text(
                &json!({"kind":"assistant","content":{"type":"reasoning","content":[{"type":"text","text":"reasoning","signature":"signed"}]}})
            ).unwrap(),
            Some((true, "reasoning".into()))
        );
    }

    fn provider_state_block() -> ReasoningContent {
        ReasoningContent::ProviderState {
            namespace: "zed.language-model.v1".into(),
            value: json!({"kind":"stop","reason":"end_turn"}),
        }
    }

    #[test]
    fn transcript_projection_omits_opaque_reasoning() {
        let payload = serde_json::to_value(InfinityMessage::Assistant {
            content: AssistantContent::Reasoning(Reasoning {
                id: None,
                content: vec![
                    provider_state_block(),
                    ReasoningContent::Encrypted("encrypted".into()),
                    ReasoningContent::Redacted {
                        data: "redacted".into(),
                    },
                ],
            }),
        })
        .unwrap();
        assert_eq!(history_text(&payload).unwrap(), None);
    }

    #[test]
    fn replayed_tool_cards_keep_durable_identity_and_error_status() {
        let update = history_tool_update(&json!({"kind":"tool_result","runtime_job_id":"agent:7:call","result":{"id":"call","is_error":true,"content":[{"type":"text","text":"permission denied"}],"output":{"reason":"denied"}}})).expect("tool result");
        let serialized = serde_json::to_value(update).expect("serialized update");
        assert_eq!(serialized.get("toolCallId"), Some(&json!("agent:7:call")));
        assert_eq!(serialized.get("status"), Some(&json!("failed")));
        assert_eq!(
            serialized.get("rawOutput"),
            Some(&json!({"reason":"denied"}))
        );
    }
}
