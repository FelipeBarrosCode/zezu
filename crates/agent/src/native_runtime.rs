use std::{collections::HashMap, path::PathBuf, rc::Rc, sync::Arc};

use crate::runtime_tools::{RuntimeToolHost, RuntimeToolPreparation, RuntimeToolResolver};
use acp_thread::AcpThread;
use agent_runtime::{
    AgentSnapshot, EmbeddedRuntime, InfinitySlice, RuntimeLimits, RuntimeStore, SliceFactory,
    model::{
        ModelEntry, ModelExecutionSender, ToolDefinition, ZedModelProvider, model_execution_channel,
    },
    resource::{ResourceLimits, RuntimeResources},
};
use agent_settings::{AgentProfileId, AgentSettings, language_model_to_selection};
use anyhow::{Context as _, Result, anyhow, ensure};
use futures::{FutureExt as _, channel::oneshot, future::BoxFuture};
use gpui::{App, AppContext as _, Context, Entity, Global, Subscription, Task, WeakEntity};
use language_model::{
    CompletionIntent, LanguageModel, LanguageModelCompletionError, LanguageModelId,
    LanguageModelProviderId, LanguageModelRegistry, LanguageModelRequest,
};
use serde::{Deserialize, Serialize};
use settings::{LanguageModelSelection, Settings as _};
use util::ResultExt as _;

const PREPARATION_CAPACITY: usize = 256;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct NativeRuntimeConfig {
    pub model: Option<LanguageModelSelection>,
    pub profile: Option<AgentProfileId>,
    /// The caller assembles project instructions and skills using Zed's existing loaders.
    pub system_prompt: String,
}

struct ConfigurationRequest {
    agent: AgentSnapshot,
    configuration: NativeRuntimeConfig,
    explicit_child_model: bool,
    response: oneshot::Sender<Result<ResolvedConfiguration>>,
}

pub type RuntimeContextLoader =
    Rc<dyn Fn(NativeRuntimeConfig, AgentSnapshot, &mut App) -> Task<Result<NativeRuntimeConfig>>>;

struct ResolvedConfiguration {
    model: ModelEntry,
    template: LanguageModelRequest,
    system_prompt: String,
    tools: Vec<ToolDefinition>,
    profile_id: String,
}

struct RuntimeRunBinding {
    resolver: RuntimeToolResolver,
    preparation: Option<RuntimeToolPreparation>,
    context_loader: Option<RuntimeContextLoader>,
}

struct GlobalNativeRuntime(Entity<NativeRuntime>);

impl Global for GlobalNativeRuntime {}

pub struct NativeRuntime {
    runtime: Option<EmbeddedRuntime>,
    database: Option<(PathBuf, RuntimeLimits)>,
    starting: bool,
    quitting: bool,
    bindings: HashMap<String, RuntimeRunBinding>,
    views: HashMap<String, WeakEntity<AcpThread>>,
    tool_host: Option<Entity<RuntimeToolHost>>,
    resources: Option<RuntimeResources>,
    compactions: HashMap<String, Task<()>>,
    configuration_sender: async_channel::Sender<ConfigurationRequest>,
    execution_sender: ModelExecutionSender,
    _preparation_tasks: Vec<Task<()>>,
    _startup: Option<Task<()>>,
    _quit_subscription: Subscription,
}

impl NativeRuntime {
    pub fn global(cx: &mut App) -> Result<Entity<Self>> {
        if let Some(global) = cx.try_global::<GlobalNativeRuntime>() {
            return Ok(global.0.clone());
        }
        let (execution_sender, executions) = model_execution_channel(PREPARATION_CAPACITY)?;
        let (configuration_sender, configurations) =
            async_channel::bounded::<ConfigurationRequest>(PREPARATION_CAPACITY);
        let host = cx.new(|cx: &mut Context<Self>| {
            let configuration_task = cx.spawn(async move |this, cx| {
                while let Ok(mut request) = configurations.recv().await {
                    futures_lite::future::yield_now().await;
                    if request.response.is_canceled() { continue; }
                    let this = this.clone();
                    // Only admitted slices send requests; independent preparation must not block other roots during compaction or filesystem scans.
                    cx.spawn(async move |cx| {
                        let result = async {
                            let (loader, runtime, resources) = this.read_with(cx, |host, _| (
                                host.bindings.get(&request.agent.root_id).and_then(|binding| binding.context_loader.clone()),
                                host.runtime.clone(), host.resources.clone(),
                            ))?;
                            if let Some(loader) = loader {
                                let configuration = std::mem::take(&mut request.configuration);
                                request.configuration = cx.update(|cx| loader(configuration, request.agent.clone(), cx)).await?;
                            }
                            ensure!(!request.response.is_canceled(), "runtime configuration requester detached");
                            if let Some(runtime) = runtime {
                                let mut automatic_configuration = request.configuration.clone();
                                if request.agent.parent_id.is_some() && !request.explicit_child_model {
                                    let subagent_model = cx.update(|cx| AgentSettings::get_global(cx).subagent_model.clone());
                                    if subagent_model.is_some() { automatic_configuration.model = subagent_model; }
                                }
                                cx.update(|cx| crate::runtime_compaction::maybe_compact_agent(runtime, request.agent.id.clone(), request.agent.generation, resources, automatic_configuration, cx)).await?;
                            }
                            this.update(cx, |host, cx| {
                                ensure!(!host.quitting, "the app is shutting down");
                                let root_id = request.agent.root_id.clone();
                                let mut resolved = resolve_configuration_with_selection(request.agent.clone(), std::mem::take(&mut request.configuration), request.explicit_child_model, cx)?;
                                if let Some(binding) = host.bindings.get(&root_id) {
                                    let job = agent_runtime::Job {
                                        id: "runtime-tool-catalog".into(), agent_id: request.agent.id.clone(), generation: request.agent.generation, state: agent_runtime::JobState::Queued,
                                        payload: serde_json::json!({"runtime_root_id": root_id, "runtime_model_id": resolved.model.model_id, "runtime_profile_id": resolved.profile_id}), result: None,
                                    };
                                    resolved.tools = (binding.resolver)(&job, &request.agent, cx)?.definitions(cx);
                                }
                                Ok(resolved)
                            })?
                        }.await;
                        if request.response.send(result).is_err() { log::debug!("runtime configuration requester detached"); }
                    }).detach();
                }
            });
            let execution_task = cx.spawn(async move |this, cx| {
                while let Ok(request) = executions.recv().await {
                    futures_lite::future::yield_now().await;
                    if request.is_cancelled() {
                        continue;
                    }
                    let model = this
                        .read_with(cx, |host, cx| {
                            ensure!(!host.quitting, "the app is shutting down");
                            ensure!(
                                AgentSettings::get_global(cx).enabled(cx),
                                "AI features are disabled in Zed settings"
                            );
                            let model = resolve_model_id(&request.model_id, cx)?;
                            let provider = LanguageModelRegistry::read_global(cx)
                                .provider_for_model(&model)?;
                            Ok((model, provider))
                        })
                        .and_then(|result| result);
                    request.prepare_with(|request| {
                        model
                            .map(|(model, provider)| {
                                provider.prepare_completion(&model, request, cx)
                            })
                            .map_err(LanguageModelCompletionError::from)
                    });
                }
            });
            Self {
                runtime: None,
                database: None,
                starting: false,
                quitting: false,
                bindings: HashMap::new(),
                views: HashMap::new(),
                tool_host: None,
                resources: None,
                compactions: HashMap::new(),
                configuration_sender,
                execution_sender,
                _preparation_tasks: vec![configuration_task, execution_task],
                _startup: None,
                _quit_subscription: cx.on_app_quit(Self::shutdown_on_quit),
            }
        });
        cx.set_global(GlobalNativeRuntime(host.clone()));
        Ok(host)
    }

    pub fn enabled(cx: &App) -> bool {
        AgentSettings::get_global(cx)
            .runtime
            .enabled
            .unwrap_or(true)
    }

    pub fn bind_run(
        &mut self,
        root_id: String,
        resolver: RuntimeToolResolver,
        definitions: Vec<ToolDefinition>,
        _: &mut Context<Self>,
    ) -> Result<()> {
        ensure!(!self.quitting, "the app is shutting down");
        ensure!(!root_id.is_empty(), "runtime root identity cannot be empty");
        let mut names = std::collections::HashSet::new();
        for definition in &definitions {
            ensure!(
                names.insert(definition.name.clone()),
                "duplicate runtime tool name: {}",
                definition.name
            );
        }
        self.bindings.insert(
            root_id,
            RuntimeRunBinding {
                resolver,
                preparation: None,
                context_loader: None,
            },
        );
        Ok(())
    }

    pub fn bind_run_preparation(
        &mut self,
        root_id: String,
        preparation: RuntimeToolPreparation,
    ) -> Result<()> {
        self.bindings
            .get_mut(&root_id)
            .context("bind the runtime project before installing its workspace preparation")?
            .preparation = Some(preparation);
        Ok(())
    }

    pub fn bind_run_context(
        &mut self,
        root_id: String,
        loader: RuntimeContextLoader,
    ) -> Result<()> {
        self.bindings
            .get_mut(&root_id)
            .context("bind the runtime project before installing its context loader")?
            .context_loader = Some(loader);
        Ok(())
    }

    pub fn compact(&mut self, agent_id: String, cx: &mut Context<Self>) -> Task<Result<()>> {
        if self.quitting {
            return Task::ready(Err(anyhow!("the app is shutting down")));
        }
        let Some(runtime) = self.runtime.clone() else {
            return Task::ready(Err(anyhow!("start the native runtime before compacting")));
        };
        if self.compactions.contains_key(&agent_id) {
            return Task::ready(Err(anyhow!("this agent is already compacting")));
        }
        let resources = self.resources.clone();
        let (response, receiver) = oneshot::channel();
        let identity = agent_id.clone();
        let task = cx.spawn(async move |this, cx| {
            let result = async {
                let snapshot = runtime.store.snapshot(identity.clone()).await?;
                let configuration: NativeRuntimeConfig = serde_json::from_value(
                    runtime.store.metadata(snapshot.root_id.clone()).await?,
                )?;
                let loader = this.read_with(cx, |host, _| {
                    host.bindings
                        .get(&snapshot.root_id)
                        .and_then(|binding| binding.context_loader.clone())
                })?;
                let configuration = if let Some(loader) = loader {
                    cx.update(|cx| loader(configuration, snapshot, cx)).await?
                } else {
                    configuration
                };
                cx.update(|cx| {
                    crate::runtime_compaction::compact_with_configuration(
                        runtime,
                        identity.clone(),
                        resources,
                        crate::runtime_compaction::CompactionMode::Manual,
                        Some(configuration),
                        cx,
                    )
                })
                .await
            }
            .await;
            if let Err(error) = &result {
                log::error!("native context compaction failed: {error:#}");
            }
            if response.send(result).is_err() {
                log::debug!("compaction requester detached");
            }
            if let Err(error) = this.update(cx, |host, _| {
                host.compactions.remove(&identity);
            }) {
                log::debug!("compaction owner detached: {error}");
            }
        });
        self.compactions.insert(agent_id, task);
        cx.spawn(async move |_, _| receiver.await.context("native compaction owner detached")?)
    }

    pub fn bind_view(&mut self, agent_id: String, view: WeakEntity<AcpThread>) {
        self.views.retain(|_, view| view.upgrade().is_some());
        self.views.insert(agent_id, view);
    }

    pub fn resources(&self) -> Option<RuntimeResources> {
        self.resources.clone()
    }

    pub fn start_default(&mut self, cx: &mut Context<Self>) -> Task<Result<EmbeddedRuntime>> {
        if !Self::enabled(cx) {
            return Task::ready(Err(anyhow!("the native concurrent runtime is disabled")));
        }
        let configured = &AgentSettings::get_global(cx).runtime;
        let defaults = RuntimeLimits::default();
        let maximum = configured.max_agents.unwrap_or(defaults.max_agents_per_run);
        let limits = RuntimeLimits {
            max_agents_per_run: maximum,
            max_active_agents: configured
                .max_active_agents
                .unwrap_or(defaults.max_active_agents)
                .min(maximum),
            command_queue_capacity: configured
                .max_queued_jobs
                .unwrap_or(defaults.command_queue_capacity),
            ..defaults
        };
        self.start_database(paths::data_dir().join("agent-runtime.sqlite"), limits, cx)
    }

    pub fn active(&self) -> Option<EmbeddedRuntime> {
        self.runtime.clone()
    }

    pub fn start_database(
        &mut self,
        path: PathBuf,
        limits: RuntimeLimits,
        cx: &mut Context<Self>,
    ) -> Task<Result<EmbeddedRuntime>> {
        if self.quitting {
            return Task::ready(Err(anyhow!("the app is shutting down")));
        }
        if let Some(runtime) = &self.runtime {
            return Task::ready(if self.database.as_ref() == Some(&(path, limits)) {
                Ok(runtime.clone())
            } else {
                Err(anyhow!(
                    "the native runtime already owns another database or configuration"
                ))
            });
        }
        if self.starting {
            return Task::ready(Err(anyhow!("the native runtime is already starting")));
        }
        if let Err(error) = limits.validate() {
            return Task::ready(Err(error));
        }
        let configured = &AgentSettings::get_global(cx).runtime;
        let resource_defaults = ResourceLimits::default();
        let resources = match RuntimeResources::new(ResourceLimits {
            max_agents: limits.max_agents_per_run,
            max_active_agents: limits.max_active_agents.min(limits.max_agents_per_run),
            max_tool_jobs: configured
                .max_tool_jobs
                .unwrap_or(resource_defaults.max_tool_jobs),
            max_workspaces: configured
                .max_workspaces
                .unwrap_or(resource_defaults.max_workspaces),
            max_queued_jobs: limits.command_queue_capacity,
            max_memory_bytes: configured
                .max_memory_bytes
                .unwrap_or(resource_defaults.max_memory_bytes),
        }) {
            Ok(resources) => resources,
            Err(error) => return Task::ready(Err(error)),
        };
        self.starting = true;
        let configurations = self.configuration_sender.clone();
        let executions = self.execution_sender.clone();
        let database = (path.clone(), limits.clone());
        let (response, receiver) = oneshot::channel();
        self._startup = Some(cx.spawn(async move |this, cx| {
            let worker_resources = resources.clone();
            let result = cx
                .background_spawn(async move {
                    let store = RuntimeStore::open(path, limits)?;
                    let factory =
                        slice_factory(store.clone(), configurations, executions, worker_resources);
                    EmbeddedRuntime::start(store, factory)
                })
                .await;
            let result = this
                .update(cx, |host, cx| {
                    host.starting = false;
                    let runtime = result?;
                    if host.quitting {
                        runtime.request_shutdown();
                        return Err(anyhow!("the app shut down during runtime startup"));
                    }
                    let resolver_owner = cx.entity().downgrade();
                    let resolver: RuntimeToolResolver = Rc::new(move |job, snapshot, cx| {
                        let root = job
                            .payload
                            .get("runtime_root_id")
                            .and_then(serde_json::Value::as_str)
                            .unwrap_or(&snapshot.root_id);
                        ensure!(
                            root == snapshot.root_id,
                            "tool job root identity does not match its agent"
                        );
                        let resolver = resolver_owner
                            .read_with(cx, |host, _| {
                                host.bindings
                                    .get(root)
                                    .map(|binding| binding.resolver.clone())
                            })?
                            .context(
                                "restore the runtime project binding before resuming this run",
                            )?;
                        resolver(job, snapshot, cx)
                    });
                    let event_owner = cx.entity().downgrade();
                    let sink = Rc::new(
                        move |_job: &agent_runtime::Job,
                              snapshot: &AgentSnapshot,
                              event: &mut Option<crate::ThreadEvent>,
                              cx: &mut App| {
                            let view = event_owner.update(cx, |host, _| {
                                host.views.retain(|_, view| view.upgrade().is_some());
                                host.views
                                    .get(&snapshot.id)
                                    .or_else(|| host.views.get(&snapshot.root_id))
                                    .cloned()
                            })?;
                            if let Some(view) = view {
                                crate::runtime_connection::project_runtime_tool_event(
                                    &view, event, cx,
                                )?;
                            }
                            Ok(())
                        },
                    );
                    let tool_host = RuntimeToolHost::start(
                        runtime.store.clone(),
                        resolver,
                        sink,
                        resources.limits().max_tool_jobs,
                        cx,
                    )?;
                    let preparation_owner = cx.entity().downgrade();
                    let preparation: RuntimeToolPreparation = Rc::new(move |job, snapshot, cx| {
                        match preparation_owner.read_with(cx, |host, _| {
                            host.bindings
                                .get(&snapshot.root_id)
                                .and_then(|binding| binding.preparation.clone())
                        }) {
                            Ok(Some(prepare)) => prepare(job, snapshot, cx),
                            Ok(None) => Task::ready(Ok(())),
                            Err(error) => Task::ready(Err(error)),
                        }
                    });
                    tool_host.update(cx, |host, _| {
                        host.set_resources(resources.clone());
                        host.set_preparation(preparation);
                    });
                    host.tool_host = Some(tool_host);
                    host.resources = Some(resources);
                    host.database = Some(database);
                    host.runtime = Some(runtime.clone());
                    Ok(runtime)
                })
                .and_then(|result| result);
            if let Err(error) = &result {
                log::error!("native runtime startup failed: {error:#}");
            }
            if response.send(result).is_err() {
                log::debug!("runtime startup requester detached");
            }
        }));
        cx.spawn(async move |_, _| {
            receiver
                .await
                .context("native runtime host disappeared during startup")?
        })
    }

    pub fn submit_text(
        &self,
        configuration: NativeRuntimeConfig,
        text: String,
    ) -> BoxFuture<'static, Result<String>> {
        let runtime = self.runtime.clone();
        let quitting = self.quitting;
        async move {
            ensure!(!quitting, "the app is shutting down");
            let runtime = runtime.context("start the native runtime before submitting work")?;
            let id = uuid::Uuid::new_v4().to_string();
            runtime
                .store
                .create_root(id.clone(), serde_json::to_value(configuration)?)
                .await?;
            runtime
                .store
                .enqueue_text_input(id.clone(), uuid::Uuid::new_v4().to_string(), text)
                .await?;
            Ok(id)
        }
        .boxed()
    }

    fn shutdown_on_quit(&mut self, _: &mut Context<Self>) -> impl Future<Output = ()> + use<> {
        self.quitting = true;
        self.compactions.clear();
        let runtime = self.runtime.clone();
        if let Some(runtime) = &runtime {
            // GPUI's quit deadline may expire before its asynchronous observers finish.
            runtime.request_shutdown();
        }
        async move {
            if let Some(runtime) = runtime {
                runtime.shutdown().await.log_err();
            }
        }
    }
}

fn slice_factory(
    store: RuntimeStore,
    configurations: async_channel::Sender<ConfigurationRequest>,
    executions: ModelExecutionSender,
    resources: RuntimeResources,
) -> SliceFactory {
    Arc::new(move |agent| {
        let store = store.clone();
        let configurations = configurations.clone();
        let executions = executions.clone();
        let resources = resources.clone();
        async move {
            let mut configuration: NativeRuntimeConfig =
                serde_json::from_value(store.metadata(agent.root_id.clone()).await?)?;
            let mut explicit_child_model = false;
            if agent.parent_id.is_some() {
                let child_metadata = store.metadata(agent.id.clone()).await?;
                if let Some(model) = child_metadata.get("model").filter(|model| !model.is_null()) {
                    configuration.model = Some(serde_json::from_value(model.clone())?);
                    explicit_child_model = true;
                }
                if let Some(profile) = child_metadata
                    .get("profile")
                    .filter(|profile| !profile.is_null())
                {
                    configuration.profile = Some(serde_json::from_value(profile.clone())?);
                }
            }
            let agent_id = agent.id.clone();
            let (response, receiver) = oneshot::channel();
            configurations
                .send(ConfigurationRequest {
                    agent,
                    configuration,
                    explicit_child_model,
                    response,
                })
                .await
                .map_err(|_| anyhow!("native configuration host has stopped"))?;
            let resolved = receiver
                .await
                .context("native configuration preparation was interrupted")??;
            Ok(InfinitySlice {
                provider: Arc::new(
                    ZedModelProvider::new(resolved.model.clone(), resolved.template, executions)
                        .with_state(store.clone(), agent_id)
                        .with_resources(resources),
                ),
                model: resolved.model,
                tools: resolved.tools,
                system_prompt: resolved.system_prompt,
            })
        }
        .boxed()
    })
}

pub(crate) fn resolve_model_id(id: &str, cx: &App) -> Result<LanguageModel> {
    let (provider, model) = id
        .split_once('/')
        .context("expected a provider/model identifier")?;
    LanguageModelRegistry::read_global(cx)
        .provider(&LanguageModelProviderId(provider.to_owned().into()))
        .context("the configured model provider is unavailable")?
        .provided_models(cx)
        .into_iter()
        .find(|candidate| candidate.id == LanguageModelId(model.to_owned().into()))
        .with_context(|| format!("the configured model is unavailable: {id}"))
}

#[cfg(test)]
fn resolve_configuration(
    agent: AgentSnapshot,
    configuration: NativeRuntimeConfig,
    cx: &App,
) -> Result<ResolvedConfiguration> {
    resolve_configuration_with_selection(agent, configuration, false, cx)
}

fn resolve_configuration_with_selection(
    agent: AgentSnapshot,
    configuration: NativeRuntimeConfig,
    explicit_child_model: bool,
    cx: &App,
) -> Result<ResolvedConfiguration> {
    let settings = AgentSettings::get_global(cx);
    ensure!(
        settings.enabled(cx),
        "AI features are disabled in Zed settings"
    );
    let profile_id = configuration
        .profile
        .as_ref()
        .unwrap_or(&settings.default_profile);
    let profile = settings
        .profiles
        .get(profile_id)
        .with_context(|| format!("agent profile is unavailable: {profile_id}"))?;
    let selection = if explicit_child_model {
        configuration.model.as_ref()
    } else {
        agent
            .parent_id
            .as_ref()
            .and(settings.subagent_model.as_ref())
    }
    .or(configuration.model.as_ref())
    .or(profile.default_model.as_ref());
    let model = match selection {
        Some(selection) => {
            resolve_model_id(&format!("{}/{}", selection.provider.0, selection.model), cx)?
        }
        None => LanguageModelRegistry::read_global(cx)
            .default_model()
            .context("select a model in Zed before starting an agent")?,
    };
    let override_selection = selection.or(settings.default_model.as_ref().filter(|selection| {
        selection.provider.0 == model.provider_id.0.as_ref()
            && selection.model == model.id.0.as_ref()
    }));
    let selection = language_model_to_selection(&model, override_selection);
    let template = LanguageModelRequest {
        thread_id: Some(agent.id.clone()),
        prompt_id: Some(uuid::Uuid::new_v4().to_string()),
        intent: Some(CompletionIntent::UserPrompt),
        temperature: AgentSettings::temperature_for_model(&model, cx),
        thinking_allowed: selection.enable_thinking
            || !model.supports_disabling_thinking_at_effort(selection.effort.as_deref()),
        thinking_effort: selection.effort,
        speed: selection.speed,
        max_output_tokens: model.max_output_tokens(),
        ..Default::default()
    };
    Ok(ResolvedConfiguration {
        model: ModelEntry {
            model_id: format!("{}/{}", model.provider_id.0, model.id.0),
            display_name: model.name.0.to_string(),
            context_window: usize::try_from(model.max_token_count())?,
            max_output_tokens: model.max_output_tokens(),
            supports_image_input: model.supports_images(),
        },
        template,
        system_prompt: configuration.system_prompt,
        tools: Vec::new(),
        profile_id: profile_id.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::future::{Either, select};
    use gpui::TestAppContext;
    use language_model::{
        LanguageModelCompletionEvent, StopReason, fake_provider::FakeLanguageModelProvider,
    };
    use std::time::{Duration, Instant};

    struct FailureTool;

    impl crate::AgentTool for FailureTool {
        type Input = serde_json::Value;
        type Output = String;
        const NAME: &'static str = "runtime_test_failure";
        fn kind() -> agent_client_protocol::v2::schema::ToolKind {
            agent_client_protocol::v2::schema::ToolKind::Read
        }
        fn initial_title(
            &self,
            _: Result<Self::Input, serde_json::Value>,
            _: &mut App,
        ) -> gpui::SharedString {
            "Exercise failed tool result".into()
        }
        fn run(
            self: Arc<Self>,
            _: crate::ToolInput<Self::Input>,
            _: crate::ToolCallEventStream,
            _: &mut App,
        ) -> Task<Result<String, String>> {
            Task::ready(Err("intentional tool failure".into()))
        }
    }

    #[gpui::test]
    async fn native_tool_failure_round_preserves_configuration_and_result_flags(
        cx: &mut TestAppContext,
    ) {
        use crate::AgentTool as _;
        cx.executor().allow_parking();
        let provider = crate::tests::init_test(cx);
        let model = provider.model("fake");
        let fs = fs::FakeFs::new(cx.executor());
        let project = project::Project::test(fs, [], cx).await;
        let directory = tempfile::tempdir().expect("directory");
        let host = cx.update(|cx| NativeRuntime::global(cx).expect("host"));
        let runtime = host
            .update(cx, |host, cx| {
                host.start_database(
                    directory.path().join("native.sqlite"),
                    RuntimeLimits::default(),
                    cx,
                )
            })
            .await
            .expect("runtime");
        let mut notices = runtime.subscribe();
        let id = "failed-tool-run".to_owned();
        let configuration = serde_json::to_value(NativeRuntimeConfig {
            system_prompt: "Retain these project rules across tool results.".into(),
            ..Default::default()
        })
        .expect("configuration");
        runtime
            .store
            .create_root(id.clone(), configuration.clone())
            .await
            .expect("root");
        host.update(cx, |host, cx| {
            let provider_id = model.provider_id();
            let resolver: RuntimeToolResolver = Rc::new(move |_, _, cx| {
                Ok(crate::runtime_tools::RuntimeToolContext::new(
                    project.clone(),
                    collections::BTreeMap::from([(
                        "runtime_test_failure".into(),
                        FailureTool.erase(),
                    )]),
                    provider_id.clone(),
                    cx,
                ))
            });
            host.bind_run(id.clone(), resolver, Vec::new(), cx)
                .expect("binding");
        });
        runtime
            .store
            .enqueue_text_input(
                id.clone(),
                "user".into(),
                "Run a tool and report its failure.".into(),
            )
            .await
            .expect("input");
        let first = pending_request(&provider, &model).await;
        provider.send_event(
            &model,
            &first,
            LanguageModelCompletionEvent::ToolUse(language_model::LanguageModelToolUse {
                id: "failed-call".into(),
                name: "runtime_test_failure".into(),
                raw_input: "{}".into(),
                input: language_model::LanguageModelToolUseInput::Json(serde_json::json!({})),
                is_input_complete: true,
                thought_signature: None,
            }),
        );
        provider.send_event(
            &model,
            &first,
            LanguageModelCompletionEvent::Stop(StopReason::ToolUse),
        );
        provider.end_stream(&model, &first);
        assert!(next_notice(&mut notices).await.error.is_none());
        let second = pending_request(&provider, &model).await;
        let result = second
            .messages
            .iter()
            .flat_map(|message| &message.content)
            .find_map(|content| match content {
                language_model::MessageContent::ToolResult(result) => Some(result),
                _ => None,
            })
            .expect("native tool result");
        assert!(result.is_error);
        assert_eq!(
            result.output,
            Some(serde_json::json!("intentional tool failure"))
        );
        assert_eq!(
            runtime
                .store
                .metadata(id.clone())
                .await
                .expect("root metadata"),
            configuration
        );
        provider.send_text(&model, &second, "The tool failed as expected.");
        provider.send_event(
            &model,
            &second,
            LanguageModelCompletionEvent::Stop(StopReason::EndTurn),
        );
        provider.end_stream(&model, &second);
        assert!(next_notice(&mut notices).await.error.is_none());
        assert_eq!(
            runtime
                .store
                .metadata(id)
                .await
                .expect("metadata after completed round"),
            configuration
        );
        runtime.shutdown().await.expect("shutdown");
    }

    async fn pending_request(
        provider: &FakeLanguageModelProvider,
        model: &LanguageModel,
    ) -> LanguageModelRequest {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(request) = provider.pending_completions_for(model).into_iter().next() {
                return request;
            }
            assert!(
                Instant::now() < deadline,
                "runtime did not prepare a completion"
            );
            // The embedded worker uses OS threads, outside the simulated GPUI clock.
            async_io::Timer::after(Duration::from_millis(2)).await;
        }
    }

    async fn next_notice(
        notices: &mut agent_runtime::RuntimeNoticeReceiver,
    ) -> agent_runtime::RuntimeNotice {
        let response = notices.recv();
        futures::pin_mut!(response);
        match select(
            response,
            Box::pin(async_io::Timer::after(Duration::from_secs(10))),
        )
        .await
        {
            Either::Left((notice, _)) => notice.expect("runtime notice"),
            Either::Right(_) => panic!("runtime did not finish the slice"),
        }
    }

    fn selection(provider: &FakeLanguageModelProvider, model: &str) -> LanguageModelSelection {
        language_model_to_selection(&provider.model(model), None)
    }

    #[gpui::test]
    async fn native_runtime_uses_zed_models_and_survives_detached_handles(cx: &mut TestAppContext) {
        cx.executor().allow_parking();
        let provider = crate::tests::init_test(cx);
        let parent_model = provider.update_model("parent", |model| {
            model.max_output_tokens = Some(700);
        });
        let child_model = provider.model("worker");
        cx.update(|cx| {
            let mut settings = AgentSettings::get_global(cx).clone();
            settings.subagent_model = Some(selection(&provider, "worker"));
            AgentSettings::override_global(settings, cx);
        });
        let directory = tempfile::tempdir().expect("directory");
        let path = directory.path().join("native.sqlite");
        let host = cx.update(|cx| NativeRuntime::global(cx).expect("host"));
        let runtime = host
            .update(cx, |host, cx| {
                host.start_database(path.clone(), RuntimeLimits::default(), cx)
            })
            .await
            .expect("runtime");
        let mut notices = runtime.subscribe();
        let configuration = NativeRuntimeConfig {
            model: Some(selection(&provider, "parent")),
            system_prompt: "project-specific instructions".into(),
            ..Default::default()
        };
        let id = host
            .read_with(cx, |host, _| {
                host.submit_text(configuration, "inspect the project".into())
            })
            .await
            .expect("submission");
        let entity_id = host.entity_id();
        drop(host);
        let request = pending_request(&provider, &parent_model).await;
        assert_eq!(request.thread_id.as_deref(), Some(id.as_str()));
        assert_eq!(request.max_output_tokens, Some(700));
        assert!(request.messages.iter().any(|message| {
            message
                .string_contents()
                .contains("project-specific instructions")
        }));
        provider.send_text(&parent_model, &request, "parent response");
        provider.send_event(
            &parent_model,
            &request,
            LanguageModelCompletionEvent::Stop(StopReason::EndTurn),
        );
        provider.end_stream(&parent_model, &request);
        let notice = next_notice(&mut notices).await;
        assert_eq!(notice.agent_id, id);
        assert!(notice.error.is_none(), "{:?}", notice.error);
        let host = cx.update(|cx| NativeRuntime::global(cx).expect("global host"));
        assert_eq!(entity_id, host.entity_id());
        assert!(
            host.update(cx, |host, cx| host.start_database(
                path.clone(),
                RuntimeLimits::default(),
                cx
            ))
            .await
            .is_ok()
        );
        assert!(
            host.update(cx, |host, cx| host.start_database(
                path.with_file_name("other.sqlite"),
                RuntimeLimits::default(),
                cx
            ))
            .await
            .is_err()
        );
        let child = runtime
            .store
            .spawn_child(id.clone(), "worker".into(), 0, true)
            .await
            .expect("child");
        runtime
            .store
            .enqueue_text_input(child.clone(), "child-input".into(), "parallel task".into())
            .await
            .expect("enqueue");
        let request = pending_request(&provider, &child_model).await;
        assert_eq!(request.thread_id.as_deref(), Some(child.as_str()));
        provider.send_event(
            &child_model,
            &request,
            LanguageModelCompletionEvent::Stop(StopReason::EndTurn),
        );
        provider.end_stream(&child_model, &request);
        assert!(next_notice(&mut notices).await.error.is_none());
        runtime.shutdown().await.expect("shutdown");
        let restored = RuntimeStore::open(path, RuntimeLimits::default()).expect("reopen");
        assert_eq!(
            restored.snapshot(id.clone()).await.expect("root").state,
            "paused"
        );
        assert_eq!(
            restored.snapshot(child).await.expect("child").state,
            "completed"
        );
        assert_eq!(
            restored.metadata(id).await.expect("metadata")["system_prompt"],
            "project-specific instructions"
        );
        restored.shutdown().await.expect("store shutdown");
    }

    #[gpui::test]
    async fn native_runtime_pauses_on_malformed_provider_output(cx: &mut TestAppContext) {
        cx.executor().allow_parking();
        let provider = crate::tests::init_test(cx);
        let model = provider.model("fake");
        let directory = tempfile::tempdir().expect("directory");
        let host = cx.update(|cx| NativeRuntime::global(cx).expect("host"));
        let runtime = host
            .update(cx, |host, cx| {
                host.start_database(
                    directory.path().join("native.sqlite"),
                    RuntimeLimits::default(),
                    cx,
                )
            })
            .await
            .expect("runtime");
        let mut notices = runtime.subscribe();
        let id = host
            .read_with(cx, |host, _| {
                host.submit_text(NativeRuntimeConfig::default(), "start".into())
            })
            .await
            .expect("submit");
        let request = pending_request(&provider, &model).await;
        provider.send_text(&model, &request, "partial answer");
        provider.send_event(
            &model,
            &request,
            LanguageModelCompletionEvent::ToolUseJsonParseError {
                id: "invalid-call".into(),
                tool_name: "read_file".into(),
                raw_input: "{".into(),
                json_parse_error: "incomplete JSON input".into(),
            },
        );
        let notice = next_notice(&mut notices).await;
        assert_eq!(notice.agent_id, id);
        assert!(
            notice
                .error
                .as_deref()
                .is_some_and(|error| error.contains("invalid tool arguments")),
            "{:?}",
            notice.error
        );
        assert_eq!(
            runtime.store.snapshot(id).await.expect("state").state,
            "paused"
        );
        assert!(provider.is_stream_closed(&model, &request));
        provider.end_stream(&model, &request);
        runtime.shutdown().await.expect("shutdown");
    }

    #[gpui::test]
    async fn native_runtime_quit_cancels_provider_and_reopens_paused(cx: &mut TestAppContext) {
        cx.executor().allow_parking();
        let provider = crate::tests::init_test(cx);
        let model = provider.model("fake");
        let directory = tempfile::tempdir().expect("directory");
        let path = directory.path().join("native.sqlite");
        let host = cx.update(|cx| NativeRuntime::global(cx).expect("host"));
        let runtime = host
            .update(cx, |host, cx| {
                host.start_database(path.clone(), RuntimeLimits::default(), cx)
            })
            .await
            .expect("runtime");
        let id = host
            .read_with(cx, |host, _| {
                host.submit_text(NativeRuntimeConfig::default(), "in progress".into())
            })
            .await
            .expect("submit");
        let request = pending_request(&provider, &model).await;
        provider.send_text(&model, &request, "uncommitted partial response");
        cx.update(|cx| cx.shutdown());
        runtime.shutdown().await.expect("join");
        let deadline = Instant::now() + Duration::from_secs(10);
        let restored = loop {
            match RuntimeStore::open(&path, RuntimeLimits::default()) {
                Ok(store) => break store,
                Err(error) => {
                    assert!(
                        Instant::now() < deadline,
                        "database owner did not stop: {error:#}"
                    );
                    async_io::Timer::after(Duration::from_millis(2)).await;
                }
            }
        };
        assert!(provider.is_stream_closed(&model, &request));
        assert_eq!(restored.snapshot(id).await.expect("state").state, "paused");
        restored.shutdown().await.expect("store shutdown");
    }

    #[gpui::test]
    fn explicit_child_model_wins_configured_subagent_model(cx: &mut TestAppContext) {
        let provider = crate::tests::init_test(cx);
        cx.update(|cx| {
            let mut settings = AgentSettings::get_global(cx).clone();
            settings.subagent_model = Some(selection(&provider, "default-worker"));
            AgentSettings::override_global(settings, cx);
            let configuration = NativeRuntimeConfig {
                model: Some(selection(&provider, "selected-worker")),
                ..Default::default()
            };
            let agent = AgentSnapshot {
                id: "child".into(),
                root_id: "root".into(),
                parent_id: Some("root".into()),
                state: "running".into(),
                generation: 0,
            };
            let inherited = resolve_configuration_with_selection(
                agent.clone(),
                configuration.clone(),
                false,
                cx,
            )
            .expect("inherited model");
            let explicit = resolve_configuration_with_selection(agent, configuration, true, cx)
                .expect("selected model");
            assert!(inherited.model.model_id.ends_with("/default-worker"));
            assert!(explicit.model.model_id.ends_with("/selected-worker"));
        });
    }

    #[gpui::test]
    fn native_runtime_rejects_missing_models_and_uses_profile_configuration(
        cx: &mut TestAppContext,
    ) {
        let provider = crate::tests::init_test(cx);
        cx.update(|cx| {
            let mut settings = AgentSettings::get_global(cx).clone();
            let profile_id = settings.default_profile.clone();
            settings
                .profiles
                .get_mut(&profile_id)
                .expect("default profile")
                .default_model = Some(selection(&provider, "profile"));
            AgentSettings::override_global(settings, cx);
            let agent = AgentSnapshot {
                id: "agent".into(),
                root_id: "agent".into(),
                parent_id: None,
                state: "running".into(),
                generation: 0,
            };
            assert!(
                resolve_configuration(
                    agent.clone(),
                    NativeRuntimeConfig {
                        model: Some(LanguageModelSelection {
                            model: "missing".into(),
                            ..selection(&provider, "fake")
                        }),
                        ..Default::default()
                    },
                    cx
                )
                .is_err()
            );
            let resolved = resolve_configuration(agent.clone(), NativeRuntimeConfig::default(), cx)
                .expect("profile");
            assert!(resolved.model.model_id.ends_with("/profile"));
            let mut settings = AgentSettings::get_global(cx).clone();
            settings.enabled = false;
            AgentSettings::override_global(settings, cx);
            assert!(resolve_configuration(agent, NativeRuntimeConfig::default(), cx).is_err());
        });
    }
}
