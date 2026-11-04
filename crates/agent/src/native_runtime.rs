use std::{path::PathBuf, sync::Arc};

use agent_runtime::{
    AgentSnapshot, EmbeddedRuntime, InfinitySlice, RuntimeLimits, RuntimeStore, SliceFactory,
    model::{ModelEntry, ModelExecutionSender, ZedModelProvider, model_execution_channel},
};
use agent_settings::{AgentProfileId, AgentSettings, language_model_to_selection};
use anyhow::{Context as _, Result, anyhow, ensure};
use futures::{FutureExt as _, channel::oneshot, future::BoxFuture};
use gpui::{App, AppContext as _, Context, Entity, Global, Subscription, Task};
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
    response: oneshot::Sender<Result<ResolvedConfiguration>>,
}

struct ResolvedConfiguration {
    model: ModelEntry,
    template: LanguageModelRequest,
    system_prompt: String,
}

struct GlobalNativeRuntime(Entity<NativeRuntime>);

impl Global for GlobalNativeRuntime {}

pub struct NativeRuntime {
    runtime: Option<EmbeddedRuntime>,
    database: Option<(PathBuf, RuntimeLimits)>,
    starting: bool,
    quitting: bool,
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
                while let Ok(request) = configurations.recv().await {
                    futures_lite::future::yield_now().await;
                    if request.response.is_canceled() {
                        continue;
                    }
                    let result = this
                        .read_with(cx, |host, cx| {
                            ensure!(!host.quitting, "the app is shutting down");
                            resolve_configuration(request.agent, request.configuration, cx)
                        })
                        .and_then(|result| result);
                    if request.response.send(result).is_err() {
                        log::debug!("runtime configuration requester detached");
                    }
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
        self.starting = true;
        let configurations = self.configuration_sender.clone();
        let executions = self.execution_sender.clone();
        let database = (path.clone(), limits.clone());
        let (response, receiver) = oneshot::channel();
        self._startup = Some(cx.spawn(async move |this, cx| {
            let result = cx
                .background_spawn(async move {
                    let store = RuntimeStore::open(path, limits)?;
                    let factory = slice_factory(store.clone(), configurations, executions);
                    EmbeddedRuntime::start(store, factory)
                })
                .await;
            let result = this
                .update(cx, |host, _| {
                    host.starting = false;
                    let runtime = result?;
                    if host.quitting {
                        runtime.request_shutdown();
                        return Err(anyhow!("the app shut down during runtime startup"));
                    }
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
) -> SliceFactory {
    Arc::new(move |agent| {
        let store = store.clone();
        let configurations = configurations.clone();
        let executions = executions.clone();
        async move {
            let configuration =
                serde_json::from_value(store.metadata(agent.root_id.clone()).await?)?;
            let (response, receiver) = oneshot::channel();
            configurations
                .send(ConfigurationRequest {
                    agent,
                    configuration,
                    response,
                })
                .await
                .map_err(|_| anyhow!("native configuration host has stopped"))?;
            let resolved = receiver
                .await
                .context("native configuration preparation was interrupted")??;
            Ok(InfinitySlice {
                provider: Arc::new(ZedModelProvider::new(
                    resolved.model.clone(),
                    resolved.template,
                    executions,
                )),
                model: resolved.model,
                tools: Vec::new(),
                system_prompt: resolved.system_prompt,
            })
        }
        .boxed()
    })
}

fn resolve_model_id(id: &str, cx: &App) -> Result<LanguageModel> {
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

fn resolve_configuration(
    agent: AgentSnapshot,
    configuration: NativeRuntimeConfig,
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
    let selection = agent
        .parent_id
        .as_ref()
        .and(settings.subagent_model.as_ref())
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
            "paused"
        );
        assert_eq!(
            restored.metadata(id).await.expect("metadata")["system_prompt"],
            "project-specific instructions"
        );
        restored.shutdown().await.expect("store shutdown");
    }

    #[gpui::test]
    async fn native_runtime_pauses_on_unsupported_provider_state(cx: &mut TestAppContext) {
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
            LanguageModelCompletionEvent::Stop(StopReason::MaxTokens),
        );
        let notice = next_notice(&mut notices).await;
        assert_eq!(notice.agent_id, id);
        assert!(
            notice
                .error
                .as_deref()
                .is_some_and(|error| error.contains("MaxTokens")),
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
