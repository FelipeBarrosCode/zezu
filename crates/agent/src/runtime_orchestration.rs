use std::{cell::RefCell, path::PathBuf, rc::Rc};

use agent_runtime::{
    AgentSnapshot, Job, RuntimeStore,
    integration::{GitIntegration, IntegrationCheck},
    orchestration_tools::{
        RuntimeIntegration, definitions_with_integration, dispatch_runtime_job_with_integration,
    },
    resource::RuntimeResources,
    workspaces::{InvalidWorkspace, WorkspacePool},
};
use agent_settings::{AgentProfileId, AgentProfileSettings, AgentSettings};
use anyhow::{Context as _, Result, ensure};
use futures::{FutureExt as _, StreamExt as _};
use gpui::{App, AppContext as _, AsyncApp, Entity, Subscription, Task};
use project::{Project, trusted_worktrees::TrustedWorktrees};
use settings::Settings as _;

use crate::{
    Thread, ThreadEvent, ToolCallEventStream, ToolPermissionContext,
    runtime_tools::{RuntimeToolEventSink, RuntimeToolResolver},
    sandboxing::ThreadSandboxGrants,
};

#[derive(Clone)]
pub struct RuntimeOrchestrationServices {
    pub store: RuntimeStore,
    pub workspace: Option<WorkspacePool>,
    pub workspace_error: Option<String>,
    pub integration: Option<RuntimeIntegration>,
    _quit: Option<Rc<Subscription>>,
}

impl RuntimeOrchestrationServices {
    pub fn metadata_only(store: RuntimeStore) -> Self {
        Self {
            store,
            workspace: None,
            workspace_error: None,
            integration: None,
            _quit: None,
        }
    }
}

#[allow(
    clippy::disallowed_methods,
    reason = "The branch lookup runs on GPUI's background executor and its stdio configuration is never converted to smol."
)]
pub fn prepare_runtime_orchestration(
    store: RuntimeStore,
    repository: PathBuf,
    pool_directory: PathBuf,
    resources: RuntimeResources,
    project: Entity<Project>,
    cx: &mut App,
) -> Task<Result<RuntimeOrchestrationServices>> {
    if TrustedWorktrees::has_restricted_worktrees(&project.read(cx).worktree_store(), cx) {
        return Task::ready(Ok(RuntimeOrchestrationServices::metadata_only(store)));
    }
    let settings = AgentSettings::get_global(cx).runtime.clone();
    let capacity = resources.limits().max_workspaces;
    cx.spawn(async move |cx| {
        let workspace = match WorkspacePool::open(store.clone(), repository.clone(), pool_directory, capacity).await {
            Ok(workspace) => workspace.with_resources(resources),
            Err(error) if error.is::<InvalidWorkspace>() => {
                log::warn!("runtime editing workspace is unavailable: {error:#}");
                return Ok(RuntimeOrchestrationServices {
                    workspace_error: Some(error.to_string()),
                    ..RuntimeOrchestrationServices::metadata_only(store)
                });
            }
            Err(error) => return Err(error),
        };
        let recovered=workspace.recover().await?;
        if !recovered.is_empty() {log::warn!("{} editing workspaces require explicit recovery",recovered.len());}
        let checks=settings.integration_checks.unwrap_or_default();
        let integration_repository = repository.clone();
        let integration=if checks.is_empty() {None} else {
            let target_branch=match settings.target_branch {
                Some(branch)=>branch,
                None=>cx.background_spawn(async move {
                    let output=std::process::Command::new("git").arg("-C").arg(repository).args(["symbolic-ref","--short","HEAD"]).env_remove("GIT_DIR").env_remove("GIT_WORK_TREE").stdin(std::process::Stdio::null()).output()?;
                    ensure!(output.status.success(),"configure agent.runtime.target_branch when the repository has a detached HEAD");
                    Ok::<_,anyhow::Error>(String::from_utf8(output.stdout)?.trim().to_owned())
                }).await?,
            };
            let integration=GitIntegration::open(store.clone(),integration_repository).await?;
            let recovered=integration.recover().await?;
            if !recovered.is_empty() {log::warn!("{} integrations require reconciliation",recovered.len());}
            Some(RuntimeIntegration {integration,target_branch,checks:checks.into_iter().map(|check|IntegrationCheck {program:check.program,arguments:check.arguments,timeout_seconds:check.timeout_seconds}).collect(),auto_integrate:settings.auto_integrate.unwrap_or(true)})
        };
        let pool=workspace.clone();
        let integration_shutdown=integration.as_ref().map(|policy|policy.integration.clone());
        let quit=Some(Rc::new(cx.update(|cx|cx.on_app_quit(move |_| {
            pool.request_shutdown();
            if let Some(integration)=&integration_shutdown {integration.request_shutdown();}
            async {}
        }))));
        Ok(RuntimeOrchestrationServices {store,workspace:Some(workspace),workspace_error:None,integration,_quit:quit})
    })
}

pub fn append_runtime_instructions(prompt: &mut String, services: &RuntimeOrchestrationServices) {
    prompt.push_str("\nNative runtime coordination: delegate with runtime_spawn, park dependencies with runtime_wait, and inspect results with runtime_reports. A child ending its turn with no pending effects automatically reports its final text. Use runtime_report for explicit structured success or failure. Never use legacy spawn_agent/create_thread execution paths.\n");
    prompt.push_str(&format!(
        "Save each runtime_spawn result's agent_id. Wait for those results before calling runtime_wait, then pass those exact agent_id values as dependencies. Spawn keys and labels are not agent IDs. Use at most {} tool calls in a model response and split larger batches across responses. Each wait accepts up to 256 dependency IDs.\n",
        services.store.limits().max_tool_calls_per_response(),
    ));
    if services.workspace.is_some() {
        prompt.push_str("Before modifying project files, acquire your editing lease with runtime_workspace_acquire. All editing and terminal operations must target that leased project. Preserve the lease response for checkpoint and release. Checkpoint edits with runtime_workspace_checkpoint, then release a clean workspace with runtime_workspace_release. Do not create or delete worktrees manually.\n");
    } else if let Some(error) = &services.workspace_error {
        prompt.push_str(&format!(
            "Editing workspaces are unavailable for this project. {error}\n"
        ));
    }
    if services
        .integration
        .as_ref()
        .is_some_and(|policy| policy.auto_integrate)
    {
        prompt.push_str("When your profile permits integration, workspace checkpoints automatically run the user's configured checks and integrate into the configured target branch. Check commands and destination are application settings. Inspect the returned integration status; report failed checks or conflicts instead of bypassing them. runtime_revert selectively reverses an integration with validation.\n");
    }
}

pub fn runtime_tool_enabled(profile: &AgentProfileSettings, name: &str) -> bool {
    if let Some(enabled) = profile.tools.get(name) {
        return *enabled;
    }
    match name {
        "runtime_spawn" | "runtime_wait" => profile.is_tool_enabled("spawn_agent"),
        "runtime_report" | "runtime_reports" => {
            profile.is_tool_enabled("read_file") || profile.is_tool_enabled("spawn_agent")
        }
        "runtime_workspace_acquire"
        | "runtime_workspace_checkpoint"
        | "runtime_workspace_release" => {
            profile.is_tool_enabled("edit_file")
                || profile.is_tool_enabled("write_file")
                || profile.enable_all_context_servers
                || !profile.context_servers.is_empty()
        }
        "runtime_integrate" | "runtime_revert" => {
            profile.is_tool_enabled("terminal") || profile.is_tool_enabled("write_file")
        }
        _ => false,
    }
}

fn authoritative_profile(job: &Job, fallback: &AgentProfileId) -> Result<AgentProfileId> {
    match job.payload.get("runtime_profile_id") {
        Some(serde_json::Value::String(profile)) if !profile.is_empty() => {
            Ok(AgentProfileId(profile.clone().into()))
        }
        None | Some(serde_json::Value::Null) => Ok(fallback.clone()),
        _ => anyhow::bail!("runtime job profile metadata is invalid"),
    }
}

fn automatic_integration_allowed(profile: &AgentProfileSettings, configured: bool) -> bool {
    configured && runtime_tool_enabled(profile, "runtime_integrate")
}

pub fn with_runtime_orchestration(
    resolver: RuntimeToolResolver,
    profile_id: AgentProfileId,
    services: RuntimeOrchestrationServices,
    sink: RuntimeToolEventSink,
) -> RuntimeToolResolver {
    Rc::new(move |job, snapshot, cx| {
        let mut context = resolver(job, snapshot, cx)?;
        context.tools.remove("spawn_agent");
        context.tools.remove("create_thread");
        if services.workspace_error.is_some() {
            context
                .tools
                .retain(|name, _| !crate::runtime_workspaces::requires_workspace(name));
        }
        let requested_profile = authoritative_profile(job, &profile_id)?;
        let (effective_profile_id, _) =
            Thread::profile_for_restricted_workspace(requested_profile, &context.project, cx);
        let profile = AgentSettings::get_global(cx)
            .profiles
            .get(&effective_profile_id)
            .context("runtime agent profile unavailable")?;
        let restricted = TrustedWorktrees::has_restricted_worktrees(
            &context.project.read(cx).worktree_store(),
            cx,
        );
        let acquiring_workspace = job.payload.get("tool").and_then(serde_json::Value::as_str)
            == Some("runtime_workspace_acquire");
        let definitions = definitions_with_integration(
            (services.workspace.is_some() || acquiring_workspace) && !restricted,
            services.integration.is_some() && !restricted,
        )
        .into_iter()
        .filter(|definition| runtime_tool_enabled(profile, &definition.name))
        .collect::<Vec<_>>();
        let enabled = definitions
            .iter()
            .map(|definition| definition.name.clone())
            .collect::<Vec<_>>();
        let mut services = services.clone();
        if let Some(policy) = &mut services.integration {
            policy.auto_integrate = automatic_integration_allowed(profile, policy.auto_integrate);
        }
        let snapshot = snapshot.clone();
        let project = context.project.clone();
        let sink = sink.clone();
        let app = cx.to_async();
        context.install_internal_tools(
            definitions,
            Rc::new(move |job| {
                let mut services = services.clone();
                let project = project.clone();
                let sink = sink.clone();
                let snapshot = snapshot.clone();
                let mut app = app.clone();
                let enabled = enabled.clone();
                let profile_id = effective_profile_id.clone();
                async move {
                    let name = job
                        .payload
                        .get("tool")
                        .and_then(serde_json::Value::as_str)
                        .context("internal tool name")?;
                    ensure!(
                        enabled.iter().any(|enabled| enabled == name),
                        "internal tool is disabled by the selected profile"
                    );
                    if (name.starts_with("runtime_workspace_") && services.workspace.is_some())
                        || matches!(name, "runtime_integrate" | "runtime_revert")
                    {
                        authorize_effect(
                            &services.store,
                            &job,
                            &snapshot,
                            project.clone(),
                            &sink,
                            &mut app,
                        )
                        .await?;
                    }
                    app.update(|cx| {
                        let requested_profile = authoritative_profile(&job, &profile_id)?;
                        let (profile_id, _) = Thread::profile_for_restricted_workspace(
                            requested_profile,
                            &project,
                            cx,
                        );
                        let profile = AgentSettings::get_global(cx)
                            .profiles
                            .get(&profile_id)
                            .context("runtime agent profile unavailable")?;
                        ensure!(
                            runtime_tool_enabled(profile, name),
                            "internal tool was disabled while awaiting execution"
                        );
                        if let Some(policy) = &mut services.integration {
                            policy.auto_integrate =
                                automatic_integration_allowed(profile, policy.auto_integrate);
                        }
                        Ok::<_, anyhow::Error>(())
                    })?;
                    ensure!(
                        dispatch_runtime_job_with_integration(
                            services.store,
                            job,
                            services.workspace,
                            services.integration
                        )
                        .await?,
                        "internal tool was not handled"
                    );
                    Ok(())
                }
                .boxed_local()
            }),
        )?;
        Ok(context)
    })
}

async fn authorize_effect(
    store: &RuntimeStore,
    job: &Job,
    snapshot: &AgentSnapshot,
    project: Entity<Project>,
    sink: &RuntimeToolEventSink,
    cx: &mut AsyncApp,
) -> Result<()> {
    let name = job
        .payload
        .get("tool")
        .and_then(serde_json::Value::as_str)
        .context("tool name")?
        .to_owned();
    let arguments = job
        .payload
        .get("arguments")
        .context("tool arguments")?
        .to_string();
    let call_id = job
        .payload
        .get("tool_call_id")
        .and_then(serde_json::Value::as_str)
        .context("tool call id")?
        .to_owned();
    let (cancellation, receiver) = watch::channel(false);
    let (authorization, _stream, mut events) = cx.update(|cx| {
        ensure!(
            !TrustedWorktrees::has_restricted_worktrees(&project.read(cx).worktree_store(), cx),
            "editing workspace has become restricted"
        );
        let (stream, events) = ToolCallEventStream::for_runtime(
            call_id.into(),
            job.id.clone(),
            project.read(cx).fs().clone(),
            receiver,
            Rc::new(RefCell::new(ThreadSandboxGrants::default())),
        );
        let authorization = stream.authorize(
            format!("Run {name} using the configured workspace and integration policy"),
            ToolPermissionContext::new(name, vec![arguments]),
            cx,
        );
        Ok::<_, anyhow::Error>((authorization, stream, events))
    })?;
    let authorization = authorization.fuse();
    futures::pin_mut!(authorization);
    let mut pending = None;
    loop {
        if let Some(event) = pending.take() {
            pending = Some(event);
            cx.update(|cx| sink(job, snapshot, &mut pending, cx))?;
            if pending.is_some() {
                if let Some(result) = authorization.as_mut().now_or_never() {
                    result?;
                    break;
                }
                cx.background_executor()
                    .timer(std::time::Duration::from_millis(25))
                    .await;
                continue;
            }
        }
        futures::select_biased! {
            result=authorization=>{result?;break;},
            event=events.next().fuse()=> {
                let event=event.context("authorization event stream closed")??;
                if matches!(event,ThreadEvent::ToolCallAuthorization(_)) {store.record_job_event(job.id.clone(),"approval_requested".into(),serde_json::json!({"internal":true})).await?;}
                pending=Some(event);
            }
        }
    }
    drop(cancellation);
    store
        .record_job_event(
            job.id.clone(),
            "approval_completed".into(),
            serde_json::json!({"allowed":true,"internal":true}),
        )
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn readonly_profile() -> AgentProfileSettings {
        AgentProfileSettings {
            name: "Read-only MCP".into(),
            tools: [("read_file".into(), true)].into_iter().collect(),
            enable_all_context_servers: true,
            context_servers: Default::default(),
            default_model: None,
        }
    }

    #[test]
    fn readonly_mcp_profiles_can_use_leases_without_native_edit_or_integration() {
        let mut profile = readonly_profile();
        for tool in [
            "runtime_workspace_acquire",
            "runtime_workspace_checkpoint",
            "runtime_workspace_release",
        ] {
            assert!(runtime_tool_enabled(&profile, tool));
        }
        assert!(!profile.is_tool_enabled("edit_file"));
        assert!(!profile.is_tool_enabled("write_file"));
        assert!(!runtime_tool_enabled(&profile, "runtime_integrate"));
        assert!(!automatic_integration_allowed(&profile, true));
        profile.enable_all_context_servers = false;
        assert!(!runtime_tool_enabled(&profile, "runtime_workspace_acquire"));
        profile.context_servers.insert(
            "server".into(),
            agent_settings::ContextServerPreset {
                tools: Default::default(),
            },
        );
        assert!(runtime_tool_enabled(&profile, "runtime_workspace_acquire"));
    }

    #[test]
    fn explicit_integration_denial_disables_automatic_checkpoint_integration() {
        let mut profile = readonly_profile();
        profile.tools.insert("write_file".into(), true);
        assert!(automatic_integration_allowed(&profile, true));
        profile.tools.insert("runtime_integrate".into(), false);
        assert!(runtime_tool_enabled(
            &profile,
            "runtime_workspace_checkpoint"
        ));
        assert!(!runtime_tool_enabled(&profile, "runtime_integrate"));
        assert!(!automatic_integration_allowed(&profile, true));
        profile
            .tools
            .insert("runtime_workspace_checkpoint".into(), false);
        assert!(!runtime_tool_enabled(
            &profile,
            "runtime_workspace_checkpoint"
        ));
    }

    #[test]
    fn child_job_profile_is_authoritative_over_parent_fallback() -> Result<()> {
        let mut job = Job {
            id: "job".into(),
            agent_id: "child".into(),
            generation: 0,
            state: agent_runtime::JobState::Queued,
            payload: json!({"runtime_profile_id":"ask"}),
            result: None,
        };
        let fallback = AgentProfileId("write".into());
        assert_eq!(authoritative_profile(&job, &fallback)?.as_str(), "ask");
        job.payload = json!({"runtime_profile_id":42});
        assert!(authoritative_profile(&job, &fallback).is_err());
        Ok(())
    }
}
