use std::sync::Arc;

use agent_settings::{AgentProfileId, AgentSettings, UserAgentsMd};
use anyhow::{Context as _, Result, ensure};
use fs::Fs;
use gpui::{App, Entity, SharedString, Task};
use project::Project;
use settings::{LanguageModelSelection, Settings as _};

use crate::{
    NativeAgent, NativeRuntimeConfig, Skill, SystemPromptTemplate, Template, Templates, Thread,
};

pub struct PreparedRuntimeContext {
    pub system_prompt: String,
    pub profile_id: AgentProfileId,
    pub profile_was_downgraded: bool,
    pub enabled_tool_names: Vec<SharedString>,
    pub skills: Vec<Skill>,
    pub warnings: Vec<SharedString>,
}

impl PreparedRuntimeContext {
    pub fn configuration(&self, model: Option<LanguageModelSelection>) -> NativeRuntimeConfig {
        NativeRuntimeConfig {
            model,
            profile: Some(self.profile_id.clone()),
            system_prompt: self.system_prompt.clone(),
        }
    }
}

pub fn prepare_runtime_context(
    project: Entity<Project>,
    fs: Arc<dyn Fs>,
    requested_profile: Option<AgentProfileId>,
    available_tool_names: Vec<SharedString>,
    model_name: Option<String>,
    cx: &mut App,
) -> Task<Result<PreparedRuntimeContext>> {
    let settings = AgentSettings::get_global(cx);
    if !settings.enabled(cx) {
        return Task::ready(Err(anyhow::anyhow!(
            "AI features are disabled in Zed settings"
        )));
    }
    let requested_profile = requested_profile.unwrap_or_else(|| settings.default_profile.clone());
    let (profile_id, profile_was_downgraded) =
        Thread::profile_for_restricted_workspace(requested_profile, &project, cx);
    let Some(profile) = settings.profiles.get(&profile_id) else {
        return Task::ready(Err(anyhow::anyhow!(
            "agent profile is unavailable: {profile_id}"
        )));
    };
    let enabled_tool_names = available_tool_names
        .into_iter()
        .filter(|name| {
            !crate::tools::ALL_TOOL_NAMES.contains(&name.as_ref())
                || profile.is_tool_enabled(name.as_ref())
        })
        .filter(|name| crate::tools::tool_feature_flag_enabled(name, cx))
        .collect::<Vec<_>>();
    let user_agents_md = UserAgentsMd::global(cx).and_then(|state| state.content().cloned());
    let sandboxing = crate::sandboxing::sandboxing_enabled_for_project(project.read(cx), cx);
    let context = NativeAgent::build_project_context(&project, fs, cx);
    cx.spawn(async move |cx| {
        let (project_context, skills, issues) = context.await;
        let enabled = cx.update(|cx| AgentSettings::get_global(cx).enabled(cx));
        ensure!(
            enabled,
            "AI features were disabled while loading agent instructions"
        );
        let system_prompt = SystemPromptTemplate {
            project: &project_context,
            available_tools: enabled_tool_names.clone(),
            model_name,
            date: chrono::Local::now().format("%Y-%m-%d").to_string(),
            user_agents_md,
            sandboxing,
            is_linux: cfg!(target_os = "linux"),
            is_windows: cfg!(target_os = "windows"),
        }
        .render(&Templates::new())
        .context("failed to render runtime instructions")?;
        Ok(PreparedRuntimeContext {
            system_prompt,
            profile_id,
            profile_was_downgraded,
            enabled_tool_names,
            skills,
            warnings: issues
                .into_iter()
                .map(|issue| format!("{}: {}", issue.path.display(), issue.message).into())
                .collect(),
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use fs::FakeFs;
    use gpui::TestAppContext;
    use serde_json::json;

    #[gpui::test]
    async fn runtime_context_loads_live_rules_and_skill_catalog(cx: &mut TestAppContext) {
        crate::tests::init_test(cx);
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree("/project", json!({
            "AGENTS.md": "Use the native runtime project rules.",
            ".agents": {"skills": {"verification": {"SKILL.md": "---\nname: verification\ndescription: Verify project results\n---\nRun the checks."}}}
        })).await;
        let project = Project::test(fs.clone(), [], cx).await;
        project
            .update(cx, |project, cx| {
                project.create_worktree("/project", true, cx)
            })
            .await
            .expect("worktree");
        let prepared = cx
            .update(|cx| {
                prepare_runtime_context(
                    project,
                    fs,
                    None,
                    vec!["read_file".into()],
                    Some("Fake".into()),
                    cx,
                )
            })
            .await
            .expect("context");
        assert!(
            prepared
                .system_prompt
                .contains("Use the native runtime project rules.")
        );
        assert!(prepared.system_prompt.contains("Verify project results"));
        assert!(
            prepared
                .skills
                .iter()
                .any(|skill| skill.name == "verification")
        );
        assert_eq!(
            prepared.configuration(None).profile,
            Some(prepared.profile_id)
        );
    }

    #[gpui::test]
    async fn runtime_context_rejects_unknown_profiles_and_disabled_ai(cx: &mut TestAppContext) {
        crate::tests::init_test(cx);
        let fs = FakeFs::new(cx.executor());
        let project = Project::test(fs.clone(), [], cx).await;
        let result = cx
            .update(|cx| {
                prepare_runtime_context(
                    project.clone(),
                    fs.clone(),
                    Some(AgentProfileId("missing".into())),
                    Vec::new(),
                    None,
                    cx,
                )
            })
            .await;
        assert!(result.is_err());
        cx.update(|cx| {
            let mut settings = AgentSettings::get_global(cx).clone();
            settings.enabled = false;
            AgentSettings::override_global(settings, cx);
        });
        assert!(
            cx.update(|cx| prepare_runtime_context(project, fs, None, Vec::new(), None, cx))
                .await
                .is_err()
        );
    }
}
