use std::{path::PathBuf, rc::Rc};

use acp_thread::{AgentModelId, ProcessOutput};
use agent_client_protocol::v2::schema as acp;
use agent_settings::{AgentProfileId, AgentSettings};
use anyhow::{Context as _, Result, anyhow};
use collections::BTreeMap;
use futures::future::Shared;
use gpui::{App, AsyncApp, Entity, SharedString, Task};
use language_model::LanguageModelProviderId;
use project::Project;
use settings::Settings as _;

use crate::{
    AgentTool as _, AnyAgentTool, SandboxedTerminalTool, SubagentHandle, TerminalHandle,
    TerminalTool, ThreadEnvironment, sandboxing::sandboxing_enabled_for_project,
};

pub fn runtime_terminal_tools(
    project: Entity<Project>,
    provider: &LanguageModelProviderId,
    profile_id: &AgentProfileId,
    cx: &App,
) -> Result<BTreeMap<SharedString, std::sync::Arc<dyn AnyAgentTool>>> {
    let settings = AgentSettings::get_global(cx);
    let profile = settings
        .profiles
        .get(profile_id)
        .context("terminal agent profile is unavailable")?;
    let mut tools = BTreeMap::new();
    if !profile.is_tool_enabled(TerminalTool::NAME) {
        return Ok(tools);
    }
    let environment = Rc::new(RuntimeThreadEnvironment {
        project: project.clone(),
    });
    let tool = if sandboxing_enabled_for_project(project.read(cx), cx) {
        SandboxedTerminalTool::new(project, environment).erase()
    } else {
        TerminalTool::new(project, environment).erase()
    };
    if tool.supports_provider(provider) {
        tools.insert(SharedString::from(TerminalTool::NAME), tool);
    }
    Ok(tools)
}

struct RuntimeThreadEnvironment {
    project: Entity<Project>,
}

impl ThreadEnvironment for RuntimeThreadEnvironment {
    fn create_terminal(
        &self,
        command: String,
        extra_env: Vec<(String, String)>,
        cwd: Option<PathBuf>,
        output_byte_limit: Option<u64>,
        sandbox_wrap: Option<acp_thread::SandboxWrap>,
        cx: &mut AsyncApp,
    ) -> Task<Result<Rc<dyn TerminalHandle>>> {
        let project = self.project.clone();
        cx.spawn(async move |cx| {
            #[cfg_attr(any(target_os = "linux", target_os = "windows"), allow(unused_mut))]
            let mut extra_env = extra_env;
            #[cfg_attr(any(target_os = "linux", target_os = "windows"), allow(unused_mut))]
            let mut sandbox_wrap = sandbox_wrap;
            #[cfg(not(any(target_os = "linux", target_os = "windows")))]
            let temporary = if project.read_with(cx, |project, _| project.is_local()) {
                let directory = cx.background_spawn(async { tempfile::tempdir() }).await?;
                let path = directory.path().canonicalize()?;
                let value = path.to_string_lossy().to_string();
                extra_env.extend([
                    ("TMPDIR".into(), value.clone()),
                    ("TMP".into(), value.clone()),
                    ("TEMP".into(), value),
                ]);
                if let Some(wrap) = &mut sandbox_wrap {
                    wrap.writable_paths.push(path);
                }
                Some(directory)
            } else {
                None
            };
            let terminal = cx
                .update(|cx| {
                    acp_thread::create_runtime_terminal(
                        project,
                        command,
                        vec![],
                        extra_env,
                        cwd,
                        output_byte_limit,
                        sandbox_wrap,
                        cx,
                    )
                })
                .await?;
            Ok(Rc::new(RuntimeTerminalHandle {
                terminal,
                #[cfg(not(any(target_os = "linux", target_os = "windows")))]
                _temporary: temporary,
            }) as Rc<dyn TerminalHandle>)
        })
    }

    fn create_subagent(
        &self,
        _label: String,
        _model: Option<AgentModelId>,
        _: &mut App,
    ) -> Result<Rc<dyn SubagentHandle>> {
        Err(anyhow!(
            "use the native runtime delegation tools to create children"
        ))
    }
}

struct RuntimeTerminalHandle {
    terminal: Entity<acp_thread::Terminal>,
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    _temporary: Option<tempfile::TempDir>,
}

impl TerminalHandle for RuntimeTerminalHandle {
    fn id(&self, cx: &AsyncApp) -> Result<acp::TerminalId> {
        Ok(self
            .terminal
            .read_with(cx, |terminal, _| terminal.id().clone()))
    }
    fn current_output(&self, cx: &AsyncApp) -> Result<ProcessOutput> {
        Ok(self
            .terminal
            .read_with(cx, |terminal, cx| terminal.current_output(cx)))
    }
    fn wait_for_exit(&self, cx: &AsyncApp) -> Result<Shared<Task<acp::TerminalExitStatus>>> {
        self.terminal
            .read_with(cx, |terminal, _| terminal.wait_for_exit())
    }
    fn kill(&self, cx: &AsyncApp) -> Result<()> {
        cx.update(|cx| self.terminal.update(cx, |terminal, cx| terminal.kill(cx)));
        Ok(())
    }
    fn was_stopped_by_user(&self, cx: &AsyncApp) -> Result<bool> {
        Ok(self
            .terminal
            .read_with(cx, |terminal, _| terminal.was_stopped_by_user()))
    }
}
