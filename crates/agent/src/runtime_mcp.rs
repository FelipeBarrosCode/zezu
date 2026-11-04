use std::sync::Arc;

use agent_settings::{AgentProfileId, AgentSettings};
use anyhow::{Context as _, Result, ensure};
use collections::BTreeMap;
use context_server::ContextServerId;
use gpui::{App, AppContext as _, Entity, SharedString, Task, WeakEntity};
use language_model::LanguageModelProviderId;
use project::Project;
use settings::Settings as _;

use crate::{AgentToolOutput, AnyAgentTool, ContextServerRegistry, ToolCallEventStream, ToolInput};

#[derive(Clone)]
pub struct RuntimeMcpCatalog {
    registry: Entity<ContextServerRegistry>,
}

impl RuntimeMcpCatalog {
    pub fn new(project: Entity<Project>, cx: &mut App) -> Self {
        let store = project.read(cx).context_server_store();
        Self {
            registry: cx.new(|cx| ContextServerRegistry::new(store, cx)),
        }
    }

    pub fn from_registry(registry: Entity<ContextServerRegistry>) -> Self {
        Self { registry }
    }

    pub fn merge_into(
        &self,
        tools: &mut BTreeMap<SharedString, Arc<dyn AnyAgentTool>>,
        profile_id: &AgentProfileId,
        provider: &LanguageModelProviderId,
        cx: &App,
    ) -> Result<()> {
        let settings = AgentSettings::get_global(cx);
        let profile = settings
            .profiles
            .get(profile_id)
            .context("MCP agent profile is unavailable")?;
        for (server_id, server_tools) in self.registry.read(cx).servers() {
            for (tool_name, tool) in server_tools {
                if !profile.is_context_server_tool_enabled(&server_id.0, tool_name)
                    || !tool.supports_provider(provider)
                {
                    continue;
                }
                let alias: SharedString = runtime_mcp_tool_name(&server_id.0, tool_name).into();
                ensure!(
                    !tools.contains_key(&alias),
                    "MCP tool alias collision: {alias}"
                );
                tools.insert(
                    alias.clone(),
                    Arc::new(LiveMcpTool {
                        registry: self.registry.downgrade(),
                        server_id: server_id.clone(),
                        canonical_name: tool_name.clone(),
                        alias,
                        profile_id: profile_id.clone(),
                        schema: tool.input_schema(),
                        snapshot: tool.clone(),
                    }),
                );
            }
        }
        Ok(())
    }
}

fn runtime_mcp_tool_name(server: &str, tool: &str) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in (server.len() as u64)
        .to_le_bytes()
        .into_iter()
        .chain(server.bytes())
        .chain((tool.len() as u64).to_le_bytes())
        .chain(tool.bytes())
    {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    let mut suffix = crate::thread::provider_compatible_tool_name(tool);
    suffix.truncate(43);
    format!("mcp_{hash:016x}_{suffix}")
}

struct LiveMcpTool {
    registry: WeakEntity<ContextServerRegistry>,
    server_id: ContextServerId,
    canonical_name: SharedString,
    alias: SharedString,
    profile_id: AgentProfileId,
    schema: serde_json::Value,
    snapshot: Arc<dyn AnyAgentTool>,
}

impl LiveMcpTool {
    fn current(&self, cx: &App) -> Result<Arc<dyn AnyAgentTool>> {
        let settings = AgentSettings::get_global(cx);
        let profile = settings
            .profiles
            .get(&self.profile_id)
            .context("MCP profile was removed")?;
        ensure!(
            profile.is_context_server_tool_enabled(&self.server_id.0, &self.canonical_name),
            "MCP tool is disabled by the current profile"
        );
        let registry = self
            .registry
            .upgrade()
            .context("MCP registry has stopped")?;
        let tool = registry
            .read(cx)
            .tools_for_server(&self.server_id)
            .find(|tool| tool.name() == self.canonical_name)
            .cloned()
            .context("MCP tool is no longer available")?;
        ensure!(
            tool.input_schema() == self.schema,
            "MCP tool schema changed; regenerate the model request before execution"
        );
        Ok(tool)
    }
}

impl AnyAgentTool for LiveMcpTool {
    fn name(&self) -> SharedString {
        self.alias.clone()
    }
    fn description(&self) -> SharedString {
        self.snapshot.description()
    }
    fn kind(&self) -> agent_client_protocol::v2::schema::ToolKind {
        self.snapshot.kind()
    }
    fn initial_title(&self, input: serde_json::Value, cx: &mut App) -> SharedString {
        self.snapshot.initial_title(input, cx)
    }
    fn input_schema(&self) -> serde_json::Value {
        self.schema.clone()
    }
    fn supports_provider(&self, provider: &LanguageModelProviderId) -> bool {
        self.snapshot.supports_provider(provider)
    }
    fn allow_in_restricted_mode(&self) -> bool {
        self.snapshot.allow_in_restricted_mode()
    }
    fn run(
        self: Arc<Self>,
        input: ToolInput<serde_json::Value>,
        events: ToolCallEventStream,
        cx: &mut App,
    ) -> Task<Result<AgentToolOutput, AgentToolOutput>> {
        cx.spawn(async move |cx| {
            let arguments = input.recv().await.map_err(AgentToolOutput::from)?;
            if !arguments.is_object() && !arguments.is_null() {
                return Err(anyhow::anyhow!(
                    "MCP tool arguments must be an object; the supplied arguments were not sent"
                )
                .into());
            }
            let task = cx.update(|cx| {
                self.current(cx)
                    .map(|tool| tool.run(ToolInput::ready(arguments), events, cx))
            });
            match task {
                Ok(task) => task.await,
                Err(error) => Err(error.into()),
            }
        })
    }
    fn replay(
        &self,
        input: serde_json::Value,
        output: serde_json::Value,
        events: ToolCallEventStream,
        cx: &mut App,
    ) -> Result<()> {
        self.current(cx)?.replay(input, output, events, cx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mcp_names_are_stable_bounded_and_distinguish_sanitization_collisions() {
        let examples = [
            ("server/a", "search"),
            ("server-a", "search"),
            ("server/a", "search!"),
            ("server/a", "search?"),
            ("x", "工具"),
        ];
        let names = examples
            .iter()
            .map(|(server, tool)| runtime_mcp_tool_name(server, tool))
            .collect::<Vec<_>>();
        for ((server, tool), name) in examples.iter().zip(&names) {
            assert_eq!(&runtime_mcp_tool_name(server, tool), name);
            assert!(name.len() <= 64);
            assert!(
                name.chars()
                    .all(|character| character.is_ascii_alphanumeric()
                        || character == '_'
                        || character == '-')
            );
        }
        let unique = names.iter().collect::<std::collections::HashSet<_>>();
        assert_eq!(unique.len(), names.len());
        assert!(runtime_mcp_tool_name(&"a".repeat(200), &"b".repeat(200)).len() <= 64);
    }
}
