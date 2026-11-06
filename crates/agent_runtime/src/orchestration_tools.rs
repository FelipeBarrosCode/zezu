use anyhow::{Context as _, Result, ensure};
use infinity_agent_core::message::{InputMessage, InputMessageContent};
use infinity_provider_protocol::{ToolDefinition, ToolResult, ToolResultContent, UserContent};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{
    Job, RuntimeStore,
    orchestration::{FailurePolicy, InvalidWait, Orchestrator, TaskIdentity, TaskResult},
    workspaces::{InvalidWorkspace, WorkspaceLease, WorkspacePool},
};

#[derive(Clone)]
pub struct RuntimeIntegration {
    pub integration: crate::integration::GitIntegration,
    pub target_branch: String,
    pub checks: Vec<crate::integration::IntegrationCheck>,
    pub auto_integrate: bool,
}

pub fn definitions_with_integration(workspaces: bool, integration: bool) -> Vec<ToolDefinition> {
    let mut tools = definitions(workspaces);
    if integration {
        tools.extend([
            definition("runtime_integrate", "Apply a durable workspace artifact through the configured checks and serialized integration policy. The target branch and check commands come from trusted application configuration.",json!({"artifact_id":{"type":"string"}}),&["artifact_id"]),
            definition("runtime_revert", "Selectively revert a previous integration using the configured validation checks.",json!({"integration_id":{"type":"string"}}),&["integration_id"]),
        ]);
    }
    tools
}

pub fn definitions(workspaces: bool) -> Vec<ToolDefinition> {
    let mut tools = vec![
        definition(
            "runtime_spawn",
            "Delegate an isolated child task with a durable initial prompt. Save the returned agent_id for runtime_wait and runtime_reports. The spawn key is a label, not the agent ID.",
            json!({"key":{"type":"string"},"prompt":{"type":"string"},"fresh":{"type":"boolean"}}),
            &["key", "prompt"],
        ),
        definition(
            "runtime_wait",
            "Park this agent until its dependencies report. Wait for the runtime_spawn results first, then pass their returned agent_id values as dependencies. Set fail_fast to wake on the first failure.",
            json!({"key":{"type":"string"},"dependencies":{"type":"array","items":{"type":"string","description":"Returned agent_id of a task in this run. Do not pass a spawn key or label."},"minItems":1,"maxItems":256},"fail_fast":{"type":"boolean"}}),
            &["key", "dependencies"],
        ),
        definition(
            "runtime_report",
            "Finish this task with its final result and notify waiting agents. Pending effects must be completed or reconciled first.",
            json!({"status":{"type":"string","enum":["success","failed"]},"result":{}}),
            &["status", "result"],
        ),
        definition(
            "runtime_reports",
            "Read a bounded page of child reports, or fetch a particular task result. Large join results are available here by task id.",
            json!({"after":{"type":"integer","minimum":0},"limit":{"type":"integer","minimum":1,"maximum":64},"agent_id":{"type":"string"},"generation":{"type":"integer","minimum":0}}),
            &[],
        ),
    ];
    if workspaces {
        tools.extend([
            definition("runtime_workspace_acquire", "Acquire an isolated editing workspace from the configured bounded pool. A null result means every workspace is currently leased.",json!({"base_ref":{"type":"string"}}),&[]),
            definition("runtime_workspace_checkpoint", "Commit edits in your leased workspace to a durable recoverable artifact before releasing it.",json!({"lease":{"type":"object"},"message":{"type":"string"}}),&["lease","message"]),
            definition("runtime_workspace_release", "Release your clean checkpointed workspace back to the bounded pool. Dirty workspaces are preserved and cannot be released.",json!({"lease":{"type":"object"}}),&["lease"]),
        ]);
    }
    tools
}

fn definition(
    name: &str,
    description: &str,
    properties: Value,
    required: &[&str],
) -> ToolDefinition {
    ToolDefinition {
        name: name.into(),
        description: description.into(),
        parameters: json!({"type":"object","properties":properties,"required":required,"additionalProperties":false}),
    }
}

/// The native broker must apply its enabled-tool/profile/trust/approval checks before calling this dispatcher.
pub async fn dispatch_runtime_job(
    store: RuntimeStore,
    job: Job,
    workspace: Option<WorkspacePool>,
) -> Result<bool> {
    dispatch_runtime_job_with_integration(store, job, workspace, None).await
}

pub async fn dispatch_runtime_job_with_integration(
    store: RuntimeStore,
    job: Job,
    workspace: Option<WorkspacePool>,
    integration: Option<RuntimeIntegration>,
) -> Result<bool> {
    let name = job
        .payload
        .get("tool")
        .and_then(Value::as_str)
        .context("job tool name")?;
    if name != "runtime_workspace_acquire"
        && !definitions_with_integration(workspace.is_some(), integration.is_some())
            .iter()
            .any(|tool| tool.name == name)
    {
        return Ok(false);
    }
    let current = store.job(job.id.clone()).await?;
    ensure!(
        current.state == crate::JobState::Dispatched
            && current.agent_id == job.agent_id
            && current.generation == job.generation
            && current.payload == job.payload,
        "runtime tool requires the current dispatched job"
    );
    let task = TaskIdentity {
        agent_id: job.agent_id.clone(),
        generation: job.generation,
    };
    let arguments = job
        .payload
        .get("arguments")
        .cloned()
        .context("job arguments")?;
    let orchestrator = Orchestrator::open(store.clone()).await?;
    if name == "runtime_report" {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Arguments {
            status: String,
            result: Value,
        }
        let arguments: Arguments = serde_json::from_value(arguments)?;
        let result = match arguments.status.as_str() {
            "success" => TaskResult::Success(arguments.result),
            "failed" => TaskResult::Failed(
                arguments
                    .result
                    .as_str()
                    .map(str::to_owned)
                    .unwrap_or_else(|| arguments.result.to_string()),
            ),
            _ => anyhow::bail!("task report status must be success or failed"),
        };
        if store
            .workspace_has_uncheckpointed_edits(task.clone())
            .await?
        {
            let input = tool_result_input(
                &job,
                json!({"error":"Checkpoint and release your editing workspace before reporting completion."}),
                true,
            )?;
            store
                .complete_job(job.id, serde_json::to_value(input)?)
                .await?;
            return Ok(true);
        }
        orchestrator.report_dispatched_job(job, result).await?;
        return Ok(true);
    }
    let result = match name {
        "runtime_spawn" => {
            #[derive(Deserialize)]
            #[serde(deny_unknown_fields)]
            struct Arguments {
                key: String,
                prompt: String,
                #[serde(default)]
                fresh: bool,
            }
            let arguments: Arguments = serde_json::from_value(arguments)?;
            ensure!(
                !arguments.key.is_empty() && arguments.key.len() <= 256,
                "spawn key must contain 1 to 256 bytes"
            );
            ensure!(
                !arguments.prompt.is_empty()
                    && arguments.prompt.len() <= store.limits().max_input_bytes / 2,
                "child prompt exceeds its byte budget"
            );
            let parent = job.agent_id.clone();
            let cutoff = store
                .read(move |connection| {
                    connection
                        .query_row(
                            "SELECT max(count(*)-1,0) FROM history WHERE agent_id=?",
                            [parent],
                            |row| row.get::<_, i64>(0),
                        )
                        .map_err(Into::into)
                })
                .await?;
            let child = orchestrator
                .spawn_child(task, arguments.key.clone(), cutoff, arguments.fresh)
                .await?;
            let root_id = store.snapshot(job.agent_id.clone()).await?.root_id;
            let input = InputMessage::user_text(child.clone(), arguments.prompt);
            store
                .enqueue_from(
                    crate::Lease {
                        agent_id: job.agent_id.clone(),
                        root_id,
                        generation: job.generation,
                        id: String::new(),
                        inputs: Vec::new(),
                    },
                    child.clone(),
                    format!("spawn:{}:{}:initial", job.agent_id, arguments.key),
                    serde_json::to_value(input)?,
                )
                .await?;
            json!({"agent_id":child})
        }
        "runtime_wait" => {
            #[derive(Deserialize)]
            #[serde(deny_unknown_fields)]
            struct Arguments {
                key: String,
                dependencies: Vec<String>,
                #[serde(default)]
                fail_fast: bool,
            }
            let arguments: Arguments = match serde_json::from_value(arguments) {
                Ok(arguments) => arguments,
                Err(_) => {
                    return reject_tool(
                        &store,
                        job,
                        "runtime_wait requires a string key, a dependencies array of agent_id strings, and an optional boolean fail_fast",
                    )
                    .await;
                }
            };
            let status = orchestrator
                .wait_for(
                    task,
                    arguments.key,
                    arguments.dependencies,
                    if arguments.fail_fast {
                        FailurePolicy::FailFast
                    } else {
                        FailurePolicy::Collect
                    },
                )
                .await;
            match status {
                Ok(status) => serde_json::to_value(status)?,
                Err(error) if error.is::<InvalidWait>() => {
                    return reject_tool(&store, job, &error.to_string()).await;
                }
                Err(error) => return Err(error),
            }
        }
        "runtime_reports" => {
            #[derive(Deserialize)]
            #[serde(deny_unknown_fields)]
            struct Arguments {
                after: Option<i64>,
                limit: Option<usize>,
                agent_id: Option<String>,
                generation: Option<i64>,
            }
            let arguments: Arguments = serde_json::from_value(arguments)?;
            if let Some(agent_id) = arguments.agent_id {
                let root = store.snapshot(job.agent_id.clone()).await?.root_id;
                let requested = store.snapshot(agent_id.clone()).await?;
                ensure!(root == requested.root_id, "task reports cannot cross runs");
                serde_json::to_value(
                    orchestrator
                        .task_report(TaskIdentity {
                            agent_id,
                            generation: arguments.generation.unwrap_or(requested.generation),
                        })
                        .await?,
                )?
            } else {
                let limit = arguments.limit.unwrap_or(16);
                ensure!(
                    (1..=64).contains(&limit),
                    "report page size must be between 1 and 64"
                );
                serde_json::to_value(
                    orchestrator
                        .reports(job.agent_id.clone(), arguments.after.unwrap_or(0), limit)
                        .await?,
                )?
            }
        }
        "runtime_integrate" => {
            #[derive(Deserialize)]
            #[serde(deny_unknown_fields)]
            struct Arguments {
                artifact_id: String,
            }
            let arguments: Arguments = serde_json::from_value(arguments)?;
            let policy = integration.context("integration policy is not configured")?;
            let artifact = workspace
                .context("workspace pool is not configured")?
                .artifact(arguments.artifact_id)
                .await?
                .context("workspace artifact was not found")?;
            let root = store.snapshot(job.agent_id.clone()).await?.root_id;
            let owner = store.snapshot(artifact.agent_id.clone()).await?;
            ensure!(
                owner.root_id == root,
                "workspace artifacts cannot cross runs"
            );
            ensure!(
                owner.generation == artifact.generation && owner.state != "cancelled",
                "workspace artifact belongs to an interrupted task generation"
            );
            serde_json::to_value(
                policy
                    .integration
                    .integrate(crate::integration::IntegrationRequest {
                        id: job.id.clone(),
                        agent_id: job.agent_id.clone(),
                        generation: job.generation,
                        target_branch: policy.target_branch,
                        source_commit: artifact.commit,
                        source_owner: Some(TaskIdentity {
                            agent_id: artifact.agent_id.clone(),
                            generation: artifact.generation,
                        }),
                        checks: policy.checks,
                    })
                    .await?,
            )?
        }
        "runtime_revert" => {
            #[derive(Deserialize)]
            #[serde(deny_unknown_fields)]
            struct Arguments {
                integration_id: String,
            }
            let arguments: Arguments = serde_json::from_value(arguments)?;
            let policy = integration.context("integration policy is not configured")?;
            let record = policy
                .integration
                .record(arguments.integration_id.clone())
                .await?
                .context("integration was not found")?;
            let root = store.snapshot(job.agent_id.clone()).await?.root_id;
            ensure!(
                store.snapshot(record.request.agent_id).await?.root_id == root,
                "integrations cannot cross runs"
            );
            ensure!(
                record.request.target_branch == policy.target_branch,
                "integration belongs to another target branch"
            );
            serde_json::to_value(
                policy
                    .integration
                    .revert(arguments.integration_id, policy.checks)
                    .await?,
            )?
        }
        "runtime_workspace_acquire" => {
            #[derive(Deserialize)]
            #[serde(deny_unknown_fields)]
            struct Arguments {
                base_ref: Option<String>,
            }
            let arguments: Arguments = match serde_json::from_value(arguments) {
                Ok(arguments) => arguments,
                Err(_) => {
                    return reject_tool(
                        &store,
                        job,
                        "runtime_workspace_acquire accepts an optional Git revision string in base_ref.",
                    )
                    .await;
                }
            };
            let Some(workspace) = workspace else {
                return reject_tool(
                    &store,
                    job,
                    "Editing workspaces are unavailable. Open a trusted project repository with a commit, then reopen the conversation.",
                )
                .await;
            };
            let epoch = store.workspace_resource_epoch().await?;
            let acquisition = workspace
                .acquire(task, arguments.base_ref.unwrap_or_else(|| "HEAD".into()))
                .await;
            match acquisition {
                Ok(Some(lease)) => serde_json::to_value(lease)?,
                Ok(None) => {
                    store.park_workspace_job(job, epoch).await?;
                    return Ok(true);
                }
                Err(error) if error.is::<InvalidWorkspace>() => {
                    return reject_tool(&store, job, &error.to_string()).await;
                }
                Err(error) => return Err(error),
            }
        }
        "runtime_workspace_checkpoint" => {
            #[derive(Deserialize)]
            #[serde(deny_unknown_fields)]
            struct Arguments {
                lease: WorkspaceLease,
                message: String,
            }
            let arguments: Arguments = serde_json::from_value(arguments)?;
            ensure_owner(&arguments.lease, &job)?;
            let artifact = workspace
                .context("editing workspace pool is not configured")?
                .checkpoint(arguments.lease, arguments.message)
                .await?;
            if let Some(policy) = integration.filter(|policy| policy.auto_integrate) {
                let integrated = policy
                    .integration
                    .integrate(crate::integration::IntegrationRequest {
                        id: format!("checkpoint:{}", artifact.id),
                        agent_id: job.agent_id.clone(),
                        generation: job.generation,
                        target_branch: policy.target_branch,
                        source_commit: artifact.commit.clone(),
                        source_owner: Some(TaskIdentity {
                            agent_id: artifact.agent_id.clone(),
                            generation: artifact.generation,
                        }),
                        checks: policy.checks,
                    })
                    .await?;
                json!({"artifact":artifact,"integration":integrated})
            } else {
                serde_json::to_value(artifact)?
            }
        }
        "runtime_workspace_release" => {
            #[derive(Deserialize)]
            #[serde(deny_unknown_fields)]
            struct Arguments {
                lease: WorkspaceLease,
            }
            let arguments: Arguments = serde_json::from_value(arguments)?;
            ensure_owner(&arguments.lease, &job)?;
            workspace
                .context("editing workspace pool is not configured")?
                .release(arguments.lease)
                .await?;
            json!({"released":true})
        }
        _ => return Ok(false),
    };
    let input = tool_result_input(&job, result, false)?;
    store
        .complete_job(job.id, serde_json::to_value(input)?)
        .await?;
    Ok(true)
}

async fn reject_tool(store: &RuntimeStore, job: Job, message: &str) -> Result<bool> {
    store
        .complete_tool_job(
            job,
            vec![json!({"type":"text","text":message})],
            true,
            Value::Null,
        )
        .await?;
    Ok(true)
}

fn ensure_owner(lease: &WorkspaceLease, job: &Job) -> Result<()> {
    ensure!(
        lease.agent_id == job.agent_id && lease.generation == job.generation,
        "tool cannot use another task's workspace lease"
    );
    Ok(())
}

pub(crate) fn tool_result_input(job: &Job, result: Value, is_error: bool) -> Result<InputMessage> {
    let id = job
        .payload
        .get("tool_call_id")
        .and_then(Value::as_str)
        .context("job tool call id")?;
    let call_id = job
        .payload
        .get("provider_call_id")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let text = serde_json::to_string(&json!({"is_error":is_error,"result":result}))?;
    Ok(InputMessage {
        content: InputMessageContent::User(UserContent::ToolResult(ToolResult {
            id: id.into(),
            call_id,
            content: vec![ToolResultContent::text(text)],

            ..Default::default()
        })),
        group_id: job.agent_id.clone().into(),
        metadata: None,
        synthetic: None,
        display_as: None,
        subscription: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{JobState, RuntimeLimits, SliceOutcome};

    #[tokio::test]
    async fn invalid_wait_returns_tool_error_then_accepts_the_completed_child_id() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let store = RuntimeStore::open(
            directory.path().join("runtime.sqlite"),
            RuntimeLimits::default(),
        )?;
        let orchestrator = Orchestrator::open(store.clone()).await?;
        store.create_root("root".into(), json!({})).await?;
        let parent = TaskIdentity {
            agent_id: "root".into(),
            generation: 0,
        };
        let child = orchestrator
            .spawn_child(parent, "agent-01".into(), 0, true)
            .await?;
        orchestrator
            .report(
                TaskIdentity {
                    agent_id: child.clone(),
                    generation: 0,
                },
                "done".into(),
                TaskResult::Success(json!("completed child")),
            )
            .await?;
        assert_eq!(store.snapshot(child.clone()).await?.state, "completed");

        store
            .enqueue_text_input("root".into(), "start".into(), "collect results".into())
            .await?;
        let lease = store.admit().await?.context("parent admission")?;
        let invalid_waits = [
            ("label", json!([child, "agent-01"])),
            ("malformed", json!("agent-01")),
        ];
        for (id, dependencies) in &invalid_waits {
            store
                .record_job(
                    lease.clone(),
                    (*id).into(),
                    json!({
                        "tool": "runtime_wait",
                        "tool_call_id": id,
                        "provider_call_id": format!("provider-{id}"),
                        "arguments": {"key":"join", "dependencies":dependencies},
                    }),
                )
                .await?;
        }
        store
            .finish(lease, json!({}), SliceOutcome::Waiting)
            .await?;
        for (id, _) in invalid_waits {
            let job = store.dispatch_job(id.into()).await?;
            assert!(dispatch_runtime_job(store.clone(), job, None).await?);
            let job = store.job(id.into()).await?;
            assert_eq!(job.state, JobState::Completed);
            let result = job.result.context("completed tool error")?;
            assert_eq!(result["content"]["is_error"], true);
            assert_eq!(result["content"]["id"], id);
            assert_eq!(result["content"]["call_id"], format!("provider-{id}"));
            assert!(
                result["content"]["content"][0]["text"]
                    .as_str()
                    .context("error text")?
                    .contains("agent_id")
            );
        }
        assert!(store.uncertain_jobs("root".into(), 1).await?.is_empty());
        let lease = store
            .admit()
            .await?
            .context("parent resumes after tool errors")?;
        assert_eq!(lease.inputs.len(), 2);
        assert_eq!(lease.agent_id, "root");
        store
            .record_job(
                lease.clone(),
                "valid".into(),
                json!({
                    "tool":"runtime_wait",
                    "tool_call_id":"valid",
                    "arguments":{"key":"join", "dependencies":[child]},
                }),
            )
            .await?;
        store
            .finish(lease, json!({}), SliceOutcome::Waiting)
            .await?;

        let job = store.dispatch_job("valid".into()).await?;
        assert!(dispatch_runtime_job(store.clone(), job, None).await?);
        let lease = store
            .admit()
            .await?
            .context("parent resumes with the child report")?;
        assert_eq!(lease.inputs.len(), 2);
        assert_eq!(lease.inputs[0].0, "job:valid:result");
        assert!(lease.inputs[1].0.starts_with("join:"));
        let report: Value = serde_json::from_str(
            lease.inputs[1].1["content"]["text"]
                .as_str()
                .context("join report text")?,
        )?;
        assert_eq!(report["remaining"], 0);
        assert_eq!(report["results"][0]["agent_id"], child);
        assert_eq!(report["results"][0]["result"]["result"], "completed child");
        store.shutdown().await
    }

    #[tokio::test]
    async fn native_spawn_and_terminal_report_drive_parent_join() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let store = RuntimeStore::open(
            directory.path().join("runtime.sqlite"),
            RuntimeLimits::default(),
        )?;
        let orchestrator = Orchestrator::open(store.clone()).await?;
        store.create_root("root".into(), json!({})).await?;
        store
            .enqueue_text_input("root".into(), "start".into(), "start".into())
            .await?;
        let lease = store.admit().await?.context("lease")?;
        store.record_job(lease.clone(),"spawn".into(),json!({"tool":"runtime_spawn","arguments":{"key":"child","prompt":"work","fresh":true},"tool_call_id":"spawn-call"})).await?;
        store
            .finish(lease, json!({}), SliceOutcome::Waiting)
            .await?;
        let job = store.dispatch_job("spawn".into()).await?;
        assert!(dispatch_runtime_job(store.clone(), job, None).await?);
        let child = store
            .list_agents("root".into(), None, 10)
            .await?
            .agents
            .into_iter()
            .find(|agent| agent.id != "root")
            .context("child")?;
        orchestrator
            .wait_for(
                TaskIdentity {
                    agent_id: "root".into(),
                    generation: 0,
                },
                "join".into(),
                vec![child.id.clone()],
                FailurePolicy::Collect,
            )
            .await?;
        let lease = store.admit().await?.context("child inference")?;
        assert_eq!(lease.agent_id, child.id);
        store.record_job(lease.clone(),"report".into(),json!({"tool":"runtime_report","arguments":{"status":"success","result":"done"},"tool_call_id":"report-call"})).await?;
        store
            .finish(lease, json!({}), SliceOutcome::Waiting)
            .await?;
        let job = store.dispatch_job("report".into()).await?;
        assert!(dispatch_runtime_job(store.clone(), job, None).await?);
        assert_eq!(store.snapshot(child.id).await?.state, "completed");
        assert_eq!(
            store.admit().await?.context("parent wake")?.agent_id,
            "root"
        );
        store.shutdown().await
    }
}
