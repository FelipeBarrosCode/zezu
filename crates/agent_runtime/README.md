# Embedded concurrent agent runtime

Zezu embeds Infinity inside the native Zed application. New native Agent Panel
sessions submit to this runtime by default; existing legacy sessions retain their
original connection. The embedded worker owns execution while ACP thread objects
and the fleet panel project bounded portions of its durable state for display.
It is not a separate daemon or a cloud execution service.

The design supports a large population of logical agents with a bounded number
of active model requests, tools and editing workspaces. A parked agent is a
SQLite record, not a live GPUI entity, operating-system thread, model stream or
worktree. The default logical limit is 75,000 agents **per root run**. This is a
configured storage/admission limit, not evidence that 75,000 agents can execute
inference simultaneously or that every editor workload fits in memory.

## Application lifecycle and ownership

`NativeRuntime` is application-owned and starts lazily. It opens
`agent-runtime.sqlite` in Zed's data directory and runs Infinity on a dedicated
Tokio `LocalSet`. SQLite has an exclusive process-owner lock, WAL with full
synchronization, one bounded writer and two bounded reader workers. GPUI callers
do not need their own Tokio runtime.

The runtime runs only while the application is open. Pause prevents further
admission and lets active slices reach their boundary. Stop invalidates the
selected agent generation and its descendants, cancels owned inference/tool
execution and rejects late results. App shutdown cancels execution and checkpoints
the store. Concurrent shutdown callers await the same result.

Reopening the database leaves unfinished agents paused. The user explicitly
resumes them. Dispatched effects whose outcome cannot be established become
`uncertain`; resume does not authorize executing them again. Recovery APIs require
the effect owner to reconcile the result or obtain the existing permission
decision. Dirty editing leases and interrupted integrations also require explicit
recovery rather than automatic reuse.

The fleet inspector lists up to 20 interrupted tools for the selected agent.
After inspecting possible effects, the user can choose **Continue without
retrying**. This records a correlated error acknowledging an unknown outcome;
it does not claim success, replay the tool or resume the agent automatically.
Explicit resume or a new user turn remains necessary.

## Scheduling, orchestration and durable effects

`RuntimeStore::admit` grants one generation-fenced slice lease per agent. A slice
loads a bounded effective history, processes an atomic input batch and executes
one Infinity model round. Its loaded execution objects are released afterwards.
Admission is bounded and fair between root runs. There is no task per parked
agent. Empty admission and job scans remain cached until durable eligibility
changes, so an unchanged blocked fleet is not rescanned on every worker tick.

Agents have stable IDs, parent IDs and immutable inherited-history cutoffs. The
orchestrator stores dependency joins and structured child reports in SQLite,
rejects cycles and cross-run joins, and supports collect and fail-fast policies.
Waiting on children parks the parent. A child that finishes without pending
effects reports its final text automatically; explicit reports can carry a
structured result. Native coordination tools are `runtime_spawn`, `runtime_wait`,
`runtime_reports` and `runtime_report`.

Tool calls and their durable intents commit with history before dispatch. The
native tool host then uses Zed's existing tool implementations, approval event
streams and sandbox policy. Effects for one agent are serialized; independent
agents can use the shared bounded tool pool. The next model round waits until
the same-generation tool jobs and joins are resolved. Resource waiters park
durably; revision-aware dispatch avoids repeatedly scanning an unchanged fleet.

History append and processed input IDs share one SQLite transaction. No database
transaction spans inference or external tool execution. External effects cannot
be made exactly once by a local transaction: duplicate result delivery is
idempotent, but interrupted dispatched work still needs reconciliation.

## Native Zed behavior

The host reuses registered providers and credentials, selected root/subagent
models, model parameters, profiles, instructions, AGENTS files, skills, tool
permissions, MCP transports/authentication and terminal sandbox behavior. Live
instruction and skill context is prepared on admission. Profile and tool schemas
are checked again at dispatch. MCP tools retain stable aliases and are exposed
only when enabled by the current profile.

Filesystem, buffer, editing and LSP adapters call the existing Zed implementations.
Leased projects retain their own edit action log, MCP context and code-action
handles. Code-action IDs are bounded and checked against the document snapshot.
Approval requests remain owned by the application when a view closes; the
runtime never treats a detached view as permission to proceed.

New runtime sessions use an ACP facade rather than the legacy `Thread` execution
loop. The selected conversation materializes a bounded transcript projection;
the fleet panel pages 100 agent rows and 50 transcript entries and coalesces
refreshes. Model, profile, thinking, effort and speed controls use configuration
leases: changes are validated while the affected subtree is idle and admission
is blocked until the change commits or aborts. Concurrent edits cannot change
configuration underneath an active slice.

## Isolated editing and integration

Editing uses a bounded pool of detached Git worktrees. Agents acquire a
generation-fenced workspace lease, operate through its native Zed project,
checkpoint an artifact and release a clean workspace. The pool is shared across
the fleet; it does not create one worktree per parked agent. Artifact refs keep
checkpoint commits reachable across cleanup and recovery.

The workspace and integration tools are:

- `runtime_workspace_acquire`, `runtime_workspace_checkpoint` and
  `runtime_workspace_release`;
- `runtime_integrate` and `runtime_revert` when configured and permitted.

Integration runs the user's configured checks in an isolated staging worktree,
then serializes publication to the target branch with compare-and-swap checks.
Failures or conflicts leave the target unchanged. A selective revert validates
the inverse change while preserving unrelated later work. Integration checks and
the target branch come from application settings, not model-supplied arguments.
Automatic integration runs after a successful checkpoint only when checks are
configured and the current profile permits integration. Disabling
`runtime_integrate` also disables that automatic path.

Isolated editing currently requires a trusted **local Git project**. Remote and
non-Git projects do not gain equivalent editing isolation from this runtime.
Restricted worktrees retain metadata-only orchestration. Missing checks disable
integration; they do not cause unchecked publication.

## Providers, context and accounting

`PreparedLanguageModelCompletion` is the shared provider execution seam. Zed
resolves configuration on the foreground executor; owned request/stream futures
run on the embedded worker. Explicitly unavailable models fail instead of falling
back silently. AI-disable settings apply at configuration and preparation.

The vendored Infinity crates retain their Apache-2.0 license and pinned upstream
revision; see their `UPSTREAM.md` files. Local protocol extensions preserve native
reasoning details, signed/redacted reasoning, raw and custom tool input, tool
signatures, native compaction state and tool-result error/output data. Parallel
asynchronous tool calls are dispatched only after the durable history barrier.
Unsigned reasoning and complete parallel response handling are opt-in provider
capabilities, preserving upstream behavior for other providers.

Provider attempts and exact native token usage are recorded durably. Partial
inference is not a checkpoint: the original native stream must finish
successfully, including usage events after a stop event. Errors, truncation and
unsupported state pause the agent without committing a partial response. Dropping
the slice drops the original provider stream.

Manual `/compact` and configured automatic compaction use the native compaction
model/provider path. Compaction writes an effective-context checkpoint while
retaining the immutable archive and inherited-history boundaries. Installation
is fenced against generation and configuration changes. Native explicit
compaction is used when supported; otherwise bounded transcript pages feed the
summary path. Typed provider context-overflow state can trigger recovery without
matching error strings.

The generic bridge accepts base64 PNG images. URL images require an authorized
tool to resolve them before inference; non-PNG input needs conversion. Foreign
protocol-only secondary call/result IDs that cannot be represented by the native
provider state fail explicitly. Preserving native provider fields is not a claim
that every registered provider has been exercised against its live service.

## Settings and bounds

Configure these under `agent.runtime` in Zed settings:

| Setting | Default | Meaning |
| --- | --- | --- |
| `enabled` | `true` | Use the embedded runtime for new native sessions |
| `max_agents` | `75000` | Logical agents in one root run |
| `max_active_agents` | `16` | Concurrent admitted inference slices |
| `max_tool_jobs` | `8` | Concurrent native tool jobs |
| `max_workspaces` | `4` | Editing workspace leases held at once |
| `max_queued_jobs` | `256` | Bounded admitted command/job capacity |
| `max_memory_bytes` | `536870912` | Shared retained execution-byte budget |
| `integration_checks` | empty | Check programs, argument arrays and timeouts |
| `target_branch` | current branch | Integration destination; required for detached HEAD |
| `auto_integrate` | `true` | Integrate checkpoints when checks and permissions allow |

Limits must be positive; the native tool host accepts 1–256 active tool jobs.
Active inference is capped by the logical-agent limit.
An already-open runtime rejects a different database/limit configuration; change
startup limits before starting it or restart the application. Model/profile
controls use the separate per-run configuration lease path described above.

The execution-byte budget accounts for retained runtime work; it is **not** a
hard cap on whole-editor RSS, provider-library allocations, language servers or
filesystem caches. Effective history is preflighted against a 32 MiB budget before
deserialization. Other guards include 1 MiB per input, 32 inputs per slice, 8 MiB
retained model output and at most `min(256, max_inputs_per_slice)` completed tool
calls per response (32 by default). An oversized response pauses before any of
its history or jobs commit, so sibling results fit one bounded input slice.
Resource admission and
provider cooldown prevent these independently bounded operations from all
becoming active at once.

## Validation and remaining coverage

Useful focused commands from the repository root:

```sh
cargo test -p agent_runtime --lib
cargo test -p infinity-agent-core -p infinity-provider-protocol --lib
cargo test -p agent --lib runtime_
cargo check -p agent -p agent_ui
./script/clippy -p agent_runtime
cargo test -p agent_runtime --lib -- --ignored --nocapture
```

The explicit capacity tests seed 75,000 parked metadata/jobs, exercise bounded
pagination, dispatch-scan caching and recovery/admission, and report measurements
for their synthetic workload. They use no live model requests. Native integration
tests use Zed's registered fake provider and real Infinity/SQLite paths, including
tool approvals, project isolation, cancellation and paused recovery.

These tests do not establish 75,000 simultaneous model streams, a full editor RAM
ceiling or production throughput. Live-provider matrices, prolonged full-editor
load, remote editing parity, and platform-specific process-tree termination need
separate validation. Unix process groups are used for owned Git/check processes;
equivalent Windows descendant-process containment has not been established by
the Linux test suite. Upstream socket-transport tests also require an environment
that permits local socket binding.
