# Embedded agent runtime

This crate is the first implementation of Zezu's native concurrent agent runtime.
It embeds Infinity at revision `1abc6fcff1f186124f924b56aa92c3a3684c90c5` on a
dedicated Tokio `LocalSet`, with SQLite ownership separate from GPUI.

## Current integration status

The runtime executes actual Infinity steps against durable SQLite stores. It is
a workspace library, not a separate daemon, external ACP agent, or cloud service.
`NativeAgentServer` attaches an app-owned `NativeRuntime` host to native
connections. The host starts its database and worker lazily through its native
API. The existing interactive agent loop continues to run normally, using the
owned completion preparation boundary; the panel does not submit to Infinity yet.
The complete K1–K7 and N1–N10 integration is not finished by this change.

Implemented foundations:

- Stable agent IDs, parent relationships and immutable history cutoffs.
- A bounded database writer and two bounded reader workers; callers can await
  them without a Tokio runtime on the calling thread.
- SQLite WAL with full synchronization and an exclusive process owner lock.
- Per-agent input ordering, durable input IDs and a single active slice lease.
- Bounded active slices and fairness between root runs, with durable scheduling
  metadata and no per-parked-agent task or UI entity.
- Idempotent child creation and a transactional run-wide agent count.
- Durable tool intents committed with completed tool-call history, before dispatch.
- Separate job dispatch/completion, duplicate result handling and explicit
  reconciliation for uncertain effects.
- Cancellation generations, subtree cancellation, graceful pause requests and
  restart in paused state.
- Paginated agent metadata queries and fixed parent history references.
- Completed-response guards around Infinity's step API. Partial inference is not
  a durable execution checkpoint.
- A bounded preparation bridge to Zed's registered model providers. Requests use
  the existing provider configuration, credentials and provider-side limits.
- Root and subagent model selection, profile model defaults, supported thinking,
  effort, speed, temperature and output limits from Zed's existing settings.
- Native app ownership independent of view handles, synchronous quit cancellation
  and a shared shutdown result for concurrent shutdown callers.

`PreparedLanguageModelCompletion` in `language_model_core` is the shared provider
execution seam. Zed resolves configuration on the foreground executor; the owned
execution and stream are polled on the runtime worker. Only admitted slices have
preparation requests, and both host queues have a fixed capacity of 256.

## Native host API and provider coverage

`NativeAgentConnection::runtime` exposes the shared host. Call
`NativeRuntime::start_database(path, limits, cx)` before `submit_text(config, text)`.
The native API returns the run ID and exposes the existing store operations and
bounded runtime notices. Repeated startup accepts the same path and limits and
rejects another database/configuration. Opening the database happens off the UI
thread. Recovered agents remain paused; starting the host does not resume them.

`NativeRuntimeConfig` persists model/profile identifiers and an assembled system
prompt, without credentials. The caller must use the existing project instruction
and skill loaders to assemble that prompt. This batch applies profile model
defaults, not profile tool permissions. Children inherit the run prompt and use
Zed's configured subagent model when set. Explicit unavailable models fail; they
never silently fall back to another provider. AI-disable settings are checked
both at configuration resolution and provider preparation.

The bridge supports text, JSON function calls and correlated results, base64 PNG
images, plain/signed thinking, redacted thinking, and token usage. Complete tool
arguments are emitted once. A stop event alone is insufficient to checkpoint:
the native stream must finish successfully, including any usage events after
stop. Raw usage/status events can be observed; cache-creation usage is combined
with input usage in Infinity's narrower usage structure. Durable native usage
accounting and the panel projection remain integration work.

Unrepresentable fields fail explicitly: provider reasoning details/IDs, tool
thought signatures, compaction, custom text tool input, distinct secondary call
IDs, image URLs and non-PNG images. These need protocol extensions or authorized
conversion before full provider parity. The bridge bounds retained output to
8 MiB and 256 tool calls per response. Unsupported state, incomplete streams,
provider errors and truncated/refused output pause the agent without committing
partial inference. Dropping a slice drops the original Zed provider stream.

## Execution and recovery

`RuntimeStore::admit` claims at most one bounded input batch for an agent and
returns a generation-fenced lease. The worker constructs an `InfinitySlice`
only after admission. Each slice loads the needed history, runs one model round,
and drops its loaded execution objects afterwards.

Only registered asynchronous tool wrappers are exposed to Infinity. Its default
synchronous built-ins are disabled because they can invoke additional completions
inside one step. Each wrapper records an intent; a Zed tool adapter must authorize
and execute it before reporting completion. Recording or dispatching a job in this
crate never executes a shell command or MCP operation itself.

History and the processed IDs derived from it commit in one SQLite transaction.
Tool-call history and its outgoing job intent also commit together. This closes
the upstream split between history append and processed-ID writes for this store.
No transaction spans inference or tool execution.

External effects still cannot be made exactly once by a local transaction.
Dispatched jobs become uncertain on interruption. Reconciliation requires the tool
owner to verify what happened or obtain the existing permission decision. Resume
alone never grants permission or authorizes repeating an uncertain effect.

Stop invalidates the selected agent generation and its active descendants. A new
explicit user submission may start another turn; late results from the old turn
cannot do that. Pause lets active slices reach their boundary and prevents further
admission. Shutdown drops active inference futures, checkpoints the store and
releases its owner lock. Owned tool-process termination will be supplied by the
Zed job adapter; no process executor is installed here yet.

## Remaining integration gates

1. Connect panel submissions and error presentation to the native runtime API
   after permission handling and bounded UI projections are available.
2. Extend the protocol for the unsupported provider fields above, preserve full
   usage accounting, and verify every supported provider end to end.
3. Bind existing independent tools and narrow project, buffer and LSP bridges to
   the durable job broker. Preserve instruction, skill, MCP and approval behavior.
4. Add parent reports and dependency joins. Existing child IDs and inherited
   history provide storage primitives; a complete orchestration graph is not
   implemented yet.
5. Replace eager child views and native-thread access with bounded projections.
6. Add workspace leases, recoverable editing artifacts, configured checks and
   serialized integration with selective revert.
7. Wire resource settings, tool/workspace admission and memory accounting. Tune
   adaptive caches and resource defaults using representative workloads.
8. Exercise crash injection, live-provider paths and the full editor under load.

## Validation

```sh
cargo test -p agent_runtime -p language_model_core --lib
cargo test -p agent --lib native_runtime::tests
./script/clippy -p agent_runtime
cargo test -p agent_runtime large_parked_fleet_uses_paged_metadata -- --ignored --nocapture
```

Native integration tests drive Zed's fake registered provider through the actual
Infinity worker and SQLite store. They cover detached handles, configured root
and subagent models, provider-state rejection, and app-quit cancellation with a
paused reopen. No live model credentials or paid requests are used.

The explicit capacity test seeds 75,000 parked metadata records and checks bounded
pagination, idle admission and enforcement of the total-agent limit. It does not
run 75,000 inference streams, create 75,000 worktrees, measure total editor memory,
or establish a supported production capacity.
