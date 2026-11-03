# Embedded agent runtime

This crate is the first implementation of Zezu's native concurrent agent runtime.
It embeds Infinity at revision `1abc6fcff1f186124f924b56aa92c3a3684c90c5` on a
dedicated Tokio `LocalSet`, with SQLite ownership separate from GPUI.

## Current integration status

The runtime executes actual Infinity steps against durable SQLite stores. It is
a workspace library, not a separate daemon, external ACP agent, or cloud service.
`NativeAgentServer` does not yet instantiate it. The existing native agent loop
continues to run normally, using the new owned completion preparation boundary.
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

`PreparedLanguageModelCompletion` in `language_model_core` is the initial shared
provider execution seam. Zed still owns provider preparation and configuration;
an owned execution can cross threads. This is not yet the full provider-to-Infinity
request, capability and stream conversion layer.

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

1. Connect the runtime owner to the app lifecycle and native connection facade,
   with configured model selection, error presentation and permission handling.
2. Convert Zed and Infinity requests and stream events without losing provider
   capabilities, multimodal data, usage, reasoning or cancellation semantics.
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
./script/clippy -p agent_runtime
cargo test -p agent_runtime large_parked_fleet_uses_paged_metadata -- --ignored --nocapture
```

The explicit capacity test seeds 75,000 parked metadata records and checks bounded
pagination, idle admission and enforcement of the total-agent limit. It does not
run 75,000 inference streams, create 75,000 worktrees, measure total editor memory,
or establish a supported production capacity.
