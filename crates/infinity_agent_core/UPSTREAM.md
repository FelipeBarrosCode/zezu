Vendored from hydro-project/infinity revision 1abc6fcff1f186124f924b56aa92c3a3684c90c5, crates/infinity-agent-core. Apache-2.0 license is retained in LICENSE.txt.

Local changes preserve structured provider state and opt-in complete native responses, including unsigned reasoning and parallel asynchronous tool calls. Deferred actions are collected and dispatched after the single durable history barrier. Pending tool matching walks all outstanding calls across assistant continuations, while user/synthetic input boundaries still fence obsolete results.

JSON object ordering explicitly matches the native workspace's `preserve_order` setting, including when this package is tested alone.

The native host must admit a follow-up model round only after every same-generation dispatched tool job has reached a terminal state. Upstream synchronous built-ins remain disabled in the embedded Zezu runtime.
