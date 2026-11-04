Vendored from hydro-project/infinity revision 1abc6fcff1f186124f924b56aa92c3a3684c90c5, crates/infinity-provider-protocol.

Local extension: ReasoningContent::ProviderState is a structured provider-owned state block. It is retained in durable conversation history and omitted from display text. Zed uses a versioned namespace to round-trip native reasoning, compaction and tool metadata without putting opaque provider state into model-visible text.

ToolResult also has serde-defaulted, omitted-when-empty `is_error` and `output` fields so native tool failures and structured output survive durable replay. Legacy successful results retain their original JSON shape.

Original Apache-2.0 license is preserved in LICENSE.txt. Optional transport features are unchanged.
