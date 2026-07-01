/// A single tool call requested by the model.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ToolCall {
    /// Provider-assigned identifier for this call, echoed back in the matching
    /// [`Message::ToolResult::tool_use_id`] so the model can correlate them.
    pub id: String,
    /// Name of the tool the model wants to invoke; matches a [`ToolDef::name`].
    pub name: String,
    /// Arguments for the call, shaped according to the tool's
    /// [`ToolDef::input_schema`].
    pub input: serde_json::Value,
}

/// A message in the conversation.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Message {
    /// A user turn carrying plain text.
    User(String),
    /// Assistant turn: optional text and zero or more tool calls.
    Assistant {
        /// The assistant's text reply, if any. `None` when the turn contains
        /// only tool calls.
        text: Option<String>,
        /// Tool calls the model requested on this turn; empty for a plain reply.
        tool_calls: Vec<ToolCall>,
    },
    /// Result of a tool call, fed back to the model.
    ToolResult {
        /// The [`ToolCall::id`] this result answers.
        tool_use_id: String,
        /// The tool's output, returned to the model as text.
        content: String,
        /// Whether `content` represents an error rather than a successful result.
        is_error: bool,
    },
}

/// Describes a tool the model may call.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ToolDef {
    /// Unique tool name the model uses to invoke it.
    pub name: String,
    /// Natural-language description of what the tool does; the model uses this
    /// to decide when to call it.
    pub description: String,
    /// JSON Schema object describing the tool's `input` parameter.
    pub input_schema: serde_json::Value,
}

/// Token usage for one provider round-trip.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Usage {
    /// Number of prompt (input) tokens billed for the request.
    pub input_tokens: u64,
    /// Number of completion (output) tokens generated in the response.
    pub output_tokens: u64,
}

impl Usage {
    /// Add another round-trip's token counts into this one.
    pub fn append(&mut self, other: Usage) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
    }
}

/// The full request sent to a provider.
#[derive(Debug, Clone)]
pub struct LLMRequest {
    /// Provider-specific model identifier (e.g. `"gpt-4o"`, `"claude-sonnet-4-6"`).
    pub model: String,
    /// System prompt sent to the model. Use an empty string for no system prompt.
    pub system: String,
    /// Conversation history, oldest first.
    pub messages: Vec<Message>,
    /// Tools the model is allowed to call; empty to disable tool use.
    pub tools: Vec<ToolDef>,
    /// Upper bound on tokens the model may generate in its response.
    pub max_tokens: u32,
}

/// The provider's response.
#[derive(Debug, Clone)]
pub struct LLMResponse {
    /// The model's text output, if any. `None` when the turn produced only
    /// tool calls.
    pub text: Option<String>,
    /// Tool calls the model requested; empty for a plain text reply.
    pub tool_calls: Vec<ToolCall>,
    /// Token usage reported by the provider for this round-trip.
    pub usage: Usage,
    /// Why generation stopped, e.g. `"end_turn"`, `"tool_use"`, `"max_tokens"`.
    pub stop_reason: String,
}
