/// A single tool call requested by the model.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub input: serde_json::Value,
}

/// A message in the conversation.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Message {
    User(String),
    /// Assistant turn: optional text and zero or more tool calls.
    Assistant {
        text: Option<String>,
        tool_calls: Vec<ToolCall>,
    },
    /// Result of a tool call, fed back to the model.
    ToolResult {
        tool_use_id: String,
        content: String,
        is_error: bool,
    },
}

/// Describes a tool the model may call.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    /// JSON Schema object describing the tool's `input` parameter.
    pub input_schema: serde_json::Value,
}

/// Token usage for one provider round-trip.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Usage {
    pub input_tokens: u64,
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
    pub model: String,
    pub system: String,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDef>,
    pub max_tokens: u32,
}

/// The provider's response.
#[derive(Debug, Clone)]
pub struct LLMResponse {
    pub text: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub usage: Usage,
    /// e.g. `"end_turn"`, `"tool_use"`, `"max_tokens"`
    pub stop_reason: String,
}
