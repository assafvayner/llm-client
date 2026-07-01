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
#[non_exhaustive]
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
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
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
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
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

impl LLMRequest {
    /// Start building a request for the given model. The default `system` prompt
    /// is empty, `messages` and `tools` are empty, and `max_tokens` defaults to
    /// `1024`.
    pub fn builder(model: impl Into<String>) -> LLMRequestBuilder {
        LLMRequestBuilder {
            model: model.into(),
            system: String::new(),
            messages: vec![],
            tools: vec![],
            max_tokens: 1024,
        }
    }
}

/// Builder for [`LLMRequest`].
pub struct LLMRequestBuilder {
    model: String,
    system: String,
    messages: Vec<Message>,
    tools: Vec<ToolDef>,
    max_tokens: u32,
}

impl LLMRequestBuilder {
    /// Set the system prompt.
    pub fn system(mut self, system: impl Into<String>) -> Self {
        self.system = system.into();
        self
    }

    /// Append one message to the conversation history.
    pub fn message(mut self, message: Message) -> Self {
        self.messages.push(message);
        self
    }

    /// Append messages to the conversation history.
    pub fn messages(mut self, messages: impl IntoIterator<Item = Message>) -> Self {
        self.messages.extend(messages);
        self
    }

    /// Append one tool the model is allowed to call.
    pub fn tool(mut self, tool: ToolDef) -> Self {
        self.tools.push(tool);
        self
    }

    /// Append tools the model is allowed to call.
    pub fn tools(mut self, tools: impl IntoIterator<Item = ToolDef>) -> Self {
        self.tools.extend(tools);
        self
    }

    /// Set the upper bound on tokens the model may generate. Defaults to
    /// `1024` when not set.
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Build the [`LLMRequest`].
    pub fn build(self) -> LLMRequest {
        LLMRequest {
            model: self.model,
            system: self.system,
            messages: self.messages,
            tools: self.tools,
            max_tokens: self.max_tokens,
        }
    }
}

/// The provider's response.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
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

impl LLMResponse {
    /// Construct a response from its fields.
    pub fn new(text: Option<String>, tool_calls: Vec<ToolCall>, usage: Usage, stop_reason: impl Into<String>) -> Self {
        Self {
            text,
            tool_calls,
            usage,
            stop_reason: stop_reason.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn request_and_response_round_trip_through_serde() {
        let req = LLMRequest::builder("test-model")
            .system("sys")
            .message(Message::User("hi".into()))
            .tool(ToolDef {
                name: "run_query".into(),
                description: "Run a query".into(),
                input_schema: json!({ "type": "object" }),
            })
            .max_tokens(64)
            .build();
        let req2: LLMRequest = serde_json::from_str(&serde_json::to_string(&req).unwrap()).unwrap();
        assert_eq!(req, req2);

        let resp = LLMResponse::new(
            Some("answer".into()),
            vec![ToolCall {
                id: "c1".into(),
                name: "run_query".into(),
                input: json!({ "sql": "SELECT 1" }),
            }],
            Usage {
                input_tokens: 10,
                output_tokens: 5,
            },
            "tool_use",
        );
        let resp2: LLMResponse = serde_json::from_str(&serde_json::to_string(&resp).unwrap()).unwrap();
        assert_eq!(resp, resp2);
    }
}
