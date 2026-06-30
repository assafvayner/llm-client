use futures_util::StreamExt;
use serde_json::{Value, json};

use crate::{LLMError, LLMRequest, LLMResponse, Message, ToolCall, Usage};

pub struct ClaudeProvider {
    api_key: String,
    base_url: String,
    client: reqwest::Client,
}

impl ClaudeProvider {
    pub fn new(api_key: String, base_url: Option<String>) -> Self {
        let mut builder = Self::builder(api_key);
        builder.base_url = base_url;
        builder.build()
    }

    /// Start building a provider. The API key is required; the base URL and HTTP
    /// client are optional.
    pub fn builder(api_key: impl Into<String>) -> ClaudeProviderBuilder {
        ClaudeProviderBuilder {
            api_key: api_key.into(),
            base_url: None,
            client: None,
        }
    }

    pub fn from_env() -> Result<Self, LLMError> {
        let key = std::env::var("ANTHROPIC_API_KEY")
            .map_err(|_| LLMError::Provider("ANTHROPIC_API_KEY not set".to_string()))?;
        Ok(Self::new(key, None))
    }

    pub fn build_body(&self, req: &LLMRequest) -> Value {
        let messages: Vec<Value> = req.messages.iter().map(map_message).collect();

        let tools: Vec<Value> = req
            .tools
            .iter()
            .map(|t| {
                json!({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                })
            })
            .collect();

        json!({
            "model": req.model,
            "max_tokens": req.max_tokens,
            "system": req.system,
            "tools": tools,
            "messages": messages,
        })
    }

    pub fn parse_response(json: &Value) -> Result<LLMResponse, LLMError> {
        let content = json
            .get("content")
            .and_then(Value::as_array)
            .ok_or_else(|| LLMError::Provider("missing content array".to_string()))?;

        let mut text_parts: Vec<&str> = Vec::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();

        for block in content {
            match block.get("type").and_then(Value::as_str) {
                Some("text") => {
                    if let Some(t) = block.get("text").and_then(Value::as_str) {
                        text_parts.push(t);
                    }
                },
                Some("tool_use") => {
                    let id = block.get("id").and_then(Value::as_str).unwrap_or("").to_string();
                    let name = block.get("name").and_then(Value::as_str).unwrap_or("").to_string();
                    let input = block.get("input").cloned().unwrap_or(Value::Null);
                    tool_calls.push(ToolCall { id, name, input });
                },
                // Ignore "thinking" and any other block types.
                _ => {},
            }
        }

        let text = if text_parts.is_empty() {
            None
        } else {
            Some(text_parts.join(""))
        };

        let stop_reason = json.get("stop_reason").and_then(Value::as_str).unwrap_or("").to_string();

        let usage = {
            let u = json.get("usage");
            Usage {
                input_tokens: u.and_then(|v| v.get("input_tokens")).and_then(Value::as_u64).unwrap_or(0),
                output_tokens: u.and_then(|v| v.get("output_tokens")).and_then(Value::as_u64).unwrap_or(0),
            }
        };

        Ok(LLMResponse {
            text,
            tool_calls,
            usage,
            stop_reason,
        })
    }
}

/// Builder for [`ClaudeProvider`].
pub struct ClaudeProviderBuilder {
    api_key: String,
    base_url: Option<String>,
    client: Option<reqwest::Client>,
}

impl ClaudeProviderBuilder {
    /// Override the API base URL (defaults to `https://api.anthropic.com`).
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Override the [`reqwest::Client`] used for requests.
    pub fn client(mut self, client: reqwest::Client) -> Self {
        self.client = Some(client);
        self
    }

    /// Build the [`ClaudeProvider`].
    pub fn build(self) -> ClaudeProvider {
        ClaudeProvider {
            api_key: self.api_key,
            base_url: self.base_url.unwrap_or_else(|| "https://api.anthropic.com".to_string()),
            client: self.client.unwrap_or_default(),
        }
    }
}

fn map_message(msg: &Message) -> Value {
    match msg {
        Message::User(s) => json!({ "role": "user", "content": s }),
        Message::Assistant { text, tool_calls } => {
            let mut content: Vec<Value> = Vec::new();
            if let Some(t) = text
                && !t.is_empty()
            {
                content.push(json!({ "type": "text", "text": t }));
            }
            for call in tool_calls {
                content.push(json!({
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": call.input,
                }));
            }
            json!({ "role": "assistant", "content": content })
        },
        Message::ToolResult {
            tool_use_id,
            content,
            is_error,
        } => {
            json!({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": content,
                    "is_error": is_error,
                }]
            })
        },
    }
}

#[async_trait::async_trait]
impl crate::LLMClient for ClaudeProvider {
    fn name(&self) -> &str {
        "claude"
    }

    async fn generate(&self, req: &LLMRequest) -> Result<LLMResponse, LLMError> {
        let url = format!("{}/v1/messages", self.base_url);
        let body = self.build_body(req);

        let resp = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| LLMError::Provider(e.to_string()))?;

        let status = resp.status();
        let json: Value = resp.json().await.map_err(|e| LLMError::Provider(e.to_string()))?;

        if !status.is_success() {
            let msg = json
                .get("error")
                .and_then(|e| e.get("message"))
                .and_then(Value::as_str)
                .unwrap_or("unknown error");
            return Err(LLMError::Provider(format!("HTTP {status}: {msg}")));
        }

        Self::parse_response(&json)
    }
}

#[async_trait::async_trait]
impl crate::LLMStreamingClient for ClaudeProvider {
    async fn stream(&self, req: &LLMRequest, on_text: &mut crate::TextSink<'_>) -> Result<LLMResponse, LLMError> {
        let url = format!("{}/v1/messages", self.base_url);
        let mut body = self.build_body(req);
        body["stream"] = serde_json::Value::Bool(true);

        let resp = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| LLMError::Provider(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(LLMError::Provider(format!("HTTP {status}: {text}")));
        }

        let mut acc = AnthropicStreamAcc::new();
        let mut sse = crate::streaming::SseBuffer::new();
        let mut stream = resp.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| LLMError::Provider(e.to_string()))?;
            sse.push(&chunk);
            while let Some(data) = sse.next_event() {
                acc.handle(&data, on_text);
            }
        }
        Ok(acc.finish())
    }
}

/// Assembles an Anthropic streaming response from SSE event payloads.
#[derive(Default)]
struct AnthropicStreamAcc {
    text: String,
    stop_reason: String,
    usage: Usage,
    /// content block index -> partial tool block
    blocks: std::collections::BTreeMap<u64, ToolBlock>,
}

#[derive(Default)]
struct ToolBlock {
    id: String,
    name: String,
    json: String,
}

impl AnthropicStreamAcc {
    fn new() -> Self {
        Self::default()
    }

    /// Handle one SSE `data:` payload (a JSON object). Unknown/`[DONE]`/`ping`
    /// payloads are ignored. Text deltas are forwarded to `on_text`.
    fn handle(&mut self, data: &str, on_text: &mut dyn FnMut(&str)) {
        if data.trim().is_empty() || data.trim() == "[DONE]" {
            return;
        }
        let Ok(v) = serde_json::from_str::<Value>(data) else {
            return;
        };
        match v.get("type").and_then(Value::as_str) {
            Some("message_start") => {
                if let Some(u) = v.get("message").and_then(|m| m.get("usage")) {
                    self.usage.input_tokens = u.get("input_tokens").and_then(Value::as_u64).unwrap_or(0);
                    self.usage.output_tokens = u.get("output_tokens").and_then(Value::as_u64).unwrap_or(0);
                }
            },
            Some("content_block_start") => {
                let idx = v.get("index").and_then(Value::as_u64).unwrap_or(0);
                let block = v.get("content_block");
                if block.and_then(|b| b.get("type")).and_then(Value::as_str) == Some("tool_use") {
                    self.blocks.insert(
                        idx,
                        ToolBlock {
                            id: block
                                .and_then(|b| b.get("id"))
                                .and_then(Value::as_str)
                                .unwrap_or("")
                                .to_string(),
                            name: block
                                .and_then(|b| b.get("name"))
                                .and_then(Value::as_str)
                                .unwrap_or("")
                                .to_string(),
                            json: String::new(),
                        },
                    );
                }
            },
            Some("content_block_delta") => {
                let idx = v.get("index").and_then(Value::as_u64).unwrap_or(0);
                let delta = v.get("delta");
                match delta.and_then(|d| d.get("type")).and_then(Value::as_str) {
                    Some("text_delta") => {
                        if let Some(t) = delta.and_then(|d| d.get("text")).and_then(Value::as_str) {
                            self.text.push_str(t);
                            on_text(t);
                        }
                    },
                    Some("input_json_delta") => {
                        if let Some(p) = delta.and_then(|d| d.get("partial_json")).and_then(Value::as_str)
                            && let Some(b) = self.blocks.get_mut(&idx)
                        {
                            b.json.push_str(p);
                        }
                    },
                    _ => {},
                }
            },
            Some("message_delta") => {
                if let Some(sr) = v.get("delta").and_then(|d| d.get("stop_reason")).and_then(Value::as_str) {
                    self.stop_reason = sr.to_string();
                }
                if let Some(ot) = v.get("usage").and_then(|u| u.get("output_tokens")).and_then(Value::as_u64) {
                    self.usage.output_tokens = ot;
                }
            },
            _ => {},
        }
    }

    /// Finalize into an [`LLMResponse`].
    fn finish(self) -> LLMResponse {
        let tool_calls = self
            .blocks
            .into_values()
            .map(|b| ToolCall {
                id: b.id,
                name: b.name,
                input: serde_json::from_str(&b.json).unwrap_or(Value::Null),
            })
            .collect();
        LLMResponse {
            text: if self.text.is_empty() { None } else { Some(self.text) },
            tool_calls,
            usage: self.usage,
            stop_reason: self.stop_reason,
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn anthropic_accumulates_text_and_tool_call() {
        let mut acc = AnthropicStreamAcc::new();
        let mut text = String::new();
        let feed = |v: serde_json::Value, acc: &mut AnthropicStreamAcc, t: &mut String| {
            acc.handle(&v.to_string(), &mut |s| t.push_str(s));
        };
        feed(
            json!({"type":"message_start","message":{"usage":{"input_tokens":10,"output_tokens":0}}}),
            &mut acc,
            &mut text,
        );
        feed(json!({"type":"content_block_start","index":0,"content_block":{"type":"text"}}), &mut acc, &mut text);
        feed(
            json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hel"}}),
            &mut acc,
            &mut text,
        );
        feed(
            json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"lo"}}),
            &mut acc,
            &mut text,
        );
        feed(
            json!({"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"tu1","name":"run_query"}}),
            &mut acc,
            &mut text,
        );
        feed(
            json!({"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"sql\":\"SEL"}}),
            &mut acc,
            &mut text,
        );
        feed(
            json!({"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"ECT 1\"}"}}),
            &mut acc,
            &mut text,
        );
        feed(
            json!({"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":7}}),
            &mut acc,
            &mut text,
        );

        assert_eq!(text, "Hello");
        let resp = acc.finish();
        assert_eq!(resp.text.as_deref(), Some("Hello"));
        assert_eq!(resp.stop_reason, "tool_use");
        assert_eq!(resp.usage.input_tokens, 10);
        assert_eq!(resp.usage.output_tokens, 7);
        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_calls[0].name, "run_query");
        assert_eq!(resp.tool_calls[0].id, "tu1");
        assert_eq!(resp.tool_calls[0].input, serde_json::json!({ "sql": "SELECT 1" }));
    }

    #[test]
    fn anthropic_orphan_input_delta_produces_no_tool_call() {
        let mut acc = AnthropicStreamAcc::new();
        // input_json_delta for index 0 with no content_block_start first
        acc.handle(
            &serde_json::json!({"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"sql\":\"X\"}"}}).to_string(),
            &mut |_| {},
        );
        let resp = acc.finish();
        assert!(resp.tool_calls.is_empty(), "orphan delta must not create a tool call");
    }
}
