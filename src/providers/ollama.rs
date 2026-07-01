use serde_json::{Value, json};

use crate::{LLMError, LLMRequest, LLMResponse, Message, ToolCall, Usage};

/// Client for a local [Ollama](https://ollama.com) server's chat API.
pub struct OllamaClient {
    base_url: String,
    client: reqwest::Client,
}

impl OllamaClient {
    /// Create a client with an optional base URL override (defaults to
    /// `http://localhost:11434`). For more options use [`OllamaClient::builder`].
    pub fn new(base_url: Option<String>) -> Self {
        let mut builder = Self::builder();
        builder.base_url = base_url;
        builder.build()
    }

    /// Start building a provider. Ollama needs no API key; the base URL and HTTP
    /// client are optional.
    pub fn builder() -> OllamaClientBuilder {
        OllamaClientBuilder {
            base_url: None,
            client: None,
        }
    }

    pub(crate) fn build_body(&self, req: &LLMRequest) -> Value {
        let mut messages: Vec<Value> = Vec::new();
        messages.push(json!({ "role": "system", "content": req.system }));
        for msg in &req.messages {
            messages.push(map_message(msg));
        }

        let mut body = json!({
            "model": req.model,
            "stream": false,
            "messages": messages,
        });

        if !req.tools.is_empty() {
            let tools: Vec<Value> = req
                .tools
                .iter()
                .map(|t| {
                    json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.input_schema,
                        }
                    })
                })
                .collect();
            body["tools"] = Value::Array(tools);
        }

        body
    }

    pub(crate) fn parse_response(json: &Value) -> Result<LLMResponse, LLMError> {
        let message = json
            .get("message")
            .ok_or_else(|| LLMError::Provider("missing message".to_string()))?;

        let text = message
            .get("content")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(str::to_string);

        let tool_calls = message
            .get("tool_calls")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .enumerate()
                    .filter_map(|(i, tc)| {
                        let func = tc.get("function")?;
                        let name = func.get("name")?.as_str()?.to_string();
                        let input = func.get("arguments")?.clone();
                        let id = format!("call_{i}");
                        Some(ToolCall { id, name, input })
                    })
                    .collect()
            })
            .unwrap_or_default();

        let stop_reason = json.get("done_reason").and_then(Value::as_str).unwrap_or("stop").to_string();

        let usage = Usage {
            input_tokens: json.get("prompt_eval_count").and_then(Value::as_u64).unwrap_or(0),
            output_tokens: json.get("eval_count").and_then(Value::as_u64).unwrap_or(0),
        };

        Ok(LLMResponse {
            text,
            tool_calls,
            usage,
            stop_reason,
        })
    }
}

/// Builder for [`OllamaClient`].
pub struct OllamaClientBuilder {
    base_url: Option<String>,
    client: Option<reqwest::Client>,
}

impl OllamaClientBuilder {
    /// Override the API base URL (defaults to `http://localhost:11434`).
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Override the [`reqwest::Client`] used for requests.
    pub fn client(mut self, client: reqwest::Client) -> Self {
        self.client = Some(client);
        self
    }

    /// Build the [`OllamaClient`].
    pub fn build(self) -> OllamaClient {
        OllamaClient {
            base_url: self.base_url.unwrap_or_else(|| "http://localhost:11434".to_string()),
            client: self.client.unwrap_or_default(),
        }
    }
}

fn map_message(msg: &Message) -> Value {
    match msg {
        Message::User(s) => json!({ "role": "user", "content": s }),
        Message::Assistant { text, tool_calls } => {
            let content = text.as_deref().unwrap_or("");
            if tool_calls.is_empty() {
                json!({ "role": "assistant", "content": content })
            } else {
                let tc: Vec<Value> = tool_calls
                    .iter()
                    .map(|call| {
                        json!({
                            "function": {
                                "name": call.name,
                                "arguments": call.input,
                            }
                        })
                    })
                    .collect();
                json!({
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tc,
                })
            }
        },
        Message::ToolResult { content, .. } => {
            json!({ "role": "tool", "content": content })
        },
    }
}

#[async_trait::async_trait]
impl crate::LLMClient for OllamaClient {
    fn name(&self) -> &str {
        "ollama"
    }

    async fn generate(&self, req: &LLMRequest) -> Result<LLMResponse, LLMError> {
        let url = format!("{}/api/chat", self.base_url);
        let body = self.build_body(req);

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| LLMError::Provider(e.to_string()))?;

        let status = resp.status();
        let json: Value = resp.json().await.map_err(|e| LLMError::Provider(e.to_string()))?;

        if !status.is_success() {
            let msg = json.get("error").and_then(Value::as_str).unwrap_or("unknown error");
            return Err(LLMError::Provider(format!("HTTP {status}: {msg}")));
        }

        Self::parse_response(&json)
    }
}
