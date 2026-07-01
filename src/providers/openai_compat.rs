//! The OpenAI-compatible chat-completions dialect.
//!
//! Holds the shared request/response mapping, the HTTP round-trip helpers, and
//! the streaming accumulator used by every provider that speaks this dialect
//! (`OpenAIClient`, `GeminiClient`, `HfClient`), plus [`OpenAICompatClient`]
//! — a generic client for any OpenAI-compatible endpoint.

use futures_util::StreamExt;
use serde_json::{Value, json};

use crate::{LLMError, LLMRequest, LLMResponse, Message, ToolCall, Usage};

/// Build an OpenAI-compatible chat completions request body.
pub(crate) fn openai_build_body(req: &LLMRequest) -> Value {
    let mut messages: Vec<Value> = Vec::new();
    messages.push(json!({ "role": "system", "content": req.system }));
    for msg in &req.messages {
        messages.push(map_message(msg));
    }

    let mut body = json!({
        "model": req.model,
        "max_tokens": req.max_tokens,
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

/// Parse an OpenAI-compatible chat completions response.
pub(crate) fn openai_parse_response(json: &Value) -> Result<LLMResponse, LLMError> {
    let choice = json
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|a| a.first())
        .ok_or_else(|| LLMError::Provider("missing choices[0]".to_string()))?;

    let message = choice
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
                .filter_map(|tc| {
                    let id = tc.get("id")?.as_str()?.to_string();
                    let func = tc.get("function")?;
                    let name = func.get("name")?.as_str()?.to_string();
                    let arguments_str = func.get("arguments").and_then(Value::as_str).unwrap_or("{}");
                    let input: Value = serde_json::from_str(arguments_str)
                        .unwrap_or_else(|_| Value::String(arguments_str.to_string()));
                    Some(ToolCall { id, name, input })
                })
                .collect()
        })
        .unwrap_or_default();

    let stop_reason = choice.get("finish_reason").and_then(Value::as_str).unwrap_or("").to_string();

    let usage = {
        let u = json.get("usage");
        Usage {
            input_tokens: u.and_then(|v| v.get("prompt_tokens")).and_then(Value::as_u64).unwrap_or(0),
            output_tokens: u.and_then(|v| v.get("completion_tokens")).and_then(Value::as_u64).unwrap_or(0),
        }
    };

    Ok(LLMResponse {
        text,
        tool_calls,
        usage,
        stop_reason,
    })
}

/// Perform an OpenAI-compatible chat-completions POST and parse the response.
///
/// The caller supplies the fully-formed endpoint `url`. `api_key`, when
/// present, is sent as a bearer token; pass `None` for keyless servers.
pub(crate) async fn openai_chat_completion(
    client: &reqwest::Client,
    url: &str,
    api_key: Option<&str>,
    req: &LLMRequest,
) -> Result<LLMResponse, LLMError> {
    let body = openai_build_body(req);

    let mut request = client.post(url).json(&body);
    if let Some(key) = api_key {
        request = request.header("Authorization", format!("Bearer {key}"));
    }
    let resp = request.send().await.map_err(|e| LLMError::Provider(e.to_string()))?;

    let status = resp.status();
    let text = resp.text().await.map_err(|e| LLMError::Provider(e.to_string()))?;

    if !status.is_success() {
        let msg = serde_json::from_str::<Value>(&text)
            .ok()
            .and_then(|j| j.get("error")?.get("message")?.as_str().map(str::to_string))
            .unwrap_or(text);
        return Err(LLMError::Http {
            status: status.as_u16(),
            message: msg,
        });
    }

    let json: Value = serde_json::from_str(&text).map_err(|e| LLMError::Provider(e.to_string()))?;
    openai_parse_response(&json)
}

/// Streaming variant of [`openai_chat_completion`]: POSTs with `stream: true`,
/// forwards text deltas to `on_text`, and assembles the full response.
pub(crate) async fn openai_chat_completion_streaming(
    client: &reqwest::Client,
    url: &str,
    api_key: Option<&str>,
    req: &LLMRequest,
    on_text: &mut crate::TextSink<'_>,
) -> Result<LLMResponse, LLMError> {
    let mut body = openai_build_body(req);
    body["stream"] = Value::Bool(true);
    body["stream_options"] = json!({ "include_usage": true });

    let mut request = client.post(url).json(&body);
    if let Some(key) = api_key {
        request = request.header("Authorization", format!("Bearer {key}"));
    }
    let resp = request.send().await.map_err(|e| LLMError::Provider(e.to_string()))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(LLMError::Http {
            status: status.as_u16(),
            message: text,
        });
    }

    let mut acc = OpenAiStreamAcc::new();
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

pub(crate) fn map_message(msg: &Message) -> Value {
    match msg {
        Message::User(s) => json!({ "role": "user", "content": s }),
        Message::Assistant { text, tool_calls } => {
            let mut obj = json!({
                "role": "assistant",
                "content": text.as_deref().unwrap_or(""),
            });
            if !tool_calls.is_empty() {
                let tc: Vec<Value> = tool_calls
                    .iter()
                    .map(|call| {
                        let arguments = serde_json::to_string(&call.input).unwrap_or_else(|_| "{}".to_string());
                        json!({
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": arguments,
                            }
                        })
                    })
                    .collect();
                obj["tool_calls"] = Value::Array(tc);
            }
            obj
        },
        Message::ToolResult {
            tool_use_id, content, ..
        } => {
            json!({
                "role": "tool",
                "tool_call_id": tool_use_id,
                "content": content,
            })
        },
    }
}

/// A generic client for any OpenAI-compatible chat-completions endpoint
/// (vLLM, LM Studio, Together, OpenRouter, local servers, …).
///
/// Unlike the preset providers, the base URL, path, and display name are all
/// caller-supplied, and the API key is optional for keyless servers.
pub struct OpenAICompatClient {
    name: String,
    url: String,
    api_key: Option<String>,
    client: reqwest::Client,
}

impl OpenAICompatClient {
    /// Create a client for an OpenAI-compatible endpoint.
    ///
    /// `base_url` is the server root (e.g. `https://api.together.xyz`) and
    /// `path` is the chat-completions path (e.g. `/v1/chat/completions`, or a
    /// bare `/chat/completions` that some servers use). `name` is the label
    /// returned by [`crate::LLMClient::name`]. `api_key` is sent as a bearer
    /// token when `Some`; pass `None` for keyless local servers.
    ///
    /// For control over the HTTP client, use [`OpenAICompatClient::builder`].
    pub fn new(
        name: impl Into<String>,
        base_url: impl Into<String>,
        path: impl Into<String>,
        api_key: Option<String>,
    ) -> Self {
        let mut builder = Self::builder(name, base_url, path);
        builder.api_key = api_key;
        builder.build()
    }

    /// Start building a client for an OpenAI-compatible endpoint. `name`,
    /// `base_url`, and `path` are required; the API key and HTTP client are
    /// optional. See [`OpenAICompatClient::new`] for the meaning of each
    /// required field.
    pub fn builder(
        name: impl Into<String>,
        base_url: impl Into<String>,
        path: impl Into<String>,
    ) -> OpenAICompatClientBuilder {
        OpenAICompatClientBuilder {
            name: name.into(),
            base_url: base_url.into(),
            path: path.into(),
            api_key: None,
            client: None,
        }
    }
}

/// Builder for [`OpenAICompatClient`].
pub struct OpenAICompatClientBuilder {
    name: String,
    base_url: String,
    path: String,
    api_key: Option<String>,
    client: Option<reqwest::Client>,
}

impl OpenAICompatClientBuilder {
    /// Set the bearer API key. Leave unset for keyless local servers.
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Override the [`reqwest::Client`] used for requests. When unset, a default
    /// client with a `llm-client/<version>` user agent is built.
    pub fn client(mut self, client: reqwest::Client) -> Self {
        self.client = Some(client);
        self
    }

    /// Build the [`OpenAICompatClient`].
    pub fn build(self) -> OpenAICompatClient {
        let base = self.base_url.trim_end_matches('/');
        let url = if self.path.starts_with('/') {
            format!("{base}{}", self.path)
        } else {
            format!("{base}/{}", self.path)
        };
        OpenAICompatClient {
            name: self.name,
            url,
            api_key: self.api_key,
            client: self.client.unwrap_or_else(super::default_client),
        }
    }
}

#[cfg(test)]
impl OpenAICompatClient {
    /// The fully-formed chat-completions endpoint URL. Test-only accessor.
    pub(crate) fn endpoint(&self) -> &str {
        &self.url
    }
}

#[async_trait::async_trait]
impl crate::LLMClient for OpenAICompatClient {
    fn name(&self) -> &str {
        &self.name
    }

    async fn generate(&self, req: &LLMRequest) -> Result<LLMResponse, LLMError> {
        openai_chat_completion(&self.client, &self.url, self.api_key.as_deref(), req).await
    }
}

#[async_trait::async_trait]
impl crate::LLMStreamingClient for OpenAICompatClient {
    async fn stream(&self, req: &LLMRequest, on_text: &mut crate::TextSink<'_>) -> Result<LLMResponse, LLMError> {
        openai_chat_completion_streaming(&self.client, &self.url, self.api_key.as_deref(), req, on_text).await
    }
}

/// Assembles an OpenAI-compatible streaming response from SSE `data:` payloads.
#[derive(Default)]
struct OpenAiStreamAcc {
    text: String,
    stop_reason: String,
    usage: Usage,
    /// tool_call index -> (id, name, arguments-so-far)
    tools: std::collections::BTreeMap<u64, OpenAiToolFrag>,
}

#[derive(Default)]
struct OpenAiToolFrag {
    id: String,
    name: String,
    args: String,
}

impl OpenAiStreamAcc {
    fn new() -> Self {
        Self::default()
    }

    fn handle(&mut self, data: &str, on_text: &mut dyn FnMut(&str)) {
        if data.trim().is_empty() || data.trim() == "[DONE]" {
            return;
        }
        let Ok(v) = serde_json::from_str::<Value>(data) else {
            return;
        };
        if let Some(u) = v.get("usage") {
            if let Some(p) = u.get("prompt_tokens").and_then(Value::as_u64) {
                self.usage.input_tokens = p;
            }
            if let Some(c) = u.get("completion_tokens").and_then(Value::as_u64) {
                self.usage.output_tokens = c;
            }
        }
        let Some(choice) = v.get("choices").and_then(Value::as_array).and_then(|a| a.first()) else {
            return;
        };
        if let Some(fr) = choice.get("finish_reason").and_then(Value::as_str) {
            self.stop_reason = fr.to_string();
        }
        let Some(delta) = choice.get("delta") else {
            return;
        };
        if let Some(c) = delta.get("content").and_then(Value::as_str)
            && !c.is_empty()
        {
            self.text.push_str(c);
            on_text(c);
        }
        if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
            for call in calls {
                let idx = call.get("index").and_then(Value::as_u64).unwrap_or(0);
                let frag = self.tools.entry(idx).or_default();
                if let Some(id) = call.get("id").and_then(Value::as_str)
                    && !id.is_empty()
                {
                    frag.id = id.to_string();
                }
                if let Some(func) = call.get("function") {
                    if let Some(n) = func.get("name").and_then(Value::as_str)
                        && !n.is_empty()
                    {
                        frag.name = n.to_string();
                    }
                    if let Some(a) = func.get("arguments").and_then(Value::as_str) {
                        frag.args.push_str(a);
                    }
                }
            }
        }
    }

    fn finish(self) -> LLMResponse {
        let tool_calls = self
            .tools
            .into_values()
            .map(|f| ToolCall {
                id: f.id,
                name: f.name,
                input: serde_json::from_str(if f.args.is_empty() { "{}" } else { &f.args }).unwrap_or(Value::Null),
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
    use super::*;

    #[test]
    fn openai_accumulates_text_and_tool_call() {
        use serde_json::json;
        let mut acc = OpenAiStreamAcc::new();
        let mut text = String::new();
        let feed = |v: serde_json::Value, acc: &mut OpenAiStreamAcc, t: &mut String| {
            acc.handle(&v.to_string(), &mut |s| t.push_str(s));
        };
        feed(json!({"choices":[{"delta":{"content":"Hi"}}]}), &mut acc, &mut text);
        feed(json!({"choices":[{"delta":{"content":" there"}}]}), &mut acc, &mut text);
        feed(
            json!({"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"run_query","arguments":"{\"sql\":\""}}]}}]}),
            &mut acc,
            &mut text,
        );
        feed(
            json!({"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"SELECT 1\"}"}}]}}]}),
            &mut acc,
            &mut text,
        );
        feed(
            json!({"choices":[{"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":5,"completion_tokens":9}}),
            &mut acc,
            &mut text,
        );
        acc.handle("[DONE]", &mut |s| text.push_str(s));

        assert_eq!(text, "Hi there");
        let resp = acc.finish();
        assert_eq!(resp.text.as_deref(), Some("Hi there"));
        assert_eq!(resp.stop_reason, "tool_calls");
        assert_eq!(resp.usage.input_tokens, 5);
        assert_eq!(resp.usage.output_tokens, 9);
        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_calls[0].id, "c1");
        assert_eq!(resp.tool_calls[0].name, "run_query");
        assert_eq!(resp.tool_calls[0].input, serde_json::json!({ "sql": "SELECT 1" }));
    }

    #[test]
    fn openai_stream_no_arg_tool_call_defaults_to_empty_object() {
        use serde_json::json;
        let mut acc = OpenAiStreamAcc::new();
        // A tool call streamed with a name/id but no argument fragments.
        acc.handle(
            &json!({"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"get_time"}}]}}]})
                .to_string(),
            &mut |_| {},
        );
        let resp = acc.finish();
        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_calls[0].input, json!({}), "empty args must default to {{}}, not null");
    }
}
