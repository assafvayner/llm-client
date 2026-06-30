use std::collections::VecDeque;
use std::sync::Mutex;

use crate::{LLMClient, LLMError, LLMRequest, LLMResponse, LLMStreamingClient, TextSink, Usage};

// ---------------------------------------------------------------------------
// PendingClient
// ---------------------------------------------------------------------------

/// Test provider whose streaming call never completes on its own — it awaits
/// forever, so a cancellation race can interrupt it.
pub struct PendingClient;

#[async_trait::async_trait]
impl LLMClient for PendingClient {
    fn name(&self) -> &str {
        "pending"
    }
    async fn generate(&self, _req: &LLMRequest) -> Result<LLMResponse, LLMError> {
        std::future::pending().await
    }
}

#[async_trait::async_trait]
impl LLMStreamingClient for PendingClient {
    async fn stream(&self, _req: &LLMRequest, _on_text: &mut TextSink<'_>) -> Result<LLMResponse, LLMError> {
        std::future::pending().await
    }
}

// ---------------------------------------------------------------------------
// MockClient
// ---------------------------------------------------------------------------

/// A provider backed by a queue of pre-scripted responses.
///
/// Responses are returned in FIFO order. If the queue is exhausted and the
/// provider is called again, it returns an error.
///
/// Use `last_request_message_count()` to inspect what the agent sent on the
/// most recent call — useful for testing conversation-memory behaviour.
pub struct MockClient {
    responses: Mutex<VecDeque<LLMResponse>>,
    last_message_count: Mutex<usize>,
}

impl MockClient {
    pub fn new(responses: impl IntoIterator<Item = LLMResponse>) -> Self {
        Self {
            responses: Mutex::new(responses.into_iter().collect()),
            last_message_count: Mutex::new(0),
        }
    }

    /// Number of messages in the last `generate` call's request.
    pub fn last_request_message_count(&self) -> usize {
        *self.last_message_count.lock().unwrap()
    }
}

#[async_trait::async_trait]
impl LLMClient for MockClient {
    async fn generate(&self, req: &LLMRequest) -> Result<LLMResponse, LLMError> {
        *self.last_message_count.lock().unwrap() = req.messages.len();
        self.responses
            .lock()
            .unwrap()
            .pop_front()
            .ok_or_else(|| LLMError::Provider("MockClient: response queue exhausted".to_string()))
    }

    fn name(&self) -> &str {
        "mock"
    }
}

// ---------------------------------------------------------------------------
// ScriptedStreamClient
// ---------------------------------------------------------------------------

/// Test provider that streams pre-scripted text fragments per round-trip.
pub struct ScriptedStreamClient {
    rounds: Mutex<VecDeque<Vec<String>>>,
}

impl ScriptedStreamClient {
    pub fn new<I, J, S>(rounds: I) -> Self
    where
        I: IntoIterator<Item = J>,
        J: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let rounds = rounds
            .into_iter()
            .map(|frags| frags.into_iter().map(Into::into).collect::<Vec<_>>())
            .collect();
        Self {
            rounds: Mutex::new(rounds),
        }
    }
}

#[async_trait::async_trait]
impl LLMStreamingClient for ScriptedStreamClient {
    async fn stream(&self, _req: &LLMRequest, on_text: &mut TextSink<'_>) -> Result<LLMResponse, LLMError> {
        let frags = self.rounds.lock().unwrap().pop_front().unwrap_or_default();
        let mut text = String::new();
        for f in &frags {
            on_text(f);
            text.push_str(f);
        }
        Ok(LLMResponse {
            text: if text.is_empty() { None } else { Some(text) },
            tool_calls: vec![],
            usage: Usage {
                input_tokens: 1,
                output_tokens: 1,
            },
            stop_reason: "end_turn".into(),
        })
    }
}
