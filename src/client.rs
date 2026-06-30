use crate::{LLMError, LLMRequest, LLMResponse};

/// Sink for streamed text fragments. `Send` so it can cross the provider's
/// async boundary inside a spawned turn.
pub type TextSink<'a> = dyn FnMut(&str) + Send + 'a;

/// A client that can generate a response to an LLM request in a single
/// round-trip.
///
/// Implementations are expected to be cheaply clonable (e.g. `Arc`-wrapped
/// HTTP clients) so they can be shared across threads, but the trait itself
/// only requires `Send + Sync`.
#[async_trait::async_trait]
pub trait LLMClient: Send + Sync {
    async fn generate(&self, req: &LLMRequest) -> Result<LLMResponse, LLMError>;
    fn name(&self) -> &str;
}

/// A client that can stream a completion, emitting text fragments as they
/// arrive.
///
/// This is independent of [`LLMClient`]: a type may implement either, both, or
/// neither. Providers that speak a real streaming protocol (SSE) implement it;
/// non-streaming backends do not.
#[async_trait::async_trait]
pub trait LLMStreamingClient: Send + Sync {
    /// Stream a completion, calling `on_text` for each text fragment as it
    /// arrives, and returning the fully assembled response (incl. tool calls).
    async fn stream(&self, req: &LLMRequest, on_text: &mut TextSink<'_>) -> Result<LLMResponse, LLMError>;
}
