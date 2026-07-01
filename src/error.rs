/// Errors produced by the LLM layer.
#[derive(Debug, thiserror::Error)]
pub enum LLMError {
    /// The provider rejected the request or returned an error response —
    /// covers HTTP/transport failures, non-success status codes, unparseable
    /// payloads, and missing configuration (e.g. an unset API-key env var).
    #[error("provider error: {0}")]
    Provider(String),
    /// A tool invocation failed while handling the model's request.
    #[error("tool error: {0}")]
    Tool(String),
    /// The request was malformed or invalid before it could be sent.
    #[error("request error: {0}")]
    Request(String),
}
