/// Errors produced by the LLM layer.
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum LLMError {
    /// A non-success HTTP status, so callers can match on `status` (retry
    /// `429`/`5xx`, surface auth failures on `401`/`403`, etc.).
    #[error("HTTP {status}: {message}")]
    Http {
        /// The HTTP status code returned by the provider.
        status: u16,
        /// The provider's error message, or the raw body if unstructured.
        message: String,
    },
    /// A transport, serialization, or configuration failure (connection error,
    /// unparseable payload, unset API-key env var).
    #[error("provider error: {0}")]
    Provider(String),
}
