//! [llama.cpp](https://github.com/ggml-org/llama.cpp) server provider.
//!
//! The llama.cpp server exposes an OpenAI-compatible `/v1/chat/completions`
//! endpoint, so [`LlamaCppClient`] is a thin wrapper over
//! [`OpenAICompatClient`](super::openai_compat::OpenAICompatClient) — request
//! mapping, SSE streaming, and tool calls are all shared with the other
//! OpenAI-compatible providers.
//!
//! The transport (HTTP vs. Unix domain socket) is fixed at construction time:
//! [`LlamaCppClient::http`]/[`LlamaCppClient::local`] yield a
//! [`LlamaCppHttpBuilder`], while [`LlamaCppClient::unix_socket`] yields a
//! [`LlamaCppUdsBuilder`]. There is no single builder that exposes both, so the
//! two modes are mutually exclusive by construction.

use super::openai_compat::OpenAICompatClient;
use crate::{LLMClient, LLMError, LLMRequest, LLMResponse, LLMStreamingClient};

/// llama.cpp's default server bind.
const DEFAULT_HTTP_URL: &str = "http://localhost:8080";

/// A client for a llama.cpp server's OpenAI-compatible `/v1/chat/completions`
/// endpoint, reachable over HTTP or a Unix domain socket.
pub struct LlamaCppClient(OpenAICompatClient);

impl LlamaCppClient {
    /// Start building a client that talks HTTP to `base_url` (e.g.
    /// `http://localhost:8080`). A trailing slash on `base_url` is trimmed.
    pub fn http(base_url: impl Into<String>) -> LlamaCppHttpBuilder {
        LlamaCppHttpBuilder {
            base_url: base_url.into(),
            api_key: None,
            client: None,
        }
    }

    /// Start building a client for llama.cpp's default local bind
    /// (`http://localhost:8080`). Shorthand for [`LlamaCppClient::http`].
    pub fn local() -> LlamaCppHttpBuilder {
        Self::http(DEFAULT_HTTP_URL)
    }

    /// Start building a client that connects over the Unix domain socket at
    /// `path`.
    ///
    /// All connections are routed over the socket; the request host is a
    /// synthetic `http://localhost` and no DNS resolution occurs. There is no
    /// HTTP-client override in this mode — the socket-bound client is built
    /// internally.
    #[cfg(unix)]
    pub fn unix_socket(path: impl AsRef<std::path::Path>) -> LlamaCppUdsBuilder {
        LlamaCppUdsBuilder {
            socket_path: path.as_ref().to_path_buf(),
            api_key: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn endpoint(&self) -> &str {
        self.0.endpoint()
    }
}

/// Builder for an HTTP-mode [`LlamaCppClient`]. See [`LlamaCppClient::http`].
pub struct LlamaCppHttpBuilder {
    base_url: String,
    api_key: Option<String>,
    client: Option<reqwest::Client>,
}

impl LlamaCppHttpBuilder {
    /// Set the bearer API key (llama.cpp `--api-key`). Leave unset for the
    /// default keyless server.
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Override the [`reqwest::Client`] used for requests.
    pub fn client(mut self, client: reqwest::Client) -> Self {
        self.client = Some(client);
        self
    }

    /// Build the [`LlamaCppClient`].
    pub fn build(self) -> LlamaCppClient {
        let mut inner = OpenAICompatClient::builder("llama.cpp", self.base_url, "/v1/chat/completions");
        if let Some(key) = self.api_key {
            inner = inner.api_key(key);
        }
        if let Some(client) = self.client {
            inner = inner.client(client);
        }
        LlamaCppClient(inner.build())
    }
}

/// Builder for a Unix-domain-socket-mode [`LlamaCppClient`]. See
/// [`LlamaCppClient::unix_socket`].
#[cfg(unix)]
pub struct LlamaCppUdsBuilder {
    socket_path: std::path::PathBuf,
    api_key: Option<String>,
}

#[cfg(unix)]
impl LlamaCppUdsBuilder {
    /// Set the bearer API key (llama.cpp `--api-key`). Leave unset for the
    /// default keyless server.
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Build the [`LlamaCppClient`].
    pub fn build(self) -> LlamaCppClient {
        let client = reqwest::Client::builder()
            .user_agent(concat!("llmeh/", env!("CARGO_PKG_VERSION")))
            .connect_timeout(std::time::Duration::from_secs(10))
            .unix_socket(self.socket_path)
            .build()
            .expect("failed to build unix-socket reqwest client");
        let mut inner =
            OpenAICompatClient::builder("llama.cpp", "http://localhost", "/v1/chat/completions").client(client);
        if let Some(key) = self.api_key {
            inner = inner.api_key(key);
        }
        LlamaCppClient(inner.build())
    }
}

#[async_trait::async_trait]
impl LLMClient for LlamaCppClient {
    fn name(&self) -> &str {
        self.0.name()
    }

    async fn generate(&self, req: &LLMRequest) -> Result<LLMResponse, LLMError> {
        self.0.generate(req).await
    }
}

#[async_trait::async_trait]
impl LLMStreamingClient for LlamaCppClient {
    async fn stream(&self, req: &LLMRequest, on_text: &mut crate::TextSink<'_>) -> Result<LLMResponse, LLMError> {
        self.0.stream(req, on_text).await
    }
}
