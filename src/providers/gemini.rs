use super::openai_compat::OpenAICompatProvider;
use crate::{LLMClient, LLMError, LLMRequest, LLMResponse, LLMStreamingClient};

/// Google Gemini via its OpenAI-compatible endpoint.
///
/// Gemini exposes an OpenAI-compatible chat-completions surface at
/// `https://generativelanguage.googleapis.com/v1beta/openai/chat/completions`.
/// The base URL already ends in `/v1beta/openai`, so only `/chat/completions`
/// is appended (unlike OpenAI/HF which use `/v1/chat/completions`).
pub struct GeminiProvider(OpenAICompatProvider);

impl GeminiProvider {
    pub fn new(api_key: String, base_url: Option<String>) -> Self {
        let mut builder = Self::builder(api_key);
        builder.base_url = base_url;
        builder.build()
    }

    /// Start building a provider. The API key is required; the base URL and HTTP
    /// client are optional.
    pub fn builder(api_key: impl Into<String>) -> GeminiProviderBuilder {
        GeminiProviderBuilder {
            api_key: api_key.into(),
            base_url: None,
            client: None,
        }
    }

    /// Read the API key from `GEMINI_API_KEY`, falling back to `GOOGLE_API_KEY`.
    pub fn from_env() -> Result<Self, LLMError> {
        let key = std::env::var("GEMINI_API_KEY")
            .or_else(|_| std::env::var("GOOGLE_API_KEY"))
            .map_err(|_| LLMError::Provider("GEMINI_API_KEY (or GOOGLE_API_KEY) not set".into()))?;
        Ok(Self::new(key, None))
    }
}

/// Builder for [`GeminiProvider`].
pub struct GeminiProviderBuilder {
    api_key: String,
    base_url: Option<String>,
    client: Option<reqwest::Client>,
}

impl GeminiProviderBuilder {
    /// Override the API base URL (defaults to
    /// `https://generativelanguage.googleapis.com/v1beta/openai`).
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Override the [`reqwest::Client`] used for requests.
    pub fn client(mut self, client: reqwest::Client) -> Self {
        self.client = Some(client);
        self
    }

    /// Build the [`GeminiProvider`].
    pub fn build(self) -> GeminiProvider {
        let base = self
            .base_url
            .unwrap_or_else(|| "https://generativelanguage.googleapis.com/v1beta/openai".to_string());
        let mut inner = OpenAICompatProvider::builder("gemini", base, "/chat/completions").api_key(self.api_key);
        if let Some(client) = self.client {
            inner = inner.client(client);
        }
        GeminiProvider(inner.build())
    }
}

#[async_trait::async_trait]
impl LLMClient for GeminiProvider {
    fn name(&self) -> &str {
        self.0.name()
    }

    async fn generate(&self, req: &LLMRequest) -> Result<LLMResponse, LLMError> {
        self.0.generate(req).await
    }
}

#[async_trait::async_trait]
impl LLMStreamingClient for GeminiProvider {
    async fn stream(&self, req: &LLMRequest, on_text: &mut crate::TextSink<'_>) -> Result<LLMResponse, LLMError> {
        self.0.stream(req, on_text).await
    }
}
