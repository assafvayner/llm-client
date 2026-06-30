use super::openai_compat::OpenAICompatProvider;
use crate::{LLMClient, LLMError, LLMRequest, LLMResponse, LLMStreamingClient};

/// Hugging Face Inference Providers via the OpenAI-compatible router at
/// `https://router.huggingface.co/v1/chat/completions`.
pub struct HfProvider(OpenAICompatProvider);

impl HfProvider {
    pub fn new(api_key: String, base_url: Option<String>) -> Self {
        let mut builder = Self::builder(api_key);
        builder.base_url = base_url;
        builder.build()
    }

    /// Start building a provider. The API key is required; the base URL and HTTP
    /// client are optional.
    pub fn builder(api_key: impl Into<String>) -> HfProviderBuilder {
        HfProviderBuilder {
            api_key: api_key.into(),
            base_url: None,
            client: None,
        }
    }

    pub fn from_env() -> Result<Self, LLMError> {
        let key = std::env::var("HF_TOKEN").map_err(|_| LLMError::Provider("HF_TOKEN not set".to_string()))?;
        Ok(Self::new(key, None))
    }
}

/// Builder for [`HfProvider`].
pub struct HfProviderBuilder {
    api_key: String,
    base_url: Option<String>,
    client: Option<reqwest::Client>,
}

impl HfProviderBuilder {
    /// Override the API base URL (defaults to `https://router.huggingface.co`).
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Override the [`reqwest::Client`] used for requests.
    pub fn client(mut self, client: reqwest::Client) -> Self {
        self.client = Some(client);
        self
    }

    /// Build the [`HfProvider`].
    pub fn build(self) -> HfProvider {
        let base = self.base_url.unwrap_or_else(|| "https://router.huggingface.co".to_string());
        let mut inner = OpenAICompatProvider::builder("hf", base, "/v1/chat/completions").api_key(self.api_key);
        if let Some(client) = self.client {
            inner = inner.client(client);
        }
        HfProvider(inner.build())
    }
}

#[async_trait::async_trait]
impl LLMClient for HfProvider {
    fn name(&self) -> &str {
        self.0.name()
    }

    async fn generate(&self, req: &LLMRequest) -> Result<LLMResponse, LLMError> {
        self.0.generate(req).await
    }
}

#[async_trait::async_trait]
impl LLMStreamingClient for HfProvider {
    async fn stream(&self, req: &LLMRequest, on_text: &mut crate::TextSink<'_>) -> Result<LLMResponse, LLMError> {
        self.0.stream(req, on_text).await
    }
}
