use serde_json::Value;

use super::openai_compat::{OpenAICompatClient, openai_build_body, openai_parse_response};
use crate::{LLMClient, LLMError, LLMRequest, LLMResponse, LLMStreamingClient};

pub struct OpenAIClient(OpenAICompatClient);

impl OpenAIClient {
    pub fn new(api_key: String, base_url: Option<String>) -> Self {
        let mut builder = Self::builder(api_key);
        builder.base_url = base_url;
        builder.build()
    }

    /// Start building a provider. The API key is required; the base URL and HTTP
    /// client are optional.
    pub fn builder(api_key: impl Into<String>) -> OpenAIClientBuilder {
        OpenAIClientBuilder {
            api_key: api_key.into(),
            base_url: None,
            client: None,
        }
    }

    pub fn from_env() -> Result<Self, LLMError> {
        let key =
            std::env::var("OPENAI_API_KEY").map_err(|_| LLMError::Provider("OPENAI_API_KEY not set".to_string()))?;
        Ok(Self::new(key, None))
    }

    pub fn build_body(&self, req: &LLMRequest) -> Value {
        openai_build_body(req)
    }

    pub fn parse_response(json: &Value) -> Result<LLMResponse, LLMError> {
        openai_parse_response(json)
    }
}

/// Builder for [`OpenAIClient`].
pub struct OpenAIClientBuilder {
    api_key: String,
    base_url: Option<String>,
    client: Option<reqwest::Client>,
}

impl OpenAIClientBuilder {
    /// Override the API base URL (defaults to `https://api.openai.com`).
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Override the [`reqwest::Client`] used for requests.
    pub fn client(mut self, client: reqwest::Client) -> Self {
        self.client = Some(client);
        self
    }

    /// Build the [`OpenAIClient`].
    pub fn build(self) -> OpenAIClient {
        let base = self.base_url.unwrap_or_else(|| "https://api.openai.com".to_string());
        let mut inner = OpenAICompatClient::builder("openai", base, "/v1/chat/completions").api_key(self.api_key);
        if let Some(client) = self.client {
            inner = inner.client(client);
        }
        OpenAIClient(inner.build())
    }
}

#[async_trait::async_trait]
impl LLMClient for OpenAIClient {
    fn name(&self) -> &str {
        self.0.name()
    }

    async fn generate(&self, req: &LLMRequest) -> Result<LLMResponse, LLMError> {
        self.0.generate(req).await
    }
}

#[async_trait::async_trait]
impl LLMStreamingClient for OpenAIClient {
    async fn stream(&self, req: &LLMRequest, on_text: &mut crate::TextSink<'_>) -> Result<LLMResponse, LLMError> {
        self.0.stream(req, on_text).await
    }
}
