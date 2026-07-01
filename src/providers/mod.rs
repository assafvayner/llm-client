#[cfg(feature = "claude")]
mod claude;
#[cfg(feature = "gemini")]
mod gemini;
#[cfg(feature = "hf")]
mod hf;
#[cfg(feature = "llama-cpp")]
mod llama_cpp;
#[cfg(feature = "ollama")]
mod ollama;
#[cfg(feature = "openai")]
mod openai;
// The shared OpenAI-compatible dialect (request/response mapping, HTTP helpers,
// stream accumulator) plus the generic `OpenAICompatClient`. The openai/gemini/hf
// providers all enable `openai-compat`, so this covers them too.
#[cfg(feature = "openai-compat")]
mod openai_compat;

#[cfg(all(
    test,
    any(
        feature = "claude",
        feature = "openai",
        feature = "gemini",
        feature = "hf",
        feature = "ollama",
        feature = "llama-cpp"
    )
))]
mod tests;

#[cfg(feature = "claude")]
pub use claude::{ClaudeClient, ClaudeClientBuilder};
#[cfg(feature = "gemini")]
pub use gemini::{GeminiClient, GeminiClientBuilder};
#[cfg(feature = "hf")]
pub use hf::{HfClient, HfClientBuilder};
#[cfg(all(feature = "llama-cpp", unix))]
pub use llama_cpp::LlamaCppUdsBuilder;
#[cfg(feature = "llama-cpp")]
pub use llama_cpp::{LlamaCppClient, LlamaCppHttpBuilder};
#[cfg(feature = "ollama")]
pub use ollama::{OllamaClient, OllamaClientBuilder};
#[cfg(feature = "openai")]
pub use openai::{OpenAIClient, OpenAIClientBuilder};
#[cfg(feature = "openai-compat")]
pub use openai_compat::{OpenAICompatClient, OpenAICompatClientBuilder};

/// Default HTTP client used when the caller supplies none via `.client(...)`.
///
/// Sets a user agent and a connect timeout so a dead host fails fast. No overall
/// timeout is imposed, since long or streamed completions are legitimate —
/// callers wanting a hard deadline should pass their own [`reqwest::Client`].
#[cfg(any(feature = "claude", feature = "openai-compat", feature = "ollama"))]
pub(crate) fn default_client() -> reqwest::Client {
    reqwest::Client::builder()
        .user_agent(concat!("llmeh/", env!("CARGO_PKG_VERSION")))
        .connect_timeout(std::time::Duration::from_secs(10))
        .build()
        .expect("failed to build reqwest client")
}
