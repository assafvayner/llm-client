#[cfg(feature = "claude")]
mod claude;
#[cfg(feature = "gemini")]
mod gemini;
#[cfg(feature = "hf")]
mod hf;
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
        feature = "ollama"
    )
))]
mod tests;

#[cfg(feature = "claude")]
pub use claude::{ClaudeClient, ClaudeClientBuilder};
#[cfg(feature = "gemini")]
pub use gemini::{GeminiClient, GeminiClientBuilder};
#[cfg(feature = "hf")]
pub use hf::{HfClient, HfClientBuilder};
#[cfg(feature = "ollama")]
pub use ollama::{OllamaClient, OllamaClientBuilder};
#[cfg(feature = "openai")]
pub use openai::{OpenAIClient, OpenAIClientBuilder};
#[cfg(feature = "openai-compat")]
pub use openai_compat::{OpenAICompatClient, OpenAICompatClientBuilder};
