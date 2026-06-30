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
// stream accumulator) plus the generic `OpenAICompatProvider`. The openai/gemini/hf
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
pub use claude::{ClaudeProvider, ClaudeProviderBuilder};
#[cfg(feature = "gemini")]
pub use gemini::{GeminiProvider, GeminiProviderBuilder};
#[cfg(feature = "hf")]
pub use hf::{HfProvider, HfProviderBuilder};
#[cfg(feature = "ollama")]
pub use ollama::{OllamaProvider, OllamaProviderBuilder};
#[cfg(feature = "openai")]
pub use openai::{OpenAIProvider, OpenAIProviderBuilder};
#[cfg(feature = "openai-compat")]
pub use openai_compat::{OpenAICompatProvider, OpenAICompatProviderBuilder};
