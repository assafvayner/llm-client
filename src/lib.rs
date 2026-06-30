//! `llm-client` — a provider-agnostic LLM client abstraction.
//!
//! Provides the [`LLMClient`] trait and a small set of shared request/response
//! types, plus client implementations for several providers (Claude, OpenAI,
//! Gemini, Ollama, Hugging Face) plus a generic OpenAI-compatible client. Each
//! provider lives behind a cargo feature so consumers only compile the ones they
//! use.

mod client;
mod error;
mod providers;
mod types;

#[cfg(any(feature = "claude", feature = "openai-compat"))]
mod streaming;

#[cfg(feature = "mock")]
mod mock;

pub use client::{LLMClient, LLMStreamingClient, TextSink};
pub use error::LLMError;
#[cfg(feature = "mock")]
pub use mock::{MockProvider, PendingProvider, ScriptedStreamProvider};
#[cfg(feature = "claude")]
pub use providers::{ClaudeProvider, ClaudeProviderBuilder};
#[cfg(feature = "gemini")]
pub use providers::{GeminiProvider, GeminiProviderBuilder};
#[cfg(feature = "hf")]
pub use providers::{HfProvider, HfProviderBuilder};
#[cfg(feature = "ollama")]
pub use providers::{OllamaProvider, OllamaProviderBuilder};
#[cfg(feature = "openai-compat")]
pub use providers::{OpenAICompatProvider, OpenAICompatProviderBuilder};
#[cfg(feature = "openai")]
pub use providers::{OpenAIProvider, OpenAIProviderBuilder};
pub use types::{LLMRequest, LLMResponse, Message, ToolCall, ToolDef, Usage};
