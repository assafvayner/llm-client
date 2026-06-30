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
pub use mock::{MockClient, PendingClient, ScriptedStreamClient};
#[cfg(feature = "claude")]
pub use providers::{ClaudeClient, ClaudeClientBuilder};
#[cfg(feature = "gemini")]
pub use providers::{GeminiClient, GeminiClientBuilder};
#[cfg(feature = "hf")]
pub use providers::{HfClient, HfClientBuilder};
#[cfg(feature = "ollama")]
pub use providers::{OllamaClient, OllamaClientBuilder};
#[cfg(feature = "openai")]
pub use providers::{OpenAIClient, OpenAIClientBuilder};
#[cfg(feature = "openai-compat")]
pub use providers::{OpenAICompatClient, OpenAICompatClientBuilder};
pub use types::{LLMRequest, LLMResponse, Message, ToolCall, ToolDef, Usage};
