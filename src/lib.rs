#![doc = include_str!("../README.md")]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

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
#[cfg_attr(docsrs, doc(cfg(feature = "mock")))]
pub use mock::{MockClient, PendingClient, ScriptedStreamClient};
#[cfg(all(feature = "llama-cpp", unix))]
#[cfg_attr(docsrs, doc(cfg(all(feature = "llama-cpp", unix))))]
pub use providers::LlamaCppUdsBuilder;
#[cfg(feature = "claude")]
#[cfg_attr(docsrs, doc(cfg(feature = "claude")))]
pub use providers::{ClaudeClient, ClaudeClientBuilder};
#[cfg(feature = "gemini")]
#[cfg_attr(docsrs, doc(cfg(feature = "gemini")))]
pub use providers::{GeminiClient, GeminiClientBuilder};
#[cfg(feature = "hf")]
#[cfg_attr(docsrs, doc(cfg(feature = "hf")))]
pub use providers::{HfClient, HfClientBuilder};
#[cfg(feature = "llama-cpp")]
#[cfg_attr(docsrs, doc(cfg(feature = "llama-cpp")))]
pub use providers::{LlamaCppClient, LlamaCppHttpBuilder};
#[cfg(feature = "ollama")]
#[cfg_attr(docsrs, doc(cfg(feature = "ollama")))]
pub use providers::{OllamaClient, OllamaClientBuilder};
#[cfg(feature = "openai")]
#[cfg_attr(docsrs, doc(cfg(feature = "openai")))]
pub use providers::{OpenAIClient, OpenAIClientBuilder};
#[cfg(feature = "openai-compat")]
#[cfg_attr(docsrs, doc(cfg(feature = "openai-compat")))]
pub use providers::{OpenAICompatClient, OpenAICompatClientBuilder};
pub use types::{LLMRequest, LLMResponse, Message, ToolCall, ToolDef, Usage};
