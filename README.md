# llm-client

`llm-client` is a provider-agnostic Rust client abstraction for large language
model APIs. It gives applications one request and response shape, plus concrete
clients for Claude, OpenAI, Gemini, Ollama, Hugging Face, and generic
OpenAI-compatible chat completions servers.

## Features

- One `LLMClient` trait for non-streaming chat completions.
- Optional `LLMStreamingClient` support for providers that expose streaming.
- Shared request, response, message, usage, and tool-call types.
- Feature-gated provider implementations so downstream crates can compile only
  the backends they need.
- A generic `OpenAICompatClient` for vLLM, LM Studio, Together, OpenRouter,
  local gateways, and other OpenAI-compatible endpoints.
- Test helpers behind the `mock` feature.

## Installation

Add the crate:

```toml
[dependencies]
llm-client = "0.1"
```

By default, all provider features are enabled. To select a smaller provider set:

```toml
[dependencies]
llm-client = {
  version = "0.1",
  default-features = false,
  features = ["openai", "ollama"]
}
```

This crate uses Rust 1.95+ and edition 2024.

## Quick Start

```rust,no_run
use llm_client::{LLMClient, LLMRequest, Message, OpenAIClient};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let provider = OpenAIClient::from_env()?;

    let req = LLMRequest {
        model: "gpt-4o".into(),
        system: "You are concise.".into(),
        messages: vec![Message::User("Explain Rust ownership in one sentence.".into())],
        tools: vec![],
        max_tokens: 128,
    };

    let resp = provider.generate(&req).await?;
    println!("{}", resp.text.unwrap_or_default());

    Ok(())
}
```

Set the provider's API key before running:

```sh
OPENAI_API_KEY=... cargo run
```

## Providers

| Provider | Feature | Client | Environment variable |
| --- | --- | --- | --- |
| Claude | `claude` | `ClaudeClient` | `ANTHROPIC_API_KEY` |
| OpenAI | `openai` | `OpenAIClient` | `OPENAI_API_KEY` |
| Gemini | `gemini` | `GeminiClient` | `GEMINI_API_KEY` or `GOOGLE_API_KEY` |
| Hugging Face | `hf` | `HfClient` | `HF_TOKEN` |
| Ollama | `ollama` | `OllamaClient` | none |
| OpenAI-compatible | `openai-compat` | `OpenAICompatClient` | caller-defined |

Each hosted provider supports `builder(...)` for explicit configuration and
`from_env()` for the common API-key environment variable. Ollama defaults to
`http://localhost:11434` and can be pointed elsewhere with
`OllamaClient::builder().base_url(...)`.

## Examples

Run the included examples from the repository root:

```sh
HF_TOKEN=... cargo run --example basic
ANTHROPIC_API_KEY=... cargo run --example streaming
HF_TOKEN=... cargo run --example streaming_hf
OPENAI_API_KEY=... cargo run --example tool_use
cargo run --example multi_provider -- ollama
cargo run --example custom_client
```

The examples cover:

- `basic`: one-shot completion with Hugging Face.
- `multi_provider`: select a provider at runtime behind `Box<dyn LLMClient>`.
- `streaming` and `streaming_hf`: print text fragments as they arrive.
- `tool_use`: define a tool, handle model tool calls, and continue the turn.
- `custom_client`: use a custom `reqwest::Client` and OpenAI-compatible endpoint.

## Tool Use

Tools are described with `ToolDef` and passed on `LLMRequest`. Providers return
requested tool calls as `ToolCall` values in `LLMResponse::tool_calls`. To
continue the conversation, append the assistant message and corresponding
`Message::ToolResult` entries to the next request.

See `examples/tool_use.rs` for a complete flow.

## Streaming

Providers that implement streaming expose `LLMStreamingClient::stream`. The
method calls a text sink for each fragment and returns the fully assembled
`LLMResponse` when the stream completes.

```rust,ignore
use llm_client::{ClaudeClient, LLMRequest, LLMStreamingClient, Message};

let provider = ClaudeClient::from_env()?;
let req = LLMRequest {
    model: "claude-sonnet-4-6".into(),
    system: "You are helpful.".into(),
    messages: vec![Message::User("Write a haiku about Rust.".into())],
    tools: vec![],
    max_tokens: 256,
};

let mut sink = |fragment: &str| print!("{fragment}");
let response = provider.stream(&req, &mut sink).await?;
```

## OpenAI-Compatible Servers

Use `OpenAICompatClient` for services that expose an OpenAI-style
`/chat/completions` endpoint:

```rust,ignore
use llm_client::OpenAICompatClient;

let provider = OpenAICompatClient::builder(
    "local-vllm",
    "http://localhost:8000",
    "/v1/chat/completions",
)
.build();
```

Call `.api_key(...)` when the endpoint requires bearer-token authentication, or
`.client(...)` to provide a custom `reqwest::Client`.

## Development

Run formatting and tests locally:

```sh
cargo fmt --check
cargo test
```

Some tests and examples are live provider checks. They are skipped or fail fast
when the relevant API key is not configured.

## License

Licensed under the Apache License, Version 2.0. See `LICENSE` for details.
