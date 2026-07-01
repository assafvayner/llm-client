//! Custom HTTP client + custom endpoint via the builder.
//!
//! Points at any OpenAI-compatible server (vLLM, LM Studio, Together, OpenRouter,
//! a local gateway, …) using a caller-supplied `reqwest::Client` — so you control
//! timeouts, proxies, connection pooling, default headers, and so on.
//!
//! Run with: `cargo run --example custom_client`

use std::time::Duration;

use llmeh::{LLMClient, LLMRequest, Message, OpenAICompatClient};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Bring your own client — here with a 10s timeout and a custom user agent.
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .user_agent("my-app/1.0")
        .build()?;

    // name/base_url/path are required in `builder(..)`; api_key + client are optional.
    let provider = OpenAICompatClient::builder("local-vllm", "http://localhost:8000", "/v1/chat/completions")
        .client(http) // override the default reqwest client
        .build(); // no `.api_key(..)` -> keyless server

    let req = LLMRequest::builder("Qwen/Qwen2.5-7B-Instruct")
        .system("You are concise.")
        .message(Message::User("Say hi in one word.".into()))
        .max_tokens(64)
        .build();

    let resp = provider.generate(&req).await?;
    println!("[{}] {}", provider.name(), resp.text.unwrap_or_default());
    Ok(())
}
