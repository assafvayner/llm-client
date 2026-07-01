//! Streaming completion — text fragments are printed as they arrive.
//!
//! Run with: `ANTHROPIC_API_KEY=... cargo run --example streaming`

use std::io::Write;

use llm_client::{ClaudeClient, LLMRequest, LLMStreamingClient, Message};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let provider = ClaudeClient::from_env()?;

    let req = LLMRequest::builder("claude-sonnet-4-6")
        .system("You are a helpful assistant.")
        .message(Message::User("Write a haiku about Rust.".into()))
        .max_tokens(256)
        .build();

    // The sink is called for each text fragment as it streams in.
    let mut sink = |fragment: &str| {
        print!("{fragment}");
        let _ = std::io::stdout().flush();
    };

    // `stream` still returns the fully assembled response (text + tool calls).
    let resp = provider.stream(&req, &mut sink).await?;

    println!("\n--\nstop_reason: {}", resp.stop_reason);
    Ok(())
}
