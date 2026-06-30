//! Streaming completion — text fragments are printed as they arrive.
//!
//! Run with: `ANTHROPIC_API_KEY=... cargo run --example streaming`

use std::io::Write;

use llm_client::{ClaudeProvider, LLMRequest, LLMStreamingClient, Message};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let provider = ClaudeProvider::from_env()?;

    let req = LLMRequest {
        model: "claude-sonnet-4-6".into(),
        system: "You are a helpful assistant.".into(),
        messages: vec![Message::User("Write a haiku about Rust.".into())],
        tools: vec![],
        max_tokens: 256,
    };

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
