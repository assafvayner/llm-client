//! Streaming completion against the Hugging Face router.
//!
//! Same streaming interface as the Claude example — `HfClient` speaks the
//! OpenAI-compatible SSE dialect, so it implements `LLMStreamingClient` too.
//!
//! Run with: `HF_TOKEN=... cargo run --example streaming_hf`

use std::io::Write;

use llm_client::{HfClient, LLMRequest, LLMStreamingClient, Message};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let provider = HfClient::from_env()?;

    let req = LLMRequest::builder("meta-llama/Llama-3.3-70B-Instruct")
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
    println!("tokens: in={} out={}", resp.usage.input_tokens, resp.usage.output_tokens);
    Ok(())
}
