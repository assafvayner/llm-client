//! One-shot completion against a local llama.cpp server.
//!
//! llama.cpp's server speaks the OpenAI-compatible `/v1/chat/completions`
//! dialect, so `LlamaCppClient` supports both `generate` and streaming.
//!
//! HTTP (the default): start `llama-server --port 8080`, then
//!   `cargo run --example llama_cpp`
//!
//! Unix domain socket: start the server bound to a socket, then pass the path
//!   `cargo run --example llama_cpp -- /run/llama.sock`

use llm_client::{LLMClient, LLMRequest, LlamaCppClient, Message};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // First CLI arg, when present, is a Unix socket path; otherwise use HTTP.
    let client = match std::env::args().nth(1) {
        #[cfg(unix)]
        Some(socket) => LlamaCppClient::unix_socket(socket).build(),
        #[cfg(not(unix))]
        Some(_) => return Err("Unix domain sockets are only supported on unix targets".into()),
        None => LlamaCppClient::local().build(),
    };

    let req = LLMRequest {
        model: "llama".into(), // llama.cpp serves whatever model is loaded
        system: "You are a terse assistant.".into(),
        messages: vec![Message::User("Name three primary colors.".into())],
        tools: vec![],
        max_tokens: 256,
    };

    let resp = client.generate(&req).await?;

    println!("[{}] {}", client.name(), resp.text.unwrap_or_default());
    println!("tokens: in={} out={}", resp.usage.input_tokens, resp.usage.output_tokens);
    Ok(())
}
