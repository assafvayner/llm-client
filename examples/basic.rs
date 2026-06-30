//! Minimal one-shot completion.
//!
//! The provider holds the endpoint + auth; the model is chosen per request.
//!
//! Run with: `HF_TOKEN=... cargo run --example basic`

use llm_client::{HfProvider, LLMClient, LLMRequest, Message};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // `from_env` reads HF_TOKEN; equivalent to `HfProvider::builder(token).build()`.
    let provider = HfProvider::from_env()?;

    let req = LLMRequest {
        model: "meta-llama/Llama-3.3-70B-Instruct".into(),
        system: "You are a terse assistant.".into(),
        messages: vec![Message::User("Name three primary colors.".into())],
        tools: vec![],
        max_tokens: 256,
    };

    let resp = provider.generate(&req).await?;

    println!("[{}] {}", provider.name(), resp.text.unwrap_or_default());
    println!("tokens: in={} out={}", resp.usage.input_tokens, resp.usage.output_tokens);
    Ok(())
}
