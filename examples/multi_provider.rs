//! Pick a provider at runtime behind `Box<dyn LLMClient>`.
//!
//! One `LLMRequest` shape works against any provider — only the model id and the
//! builder differ. Each provider's `builder(..)` takes its required fields.
//!
//! Run with: `cargo run --example multi_provider -- claude`
//!           `cargo run --example multi_provider -- ollama`  (no key needed)

use llm_client::{
    ClaudeProvider, GeminiProvider, HfProvider, LLMClient, LLMRequest, Message, OllamaProvider, OpenAIProvider,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let which = std::env::args().nth(1).unwrap_or_else(|| "ollama".into());
    let (client, model) = provider_for(&which)?;

    let req = LLMRequest {
        model: model.into(),
        system: "You are helpful.".into(),
        messages: vec![Message::User("In one sentence, what is Rust?".into())],
        tools: vec![],
        max_tokens: 128,
    };

    let resp = client.generate(&req).await?;
    println!("[{}] {}", client.name(), resp.text.unwrap_or_default());
    Ok(())
}

/// A boxed provider paired with a default model id to use for it.
type Selected = (Box<dyn LLMClient>, &'static str);

/// Map a name to a boxed provider and a sensible default model for it.
fn provider_for(name: &str) -> Result<Selected, Box<dyn std::error::Error>> {
    Ok(match name {
        "claude" => (Box::new(ClaudeProvider::builder(env("ANTHROPIC_API_KEY")?).build()), "claude-sonnet-4-6"),
        "openai" => (Box::new(OpenAIProvider::builder(env("OPENAI_API_KEY")?).build()), "gpt-4o"),
        "gemini" => (Box::new(GeminiProvider::builder(env("GEMINI_API_KEY")?).build()), "gemini-2.0-flash"),
        "hf" => (Box::new(HfProvider::builder(env("HF_TOKEN")?).build()), "meta-llama/Llama-3.3-70B-Instruct"),
        "ollama" => (Box::new(OllamaProvider::builder().build()), "llama3.3"),
        other => return Err(format!("unknown provider: {other}").into()),
    })
}

fn env(key: &str) -> Result<String, Box<dyn std::error::Error>> {
    std::env::var(key).map_err(|_| format!("{key} not set").into())
}
