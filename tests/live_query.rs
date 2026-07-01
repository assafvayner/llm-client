//! Live integration tests: run a basic query against real provider APIs.
//!
//! Offline (`CI` unset) each test is skipped when its API-key env var is
//! missing, so a plain `cargo test` stays green. In CI (`CI` set, as GitHub
//! Actions does) a missing key hard-fails the test, so a misconfigured secret
//! is caught rather than silently skipped.
//!
//! Env vars: `HF_TOKEN`, `GEMINI_API_KEY`.

use llmeh::{LLMClient, LLMRequest, Message};

/// Fetch a required API key. Returns `Some(key)` when set. When unset, panics in
/// CI (so the build fails) and returns `None` (test skips) otherwise.
fn key_or_skip(var: &str) -> Option<String> {
    match std::env::var(var) {
        Ok(key) => Some(key),
        Err(_) if std::env::var_os("CI").is_some() => {
            panic!("{var} not set in CI — required for live integration tests");
        },
        Err(_) => {
            eprintln!("[skip] {var} not set — skipping live test");
            None
        },
    }
}

/// A minimal one-turn request that should elicit a short text reply.
fn basic_request(model: &str) -> LLMRequest {
    LLMRequest::builder(model)
        .system("You are a terse assistant.")
        .message(Message::User("Reply with the single word: pong".to_string()))
        .max_tokens(32)
        .build()
}

/// Run the request and assert the provider returned non-empty text.
async fn assert_basic_query(client: &dyn LLMClient, model: &str) {
    let resp = client
        .generate(&basic_request(model))
        .await
        .unwrap_or_else(|e| panic!("[{}] live query failed: {e}", client.name()));

    assert!(
        resp.text.as_deref().is_some_and(|t| !t.trim().is_empty()),
        "[{}] expected non-empty text, got {resp:?}",
        client.name(),
    );
}

#[cfg(feature = "hf")]
#[tokio::test]
async fn hf_basic_query() {
    let Some(key) = key_or_skip("HF_TOKEN") else {
        return;
    };
    let client = llmeh::HfClient::builder(key).build();
    assert_basic_query(&client, "meta-llama/Llama-3.3-70B-Instruct").await;
}

#[cfg(feature = "gemini")]
#[tokio::test]
async fn gemini_basic_query() {
    let Some(key) = key_or_skip("GEMINI_API_KEY") else {
        return;
    };
    let client = llmeh::GeminiClient::builder(key).build();
    assert_basic_query(&client, "gemini-3.1-flash-lite").await;
}
