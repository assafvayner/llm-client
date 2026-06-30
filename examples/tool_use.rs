//! Tool use: define a tool, run the model, answer the tool call, continue.
//!
//! Run with: `OPENAI_API_KEY=... cargo run --example tool_use`

use llm_client::{LLMClient, LLMRequest, Message, OpenAIProvider, ToolCall, ToolDef};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let provider = OpenAIProvider::from_env()?;

    let weather = ToolDef {
        name: "get_weather".into(),
        description: "Get the current weather for a city.".into(),
        input_schema: json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"],
        }),
    };

    let mut messages = vec![Message::User("What's the weather in Paris?".into())];

    let req = LLMRequest {
        model: "gpt-4o".into(),
        system: "Use tools when needed.".into(),
        messages: messages.clone(),
        tools: vec![weather],
        max_tokens: 512,
    };
    let resp = provider.generate(&req).await?;

    // No tools wanted — we already have the answer.
    if resp.tool_calls.is_empty() {
        println!("{}", resp.text.unwrap_or_default());
        return Ok(());
    }

    // Record the assistant turn, then answer each tool call.
    messages.push(Message::Assistant {
        text: resp.text.clone(),
        tool_calls: resp.tool_calls.clone(),
    });
    for ToolCall { id, name, input } in &resp.tool_calls {
        println!("model called {name}({input})");
        // A real app would dispatch on `name` and run the tool; we fake a result.
        let result = json!({ "temp_c": 18, "conditions": "cloudy" }).to_string();
        messages.push(Message::ToolResult {
            tool_use_id: id.clone(),
            content: result,
            is_error: false,
        });
    }

    // Continue the conversation with the tool results appended.
    let followup = LLMRequest { messages, ..req };
    let final_resp = provider.generate(&followup).await?;
    println!("{}", final_resp.text.unwrap_or_default());
    Ok(())
}
