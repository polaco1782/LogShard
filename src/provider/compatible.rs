use anyhow::{Context, Result, anyhow, bail};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{ChatMessage, ModelReply, ProviderConfig, ProviderKind, llama_cpp};
use crate::debug::ActivityLog;

pub struct OpenAiCompatibleClient {
    config: ProviderConfig,
    client: Client,
}

impl OpenAiCompatibleClient {
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(config.timeout)
            .build()
            .context("failed to construct HTTP client")?;
        Ok(Self { config, client })
    }

    pub fn config(&self) -> &ProviderConfig {
        &self.config
    }

    pub async fn complete(
        &self,
        messages: &[ChatMessage],
        tools: &[Value],
        activity: &ActivityLog,
    ) -> Result<ModelReply> {
        let request = ChatRequest {
            model: &self.config.model,
            messages,
            tools,
            tool_choice: "auto",
            parallel_tool_calls: false,
            max_completion_tokens: matches!(
                self.config.kind,
                ProviderKind::Openai | ProviderKind::Openrouter
            )
            .then_some(self.config.max_output_tokens),
            max_tokens: matches!(self.config.kind, ProviderKind::LlamaCpp)
                .then_some(self.config.max_output_tokens),
        };
        activity.emit(
            "provider.request",
            serde_json::json!({
                "provider": self.config.kind.to_string(),
                "endpoint": self.config.endpoint.as_ref().map(reqwest::Url::as_str),
                "request": {
                    "model": request.model,
                    "message_count": request.messages.len(),
                    "tools": tool_names(request.tools),
                    "tool_choice": request.tool_choice,
                    "parallel_tool_calls": request.parallel_tool_calls,
                    "max_completion_tokens": request.max_completion_tokens,
                    "max_tokens": request.max_tokens
                }
            }),
        );
        let mut builder = self
            .client
            .post(
                self.config
                    .endpoint
                    .clone()
                    .context("OpenAI-compatible provider has no endpoint")?,
            )
            .json(&request);
        if let Some(key) = &self.config.api_key {
            builder = builder.bearer_auth(key);
        }

        let response = match builder.send().await {
            Ok(response) => response,
            Err(error) => {
                activity.emit(
                    "provider.error",
                    serde_json::json!({ "error": error.to_string() }),
                );
                return Err(error).context("provider request failed");
            }
        };
        let status = response.status();
        let bytes = match response.bytes().await {
            Ok(bytes) => bytes,
            Err(error) => {
                activity.emit(
                    "provider.error",
                    serde_json::json!({
                        "status": status.as_u16(),
                        "error": error.to_string()
                    }),
                );
                return Err(error).context("failed to read provider response");
            }
        };
        activity.emit(
            "provider.response",
            serde_json::json!({
                "status": status.as_u16(),
                "body": debug_body(&bytes)
            }),
        );
        if !status.is_success() {
            bail!("provider returned HTTP {status}: {}", error_body(&bytes));
        }

        let response: ChatResponse = serde_json::from_slice(&bytes)
            .with_context(|| format!("provider returned invalid JSON: {}", error_body(&bytes)))?;
        let choice = response
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("provider returned no completion choices"))?;
        let mut message = choice.message;
        let mut finish_reason = choice.finish_reason;
        if self.config.kind == ProviderKind::LlamaCpp
            && message
                .tool_calls
                .as_ref()
                .is_none_or(|calls| calls.is_empty())
            && let Some(call) = message
                .content
                .as_deref()
                .and_then(|content| llama_cpp::content_tool_call(content, tools, messages.len()))
        {
            activity.emit(
                "provider.content_tool_call",
                serde_json::json!({
                    "call_id": call.id,
                    "name": call.function.name
                }),
            );
            message.content = None;
            message.tool_calls = Some(vec![call]);
            finish_reason = Some("tool_calls".to_owned());
        }
        Ok(ModelReply {
            message,
            finish_reason,
        })
    }
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [ChatMessage],
    tools: &'a [Value],
    tool_choice: &'static str,
    parallel_tool_calls: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: ChatMessage,
    finish_reason: Option<String>,
}

fn error_body(bytes: &[u8]) -> String {
    const LIMIT: usize = 4096;
    let end = bytes.len().min(LIMIT);
    let mut body = String::from_utf8_lossy(&bytes[..end]).into_owned();
    if bytes.len() > LIMIT {
        body.push_str("…[truncated]");
    }
    body
}

fn debug_body(bytes: &[u8]) -> Value {
    serde_json::from_slice(bytes)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(bytes).into_owned()))
}

fn tool_names(tools: &[Value]) -> Vec<&str> {
    tools
        .iter()
        .filter_map(|tool| tool.pointer("/function/name").and_then(Value::as_str))
        .collect()
}
