mod compatible;
mod llama_cpp;
mod openai;
mod openrouter;

use std::{env, fmt, net::IpAddr, time::Duration};

use anyhow::{Context, Result, bail};
use clap::ValueEnum;
use reqwest::Url;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

pub use compatible::OpenAiCompatibleClient;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    Openai,
    Openrouter,
    LlamaCpp,
}

impl fmt::Display for ProviderKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Openai => "openai",
            Self::Openrouter => "openrouter",
            Self::LlamaCpp => "llama_cpp",
        })
    }
}

#[derive(Clone, Debug)]
pub struct ProviderConfig {
    pub kind: ProviderKind,
    pub model: String,
    pub endpoint: Url,
    pub api_key: Option<String>,
    pub timeout: Duration,
    pub max_output_tokens: u32,
}

impl ProviderConfig {
    pub fn resolve(
        kind: ProviderKind,
        model: Option<String>,
        base_url: Option<String>,
        api_key: Option<String>,
        timeout_seconds: u64,
        max_output_tokens: u32,
    ) -> Result<Self> {
        let defaults = match kind {
            ProviderKind::Openai => openai::defaults(),
            ProviderKind::Openrouter => openrouter::defaults(),
            ProviderKind::LlamaCpp => llama_cpp::defaults(),
        };

        let model = nonempty(model.unwrap_or_else(|| defaults.model.to_owned()), "model")?;
        let base_url = base_url.unwrap_or_else(|| defaults.base_url.to_owned());
        let endpoint = completion_endpoint(&base_url)?;
        let api_key = api_key
            .or_else(|| defaults.api_key_env.and_then(|name| env::var(name).ok()))
            .and_then(|value| {
                let trimmed = value.trim().to_owned();
                (!trimmed.is_empty()).then_some(trimmed)
            });

        if defaults.api_key_required && api_key.is_none() {
            bail!(
                "{} requires an API key; pass --api-key or set {}",
                kind,
                defaults.api_key_env.unwrap_or("LOGSHARD_API_KEY")
            );
        }

        Ok(Self {
            kind,
            model,
            endpoint,
            api_key,
            timeout: Duration::from_secs(timeout_seconds),
            max_output_tokens,
        })
    }
}

pub(crate) struct ProviderDefaults {
    pub model: &'static str,
    pub base_url: &'static str,
    pub api_key_env: Option<&'static str>,
    pub api_key_required: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl ChatMessage {
    pub fn text(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
            extra: Map::new(),
        }
    }

    pub fn tool_result(call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: "tool".to_owned(),
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: Some(call_id.into()),
            extra: Map::new(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionCall,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FunctionCall {
    pub name: String,
    #[serde(deserialize_with = "deserialize_arguments")]
    pub arguments: String,
}

#[derive(Debug)]
pub struct ModelReply {
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
}

fn completion_endpoint(base_url: &str) -> Result<Url> {
    let base_url = nonempty(base_url.to_owned(), "base URL")?;
    let parsed = Url::parse(&base_url).with_context(|| format!("invalid base URL: {base_url}"))?;

    if !parsed.username().is_empty() || parsed.password().is_some() {
        bail!("base URL must not contain credentials");
    }
    if parsed.query().is_some() || parsed.fragment().is_some() {
        bail!("base URL must not contain a query or fragment");
    }

    match parsed.scheme() {
        "https" => {}
        "http" if is_loopback(&parsed) => {}
        "http" => bail!("plain HTTP is allowed only for loopback provider URLs"),
        scheme => bail!("unsupported provider URL scheme: {scheme}"),
    }

    let endpoint = format!("{}/chat/completions", parsed.as_str().trim_end_matches('/'));
    Url::parse(&endpoint).context("failed to construct chat completions endpoint")
}

fn is_loopback(url: &Url) -> bool {
    match url.host_str() {
        Some("localhost") => true,
        Some(host) => host
            .trim_start_matches('[')
            .trim_end_matches(']')
            .parse::<IpAddr>()
            .is_ok_and(|ip| ip.is_loopback()),
        None => false,
    }
}

fn nonempty(value: String, field: &str) -> Result<String> {
    let value = value.trim().to_owned();
    if value.is_empty() {
        bail!("{field} must not be empty");
    }
    Ok(value)
}

fn deserialize_arguments<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Value::deserialize(deserializer)?;
    match value {
        Value::String(arguments) => Ok(arguments),
        other => serde_json::to_string(&other).map_err(serde::de::Error::custom),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_accepts_https_and_loopback_http() {
        assert_eq!(
            completion_endpoint("https://api.example/v1")
                .unwrap()
                .as_str(),
            "https://api.example/v1/chat/completions"
        );
        assert_eq!(
            completion_endpoint("http://[::1]:8080/v1/")
                .unwrap()
                .as_str(),
            "http://[::1]:8080/v1/chat/completions"
        );
    }

    #[test]
    fn endpoint_rejects_insecure_remote_hosts_and_credentials() {
        assert!(completion_endpoint("http://example.com/v1").is_err());
        assert!(completion_endpoint("https://secret@example.com/v1").is_err());
    }

    #[test]
    fn accepts_object_arguments_from_compatible_servers() {
        let call: FunctionCall =
            serde_json::from_str(r#"{"name":"search_logs","arguments":{"pattern":"error"}}"#)
                .unwrap();
        assert_eq!(call.arguments, r#"{"pattern":"error"}"#);
    }
}
