mod compatible;
mod github_copilot;
mod llama_cpp;
mod openai;
mod openrouter;

use std::{fmt, net::IpAddr, str::FromStr, time::Duration};

use anyhow::{Context, Result, bail};
use reqwest::Url;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

pub use compatible::OpenAiCompatibleClient;
pub(crate) use github_copilot::authenticate as authenticate_copilot;
pub use github_copilot::{CopilotRunResult, GithubCopilotClient};

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    Openai,
    Openrouter,
    LlamaCpp,
    GithubCopilot,
}

impl fmt::Display for ProviderKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Openai => "openai",
            Self::Openrouter => "openrouter",
            Self::LlamaCpp => "llama_cpp",
            Self::GithubCopilot => "github_copilot",
        })
    }
}

impl FromStr for ProviderKind {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "1" | "openai" => Ok(Self::Openai),
            "2" | "openrouter" => Ok(Self::Openrouter),
            "3" | "llama.cpp" | "llama-cpp" | "llama_cpp" => Ok(Self::LlamaCpp),
            "4" | "copilot" | "github-copilot" | "github_copilot" => Ok(Self::GithubCopilot),
            _ => bail!(
                "unsupported provider {value:?}; expected openai, openrouter, llama_cpp, or github_copilot"
            ),
        }
    }
}

impl ProviderKind {
    pub const fn display_name(self) -> &'static str {
        match self {
            Self::Openai => "OpenAI",
            Self::Openrouter => "OpenRouter",
            Self::LlamaCpp => "llama.cpp",
            Self::GithubCopilot => "GitHub Copilot",
        }
    }

    pub(crate) fn default_model(self) -> &'static str {
        self.defaults().model
    }

    pub(crate) fn default_base_url(self) -> Option<&'static str> {
        self.defaults().base_url
    }

    pub(crate) fn requires_api_key(self) -> bool {
        self.defaults().api_key_required
    }

    fn defaults(self) -> ProviderDefaults {
        match self {
            Self::Openai => openai::defaults(),
            Self::Openrouter => openrouter::defaults(),
            Self::LlamaCpp => llama_cpp::defaults(),
            Self::GithubCopilot => github_copilot::defaults(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ProviderConfig {
    pub kind: ProviderKind,
    pub model: String,
    pub endpoint: Option<Url>,
    pub api_key: Option<String>,
    pub timeout: Duration,
    pub max_output_tokens: u32,
    pub(crate) ca_certificates: Vec<reqwest::Certificate>,
}

impl ProviderConfig {
    /// Add trust only to this provider, never to the OS or other SDK clients.
    pub fn with_ca_cert_pem(mut self, pem: Option<String>) -> Result<Self> {
        if let Some(pem) = pem {
            if self.kind == ProviderKind::GithubCopilot {
                bail!("provider.ca_cert_pem is not supported for github_copilot");
            }
            if self
                .endpoint
                .as_ref()
                .is_none_or(|url| url.scheme() != "https")
            {
                bail!("provider.ca_cert_pem requires an HTTPS provider.base_url");
            }
            if pem.len() > 256 * 1024 {
                bail!("provider.ca_cert_pem exceeds 256 KiB");
            }
            if pem.contains("PRIVATE KEY") {
                bail!("provider.ca_cert_pem must contain public certificates, not private keys");
            }
            let mut remaining = pem.trim();
            while !remaining.is_empty() {
                if !remaining.starts_with("-----BEGIN CERTIFICATE-----") {
                    bail!("provider.ca_cert_pem must contain only PEM certificates");
                }
                let end = remaining
                    .find("-----END CERTIFICATE-----")
                    .context("provider.ca_cert_pem contains an incomplete certificate")?;
                remaining = remaining[end + "-----END CERTIFICATE-----".len()..].trim();
            }
            self.ca_certificates = reqwest::Certificate::from_pem_bundle(pem.as_bytes())
                .context("provider.ca_cert_pem must be a valid PEM certificate bundle")?;
            if self.ca_certificates.is_empty() {
                bail!("provider.ca_cert_pem must contain at least one PEM certificate");
            }
        }
        Ok(self)
    }
    pub fn new(
        kind: ProviderKind,
        model: String,
        base_url: Option<String>,
        api_key: Option<String>,
        timeout_seconds: u64,
        max_output_tokens: u32,
    ) -> Result<Self> {
        let model = nonempty(model, "provider.model")?;
        if model.chars().any(char::is_whitespace) {
            bail!("provider.model must not contain whitespace");
        }
        let endpoint = match (kind, base_url) {
            (ProviderKind::GithubCopilot, None) => None,
            (ProviderKind::GithubCopilot, Some(_)) => {
                bail!("provider.base_url is not supported for github_copilot")
            }
            (_, Some(base_url)) => Some(completion_endpoint(&base_url)?),
            (_, None) => bail!("provider.base_url is required for {kind}"),
        };
        let api_key = api_key.and_then(|value| {
            let trimmed = value.trim().to_owned();
            (!trimmed.is_empty()).then_some(trimmed)
        });

        if kind.requires_api_key() && api_key.is_none() {
            bail!("provider.api_key is required for {kind}");
        }
        if kind == ProviderKind::GithubCopilot
            && api_key
                .as_deref()
                .is_some_and(|key| key.starts_with("ghp_"))
        {
            bail!(
                "github_copilot does not support classic personal access tokens; use a fine-grained token with Copilot Requests permission"
            );
        }
        if api_key
            .as_deref()
            .is_some_and(|key| key.chars().any(char::is_control))
        {
            bail!("provider.api_key must not contain control characters");
        }

        Ok(Self {
            kind,
            model,
            endpoint,
            api_key,
            timeout: Duration::from_secs(timeout_seconds),
            max_output_tokens,
            ca_certificates: Vec::new(),
        })
    }
}

pub(crate) struct ProviderDefaults {
    pub model: &'static str,
    pub base_url: Option<&'static str>,
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
    let base_url = nonempty(base_url.to_owned(), "provider.base_url")?;
    let parsed =
        Url::parse(&base_url).with_context(|| format!("invalid provider.base_url: {base_url}"))?;

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

    #[test]
    fn github_copilot_uses_sdk_managed_routing() {
        assert_eq!(
            ProviderKind::from_str("copilot").unwrap(),
            ProviderKind::GithubCopilot
        );
        let config = ProviderConfig::new(
            ProviderKind::GithubCopilot,
            "gpt-5.4".into(),
            None,
            None,
            120,
            4096,
        )
        .unwrap();

        assert!(config.endpoint.is_none());
        assert!(config.api_key.is_none());
    }

    #[test]
    fn github_copilot_rejects_classic_tokens_and_custom_endpoints() {
        assert!(
            ProviderConfig::new(
                ProviderKind::GithubCopilot,
                "gpt-5.4".into(),
                None,
                Some("ghp_classic".into()),
                120,
                4096,
            )
            .is_err()
        );
        assert!(
            ProviderConfig::new(
                ProviderKind::GithubCopilot,
                "gpt-5.4".into(),
                Some("https://api.githubcopilot.com".into()),
                Some("github_pat_test".into()),
                120,
                4096,
            )
            .is_err()
        );
    }
}
