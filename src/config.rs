use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::paths;
use crate::provider::{ProviderConfig, ProviderKind, authenticate_copilot};
use crate::secure_fs::{atomic_write_private, verify_private_file};

const DEFAULT_TIMEOUT_SECONDS: u64 = 120;
const DEFAULT_MAX_OUTPUT_TOKENS: u32 = 4096;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub provider: ProviderSettings,
    #[serde(default)]
    pub generation: GenerationSettings,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderSettings {
    #[serde(rename = "type")]
    pub kind: ProviderKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationSettings {
    #[serde(default = "default_timeout_seconds")]
    pub timeout_seconds: u64,
    #[serde(default = "default_max_output_tokens")]
    pub max_output_tokens: u32,
}

impl Default for GenerationSettings {
    fn default() -> Self {
        Self {
            timeout_seconds: DEFAULT_TIMEOUT_SECONDS,
            max_output_tokens: DEFAULT_MAX_OUTPUT_TOKENS,
        }
    }
}

impl Config {
    pub fn path() -> Result<PathBuf> {
        paths::config_file()
    }

    pub fn load() -> Result<Self> {
        let path = Self::path()?;
        Self::load_from(&path).with_context(|| {
            format!(
                "could not load {}; run `logshard setup` to create it",
                path.display()
            )
        })
    }

    pub fn load_from(path: &Path) -> Result<Self> {
        let contents = Self::read_from(path)?;
        Self::parse_from(&contents, path)
    }

    fn read_from(path: &Path) -> Result<String> {
        verify_private_file(path, "configuration")?;
        std::fs::read_to_string(path)
            .with_context(|| format!("could not read configuration at {}", path.display()))
    }

    fn parse_from(contents: &str, path: &Path) -> Result<Self> {
        let config: Self = toml::from_str(contents)
            .with_context(|| format!("invalid configuration in {}", path.display()))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if !(1..=3600).contains(&self.generation.timeout_seconds) {
            bail!("generation.timeout_seconds must be between 1 and 3600");
        }
        if !(64..=32768).contains(&self.generation.max_output_tokens) {
            bail!("generation.max_output_tokens must be between 64 and 32768");
        }
        self.provider_config().map(drop)
    }

    pub fn provider_config(&self) -> Result<ProviderConfig> {
        ProviderConfig::new(
            self.provider.kind,
            self.provider.model.clone(),
            self.provider.base_url.clone(),
            self.provider.api_key.clone(),
            self.generation.timeout_seconds,
            self.generation.max_output_tokens,
        )
    }

    pub fn save_to(&self, path: &Path) -> Result<()> {
        self.validate()?;
        let contents = toml::to_string_pretty(self).context("could not serialize configuration")?;
        atomic_write_private(path, contents.as_bytes(), "configuration")
    }
}

pub fn interactive_setup() -> Result<PathBuf> {
    if !std::io::stdin().is_terminal() {
        bail!("`logshard setup` must be run from an interactive terminal");
    }

    let path = Config::path()?;
    let existing = if path.exists() {
        let contents = Config::read_from(&path)?;
        match Config::parse_from(&contents, &path) {
            Ok(config) => Some(config),
            Err(error) => {
                eprintln!(
                    "warning: existing configuration is invalid and will be replaced if setup completes: {error:#}"
                );
                None
            }
        }
    } else {
        None
    };

    let default_provider = existing
        .as_ref()
        .map_or(ProviderKind::Openai, |config| config.provider.kind);
    println!("AI provider");
    println!("  1) OpenAI");
    println!("  2) OpenRouter");
    println!("  3) llama.cpp");
    println!("  4) GitHub Copilot");
    let provider = ProviderKind::from_str(&prompt_value(
        &format!("Select provider [{default_provider}] > "),
        Some(&default_provider.to_string()),
    )?)?;
    let existing_provider = existing
        .as_ref()
        .filter(|config| config.provider.kind == provider);

    let api_key = if provider == ProviderKind::GithubCopilot {
        println!("Starting GitHub Copilot device authentication...");
        authenticate_copilot()?;
        println!("GitHub Copilot authentication complete.");
        None
    } else {
        prompt_api_key(provider, existing_provider)?
    };

    let default_model = existing_provider.map_or(provider.default_model(), |config| {
        config.provider.model.as_str()
    });
    let model = prompt_value(
        &format!("{} model [{default_model}] > ", provider.display_name()),
        Some(default_model),
    )?;
    let base_url = if let Some(provider_default) = provider.default_base_url() {
        let default_base_url = existing_provider
            .and_then(|config| config.provider.base_url.as_deref())
            .unwrap_or(provider_default);
        Some(prompt_value(
            &format!(
                "{} API base URL [{default_base_url}] > ",
                provider.display_name()
            ),
            Some(default_base_url),
        )?)
    } else {
        None
    };

    let config = Config {
        provider: ProviderSettings {
            kind: provider,
            api_key,
            model,
            base_url,
        },
        generation: existing.map(|config| config.generation).unwrap_or_default(),
    };
    config.save_to(&path)?;
    Ok(path)
}

fn prompt_api_key(
    provider: ProviderKind,
    existing_provider: Option<&Config>,
) -> Result<Option<String>> {
    let credential_name = "API key";
    let key_prompt = match (existing_provider, provider.requires_api_key()) {
        (Some(_), true) => format!(
            "{} {credential_name} (Enter keeps the current value) > ",
            provider.display_name(),
        ),
        (None, true) => format!("{} {credential_name} > ", provider.display_name()),
        (Some(_), false) => format!(
            "{} {credential_name} (optional; Enter keeps it, - clears it) > ",
            provider.display_name(),
        ),
        (None, false) => format!(
            "{} {credential_name} (optional) > ",
            provider.display_name()
        ),
    };
    let password_config = rpassword::ConfigBuilder::new()
        .password_feedback_mask('*')
        .build();
    let entered_key = rpassword::prompt_password_with_config(key_prompt, password_config)
        .with_context(|| format!("could not read the {credential_name}"))?;
    let api_key = match (entered_key.trim(), existing_provider) {
        ("-", _) if !provider.requires_api_key() => None,
        ("", Some(config)) => config.provider.api_key.clone(),
        ("", None) if provider.requires_api_key() => {
            bail!(
                "a {} {credential_name} is required",
                provider.display_name()
            )
        }
        ("", None) => None,
        (value, _) => Some(value.to_owned()),
    };
    Ok(api_key)
}

fn prompt_value(prompt: &str, default: Option<&str>) -> Result<String> {
    print!("{prompt}");
    std::io::stdout()
        .flush()
        .context("could not display setup prompt")?;
    let mut entered = String::new();
    std::io::stdin()
        .read_line(&mut entered)
        .context("could not read setup input")?;
    Ok(match entered.trim() {
        "" => default.unwrap_or_default().to_owned(),
        value => value.to_owned(),
    })
}

const fn default_timeout_seconds() -> u64 {
    DEFAULT_TIMEOUT_SECONDS
}

const fn default_max_output_tokens() -> u32 {
    DEFAULT_MAX_OUTPUT_TOKENS
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::tempdir;

    use super::{Config, GenerationSettings, ProviderKind, ProviderSettings};

    fn config() -> Config {
        Config {
            provider: ProviderSettings {
                kind: ProviderKind::Openrouter,
                api_key: Some("secret-value".into()),
                model: "openrouter/auto".into(),
                base_url: Some("https://openrouter.ai/api/v1".into()),
            },
            generation: GenerationSettings::default(),
        }
    }

    #[test]
    fn saves_and_loads_a_private_config() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("logshard/config.toml");
        config().save_to(&path).unwrap();

        let loaded = Config::load_from(&path).unwrap();
        assert_eq!(loaded.provider.kind, ProviderKind::Openrouter);
        assert_eq!(loaded.provider.api_key.as_deref(), Some("secret-value"));
        assert_eq!(loaded.generation.max_output_tokens, 4096);

        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;

            assert_eq!(fs::metadata(&path).unwrap().mode() & 0o777, 0o600);
            assert_eq!(
                fs::metadata(path.parent().unwrap()).unwrap().mode() & 0o777,
                0o700
            );
        }
    }

    #[test]
    fn rejects_unknown_and_incomplete_configuration() {
        let old_cli_shape = toml::from_str::<Config>(
            r#"provider = "openai"
model = "gpt-5.6"
base_url = "https://api.openai.com/v1"
api_key = "secret"
"#,
        );
        assert!(old_cli_shape.is_err());

        let missing_key = toml::from_str::<Config>(
            r#"[provider]
type = "openai"
model = "gpt-5.6"
base_url = "https://api.openai.com/v1"
"#,
        )
        .unwrap();
        assert!(missing_key.validate().is_err());

        let missing_base_url = toml::from_str::<Config>(
            r#"[provider]
type = "openai"
api_key = "secret"
model = "gpt-5.6"
"#,
        )
        .unwrap();
        assert!(missing_base_url.validate().is_err());
    }

    #[test]
    fn accepts_keyless_loopback_llama_cpp() {
        let config = Config {
            provider: ProviderSettings {
                kind: ProviderKind::LlamaCpp,
                api_key: None,
                model: "local".into(),
                base_url: Some("http://[::1]:8080/v1".into()),
            },
            generation: GenerationSettings::default(),
        };
        config.validate().unwrap();
    }

    #[test]
    fn accepts_github_copilot_without_a_base_url() {
        let config = Config {
            provider: ProviderSettings {
                kind: ProviderKind::GithubCopilot,
                api_key: None,
                model: "gpt-5.4".into(),
                base_url: None,
            },
            generation: GenerationSettings::default(),
        };

        config.validate().unwrap();
        let serialized = toml::to_string(&config).unwrap();
        assert!(!serialized.contains("base_url"));
        assert!(!serialized.contains("api_key"));
    }

    #[cfg(unix)]
    #[test]
    fn rejects_a_group_readable_config() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempdir().unwrap();
        let path = directory.path().join("logshard/config.toml");
        config().save_to(&path).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o640)).unwrap();

        let error = Config::load_from(&path).unwrap_err().to_string();
        assert!(error.contains("insecure permissions"));
    }
}
