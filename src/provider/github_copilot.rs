use std::sync::{
    Arc,
    atomic::{AtomicU32, Ordering},
};
use std::{
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::{Context, Result, anyhow, bail};
use async_trait::async_trait;
use github_copilot_sdk::{
    Client, ClientMode, ClientOptions, DeferMode, InfiniteSessionConfig, MessageOptions,
    SessionConfig, SystemMessageConfig, Tool, ToolInvocation, ToolResult, ToolSet,
    install_bundled_cli,
    rpc::{Model, ModelCapabilitiesOverride, ModelCapabilitiesOverrideLimits, ModelPolicyState},
    tool::ToolHandler,
};

use super::{ProviderConfig, ProviderDefaults, ProviderKind};
use crate::{debug::ActivityLog, paths, secure_fs::ensure_private_directory, tools::ToolRegistry};

pub(super) fn defaults() -> ProviderDefaults {
    ProviderDefaults {
        model: "gpt-5.4",
        base_url: None,
        api_key_required: false,
    }
}

pub(crate) fn authenticate() -> Result<()> {
    let copilot_home = prepare_copilot_home()?;
    let program = install_bundled_cli()
        .context("the bundled GitHub Copilot CLI is unavailable for this platform")?;
    let status = login_command(&program, &copilot_home)
        .status()
        .context("failed to start GitHub Copilot authentication")?;
    if !status.success() {
        bail!("GitHub Copilot authentication failed with {status}");
    }
    Ok(())
}

fn prepare_copilot_home() -> Result<PathBuf> {
    let copilot_home = paths::copilot_home()?;
    secure_copilot_home(&copilot_home)?;
    Ok(copilot_home)
}

fn secure_copilot_home(copilot_home: &Path) -> Result<()> {
    let logshard_home = copilot_home
        .parent()
        .context("GitHub Copilot home has no parent directory")?;
    ensure_private_directory(logshard_home)?;
    ensure_private_directory(copilot_home)
}

fn login_command(program: &Path, copilot_home: &Path) -> Command {
    let mut command = Command::new(program);
    command
        .args(["login", "--device-code"])
        .current_dir(copilot_home)
        .env("COPILOT_HOME", copilot_home)
        .env("COPILOT_DISABLE_KEYTAR", "1")
        .env_remove("COPILOT_GITHUB_TOKEN")
        .env_remove("GH_TOKEN")
        .env_remove("GITHUB_TOKEN")
        .env_remove("GITHUB_COPILOT_API_TOKEN")
        .env_remove("COPILOT_API_URL")
        .env_remove("COPILOT_SDK_AUTH_TOKEN");
    command
}

pub struct GithubCopilotClient {
    config: ProviderConfig,
}

#[derive(Debug)]
pub struct CopilotRunResult {
    pub answer: String,
    pub tool_calls: u32,
    pub steps: u32,
}

impl GithubCopilotClient {
    pub fn new(config: ProviderConfig) -> Result<Self> {
        if config.kind != ProviderKind::GithubCopilot {
            bail!("GitHub Copilot client requires a github_copilot configuration");
        }
        Ok(Self { config })
    }

    pub async fn run(
        &self,
        system_prompt: &str,
        user_prompt: &str,
        tools: ToolRegistry,
        max_steps: u32,
        activity: &ActivityLog,
    ) -> Result<CopilotRunResult> {
        let copilot_home = prepare_copilot_home()?;
        let runtime_directory = tempfile::Builder::new()
            .prefix("logshard-copilot-")
            .tempdir()
            .context("failed to create an isolated GitHub Copilot runtime directory")?;
        let mut options = ClientOptions::new()
            .with_use_logged_in_user(self.config.api_key.is_none())
            .with_base_directory(copilot_home)
            .with_cwd(runtime_directory.path())
            .with_env_remove([
                "COPILOT_GITHUB_TOKEN",
                "GH_TOKEN",
                "GITHUB_TOKEN",
                "GITHUB_COPILOT_API_TOKEN",
                "COPILOT_API_URL",
            ])
            .with_mode(ClientMode::Empty);
        if let Some(github_token) = &self.config.api_key {
            options = options.with_github_token(github_token);
        }
        let client = Client::start(options)
            .await
            .context("failed to start the GitHub Copilot runtime")?;

        let outcome = async {
            let auth = client
                .get_auth_status()
                .await
                .context("failed to check GitHub Copilot authentication")?;
            if !auth.is_authenticated {
                bail!("GitHub Copilot is not authenticated; run `logshard setup`");
            }
            activity.emit(
                "provider.authenticated",
                serde_json::json!({
                    "provider": self.config.kind.to_string(),
                    "login": auth.login,
                    "host": auth.host
                }),
            );
            self.run_with_client(
                &client,
                system_prompt,
                user_prompt,
                tools,
                max_steps,
                activity,
            )
            .await
        }
        .await;
        if let Err(error) = client.stop().await {
            activity.emit(
                "provider.cleanup_error",
                serde_json::json!({ "error": error.to_string() }),
            );
            if outcome.is_ok() {
                return Err(anyhow!(error)).context("failed to stop the GitHub Copilot runtime");
            }
        }
        outcome
    }

    async fn run_with_client(
        &self,
        client: &Client,
        system_prompt: &str,
        user_prompt: &str,
        tools: ToolRegistry,
        max_steps: u32,
        activity: &ActivityLog,
    ) -> Result<CopilotRunResult> {
        let models = tokio::time::timeout(self.config.timeout, client.list_models())
            .await
            .context("timed out while loading GitHub Copilot models")?
            .context("failed to load GitHub Copilot models")?;
        let selected_model = models.iter().find(|model| model.id == self.config.model);
        if selected_model.is_none() {
            let available = models
                .iter()
                .filter(|model| !model_is_disabled(model))
                .map(|model| model.id.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            bail!(
                "GitHub Copilot model {:?} is not available for this account; available models: {available}",
                self.config.model
            );
        }
        if selected_model.is_some_and(model_is_disabled) {
            bail!(
                "GitHub Copilot model {:?} is disabled by account or organization policy",
                self.config.model
            );
        }
        activity.emit(
            "provider.models",
            serde_json::json!({
                "provider": self.config.kind.to_string(),
                "selected": self.config.model,
                "available": models.iter().map(|model| &model.id).collect::<Vec<_>>()
            }),
        );

        let tool_calls = Arc::new(AtomicU32::new(0));
        let copilot_tools = copilot_tools(tools, activity.clone(), tool_calls.clone())?;
        let available_tools = ToolSet::new().add_custom("*")?.into_vec();
        let capabilities = ModelCapabilitiesOverride {
            limits: Some(ModelCapabilitiesOverrideLimits {
                max_output_tokens: Some(i64::from(self.config.max_output_tokens)),
                ..Default::default()
            }),
            ..Default::default()
        };
        let session_config = SessionConfig::default()
            .with_model(&self.config.model)
            .with_client_name("logshard")
            .with_system_message(
                SystemMessageConfig::new()
                    .with_mode("replace")
                    .with_content(system_prompt),
            )
            .with_tools(copilot_tools)
            .with_available_tools(available_tools)
            .with_model_capabilities(capabilities)
            .with_enable_config_discovery(false)
            .with_infinite_sessions(InfiniteSessionConfig::new().with_enabled(false));
        let session = client
            .create_session(session_config)
            .await
            .context("failed to create a GitHub Copilot session")?;
        let mut events = session.subscribe();
        activity.emit(
            "provider.request",
            serde_json::json!({
                "provider": self.config.kind.to_string(),
                "model": self.config.model,
                "tools": ToolRegistry::definitions()
                    .iter()
                    .filter_map(|tool| tool.pointer("/function/name").and_then(serde_json::Value::as_str))
                    .collect::<Vec<_>>(),
                "max_output_tokens": self.config.max_output_tokens,
                "max_steps": max_steps
            }),
        );

        let run_timeout = self.config.timeout.saturating_mul(max_steps);
        let run = tokio::time::timeout(run_timeout, async {
            session
                .send(MessageOptions::new(user_prompt))
                .await
                .context("failed to send the GitHub Copilot request")?;
            collect_run(&session, &mut events, max_steps, activity).await
        })
        .await;
        let outcome = match run {
            Ok(outcome) => outcome,
            Err(_) => {
                let _ = session.abort().await;
                Err(anyhow!(
                    "GitHub Copilot request timed out after {} seconds",
                    run_timeout.as_secs()
                ))
            }
        };
        let cleanup = session.disconnect().await;
        let mut result = outcome?;
        cleanup.context("failed to close the GitHub Copilot session")?;
        result.tool_calls = tool_calls.load(Ordering::Relaxed);
        Ok(result)
    }
}

fn model_is_disabled(model: &Model) -> bool {
    matches!(
        model.policy.as_ref().map(|policy| &policy.state),
        Some(ModelPolicyState::Disabled)
    )
}

async fn collect_run(
    session: &github_copilot_sdk::session::Session,
    events: &mut github_copilot_sdk::subscription::EventSubscription,
    max_steps: u32,
    activity: &ActivityLog,
) -> Result<CopilotRunResult> {
    let mut answer = None;
    let mut steps = 0_u32;
    let mut seen_turn = false;
    loop {
        let event = events
            .recv()
            .await
            .context("GitHub Copilot event stream closed unexpectedly")?;
        if event.agent_id.is_some() {
            continue;
        }
        match event.event_type.as_str() {
            "assistant.turn_start" => {
                seen_turn = true;
                steps = steps
                    .checked_add(1)
                    .context("GitHub Copilot step counter overflow")?;
                activity.emit("agent.step", serde_json::json!({ "step": steps }));
                if steps > max_steps {
                    session.abort().await.ok();
                    bail!(
                        "agent reached the configured {max_steps} step limit without a final answer"
                    );
                }
            }
            "assistant.message" => {
                if let Some(content) = event.data.get("content").and_then(|value| value.as_str()) {
                    activity.emit(
                        "provider.response",
                        serde_json::json!({ "body": event.data }),
                    );
                    if !content.trim().is_empty() {
                        answer = Some(content.trim().to_owned());
                    }
                }
            }
            "session.error" if !event.is_transient_error() => {
                let message = event
                    .data
                    .get("message")
                    .and_then(|value| value.as_str())
                    .unwrap_or("unknown GitHub Copilot error");
                bail!("GitHub Copilot session failed: {message}");
            }
            "session.idle" if seen_turn => break,
            _ => {}
        }
    }

    Ok(CopilotRunResult {
        answer: answer.context("GitHub Copilot returned no final answer")?,
        tool_calls: 0,
        steps,
    })
}

fn copilot_tools(
    registry: ToolRegistry,
    activity: ActivityLog,
    tool_calls: Arc<AtomicU32>,
) -> Result<Vec<Tool>> {
    let handler = Arc::new(LogShardToolHandler {
        registry,
        activity,
        tool_calls,
    });
    ToolRegistry::definitions()
        .into_iter()
        .map(|definition| {
            let function = definition
                .get("function")
                .and_then(serde_json::Value::as_object)
                .context("invalid LogShard tool definition")?;
            let name = function
                .get("name")
                .and_then(serde_json::Value::as_str)
                .context("LogShard tool definition has no name")?;
            let description = function
                .get("description")
                .and_then(serde_json::Value::as_str)
                .context("LogShard tool definition has no description")?;
            let parameters = function
                .get("parameters")
                .cloned()
                .context("LogShard tool definition has no parameters")?;
            Ok(Tool::new(name)
                .with_description(description)
                .with_parameters(parameters)
                .with_skip_permission(true)
                .with_defer(DeferMode::Never)
                .with_handler(handler.clone()))
        })
        .collect()
}

struct LogShardToolHandler {
    registry: ToolRegistry,
    activity: ActivityLog,
    tool_calls: Arc<AtomicU32>,
}

#[async_trait]
impl ToolHandler for LogShardToolHandler {
    async fn call(&self, invocation: ToolInvocation) -> github_copilot_sdk::Result<ToolResult> {
        let arguments = serde_json::to_string(&invocation.arguments)?;
        let call_number = self.tool_calls.fetch_add(1, Ordering::Relaxed) + 1;
        self.activity.emit(
            "tool.call",
            serde_json::json!({
                "call": call_number,
                "call_id": invocation.tool_call_id,
                "type": "function",
                "name": invocation.tool_name,
                "arguments": invocation.arguments
            }),
        );
        let result = self.registry.execute(&invocation.tool_name, &arguments);
        self.activity.emit(
            "tool.result",
            serde_json::json!({
                "call": call_number,
                "call_id": invocation.tool_call_id,
                "name": invocation.tool_name,
                "result": serde_json::from_str::<serde_json::Value>(&result)
                    .unwrap_or_else(|_| serde_json::Value::String(result.clone()))
            }),
        );
        Ok(ToolResult::Text(result))
    }
}

#[cfg(test)]
mod tests {
    use github_copilot_sdk::rpc::ModelPolicy;

    use crate::corpus::{Corpus, IngestOptions};

    use super::*;

    #[test]
    fn defaults_use_the_sdk_managed_endpoint() {
        let defaults = defaults();
        assert_eq!(defaults.model, "gpt-5.4");
        assert!(defaults.base_url.is_none());
        assert!(!defaults.api_key_required);
    }

    #[test]
    fn login_uses_the_device_flow_and_app_scoped_credentials() {
        use std::ffi::OsStr;

        let command = login_command(Path::new("copilot"), Path::new("private/copilot"));
        assert_eq!(
            command.get_args().collect::<Vec<_>>(),
            [OsStr::new("login"), OsStr::new("--device-code")]
        );
        assert_eq!(
            command.get_current_dir(),
            Some(Path::new("private/copilot"))
        );

        let environment = command
            .get_envs()
            .collect::<std::collections::HashMap<_, _>>();
        assert_eq!(
            environment.get(OsStr::new("COPILOT_HOME")),
            Some(&Some(OsStr::new("private/copilot")))
        );
        assert_eq!(
            environment.get(OsStr::new("COPILOT_DISABLE_KEYTAR")),
            Some(&Some(OsStr::new("1")))
        );
        assert_eq!(environment.get(OsStr::new("GH_TOKEN")), Some(&None));
        assert_eq!(environment.get(OsStr::new("GITHUB_TOKEN")), Some(&None));
    }

    #[cfg(unix)]
    #[test]
    fn copilot_home_is_private() {
        use std::os::unix::fs::MetadataExt;

        let directory = tempfile::tempdir().unwrap();
        let copilot_home = directory.path().join("logshard/copilot");
        secure_copilot_home(&copilot_home).unwrap();

        assert_eq!(
            std::fs::metadata(copilot_home.parent().unwrap())
                .unwrap()
                .mode()
                & 0o777,
            0o700
        );
        assert_eq!(
            std::fs::metadata(copilot_home).unwrap().mode() & 0o777,
            0o700
        );
    }

    #[test]
    fn exposes_only_the_bounded_log_tools() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("test.log");
        std::fs::write(&path, "INFO ready\n").unwrap();
        let corpus = Corpus::ingest(
            &[path],
            IngestOptions {
                max_input_bytes: 1024,
                max_line_bytes: 1024,
                redact_secrets: true,
            },
        )
        .unwrap();
        let tools = copilot_tools(
            ToolRegistry::new(corpus.into(), 1024),
            ActivityLog::disabled(),
            Arc::new(AtomicU32::new(0)),
        )
        .unwrap();

        assert_eq!(tools.len(), 5);
        assert!(tools.iter().all(|tool| tool.skip_permission));
        assert!(
            tools
                .iter()
                .all(|tool| tool.defer == Some(DeferMode::Never))
        );
        assert!(tools.iter().all(|tool| tool.handler().is_some()));
        assert_eq!(
            tools
                .iter()
                .map(|tool| tool.name.as_str())
                .collect::<Vec<_>>(),
            vec![
                "corpus_stats",
                "list_sources",
                "search_logs",
                "read_lines",
                "top_patterns"
            ]
        );
    }

    #[test]
    fn honors_copilot_model_policy() {
        let mut model = Model::default();
        assert!(!model_is_disabled(&model));

        model.policy = Some(ModelPolicy {
            state: ModelPolicyState::Disabled,
            terms: None,
        });
        assert!(model_is_disabled(&model));
    }
}
