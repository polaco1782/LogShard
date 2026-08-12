use std::{fs, io, path::Path, sync::Arc};

use anyhow::{Context, Result, bail};
use clap::Parser;
use logshard::{
    agent,
    cli::{Cli, Command, OutputFormat},
    config::{self, Config},
    corpus::{Corpus, IngestOptions},
    debug::ActivityLog,
    provider::{GithubCopilotClient, OpenAiCompatibleClient, ProviderConfig, ProviderKind},
    tools::ToolRegistry,
    ui::ActivityIndicator,
};
use serde::Serialize;
use serde_json::Value;

const MAX_PROMPT_BYTES: usize = 256 * 1024;

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("error: {error:#}");
        std::process::exit(1);
    }
}

async fn run() -> Result<()> {
    let cli = Cli::parse();
    if matches!(cli.command, Some(Command::Setup)) {
        let path = config::interactive_setup()?;
        println!("Configuration saved to {}", path.display());
        return Ok(());
    }

    if cli.inputs.is_empty() {
        bail!("at least one log input is required");
    }
    let prompt = load_prompt(&cli)?;
    let provider_config: ProviderConfig = Config::load()?.provider_config()?;
    let activity = ActivityLog::new(
        cli.debug,
        provider_config.api_key.as_deref(),
        cli.debug_max_text_bytes,
    );
    let corpus = Arc::new(Corpus::ingest(
        &cli.inputs,
        IngestOptions {
            max_input_bytes: cli.max_input_bytes,
            max_line_bytes: cli.max_line_bytes,
            redact_secrets: !cli.no_redact_secrets,
        },
    )?);
    activity.emit("corpus.ingested", corpus.stats());
    let tools = ToolRegistry::new(corpus.clone(), cli.max_tool_bytes);
    let answer_format = match cli.output {
        OutputFormat::Plain => agent::AnswerFormat::Plain,
        OutputFormat::Json => agent::AnswerFormat::Json,
    };
    let indicator = ActivityIndicator::start("Analyzing logs", !cli.debug);
    let investigation = if provider_config.kind == ProviderKind::GithubCopilot {
        let client = GithubCopilotClient::new(provider_config.clone())?;
        agent::investigate_with_github_copilot(
            &client,
            &tools,
            &prompt,
            answer_format,
            cli.max_steps,
            cli.max_answer_bytes,
            &activity,
        )
        .await
    } else {
        let client = OpenAiCompatibleClient::new(provider_config.clone())?;
        agent::investigate(
            &client,
            &tools,
            &prompt,
            answer_format,
            cli.max_steps,
            cli.max_answer_bytes,
            &activity,
        )
        .await
    };
    drop(indicator);
    let result = investigation?;

    match cli.output {
        OutputFormat::Plain => println!("{}", result.answer),
        OutputFormat::Json => {
            if result.answer_truncated {
                bail!(
                    "model's JSON answer exceeded the {}-byte limit; increase --max-answer-bytes",
                    cli.max_answer_bytes
                );
            }
            let output = JsonOutput {
                answer: parse_json_answer(&result.answer)?,
                answer_truncated: result.answer_truncated,
                provider: provider_config.kind,
                model: &provider_config.model,
                tool_calls: result.tool_calls,
                steps: result.steps,
                corpus: corpus.stats(),
            };
            serde_json::to_writer_pretty(io::stdout().lock(), &output)
                .context("failed to write JSON output")?;
            println!();
        }
    }
    Ok(())
}

fn parse_json_answer(answer: &str) -> Result<Value> {
    let answer: Value =
        serde_json::from_str(answer).context("model returned invalid JSON for --output json")?;
    if !answer.is_object() {
        bail!("model returned a JSON value instead of the required JSON object");
    }
    Ok(answer)
}

fn load_prompt(cli: &Cli) -> Result<String> {
    let prompt = if let Some(prompt) = &cli.prompt {
        prompt.clone()
    } else if let Some(path) = &cli.prompt_file {
        read_prompt_file(path)?
    } else {
        bail!("an investigation instruction is required; use --prompt or --prompt-file")
    };
    let prompt = prompt.trim().to_owned();
    if prompt.is_empty() {
        bail!("investigation prompt must not be empty");
    }
    if prompt.len() > MAX_PROMPT_BYTES {
        bail!("investigation prompt exceeds the {MAX_PROMPT_BYTES} byte limit");
    }
    Ok(prompt)
}

fn read_prompt_file(path: &Path) -> Result<String> {
    if path.as_os_str() == "-" {
        bail!("--prompt-file requires a file path; standard input is reserved for log input");
    }
    let metadata = fs::metadata(path)
        .with_context(|| format!("failed to inspect prompt file {}", path.display()))?;
    if metadata.len() > MAX_PROMPT_BYTES as u64 {
        bail!("investigation prompt exceeds the {MAX_PROMPT_BYTES} byte limit");
    }
    fs::read_to_string(path)
        .with_context(|| format!("failed to read UTF-8 prompt file {}", path.display()))
}

#[derive(Serialize)]
struct JsonOutput<'a> {
    answer: Value,
    answer_truncated: bool,
    provider: logshard::provider::ProviderKind,
    model: &'a str,
    tool_calls: u32,
    steps: u32,
    corpus: &'a logshard::corpus::CorpusStats,
}

#[cfg(test)]
mod tests {
    use super::*;
    use logshard::{corpus::CorpusStats, provider::ProviderKind};
    use serde_json::json;

    #[test]
    fn json_output_embeds_the_model_answer_as_an_object() {
        let stats = CorpusStats::default();
        let output = JsonOutput {
            answer: parse_json_answer(r#"{"summary":"failed","warnings":[]}"#).unwrap(),
            answer_truncated: false,
            provider: ProviderKind::Openai,
            model: "test-model",
            tool_calls: 1,
            steps: 2,
            corpus: &stats,
        };

        let serialized = serde_json::to_value(output).unwrap();

        assert_eq!(
            serialized["answer"],
            json!({
                "summary": "failed",
                "warnings": []
            })
        );
    }

    #[test]
    fn json_output_rejects_non_object_answers() {
        assert!(parse_json_answer(r#"["failure"]"#).is_err());
        assert!(parse_json_answer("not JSON").is_err());
    }
}
