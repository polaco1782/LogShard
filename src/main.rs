use std::{fs, io, path::Path};

use anyhow::{Context, Result, bail};
use clap::Parser;
use logshard::{
    agent,
    cli::{Cli, OutputFormat},
    corpus::{Corpus, IngestOptions},
    debug::ActivityLog,
    provider::{OpenAiCompatibleClient, ProviderConfig},
    tools::ToolRegistry,
};
use serde::Serialize;

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
    let prompt = load_prompt(&cli)?;
    let provider_config = ProviderConfig::resolve(
        cli.provider,
        cli.model,
        cli.base_url,
        cli.api_key,
        cli.timeout_seconds,
        cli.max_output_tokens,
    )?;
    let activity = ActivityLog::new(
        cli.debug,
        provider_config.api_key.as_deref(),
        cli.debug_max_text_bytes,
    );
    let corpus = Corpus::ingest(
        &cli.inputs,
        IngestOptions {
            max_input_bytes: cli.max_input_bytes,
            max_line_bytes: cli.max_line_bytes,
            redact_secrets: !cli.no_redact_secrets,
        },
    )?;
    activity.emit("corpus.ingested", corpus.stats());
    let client = OpenAiCompatibleClient::new(provider_config)?;
    let tools = ToolRegistry::new(&corpus, cli.max_tool_bytes);
    let result = agent::investigate(
        &client,
        &tools,
        &prompt,
        cli.max_steps,
        cli.max_answer_bytes,
        &activity,
    )
    .await?;

    match cli.output {
        OutputFormat::Plain => println!("{}", result.answer),
        OutputFormat::Json => {
            let output = JsonOutput {
                answer: &result.answer,
                answer_truncated: result.answer_truncated,
                provider: client.config().kind,
                model: &client.config().model,
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

fn load_prompt(cli: &Cli) -> Result<String> {
    let prompt = if let Some(prompt) = &cli.prompt {
        prompt.clone()
    } else if let Some(path) = &cli.prompt_file {
        read_prompt_file(path)?
    } else {
        unreachable!("clap requires an investigation instruction")
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
    answer: &'a str,
    answer_truncated: bool,
    provider: logshard::provider::ProviderKind,
    model: &'a str,
    tool_calls: u32,
    steps: u32,
    corpus: &'a logshard::corpus::CorpusStats,
}
