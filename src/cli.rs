use std::path::PathBuf;

use clap::{ArgAction, Parser, Subcommand, ValueEnum};
use serde::Serialize;

#[derive(Debug, Parser)]
#[command(
    name = "logshard",
    version,
    about = "Analyze large logs with an AI-guided, token-efficient local tool loop",
    args_conflicts_with_subcommands = true
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,

    /// Log files, directories, or '-' for standard input
    #[arg(value_name = "INPUT")]
    pub inputs: Vec<PathBuf>,

    /// Investigation instructions for the AI
    #[arg(short, long, conflicts_with = "prompt_file")]
    pub prompt: Option<String>,

    /// Read investigation instructions from a UTF-8 file
    #[arg(long, value_name = "PATH", conflicts_with = "prompt")]
    pub prompt_file: Option<PathBuf>,

    /// Final output format
    #[arg(long, value_enum, default_value = "plain")]
    pub output: OutputFormat,

    /// Maximum agent/tool round trips
    #[arg(long, default_value_t = 12, value_parser = clap::value_parser!(u32).range(1..=64))]
    pub max_steps: u32,

    /// Maximum UTF-8 bytes printed from the model's final answer
    #[arg(long, default_value_t = 8192, value_parser = parse_answer_bytes)]
    pub max_answer_bytes: usize,

    /// Maximum bytes accepted across all inputs
    #[arg(long, default_value_t = 134_217_728, value_parser = clap::value_parser!(u64).range(1..))]
    pub max_input_bytes: u64,

    /// Maximum bytes returned by one local tool call
    #[arg(long, default_value_t = 65_536, value_parser = parse_tool_bytes)]
    pub max_tool_bytes: usize,

    /// Maximum retained bytes from any individual log line
    #[arg(long, default_value_t = 16_384, value_parser = parse_line_bytes)]
    pub max_line_bytes: usize,

    /// Preserve secrets instead of redacting common credentials locally
    #[arg(long, action = ArgAction::SetTrue)]
    pub no_redact_secrets: bool,

    /// Log full AI requests, responses, and tool activity to stderr; may expose log content
    #[arg(long, action = ArgAction::SetTrue)]
    pub debug: bool,

    /// Maximum UTF-8 bytes retained in each debug JSON string
    #[arg(long, default_value_t = 2048, value_parser = parse_debug_text_bytes)]
    pub debug_max_text_bytes: usize,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Interactively create or update the private AI provider configuration
    Setup,
}

#[derive(Clone, Copy, Debug, Serialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum OutputFormat {
    Plain,
    Json,
}

fn parse_tool_bytes(value: &str) -> Result<usize, String> {
    parse_byte_limit(value, 1_024)
}

fn parse_line_bytes(value: &str) -> Result<usize, String> {
    parse_byte_limit(value, 256)
}

fn parse_answer_bytes(value: &str) -> Result<usize, String> {
    parse_byte_limit(value, 512)
}

fn parse_debug_text_bytes(value: &str) -> Result<usize, String> {
    parse_byte_limit(value, 128)
}

fn parse_byte_limit(value: &str, minimum: usize) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|error| format!("invalid byte count: {error}"))?;
    if (minimum..=1_048_576).contains(&parsed) {
        Ok(parsed)
    } else {
        Err(format!("value must be between {minimum} and 1048576 bytes"))
    }
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::{Cli, Command};

    #[test]
    fn setup_does_not_require_analysis_arguments() {
        let cli = Cli::try_parse_from(["logshard", "setup"]).unwrap();
        assert!(matches!(cli.command, Some(Command::Setup)));
        assert!(cli.inputs.is_empty());
    }

    #[test]
    fn provider_flags_are_not_accepted() {
        assert!(
            Cli::try_parse_from([
                "logshard",
                "service.log",
                "--prompt",
                "find the failure",
                "--provider",
                "openai"
            ])
            .is_err()
        );
    }
}
