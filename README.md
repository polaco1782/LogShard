# LogShard

LogShard is a Rust CLI for investigating large log sets without placing the entire input in an AI context window. It ingests and cleans logs locally, gives the model a small set of bounded read-only tools, and lets the model retrieve only the evidence needed for the investigation prompt.

It supports OpenAI, OpenRouter, and a local `llama-server` from llama.cpp through one OpenAI-compatible Chat Completions tool loop.

## How it works

1. LogShard recursively ingests regular files or reads logs from standard input.
2. It removes blank/noise-only lines and ANSI/control data, redacts common credentials, truncates pathological lines, and compresses consecutive duplicates while retaining original line ranges and occurrence counts.
3. The model receives the investigation prompt, compact corpus statistics, and five local tools—not the logs themselves.
4. The model can list sources, search literal text or regular expressions, read bounded line ranges, and inspect repeated normalized patterns.
5. LogShard prints either a plain answer or a stable JSON envelope.

Search results group repeated patterns whose timestamps, UUIDs, IP addresses, long IDs, or hexadecimal addresses differ. This grouping affects only tool output; the ordered corpus remains available to `read_lines`.

## Build

```sh
cargo build --release --locked
```

The binary is written to `target/release/logshard`.

## Usage

Analyze one file with OpenAI:

```sh
export OPENAI_API_KEY='...'
logshard service.log \
  --prompt 'Find the root cause of the last failed startup and cite the evidence'
```

Analyze a directory with OpenRouter and return JSON:

```sh
export OPENROUTER_API_KEY='...'
logshard ./logs \
  --provider openrouter \
  --model openai/gpt-5.6 \
  --prompt-file investigation.txt \
  --output json
```

Pipe logs from another command:

```sh
journalctl -u my-service --since today | \
  logshard - --prompt 'Explain the outage timeline and identify the first causal error'
```

Run against llama.cpp:

```sh
llama-server \
  --model model.gguf \
  --alias local \
  --jinja \
  --host 127.0.0.1 \
  --port 8080

logshard app.log \
  --provider llama-cpp \
  --model local \
  --prompt 'Find memory exhaustion evidence'
```

The selected llama.cpp model and chat template must support tool calling. If the server uses a different alias or address, set `--model` and `--base-url` accordingly.

Run `logshard --help` for ingestion and tool-size limits. Important environment variables are:

| Setting | CLI option | Environment |
| --- | --- | --- |
| Provider | `--provider` | `LOGSHARD_PROVIDER` |
| Model | `--model` | `LOGSHARD_MODEL` |
| Base URL | `--base-url` | `LOGSHARD_BASE_URL` |
| Generic API key | `--api-key` | `LOGSHARD_API_KEY` |
| OpenAI API key | — | `OPENAI_API_KEY` |
| OpenRouter API key | — | `OPENROUTER_API_KEY` |

`LOGSHARD_API_KEY` takes precedence over the provider-specific key. Plain HTTP provider URLs are accepted only for loopback hosts; remote providers must use HTTPS.

Each provider response is capped at 4096 generated tokens by default. Because that token budget also covers non-visible reasoning and message structure, the final visible answer has a separate 8192-byte hard limit. Adjust these independently with `--max-output-tokens` and `--max-answer-bytes`.

## Debugging AI activity

Pass `--debug` to write every AI activity event as readable, pretty-printed JSON to standard error. The trace includes the prompt and corpus summary, provider request metadata and model responses, agent steps, tool arguments, tool results, and the final answer. Static tool schemas and accumulated conversation history are summarized instead of repeated on every request:

```sh
logshard app.log --prompt 'Find the failure' --debug 2>logshard-debug.log
```

Debugging never changes plain or JSON output on standard output, so pipelines remain usable. Long JSON string values are limited to 2048 bytes by default; use `--debug-max-text-bytes` to adjust the preview size. Authorization headers are never included and the configured API key is scrubbed if it appears in an event. Prompts and selected log excerpts are included, however, so treat the debug file as sensitive.

## JSON output

`--output json` always emits valid JSON even when a provider only supports plain text completions:

```json
{
  "answer": "The startup failed after ...",
  "answer_truncated": false,
  "provider": "openai",
  "model": "gpt-5.6",
  "tool_calls": 4,
  "steps": 5,
  "corpus": {
    "input_bytes": 481920,
    "total_lines": 8201,
    "retained_entries": 3110,
    "retained_occurrences": 7890,
    "skipped_empty": 120,
    "skipped_noise": 3,
    "compressed_duplicates": 4780,
    "truncated_lines": 0,
    "redacted_lines": 2,
    "source_count": 4
  }
}
```

## Privacy and limits

- Common bearer tokens, API keys, access tokens, passwords, and secrets are redacted before any search result can reach a provider. Use `--no-redact-secrets` only when the investigation truly needs raw values.
- Tool arguments, regular-expression size, result count, result bytes, input bytes, line bytes, HTTP duration, and agent steps are bounded.
- Symlink inputs are not followed. Local tools are read-only and do not invoke a shell.
- Using OpenAI or OpenRouter sends the selected excerpts and prompt to that provider. llama.cpp keeps inference local when its server is local.

## Development

```sh
cargo fmt --all -- --check
cargo clippy --all-targets --all-features --offline -- -D warnings
cargo test --all-targets --offline
```
