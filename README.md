# LogShard

LogShard is a Rust CLI for investigating large log sets without placing the entire input in an AI context window. It ingests and cleans logs locally, gives the model a small set of bounded read-only tools, and lets the model retrieve only the evidence needed for the investigation prompt.

It supports OpenAI, OpenRouter, GitHub Copilot subscription models, and a local `llama-server` from llama.cpp. OpenAI-compatible providers use LogShard's Chat Completions loop; GitHub Copilot uses GitHub's official Copilot SDK with the same bounded local tools.

![LogShard terminal demo](logshard-demo.gif)

## How it works

1. LogShard recursively ingests regular files or reads logs from standard input.
2. It removes blank/noise-only lines and ANSI/control data, redacts common credentials, truncates pathological lines, and compresses consecutive duplicates while retaining original line ranges and occurrence counts.
3. The model receives the investigation prompt, compact corpus statistics, and five local tools—not the logs themselves.
4. The model can list sources, search literal text or regular expressions, read bounded line ranges, and inspect repeated normalized patterns.
5. LogShard prints either a plain answer or a stable JSON envelope.

Search results group repeated patterns whose timestamps, UUIDs, IP addresses, long IDs, or hexadecimal addresses differ. This grouping affects only tool output; the ordered corpus remains available to `read_lines`.

## Build

LogShard requires Rust 1.94 or newer.

```sh
cargo build --release --locked
```

The binary is written to `target/release/logshard`.

To install it in Cargo's binary directory instead:

```sh
cargo install --path . --locked
```

## Usage

Configure the upstream AI server first:

```sh
logshard setup
```

Setup interactively selects OpenAI, OpenRouter, llama.cpp, or GitHub Copilot and asks for the applicable settings. It writes one private configuration file at `$XDG_CONFIG_HOME/logshard/config.toml`, falling back to `~/.config/logshard/config.toml` on Unix and `%LOCALAPPDATA%\logshard\config.toml` on Windows.

Analyze one file:

```sh
logshard service.log \
  --prompt 'Find the root cause of the last failed startup and cite the evidence'
```

Analyze a directory and return JSON:

```sh
logshard ./logs \
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

logshard setup
logshard app.log \
  --prompt 'Find memory exhaustion evidence'
```

Choose llama.cpp during setup and accept its default `http://127.0.0.1:8080/v1` base URL, or enter the address exposed by your server. The selected model and chat template must support tool calling.

Run with a GitHub Copilot subscription:

```sh
logshard setup
logshard app.log \
  --prompt 'Find the first error that caused the deployment rollback'
```

Choose GitHub Copilot during setup. LogShard starts GitHub's OAuth device flow and displays the verification URL and one-time code. Open the URL, enter the code, and authorize Copilot; setup waits for that authorization before asking for a model ID such as `gpt-5.4`. LogShard validates the configured model against the live model catalog before starting an investigation, so organization policy and account-specific availability are respected.

The OAuth credential is kept in LogShard's private, app-scoped Copilot directory rather than in `config.toml`. The official Copilot runtime is bundled into LogShard at build time, so users do not need to install the `copilot` CLI separately. GitHub Copilot SDK support is currently a preview feature upstream and requires a GitHub Copilot subscription.

The generated TOML has this shape:

```toml
[provider]
type = "openai"
api_key = "..."
model = "gpt-5.6"
base_url = "https://api.openai.com/v1"

[generation]
timeout_seconds = 120
max_output_tokens = 4096
```

Valid provider types are `openai`, `openrouter`, `llama_cpp`, and `github_copilot`. `api_key` may be omitted for llama.cpp and must be omitted for the normal GitHub Copilot device-login flow. Plain HTTP URLs are accepted only for loopback servers; remote providers must use HTTPS. GitHub Copilot routing is managed by its SDK, so `base_url` must also be omitted for that provider:

```toml
[provider]
type = "github_copilot"
model = "gpt-5.4"

[generation]
timeout_seconds = 120
max_output_tokens = 4096
```

The configuration directory and file are written with private permissions on Unix, and linked or group/world-readable configuration files are rejected.

Provider settings are read only from this file. The former provider/model/base-URL/API-key command-line options and their environment-variable equivalents are not supported.

Each provider response is capped at 4096 generated tokens by default. Because that token budget also covers non-visible reasoning and message structure, the final visible answer has a separate 8192-byte hard limit. Edit `generation.max_output_tokens` in the TOML file and use `--max-answer-bytes` to adjust these independently.

Run `logshard --help` for ingestion and tool-size limits.

While the model is working, interactive terminals show an in-place spinner with elapsed time on standard error. It is disabled when standard error is redirected or `--debug` is active, so JSON output and pipelines remain clean.

### Private HTTPS proxies and additional CA trust

An OpenAI-compatible proxy uses `type = "openai"` with its HTTPS `base_url`, model and API key. For a proxy signed by a private CA, add `provider.ca_cert_pem` as a TOML multiline string containing the public CA certificate(s):

```toml
[provider]
type = "openai"
model = "your-model"
base_url = "https://proxy.example.internal/v1"
api_key = "..."
ca_cert_pem = """
-----BEGIN CERTIFICATE-----
...your trusted CA certificate, not a private key...
-----END CERTIFICATE-----
"""
```

When present, this bundle **replaces** the built-in root trust for this provider's HTTP client only. It does not modify OS trust, other providers, or GitHub Copilot's SDK. TLS chain, expiry and hostname/IP verification remain enabled, and redirects are not followed. Use the actual trusted CA supplied by your administrator, not a certificate fetched from an unverified endpoint. Without this setting, existing default trust behavior is unchanged.

The bundle must be nonempty PEM, at most 256 KiB, and contain no private key. It is rejected for plain HTTP and `github_copilot`. Integrators such as JirAIya can copy the selected proxy profile's CA into the isolated per-run TOML; no additional global certificate installation or verification bypass is needed. Existing `setup` preserves the CA setting for the same provider.

## Debugging AI activity

Pass `--debug` to write every AI activity event as readable, pretty-printed JSON to standard error. The trace includes the prompt and corpus summary, provider request metadata and model responses, agent steps, tool arguments, tool results, and the final answer. Static tool schemas and accumulated conversation history are summarized instead of repeated on every request:

```sh
logshard app.log --prompt 'Find the failure' --debug 2>logshard-debug.log
```

Debugging never changes plain or JSON output on standard output, so pipelines remain usable. Long JSON string values are limited to 2048 bytes by default; use `--debug-max-text-bytes` to adjust the preview size. Authorization headers are never included and the configured API key is scrubbed if it appears in an event. Prompts and selected log excerpts are included, however, so treat the debug file as sensitive.

## JSON output

`--output json` instructs the model to return a bare JSON object without Markdown fences or surrounding prose, then emits it inside LogShard's stable JSON envelope:

```json
{
  "answer": {
    "summary": "The startup failed after ...",
    "evidence": ["service.log:42"]
  },
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
- Using OpenAI, OpenRouter, or GitHub Copilot sends the selected excerpts and prompt to that provider. The Copilot backend runs in the SDK's empty mode with only LogShard's five read-only tools; it cannot access the repository, shell, ambient instructions, SDK memory, or session persistence. llama.cpp keeps inference local when its server is local.

## Development

```sh
cargo fmt --all -- --check
cargo clippy --all-targets --all-features --offline -- -D warnings
cargo test --all-targets --offline
```

## License

LogShard is available under the [MIT License](LICENSE).
