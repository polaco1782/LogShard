use std::{
    fs::{self, File},
    io::{self, BufRead, BufReader},
    path::{Path, PathBuf},
    sync::OnceLock,
};

use anyhow::{Context, Result, bail};
use regex::Regex;
use serde::Serialize;

#[derive(Clone, Copy, Debug)]
pub struct IngestOptions {
    pub max_input_bytes: u64,
    pub max_line_bytes: usize,
    pub redact_secrets: bool,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct CorpusStats {
    pub input_bytes: u64,
    pub total_lines: u64,
    pub retained_entries: u64,
    pub retained_occurrences: u64,
    pub skipped_empty: u64,
    pub skipped_noise: u64,
    pub compressed_duplicates: u64,
    pub truncated_lines: u64,
    pub redacted_lines: u64,
    pub source_count: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct SourceStats {
    pub name: String,
    pub input_bytes: u64,
    pub total_lines: u64,
    pub retained_entries: usize,
}

#[derive(Clone, Debug)]
pub struct LogEntry {
    pub line_start: u64,
    pub line_end: u64,
    pub repeats: u64,
    pub text: String,
}

#[derive(Clone, Debug)]
pub struct LogSource {
    stats: SourceStats,
    entries: Vec<LogEntry>,
}

impl LogSource {
    pub fn stats(&self) -> &SourceStats {
        &self.stats
    }

    pub fn entries(&self) -> &[LogEntry] {
        &self.entries
    }
}

#[derive(Clone, Debug, Default)]
pub struct Corpus {
    sources: Vec<LogSource>,
    stats: CorpusStats,
}

impl Corpus {
    pub fn ingest(paths: &[PathBuf], options: IngestOptions) -> Result<Self> {
        let stdin_count = paths.iter().filter(|path| path.as_os_str() == "-").count();
        if stdin_count > 1 {
            bail!("standard input may be specified only once");
        }

        let mut files = Vec::new();
        for path in paths.iter().filter(|path| path.as_os_str() != "-") {
            collect_files(path, &mut files)?;
        }
        files.sort();
        files.dedup();

        let mut corpus = Self::default();
        let mut budget = InputBudget::new(options.max_input_bytes);
        for path in files {
            let file = File::open(&path)
                .with_context(|| format!("failed to open log input {}", path.display()))?;
            corpus.add_reader(
                path.display().to_string(),
                BufReader::new(file),
                options,
                &mut budget,
            )?;
        }

        if stdin_count == 1 {
            let stdin = io::stdin();
            corpus.add_reader("<stdin>".to_owned(), stdin.lock(), options, &mut budget)?;
        }

        if corpus.sources.is_empty() {
            bail!("no regular log files were found in the supplied inputs");
        }
        corpus.finish();
        Ok(corpus)
    }

    pub fn stats(&self) -> &CorpusStats {
        &self.stats
    }

    pub fn sources(&self) -> &[LogSource] {
        &self.sources
    }

    pub fn source(&self, name: &str) -> Option<&LogSource> {
        self.sources.iter().find(|source| source.stats.name == name)
    }

    fn add_reader(
        &mut self,
        name: String,
        mut reader: impl BufRead,
        options: IngestOptions,
        budget: &mut InputBudget,
    ) -> Result<()> {
        let mut entries: Vec<LogEntry> = Vec::new();
        let mut buffer = Vec::new();
        let mut source_bytes = 0_u64;
        let mut line_number = 0_u64;

        loop {
            buffer.clear();
            let bounded_line =
                read_bounded_line(&mut reader, &mut buffer, options.max_line_bytes, budget)
                    .with_context(|| format!("failed while reading {name}"))?;
            if bounded_line.bytes_read == 0 {
                break;
            }
            source_bytes += bounded_line.bytes_read;
            line_number += 1;
            self.stats.total_lines += 1;

            while matches!(buffer.last(), Some(b'\n' | b'\r')) {
                buffer.pop();
            }
            let raw = String::from_utf8_lossy(&buffer);
            let Some(mut sanitized) = sanitize_line(&raw, options.redact_secrets) else {
                if raw.trim().is_empty() {
                    self.stats.skipped_empty += 1;
                } else {
                    self.stats.skipped_noise += 1;
                }
                continue;
            };
            if sanitized.redacted {
                self.stats.redacted_lines += 1;
            }
            if bounded_line.truncated || sanitized.text.len() > options.max_line_bytes {
                sanitized.text = truncate_utf8(&sanitized.text, options.max_line_bytes);
                sanitized.text.push_str(" …[truncated]");
                self.stats.truncated_lines += 1;
            }

            self.stats.retained_occurrences += 1;
            if let Some(previous) = entries.last_mut()
                && previous.text == sanitized.text
            {
                previous.line_end = line_number;
                previous.repeats += 1;
                self.stats.compressed_duplicates += 1;
                continue;
            }

            entries.push(LogEntry {
                line_start: line_number,
                line_end: line_number,
                repeats: 1,
                text: sanitized.text,
            });
        }

        self.stats.input_bytes = budget.used;
        self.sources.push(LogSource {
            stats: SourceStats {
                name,
                input_bytes: source_bytes,
                total_lines: line_number,
                retained_entries: entries.len(),
            },
            entries,
        });
        Ok(())
    }

    fn finish(&mut self) {
        self.stats.retained_entries = self
            .sources
            .iter()
            .map(|source| source.entries.len() as u64)
            .sum();
        self.stats.source_count = self.sources.len();
    }
}

pub fn fingerprint(text: &str) -> String {
    static PREFIX_TIMESTAMP: OnceLock<Regex> = OnceLock::new();
    static UUID: OnceLock<Regex> = OnceLock::new();
    static IPV4: OnceLock<Regex> = OnceLock::new();
    static HEX: OnceLock<Regex> = OnceLock::new();
    static LONG_NUMBER: OnceLock<Regex> = OnceLock::new();

    let mut value = PREFIX_TIMESTAMP
        .get_or_init(|| {
            Regex::new(
                r"(?ix)^\s*(?:\[[^\]]*(?:\d{2}:\d{2}:\d{2}|T\d{2}:\d{2})[^\]]*\]|\d{4}-\d{2}-\d{2}[T ][0-9:.+\-Z]+|\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s*",
            )
            .expect("valid timestamp regex")
        })
        .replace(text, "<timestamp> ")
        .into_owned();
    value = UUID
        .get_or_init(|| {
            Regex::new(
                r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
            )
            .expect("valid UUID regex")
        })
        .replace_all(&value, "<uuid>")
        .into_owned();
    value = IPV4
        .get_or_init(|| Regex::new(r"\b(?:\d{1,3}\.){3}\d{1,3}\b").expect("valid IPv4 regex"))
        .replace_all(&value, "<ip>")
        .into_owned();
    value = HEX
        .get_or_init(|| Regex::new(r"(?i)\b0x[0-9a-f]+\b").expect("valid hex regex"))
        .replace_all(&value, "<hex>")
        .into_owned();
    value = LONG_NUMBER
        .get_or_init(|| Regex::new(r"\b\d{5,}\b").expect("valid number regex"))
        .replace_all(&value, "<number>")
        .into_owned();

    value
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

fn collect_files(path: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("failed to inspect input {}", path.display()))?;
    if metadata.file_type().is_symlink() {
        bail!("symlink inputs are not followed: {}", path.display());
    }
    if metadata.is_file() {
        files.push(path.to_owned());
        return Ok(());
    }
    if !metadata.is_dir() {
        bail!(
            "input is not a regular file or directory: {}",
            path.display()
        );
    }

    let mut children = fs::read_dir(path)
        .with_context(|| format!("failed to read input directory {}", path.display()))?
        .collect::<std::result::Result<Vec<_>, _>>()
        .with_context(|| format!("failed to enumerate input directory {}", path.display()))?;
    children.sort_by_key(|entry| entry.path());
    for child in children {
        let child_path = child.path();
        let child_type = child
            .file_type()
            .with_context(|| format!("failed to inspect {}", child_path.display()))?;
        if child_type.is_symlink() {
            continue;
        }
        if child_type.is_dir() {
            collect_files(&child_path, files)?;
        } else if child_type.is_file() {
            files.push(child_path);
        }
    }
    Ok(())
}

struct SanitizedLine {
    text: String,
    redacted: bool,
}

fn sanitize_line(raw: &str, redact_secrets: bool) -> Option<SanitizedLine> {
    static ANSI: OnceLock<Regex> = OnceLock::new();
    static BEARER: OnceLock<Regex> = OnceLock::new();
    static SECRET: OnceLock<Regex> = OnceLock::new();

    let without_ansi = ANSI
        .get_or_init(|| {
            Regex::new(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])").expect("valid ANSI regex")
        })
        .replace_all(raw, "");

    let control_count = without_ansi
        .chars()
        .filter(|character| character.is_control() && *character != '\t')
        .count();
    let character_count = without_ansi.chars().count();
    if character_count > 0 && control_count * 5 > character_count {
        return None;
    }

    let cleaned: String = without_ansi
        .chars()
        .filter(|character| !character.is_control() || *character == '\t')
        .collect();
    let cleaned = cleaned.trim();
    if cleaned.is_empty() || !cleaned.chars().any(char::is_alphanumeric) {
        return None;
    }

    if !redact_secrets {
        return Some(SanitizedLine {
            text: cleaned.to_owned(),
            redacted: false,
        });
    }

    let bearer = BEARER.get_or_init(|| {
        Regex::new(r"(?i)\bauthorization\s*:\s*bearer\s+[a-z0-9._~+/=-]+")
            .expect("valid bearer token regex")
    });
    let secret = SECRET.get_or_init(|| {
        Regex::new(
            r#"(?ix)(["']?)(api[_-]?key|access[_-]?token|token|password|passwd|secret)(["']?)\s*[:=]\s*(?:"[^"]*"|'[^']*'|[^\s,;}]+)"#,
        )
        .expect("valid secret regex")
    });
    let redacted = bearer.is_match(cleaned) || secret.is_match(cleaned);
    let cleaned = bearer.replace_all(cleaned, "Authorization: Bearer <redacted>");
    Some(SanitizedLine {
        text: secret
            .replace_all(&cleaned, "$1$2$3=<redacted>")
            .into_owned(),
        redacted,
    })
}

fn truncate_utf8(value: &str, max_bytes: usize) -> String {
    let mut end = max_bytes.min(value.len());
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    value[..end].to_owned()
}

struct BoundedLine {
    bytes_read: u64,
    truncated: bool,
}

fn read_bounded_line(
    reader: &mut impl BufRead,
    output: &mut Vec<u8>,
    retained_limit: usize,
    budget: &mut InputBudget,
) -> Result<BoundedLine> {
    let mut bytes_read = 0_u64;
    let mut truncated = false;

    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            break;
        }
        let newline = available.iter().position(|byte| *byte == b'\n');
        let consumed = newline.map_or(available.len(), |position| position + 1);
        let content_bytes = newline.unwrap_or(consumed);
        let remaining = retained_limit.saturating_sub(output.len());
        let copied = content_bytes.min(remaining);
        output.extend_from_slice(&available[..copied]);
        truncated |= copied < content_bytes;

        budget.consume(consumed as u64)?;
        bytes_read += consumed as u64;
        reader.consume(consumed);
        if newline.is_some() {
            break;
        }
    }

    Ok(BoundedLine {
        bytes_read,
        truncated,
    })
}

struct InputBudget {
    used: u64,
    maximum: u64,
}

impl InputBudget {
    fn new(maximum: u64) -> Self {
        Self { used: 0, maximum }
    }

    fn consume(&mut self, bytes: u64) -> Result<()> {
        self.used = self
            .used
            .checked_add(bytes)
            .context("input size overflow")?;
        if self.used > self.maximum {
            bail!(
                "log inputs exceed the configured {} byte limit",
                self.maximum
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    fn ingest(input: &str) -> Corpus {
        let options = IngestOptions {
            max_input_bytes: 1_000_000,
            max_line_bytes: 1024,
            redact_secrets: true,
        };
        let mut corpus = Corpus::default();
        corpus
            .add_reader(
                "test.log".to_owned(),
                Cursor::new(input),
                options,
                &mut InputBudget::new(options.max_input_bytes),
            )
            .unwrap();
        corpus.finish();
        corpus
    }

    #[test]
    fn compacts_noise_duplicates_and_secrets() {
        let corpus = ingest(
            "\x1b[31mERROR failed\x1b[0m\nERROR failed\n\n---\napi_key=super-secret user=4\nAuthorization: Bearer eyJ.secret\n",
        );
        assert_eq!(corpus.stats.retained_entries, 3);
        assert_eq!(corpus.stats.compressed_duplicates, 1);
        assert_eq!(corpus.stats.skipped_empty, 1);
        assert_eq!(corpus.stats.skipped_noise, 1);
        assert_eq!(corpus.stats.redacted_lines, 2);
        assert_eq!(corpus.sources[0].entries[0].repeats, 2);
        assert_eq!(
            corpus.sources[0].entries[1].text,
            "api_key=<redacted> user=4"
        );
        assert_eq!(
            corpus.sources[0].entries[2].text,
            "Authorization: Bearer <redacted>"
        );
    }

    #[test]
    fn fingerprints_volatile_values_without_erasing_status_codes() {
        let first =
            fingerprint("2026-08-10T01:02:03Z request 123456 from 10.0.0.1 failed with HTTP 503");
        let second =
            fingerprint("2026-08-10T01:02:04Z request 999999 from 10.0.0.2 failed with HTTP 503");
        assert_eq!(first, second);
        assert!(first.contains("503"));
    }

    #[test]
    fn enforces_input_budget() {
        let options = IngestOptions {
            max_input_bytes: 3,
            max_line_bytes: 1024,
            redact_secrets: true,
        };
        let mut corpus = Corpus::default();
        let error = corpus
            .add_reader(
                "test.log".to_owned(),
                Cursor::new("four"),
                options,
                &mut InputBudget::new(options.max_input_bytes),
            )
            .unwrap_err();
        assert!(format!("{error:#}").contains("configured 3 byte limit"));
    }

    #[test]
    fn retains_only_the_configured_prefix_of_long_lines() {
        let options = IngestOptions {
            max_input_bytes: 1024,
            max_line_bytes: 8,
            redact_secrets: true,
        };
        let mut corpus = Corpus::default();
        corpus
            .add_reader(
                "test.log".to_owned(),
                Cursor::new("123456789012345\nnext\n"),
                options,
                &mut InputBudget::new(options.max_input_bytes),
            )
            .unwrap();
        corpus.finish();
        assert_eq!(corpus.stats.input_bytes, 21);
        assert_eq!(corpus.stats.truncated_lines, 1);
        assert_eq!(corpus.sources[0].entries[0].text, "12345678 …[truncated]");
        assert_eq!(corpus.sources[0].entries[1].text, "next");
    }
}
