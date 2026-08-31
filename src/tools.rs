use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use anyhow::{Result, anyhow, bail};
use regex::RegexBuilder;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::corpus::{Corpus, LogEntry, fingerprint};

const MAX_PATTERN_BYTES: usize = 512;
const MAX_SEARCH_RESULTS: usize = 100;
const MAX_READ_LINES: u64 = 500;
const MAX_SOURCE_RESULTS: usize = 500;
const MAX_PATTERN_RESULTS: usize = 100;
const MAX_TEXT_BYTES: usize = 2_048;

#[derive(Clone)]
pub struct ToolRegistry {
    corpus: Arc<Corpus>,
    max_result_bytes: usize,
}

impl ToolRegistry {
    pub fn new(corpus: Arc<Corpus>, max_result_bytes: usize) -> Self {
        Self {
            corpus,
            max_result_bytes,
        }
    }

    pub fn definitions() -> Vec<Value> {
        vec![
            function_tool(
                "corpus_stats",
                "Get ingestion, compaction, and size statistics. Use this first to understand the corpus.",
                json!({
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": false
                }),
            ),
            function_tool(
                "list_sources",
                "List available log source names and sizes. An empty filter lists every source up to the limit.",
                json!({
                    "type": "object",
                    "properties": {
                        "filter": {
                            "type": "string",
                            "description": "Case-insensitive literal substring to match in source names, or an empty string"
                        },
                        "max_results": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": MAX_SOURCE_RESULTS
                        }
                    },
                    "required": ["filter", "max_results"],
                    "additionalProperties": false
                }),
            ),
            function_tool(
                "search_logs",
                "Search all retained log lines. Results group volatile duplicates and include representative source/line locations for follow-up reads.",
                json!({
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Literal text or Rust-compatible regular expression"
                        },
                        "regex": {
                            "type": "boolean",
                            "description": "Interpret pattern as a regular expression"
                        },
                        "case_sensitive": {
                            "type": "boolean"
                        },
                        "max_results": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": MAX_SEARCH_RESULTS
                        }
                    },
                    "required": ["pattern", "regex", "case_sensitive", "max_results"],
                    "additionalProperties": false
                }),
            ),
            function_tool(
                "read_lines",
                "Read bounded context from one source using original 1-based line numbers returned by search_logs.",
                json!({
                    "type": "object",
                    "properties": {
                        "source": {
                            "type": "string",
                            "description": "Exact source name returned by list_sources or search_logs"
                        },
                        "start_line": {
                            "type": "integer",
                            "minimum": 1
                        },
                        "line_count": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": MAX_READ_LINES
                        }
                    },
                    "required": ["source", "start_line", "line_count"],
                    "additionalProperties": false
                }),
            ),
            function_tool(
                "top_patterns",
                "Find the most repeated normalized log patterns. Timestamps, UUIDs, IP addresses, long IDs, and hex addresses are treated as volatile values.",
                json!({
                    "type": "object",
                    "properties": {
                        "min_occurrences": {
                            "type": "integer",
                            "minimum": 2
                        },
                        "max_results": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": MAX_PATTERN_RESULTS
                        }
                    },
                    "required": ["min_occurrences", "max_results"],
                    "additionalProperties": false
                }),
            ),
        ]
    }

    pub fn initial_summary(&self) -> String {
        serde_json::to_string(self.corpus.stats()).expect("corpus statistics are serializable")
    }

    pub fn execute(&self, name: &str, arguments: &str) -> String {
        let result = match name {
            "corpus_stats" => self.corpus_stats(arguments),
            "list_sources" => self.list_sources(arguments),
            "search_logs" => self.search_logs(arguments),
            "read_lines" => self.read_lines(arguments),
            "top_patterns" => self.top_patterns(arguments),
            _ => Err(anyhow!("unknown tool: {name}")),
        };
        match result {
            Ok(value) => self.bound_result(value),
            Err(error) => serde_json::to_string(&json!({ "error": error.to_string() }))
                .expect("error object is serializable"),
        }
    }

    fn corpus_stats(&self, arguments: &str) -> Result<Value> {
        let _: EmptyArgs = parse_args(arguments)?;
        Ok(json!({ "stats": self.corpus.stats() }))
    }

    fn list_sources(&self, arguments: &str) -> Result<Value> {
        let arguments: ListSourcesArgs = parse_args(arguments)?;
        let limit = bounded_usize(arguments.max_results, "max_results", MAX_SOURCE_RESULTS)?;
        let filter = arguments.filter.to_lowercase();
        let mut sources = Vec::new();
        let mut matching = 0_usize;
        for source in self.corpus.sources() {
            if !source.stats().name.to_lowercase().contains(&filter) {
                continue;
            }
            matching += 1;
            if sources.len() < limit {
                sources.push(json!(source.stats()));
            }
        }
        Ok(json!({
            "sources": sources,
            "matching_sources": matching,
            "truncated": matching > limit
        }))
    }

    fn search_logs(&self, arguments: &str) -> Result<Value> {
        let arguments: SearchArgs = parse_args(arguments)?;
        if arguments.pattern.is_empty() {
            bail!("pattern must not be empty");
        }
        if arguments.pattern.len() > MAX_PATTERN_BYTES {
            bail!("pattern exceeds the {MAX_PATTERN_BYTES} byte limit");
        }
        let limit = bounded_usize(arguments.max_results, "max_results", MAX_SEARCH_RESULTS)?;
        let expression = if arguments.regex {
            arguments.pattern
        } else {
            regex::escape(&arguments.pattern)
        };
        let matcher = RegexBuilder::new(&expression)
            .case_insensitive(!arguments.case_sensitive)
            .size_limit(2 * 1024 * 1024)
            .build()
            .map_err(|error| anyhow!("invalid search expression: {error}"))?;

        let mut groups: Vec<SearchGroup> = Vec::new();
        let mut group_indices: HashMap<String, usize> = HashMap::new();
        let mut omitted_patterns = HashSet::new();
        let mut scanned_entries = 0_u64;
        for source in self.corpus.sources() {
            for entry in source.entries() {
                scanned_entries += 1;
                if !matcher.is_match(&entry.text) {
                    continue;
                }
                let key = fingerprint(&entry.text);
                if let Some(index) = group_indices.get(&key).copied() {
                    groups[index].add_location(source.stats().name.clone(), entry);
                } else if groups.len() < limit {
                    let index = groups.len();
                    group_indices.insert(key.clone(), index);
                    groups.push(SearchGroup::new(key, source.stats().name.clone(), entry));
                } else {
                    omitted_patterns.insert(key);
                }
            }
        }

        Ok(json!({
            "matches": groups.into_iter().map(SearchGroup::into_json).collect::<Vec<_>>(),
            "scanned_entries": scanned_entries,
            "omitted_patterns": omitted_patterns.len(),
            "truncated": !omitted_patterns.is_empty()
        }))
    }

    fn read_lines(&self, arguments: &str) -> Result<Value> {
        let arguments: ReadLinesArgs = parse_args(arguments)?;
        if arguments.start_line == 0 {
            bail!("start_line must be at least 1");
        }
        if arguments.line_count == 0 || arguments.line_count > MAX_READ_LINES {
            bail!("line_count must be between 1 and {MAX_READ_LINES}");
        }
        let source = self
            .corpus
            .source(&arguments.source)
            .ok_or_else(|| anyhow!("unknown source: {}", arguments.source))?;
        let end_line = arguments
            .start_line
            .saturating_add(arguments.line_count - 1);
        let entries = source
            .entries()
            .iter()
            .filter(|entry| entry.line_end >= arguments.start_line && entry.line_start <= end_line)
            .map(|entry| {
                json!({
                    "line_start": entry.line_start,
                    "line_end": entry.line_end,
                    "repeats": entry.repeats,
                    "text": truncate_text(&entry.text)
                })
            })
            .collect::<Vec<_>>();
        Ok(json!({
            "source": source.stats().name,
            "requested_start_line": arguments.start_line,
            "requested_end_line": end_line,
            "entries": entries,
            "source_total_lines": source.stats().total_lines,
            "truncated": false
        }))
    }

    fn top_patterns(&self, arguments: &str) -> Result<Value> {
        let arguments: TopPatternsArgs = parse_args(arguments)?;
        if arguments.min_occurrences < 2 {
            bail!("min_occurrences must be at least 2");
        }
        let limit = bounded_usize(arguments.max_results, "max_results", MAX_PATTERN_RESULTS)?;
        let mut patterns: HashMap<String, PatternGroup> = HashMap::new();
        let mut sequence = 0_u64;
        for source in self.corpus.sources() {
            for entry in source.entries() {
                let key = fingerprint(&entry.text);
                let pattern = patterns.entry(key.clone()).or_insert_with(|| {
                    let first_seen = sequence;
                    sequence += 1;
                    PatternGroup {
                        fingerprint: key,
                        example: truncate_text(&entry.text),
                        occurrences: 0,
                        first_seen,
                    }
                });
                pattern.occurrences += entry.repeats;
            }
        }
        let mut patterns = patterns
            .into_values()
            .filter(|pattern| pattern.occurrences >= arguments.min_occurrences)
            .collect::<Vec<_>>();
        patterns.sort_by(|left, right| {
            right
                .occurrences
                .cmp(&left.occurrences)
                .then_with(|| left.first_seen.cmp(&right.first_seen))
        });
        let matching = patterns.len();
        patterns.truncate(limit);

        Ok(json!({
            "patterns": patterns.into_iter().map(PatternGroup::into_json).collect::<Vec<_>>(),
            "matching_patterns": matching,
            "truncated": matching > limit
        }))
    }

    fn bound_result(&self, mut value: Value) -> String {
        loop {
            let serialized = serde_json::to_string(&value).expect("tool result is serializable");
            if serialized.len() <= self.max_result_bytes {
                return serialized;
            }

            let Some(object) = value.as_object_mut() else {
                return oversized_error(self.max_result_bytes);
            };
            let array_name = ["matches", "entries", "sources", "patterns"]
                .into_iter()
                .find(|name| object.get(*name).is_some_and(Value::is_array));
            let Some(array_name) = array_name else {
                return oversized_error(self.max_result_bytes);
            };
            let Some(array) = object.get_mut(array_name).and_then(Value::as_array_mut) else {
                return oversized_error(self.max_result_bytes);
            };
            if array.pop().is_none() {
                return oversized_error(self.max_result_bytes);
            }
            object.insert("truncated".to_owned(), Value::Bool(true));
            object.insert(
                "truncation_reason".to_owned(),
                Value::String("tool byte limit".to_owned()),
            );
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EmptyArgs {}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ListSourcesArgs {
    filter: String,
    max_results: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SearchArgs {
    pattern: String,
    regex: bool,
    case_sensitive: bool,
    max_results: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReadLinesArgs {
    source: String,
    start_line: u64,
    line_count: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TopPatternsArgs {
    min_occurrences: u64,
    max_results: u64,
}

struct SearchGroup {
    fingerprint: String,
    example: String,
    occurrences: u64,
    locations: Vec<Value>,
}

impl SearchGroup {
    fn new(fingerprint: String, source: String, entry: &LogEntry) -> Self {
        let mut group = Self {
            fingerprint,
            example: truncate_text(&entry.text),
            occurrences: 0,
            locations: Vec::new(),
        };
        group.add_location(source, entry);
        group
    }

    fn add_location(&mut self, source: String, entry: &LogEntry) {
        self.occurrences += entry.repeats;
        if self.locations.len() < 3 {
            self.locations.push(json!({
                "source": source,
                "line_start": entry.line_start,
                "line_end": entry.line_end,
                "repeats": entry.repeats
            }));
        }
    }

    fn into_json(self) -> Value {
        json!({
            "pattern": self.fingerprint,
            "example": self.example,
            "occurrences": self.occurrences,
            "sample_locations": self.locations
        })
    }
}

struct PatternGroup {
    fingerprint: String,
    example: String,
    occurrences: u64,
    first_seen: u64,
}

impl PatternGroup {
    fn into_json(self) -> Value {
        json!({
            "pattern": self.fingerprint,
            "example": self.example,
            "occurrences": self.occurrences
        })
    }
}

fn function_tool(name: &str, description: &str, parameters: Value) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters
        }
    })
}

fn parse_args<T: for<'de> Deserialize<'de>>(arguments: &str) -> Result<T> {
    serde_json::from_str(arguments).map_err(|error| anyhow!("invalid tool arguments: {error}"))
}

fn bounded_usize(value: u64, name: &str, maximum: usize) -> Result<usize> {
    if value == 0 || value > maximum as u64 {
        bail!("{name} must be between 1 and {maximum}");
    }
    Ok(value as usize)
}

fn truncate_text(text: &str) -> String {
    if text.len() <= MAX_TEXT_BYTES {
        return text.to_owned();
    }
    let mut end = MAX_TEXT_BYTES;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{} …[excerpt truncated]", &text[..end])
}

fn oversized_error(limit: usize) -> String {
    serde_json::to_string(&json!({
        "error": format!("tool result cannot fit within the configured {limit} byte limit")
    }))
    .expect("error object is serializable")
}

#[cfg(test)]
mod tests {
    use std::{io::Write, path::PathBuf};

    use super::*;
    use crate::corpus::IngestOptions;

    struct TestFile {
        path: PathBuf,
    }

    impl TestFile {
        fn new(contents: &str) -> Self {
            let test_name = std::thread::current()
                .name()
                .unwrap_or("test")
                .replace("::", "-");
            let path = std::env::temp_dir().join(format!(
                "logshard-tools-{}-{}.log",
                std::process::id(),
                test_name
            ));
            let mut file = std::fs::File::create(&path).unwrap();
            file.write_all(contents.as_bytes()).unwrap();
            Self { path }
        }
    }

    impl Drop for TestFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    fn corpus(contents: &str) -> (TestFile, Corpus) {
        let file = TestFile::new(contents);
        let corpus = Corpus::ingest(
            std::slice::from_ref(&file.path),
            IngestOptions {
                max_input_bytes: 1_000_000,
                max_line_bytes: 4096,
                redact_secrets: true,
            },
        )
        .unwrap();
        (file, corpus)
    }

    #[test]
    fn search_groups_volatile_duplicates_and_keeps_locations() {
        let (_file, corpus) = corpus(
            "2026-08-10T00:00:00Z ERROR request 123456 failed\ncontext\n2026-08-10T00:00:01Z ERROR request 789012 failed\n",
        );
        let registry = ToolRegistry::new(corpus.into(), 64 * 1024);
        let output: Value = serde_json::from_str(&registry.execute(
            "search_logs",
            r#"{"pattern":"ERROR","regex":false,"case_sensitive":true,"max_results":10}"#,
        ))
        .unwrap();
        assert_eq!(output["matches"].as_array().unwrap().len(), 1);
        assert_eq!(output["matches"][0]["occurrences"], 2);
        assert_eq!(
            output["matches"][0]["sample_locations"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
    }

    #[test]
    fn read_lines_uses_original_ranges_after_compaction() {
        let (file, corpus) = corpus("same\nsame\nnext\n");
        let source = file.path.display().to_string();
        let registry = ToolRegistry::new(corpus.into(), 64 * 1024);
        let output: Value = serde_json::from_str(&registry.execute(
            "read_lines",
            &json!({"source": source, "start_line": 1, "line_count": 2}).to_string(),
        ))
        .unwrap();
        assert_eq!(output["entries"].as_array().unwrap().len(), 1);
        assert_eq!(output["entries"][0]["repeats"], 2);
    }

    #[test]
    fn invalid_regex_is_a_recoverable_tool_error() {
        let (_file, corpus) = corpus("hello\n");
        let registry = ToolRegistry::new(corpus.into(), 64 * 1024);
        let output: Value = serde_json::from_str(&registry.execute(
            "search_logs",
            r#"{"pattern":"(","regex":true,"case_sensitive":true,"max_results":10}"#,
        ))
        .unwrap();
        assert!(output["error"].as_str().unwrap().contains("invalid search"));
    }

    #[test]
    fn tool_results_obey_byte_limit() {
        let (_file, corpus) = corpus(&format!(
            "alpha {}\nbeta {}\n",
            "x".repeat(3000),
            "y".repeat(3000)
        ));
        let registry = ToolRegistry::new(corpus.into(), 1024);
        let output = registry.execute(
            "search_logs",
            r#"{"pattern":"alpha|beta","regex":true,"case_sensitive":true,"max_results":10}"#,
        );
        assert!(output.len() <= 1024);
        let parsed: Value = serde_json::from_str(&output).unwrap();
        assert_eq!(parsed["truncated"], true);
    }
}
