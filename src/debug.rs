use std::{
    io::{self, Write},
    time::{SystemTime, UNIX_EPOCH},
};

use serde::Serialize;
use serde_json::{Value, json};

#[derive(Clone, Debug)]
pub struct ActivityLog {
    enabled: bool,
    api_key: Option<String>,
    max_text_bytes: usize,
}

impl ActivityLog {
    pub fn new(enabled: bool, api_key: Option<&str>, max_text_bytes: usize) -> Self {
        Self {
            enabled,
            api_key: api_key.map(str::to_owned),
            max_text_bytes: max_text_bytes.max(64),
        }
    }

    pub fn disabled() -> Self {
        Self::new(false, None, 2048)
    }

    pub fn emit(&self, event: &str, data: impl Serialize) {
        if !self.enabled {
            return;
        }
        let line = self.format_event(event, data);
        let mut stderr = io::stderr().lock();
        let _ = writeln!(stderr, "{line}\n");
    }

    fn format_event(&self, event: &str, data: impl Serialize) -> String {
        let data = serde_json::to_value(data).unwrap_or_else(|error| {
            json!({
                "debug_serialization_error": error.to_string()
            })
        });
        let mut envelope = json!({
            "timestamp_ms": SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_millis())
                .unwrap_or_default(),
            "event": event,
            "data": data
        });
        if let Some(api_key) = &self.api_key {
            redact_secret(&mut envelope, api_key);
        }
        truncate_strings(&mut envelope, self.max_text_bytes);
        serde_json::to_string_pretty(&envelope).expect("debug event envelope is serializable")
    }
}

fn redact_secret(value: &mut Value, secret: &str) {
    if secret.is_empty() {
        return;
    }
    match value {
        Value::String(text) => {
            if text.contains(secret) {
                *text = text.replace(secret, "<redacted-api-key>");
            }
        }
        Value::Array(values) => {
            for value in values {
                redact_secret(value, secret);
            }
        }
        Value::Object(values) => {
            for value in values.values_mut() {
                redact_secret(value, secret);
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) => {}
    }
}

fn truncate_strings(value: &mut Value, max_bytes: usize) {
    match value {
        Value::String(text) => {
            if text.len() > max_bytes {
                let original_bytes = text.len();
                let marker = format!(" …[truncated; {original_bytes} bytes total]");
                let content_limit = max_bytes.saturating_sub(marker.len());
                let mut end = content_limit.min(text.len());
                while !text.is_char_boundary(end) {
                    end -= 1;
                }
                text.truncate(end);
                text.push_str(&marker);
            }
        }
        Value::Array(values) => {
            for value in values {
                truncate_strings(value, max_bytes);
            }
        }
        Value::Object(values) => {
            for value in values.values_mut() {
                truncate_strings(value, max_bytes);
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_events_are_json_and_redact_the_api_key_recursively() {
        let log = ActivityLog::new(true, Some("sk-test-secret"), 128);
        let line = log.format_event(
            "provider.response",
            json!({
                "body": {
                    "message": "echo sk-test-secret here",
                    "nested": ["sk-test-secret"],
                    "long": "x".repeat(256)
                }
            }),
        );

        assert!(!line.contains("sk-test-secret"));
        assert!(line.contains('\n'));
        let event: Value = serde_json::from_str(&line).unwrap();
        assert_eq!(event["event"], "provider.response");
        assert_eq!(event["data"]["body"]["nested"][0], "<redacted-api-key>");
        let long = event["data"]["body"]["long"].as_str().unwrap();
        assert!(long.len() <= 128);
        assert!(long.contains("truncated"));
    }
}
