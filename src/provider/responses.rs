//! Streaming, stateless Responses transport for explicitly configured reverse adapters.
use super::{ChatMessage, FunctionCall, ModelReply, ProviderConfig, ToolCall};
use crate::debug::ActivityLog;
use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashSet};

const RAW_OUTPUT: &str = "__logshard_responses_output";
const NAMESPACE: &str = "logshard";
const MAX_RESPONSE_BYTES: usize = 16 * 1024 * 1024;

fn request(config: &ProviderConfig, messages: &[ChatMessage], tools: &[Value]) -> Result<Value> {
    let mut instructions = Vec::new();
    let mut input = Vec::new();
    for message in messages {
        match message.role.as_str() {
            "system" | "developer" => instructions.push(message.content.clone().unwrap_or_default()),
            "tool" => input.push(json!({"type":"function_call_output", "call_id": message.tool_call_id.as_ref().context("missing tool call ID")?, "output": message.content.as_deref().unwrap_or("")})),
            "assistant" if message.extra.contains_key(RAW_OUTPUT) => {
                input.extend(message.extra[RAW_OUTPUT].as_array().context("invalid saved Responses output")?.iter().cloned());
            }
            "user" | "assistant" => input.push(json!({"role": message.role, "content": message.content.as_deref().unwrap_or("")})),
            _ => bail!("unsupported message role for Responses transport"),
        }
    }
    let functions: Vec<Value> = tools
        .iter()
        .map(|tool| {
            let mut function = tool
                .get("function")
                .cloned()
                .context("invalid local tool schema")?;
            function
                .as_object_mut()
                .context("invalid local tool schema")?
                .insert("type".into(), json!("function"));
            Ok(function)
        })
        .collect::<Result<_>>()?;
    Ok(
        json!({"model": config.model, "instructions": instructions.join("\n\n"), "input": input,
        "store": false, "stream": true, "include": ["reasoning.encrypted_content"],
        "tools": [{"type":"namespace", "name":NAMESPACE, "description":"Bounded read-only local log analysis tools.", "tools":functions}]}),
    )
}

pub(super) async fn complete(
    client: &reqwest::Client,
    config: &ProviderConfig,
    messages: &[ChatMessage],
    tools: &[Value],
    activity: &ActivityLog,
) -> Result<ModelReply> {
    let body = request(config, messages, tools)?;
    activity.emit("provider.request", json!({"transport":"responses", "model":config.model, "message_count":messages.len(), "tool_count":tools.len()}));
    let mut builder = client
        .post(
            config
                .endpoint
                .clone()
                .context("missing Responses endpoint")?,
        )
        .header("Accept", "text/event-stream")
        .json(&body);
    if let Some(key) = &config.api_key {
        builder = builder.bearer_auth(key);
    }
    let mut response = builder
        .send()
        .await
        .context("Responses provider request failed")?;
    if !response.status().is_success() {
        bail!("Responses provider returned HTTP {}", response.status());
    }
    if !response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| {
            v.split(';')
                .next()
                .is_some_and(|v| v.trim() == "text/event-stream")
        })
    {
        bail!("Responses provider did not return an event stream");
    }
    let mut stream = EventStream::default();
    while let Some(bytes) = response
        .chunk()
        .await
        .context("failed to read Responses event stream")?
    {
        stream.feed(&bytes)?;
        if stream.completed.is_some() {
            break;
        }
    }
    let output = stream.finish()?;
    let mut message = ChatMessage::text("assistant", "");
    let mut text = String::new();
    let mut calls = Vec::new();
    let mut ids = HashSet::new();
    for item in &output {
        match item.get("type").and_then(Value::as_str) {
            Some("reasoning") => {}
            Some("message") => {
                ensure_completed_item(item)?;
                for part in item
                    .get("content")
                    .and_then(Value::as_array)
                    .context("invalid Responses message content")?
                {
                    match part.get("type").and_then(Value::as_str) {
                        Some("output_text") => text.push_str(
                            part.get("text")
                                .and_then(Value::as_str)
                                .context("invalid output text")?,
                        ),
                        Some("refusal") => bail!("Responses provider refused the request"),
                        _ => bail!("unsupported Responses message content"),
                    }
                }
            }
            Some("function_call") => {
                ensure_completed_item(item)?;
                if item
                    .get("namespace")
                    .is_some_and(|v| v.as_str() != Some(NAMESPACE))
                {
                    bail!("Responses provider returned an unexpected tool namespace");
                }
                let name = item
                    .get("name")
                    .and_then(Value::as_str)
                    .context("missing Responses tool name")?;
                if !tools
                    .iter()
                    .any(|t| t.pointer("/function/name").and_then(Value::as_str) == Some(name))
                {
                    bail!("Responses provider requested an unauthorized tool");
                }
                let id = item
                    .get("call_id")
                    .and_then(Value::as_str)
                    .filter(|v| !v.is_empty())
                    .context("missing Responses call ID")?;
                if !ids.insert(id.to_owned()) {
                    bail!("duplicate Responses call ID");
                }
                let arguments = item
                    .get("arguments")
                    .and_then(Value::as_str)
                    .context("invalid Responses arguments")?;
                if !serde_json::from_str::<Value>(arguments)
                    .context("invalid Responses arguments JSON")?
                    .is_object()
                {
                    bail!("Responses tool arguments must be an object");
                }
                calls.push(ToolCall {
                    id: id.into(),
                    kind: "function".into(),
                    function: FunctionCall {
                        name: name.into(),
                        arguments: arguments.into(),
                    },
                });
            }
            _ => bail!("unsupported Responses output item"),
        }
    }
    message.content = Some(text);
    let has_calls = !calls.is_empty();
    if has_calls {
        message.tool_calls = Some(calls);
    }
    message.extra.insert(RAW_OUTPUT.into(), json!(output));
    activity.emit("provider.response", json!({"transport":"responses", "tool_calls": message.tool_calls.as_ref().map_or(0, Vec::len), "text_bytes":message.content.as_ref().map_or(0,String::len)}));
    Ok(ModelReply {
        message,
        finish_reason: Some(if has_calls { "tool_calls" } else { "stop" }.into()),
    })
}

fn ensure_completed_item(item: &Value) -> Result<()> {
    if item
        .get("status")
        .is_some_and(|v| v.as_str() != Some("completed"))
    {
        bail!("Responses output item was not completed");
    }
    Ok(())
}
#[derive(Default)]
struct EventStream {
    pending: Vec<u8>,
    data: String,
    total: usize,
    items: BTreeMap<u64, (Value, bool)>,
    completed: Option<Value>,
}

fn same_item(a: &Value, b: &Value) -> bool {
    a.get("type") == b.get("type")
        && ["id", "call_id"]
            .iter()
            .all(|key| match (a.get(key), b.get(key)) {
                (Some(a), Some(b)) => a == b,
                _ => true,
            })
}

impl EventStream {
    fn feed(&mut self, bytes: &[u8]) -> Result<()> {
        self.total = self
            .total
            .checked_add(bytes.len())
            .context("Responses stream too large")?;
        if self.total > MAX_RESPONSE_BYTES {
            bail!("Responses stream exceeded 16 MiB");
        }
        self.pending.extend_from_slice(bytes);
        while let Some(end) = self.pending.iter().position(|b| *b == b'\n') {
            let line: Vec<_> = self.pending.drain(..=end).collect();
            let line = std::str::from_utf8(&line)
                .context("invalid UTF-8 Responses event")?
                .trim_end_matches(['\r', '\n']);
            if line.is_empty() {
                self.dispatch()?;
            } else if let Some(data) = line.strip_prefix("data:") {
                if !self.data.is_empty() {
                    self.data.push('\n');
                }
                self.data.push_str(data.strip_prefix(' ').unwrap_or(data));
            }
        }
        Ok(())
    }
    fn dispatch(&mut self) -> Result<()> {
        if self.data.is_empty() {
            return Ok(());
        }
        let data = std::mem::take(&mut self.data);
        if data == "[DONE]" {
            return Ok(());
        }
        let event: Value = serde_json::from_str(&data).context("invalid Responses event JSON")?;
        let kind = event
            .get("type")
            .and_then(Value::as_str)
            .context("missing Responses event type")?;
        match kind {
            "response.output_item.added" | "response.output_item.done" => {
                let finalized = kind == "response.output_item.done";
                let item = event.get("item").cloned().unwrap_or(Value::Null);
                if item.is_null() && !finalized {
                    return Ok(());
                }
                if !item.is_object() {
                    bail!("invalid streamed output item");
                }
                let index = event
                    .get("output_index")
                    .and_then(Value::as_u64)
                    .context("invalid streamed output index")?;
                if self
                    .items
                    .get(&index)
                    .is_some_and(|(previous, _)| !same_item(previous, &item))
                {
                    bail!("streamed output item identity changed");
                }
                if finalized || !self.items.contains_key(&index) {
                    self.items.insert(index, (item, finalized));
                }
            }
            "response.completed" => {
                let response = event
                    .get("response")
                    .filter(|v| v.is_object())
                    .context("invalid completed Responses event")?;
                if response
                    .get("status")
                    .is_some_and(|s| s.as_str() != Some("completed"))
                {
                    bail!("Responses provider did not complete successfully");
                }
                self.completed = Some(response.clone());
            }
            "response.failed" | "response.incomplete" | "error" => {
                bail!("Responses provider failed or returned an incomplete response")
            }
            _ => {}
        }
        Ok(())
    }
    fn finish(self) -> Result<Vec<Value>> {
        let response = self
            .completed
            .context("Responses stream ended without response.completed")?;
        let terminal = response.get("output");
        if terminal.is_some_and(|v| !v.is_null() && !v.is_array()) {
            bail!("invalid completed Responses output");
        }
        let output = terminal.and_then(Value::as_array).filter(|v| !v.is_empty());
        for (item, done) in self.items.values() {
            if !done && !output.is_some_and(|values| values.iter().any(|v| same_item(item, v))) {
                bail!("Responses stream contains unfinished output items");
            }
        }
        Ok(output.cloned().unwrap_or_else(|| {
            self.items
                .into_values()
                .filter(|(_, done)| *done)
                .map(|(item, _)| item)
                .collect()
        }))
    }
}
