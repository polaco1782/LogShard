use serde::Deserialize;
use serde_json::Value;

use super::{FunctionCall, ProviderDefaults, ToolCall};

pub(super) fn defaults() -> ProviderDefaults {
    ProviderDefaults {
        model: "local",
        base_url: Some("http://127.0.0.1:8080/v1"),
        api_key_required: false,
    }
}

pub(super) fn content_tool_call(
    content: &str,
    tools: &[Value],
    message_count: usize,
) -> Option<ToolCall> {
    let encoded: ContentToolCall = serde_json::from_str(content).ok()?;
    if !encoded.params.is_object()
        || !tools.iter().any(|tool| {
            tool.pointer("/function/name").and_then(Value::as_str)
                == Some(encoded.tool_name.as_str())
        })
    {
        return None;
    }

    Some(ToolCall {
        id: format!("content-tool-call-{message_count}"),
        kind: "function".to_owned(),
        function: FunctionCall {
            name: encoded.tool_name,
            arguments: serde_json::to_string(&encoded.params).ok()?,
        },
    })
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ContentToolCall {
    tool_name: String,
    params: Value,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::content_tool_call;

    #[test]
    fn parses_the_observed_content_encoded_tool_request() {
        let tools = vec![json!({ "function": { "name": "search_logs" } })];
        let call = content_tool_call(
            r#"{"tool_name":"search_logs","params":{"pattern":"error","regex":false}}"#,
            &tools,
            2,
        )
        .unwrap();

        assert_eq!(call.id, "content-tool-call-2");
        assert_eq!(call.function.name, "search_logs");
        assert_eq!(
            call.function.arguments,
            r#"{"pattern":"error","regex":false}"#
        );
    }

    #[test]
    fn does_not_reinterpret_unknown_or_non_exact_json_as_a_tool_request() {
        let tools = vec![json!({ "function": { "name": "search_logs" } })];

        assert!(content_tool_call(r#"{"tool_name":"unknown","params":{}}"#, &tools, 2).is_none());
        assert!(
            content_tool_call(
                r#"{"tool_name":"search_logs","params":{},"result":"final"}"#,
                &tools,
                2
            )
            .is_none()
        );
    }
}
