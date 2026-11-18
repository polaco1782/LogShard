use std::net::{IpAddr, Ipv4Addr, SocketAddr};

use logshard::{
    agent::{AnswerFormat, investigate},
    corpus::{Corpus, IngestOptions},
    debug::ActivityLog,
    provider::{ChatMessage, OpenAiCompatibleClient, ProviderConfig, ProviderKind},
    tools::ToolRegistry,
};
use serde_json::{Value, json};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    sync::mpsc,
};

fn function_call(namespace: &str) -> Value {
    json!({
        "type": "function_call", "id": "fc_1", "call_id": "call_1",
        "namespace": namespace, "name": "search_logs",
        "arguments": r#"{"pattern":"FATAL","regex":false,"case_sensitive":false,"max_results":5}"#,
        "status": "completed"
    })
}

fn final_message() -> Value {
    json!({
        "type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
        "content": [{"type": "output_text", "text": "{\"summary\":\"Cache failure at the retrieved line.\",\"evidence\":[\"synthetic.log:2\"]}"}]
    })
}

fn sse(events: Vec<Value>) -> String {
    events
        .into_iter()
        .map(|event| format!("data: {event}\r\n\r\n"))
        .collect()
}

fn completed(output: Value) -> Value {
    json!({"type":"response.completed", "response":{"id":"resp_1", "status":"completed", "output":output}})
}

async fn client_for(
    replies: Vec<(&'static str, String)>,
    responses: bool,
) -> (OpenAiCompatibleClient, mpsc::Receiver<(String, Value)>) {
    let listener = TcpListener::bind(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0))
        .await
        .unwrap();
    let address = listener.local_addr().unwrap();
    let (sender, receiver) = mpsc::channel(replies.len());
    tokio::spawn(async move {
        for (content_type, body) in replies {
            let (mut stream, _) = listener.accept().await.unwrap();
            let request = read_request(&mut stream).await;
            sender.send(request).await.unwrap();
            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            stream.write_all(header.as_bytes()).await.unwrap();
            // Fragment across event, CRLF and UTF-8 boundaries like a real stream.
            for fragment in body.as_bytes().chunks(7) {
                stream.write_all(fragment).await.unwrap();
            }
        }
    });
    let config = ProviderConfig::new(
        ProviderKind::Openai,
        "mock-plan-model".into(),
        Some(format!("http://{address}/v1")),
        Some("public-test-key".into()),
        5,
        1024,
    )
    .unwrap()
    .with_responses_api(responses)
    .unwrap();
    (OpenAiCompatibleClient::new(config).unwrap(), receiver)
}

async fn read_request(stream: &mut TcpStream) -> (String, Value) {
    let mut request = Vec::new();
    let header_end = loop {
        let mut buffer = [0_u8; 4096];
        let read = stream.read(&mut buffer).await.unwrap();
        assert!(read > 0);
        request.extend_from_slice(&buffer[..read]);
        if let Some(position) = request.windows(4).position(|window| window == b"\r\n\r\n") {
            break position + 4;
        }
    };
    let headers = String::from_utf8_lossy(&request[..header_end]);
    let target = headers
        .lines()
        .next()
        .unwrap()
        .split_whitespace()
        .nth(1)
        .unwrap()
        .to_owned();
    let content_length = headers
        .lines()
        .find_map(|line| {
            line.split_once(':').and_then(|(name, value)| {
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().unwrap())
            })
        })
        .unwrap();
    while request.len() - header_end < content_length {
        let mut buffer = [0_u8; 4096];
        let read = stream.read(&mut buffer).await.unwrap();
        assert!(read > 0);
        request.extend_from_slice(&buffer[..read]);
    }
    (
        target,
        serde_json::from_slice(&request[header_end..header_end + content_length]).unwrap(),
    )
}

#[tokio::test]
async fn responses_agent_retrieves_bounded_evidence_and_replays_encrypted_reasoning() {
    let reasoning = json!({"type":"reasoning", "id":"rs_1", "encrypted_content":"opaque-test-reasoning", "summary":[]});
    let call = function_call("logshard");
    let first = sse(vec![
        json!({"type":"response.output_item.done", "output_index":0, "item":reasoning}),
        json!({"type":"response.output_item.done", "output_index":1, "item":call}),
        completed(json!([])),
    ]);
    let second = sse(vec![completed(json!([final_message()]))]);
    let (client, mut requests) = client_for(
        vec![("text/event-stream", first), ("text/event-stream", second)],
        true,
    )
    .await;
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("synthetic.log");
    std::fs::write(
        &path,
        "INFO startup\nFATAL sentinel-cache-miss\nINFO idle\n",
    )
    .unwrap();
    let corpus = Corpus::ingest(
        &[path],
        IngestOptions {
            max_input_bytes: 4096,
            max_line_bytes: 1024,
            redact_secrets: true,
        },
    )
    .unwrap();
    let registry = ToolRegistry::new(corpus.into(), 4096);
    let result = investigate(
        &client,
        &registry,
        "Find the failure",
        AnswerFormat::Json,
        4,
        4096,
        &ActivityLog::disabled(),
    )
    .await
    .unwrap();
    assert_eq!(result.steps, 2);
    assert_eq!(result.tool_calls, 1);
    assert_eq!(
        serde_json::from_str::<Value>(&result.answer).unwrap()["summary"],
        "Cache failure at the retrieved line."
    );
    let (first_target, first_request) = requests.recv().await.unwrap();
    let (second_target, second_request) = requests.recv().await.unwrap();
    for (target, request) in [
        (&first_target, &first_request),
        (&second_target, &second_request),
    ] {
        assert_eq!(target, "/v1/responses");
        assert_eq!(request["store"], false);
        assert_eq!(request["stream"], true);
        assert_eq!(request["include"], json!(["reasoning.encrypted_content"]));
        assert_eq!(request["tools"][0]["type"], "namespace");
        assert_eq!(request["tools"][0]["name"], "logshard");
        let definitions = request["tools"][0]["tools"].as_array().unwrap();
        assert!(
            definitions
                .iter()
                .any(|tool| tool["type"] == "function" && tool["name"] == "search_logs")
        );
        for forbidden in [
            "messages",
            "max_completion_tokens",
            "max_tokens",
            "parallel_tool_calls",
            "previous_response_id",
        ] {
            assert!(request.get(forbidden).is_none(), "unexpected {forbidden}");
        }
    }
    assert!(!first_request.to_string().contains("sentinel-cache-miss"));
    let input = second_request["input"].as_array().unwrap();
    assert!(input.contains(&reasoning));
    assert!(input.contains(&call));
    let output = input
        .iter()
        .find(|item| item["type"] == "function_call_output")
        .unwrap();
    assert_eq!(output["call_id"], "call_1");
    assert!(
        output["output"]
            .as_str()
            .unwrap()
            .contains("sentinel-cache-miss")
    );
    assert!(output["output"].as_str().unwrap().len() <= 4096);
}

async fn complete_events(events: Vec<Value>) -> anyhow::Result<logshard::provider::ModelReply> {
    let (client, _requests) = client_for(vec![("text/event-stream", sse(events))], true).await;
    client
        .complete(
            &[ChatMessage::text("user", "test")],
            &ToolRegistry::definitions(),
            &ActivityLog::disabled(),
        )
        .await
}

#[tokio::test]
async fn missing_completion_is_not_a_successful_tool_reply() {
    let result = complete_events(vec![json!({"type":"response.output_item.done", "output_index":0, "item":function_call("logshard")})]).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn unfinished_output_item_is_rejected() {
    let result = complete_events(vec![json!({"type":"response.output_item.added", "output_index":0, "item":function_call("logshard")}), completed(json!([]))]).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn foreign_tool_namespace_is_rejected() {
    assert!(
        complete_events(vec![completed(json!([function_call("foreign")]))])
            .await
            .is_err()
    );
}

#[tokio::test]
async fn nonempty_terminal_output_is_authoritative_and_not_duplicated() {
    let call = function_call("logshard");
    let reply = complete_events(vec![
        json!({"type":"response.output_item.done", "output_index":0, "item":call}),
        completed(json!([call])),
    ])
    .await
    .unwrap();
    assert_eq!(reply.message.tool_calls.unwrap().len(), 1);
    let reply = complete_events(vec![json!({"type":"response.output_item.done", "output_index":0, "item":function_call("logshard")}), completed(json!([final_message()]))]).await.unwrap();
    assert!(reply.message.tool_calls.unwrap_or_default().is_empty());
    assert!(reply.message.content.unwrap().contains("Cache failure"));
}

#[tokio::test]
async fn ordinary_openai_keeps_chat_completions_contract() {
    let response = json!({"choices":[{"message":{"role":"assistant","content":"plain API reply"},"finish_reason":"stop"}]}).to_string();
    let (client, mut requests) = client_for(vec![("application/json", response)], false).await;
    let reply = client
        .complete(
            &[ChatMessage::text("user", "test")],
            &ToolRegistry::definitions(),
            &ActivityLog::disabled(),
        )
        .await
        .unwrap();
    assert_eq!(reply.message.content.as_deref(), Some("plain API reply"));
    let (target, request) = requests.recv().await.unwrap();
    assert_eq!(target, "/v1/chat/completions");
    assert!(request["messages"].is_array());
    assert_eq!(request["max_completion_tokens"], 1024);
    assert_eq!(request["parallel_tool_calls"], false);
    assert_eq!(request["tool_choice"], "auto");
    assert_eq!(request["tools"][0]["type"], "function");
    assert!(request.get("input").is_none());
}

#[tokio::test]
async fn failed_and_incomplete_streams_are_rejected() {
    for kind in ["response.failed", "response.incomplete", "error"] {
        assert!(
            complete_events(vec![
                json!({"type":kind}),
                completed(json!([final_message()]))
            ])
            .await
            .is_err()
        );
    }
}

#[tokio::test]
async fn unauthorized_or_malformed_tool_calls_are_rejected() {
    let mut unknown = function_call("logshard");
    unknown["name"] = json!("execute_shell");
    assert!(
        complete_events(vec![completed(json!([unknown]))])
            .await
            .is_err()
    );
    let mut invalid = function_call("logshard");
    invalid["arguments"] = json!("not JSON");
    assert!(
        complete_events(vec![completed(json!([invalid]))])
            .await
            .is_err()
    );
    let call = function_call("logshard");
    assert!(
        complete_events(vec![completed(json!([call, call]))])
            .await
            .is_err()
    );
}

#[tokio::test]
async fn explicitly_unfinished_terminal_items_are_rejected() {
    for status in ["in_progress", "incomplete"] {
        let mut call = function_call("logshard");
        call["status"] = json!(status);
        assert!(
            complete_events(vec![completed(json!([call]))])
                .await
                .is_err()
        );
        let mut message = final_message();
        message["status"] = json!(status);
        assert!(
            complete_events(vec![completed(json!([message]))])
                .await
                .is_err()
        );
    }
}

#[test]
fn responses_configuration_is_explicit_and_openai_only() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("config.toml");
    let config: logshard::config::Config = toml::from_str(
        r#"
[provider]
type = "openai"
model = "test-model"
api_key = "public-test-key"
base_url = "http://localhost/v1"
responses_api = true
"#,
    )
    .unwrap();
    config.save_to(&path).unwrap();
    let loaded = logshard::config::Config::load_from(&path).unwrap();
    assert!(loaded.provider.responses_api);
    assert_eq!(
        loaded.provider_config().unwrap().endpoint.unwrap().path(),
        "/v1/responses"
    );
    let other = ProviderConfig::new(
        ProviderKind::LlamaCpp,
        "mock".into(),
        Some("http://localhost/v1".into()),
        None,
        5,
        1024,
    )
    .unwrap();
    assert!(other.with_responses_api(true).is_err());
}
