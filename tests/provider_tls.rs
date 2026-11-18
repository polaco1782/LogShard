//! All credentials and certificates in fixtures are public test-only material.
use logshard::debug::ActivityLog;
use logshard::provider::{OpenAiCompatibleClient, ProviderConfig, ProviderKind};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, pem::PemObject};
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio_rustls::{TlsAcceptor, rustls};

const CA: &str = include_str!("fixtures/test-ca.pem");

async fn request(host: &str, ca: Option<&str>, redirect: bool) -> anyhow::Result<()> {
    let certificates = CertificateDer::pem_slice_iter(include_bytes!("fixtures/test-server.pem"))
        .collect::<Result<Vec<_>, _>>()?;
    let key = PrivateKeyDer::from_pem_slice(include_bytes!("fixtures/test-server.key"))?;
    let config = rustls::ServerConfig::builder_with_provider(Arc::new(
        rustls::crypto::ring::default_provider(),
    ))
    .with_safe_default_protocol_versions()?
    .with_no_client_auth()
    .with_single_cert(certificates, key)?;
    let acceptor = TlsAcceptor::from(Arc::new(config));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let port = listener.local_addr()?.port();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        if let Ok(mut stream) = acceptor.accept(stream).await {
            let mut buf = [0; 8192];
            let _ = stream.read(&mut buf).await;
            let body = r#"{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}"#;
            let response = if redirect {
                format!(
                    "HTTP/1.1 302 Found\r\nLocation: https://localhost:{port}/elsewhere\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                )
            } else {
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                )
            };
            let _ = stream.write_all(response.as_bytes()).await;
        }
    });
    let config = ProviderConfig::new(
        ProviderKind::Openai,
        "test".into(),
        Some(format!("https://{host}:{port}/v1")),
        Some("public-test-key".into()),
        3,
        64,
    )?
    .with_ca_cert_pem(ca.map(str::to_owned))?;
    let result = OpenAiCompatibleClient::new(config)?
        .complete(&[], &[], &ActivityLog::disabled())
        .await;
    server.abort();
    result.map(|reply| assert_eq!(reply.message.content.as_deref(), Some("ok")))
}

#[tokio::test]
async fn custom_ca_accepts_valid_server() {
    request("localhost", Some(CA), false).await.unwrap();
}

#[tokio::test]
async fn default_trust_rejects_private_ca() {
    assert!(request("localhost", None, false).await.is_err());
}

#[tokio::test]
async fn custom_ca_still_checks_hostname() {
    assert!(request("127.0.0.1", Some(CA), false).await.is_err());
}

#[tokio::test]
async fn redirects_are_not_followed() {
    let error = request("localhost", Some(CA), true).await.unwrap_err();
    assert!(error.to_string().contains("302"), "{error:#}");
}

#[test]
fn invalid_ca_configuration_fails_closed() {
    for pem in [
        "",
        "not a certificate",
        "-----BEGIN PRIVATE KEY-----\nanything",
        "-----BEGIN CERTIFICATE-----\nbroken\n-----END CERTIFICATE-----",
    ] {
        let config = ProviderConfig::new(
            ProviderKind::Openai,
            "test".into(),
            Some("https://localhost/v1".into()),
            Some("test".into()),
            1,
            64,
        )
        .unwrap();
        assert!(config.with_ca_cert_pem(Some(pem.into())).is_err());
    }
    for (kind, url, key) in [
        (ProviderKind::GithubCopilot, None, None),
        (
            ProviderKind::LlamaCpp,
            Some("http://localhost/v1".into()),
            None,
        ),
    ] {
        let config = ProviderConfig::new(kind, "test".into(), url, key, 1, 64).unwrap();
        assert!(config.with_ca_cert_pem(Some(CA.into())).is_err());
    }
}

#[tokio::test]
async fn unrelated_custom_ca_rejects_server() {
    assert!(
        request(
            "localhost",
            Some(include_str!("fixtures/unrelated-ca.pem")),
            false
        )
        .await
        .is_err()
    );
}
