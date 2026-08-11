use super::ProviderDefaults;

pub(super) fn defaults() -> ProviderDefaults {
    ProviderDefaults {
        model: "local",
        base_url: "http://127.0.0.1:8080/v1",
        api_key_env: None,
        api_key_required: false,
    }
}
