use super::ProviderDefaults;

pub(super) fn defaults() -> ProviderDefaults {
    ProviderDefaults {
        model: "gpt-5.6",
        base_url: Some("https://api.openai.com/v1"),
        api_key_required: true,
    }
}
