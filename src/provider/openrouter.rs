use super::ProviderDefaults;

pub(super) fn defaults() -> ProviderDefaults {
    ProviderDefaults {
        model: "openrouter/auto",
        base_url: "https://openrouter.ai/api/v1",
        api_key_required: true,
    }
}
