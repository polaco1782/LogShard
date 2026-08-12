use std::{env, path::PathBuf};

use anyhow::{Context, Result};

pub fn config_file() -> Result<PathBuf> {
    Ok(platform_config_base()?.join("logshard").join("config.toml"))
}

#[cfg(windows)]
fn platform_config_base() -> Result<PathBuf> {
    absolute_environment_path("LOCALAPPDATA").context(
        "LOCALAPPDATA is not set to an absolute path, so the configuration path cannot be determined",
    )
}

#[cfg(not(windows))]
fn platform_config_base() -> Result<PathBuf> {
    if let Some(path) = absolute_environment_path("XDG_CONFIG_HOME") {
        return Ok(path);
    }

    let home = env::var_os("HOME")
        .filter(|value| !value.is_empty())
        .context("HOME is not set, so the configuration path cannot be determined")?;
    Ok(PathBuf::from(home).join(".config"))
}

fn absolute_environment_path(variable: &str) -> Option<PathBuf> {
    let value = env::var_os(variable).filter(|value| !value.is_empty())?;
    let path = PathBuf::from(value);
    path.is_absolute().then_some(path)
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::config_file;

    #[test]
    fn config_has_a_stable_central_path() {
        assert!(
            config_file()
                .unwrap()
                .ends_with(Path::new("logshard/config.toml"))
        );
    }
}
