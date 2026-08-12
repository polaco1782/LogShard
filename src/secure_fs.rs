use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};

pub(crate) fn ensure_private_directory(path: &Path) -> Result<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_dir() => {
            bail!("{} is not a private directory", path.display());
        }
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            fs::create_dir_all(path)
                .with_context(|| format!("could not create {}", path.display()))?;
        }
        Err(error) => {
            return Err(error).with_context(|| format!("could not inspect {}", path.display()));
        }
    }

    set_directory_permissions(path)
}

pub fn verify_private_file(path: &Path, description: &str) -> Result<()> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("could not inspect {description} at {}", path.display()))?;
    if metadata.file_type().is_symlink() {
        bail!(
            "refusing to read {description} through a link at {}",
            path.display()
        );
    }
    if !metadata.is_file() {
        bail!("{description} {} is not a regular file", path.display());
    }

    verify_file_permissions(path, &metadata, description)
}

pub fn atomic_write_private(path: &Path, contents: &[u8], description: &str) -> Result<()> {
    let parent = path
        .parent()
        .context("the private file path has no parent directory")?;
    ensure_private_directory(parent)?;

    match fs::symlink_metadata(path) {
        Ok(_) => verify_private_file(path, description)?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(error)
                .with_context(|| format!("could not inspect {description} at {}", path.display()));
        }
    }

    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("private-file");
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let temporary_path = parent.join(format!(".{file_name}.{}.{}.tmp", std::process::id(), nonce));

    let write_result = (|| -> Result<()> {
        let mut temporary = open_private_file(&temporary_path, "temporary configuration")?;
        temporary
            .write_all(contents)
            .context("could not write temporary configuration")?;
        temporary
            .sync_all()
            .context("could not flush temporary configuration")?;
        fs::rename(&temporary_path, path)
            .with_context(|| format!("could not replace {description} at {}", path.display()))?;
        sync_directory(parent)
    })();

    if write_result.is_err() {
        let _ = fs::remove_file(&temporary_path);
    }
    write_result
}

#[cfg(unix)]
fn sync_directory(path: &Path) -> Result<()> {
    File::open(path)
        .and_then(|directory| directory.sync_all())
        .with_context(|| format!("could not flush {}", path.display()))
}

#[cfg(not(unix))]
fn sync_directory(_path: &Path) -> Result<()> {
    Ok(())
}

#[cfg(unix)]
fn verify_file_permissions(path: &Path, metadata: &fs::Metadata, description: &str) -> Result<()> {
    use std::os::unix::fs::MetadataExt;

    let mode = metadata.mode() & 0o777;
    if mode & 0o077 != 0 {
        bail!(
            "{description} {} has insecure permissions {mode:03o}; run `chmod 600 {}`",
            path.display(),
            path.display()
        );
    }
    Ok(())
}

#[cfg(not(unix))]
fn verify_file_permissions(
    _path: &Path,
    _metadata: &fs::Metadata,
    _description: &str,
) -> Result<()> {
    Ok(())
}

#[cfg(unix)]
fn set_directory_permissions(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;

    fs::set_permissions(path, fs::Permissions::from_mode(0o700))
        .with_context(|| format!("could not secure {}", path.display()))
}

#[cfg(not(unix))]
fn set_directory_permissions(_path: &Path) -> Result<()> {
    Ok(())
}

#[cfg(unix)]
fn open_private_file(path: &Path, description: &str) -> Result<File> {
    use std::os::unix::fs::OpenOptionsExt;

    OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(0o600)
        .open(path)
        .with_context(|| format!("could not create {description} at {}", path.display()))
}

#[cfg(not(unix))]
fn open_private_file(path: &Path, description: &str) -> Result<File> {
    OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
        .with_context(|| format!("could not create {description} at {}", path.display()))
}
