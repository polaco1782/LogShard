use std::io::{self, IsTerminal, Write};
use std::sync::mpsc::{self, Sender};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const SPINNER_FRAMES: &[char] = &['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
const FRAME_INTERVAL: Duration = Duration::from_millis(80);

pub struct ActivityIndicator {
    stop: Option<Sender<()>>,
    worker: Option<JoinHandle<()>>,
}

impl ActivityIndicator {
    pub fn start(message: impl Into<String>, enabled: bool) -> Self {
        if !enabled || !io::stderr().is_terminal() {
            return Self::disabled();
        }

        let message = message.into();
        let (sender, receiver) = mpsc::channel();
        let worker = thread::Builder::new()
            .name("logshard-activity-indicator".to_owned())
            .spawn(move || {
                let started = Instant::now();
                for frame in SPINNER_FRAMES.iter().cycle() {
                    render_status(*frame, &message, started.elapsed());
                    match receiver.recv_timeout(FRAME_INTERVAL) {
                        Err(mpsc::RecvTimeoutError::Timeout) => {}
                        Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
                    }
                }
                clear_status();
            });

        match worker {
            Ok(worker) => Self {
                stop: Some(sender),
                worker: Some(worker),
            },
            Err(_) => Self::disabled(),
        }
    }

    fn disabled() -> Self {
        Self {
            stop: None,
            worker: None,
        }
    }
}

impl Drop for ActivityIndicator {
    fn drop(&mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn render_status(frame: char, message: &str, elapsed: Duration) {
    let mut stderr = io::stderr().lock();
    let _ = write!(
        stderr,
        "\r\x1b[2K{frame} {message} ({}s)",
        elapsed.as_secs()
    );
    let _ = stderr.flush();
}

fn clear_status() {
    let mut stderr = io::stderr().lock();
    let _ = write!(stderr, "\r\x1b[2K");
    let _ = stderr.flush();
}

#[cfg(test)]
mod tests {
    use super::ActivityIndicator;

    #[test]
    fn disabled_indicator_does_not_start_a_worker() {
        let indicator = ActivityIndicator::start("Analyzing logs", false);

        assert!(indicator.stop.is_none());
        assert!(indicator.worker.is_none());
    }
}
