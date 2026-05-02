use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddingProgress {
    pub completed_batches: usize,
    pub total_batches: usize,
    pub message: String,
}

impl EmbeddingProgress {
    pub fn new(completed_batches: usize, total_batches: usize, message: impl Into<String>) -> Self {
        Self {
            completed_batches,
            total_batches,
            message: message.into(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EmbeddingLogger {
    path: PathBuf,
    file: Arc<Mutex<File>>,
}

impl EmbeddingLogger {
    pub fn new(log_dir: impl AsRef<Path>) -> std::io::Result<Self> {
        fs::create_dir_all(log_dir.as_ref())?;
        let path = log_dir.as_ref().join("embedding.log");
        let file = File::create(&path)?;

        let logger = Self {
            path,
            file: Arc::new(Mutex::new(file)),
        };
        logger.trace("log initialized");
        Ok(logger)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn trace(&self, message: impl AsRef<str>) {
        let Ok(mut file) = self.file.lock() else {
            return;
        };

        let _ = writeln!(file, "{} | {}", timestamp_millis(), message.as_ref());
        let _ = file.flush();
    }
}

fn timestamp_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default()
}
