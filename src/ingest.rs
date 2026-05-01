use std::fs;
use std::path::PathBuf;

use crate::TextError;

#[derive(Debug, Clone, Copy)]
pub enum FileType {
    Txt,
}

#[derive(Debug, Clone)]
pub struct Ingest {
    path: PathBuf,
    chunk_size: usize,
    filetype: FileType,
}

impl Ingest {
    pub fn new(path: PathBuf, chunk_size: usize, filetype: FileType) -> Self {
        Self {
            path,
            chunk_size,
            filetype,
        }
    }

    pub fn chunks_from_file(&self) -> Result<Vec<String>, TextError> {
        match self.filetype {
            FileType::Txt => self.chunks_from_txt(),
        }
    }

    fn chunks_from_txt(&self) -> Result<Vec<String>, TextError> {
        let text = fs::read_to_string(&self.path).map_err(|source| TextError::Read {
            path: self.path.clone(),
            source,
        })?;

        let words: Vec<&str> = text.split_whitespace().collect();
        let mut chunks = Vec::new();

        for i in (0..words.len()).step_by(self.chunk_size) {
            let end = (i + self.chunk_size).min(words.len());
            chunks.push(words[i..end].join(" "));
        }

        Ok(chunks)
    }
}
