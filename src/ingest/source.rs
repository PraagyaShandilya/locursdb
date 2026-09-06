use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone)]
pub(crate) struct SourceChunk {
    pub(crate) content: String,
    pub(crate) start_line: usize,
    pub(crate) end_line: usize,
}

pub(crate) fn discover(path: &Path) -> std::io::Result<Vec<PathBuf>> {
    if path.is_file() {
        return Ok(if is_supported_source_file(path) {
            vec![path.to_path_buf()]
        } else {
            Vec::new()
        });
    }
    if !path.is_dir() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("path not found: {}", path.display()),
        ));
    }

    let ignore_patterns = load_gitignore_patterns(path);
    let mut files = Vec::new();
    collect_files_recursive(path, path, &ignore_patterns, &mut files)?;
    files.sort();
    Ok(files)
}

pub(crate) fn read_chunks(
    path: &Path,
    chunk_size: usize,
    chunk_overlap: usize,
) -> std::io::Result<Vec<SourceChunk>> {
    fs::read_to_string(path).map(|text| chunk_text_with_lines(&text, chunk_size, chunk_overlap))
}

pub(crate) fn language(path: &Path) -> Option<&'static str> {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some("rs") => Some("rust"),
        Some("py") => Some("python"),
        Some("toml") => Some("toml"),
        Some("json") => Some("json"),
        Some("md") => Some("markdown"),
        Some("txt") => Some("text"),
        _ => None,
    }
}

fn chunk_text_with_lines(text: &str, chunk_size: usize, chunk_overlap: usize) -> Vec<SourceChunk> {
    let chunk_size = chunk_size.max(1);
    let mut chunks = recursive_split(text, chunk_size, 0)
        .into_iter()
        .map(|chunk| chunk.trim().to_string())
        .filter(|chunk| !chunk.is_empty())
        .collect::<Vec<_>>();

    if chunk_overlap > 0 && chunks.len() >= 2 {
        for index in 1..chunks.len() {
            let overlap = suffix_chars(&chunks[index - 1], chunk_overlap);
            let overlap = overlap.trim_start();
            if !overlap.is_empty() {
                chunks[index] = format!("{overlap}{}", chunks[index]);
            }
        }
    }

    chunks_with_line_ranges(text, chunks)
}

fn chunks_with_line_ranges(text: &str, chunks: Vec<String>) -> Vec<SourceChunk> {
    let mut byte_cursor = 0usize;
    chunks
        .into_iter()
        .map(|content| {
            let search_start = byte_cursor.min(text.len());
            let found_at = text[search_start..]
                .find(content.trim())
                .map(|offset| search_start + offset)
                .or_else(|| text.find(content.trim()))
                .unwrap_or(search_start);
            let end_at = found_at.saturating_add(content.len()).min(text.len());
            byte_cursor = end_at;
            SourceChunk {
                content,
                start_line: line_number_at_byte(text, found_at),
                end_line: line_number_at_byte(text, end_at),
            }
        })
        .collect()
}

fn line_number_at_byte(text: &str, byte_index: usize) -> usize {
    text[..byte_index.min(text.len())]
        .bytes()
        .filter(|byte| *byte == b'\n')
        .count()
        + 1
}

fn recursive_split(text: &str, chunk_size: usize, separator_index: usize) -> Vec<String> {
    if text.trim().is_empty() {
        return Vec::new();
    }
    if text.chars().count() <= chunk_size {
        return vec![text.to_string()];
    }

    const SEPARATORS: [&str; 5] = ["\n\n", "\n", ". ", " ", ""];
    let separator = SEPARATORS[separator_index.min(SEPARATORS.len() - 1)];
    if separator.is_empty() {
        return split_by_chars(text, chunk_size);
    }

    let pieces = text.split_inclusive(separator).collect::<Vec<_>>();
    if pieces.len() <= 1 {
        return recursive_split(text, chunk_size, separator_index + 1);
    }

    let mut split_pieces = Vec::new();
    for piece in pieces {
        if piece.chars().count() > chunk_size {
            split_pieces.extend(recursive_split(piece, chunk_size, separator_index + 1));
        } else {
            split_pieces.push(piece.to_string());
        }
    }
    merge_pieces(split_pieces, chunk_size)
}

fn merge_pieces(pieces: Vec<String>, chunk_size: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();
    for piece in pieces {
        if piece.trim().is_empty() {
            continue;
        }
        let current_len = current.chars().count();
        let piece_len = piece.chars().count();
        if !current.is_empty() && current_len + piece_len > chunk_size {
            chunks.push(std::mem::take(&mut current));
        }
        current.push_str(&piece);
    }
    if !current.trim().is_empty() {
        chunks.push(current);
    }
    chunks
}

fn split_by_chars(text: &str, chunk_size: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        current.push(ch);
        if current.chars().count() >= chunk_size {
            chunks.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

fn suffix_chars(text: &str, count: usize) -> String {
    let len = text.chars().count();
    text.chars().skip(len.saturating_sub(count)).collect()
}

fn collect_files_recursive(
    root: &Path,
    path: &Path,
    ignore_patterns: &[String],
    files: &mut Vec<PathBuf>,
) -> std::io::Result<()> {
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let entry_path = entry.path();
        if is_ignored_path(root, &entry_path, ignore_patterns) {
            continue;
        }
        if entry.file_type()?.is_dir() {
            collect_files_recursive(root, &entry_path, ignore_patterns, files)?;
        } else if entry.file_type()?.is_file() && is_supported_source_file(&entry_path) {
            files.push(entry_path);
        }
    }
    Ok(())
}

fn is_ignored_path(root: &Path, path: &Path, ignore_patterns: &[String]) -> bool {
    if path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.starts_with('.') || name == "target" || name == "log")
    {
        return true;
    }
    let relative = path.strip_prefix(root).unwrap_or(path).to_string_lossy();
    ignore_patterns.iter().any(|pattern| {
        relative.as_ref() == pattern
            || relative.starts_with(&format!("{pattern}/"))
            || path.file_name().and_then(|name| name.to_str()) == Some(pattern.as_str())
    })
}

fn load_gitignore_patterns(root: &Path) -> Vec<String> {
    fs::read_to_string(root.join(".gitignore"))
        .ok()
        .map(|content| {
            content
                .lines()
                .map(str::trim)
                .filter(|line| !line.is_empty() && !line.starts_with('#') && !line.starts_with('!'))
                .map(|line| {
                    line.trim_start_matches('/')
                        .trim_end_matches('/')
                        .to_string()
                })
                .collect()
        })
        .unwrap_or_default()
}

fn is_supported_source_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| matches!(ext, "rs" | "py" | "toml" | "json" | "md" | "txt"))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::chunk_text_with_lines;

    fn contents(text: &str, size: usize, overlap: usize) -> Vec<String> {
        chunk_text_with_lines(text, size, overlap)
            .into_iter()
            .map(|chunk| chunk.content)
            .collect()
    }

    #[test]
    fn recursive_chunking_prefers_paragraph_boundaries() {
        let text = "alpha beta gamma.\n\ndelta epsilon zeta.\n\neta theta iota.";
        assert_eq!(
            contents(text, 35, 0),
            [
                "alpha beta gamma.",
                "delta epsilon zeta.",
                "eta theta iota."
            ]
        );
    }

    #[test]
    fn recursive_chunking_falls_back_to_words_and_adds_overlap() {
        let chunks = contents("one two three four five six seven eight", 18, 5);
        assert!(chunks.len() > 1);
        assert!(chunks[0].ends_with("three"));
        assert!(chunks[1].starts_with("three"));
        assert!(chunks.iter().all(|chunk| !chunk.is_empty()));
    }

    #[test]
    fn zero_chunk_size_still_advances_on_character_boundaries() {
        assert_eq!(contents("abc", 0, 0), ["a", "b", "c"]);
    }

    #[test]
    fn line_ranges_are_one_based() {
        let chunks = chunk_text_with_lines("first\nsecond", 6, 0);
        assert_eq!((chunks[0].start_line, chunks[0].end_line), (1, 1));
        assert_eq!((chunks[1].start_line, chunks[1].end_line), (2, 2));
    }
}
