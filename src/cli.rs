use std::{
    collections::HashMap,
    env, fs,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use serde_json::json;
use ulid::Ulid;

use crate::{
    ApiClient, AppConfig, ChunkMetadata, ContentHash, DistanceMetric, DocumentId, MainError, Point,
    SourceUri, VectorID, VectorStore,
};

#[derive(Debug, Serialize, Deserialize)]
struct StoreConfig {
    name: String,
    metric: DistanceMetric,
    dimensions: usize,
}

struct Args {
    items: Vec<String>,
}

impl Args {
    fn from_env() -> Self {
        Self {
            items: env::args().skip(1).collect(),
        }
    }

    fn command(&self) -> Option<&str> {
        self.items.first().map(String::as_str)
    }

    fn value(&self, key: &str) -> Option<String> {
        self.items
            .windows(2)
            .find_map(|w| (w[0] == key).then(|| w[1].clone()))
    }

    fn has(&self, key: &str) -> bool {
        self.items.iter().any(|item| item == key)
    }
}

pub async fn run_cli() -> Result<(), MainError> {
    match run_cli_inner().await {
        Ok(value) => {
            println!("{}", serde_json::to_string_pretty(&value)?);
            Ok(())
        }
        Err(error) => {
            eprintln!(
                "{}",
                serde_json::to_string(&json!({ "ok": false, "error": error.to_string() }))?
            );
            std::process::exit(1);
        }
    }
}

async fn run_cli_inner() -> Result<serde_json::Value, MainError> {
    let args = Args::from_env();
    match args.command() {
        Some("create") => create(&args),
        Some("add") => add(&args).await,
        Some("search") => search(&args).await,
        Some("delete") => delete(&args),
        Some("list") => list(),
        Some("help") | Some("--help") | Some("-h") | None => Ok(help()),
        Some(command) => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("unknown command: {command}"),
        ))?,
    }
}

fn create(args: &Args) -> Result<serde_json::Value, MainError> {
    let name = required(args, "--name")?;
    let metric = parse_metric(args.value("--metric").as_deref().unwrap_or("euclid"))?;
    let dimensions =
        parse_positive_usize_arg(args.value("--dimensions"), "--dimensions")?.unwrap_or(1536);
    validate_store_name(&name)?;

    let dir = store_dir(&name);
    fs::create_dir_all(&dir)?;
    let config = StoreConfig {
        name: name.clone(),
        metric,
        dimensions,
    };
    save_config(&dir, &config)?;
    if !points_path(&dir).exists() {
        save_store(
            &dir,
            &VectorStore::with_dimensions(config.metric, dimensions),
        )?;
    }

    Ok(
        json!({ "ok": true, "command": "create", "name": name, "metric": config.metric, "dimensions": dimensions, "path": dir }),
    )
}

async fn add(args: &Args) -> Result<serde_json::Value, MainError> {
    if args.value("--path").is_some() {
        add_path(args).await
    } else {
        add_content(args).await
    }
}

async fn add_content(args: &Args) -> Result<serde_json::Value, MainError> {
    let name = required(args, "--name")?;
    let dir = store_dir(&name);
    let config = load_config(&dir)?;
    let mut store = load_store(&dir, &config)?;

    let content = args.value("--content").unwrap_or_default();
    let vector = vector_or_embed_one(args, &content, config.dimensions).await?;
    let id = args
        .value("--id")
        .map(|id| {
            VectorID::try_from(id.as_str())
                .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidInput, err))
        })
        .transpose()?
        .unwrap_or_else(VectorID::new);
    let metadata = ChunkMetadata {
        document_id: DocumentId(
            args.value("--document-id")
                .unwrap_or_else(|| Ulid::new().to_string()),
        ),
        source_uri: SourceUri(
            args.value("--source")
                .unwrap_or_else(|| "cli://add".to_string()),
        ),
        chunk_index: parse_usize_arg(args.value("--chunk-index"), "--chunk-index")?
            .unwrap_or(store.len()),
        content_hash: ContentHash(blake3::hash(content.as_bytes()).to_string()),
        content,
        labels: parse_key_values(args.value("--labels"))?,
        path: None,
        start_line: None,
        end_line: None,
        language: None,
        session_folder: args.value("--session-folder"),
    };

    store.upsert(id, vector, metadata)?;
    save_store(&dir, &store)?;

    Ok(
        json!({ "ok": true, "command": "add", "mode": "content", "name": name, "id": id, "points": store.len() }),
    )
}

async fn add_path(args: &Args) -> Result<serde_json::Value, MainError> {
    let name = required(args, "--name")?;
    let path = PathBuf::from(required(args, "--path")?);
    let dir = store_dir(&name);
    let config = load_config(&dir)?;
    let mut store = load_store(&dir, &config)?;
    let labels = parse_key_values(args.value("--labels"))?;
    let chunk_size =
        parse_positive_usize_arg(args.value("--chunk-size"), "--chunk-size")?.unwrap_or(3000);
    let chunk_overlap = parse_usize_arg(args.value("--chunk-overlap"), "--chunk-overlap")?
        .unwrap_or(chunk_size / 50);
    if chunk_overlap >= chunk_size {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "--chunk-overlap must be smaller than --chunk-size",
        )
        .into());
    }
    let files = collect_files(&path)?;
    let mut added = 0usize;
    let mut skipped = 0usize;

    for file in files {
        let text = match fs::read_to_string(&file) {
            Ok(text) => text,
            Err(_) => {
                skipped += 1;
                continue;
            }
        };
        let chunks = chunk_text_with_lines(&text, chunk_size, chunk_overlap);
        let contents = chunks
            .iter()
            .map(|chunk| chunk.content.clone())
            .collect::<Vec<_>>();
        let vectors = embed_many(args, contents, config.dimensions).await?;
        if vectors.len() != chunks.len() {
            return Err(crate::ApiError::EmbeddingCountMismatch {
                expected: chunks.len(),
                actual: vectors.len(),
            })?;
        }
        let language = language_for_path(&file);
        let document_id = DocumentId(Ulid::new().to_string());
        for (chunk_index, (chunk, vector)) in chunks.into_iter().zip(vectors).enumerate() {
            let mut chunk_labels = labels.clone();
            chunk_labels.insert("path".to_string(), file.to_string_lossy().into_owned());
            chunk_labels.insert("start_line".to_string(), chunk.start_line.to_string());
            chunk_labels.insert("end_line".to_string(), chunk.end_line.to_string());
            if let Some(language) = language {
                chunk_labels.insert("language".to_string(), language.to_string());
            }
            let metadata = ChunkMetadata {
                document_id: document_id.clone(),
                source_uri: SourceUri(file.to_string_lossy().into_owned()),
                chunk_index,
                content_hash: ContentHash(blake3::hash(chunk.content.as_bytes()).to_string()),
                content: chunk.content,
                labels: chunk_labels,
                path: Some(file.to_string_lossy().into_owned()),
                start_line: Some(chunk.start_line),
                end_line: Some(chunk.end_line),
                language: language.map(str::to_string),
                session_folder: args.value("--session-folder"),
            };
            store.upsert(VectorID::new(), vector, metadata)?;
            added += 1;
        }
    }

    save_store(&dir, &store)?;

    Ok(
        json!({ "ok": true, "command": "add", "mode": "path", "name": name, "path": path, "chunks_added": added, "files_skipped": skipped, "points": store.len() }),
    )
}

async fn search(args: &Args) -> Result<serde_json::Value, MainError> {
    let name = required(args, "--name")?;
    let dir = store_dir(&name);
    let config = load_config(&dir)?;
    let store = load_store(&dir, &config)?;
    let content = args
        .value("--content")
        .or_else(|| args.value("--query"))
        .unwrap_or_default();
    let vector = vector_or_embed_one(args, &content, config.dimensions).await?;
    let top_k = parse_usize_arg(
        args.value("--top-k").or_else(|| args.value("-k")),
        "--top-k",
    )?
    .unwrap_or(5);
    let filters = parse_key_values(args.value("--filter"))?;
    let query = Point {
        id: VectorID::new(),
        vec: vector,
        metadata: ChunkMetadata::default(),
    };
    let results: Vec<_> = store
        .get_top_k_filtered_with_scores(&query, top_k, &filters)?
        .into_iter()
        .map(|(point, score)| {
            json!({ "id": point.id, "score": score, "vector": point.vec, "metadata": point.metadata, "content": point.metadata.content })
        })
        .collect();

    Ok(json!({ "ok": true, "command": "search", "name": name, "top_k": top_k, "results": results }))
}

fn delete(args: &Args) -> Result<serde_json::Value, MainError> {
    let name = required(args, "--name")?;
    let dir = store_dir(&name);
    if args.has("--store") {
        fs::remove_dir_all(&dir)?;
        return Ok(json!({ "ok": true, "command": "delete", "name": name, "deleted": "store" }));
    }
    let id = required(args, "--id")?;
    let config = load_config(&dir)?;
    let mut store = load_store(&dir, &config)?;
    let vector_id = VectorID::try_from(id.as_str())
        .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidInput, err))?;
    store.delete(vector_id);
    save_store(&dir, &store)?;
    Ok(json!({ "ok": true, "command": "delete", "name": name, "id": id, "points": store.len() }))
}

fn list() -> Result<serde_json::Value, MainError> {
    let root = root_dir();
    let mut stores = Vec::new();
    if root.exists() {
        for entry in fs::read_dir(&root)? {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let dir = entry.path();
            if let Ok(config) = load_config(&dir) {
                let points = load_store(&dir, &config)
                    .map(|store| store.len())
                    .unwrap_or(0);
                stores.push(json!({ "name": config.name, "metric": config.metric, "dimensions": config.dimensions, "points": points, "path": dir }));
            }
        }
    }
    Ok(json!({ "ok": true, "command": "list", "stores": stores }))
}

fn help() -> serde_json::Value {
    json!({
        "ok": true,
        "commands": ["create", "add", "search", "delete", "list"],
        "usage": {
            "create": "locursdb create --name <store> --metric <euclid|cos|dot> --dimensions <n>",
            "add": "locursdb add --name <store> [--content <text>|--path <file-or-dir>|--vector <f32,...>] [--embed openrouter|pseudo] [--labels k:v,...] [--session-folder <path>] [--chunk-size <chars>] [--chunk-overlap <chars>]",
            "search": "locursdb search --name <store> [--content <text>|--vector <f32,...>] [--embed openrouter|pseudo] [--top-k <n>] [--filter k:v,...]",
            "delete": "locursdb delete --name <store> --id <vector-id> | --store",
            "list": "locursdb list"
        }
    })
}

fn required(args: &Args, key: &str) -> Result<String, MainError> {
    args.value(key)
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("missing required argument {key}"),
            )
        })
        .map_err(Into::into)
}

fn parse_usize_arg(value: Option<String>, key: &str) -> Result<Option<usize>, MainError> {
    value
        .map(|value| {
            value.parse::<usize>().map_err(|err| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("invalid {key}: {err}"),
                )
            })
        })
        .transpose()
        .map_err(Into::into)
}

fn parse_positive_usize_arg(value: Option<String>, key: &str) -> Result<Option<usize>, MainError> {
    let parsed = parse_usize_arg(value, key)?;
    if parsed == Some(0) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("{key} must be greater than zero"),
        )
        .into());
    }
    Ok(parsed)
}

fn parse_key_values(value: Option<String>) -> Result<HashMap<String, String>, MainError> {
    let mut labels = HashMap::new();
    let Some(value) = value else {
        return Ok(labels);
    };
    for pair in value.split(',').filter(|pair| !pair.is_empty()) {
        let Some((key, value)) = pair.split_once(':') else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("invalid key/value pair: {pair}; expected k:v"),
            )
            .into());
        };
        labels.insert(key.to_string(), value.to_string());
    }
    Ok(labels)
}

fn parse_metric(value: &str) -> Result<DistanceMetric, MainError> {
    match value.to_ascii_lowercase().as_str() {
        "euclid" | "euclidean" => Ok(DistanceMetric::Euclid),
        "cos" | "cosine" => Ok(DistanceMetric::Cos),
        "dot" | "dot-product" => Ok(DistanceMetric::Dot),
        _ => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("invalid metric: {value}"),
        ))?,
    }
}

async fn vector_or_embed_one(
    args: &Args,
    content: &str,
    dimensions: usize,
) -> Result<Vec<f32>, MainError> {
    if let Some(raw) = args.value("--vector") {
        parse_vector(&raw, dimensions)
    } else {
        let embeddings = embed_many(args, vec![content.to_string()], dimensions).await?;
        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "embedding response was empty",
                )
            })
            .map_err(Into::into)
    }
}

async fn embed_many(
    args: &Args,
    inputs: Vec<String>,
    dimensions: usize,
) -> Result<Vec<Vec<f32>>, MainError> {
    match parse_embed_mode(args)? {
        EmbedMode::Pseudo => Ok(inputs
            .iter()
            .map(|content| pseudo_embedding(content, dimensions))
            .collect()),
        EmbedMode::OpenRouter => {
            let config = AppConfig::load()?;
            let api = ApiClient::new(
                dimensions,
                config.batch_size,
                config.embedding_concurrency,
                config.openrouter_api_key,
                config.model_name,
            );
            Ok(api.convert_input_to_embeddings(inputs).await?)
        }
    }
}

fn parse_vector(raw: &str, dimensions: usize) -> Result<Vec<f32>, MainError> {
    let vector: Result<Vec<_>, _> = raw
        .split(',')
        .filter(|part| !part.is_empty())
        .map(|part| part.parse::<f32>())
        .collect();
    let vector = vector.map_err(|err| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("invalid --vector: {err}"),
        )
    })?;
    if vector.len() != dimensions {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "vector dimensions mismatch: expected {dimensions}, got {}",
                vector.len()
            ),
        )
        .into());
    }
    if let Some(index) = vector.iter().position(|value| !value.is_finite()) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("vector contains a non-finite value at index {index}"),
        )
        .into());
    }
    Ok(vector)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EmbedMode {
    OpenRouter,
    Pseudo,
}

fn parse_embed_mode(args: &Args) -> Result<EmbedMode, MainError> {
    match args
        .value("--embed")
        .unwrap_or_else(|| "openrouter".to_string())
        .to_ascii_lowercase()
        .as_str()
    {
        "openrouter" | "api" => Ok(EmbedMode::OpenRouter),
        "pseudo" | "local" | "deterministic" => Ok(EmbedMode::Pseudo),
        value => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("invalid --embed: {value}; expected openrouter or pseudo"),
        ))?,
    }
}

fn pseudo_embedding(content: &str, dimensions: usize) -> Vec<f32> {
    let mut vector = vec![0.0; dimensions];
    if dimensions == 0 {
        return vector;
    }
    for (index, byte) in content.bytes().enumerate() {
        let slot = index % dimensions;
        vector[slot] += (byte as f32) / 255.0;
    }
    let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut vector {
            *value /= norm;
        }
    }
    vector
}

#[derive(Debug, Clone)]
struct TextChunk {
    content: String,
    start_line: usize,
    end_line: usize,
}

#[cfg(test)]
fn chunk_text(text: &str, chunk_size: usize, chunk_overlap: usize) -> Vec<String> {
    chunk_text_with_lines(text, chunk_size, chunk_overlap)
        .into_iter()
        .map(|chunk| chunk.content)
        .collect()
}

fn chunk_text_with_lines(text: &str, chunk_size: usize, chunk_overlap: usize) -> Vec<TextChunk> {
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

fn chunks_with_line_ranges(text: &str, chunks: Vec<String>) -> Vec<TextChunk> {
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
            TextChunk {
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

fn collect_files(path: &Path) -> Result<Vec<PathBuf>, MainError> {
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
        )
        .into());
    }

    let ignore_patterns = load_gitignore_patterns(path);
    let mut files = Vec::new();
    collect_files_recursive(path, path, &ignore_patterns, &mut files)?;
    files.sort();
    Ok(files)
}

fn collect_files_recursive(
    root: &Path,
    path: &Path,
    ignore_patterns: &[String],
    files: &mut Vec<PathBuf>,
) -> Result<(), MainError> {
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

fn language_for_path(path: &Path) -> Option<&'static str> {
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

fn validate_store_name(name: &str) -> Result<(), MainError> {
    if name.is_empty() || name.contains('/') || name.contains('\\') || name == "." || name == ".." {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "store name must be a simple path segment",
        )
        .into());
    }
    Ok(())
}

fn root_dir() -> PathBuf {
    env::var("LOCURSDB_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(".locursdb"))
}
fn store_dir(name: &str) -> PathBuf {
    root_dir().join(name)
}
fn config_path(dir: &Path) -> PathBuf {
    dir.join("config.json")
}
fn points_path(dir: &Path) -> PathBuf {
    dir.join("points.json")
}

fn save_config(dir: &Path, config: &StoreConfig) -> Result<(), MainError> {
    fs::write(config_path(dir), serde_json::to_vec_pretty(config)?)?;
    Ok(())
}

fn load_config(dir: &Path) -> Result<StoreConfig, MainError> {
    let config: StoreConfig = serde_json::from_slice(&fs::read(config_path(dir))?)?;
    if config.dimensions == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "stored dimensions must be greater than zero",
        )
        .into());
    }
    Ok(config)
}

fn save_store(dir: &Path, store: &VectorStore) -> Result<(), MainError> {
    store.validate()?;
    fs::write(points_path(dir), serde_json::to_vec_pretty(store)?)?;
    Ok(())
}

fn load_store(dir: &Path, config: &StoreConfig) -> Result<VectorStore, MainError> {
    let store = if points_path(dir).exists() {
        serde_json::from_slice(&fs::read(points_path(dir))?)?
    } else {
        VectorStore::with_dimensions(config.metric, config.dimensions)
    };
    store.validate()?;
    if store.dimensions() != config.dimensions {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "store dimensions mismatch: config has {}, points have {}",
                config.dimensions,
                store.dimensions()
            ),
        )
        .into());
    }
    if store.metric() != config.metric {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "store metric mismatch: config has {:?}, points have {:?}",
                config.metric,
                store.metric()
            ),
        )
        .into());
    }
    Ok(store)
}

#[cfg(test)]
mod tests {
    use super::{chunk_text, parse_positive_usize_arg, parse_vector};

    #[test]
    fn recursive_chunking_prefers_paragraph_boundaries() {
        let text = "alpha beta gamma.\n\ndelta epsilon zeta.\n\neta theta iota.";

        let chunks = chunk_text(text, 35, 0);

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], "alpha beta gamma.");
        assert_eq!(chunks[1], "delta epsilon zeta.");
        assert_eq!(chunks[2], "eta theta iota.");
    }

    #[test]
    fn recursive_chunking_falls_back_to_words_and_adds_overlap() {
        let text = "one two three four five six seven eight";

        let chunks = chunk_text(text, 18, 5);

        assert!(chunks.len() > 1);
        assert!(chunks[0].ends_with("three"));
        assert!(chunks[1].starts_with("three"));
        assert!(chunks.iter().all(|chunk| !chunk.is_empty()));
    }

    #[test]
    fn positive_usize_argument_rejects_zero() {
        let error = parse_positive_usize_arg(Some("0".to_string()), "--dimensions").unwrap_err();

        assert_eq!(error.to_string(), "--dimensions must be greater than zero");
    }

    #[test]
    fn vector_argument_rejects_non_finite_values() {
        let error = parse_vector("1,NaN", 2).unwrap_err();

        assert_eq!(
            error.to_string(),
            "vector contains a non-finite value at index 1"
        );
    }
}
