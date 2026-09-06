use std::{collections::HashMap, env, path::PathBuf};

use serde_json::json;
use ulid::Ulid;

use crate::{
    ApiClient, AppConfig, ChunkMetadata, ContentHash, DistanceMetric, DocumentId, MainError, Point,
    SourceUri, VectorID, embedding::local, ingest::source, store_repository::StoreRepository,
};

struct Args {
    items: Vec<String>,
}

impl Args {
    fn from_process() -> Self {
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
            .find_map(|window| (window[0] == key).then(|| window[1].clone()))
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
    let args = Args::from_process();
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
    let created = StoreRepository::from_environment().create(name.clone(), metric, dimensions)?;

    Ok(json!({
        "ok": true,
        "command": "create",
        "name": name,
        "metric": created.config.metric,
        "dimensions": dimensions,
        "path": created.path
    }))
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
    let repository = StoreRepository::from_environment();
    let mut open = repository.open(&name)?;
    let content = args.value("--content").unwrap_or_default();
    let vector = vector_or_embed_one(args, &content, open.config.dimensions).await?;
    let id = args
        .value("--id")
        .map(|id| {
            VectorID::try_from(id.as_str())
                .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidInput, error))
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
            .unwrap_or(open.store.len()),
        content_hash: ContentHash(blake3::hash(content.as_bytes()).to_string()),
        content,
        labels: parse_key_values(args.value("--labels"))?,
        path: None,
        start_line: None,
        end_line: None,
        language: None,
        session_folder: args.value("--session-folder"),
    };

    open.store.upsert(id, vector, metadata)?;
    repository.save(&open)?;
    Ok(json!({
        "ok": true,
        "command": "add",
        "mode": "content",
        "name": name,
        "id": id,
        "points": open.store.len()
    }))
}

async fn add_path(args: &Args) -> Result<serde_json::Value, MainError> {
    let name = required(args, "--name")?;
    let path = PathBuf::from(required(args, "--path")?);
    let repository = StoreRepository::from_environment();
    let mut open = repository.open(&name)?;
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

    let files = source::discover(&path)?;
    let mut added = 0usize;
    let mut skipped = 0usize;
    for file in files {
        let chunks = match source::read_chunks(&file, chunk_size, chunk_overlap) {
            Ok(chunks) => chunks,
            Err(_) => {
                skipped += 1;
                continue;
            }
        };
        let contents = chunks
            .iter()
            .map(|chunk| chunk.content.clone())
            .collect::<Vec<_>>();
        let vectors = embed_many(args, contents, open.config.dimensions).await?;
        if vectors.len() != chunks.len() {
            return Err(crate::ApiError::EmbeddingCountMismatch {
                expected: chunks.len(),
                actual: vectors.len(),
            })?;
        }
        let language = source::language(&file);
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
            open.store.upsert(VectorID::new(), vector, metadata)?;
            added += 1;
        }
    }
    repository.save(&open)?;

    Ok(json!({
        "ok": true,
        "command": "add",
        "mode": "path",
        "name": name,
        "path": path,
        "chunks_added": added,
        "files_skipped": skipped,
        "points": open.store.len()
    }))
}

async fn search(args: &Args) -> Result<serde_json::Value, MainError> {
    let name = required(args, "--name")?;
    let open = StoreRepository::from_environment().open(&name)?;
    let content = args
        .value("--content")
        .or_else(|| args.value("--query"))
        .unwrap_or_default();
    let vector = vector_or_embed_one(args, &content, open.config.dimensions).await?;
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
    let results: Vec<_> = open
        .store
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
    let repository = StoreRepository::from_environment();
    if args.has("--store") {
        repository.delete(&name)?;
        return Ok(json!({ "ok": true, "command": "delete", "name": name, "deleted": "store" }));
    }
    let id = required(args, "--id")?;
    let mut open = repository.open(&name)?;
    let vector_id = VectorID::try_from(id.as_str())
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidInput, error))?;
    open.store.delete(vector_id);
    repository.save(&open)?;
    Ok(
        json!({ "ok": true, "command": "delete", "name": name, "id": id, "points": open.store.len() }),
    )
}

fn list() -> Result<serde_json::Value, MainError> {
    let stores = StoreRepository::from_environment()
        .list()?
        .into_iter()
        .map(|store| {
            json!({
                "name": store.name,
                "metric": store.metric,
                "dimensions": store.dimensions,
                "points": store.points,
                "path": store.path
            })
        })
        .collect::<Vec<_>>();
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
            value.parse::<usize>().map_err(|error| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("invalid {key}: {error}"),
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
        embed_many(args, vec![content.to_string()], dimensions)
            .await?
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
        EmbedMode::Pseudo => Ok(local::embed_batch(&inputs, dimensions)),
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
    let vector = vector.map_err(|error| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("invalid --vector: {error}"),
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
