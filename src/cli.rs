use std::{collections::HashMap, env, path::PathBuf, sync::Arc};

use serde_json::json;
use tokio::sync::OnceCell;

use crate::{
    ApiClient, AppConfig, DistanceMetric, MainError, OpenRouterEmbedder, PseudoEmbedder, SourceUri,
    VectorID,
    application::{
        AddContentRequest, AddPathRequest, Application, CreateStoreRequest, DeleteRequest,
        DeleteTarget, EmbedFuture, Embedder, ProgressObserver, SearchRequest,
    },
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

fn application(embedder: Arc<dyn Embedder>) -> Application {
    Application::from_environment(embedder)
}

fn create(args: &Args) -> Result<serde_json::Value, MainError> {
    let name = required(args, "--name")?;
    let metric = parse_metric(args.value("--metric").as_deref().unwrap_or("euclid"))?;
    let dimensions =
        parse_positive_usize_arg(args.value("--dimensions"), "--dimensions")?.unwrap_or(1536);
    let result =
        application(Arc::new(LazyOpenRouterEmbedder::default())).create(CreateStoreRequest {
            name: name.clone(),
            metric,
            dimensions,
        })?;
    Ok(
        json!({ "ok": true, "command": "create", "name": name, "metric": result.metric, "dimensions": result.dimensions, "path": result.path }),
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
    let vector = args
        .value("--vector")
        .map(|raw| parse_vector(&raw))
        .transpose()?;
    let service = application(embedder_for(args)?);
    let id = args
        .value("--id")
        .map(|id| VectorID::try_from(id.as_str()).map_err(invalid_input))
        .transpose()?;
    let result = service
        .add_content(
            AddContentRequest {
                name: name.clone(),
                content: args.value("--content").unwrap_or_default(),
                vector,
                id,
                document_id: args.value("--document-id").map(crate::DocumentId),
                source_uri: Some(SourceUri(
                    args.value("--source")
                        .unwrap_or_else(|| "cli://add".to_string()),
                )),
                chunk_index: parse_usize_arg(args.value("--chunk-index"), "--chunk-index")?,
                labels: parse_key_values(args.value("--labels"))?,
                session_folder: args.value("--session-folder"),
            },
            None,
        )
        .await?;
    Ok(
        json!({ "ok": true, "command": "add", "mode": "content", "name": name, "id": result.id, "points": result.points }),
    )
}

async fn add_path(args: &Args) -> Result<serde_json::Value, MainError> {
    let name = required(args, "--name")?;
    let path = PathBuf::from(required(args, "--path")?);
    let chunk_size =
        parse_positive_usize_arg(args.value("--chunk-size"), "--chunk-size")?.unwrap_or(3000);
    let chunk_overlap = parse_usize_arg(args.value("--chunk-overlap"), "--chunk-overlap")?
        .unwrap_or(chunk_size / 50);
    // Keep path mode's historical behavior: --vector does not replace per-chunk embeddings.
    let service = application(embedder_for(args)?);
    let result = service
        .add_path(
            AddPathRequest {
                name: name.clone(),
                path: path.clone(),
                labels: parse_key_values(args.value("--labels"))?,
                session_folder: args.value("--session-folder"),
                chunk_size,
                chunk_overlap,
            },
            None,
        )
        .await?;
    Ok(
        json!({ "ok": true, "command": "add", "mode": "path", "name": name, "path": path, "chunks_added": result.chunks_added, "files_skipped": result.files_skipped, "points": result.points }),
    )
}

async fn search(args: &Args) -> Result<serde_json::Value, MainError> {
    let name = required(args, "--name")?;
    let vector = args
        .value("--vector")
        .map(|raw| parse_vector(&raw))
        .transpose()?;
    let top_k = parse_usize_arg(
        args.value("--top-k").or_else(|| args.value("-k")),
        "--top-k",
    )?
    .unwrap_or(5);
    let service = application(embedder_for(args)?);
    let hits = service
        .search(
            SearchRequest {
                name: name.clone(),
                content: args
                    .value("--content")
                    .or_else(|| args.value("--query"))
                    .unwrap_or_default(),
                vector,
                top_k,
                filters: parse_key_values(args.value("--filter"))?,
            },
            None,
        )
        .await?;
    let results = hits.into_iter().map(|hit| {
        let point = hit.point;
        json!({ "id": point.id, "score": hit.score, "vector": point.vec, "metadata": point.metadata, "content": point.metadata.content })
    }).collect::<Vec<_>>();
    Ok(json!({ "ok": true, "command": "search", "name": name, "top_k": top_k, "results": results }))
}

fn delete(args: &Args) -> Result<serde_json::Value, MainError> {
    let name = required(args, "--name")?;
    let service = application(Arc::new(LazyOpenRouterEmbedder::default()));
    if args.has("--store") {
        service.delete(DeleteRequest {
            name: name.clone(),
            target: DeleteTarget::Store,
        })?;
        return Ok(json!({ "ok": true, "command": "delete", "name": name, "deleted": "store" }));
    }
    let id = required(args, "--id")?;
    let vector_id = VectorID::try_from(id.as_str()).map_err(invalid_input)?;
    let result = service.delete(DeleteRequest {
        name: name.clone(),
        target: DeleteTarget::Point(vector_id),
    })?;
    Ok(
        json!({ "ok": true, "command": "delete", "name": name, "id": id, "points": result.points.expect("point delete returns count") }),
    )
}

fn list() -> Result<serde_json::Value, MainError> {
    let stores = application(Arc::new(LazyOpenRouterEmbedder::default()))
        .list()?
        .into_iter()
        .map(|store| {
            json!({
                "name": store.name, "metric": store.metric, "dimensions": store.dimensions,
                "points": store.points, "path": store.path
            })
        })
        .collect::<Vec<_>>();
    Ok(json!({ "ok": true, "command": "list", "stores": stores }))
}

fn embedder_for(args: &Args) -> Result<Arc<dyn Embedder>, MainError> {
    // Explicit vectors historically bypass provider selection, including --embed validation.
    if args.value("--path").is_none() && args.value("--vector").is_some() {
        return Ok(Arc::new(LazyOpenRouterEmbedder::default()));
    }
    match parse_embed_mode(args)? {
        EmbedMode::Pseudo => Ok(Arc::new(PseudoEmbedder)),
        EmbedMode::OpenRouter => Ok(Arc::new(LazyOpenRouterEmbedder::default())),
    }
}

#[derive(Debug, Default)]
struct LazyOpenRouterEmbedder {
    provider: OnceCell<OpenRouterEmbedder>,
}

impl Embedder for LazyOpenRouterEmbedder {
    fn embed<'a>(
        &'a self,
        inputs: Vec<String>,
        dimensions: usize,
        observer: Option<&'a ProgressObserver>,
    ) -> EmbedFuture<'a> {
        Box::pin(async move {
            let provider = self
                .provider
                .get_or_try_init(|| async {
                    let config = AppConfig::load()?;
                    Ok::<_, MainError>(OpenRouterEmbedder::new(ApiClient::new(
                        dimensions,
                        config.batch_size,
                        config.embedding_concurrency,
                        config.openrouter_api_key,
                        config.model_name,
                    )))
                })
                .await?;
            provider.embed(inputs, dimensions, observer).await
        })
    }
}

fn help() -> serde_json::Value {
    json!({ "ok": true, "commands": ["create", "add", "search", "delete", "list"], "usage": {
        "create": "locursdb create --name <store> --metric <euclid|cos|dot> --dimensions <n>",
        "add": "locursdb add --name <store> [--content <text>|--path <file-or-dir>|--vector <f32,...>] [--embed openrouter|pseudo] [--labels k:v,...] [--session-folder <path>] [--chunk-size <chars>] [--chunk-overlap <chars>]",
        "search": "locursdb search --name <store> [--content <text>|--vector <f32,...>] [--embed openrouter|pseudo] [--top-k <n>] [--filter k:v,...]",
        "delete": "locursdb delete --name <store> --id <vector-id> | --store", "list": "locursdb list"
    }})
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
        )
        .into()),
    }
}

fn parse_vector(raw: &str) -> Result<Vec<f32>, MainError> {
    raw.split(',')
        .filter(|part| !part.is_empty())
        .map(str::parse::<f32>)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("invalid --vector: {error}"),
            )
            .into()
        })
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
        )
        .into()),
    }
}

fn invalid_input(error: impl std::fmt::Display) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, error.to_string())
}
