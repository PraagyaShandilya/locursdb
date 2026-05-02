use std::collections::HashMap;
use std::path::PathBuf;

use crate::DotEnvError;

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub env_path: PathBuf,
    pub corpus_dir: PathBuf,
    pub corpus_path: PathBuf,
    pub batch_size: usize,
    pub embedding_concurrency: usize,
    pub chunk_size: usize,
    pub dimensions: usize,
    pub top_k: usize,
    pub openrouter_api_key: String,
    pub model_name: String,
}

impl AppConfig {
    pub fn load() -> Result<Self, DotEnvError> {
        let env_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".env");
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let values = load_env_map(&env_path)?;

        Ok(Self {
            env_path,
            corpus_dir: manifest_dir.join("corpus"),
            corpus_path: parse_path(
                &values,
                "CORPUS_PATH",
                manifest_dir.join("corpus/sample.txt"),
            ),
            batch_size: parse_usize(&values, "BATCH_SIZE")?,
            embedding_concurrency: parse_usize(&values, "EMBEDDING_CONCURRENCY")?,
            chunk_size: parse_usize(&values, "CHUNK_SIZE")?,
            dimensions: parse_usize(&values, "DIMENSIONS")?,
            top_k: parse_optional_usize(&values, "TOP_K", 5)?,
            openrouter_api_key: get_required(
                &values,
                "OPENROUTER_API_KEY",
                &manifest_dir.join(".env"),
            )?,
            model_name: get_required(&values, "MODEL_NAME", &manifest_dir.join(".env"))?,
        })
    }
}

fn load_env_map(path: &PathBuf) -> Result<HashMap<String, String>, DotEnvError> {
    let iter = dotenvy::from_path_iter(path)?;
    let mut values = HashMap::new();

    for item in iter {
        let (key, value) = item?;
        values.insert(key, value);
    }

    Ok(values)
}

fn get_required(
    values: &HashMap<String, String>,
    key: &'static str,
    path: &PathBuf,
) -> Result<String, DotEnvError> {
    values
        .get(key)
        .cloned()
        .ok_or_else(|| DotEnvError::MissingEnvVar {
            key,
            path: path.clone(),
        })
}

fn parse_usize(values: &HashMap<String, String>, key: &'static str) -> Result<usize, DotEnvError> {
    let value = get_required(
        values,
        key,
        &PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".env"),
    )?;
    parse_usize_value(key, value)
}

fn parse_optional_usize(
    values: &HashMap<String, String>,
    key: &'static str,
    default: usize,
) -> Result<usize, DotEnvError> {
    values
        .get(key)
        .cloned()
        .map(|value| parse_usize_value(key, value))
        .unwrap_or(Ok(default))
}

fn parse_usize_value(key: &'static str, value: String) -> Result<usize, DotEnvError> {
    value
        .parse::<usize>()
        .map_err(|source| DotEnvError::InvalidUsize { key, value, source })
}

fn parse_path(values: &HashMap<String, String>, key: &'static str, default: PathBuf) -> PathBuf {
    values
        .get(key)
        .map(PathBuf::from)
        .map(|path| {
            if path.is_absolute() {
                path
            } else {
                PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(path)
            }
        })
        .unwrap_or(default)
}
