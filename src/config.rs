use std::collections::HashMap;
use std::path::PathBuf;

use crate::DotEnvError;

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub env_path: PathBuf,
    pub corpus_path: PathBuf,
    pub batch_size: usize,
    pub chunk_size: usize,
    pub dimensions: usize,
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
            corpus_path: manifest_dir.join("corpus/sample.txt"),
            batch_size: parse_usize(&values, "BATCH_SIZE")?,
            chunk_size: parse_usize(&values, "CHUNK_SIZE")?,
            dimensions: parse_usize(&values, "DIMENSIONS")?,
            openrouter_api_key: get_required(&values, "OPENROUTER_API_KEY", &manifest_dir.join(".env"))?,
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
    let value = get_required(values, key, &PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".env"))?;
    value.parse::<usize>().map_err(|source| DotEnvError::InvalidUsize {
        key,
        value,
        source,
    })
}
