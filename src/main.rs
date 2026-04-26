use std::path::PathBuf;

use locursdb::{EmbeddingsRequest, EmbeddingsResponse, MainError, VectorStore};

fn load_openrouter_api_key() -> Result<String, MainError> {
    let env_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".env");
    let iter = dotenvy::from_path_iter(&env_path)?;

    for item in iter {
        let (key, value) = item?;
        if key == "OPENROUTER_API_KEY" {
            return Ok(value);
        }
    }

    Err(MainError::MissingEnvVar {
        key: "OPENROUTER_API_KEY",
        path: env_path,
    })
}


async fn embeddings_api_call(dimensions:f32, model_name:&str) -> Result<Vec<Vec<f32>>, MainError> {
    let inputs = vec![
        "The quick brown fox jumps over the lazy dog".to_string(),
        "Pack my box with five dozen liquor jugs".to_string(),
    ];
    let request = EmbeddingsRequest::new(
        inputs,
        model_name,
        dimensions
    );

    let openrouter_api_key = load_openrouter_api_key()?;
    let client = reqwest::Client::new();
    let res = client
        .post("https://openrouter.ai/api/v1/embeddings")
        .bearer_auth(&openrouter_api_key)
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .await?;

    let status = res.status();
    let body = res.text().await?;

    if !status.is_success() {
        return Err(MainError::Api { status, body });
    }

    let response: EmbeddingsResponse = serde_json::from_str(&body)?;

    Ok(response.into_embeddings())
}



#[tokio::main]
async fn main() -> Result<(), MainError> {
    
    const dimensions: f32 = 512.0;
    const model_name: &str = "nvidia/llama-nemotron-embed-vl-1b-v2:free"; 

    let embeddings = embeddings_api_call(dimensions,model_name).await?;

    println!("{}", serde_json::to_string_pretty(&embeddings)?);

    Ok(())
}
