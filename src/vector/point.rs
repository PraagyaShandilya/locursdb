use std::collections::HashMap;

use super::VectorID;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct DocumentId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct SourceUri(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ContentHash(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ChunkMetadata {
    pub document_id: DocumentId,
    pub source_uri: SourceUri,
    pub chunk_index: usize,
    #[serde(default)]
    pub content_hash: ContentHash,
    #[serde(default)]
    pub content: String,
    #[serde(default)]
    pub labels: HashMap<String, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start_line: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end_line: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_folder: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Point {
    pub id: VectorID,
    pub vec: Vec<f32>,
    pub metadata: ChunkMetadata,
}
