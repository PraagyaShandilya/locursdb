use crate::id::VectorID;
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
    pub content_hash: ContentHash,
}

#[derive(Debug, Clone)]
pub struct Point {
    pub id: VectorID,
    pub vec: Vec<f32>,
    pub metadata: ChunkMetadata,
}
