use std::collections::HashMap;

use ordered_float::OrderedFloat;
use ulid::Ulid;

use crate::error::VectorIDError;

use super::{ChunkMetadata, ContentHash, DistanceMetric, DocumentId, Point, SourceUri, VectorID};

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct VectorStore {
    points: Vec<Point>,
    dim: usize,
    metric: DistanceMetric,
}

impl VectorStore {
    pub fn new(metric: DistanceMetric) -> Self {
        Self {
            points: Vec::new(),
            dim: 0,
            metric,
        }
    }

    pub fn with_dimensions(metric: DistanceMetric, dim: usize) -> Self {
        Self {
            points: Vec::new(),
            dim,
            metric,
        }
    }

    pub fn upsert(
        &mut self,
        id: VectorID,
        vec: Vec<f32>,
        meta: ChunkMetadata,
    ) -> Result<(), VectorIDError> {
        if self.dim == 0 {
            self.dim = vec.len();
        } else if self.dim != vec.len() {
            return Err(VectorIDError::DimMismatch {
                expected: self.dim,
                actual: vec.len(),
            });
        }

        if let Some(point) = self.points.iter_mut().find(|p| p.id == id) {
            point.vec = vec;
            point.metadata = meta;
        } else {
            self.points.push(Point {
                id,
                vec,
                metadata: meta,
            });
        }

        Ok(())
    }

    pub fn create_collections(
        &mut self,
        embeddings: Vec<Vec<f32>>,
        inputs: Vec<String>,
        model_name: String,
    ) -> Result<(), VectorIDError> {
        let document_id = DocumentId(Ulid::new().to_string());

        for (idx, (embed, input)) in embeddings.into_iter().zip(inputs).enumerate() {
            let meta = ChunkMetadata {
                document_id: document_id.clone(),
                source_uri: SourceUri(model_name.clone()),
                chunk_index: idx,
                content_hash: ContentHash(blake3::hash(input.as_bytes()).to_string()),
                content: input,
                labels: HashMap::new(),
                path: None,
                start_line: None,
                end_line: None,
                language: None,
                session_folder: None,
            };

            self.upsert(VectorID::new(), embed, meta)?;
        }

        Ok(())
    }

    pub fn get(&self, id: &VectorID) -> Result<Point, VectorIDError> {
        self.points
            .iter()
            .find(|p| &p.id == id)
            .cloned()
            .ok_or_else(|| VectorIDError::NotFound(id.to_string()))
    }

    pub fn delete(&mut self, id: VectorID) {
        self.points.retain(|p| p.id != id)
    }

    pub fn get_top_k(&self, query: &Point, k: usize) -> Vec<Point> {
        self.get_top_k_filtered(query, k, &HashMap::new())
    }

    pub fn get_top_k_filtered(
        &self,
        query: &Point,
        k: usize,
        filters: &HashMap<String, String>,
    ) -> Vec<Point> {
        self.get_top_k_filtered_with_scores(query, k, filters)
            .into_iter()
            .map(|(point, _)| point)
            .collect()
    }

    pub fn get_top_k_filtered_with_scores(
        &self,
        query: &Point,
        k: usize,
        filters: &HashMap<String, String>,
    ) -> Vec<(Point, f32)> {
        let mut scored: Vec<_> = self
            .points
            .iter()
            .filter(|point| labels_match(&point.metadata.labels, filters))
            .map(|point| (OrderedFloat(self.metric.distance(query, point)), point.id))
            .collect();
        scored.sort_by_key(|(distance, id)| (*distance, id.to_string()));

        scored
            .into_iter()
            .take(k)
            .filter_map(|(distance, id)| self.get(&id).ok().map(|point| (point, distance.0)))
            .collect()
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }
}

fn labels_match(labels: &HashMap<String, String>, filters: &HashMap<String, String>) -> bool {
    filters
        .iter()
        .all(|(key, value)| labels.get(key) == Some(value))
}
