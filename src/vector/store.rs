use std::collections::{HashMap, HashSet};

use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use ulid::Ulid;

use crate::error::VectorIDError;

use super::distance::validate_vector;
use super::{ChunkMetadata, ContentHash, DistanceMetric, DocumentId, Point, SourceUri, VectorID};

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
        validate_vector(&vec)?;
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
        if embeddings.len() != inputs.len() {
            return Err(VectorIDError::CollectionLengthMismatch {
                embeddings: embeddings.len(),
                inputs: inputs.len(),
            });
        }

        let mut expected_dim = self.dim;
        for embedding in &embeddings {
            validate_vector(embedding)?;
            if expected_dim == 0 {
                expected_dim = embedding.len();
            } else if embedding.len() != expected_dim {
                return Err(VectorIDError::DimMismatch {
                    expected: expected_dim,
                    actual: embedding.len(),
                });
            }
        }

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

    pub fn get_top_k(&self, query: &Point, k: usize) -> Result<Vec<Point>, VectorIDError> {
        self.get_top_k_filtered(query, k, &HashMap::new())
    }

    pub fn get_top_k_filtered(
        &self,
        query: &Point,
        k: usize,
        filters: &HashMap<String, String>,
    ) -> Result<Vec<Point>, VectorIDError> {
        Ok(self
            .get_top_k_filtered_with_scores(query, k, filters)?
            .into_iter()
            .map(|(point, _)| point)
            .collect())
    }

    pub fn get_top_k_filtered_with_scores(
        &self,
        query: &Point,
        k: usize,
        filters: &HashMap<String, String>,
    ) -> Result<Vec<(Point, f32)>, VectorIDError> {
        validate_vector(&query.vec)?;
        if self.dim != 0 && query.vec.len() != self.dim {
            return Err(VectorIDError::DimMismatch {
                expected: self.dim,
                actual: query.vec.len(),
            });
        }

        let mut scored = self
            .points
            .iter()
            .filter(|point| labels_match(&point.metadata.labels, filters))
            .map(|point| {
                self.metric
                    .distance(query, point)
                    .map(|distance| (OrderedFloat(distance), point))
            })
            .collect::<Result<Vec<_>, _>>()?;
        scored.sort_by_key(|(distance, point)| (*distance, point.id.to_string()));

        Ok(scored
            .into_iter()
            .take(k)
            .map(|(distance, point)| (point.clone(), distance.0))
            .collect())
    }

    pub fn validate(&self) -> Result<(), VectorIDError> {
        let mut ids = HashSet::with_capacity(self.points.len());
        for point in &self.points {
            validate_vector(&point.vec)?;
            if point.vec.len() != self.dim {
                return Err(VectorIDError::DimMismatch {
                    expected: self.dim,
                    actual: point.vec.len(),
                });
            }
            if !ids.insert(point.id) {
                return Err(VectorIDError::DuplicateId(point.id.to_string()));
            }
        }
        Ok(())
    }

    pub fn dimensions(&self) -> usize {
        self.dim
    }

    pub fn metric(&self) -> DistanceMetric {
        self.metric
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

fn labels_match(labels: &HashMap<String, String>, filters: &HashMap<String, String>) -> bool {
    filters
        .iter()
        .all(|(key, value)| labels.get(key) == Some(value))
}
