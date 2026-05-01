use std::collections::BTreeMap;

use base64::prelude::*;
use ordered_float::OrderedFloat;
use ulid::Ulid;

use crate::distance::DistanceMetric;
use crate::error::VectorIDError;
use crate::id::VectorID;
use crate::point::{ChunkMetadata, ContentHash, DocumentId, Point, SourceUri};

#[derive(Debug)]
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
                content_hash: ContentHash(BASE64_STANDARD.encode(input.as_bytes())),
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
        let mut bmap = BTreeMap::new();
        let mut results = Vec::new();

        for point in &self.points {
            let distance = self.metric.distance(query, point);
            bmap.insert(OrderedFloat(distance), point.id);
        }

        for (_, id) in bmap.iter().take(k) {
            let vector = self.get(id).unwrap();
            results.push(vector);
        }

        results
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }
}
