use serde::{Deserialize, Serialize};

use crate::error::VectorIDError;

use super::Point;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DistanceMetric {
    Cos,
    Euclid,
    Dot,
}

impl DistanceMetric {
    pub fn distance(&self, point1: &Point, point2: &Point) -> Result<f32, VectorIDError> {
        validate_vector(&point1.vec)?;
        validate_vector(&point2.vec)?;
        if point1.vec.len() != point2.vec.len() {
            return Err(VectorIDError::DimMismatch {
                expected: point1.vec.len(),
                actual: point2.vec.len(),
            });
        }

        match self {
            Self::Cos => Self::score_cos(point1, point2),
            Self::Euclid => Ok(Self::score_euclid(point1, point2)),
            Self::Dot => Ok(Self::score_dot_product(point1, point2)),
        }
    }

    fn euclid_norm(vec: &[f32]) -> f32 {
        vec.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    fn score_euclid(point1: &Point, point2: &Point) -> f32 {
        point2
            .vec
            .iter()
            .zip(&point1.vec)
            .map(|(x, y)| {
                let d = y - x;
                d * d
            })
            .sum()
    }

    fn score_cos(point1: &Point, point2: &Point) -> Result<f32, VectorIDError> {
        let a = Self::euclid_norm(&point1.vec);
        let b = Self::euclid_norm(&point2.vec);
        if a == 0.0 || b == 0.0 {
            return Err(VectorIDError::ZeroNorm);
        }

        Ok(1.0 - (Self::dot_product(&point1.vec, &point2.vec) / (a * b)))
    }

    fn dot_product(vec1: &[f32], vec2: &[f32]) -> f32 {
        vec1.iter().zip(vec2).map(|(a, b)| a * b).sum()
    }

    fn score_dot_product(point1: &Point, point2: &Point) -> f32 {
        1.0 - Self::dot_product(&point1.vec, &point2.vec)
    }
}

pub(super) fn validate_vector(vector: &[f32]) -> Result<(), VectorIDError> {
    if vector.is_empty() {
        return Err(VectorIDError::EmptyVector);
    }
    if let Some(index) = vector.iter().position(|value| !value.is_finite()) {
        return Err(VectorIDError::NonFiniteValue { index });
    }
    Ok(())
}
