use super::Point;

#[derive(Debug)]
pub enum DistanceMetric {
    Cos,
    Euclid,
    Dot,
}

impl DistanceMetric {
    pub fn distance(&self, point1: &Point, point2: &Point) -> f32 {
        let distance = match self {
            Self::Cos => Self::score_cos(point1, point2),
            Self::Euclid => Self::score_euclid(point1, point2),
            Self::Dot => Self::score_dot_product(point1, point2),
        };
        distance
    }

    fn euclid_norm(vec: &Vec<f32>) -> f32 {
        let sum: f32 = vec.iter().map(|x| x * x).sum();
        sum.sqrt()
    }

    fn score_euclid(point1: &Point, point2: &Point) -> f32 {
        point2
            .vec
            .iter()
            .zip(point1.vec.clone())
            .map(|(x, y)| {
                let d = y - x;
                d * d
            })
            .sum()
    }

    fn score_cos(point1: &Point, point2: &Point) -> f32 {
        let vec1: &Vec<f32> = &point1.vec;
        let vec2: &Vec<f32> = &point2.vec;

        let a: f32 = Self::euclid_norm(&vec1.to_vec());
        let b: f32 = Self::euclid_norm(&vec2.to_vec());

        let dot_product: f32 = Self::dot_product(&point1.vec, &point2.vec);

        1.0 - (dot_product / (a * b))
    }

    fn dot_product(vec1: &[f32], vec2: &[f32]) -> f32 {
        vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum()
    }

    fn score_dot_product(point1: &Point, point2: &Point) -> f32 {
        let res: f32 = point2
            .vec
            .iter()
            .zip(point1.vec.clone())
            .map(|(a, b)| a * b)
            .sum();
        1.0 - res
    }
}
