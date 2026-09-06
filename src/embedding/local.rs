pub(crate) fn embed_batch(inputs: &[String], dimensions: usize) -> Vec<Vec<f32>> {
    inputs
        .iter()
        .map(|content| embed(content, dimensions))
        .collect()
}

fn embed(content: &str, dimensions: usize) -> Vec<f32> {
    let mut vector = vec![0.0; dimensions];
    if dimensions == 0 {
        return vector;
    }
    for (index, byte) in content.bytes().enumerate() {
        let slot = index % dimensions;
        vector[slot] += (byte as f32) / 255.0;
    }
    let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut vector {
            *value /= norm;
        }
    }
    vector
}

#[cfg(test)]
mod tests {
    use super::{embed, embed_batch};

    #[test]
    fn embeddings_are_deterministic() {
        assert_eq!(embed("same input", 8), embed("same input", 8));
    }

    #[test]
    fn embeddings_have_requested_dimension() {
        let embeddings = embed_batch(&["one".to_string(), "two".to_string()], 7);
        assert!(embeddings.iter().all(|embedding| embedding.len() == 7));
    }

    #[test]
    fn nonempty_embeddings_are_normalized() {
        let vector = embed("normalize me", 8);
        let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < f32::EPSILON * 4.0);
    }

    #[test]
    fn zero_dimensions_produces_empty_embedding() {
        assert!(embed("anything", 0).is_empty());
    }
}
