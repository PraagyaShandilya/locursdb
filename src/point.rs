use crate::id::VectorID;
use serde_json::Value;

#[derive(Debug,Clone)]
pub struct Point {
    pub id:VectorID,
    pub vec:Vec<f32>,
    pub metadata:Value,
}