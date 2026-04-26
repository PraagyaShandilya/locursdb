use crate::error::{VectorIDError};
use crate::point::Point;
use crate::id::VectorID;
use crate::distance::DistanceMetric;


#[derive(Debug)]
pub struct VectorStore {
    points: Vec<Point>,
    dim: usize,
    metric: DistanceMetric,
}

impl VectorStore {
    
    pub fn new(metric:DistanceMetric) -> Self{
        Self {
              points:Vec::new(),
              dim: 0,
              metric
            }
    }

    pub fn upsert(&mut self,id:VectorID, vec:Vec<f32>, meta: serde_json::Value) 
        -> Result<(),VectorIDError> {
            //make sure the dimension is either set to a fixed value or meets the fixed value
            if self.dim == 0{
                self.dim =vec.len();
            }   else if self.dim != vec.len(){
                return Err(VectorIDError::DimMismatch{
                    expected: self.dim,
                    actual: vec.len(),
                });
            }

            if let Some(point) = self.points.iter_mut().find(|p| p.id == id){
                
                point.vec = vec;
                point.metadata = meta;
            
            }else {

                self.points.push(Point {
                    id, vec, metadata : meta
                });

            }
            Ok(())

            
        }
    

    //get(id) 
    pub fn get(&self, id: &VectorID) -> Result<Point,VectorIDError> {
        if let Some(point) = self.points.iter().find(|p| &p.id == id) { 
            Ok(point.clone()) 
        }
        else {
            Err(VectorIDError::NotFound(id.to_string()))
        }
    }


    pub fn delete(&mut self, id:VectorID){
        self.points.retain(|p| p.id != id)
    }
    


    pub fn get_top_k(&self, query: &[f32], k:u32) -> [f32] {
        
    }


    //find the length of the vector store 
    pub fn len(&self) 
        -> usize { 
        self.points.len()
    }

}
