use locursdb::{MainError, run};

#[tokio::main]
async fn main() -> Result<(), MainError> {
    run().await
}
