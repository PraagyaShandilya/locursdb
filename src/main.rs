use locursdb::{MainError, run, run_cli};

#[tokio::main]
async fn main() -> Result<(), MainError> {
    if std::env::args().len() > 1 {
        run_cli().await
    } else {
        run().await
    }
}
