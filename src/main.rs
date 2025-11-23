use bpm_ocr::get_reading_from_file;
use tokio;

#[tokio::main]
async fn main() {
    let result = get_reading_from_file("/home/happy0/example11.jpg").await;

    println!("{:?}", result);
}
