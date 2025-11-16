use rust_blood_pressure_monitor_ocr::get_reading_from_file;
use tokio;

#[tokio::main]
async fn main() {
    println!("Testaroonie");

    let result = get_reading_from_file("/home/happy0/bp_test.jpg").await;

    println!("{:?}", result);

    
}