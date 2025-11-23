use std::fs;

use bpm_ocr::{get_reading_from_buffer};

fn main() {
    let path = "/home/happy0/example11.jpg";
    let bytes = fs::read(path).unwrap();

    let result = get_reading_from_buffer(bytes);

    //let result = get_reading_from_file("/home/happy0/example11.jpg").await;

    println!("{:?}", result);
}
