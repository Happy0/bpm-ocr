use std::{fs};
use bpm_ocr::BloodPressureReadingExtractor;

use bpm_ocr::debug::BpmOcrDebugOutputter;
use bpm_ocr::debug::TempFolderDebugger;

fn main() {
   let now = chrono::offset::Local::now();
   let folder_path = now.format("%Y-%m-%d-%H-%M-%S").to_string();

    let debugger: TempFolderDebugger = TempFolderDebugger::new(&folder_path, true);

    let blood_pressure_extractor = BloodPressureReadingExtractor::new(debugger);

    let path = "/home/happy0/example9.jpg";
    let bytes = fs::read(path).unwrap();

    let result = blood_pressure_extractor.get_reading_from_buffer(bytes);

    // let result = get_reading_from_file("/home/happy0/example11.jpg");

    println!("{:?}", result);
}
