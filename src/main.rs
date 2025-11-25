use bpm_ocr::BloodPressureReadingExtractor;
use std::fs;

use bpm_ocr::debug::TempFolderDebugger;

fn main() {
    let debugger: TempFolderDebugger = TempFolderDebugger::using_timestamp_folder_name(true);

    let blood_pressure_extractor = BloodPressureReadingExtractor::new(debugger);

    let path = "/home/happy0/example9.jpg";
    let bytes = fs::read(path).unwrap();

    let result = blood_pressure_extractor.get_reading_from_buffer(bytes);

    // let result = get_reading_from_file("/home/happy0/example11.jpg");

    println!("{:?}", result);
}
