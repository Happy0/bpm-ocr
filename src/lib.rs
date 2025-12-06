use std::sync::Arc;

use opencv::core::{Mat, Size, Vector};
use opencv::imgcodecs::ImreadModes;
use opencv::{imgcodecs, imgproc};

use crate::debug::BpmOcrDebugOutputter;
use crate::lcd_number_extractor::LcdNumberExtractor;
use crate::lcd_screen_extractor::LcdScreenExtractor;
use crate::models::{BloodPressureReading, DebuggerTrace, ProcessingError};
pub mod debug;
mod digit_extractor;
mod lcd_number_extractor;
mod lcd_screen_extractor;
pub mod models;
mod rectangle;

pub struct BloodPressureReadingExtractor<T: BpmOcrDebugOutputter> {
    screen_extractor: LcdScreenExtractor<T>,
    screen_number_extractor: LcdNumberExtractor<T>,
    debugging_session: DebuggerTrace<T>,
}

impl<T: BpmOcrDebugOutputter> BloodPressureReadingExtractor<T> {
    pub fn new(debugger_session: DebuggerTrace<T>) -> Self {
        let screen_extractor = LcdScreenExtractor::new(
            Arc::clone(&debugger_session.debugger),
            &debugger_session.unique_trace_name,
        );
        let screen_number_extractor = LcdNumberExtractor::new(
            Arc::clone(&debugger_session.debugger),
            &debugger_session.unique_trace_name,
        );

        BloodPressureReadingExtractor {
            screen_extractor,
            screen_number_extractor,
            debugging_session: debugger_session,
        }
    }

    fn process_image(self: &Self, image: &Mat) -> Result<BloodPressureReading, ProcessingError> {
        self.debugging_session
            .debugger
            .debug_original_picture(&self.debugging_session.unique_trace_name, &image)?;

        let mut resized_image = Mat::default();

        let interpolation: i32 = 0;
        imgproc::resize(
            &image,
            &mut resized_image,
            Size::new(800, 800),
            0.,
            0.,
            interpolation,
        )?;

        let birdseye_lcd_only = self.screen_extractor.extract_lcd(&resized_image)?;

        let reading = self
            .screen_number_extractor
            .extract_reading(&birdseye_lcd_only)?;

        Ok(reading)
    }
}

/// Attempts to extract a blood pressure reading from a photo file of a blood pressure monitor screen
/// * `filename` - the path to the photo file
/// * `debugger` - the debugger trace session to output debug images with
pub fn get_reading_from_file<T: BpmOcrDebugOutputter>(
    filename: &str,
    debugger: DebuggerTrace<T>,
) -> Result<BloodPressureReading, ProcessingError> {
    let extractor: BloodPressureReadingExtractor<T> = BloodPressureReadingExtractor::new(debugger);

    let gray_scale_mode: i32 = ImreadModes::IMREAD_GRAYSCALE.into();
    let image = imgcodecs::imread(filename, gray_scale_mode)?;

    extractor.process_image(&image)
}

/// Attempts to extract a blood pressure reading from a byte buffer containing a photo file of a blood pressure monitor screen
/// * `filename` - the byte buffer with the photo file
/// * `debugger` - the debugger trace session to output debug images with
pub fn get_reading_from_buffer<T: BpmOcrDebugOutputter>(
    file_contents: Vec<u8>,
    debugger: DebuggerTrace<T>,
) -> Result<BloodPressureReading, ProcessingError> {
    let extractor: BloodPressureReadingExtractor<T> = BloodPressureReadingExtractor::new(debugger);

    let contents = Vector::from_slice(&file_contents);
    let image = imgcodecs::imdecode(&contents, ImreadModes::IMREAD_GRAYSCALE.into())?;

    extractor.process_image(&image)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::debug::TempFolderDebugger;

    #[test]
    fn test_success_photo_at_angle() {
        let debug_session: DebuggerTrace<TempFolderDebugger> =
            DebuggerTrace::temp_folder_session("test_success");

        let testfile = Vec::from(include_bytes!("./test_resources/example_at_angle.jpg"));

        let expected_result = BloodPressureReading {
            systolic: 133,
            diastolic: 93,
            pulse: 65,
        };

        let result = get_reading_from_buffer(testfile, debug_session).unwrap();

        assert_eq!(result, expected_result);
    }

    #[test]
    fn test_success_topdown_photo() {
        let debug_session: DebuggerTrace<TempFolderDebugger> =
            DebuggerTrace::temp_folder_session("test_topdown_photo");

        let testfile = Vec::from(include_bytes!("./test_resources/example_top_down.jpg"));

        let expected_result = BloodPressureReading {
            systolic: 131,
            diastolic: 88,
            pulse: 77,
        };

        let result = get_reading_from_buffer(testfile, debug_session).unwrap();

        assert_eq!(result, expected_result);
    }

    #[test]
    fn test_with_2_digit() {
        let debug_session: DebuggerTrace<TempFolderDebugger> =
            DebuggerTrace::temp_folder_session("contour_candidates");

        let testfile = Vec::from(include_bytes!("./test_resources/contour_candidates.jpeg"));

        let expected_result = BloodPressureReading {
            systolic: 123,
            diastolic: 85,
            pulse: 68,
        };

        let result = get_reading_from_buffer(testfile, debug_session).unwrap();

        assert_eq!(result, expected_result);
    }
}
