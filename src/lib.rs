use std::rc::Rc;

use opencv::core::{Mat, Size, Vector};
use opencv::imgcodecs::ImreadModes;
use opencv::{imgcodecs, imgproc};

use crate::debug::BpmOcrDebugOutputter;
use crate::lcd_number_extractor::LcdNumberExtractor;
use crate::lcd_screen_extractor::LcdScreenExtractor;
use crate::models::{BloodPressureReading, ProcessingError};
pub mod debug;
mod digit_extractor;
mod lcd_number_extractor;
mod lcd_screen_extractor;
pub mod models;
mod rectangle;

pub struct BloodPressureReadingExtractor<T: BpmOcrDebugOutputter> {
    screen_extractor: LcdScreenExtractor<T>,
    screen_number_extractor: LcdNumberExtractor<T>,
}

impl<T: BpmOcrDebugOutputter> BloodPressureReadingExtractor<T> {
    pub fn new(debugger: T) -> Self {
        let shared_debugger = Rc::new(debugger);

        let screen_extractor = LcdScreenExtractor::new(&shared_debugger);
        let screen_number_extractor = LcdNumberExtractor::new(&shared_debugger);

        BloodPressureReadingExtractor {
            screen_extractor,
            screen_number_extractor,
        }
    }

    fn process_image(self: &Self, image: &Mat) -> Result<BloodPressureReading, ProcessingError> {
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

    pub fn get_reading_from_file(
        self: &Self,
        filename: &str,
    ) -> Result<BloodPressureReading, ProcessingError> {
        let gray_scale_mode: i32 = ImreadModes::IMREAD_GRAYSCALE.into();
        let image = imgcodecs::imread(filename, gray_scale_mode)?;

        self.process_image(&image)
    }

    pub fn get_reading_from_buffer(
        self: &Self,
        file_contents: Vec<u8>,
    ) -> Result<BloodPressureReading, ProcessingError> {
        let contents = Vector::from_slice(&file_contents);
        let image = imgcodecs::imdecode(&contents, ImreadModes::IMREAD_GRAYSCALE.into())?;

        self.process_image(&image)
    }
}
