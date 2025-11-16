use opencv::core::Mat;
use crate::models::{self, ProcessingError};

pub fn extract_readings(image: &Mat) -> Result<models::BloodPressureReading, ProcessingError> {
    panic!("panik")
}