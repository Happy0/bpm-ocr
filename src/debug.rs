use std::{env, fs::create_dir_all, path::Path};

use chrono::{self, Datelike, Timelike};
use opencv::{Error, core::{CV_8U, Mat, Rect2i, Scalar}, highgui, imgcodecs::imwrite_def, imgproc::{cvt_color_def, rectangle_def}};

use crate::models::{ProblemIdentifyingReadings, ProcessingError};

fn get_debug_filepath(filename: &str) -> Option<String> {
    let temp_dir = env::temp_dir();
    let now = chrono::offset::Local::now();
    
    let mut folder_path = Path::new(&temp_dir).to_path_buf();

    folder_path = folder_path.join(format!("bmp-ocr"));
    folder_path = folder_path.join(format!("{}-{}-{}-{}-{}-{}", now.day(), now.month(), now.year(), now.hour(), now.minute(), now.second()));

    create_dir_all(&folder_path).unwrap();

    folder_path = folder_path.join(filename);

    let result = folder_path.as_os_str().to_str();

    result.map(|x| x.to_string())
}

pub fn debug_digit_locations(image: &Mat, digit_locations: &Vec<Rect2i>) -> Result<(), ProcessingError> {
    let mut temp_image = Mat::default();
    cvt_color_def(&image, &mut temp_image, CV_8U)?;

    for b in digit_locations {
        rectangle_def(&mut temp_image, *b, Scalar::new(0.0, 255.0, 0.0, 0.0))?;
    }
    
    let file_path = get_debug_filepath("digit_locations.jpg");

    match file_path {
        Some(path) => {
            imwrite_def(&path, &temp_image)?;
            Ok(())
        }
        None => {
            Err(ProcessingError::AppError(ProblemIdentifyingReadings::InternalError("Could not create a temporary image for debugging for digit locations".to_string())))
        }
    }
}