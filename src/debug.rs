use std::{env, fs::create_dir_all};

use chrono::{self, Datelike, Timelike};
use opencv::{
    core::{AccessFlag, CV_8U, Mat, Rect2i, Scalar, UMat, UMatTraitConst},
    imgcodecs::imwrite_def,
    imgproc::{cvt_color_def, rectangle_def},
};

use crate::models::{ProcessingError, ReadingIdentificationError};

fn get_debug_filepath(filename: &str) -> Result<String, ProcessingError> {
    let now = chrono::offset::Local::now();

    let mut folder_path = env::temp_dir();

    folder_path = folder_path.join("bmp-ocr");
    folder_path = folder_path.join(now.format("%Y-%m-%d-%H-%M-%S").to_string());

    create_dir_all(&folder_path).map_err(|_| {
        ProcessingError::AppError(ReadingIdentificationError::InternalError(
            "Could not create a temporary folder for debugging image processing",
        ))
    })?;

    folder_path = folder_path.join(filename);

    folder_path
        .to_str()
        .ok_or(ProcessingError::AppError(
            ReadingIdentificationError::InternalError(
                "Could not create a name for a temporary folder for debugging image processing",
            ),
        ))
        .map(|x| x.to_string())
}

fn write_file(image: &Mat, file_name: &str) -> Result<(), ProcessingError> {
    let file_path = get_debug_filepath(&file_name)?;
    imwrite_def(&file_path, &image)?;

    Ok(())
}

pub fn debug_enabled() -> bool {
    env::var("DEBUG_BPM_OCR")
        .map(|value| value.to_ascii_lowercase() == "true")
        .unwrap_or(false)
}

pub fn debug_after_canny(image: &UMat) -> Result<(), ProcessingError> {
    if !debug_enabled() {
        return Ok(());
    }

    let converted_to_mat = image.get_mat(AccessFlag::ACCESS_READ)?;

    write_file(&converted_to_mat, "after_canny.jpeg")
}

pub fn debug_after_perspective_transform(image: &Mat) -> Result<(), ProcessingError> {
    if !debug_enabled() {
        return Ok(());
    }

    write_file(&image, "after_perspective_transform.jpeg")
}

pub fn debug_digits_before_morph(image: &Mat) -> Result<(), ProcessingError> {
    if !debug_enabled() {
        return Ok(());
    }

    write_file(image, "digits_before_morph.jpeg")
}

pub fn debug_digits_after_dilation(image: &Mat) -> Result<(), ProcessingError> {
    if !debug_enabled() {
        return Ok(());
    }

    write_file(&image, "digits_after_dilation.jpeg")
}

pub fn debug_digit_locations(
    image: &Mat,
    digit_locations: &Vec<Rect2i>,
) -> Result<(), ProcessingError> {
    if !debug_enabled() {
        return Ok(());
    }

    let mut temp_image = Mat::default();
    cvt_color_def(&image, &mut temp_image, CV_8U)?;

    for b in digit_locations {
        rectangle_def(&mut temp_image, *b, Scalar::new(0.0, 255.0, 0.0, 0.0))?;
    }

    write_file(&temp_image, "digit_locations.jpeg")
}
