use std::{env, fs::create_dir_all};

use chrono::{self, Datelike, Timelike};
use opencv::{
    core::{
        ACCESS_READ, AccessFlag, CV_8U, Mat, Point, Rect2i, Scalar, UMat, UMatTraitConst, Vector,
    },
    imgcodecs::imwrite_def,
    imgproc::{
        COLOR_BGR2GRAY, COLOR_GRAY2RGB, LINE_8, approx_poly_dp, arc_length, cvt_color,
        cvt_color_def, draw_contours, draw_contours_def, rectangle, rectangle_def,
    },
};

use crate::{
    models::{
        self, LcdScreenCandidate, ProcessingError, ReadingIdentificationError,
        RejectedLcdScreenCandidate,
    },
    rectangle::get_rectangle_coordinates,
};

fn get_debug_filepath(filename: &str) -> Result<String, ProcessingError> {
    let now = chrono::offset::Local::now();

    let mut folder_path = env::temp_dir()
        .join("bmp-ocr")
        .join(now.format("%Y-%m-%d-%H-%M-%S").to_string());

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

pub fn debug_lcd_contour_candidates(
    image: &Mat,
    candidates: &Vec<LcdScreenCandidate>,
    rejections: Vec<RejectedLcdScreenCandidate>,
) -> Result<(), ProcessingError> {
    if !debug_enabled() {
        return Ok(());
    }

    let mut colour: Mat = Mat::default();

    cvt_color(&image, &mut colour, COLOR_GRAY2RGB, 0)?;

    for rejection in rejections {
        let mut x: Vector<Vector<Point>> = Vector::new();
        x.push(rejection.contour);

        draw_contours(
            &mut colour,
            &x,
            0,
            Scalar::new(255.0, 0.0, 0.0, 0.1),
            1,
            LINE_8.into(),
            &Mat::default(),
            i32::MAX,
            Point::default(),
        )?;
    }

    for candidate in candidates {
        println!("{:?}", candidate.coordinates);

        let rectangle_coordinates = get_rectangle_coordinates(&candidate.coordinates).ok_or(
            ProcessingError::AppError(models::ReadingIdentificationError::InternalError(
                "Internal error: LCD candidate did not have 4 points as expected",
            )),
        )?;

        let rect = Rect2i::from_points(
            rectangle_coordinates.top_left,
            rectangle_coordinates.bottom_right,
        );

        rectangle_def(&mut colour, rect, Scalar::new(0., 255.0, 0.0, 0.1))?
    }

    write_file(&colour, "contour_candidates.jpeg")
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
