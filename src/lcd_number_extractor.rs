
use std::{collections::HashMap, hash::Hash};

use opencv::{Error, core::{CV_8U, Mat, Point, Rect2i, Scalar, Size, Vector}, imgproc::{self, MORPH_ELLIPSE, MORPH_OPEN, THRESH_BINARY_INV, THRESH_OTSU, bounding_rect, cvt_color_def, dilate_def, find_contours_def, get_structuring_element_def, morphology_ex_def, rectangle_def, threshold}};
use crate::models::{ProcessingError, ReadingLocations};
use itertools::Itertools;

fn highlight_digits(image: &Mat) -> Result<Mat, ProcessingError> {
    let mut thresholed_image = Mat::default();

    threshold(image, &mut thresholed_image, 0., 255., THRESH_BINARY_INV | THRESH_OTSU )?;

    let kernel = get_structuring_element_def(MORPH_ELLIPSE, Size::new(1,5))?;

    let mut morphed_image = Mat::default();
    morphology_ex_def(&thresholed_image, &mut morphed_image, MORPH_OPEN, &kernel)?;

    let mut dilated_image = Mat::default();

    // Fill in the gaps in the middle of the digits on the LCD screen to make it easier to identify the full digit
    let dilation_kernel = get_structuring_element_def(imgproc::MORPH_RECT, Size::new(6,4))?;
    dilate_def(&morphed_image, &mut dilated_image, &dilation_kernel)?;

    return Ok(dilated_image);
}

pub fn get_digit_borders(image:  &Mat) -> Result<Vec<Rect2i>, ProcessingError> {
    let mut contours_output : Vector<Vector<Point>> = Vector::new();
    find_contours_def(image, &mut contours_output, imgproc::RETR_EXTERNAL, imgproc::CHAIN_APPROX_SIMPLE)?;

    let predicted_digits: Vec<Result<Rect2i, Error>> = contours_output.iter().map(|contour| {
        return bounding_rect(&contour);
    }).collect();

    let possible_digits :Result<Vec<Rect2i>,Error> = predicted_digits.into_iter().collect();

    let x = possible_digits?;

    // Filter out anything that is small enough to probably not be a digit
    let result: Vec<Rect2i> = x
        .iter()
        .filter(|possible_digit| possible_digit.height > 30 )
        .cloned()
        .collect();

    return Ok(result);
}

fn group_by_similar_y_coordinate(
    digits: Vec<Rect2i>,
    difference_threshold: i32,
) -> Vec<Vec<Rect2i>> 
{
    let mut groups: Vec<Vec<Rect2i>> = Vec::new();

    'outer: for digit in digits {

        for group in groups.iter_mut() {
            if let Some(leader) = group.first() {
                if (leader.y - digit.y).abs() < difference_threshold {
                    group.push(digit);
                    continue 'outer;
                }
            }
        }

        // No matching group found — make a new one
        groups.push(vec![digit]);
    }

    groups
}

pub fn get_reading_locations(digits: Vec<Rect2i>) -> Result<ReadingLocations, ProcessingError> {
    let mut sorted_digits = digits.clone();
    sorted_digits.sort_by(|vec1,vec2| vec1.y.cmp(&vec2.y));

    let grouped_by_y_coordinate: Vec<Vec<Rect2i>> = group_by_similar_y_coordinate(sorted_digits, 5);

    println!("{:?}", grouped_by_y_coordinate);

    match &grouped_by_y_coordinate.as_slice() {
        [sys, dia, pulse] => Ok(ReadingLocations{
            systolic_region: sys.clone(),
            diastolic_region: dia.clone(),
            pulse_region: pulse.clone()
        }),
        
        _ => {Err(ProcessingError::AppError(crate::models::ProblemIdentifyingReadings::UnexpectedNumberOfRows))}
    }
}

//pub fn extract_readings(image: &Mat) -> Result<Mat, BloodPressureReading> {
pub fn extract_readings(image: &Mat) -> Result<Mat, ProcessingError> {
    let highlighted_digits = highlight_digits(image)?;

    let digit_borders = get_digit_borders(&highlighted_digits)?;

    let reading_locations = get_reading_locations(digit_borders)?;

    println!("{:?}", &reading_locations);

    let mut temp_image = Mat::default();
    
    cvt_color_def(&highlighted_digits, &mut temp_image, CV_8U)?;

    // for b in digit_borders {
    //     rectangle_def(&mut temp_image, b, Scalar::new(0.0, 255.0, 0.0, 0.0))?
    // }

    // for c in contours {
    //     draw_contours_def(&mut temp_image, &c, -1, Scalar::new(0.0, 255.0, 0.0, 0.0)).unwrap();
    // }

    return Ok(temp_image);


}