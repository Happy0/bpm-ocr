use crate::{
    debug::{self, debug_digits_after_dilation, debug_digits_before_morph},
    digit,
    models::{BloodPressureReading, ProcessingError, ReadingIdentificationError, ReadingLocations},
};
use opencv::{
    Error,
    core::{Mat, Point, Rect2i, Size, Vector},
    imgproc::{
        self, THRESH_BINARY_INV, THRESH_OTSU, bounding_rect, dilate_def, find_contours_def,
        get_structuring_element_def, threshold,
    },
};

fn highlight_digits(image: &Mat) -> Result<Mat, ProcessingError> {
    let mut thresholed_image = Mat::default();

    threshold(
        image,
        &mut thresholed_image,
        0.,
        255.,
        THRESH_BINARY_INV | THRESH_OTSU,
    )?;

    debug_digits_before_morph(&thresholed_image)?;

    let mut dilated_image = Mat::default();

    // Fill in the gaps in the middle of the digits on the LCD screen to make it easier to identify the full digit
    let dilation_kernel = get_structuring_element_def(imgproc::MORPH_RECT, Size::new(3, 3))?;
    dilate_def(&thresholed_image, &mut dilated_image, &dilation_kernel)?;

    debug_digits_after_dilation(&dilated_image)?;

    return Ok(dilated_image);
}

pub fn get_digit_borders(image: &Mat) -> Result<Vec<Rect2i>, ProcessingError> {
    let mut contours_output: Vector<Vector<Point>> = Vector::new();
    find_contours_def(
        image,
        &mut contours_output,
        imgproc::RETR_EXTERNAL,
        imgproc::CHAIN_APPROX_SIMPLE,
    )?;

    let predicted_digits: Vec<Rect2i> = contours_output
        .into_iter()
        .map(|contour| {
            return bounding_rect(&contour);
        })
        .filter(|possible_digit| match possible_digit {
            Ok(rect) => rect.y != 0 && rect.x != 0 && rect.height > 30,
            _ => true, // Make sure errors are propagated
        })
        .collect::<Result<Vec<Rect2i>, Error>>()?;

    return Ok(predicted_digits);
}

fn group_by_similar_y_coordinate(
    digits: Vec<Rect2i>,
    difference_threshold: i32,
) -> Vec<Vec<Rect2i>> {
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

pub fn get_reading_locations(mut digits: Vec<Rect2i>) -> Result<ReadingLocations, ProcessingError> {
    // Sort digits by their row

    digits.sort_by(|vec1, vec2| vec1.y.cmp(&vec2.y));

    let mut grouped_by_y_coordinate: Vec<Vec<Rect2i>> = group_by_similar_y_coordinate(digits, 5);

    // Sort numbers by their columns
    for group in grouped_by_y_coordinate.iter_mut() {
        group.sort_by(|item1, item2| item1.x.cmp(&item2.x));
    }

    match (
        grouped_by_y_coordinate.pop(),
        grouped_by_y_coordinate.pop(),
        grouped_by_y_coordinate.pop(),
        grouped_by_y_coordinate.pop(),
    ) {
        (Some(pulse), Some(diastolic), Some(systolic), None) => {
            return Ok(ReadingLocations {
                systolic_region: systolic,
                diastolic_region: diastolic,
                pulse_region: pulse,
            });
        }
        _ => {
            return Err(ProcessingError::AppError(
                crate::models::ReadingIdentificationError::UnexpectedNumberOfRows,
            ));
        }
    }
}

fn digits_to_number(image: &Mat, digits: Vec<Rect2i>) -> Result<i32, ProcessingError> {
    let mut result: i32 = 0;
    for (index, digit) in digits.iter().enumerate() {
        let digit_result: i32 = digit::parse_digit(&image, *digit)?;
        let multiplier: u32 = (digits.len() - (index + 1)).try_into().map_err(|x| {
            ProcessingError::AppError(ReadingIdentificationError::InternalError(
                "Unexpected number conversion issue",
            ))
        })?;

        let ten: i32 = 10;
        result = result + (digit_result * (ten.pow(multiplier)));
    }

    Ok(result)
}

pub fn extract_reading(image: &Mat) -> Result<BloodPressureReading, ProcessingError> {
    let highlighted_digits = highlight_digits(image)?;

    let digit_borders = get_digit_borders(&highlighted_digits)?;

    debug::debug_digit_locations(&highlighted_digits, &digit_borders)?;

    let reading_locations = get_reading_locations(digit_borders)?;

    let systolic_result = digits_to_number(&highlighted_digits, reading_locations.systolic_region)?;
    let diastolic_result =
        digits_to_number(&highlighted_digits, reading_locations.diastolic_region)?;
    let pulse_result = digits_to_number(&highlighted_digits, reading_locations.pulse_region)?;

    let blood_pressure_reading = BloodPressureReading {
        systolic: systolic_result,
        diastolic: diastolic_result,
        pulse: pulse_result,
    };

    return Ok(blood_pressure_reading);
}
