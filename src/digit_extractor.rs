use opencv::{
    Error,
    core::{Mat, MatTraitConst, Point, Rect2i, count_non_zero},
};

use crate::models::ProcessingError;

static SEGMENTS_TO_NUMBER_MAP: [([i32; 7], i32); 10] = [
    ([1, 1, 1, 0, 1, 1, 1], 0),
    ([0, 0, 1, 0, 0, 1, 0], 1),
    ([1, 0, 1, 1, 1, 0, 1], 2),
    ([1, 0, 1, 1, 0, 1, 1], 3),
    ([0, 1, 1, 1, 0, 1, 0], 4),
    ([1, 1, 0, 1, 0, 1, 1], 5),
    ([1, 1, 0, 1, 1, 1, 1], 6),
    //([1, 0, 1, 0, 0, 1, 0], 7),
    ([1, 1, 1, 0, 0, 1, 0], 7),
    ([1, 1, 1, 1, 1, 1, 1], 8),
    ([1, 1, 1, 1, 0, 1, 1], 9),
];

pub fn parse_digit(image: &Mat, full_digit_location: Rect2i) -> Result<i32, ProcessingError> {
    let focused_digit = image.roi(full_digit_location)?;
    let total_filled_in_area = count_non_zero(&focused_digit)?;
    let total_area = full_digit_location.area();

    let width_to_height_ratio = full_digit_location.width  as f32/ full_digit_location.height as f32;

    // If we're drawn a box around an area that's mostly filled in and its a thin width, then it's probably a 1
    if (total_filled_in_area as f32) / (total_area as f32) > 0.77 && width_to_height_ratio < 0.30 {
        return Ok(1);
    }

    let digit_width = ((full_digit_location.width as f32) * 0.25) as i32;
    let digit_height = ((full_digit_location.height as f32) * 0.15) as i32;
    let digit_height_centre = ((full_digit_location.height as f32) * 0.05) as i32;

    let segment_locations = [
        ((0, 0), (full_digit_location.width, digit_height)), // top row,
        ((0, 0), (digit_width, full_digit_location.height / 2)), // top left down to half,
        (
            (full_digit_location.width - digit_width, 0),
            (full_digit_location.width, full_digit_location.height / 2),
        ), // top right down to half
        (
            (0, (full_digit_location.height / 2) - digit_height_centre),
            (
                full_digit_location.width,
                (full_digit_location.height / 2) + digit_height_centre,
            ),
        ), // centre
        (
            (0, full_digit_location.height / 2),
            (digit_width, full_digit_location.height),
        ), // from centre to bottom left,
        (
            (
                full_digit_location.width - digit_width,
                full_digit_location.height / 2,
            ),
            (full_digit_location.width, full_digit_location.height),
        ), // from centre to bottom right
        (
            (0, full_digit_location.height - digit_height),
            (full_digit_location.width, full_digit_location.height),
        ), // bottom row
    ];

    let digit_segments_lit_up_result: [Result<i32, Error>; 7] =
        segment_locations.map(|segment_locations| {
            let ((x_a, y_a), (x_b, y_b)) = segment_locations;

            let rect: opencv::core::Rect_<i32> = Rect2i::from_points(
                Point::new(full_digit_location.x + x_a, full_digit_location.y + y_a),
                Point::new(full_digit_location.x + x_b, full_digit_location.y + y_b),
            );

            let focused_segment = image.roi(rect)?;

            let total_filled_in_area = count_non_zero(&focused_segment)?;

            if (total_filled_in_area as f32 / rect.area() as f32) > 0.55 {
                return Ok(1);
            } else {
                return Ok(0);
            }
        });

    let digit_segments_lit_up: Result<Vec<i32>, Error> =
        digit_segments_lit_up_result.into_iter().collect();

    let lit_up = &digit_segments_lit_up?;

    let result = SEGMENTS_TO_NUMBER_MAP.iter().find(|segments| {
        return lit_up
            .into_iter()
            .zip(segments.0.iter())
            .all(|(a, b)| *a == *b);
    });

    match result {
        Some(num) => Ok(num.1),
        None => Err(ProcessingError::AppError(
            crate::models::ReadingIdentificationError::CouldNotProcessSegments,
        )),
    }
}
