use std::cmp::max;

use opencv::Error;
use opencv::core::{Mat, Point, Point_, Point2f, Size, UMat, Vector, VectorToVec};
use opencv::imgcodecs::ImreadModes;
use opencv::imgproc::{
    approx_poly_dp, arc_length, get_perspective_transform_def, warp_perspective_def,
};
use opencv::{imgcodecs, imgproc};

use crate::debug::{debug_after_canny, debug_after_perspective_transform, debug_lcd_contour_candidates};
use crate::lcd_number_extractor::extract_reading;
use crate::models::{BloodPressureReading, LcdScreenCandidate, LcdScreenCandidateResult, ProcessingError, ReadingIdentificationError, RejectedLcdScreenCandidate};
mod debug;
mod digit;
mod lcd_number_extractor;
mod models;

fn get_lcd_candidate_points(contour: &Vector<Point>) -> Result<LcdScreenCandidateResult, Error> {
    let mut approx_curv_output: Vector<Point> = Vector::new();

    let perimeter = arc_length(&contour, true)?;

    approx_poly_dp(&contour, &mut approx_curv_output, 0.02 * perimeter, true)?;

    if approx_curv_output.len() == 4 {
        let area = imgproc::contour_area(&approx_curv_output, true)?;

        let result = LcdScreenCandidate {
            coordinates: approx_curv_output,
            area: area,
        };

        return Ok(LcdScreenCandidateResult::Success(result));
    } else {
        return Ok(LcdScreenCandidateResult::Failure(RejectedLcdScreenCandidate { contour: contour.clone() }))
    }
}

fn partition_candidates(results: Vec<LcdScreenCandidateResult>) -> (Vec<LcdScreenCandidate>, Vec<RejectedLcdScreenCandidate>) {
    let mut lcd_screen_candidates: Vec<LcdScreenCandidate> = Vec::new();
    let mut rejected_screen_candidates: Vec<RejectedLcdScreenCandidate> = Vec::new();

    for result in results.into_iter() {

        match result {
            LcdScreenCandidateResult::Failure(x) => rejected_screen_candidates.push(x),
            LcdScreenCandidateResult::Success(x) => lcd_screen_candidates.push(x)
        }

    }

    (lcd_screen_candidates, rejected_screen_candidates)
}

// Looks for rectangle shapes in the image which could be the LCD screen
fn get_lcd_candidates(contours: Vector<Vector<Point>>) -> Result<Vec<LcdScreenCandidate>, ProcessingError> {
    let candidate_results: Vec<Result<LcdScreenCandidateResult, Error>> = contours
        .to_vec()
        .iter()
        .map(|points| {
            get_lcd_candidate_points(points)
        })
        .collect();

    let candidates_or_error: Result<Vec<LcdScreenCandidateResult>, Error> =
        candidate_results.into_iter().collect();

    let candidates = candidates_or_error?;
    let (success_candidates, failure_candidates) = partition_candidates(candidates);

    debug_lcd_contour_candidates(&success_candidates, failure_candidates)?;

    Ok(success_candidates)
}

// Extracts only the LCD screen and transforms the image to a top down view of it
fn extract_lcd_birdseye_view(
    image: &Mat,
    led_coordinates: models::RectangleCoordinates,
) -> Result<Mat, ProcessingError> {
    let width_bottom = ((led_coordinates.bottom_right.x - led_coordinates.bottom_left.x).pow(2)
        + (led_coordinates.bottom_right.y - led_coordinates.bottom_left.y).pow(2))
    .isqrt();

    let width_top = ((led_coordinates.top_right.x - led_coordinates.top_left.x).pow(2)
        + (led_coordinates.top_right.y - led_coordinates.top_left.y).pow(2))
    .isqrt();

    let max_width = max(width_bottom, width_top);

    let height_bottom = ((led_coordinates.top_right.x - led_coordinates.bottom_right.x).pow(2)
        + (led_coordinates.top_right.y - led_coordinates.bottom_right.y).pow(2))
    .isqrt();

    let height_top = ((led_coordinates.top_left.x - led_coordinates.bottom_left.x).pow(2)
        + (led_coordinates.top_left.y - led_coordinates.bottom_left.y).pow(2))
    .isqrt();

    let max_height = max(height_bottom, height_top);

    let src_points: Vector<Point2f> = Vector::from_slice(&[
        Point2f::new(
            led_coordinates.top_left.x as f32,
            led_coordinates.top_left.y as f32,
        ),
        Point2f::new(
            led_coordinates.top_right.x as f32,
            led_coordinates.top_right.y as f32,
        ),
        Point2f::new(
            led_coordinates.bottom_right.x as f32,
            led_coordinates.bottom_right.y as f32,
        ),
        Point2f::new(
            led_coordinates.bottom_left.x as f32,
            led_coordinates.bottom_left.y as f32,
        ),
    ]);

    let dest_points: Vector<Point2f> = Vector::from_slice(&[
        Point2f::new(0., 0.),
        Point2f::new(max_width as f32 - 1.0, 0.),
        Point2f::new(max_width as f32 - 1.0, max_height as f32 - 1.0),
        Point2f::new(0., max_height as f32 - 1.0),
    ]);

    let src_points_mat = Mat::from_slice(src_points.as_slice())?;
    let dest_points_mat = Mat::from_slice(dest_points.as_slice())?;

    let M = get_perspective_transform_def(&src_points_mat, &dest_points_mat)?;

    let mut dest_image = Mat::default();

    warp_perspective_def(
        &image,
        &mut dest_image,
        &M,
        Size::new(max_width, max_height),
    )?;

    debug_after_perspective_transform(&dest_image)?;

    Ok(dest_image)
}

fn locate_corners(points: (Point, Point, Point, Point)) -> models::RectangleCoordinates {
    let (p1, p2, p3, p4) = points;
    let mut point_array = [p1, p2, p3, p4];

    point_array.sort_by(|point1, point2| (point1.x + point1.y).cmp(&(point2.x + point2.y)));

    match point_array {
        [p1, p2, p3, p4] => {
            let top_left = p1;
            let bottom_right = p4;
            let (bottom_left, top_right) = if p2.x < p3.x { (p2, p3) } else { (p3, p2) };

            return models::RectangleCoordinates {
                top_left,
                top_right,
                bottom_left,
                bottom_right,
            };
        }
    }
}

fn get_rectangle_coordinates(
    coordinates: &Vector<Point>,
) -> Result<models::RectangleCoordinates, ReadingIdentificationError> {
    match coordinates.as_slice() {
        [p1, p2, p3, p4] => {
            let coordinates = locate_corners((*p1, *p2, *p3, *p4));

            Ok(coordinates)
        }
        _ => Err(models::ReadingIdentificationError::InternalError(
            "Internal error: LCD candidate did not have 4 points as expected",
        )),
    }
}

fn process_image(image: &Mat) -> Result<BloodPressureReading, ProcessingError> {
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

    let mut blurred = Mat::default();
    imgproc::gaussian_blur_def(&resized_image, &mut blurred, Size::new(5, 5), 0.0)?;

    let mut edges = UMat::new_def();
    imgproc::canny_def(&blurred, &mut edges, 50., 200.)?;

    debug_after_canny(&edges)?;

    let mut contours_output: Vector<Vector<Point>> = Vector::new();
    imgproc::find_contours(
        &edges,
        &mut contours_output,
        imgproc::RETR_EXTERNAL,
        imgproc::CHAIN_APPROX_SIMPLE,
        Point::new(0, 0),
    )?;

    let mut led_candidates = get_lcd_candidates(contours_output)?;
    led_candidates.sort_by(|a1, a2| a1.area.total_cmp(&a2.area));

    let best_candidate_led: &LcdScreenCandidate = led_candidates.get(0).ok_or_else(|| {
        ProcessingError::AppError(ReadingIdentificationError::CouldNotIdentityLCDCandidate)
    })?;

    let lcd_coordinates = get_rectangle_coordinates(&best_candidate_led.coordinates)
        .map_err(ProcessingError::AppError)?;

    let birdseye_lcd_only = extract_lcd_birdseye_view(&resized_image, lcd_coordinates)?;
    let reading = extract_reading(&birdseye_lcd_only)?;

    Ok(reading)
}

pub fn get_reading_from_file(filename: &str) -> Result<BloodPressureReading, ProcessingError> {
    let gray_scale_mode: i32 = ImreadModes::IMREAD_GRAYSCALE.into();
    let image = imgcodecs::imread(filename, gray_scale_mode)?;

    process_image(&image)
}

pub fn get_reading_from_buffer(
    file_contents: Vec<u8>,
) -> Result<BloodPressureReading, ProcessingError> {
    let contents = Vector::from_slice(&file_contents);
    let image = imgcodecs::imdecode(&contents, ImreadModes::IMREAD_GRAYSCALE.into())?;

    process_image(&image)
}
