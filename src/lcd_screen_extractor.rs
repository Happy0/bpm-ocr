use std::{cmp::max, sync::Arc};

use opencv::{
    Error,
    core::{Mat, Point, Point2f, Size, UMat, Vector, VectorToVec},
    imgproc::{
        self, approx_poly_dp, arc_length, get_perspective_transform_def, warp_perspective_def,
    },
};

use crate::{
    debug::BpmOcrDebugOutputter,
    models::{
        self, LcdScreenCandidate, LcdScreenCandidateResult, ProcessingError,
        ReadingIdentificationError, RejectedLcdScreenCandidate,
    },
    rectangle::get_rectangle_coordinates,
};

pub(crate) struct LcdScreenExtractor<T: BpmOcrDebugOutputter> {
    debugger: Arc<T>,
    debug_session_name: String,
}

impl<T: BpmOcrDebugOutputter> LcdScreenExtractor<T> {
    pub fn new(debugger: Arc<T>, debug_session_name: &str) -> Self {
        LcdScreenExtractor {
            debugger: debugger,
            debug_session_name: debug_session_name.to_owned(),
        }
    }

    fn get_lcd_candidate_points(
        self: &Self,
        contour: Vector<Point>,
    ) -> Result<LcdScreenCandidateResult, Error> {
        let mut approx_curv_output: Vector<Point> = Vector::new();

        let perimeter = arc_length(&contour, true)?;

        approx_poly_dp(&contour, &mut approx_curv_output, 0.02 * perimeter, true)?;

        if approx_curv_output.len() == 4 {
            let area = imgproc::contour_area(&approx_curv_output, true)?;

            let result = LcdScreenCandidate {
                coordinates: approx_curv_output,
                area: area,
                contour: contour,
            };

            return Ok(LcdScreenCandidateResult::Success(result));
        } else {
            return Ok(LcdScreenCandidateResult::Failure(
                RejectedLcdScreenCandidate { contour: contour },
            ));
        }
    }

    fn partition_candidates(
        self: &Self,
        results: Vec<LcdScreenCandidateResult>,
    ) -> (Vec<LcdScreenCandidate>, Vec<RejectedLcdScreenCandidate>) {
        let mut lcd_screen_candidates: Vec<LcdScreenCandidate> = Vec::new();
        let mut rejected_screen_candidates: Vec<RejectedLcdScreenCandidate> = Vec::new();

        for result in results.into_iter() {
            match result {
                LcdScreenCandidateResult::Failure(x) => rejected_screen_candidates.push(x),
                LcdScreenCandidateResult::Success(x) => lcd_screen_candidates.push(x),
            }
        }

        (lcd_screen_candidates, rejected_screen_candidates)
    }

    fn extract_lcd_birdseye_view(
        self: &Self,
        image: &Mat,
        led_coordinates: models::RectangleCoordinates,
    ) -> Result<Mat, ProcessingError> {
        let width_bottom = ((led_coordinates.bottom_right.x - led_coordinates.bottom_left.x)
            .pow(2)
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

        self.debugger
            .debug_after_perspective_transform(&self.debug_session_name, &dest_image)?;

        Ok(dest_image)
    }

    fn get_lcd_candidates(
        self: &Self,
        image_blurred: &Mat,
        contours: Vector<Vector<Point>>,
    ) -> Result<Vec<LcdScreenCandidate>, ProcessingError> {
        let candidate_results: Vec<Result<LcdScreenCandidateResult, Error>> = contours
            .to_vec()
            .into_iter()
            .map(|points| self.get_lcd_candidate_points(points))
            .collect();

        let candidates_or_error: Result<Vec<LcdScreenCandidateResult>, Error> =
            candidate_results.into_iter().collect();

        let candidates = candidates_or_error?;
        let (success_candidates, failure_candidates) = self.partition_candidates(candidates);

        let _ = &self.debugger.debug_lcd_contour_candidates(
            &self.debug_session_name,
            &image_blurred,
            &success_candidates,
            failure_candidates,
        )?;

        Ok(success_candidates)
    }

    pub fn extract_lcd(&self, resized_image: &Mat) -> Result<Mat, ProcessingError> {
        let mut blurred = Mat::default();
        imgproc::gaussian_blur_def(&resized_image, &mut blurred, Size::new(5, 5), 0.0)?;

        let mut edges = UMat::new_def();
        imgproc::canny_def(&blurred, &mut edges, 50., 200.)?;

        self.debugger
            .debug_after_canny(&self.debug_session_name, &edges)?;

        let mut contours_output: Vector<Vector<Point>> = Vector::new();
        imgproc::find_contours(
            &edges,
            &mut contours_output,
            imgproc::RETR_EXTERNAL,
            imgproc::CHAIN_APPROX_SIMPLE,
            Point::new(0, 0),
        )?;

        let mut led_candidates = self.get_lcd_candidates(&blurred, contours_output)?;
        led_candidates.sort_by(|a1, a2| a1.area.total_cmp(&a2.area));

        let best_candidate_led: &LcdScreenCandidate = led_candidates.get(0).ok_or_else(|| {
            ProcessingError::AppError(ReadingIdentificationError::CouldNotIdentityLCDCandidate)
        })?;

        let lcd_coordinates = get_rectangle_coordinates(&best_candidate_led.coordinates).ok_or(
            ProcessingError::AppError(models::ReadingIdentificationError::InternalError(
                "Internal error: LCD candidate did not have 4 points as expected",
            )),
        )?;

        self.extract_lcd_birdseye_view(&resized_image, lcd_coordinates)
    }
}
