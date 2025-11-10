use imageproc::drawing::{draw_polygon, draw_polygon_mut};
use opencv::imgproc::{approx_poly_dp, arc_length, draw_contours, fill_poly_def, rectangle_def};
use opencv::{imgcodecs, imgproc};
use opencv::imgcodecs::ImreadModes;
use opencv::core::{Mat, MatTraitConstManual, Point, Rect, Size, UMat, Vector};
use opencv::Error;
use opencv::highgui;

pub struct BloodPressureReading {
    systolic: u8,
    diastolic: u8,
    pulse: u8
}

fn get_led_candidate_points(contour: &Vector<Point>) -> Result<Option<Vector<Point>>,Error> {
    let mut approx_curv_output: Vector<Point> = Vector::new();

    let perimeter = arc_length(&contour, true)?;

    approx_poly_dp(&contour, &mut approx_curv_output, 0.02 * perimeter,true)?;

    if (approx_curv_output.len() == 4) {
        return Ok(Some(approx_curv_output));
    } else {
        return Ok(None)
    }
}

fn get_led_candidates(contours: &Vector<Vector<Point>>) -> Result<Vector<Vector<Point>>, Error> {
    let mut result: Vector<Vector<Point>> = Vector::new();
    for contour in contours {
        let led_candidate_points = get_led_candidate_points(&contour)?;

        match led_candidate_points {
            None => {},
            Some(points) => result.push(points)
        }
    }
    return Ok(result);
}


pub async fn get_reading_from_file(filename: &str) -> Result<(), Error> {

    let gray_scale_mode: i32 = ImreadModes::IMREAD_GRAYSCALE.into();
    let image = imgcodecs::imread(filename, gray_scale_mode)?;

    let mut resized_image = Mat::default();

    let interpolation: i32 = 0;
    imgproc::resize(&image, &mut resized_image, Size::new(800,800), 0.,0., interpolation)?;

    let mut blurred = Mat::default();
    imgproc::gaussian_blur_def(&resized_image, &mut blurred, Size::new(5,5), 0.0)?;

    let mut edges = UMat::new_def();
    imgproc::canny_def(&blurred, &mut edges, 50., 200.)?;

    let mut contours_output : Vector<Vector<Point>> = Vector::new();
    imgproc::find_contours(&edges, &mut contours_output, imgproc::RETR_EXTERNAL, imgproc::CHAIN_APPROX_SIMPLE, Point::new(0,0))?;

    let led_candidates = get_led_candidates(&contours_output)?;

    for led_candidate in led_candidates {
        fill_poly_def(&mut resized_image, &led_candidate, (255,0,0).into())?
    }



    highgui::imshow("testaroonie", &resized_image);
    let x = highgui::wait_key(0)?;
    highgui::destroy_all_windows()
}