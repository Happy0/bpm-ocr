use imageproc::drawing::{draw_polygon, draw_polygon_mut};
use opencv::imgproc::{approx_poly_dp, arc_length, draw_contours, fill_poly_def, rectangle_def};
use opencv::{imgcodecs, imgproc};
use opencv::imgcodecs::ImreadModes;
use opencv::core::{Mat, MatTraitConstManual, Point, Rect, Size, UMat, Vector, VectorToVec};
use opencv::Error;
use opencv::highgui;

pub struct BloodPressureReading {
    systolic: u8,
    diastolic: u8,
    pulse: u8
}

#[derive(Clone, Debug)]
struct LedScreenCandidate {
    coordinates: Vector<Point>,
    area: f64
}

fn get_led_candidate_points(contour: &Vector<Point>) -> Result<Option<LedScreenCandidate>,Error> {
    let mut approx_curv_output: Vector<Point> = Vector::new();

    let perimeter = arc_length(&contour, true)?;

    approx_poly_dp(&contour, &mut approx_curv_output, 0.02 * perimeter,true)?;

    if (approx_curv_output.len() == 4) {
        let area = imgproc::contour_area(&approx_curv_output, true)?;

        let result = LedScreenCandidate {
            coordinates: approx_curv_output,
            area: area
        };

        return Ok(Some(result));
    } else {
        return Ok(None)
    }
}

fn get_led_candidates(contours: &Vector<Vector<Point>>) -> Result<Vec<LedScreenCandidate>, Error> {

    let candidate_results: Vec<Result<Option<LedScreenCandidate>,Error>> = 
        contours.to_vec().iter().map(|points|  get_led_candidate_points(points))
        .collect();

    let candidates_or_error: Result<Vec<Option<LedScreenCandidate>>, Error> = candidate_results.into_iter().collect();

    let candidates = candidates_or_error?;

    let result: Vec<LedScreenCandidate> = candidates.iter()
        .into_iter()
        .filter_map(Option::as_ref)
        .cloned()
        .collect();
    
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
    
    let mut led_candidates = get_led_candidates(&contours_output)?;
    led_candidates.sort_by(|a1,a2| a1.area.total_cmp(&a2.area) );

    println!("Num candidates: {0}", led_candidates.len());

    let best_candidate_led = led_candidates.get(0);

    match best_candidate_led {
        Some(candidate) => {
            fill_poly_def(&mut resized_image, &candidate.coordinates, (255,0,0).into())?
        }
        None => println!("Aw naw, nae candidates")
    }


    // for led_candidate in led_candidates {
        

    //     fill_poly_def(&mut resized_image, &led_candidate, (255,0,0).into())?
    // }


    highgui::imshow("testaroonie", &resized_image);
    let x = highgui::wait_key(0)?;
    highgui::destroy_all_windows()
}