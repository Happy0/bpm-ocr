use opencv::{
    Error,
    core::{Point, Rect2i, Vector},
};

#[derive(Clone, Debug)]
pub enum ReadingIdentificationError {
    InternalError(&'static str),
    CouldNotIdentifyReadings,
    CouldNotIdentityLCDCandidate,
    UnexpectedNumberOfRows,
    CouldNotProcessSegments,
}

#[derive(Clone, Debug)]
pub struct RejectedLcdScreenCandidate {
    pub contour: Vector<Point>,
}

#[derive(Clone, Debug)]
pub struct LcdScreenCandidate {
    pub coordinates: Vector<Point>,
    pub area: f64,
    pub contour: Vector<Point>,
}

pub enum LcdScreenCandidateResult {
    Success(LcdScreenCandidate),
    Failure(RejectedLcdScreenCandidate),
}

#[derive(Debug)]
pub enum ProcessingError {
    ImageDetectionLibraryError(Error),
    AppError(ReadingIdentificationError),
}

impl From<Error> for ProcessingError {
    fn from(error: Error) -> Self {
        return Self::ImageDetectionLibraryError(error);
    }
}

#[derive(Clone, Debug)]
pub(crate) struct RectangleCoordinates {
    pub top_left: Point,
    pub top_right: Point,
    pub bottom_left: Point,
    pub bottom_right: Point,
}

#[derive(Clone, Debug)]
pub(crate) struct ReadingLocations {
    pub systolic_region: Vec<Rect2i>,
    pub diastolic_region: Vec<Rect2i>,
    pub pulse_region: Vec<Rect2i>,
}

#[derive(Clone, Debug)]
pub struct BloodPressureReading {
    pub systolic: i32,
    pub diastolic: i32,
    pub pulse: i32,
}
