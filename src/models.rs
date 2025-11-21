use opencv::{Error, core::{Point, Rect2i}};

#[derive(Clone, Debug)]
pub enum ProblemIdentifyingReadings {
    InternalError(String),
    CouldNotIdentifyReadings,
    CouldNotIdentityLCDCandidate,
    UnexpectedNumberOfRows,
    CouldNotProcessSegments
}

#[derive(Debug)]
pub enum ProcessingError {
    ImageDetectionLibraryError(Error),
    AppError(ProblemIdentifyingReadings)
}

impl From<Error> for ProcessingError {
    fn from(error: Error) -> Self {
        return {
            Self::ImageDetectionLibraryError(error)
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct RectangleCoordinates {
    pub top_left: Point,
    pub top_right: Point,
    pub bottom_left: Point,
    pub bottom_right: Point
}

#[derive(Clone, Debug)]
pub(crate) struct ReadingLocations {
    pub systolic_region: Vec<Rect2i>,
    pub diastolic_region: Vec<Rect2i>,
    pub pulse_region: Vec<Rect2i>
}

pub struct BloodPressureReading {
    systolic: u8,
    diastolic: u8,
    pulse: u8
}

