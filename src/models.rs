use opencv::{Error, core::Point};

#[derive(Clone, Debug)]
pub enum ProblemIdentifyingReadings {
    InternalError(String),
    CouldNotIdentifyReadings,
    CouldNotIdentityLCDCandidate
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
pub struct RectangleCoordinates {
    pub topLeft: Point,
    pub topRight: Point,
    pub bottomLeft: Point,
    pub bottomRight: Point
}

pub struct BloodPressureReading {
    systolic: u8,
    diastolic: u8,
    pulse: u8
}

