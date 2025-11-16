use opencv::Error;

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

pub struct BloodPressureReading {
    systolic: u8,
    diastolic: u8,
    pulse: u8
}

