use std::sync::Arc;

use opencv::{
    Error,
    core::{Point, Rect2i, Vector},
};
use uuid::Uuid;

use crate::debug::{BpmOcrDebugOutputter, NoDebug, TempFolderDebugger};

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

#[derive(Clone, Debug, PartialEq)]
pub struct BloodPressureReading {
    pub systolic: i32,
    pub diastolic: i32,
    pub pulse: i32,
}

pub struct DebuggerTrace<T: BpmOcrDebugOutputter> {
    pub unique_trace_name: String,
    pub debugger: Arc<T>,
}

impl DebuggerTrace<NoDebug> {
    pub fn no_debug_session() -> Self {
        DebuggerTrace {
            debugger: Arc::new(NoDebug {}),
            unique_trace_name: "".to_owned(),
        }
    }
}

impl DebuggerTrace<TempFolderDebugger> {
    pub fn temp_folder_session(unique_session_name: &str) -> Self {
        DebuggerTrace {
            debugger: Arc::new(TempFolderDebugger::new(true)),
            unique_trace_name: unique_session_name.to_owned(),
        }
    }

    pub fn temp_folder_session_uuid() -> Self {
        let uuid = Uuid::new_v4();
        DebuggerTrace {
            debugger: Arc::new(TempFolderDebugger::new(true)),
            unique_trace_name: uuid.to_string(),
        }
    }
}
