use std::{env, fs::create_dir_all};

use opencv::{
    core::{AccessFlag, CV_8U, Mat, Point, Rect2i, Scalar, UMat, UMatTraitConst, Vector},
    imgcodecs::imwrite_def,
    imgproc::{COLOR_GRAY2RGB, LINE_8, cvt_color, cvt_color_def, draw_contours, rectangle_def},
};

use crate::models;

use models::{
    LcdScreenCandidate, ProcessingError, ReadingIdentificationError, RejectedLcdScreenCandidate,
};

pub struct TempFolderDebugger {
    folder_name: String,
    debug_enabled: bool,
}

pub struct NoDebug {}

pub trait BpmOcrDebugOutputter {
    fn new(unique_session_name: &str, debug_enabled: bool) -> Self;
    fn output(&self, image: &Mat, stage_description: &str) -> Result<(), ProcessingError>;
    fn debug_enabled(&self) -> bool;

    fn debug_after_canny(&self, image: &UMat) -> Result<(), ProcessingError> {
        if !self.debug_enabled() {
            return Ok(());
        }

        let converted_to_mat = image.get_mat(AccessFlag::ACCESS_READ)?;

        self.output(&converted_to_mat, "after_canny")
    }

    fn debug_lcd_contour_candidates(
        &self,
        image: &Mat,
        candidates: &Vec<LcdScreenCandidate>,
        rejections: Vec<RejectedLcdScreenCandidate>,
    ) -> Result<(), ProcessingError> {
        if !self.debug_enabled() {
            return Ok(());
        }

        let mut colour: Mat = Mat::default();

        cvt_color(&image, &mut colour, COLOR_GRAY2RGB, 0)?;

        for rejection in rejections {
            let mut x: Vector<Vector<Point>> = Vector::new();
            x.push(rejection.contour);

            draw_contours(
                &mut colour,
                &x,
                0,
                Scalar::new(255.0, 0.0, 0.0, 0.1),
                1,
                LINE_8.into(),
                &Mat::default(),
                i32::MAX,
                Point::default(),
            )?;
        }

        for candidate in candidates {
            let mut x: Vector<Vector<Point>> = Vector::new();
            x.push(candidate.contour.clone());

            draw_contours(
                &mut colour,
                &x,
                0,
                Scalar::new(0., 255.0, 0.0, 0.1),
                1,
                LINE_8.into(),
                &Mat::default(),
                i32::MAX,
                Point::default(),
            )?;
        }

        self.output(&colour, "contour_candidates")
    }

    fn debug_after_perspective_transform(&self, image: &Mat) -> Result<(), ProcessingError> {
        if !self.debug_enabled() {
            return Ok(());
        }

        self.output(&image, "after_perspective_transform")
    }

    fn debug_digits_before_morph(&self, image: &Mat) -> Result<(), ProcessingError> {
        if !self.debug_enabled() {
            return Ok(());
        }

        self.output(image, "digits_before_morph")
    }

    fn debug_digits_after_dilation(&self, image: &Mat) -> Result<(), ProcessingError> {
        if !self.debug_enabled() {
            return Ok(());
        }

        self.output(&image, "digits_after_dilation")
    }

    fn debug_digit_locations(
        &self,
        image: &Mat,
        digit_locations: &Vec<Rect2i>,
    ) -> Result<(), ProcessingError> {
        if !self.debug_enabled() {
            return Ok(());
        }

        let mut temp_image = Mat::default();
        cvt_color_def(&image, &mut temp_image, CV_8U)?;

        for b in digit_locations {
            rectangle_def(&mut temp_image, *b, Scalar::new(0.0, 255.0, 0.0, 0.0))?;
        }

        self.output(&temp_image, "digit_locations")
    }
}

impl TempFolderDebugger {
    pub fn using_timestamp_folder_name(debug_enabled: bool) -> Self {
        let now = chrono::offset::Local::now();
        let folder_name: String = now.format("%Y-%m-%d-%H-%M-%S").to_string();

        TempFolderDebugger {
            folder_name,
            debug_enabled,
        }
    }
}

impl BpmOcrDebugOutputter for TempFolderDebugger {
    fn new(unique_session_name: &str, debug_enabled: bool) -> Self {
        TempFolderDebugger {
            folder_name: unique_session_name.to_string(),
            debug_enabled: debug_enabled,
        }
    }

    fn output(&self, image: &Mat, stage_description: &str) -> Result<(), ProcessingError> {
        let folder_path = env::temp_dir().join("bmp-ocr").join(&self.folder_name);

        // TODO: create at construction
        create_dir_all(&folder_path).map_err(|_| {
            ProcessingError::AppError(ReadingIdentificationError::InternalError(
                "Could not create a temporary folder for debugging image processing",
            ))
        })?;

        let file_name = format!("{}.jpeg", &stage_description);

        let binding = folder_path.join(file_name);

        let file_path = binding.to_str().ok_or_else(|| {
            ProcessingError::AppError(ReadingIdentificationError::InternalError(
                &"Could not create a temporary folder for debugging image processing",
            ))
        })?;

        imwrite_def(&file_path, &image)?;
        Ok(())
    }

    fn debug_enabled(&self) -> bool {
        self.debug_enabled
    }
}

impl BpmOcrDebugOutputter for NoDebug {
    fn new(_: &str, _: bool) -> Self {
        NoDebug {  }
    }

    fn output(&self, _: &Mat, _: &str) -> Result<(), ProcessingError> {
        Ok(())
    }

    fn debug_enabled(&self) -> bool {
        false
    }
}