use opencv::core::{Point, Vector};

use crate::models::{self, ReadingIdentificationError};

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

pub fn get_rectangle_coordinates(
    coordinates: &Vector<Point>,
) -> Option<models::RectangleCoordinates> {
    match coordinates.as_slice() {
        [p1, p2, p3, p4] => {
            let coordinates = locate_corners((*p1, *p2, *p3, *p4));

            Some(coordinates)
        }
        _ => None,
    }
}
