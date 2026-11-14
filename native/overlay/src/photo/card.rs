use super::interaction::{compose, Transform};

pub fn back(front: Transform) -> Transform {
    compose(front, [
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
    ])
}

pub fn card_size(length_m: f32, [width, height]: [u32; 2]) -> [f32; 2] {
    let short = length_m * width.min(height) as f32 / width.max(height) as f32;
    if height > width { [short, length_m] } else { [length_m, short] }
}

pub fn front_facing(world: Transform, head: Transform) -> bool {
    (0..3).map(|i| world[i][2] * (head[i][3] - world[i][3])).sum::<f32>() > 0.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::interaction::{inverse, valid_transform, IDENTITY};

    #[test]
    fn the_long_side_keeps_the_selected_length_for_either_orientation() {
        assert_eq!(card_size(0.375, [1920, 1080]), [0.375, 0.2109375]);
        assert_eq!(card_size(0.375, [1080, 1920]), [0.2109375, 0.375]);
        assert_eq!(card_size(0.375, [1080, 1536]), [0.263671875, 0.375]);
        assert_eq!(card_size(0.375, [1000, 1000]), [0.375, 0.375]);
    }

    #[test]
    fn flat_back_covers_the_cropped_front_and_exactly_one_side_faces_the_viewer() {
        let front = [[0.0, -1.0, 0.0, 0.2], [0.0, 0.0, 1.0, 0.3], [-1.0, 0.0, 0.0, -0.4]];
        let height = 0.375 * 9.0 / 16.0;
        for extent in [1.0, 0.5, 0.01] {
            let width = 0.375 * extent;
            let reverse = back(front);
            assert!(valid_transform(reverse));
            let local = compose(inverse(front), reverse);
            let corners = |t: Transform| {
                let mut points = Vec::new();
                for x in [-0.5, 0.5] {
                    for y in [-0.5, 0.5] {
                        let point: [f32; 3] = std::array::from_fn(|i| t[i][3] + t[i][0] * width * x + t[i][1] * height * y);
                        points.push(point.map(|value| (value * 1_000_000.0).round() as i32));
                    }
                }
                points.sort();
                points
            };
            assert_eq!(corners(local), corners(IDENTITY));
        }
        for z in [0.5, -0.5] {
            let mut head = IDENTITY;
            head[2][3] = z;
            head = compose(front, head);
            assert_ne!(front_facing(front, head), front_facing(back(front), head));
        }
    }
}
