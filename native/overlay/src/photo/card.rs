use super::interaction::{compose, Transform};

pub const THICKNESS_M: f32 = 0.001;

#[derive(Clone, Copy)]
pub struct CardFace {
    pub transform: Transform,
    pub width: f32,
    pub height: f32,
}

pub fn faces(front: Transform, width: f32, height: f32) -> [CardFace; 5] {
    let depth = THICKNESS_M;
    let back = [
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, -depth],
    ];
    let left = [
        [0.0, 0.0, -1.0, -width * 0.5],
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, -depth * 0.5],
    ];
    let right = [
        [0.0, 0.0, 1.0, width * 0.5],
        [0.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, -depth * 0.5],
    ];
    let top = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, height * 0.5],
        [0.0, -1.0, 0.0, -depth * 0.5],
    ];
    let bottom = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, -height * 0.5],
        [0.0, 1.0, 0.0, -depth * 0.5],
    ];
    [(back, width, height), (left, depth, height), (right, depth, height), (top, width, depth), (bottom, width, depth)]
        .map(|(local, width, height)| CardFace { transform: compose(front, local), width, height })
}

pub fn front_facing(world: Transform, head: Transform) -> bool {
    (0..3).map(|i| world[i][2] * (head[i][3] - world[i][3])).sum::<f32>() > 0.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::interaction::{inverse, valid_transform, IDENTITY};

    #[test]
    fn cropped_card_faces_close_a_one_millimetre_slab_under_rotation() {
        let front = [[0.0, -1.0, 0.0, 0.2], [0.0, 0.0, 1.0, 0.3], [-1.0, 0.0, 0.0, -0.4]];
        for extent in [1.0, 0.5, 0.01] {
            let width = 0.375 * extent;
            let height = 0.375 * 9.0 / 16.0;
            let mut vertices = std::collections::HashMap::new();
            for face in std::iter::once(CardFace { transform: front, width, height }).chain(faces(front, width, height)) {
                assert!(valid_transform(face.transform));
                let local = compose(inverse(front), face.transform);
                for x in [-0.5, 0.5] {
                    for y in [-0.5, 0.5] {
                        let point: [f32; 3] = std::array::from_fn(|i|
                            local[i][3] + local[i][0] * face.width * x + local[i][1] * face.height * y
                        );
                        assert!((point[0].abs() - width * 0.5).abs() < 0.000001);
                        assert!((point[1].abs() - height * 0.5).abs() < 0.000001);
                        assert!(point[2].abs() < 0.000001 || (point[2] + THICKNESS_M).abs() < 0.000001);
                        let key = point.map(|value| (value * 1_000_000.0).round() as i32);
                        *vertices.entry(key).or_insert(0) += 1;
                    }
                }
            }
            assert_eq!(vertices.len(), 8);
            assert!(vertices.values().all(|count| *count == 3));
        }
        let mut head = IDENTITY;
        head[2][3] = 0.5;
        assert!(front_facing(IDENTITY, head));
        assert!(!front_facing(faces(IDENTITY, 0.375, 0.2)[0].transform, head));
    }
}
