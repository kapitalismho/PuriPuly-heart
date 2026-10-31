pub const ICON_SIZE: u32 = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IconState {
    Idle,
    Grab,
    Failed,
}

pub fn pixels(state: IconState) -> Vec<u8> {
    let mut rgba = vec![0; (ICON_SIZE * ICON_SIZE * 4) as usize];
    for y in 0..ICON_SIZE {
        for x in 0..ICON_SIZE {
            let x = x as i32;
            let y = y as i32;
            let failed = state == IconState::Failed;
            let border = (20..108).contains(&x)
                && (20..108).contains(&y)
                && (x < 26 || x >= 102 || y < 26 || y >= 102);
            let ink = if failed {
                (32..96).contains(&x)
                    && (32..96).contains(&y)
                    && ((x - y).abs() <= 4 || (x + y - 127).abs() <= 4)
            } else {
                border
                    || ((36..92).contains(&x) && (y - 78 + (x - 54).abs() / 2).abs() < 3)
                    || ((x - 82) * (x - 82) + (y - 45) * (y - 45) < 36)
            };
            let glow = state == IconState::Grab
                && (12..116).contains(&x)
                && (12..116).contains(&y)
                && (x < 18 || x >= 110 || y < 18 || y >= 110);
            if ink || glow {
                let offset = ((y as u32 * ICON_SIZE + x as u32) * 4) as usize;
                rgba[offset..offset + 4].copy_from_slice(if failed {
                    &[255, 96, 96, 255]
                } else {
                    &[235, 245, 255, 255]
                });
            }
        }
    }
    rgba
}
