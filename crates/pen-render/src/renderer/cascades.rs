// The sun's shadow maps. The game fits them (Ink's `shadow.cascades`): one
// orthographic window each along the frame's light view, round a slice of the
// camera's view, its middle moved in whole texels so that its shadows hold
// still as the camera moves. Fitting needs the camera, the light and the
// game's idea of how far shadows reach, which the game has; the renderer
// draws what it is given and checks it. Shaders that read them
// (`shadow_cascades`) take each surface from the slice of the view it lies in,
// blending into the next map over the end of a slice; shaders that do not read
// only the last, farthest map, at binding 1.
use pen_protocol::boundary::MAX_CASCADES;
use pen_protocol::{Matrix, RenderFrame};

/// A shadow map: what it sees (in the frame's light view), how wide a texel
/// of it is in world units, and how far ahead of the camera the slice it
/// covers ends.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Cascade {
    pub projection: Matrix,
    pub texel: f32,
    pub far: f32,
}

impl Cascade {
    pub const NONE: Cascade = Cascade { projection: crate::matrix::IDENTITY, texel: 0.0, far: 0.0 };
}

/// Whether `p` is an orthographic projection along its axes.
fn orthographic(p: &Matrix) -> bool {
    [1, 2, 3, 4, 6, 7, 8, 9, 11].iter().all(|&at| p[at] == 0.0) && p[15] == 1.0 && p[0] > 0.0 && p[5] > 0.0
}

/// What the frame's cascades must be: none without a map size, otherwise
/// orthographic, with texels and slices that grow outward.
pub(crate) fn validate(frame: &RenderFrame) -> anyhow::Result<()> {
    use anyhow::ensure;
    ensure!(frame.cascades.len() <= MAX_CASCADES, "a frame has at most {MAX_CASCADES} sun shadow maps");
    ensure!(frame.shadow.texels == 0 || !frame.cascades.is_empty(), "a sun shadow map size needs at least one cascade");
    let mut last = 0.0;
    for cascade in &frame.cascades {
        ensure!(
            orthographic(&cascade.projection)
                && cascade.projection.iter().all(|value| value.is_finite())
                && cascade.texel.is_finite()
                && cascade.texel > 0.0
                && cascade.far.is_finite()
                && cascade.far > last,
            "sun shadow maps must be orthographic with positive texels and growing slices"
        );
        last = cascade.far;
    }
    let view = &frame.shadow.view;
    ensure!(view.iter().all(|value| value.is_finite()), "the shadow view must be finite");
    Ok(())
}

/// The maps `frame` draws, nearest first, and how many: every one of the
/// game's when `all`, else the farthest alone. None without shadows.
pub(crate) fn cascades(frame: &RenderFrame, all: bool) -> ([Cascade; MAX_CASCADES], usize) {
    let mut maps = [Cascade::NONE; MAX_CASCADES];
    if frame.shadow.texels == 0 || frame.cascades.is_empty() {
        return (maps, 0);
    }
    let given = &frame.cascades[..frame.cascades.len().min(MAX_CASCADES)];
    let wanted = if all { given } else { &given[given.len() - 1..] };
    for (map, cascade) in maps.iter_mut().zip(wanted) {
        *map = Cascade { projection: cascade.projection, texel: cascade.texel, far: cascade.far };
    }
    (maps, wanted.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pen_protocol::Cascade as Wire;

    fn frame(count: usize) -> RenderFrame {
        let mut frame = RenderFrame::default();
        frame.shadow.texels = 1024;
        for index in 0..count {
            let mut projection = crate::matrix::IDENTITY;
            let reach = 10.0 * (index + 1) as f32;
            projection[0] = 1.0 / reach;
            projection[5] = 1.0 / reach;
            frame.cascades.push(Wire { projection, texel: 2.0 * reach / 1024.0, far: reach });
        }
        frame
    }

    #[test]
    fn a_frame_draws_the_maps_the_game_fitted() {
        let frame = frame(3);
        let (maps, count) = cascades(&frame, true);
        assert_eq!(count, 3);
        assert_eq!(maps[2].far, 30.0);
        assert!(maps[..3].windows(2).all(|pair| pair[0].texel < pair[1].texel));
        // Without a shader that reads them, the farthest alone, at binding 1.
        let (single, count) = cascades(&frame, false);
        assert_eq!((count, single[0].far), (1, 30.0));
        let mut dark = frame.clone();
        dark.shadow.texels = 0;
        assert_eq!(cascades(&dark, true).1, 0);
        assert_eq!(cascades(&RenderFrame::default(), true).1, 0);
    }

    #[test]
    fn maps_must_be_orthographic_and_grow_outward() {
        assert!(validate(&frame(4)).is_ok());
        let mut shrinking = frame(2);
        shrinking.cascades[1].far = 5.0;
        assert!(validate(&shrinking).is_err());
        let mut perspective = frame(1);
        perspective.cascades[0].projection[11] = -1.0;
        assert!(validate(&perspective).is_err());
        let mut none = frame(0);
        none.shadow.texels = 1024;
        assert!(validate(&none).is_err());
        let mut flat = frame(1);
        flat.cascades[0].texel = 0.0;
        assert!(validate(&flat).is_err());
    }
}
