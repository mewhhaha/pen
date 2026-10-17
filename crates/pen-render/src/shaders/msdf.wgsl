// Text from a multi-channel signed distance field (pen-texture's msdf): the
// median of the three channels is the distance to a glyph's edge, sharp at
// its corners however far the glyph is magnified. Sprite shaders add this and
// call `msdf_coverage` in the fragment stage, outside any branch (it takes
// screen-space derivatives).

fn msdf_median(field: vec3f) -> f32 {
  return max(min(field.r, field.g), min(max(field.r, field.g), field.b));
}

// How much of the pixel at `uv` a glyph covers, from the field sampled there
// (`distance`: the median, or the alpha channel's single distance), in an
// atlas of `size` texels whose field spans `range` texels either side of an
// edge. The edge is smoothed over one screen pixel at any scale.
fn msdf_coverage(distance: f32, uv: vec2f, range: f32, size: vec2f) -> f32 {
  // Screen pixels the field's range covers, across and down together.
  let pixels = max(dot(vec2f(range) / size, vec2f(1.0) / fwidth(uv)), 1.0);
  return clamp(pixels * (distance - 0.5) + 0.5, 0.0, 1.0);
}
