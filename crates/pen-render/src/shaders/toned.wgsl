fn toned(color: vec3f) -> vec3f {
  let shoulder = 0.9;
  let d = 1.0 - shoulder;
  let rolled = 1.0 - d * d / (max(color, vec3f(shoulder)) + d - shoulder);
  return select(color, rolled, color > vec3f(shoulder));
}
