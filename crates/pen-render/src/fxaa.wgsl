// FXAA 3.11 Quality, preset 12. WGSL adaptation of Timothy Lottes' algorithm:
// https://github.com/GameTechDev/CMAA2/blob/master/Projects/CMAA2/FXAA/Fxaa3_11.h
// Copyright (c) 2014, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of NVIDIA CORPORATION nor the names of its contributors
//   may be used to endorse or promote products derived from this software
//   without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY EXPRESS
// OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
// IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
// THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// fetch() is already toned to the display's UNORM RGB values. Luminance is
// measured there, before HUD/UI composition; encoding it again would change
// contrast thresholds and the scene's existing output colour contract.
fn fxaa(uv: vec2f) -> vec3f {
  // Bilinear arithmetic differs slightly across GPU backends. Consistent
  // ties avoid changing orientation/endpoints for the same 8-bit pixels.
  let epsilon = 0.0001;
  let texel = present.source.xy;
  let color = fetch(uv);
  let m = luma(color);
  let n = luma(fetch(uv - vec2f(0.0, texel.y)));
  let s = luma(fetch(uv + vec2f(0.0, texel.y)));
  let w = luma(fetch(uv - vec2f(texel.x, 0.0)));
  let e = luma(fetch(uv + vec2f(texel.x, 0.0)));
  let low = min(m, min(min(n, s), min(w, e)));
  let high = max(m, max(max(n, s), max(w, e)));
  let contrast = high - low;
  if (contrast < max(0.0312, high * 0.125)) {
    return color;
  }
  let nw = luma(fetch(uv - texel));
  let ne = luma(fetch(uv + vec2f(texel.x, -texel.y)));
  let sw = luma(fetch(uv + vec2f(-texel.x, texel.y)));
  let se = luma(fetch(uv + texel));
  let horizontal = abs(nw + sw - 2.0 * w) + 2.0 * abs(n + s - 2.0 * m) + abs(ne + se - 2.0 * e);
  let vertical = abs(nw + ne - 2.0 * n) + 2.0 * abs(w + e - 2.0 * m) + abs(sw + se - 2.0 * s);
  var across = vec2f(texel.x, 0.0);
  var along = vec2f(0.0, texel.y);
  var negative = w;
  var positive = e;
  if (horizontal + epsilon >= vertical) {
    across = vec2f(0.0, texel.y);
    along = vec2f(texel.x, 0.0);
    negative = n;
    positive = s;
  }
  var adjacent = positive;
  if (abs(negative - m) + epsilon >= abs(positive - m)) {
    across = -across;
    adjacent = negative;
  }
  let gradient = max(abs(negative - m), abs(positive - m)) * 0.25;
  let edge_luma = 0.5 * (adjacent + m);
  let edge = uv + across * 0.5;
  var distance_n = 0.0;
  var distance_p = 0.0;
  var endpoint_n = 0.0;
  var endpoint_p = 0.0;
  var done_n = false;
  var done_p = false;
  // Four endpoint reads per side at most. The final 12-pixel extension
  // bounds a span that has no detected endpoint, as in preset 12.
  let steps = array<f32, 4>(1.0, 1.5, 2.0, 4.0);
  for (var index = 0u; index < 4u; index++) {
    if (!done_n) {
      distance_n += steps[index];
      endpoint_n = luma(fetch(edge - along * distance_n)) - edge_luma;
      done_n = abs(endpoint_n) + epsilon >= gradient;
    }
    if (!done_p) {
      distance_p += steps[index];
      endpoint_p = luma(fetch(edge + along * distance_p)) - edge_luma;
      done_p = abs(endpoint_p) + epsilon >= gradient;
    }
    if (done_n && done_p) {
      break;
    }
  }
  if (!done_n) { distance_n += 12.0; }
  if (!done_p) { distance_p += 12.0; }
  var endpoint = endpoint_p;
  if (distance_n < distance_p) { endpoint = endpoint_n; }
  var edge_blend = 0.0;
  if ((endpoint < -epsilon) != (m - edge_luma < -epsilon)) {
    edge_blend = 0.5 - min(distance_n, distance_p) / (distance_n + distance_p);
  }
  let average = (2.0 * (n + s + w + e) + nw + ne + sw + se) / 12.0;
  let subpixel = clamp(abs(average - m) / contrast, 0.0, 1.0);
  let subpixel_curve = subpixel * subpixel * (3.0 - 2.0 * subpixel);
  let blend = max(edge_blend, subpixel_curve * subpixel_curve * 0.25);
  return fetch(uv + across * blend);
}
