// WGSL for every GPU pass. All fragment passes share one bind group layout:
//   0 src  1 sampler  2 params  3 aux1  4 aux2  5 aux3
// and one uniform block of four vec4s whose meaning is per pass. Each filter
// is its own entry point and pipeline (no uber-shader branching).

export const COLOR_WGSL = /* wgsl */ `
fn luma(c: vec3f) -> f32 {
  return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

fn srgbToLinear(c: vec3f) -> vec3f {
  let lo = c / 12.92;
  let hi = pow((max(c, vec3f(0.0)) + vec3f(0.055)) / 1.055, vec3f(2.4));
  return select(hi, lo, c <= vec3f(0.04045));
}

fn linearToSrgb(c: vec3f) -> vec3f {
  let x = clamp(c, vec3f(0.0), vec3f(1.0));
  let lo = x * 12.92;
  let hi = 1.055 * pow(max(x, vec3f(1e-9)), vec3f(1.0 / 2.4)) - vec3f(0.055);
  return select(hi, lo, x <= vec3f(0.0031308));
}

fn rgbToOklab(srgb: vec3f) -> vec3f {
  let c = srgbToLinear(clamp(srgb, vec3f(0.0), vec3f(1.0)));
  let lms = vec3f(
    0.4122214708 * c.r + 0.5363325363 * c.g + 0.0514459929 * c.b,
    0.2119034982 * c.r + 0.6806995451 * c.g + 0.1073969566 * c.b,
    0.0883024619 * c.r + 0.2817188376 * c.g + 0.6299787005 * c.b
  );
  let r = pow(max(lms, vec3f(1e-9)), vec3f(1.0 / 3.0));
  return vec3f(
    0.2104542553 * r.x + 0.793617785 * r.y - 0.0040720468 * r.z,
    1.9779984951 * r.x - 2.428592205 * r.y + 0.4505937099 * r.z,
    0.0259040371 * r.x + 0.7827717662 * r.y - 0.808675766 * r.z
  );
}

fn oklabToRgb(lab: vec3f) -> vec3f {
  let l1 = lab.x + 0.3963377774 * lab.y + 0.2158037573 * lab.z;
  let m1 = lab.x - 0.1055613458 * lab.y - 0.0638541728 * lab.z;
  let s1 = lab.x - 0.0894841775 * lab.y - 1.291485548 * lab.z;
  let l3 = l1 * l1 * l1;
  let m3 = m1 * m1 * m1;
  let s3 = s1 * s1 * s1;
  return linearToSrgb(vec3f(
    4.0767416621 * l3 - 3.3077115913 * m3 + 0.2309699292 * s3,
    -1.2684380046 * l3 + 2.6097574011 * m3 - 0.3413193965 * s3,
    -0.0041960863 * l3 - 0.7034186147 * m3 + 1.707614701 * s3
  ));
}

fn applyContrast(color: vec3f, contrast: f32) -> vec3f {
  return clamp((color - vec3f(0.5)) * contrast + vec3f(0.5), vec3f(0.0), vec3f(1.0));
}
`

export const PASS_SHADER = /* wgsl */ `
struct Params {
  a: vec4f,
  b: vec4f,
  c: vec4f,
  d: vec4f,
};

struct VertexOut {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> p: Params;
@group(0) @binding(3) var aux1: texture_2d<f32>;
@group(0) @binding(4) var aux2: texture_2d<f32>;
@group(0) @binding(5) var aux3: texture_2d<f32>;

${COLOR_WGSL}

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOut {
  var positions = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  let pos = positions[vertexIndex];
  var out: VertexOut;
  out.position = vec4f(pos, 0.0, 1.0);
  out.uv = vec2f(pos.x * 0.5 + 0.5, 0.5 - pos.y * 0.5);
  return out;
}

fn texelOf(t: texture_2d<f32>) -> vec2f {
  return 1.0 / vec2f(textureDimensions(t));
}

fn sampleSrc(uv: vec2f) -> vec3f {
  return textureSampleLevel(src, samp, uv, 0.0).rgb;
}

// ---------------------------------------------------------------------------
// Structure tensor (E, F, G) from Sobel derivatives.
// a.x: 0 = color (sum over RGB), 1 = OKLab lightness only.
fn tensorTap(uv: vec2f, offset: vec2f, texel: vec2f) -> vec3f {
  let c = sampleSrc(uv + offset * texel);
  if (p.a.x > 0.5) {
    return vec3f(rgbToOklab(c).x, 0.0, 0.0);
  }
  return c;
}

@fragment
fn tensorMain(in: VertexOut) -> @location(0) vec4f {
  let texel = texelOf(src);
  let c00 = tensorTap(in.uv, vec2f(-1.0, -1.0), texel);
  let c10 = tensorTap(in.uv, vec2f(0.0, -1.0), texel);
  let c20 = tensorTap(in.uv, vec2f(1.0, -1.0), texel);
  let c01 = tensorTap(in.uv, vec2f(-1.0, 0.0), texel);
  let c21 = tensorTap(in.uv, vec2f(1.0, 0.0), texel);
  let c02 = tensorTap(in.uv, vec2f(-1.0, 1.0), texel);
  let c12 = tensorTap(in.uv, vec2f(0.0, 1.0), texel);
  let c22 = tensorTap(in.uv, vec2f(1.0, 1.0), texel);
  let gx = (-c00 - 2.0 * c01 - c02 + c20 + 2.0 * c21 + c22) * 0.25;
  let gy = (-c00 - 2.0 * c10 - c20 + c02 + 2.0 * c12 + c22) * 0.25;
  return vec4f(dot(gx, gx), dot(gx, gy), dot(gy, gy), 1.0);
}

// ---------------------------------------------------------------------------
// Separable Gaussian. a.xy: direction, a.z: sigma in pixels. Large kernels
// are sampled with a stride (bilinear taps) to cap the cost at 49 taps.
@fragment
fn blurMain(in: VertexOut) -> @location(0) vec4f {
  let texel = texelOf(src);
  let sigma = max(p.a.z, 0.05);
  let radius = ceil(sigma * 2.5);
  let taps = clamp(radius, 1.0, 24.0);
  let stride = radius / taps;
  var sum = textureSampleLevel(src, samp, in.uv, 0.0);
  var total = 1.0;
  for (var i = 1.0; i <= taps; i += 1.0) {
    let d = i * stride;
    let w = exp(-(d * d) / (2.0 * sigma * sigma));
    let offset = p.a.xy * d * texel;
    sum += (textureSampleLevel(src, samp, in.uv + offset, 0.0) + textureSampleLevel(src, samp, in.uv - offset, 0.0)) * w;
    total += 2.0 * w;
  }
  return sum / total;
}

// ---------------------------------------------------------------------------
// Tangent field from a smoothed tensor: (tangent.xy, anisotropy, sqrt(lambda1)).
@fragment
fn tfmMain(in: VertexOut) -> @location(0) vec4f {
  let g = textureSampleLevel(src, samp, in.uv, 0.0).xyz;
  let root = sqrt(max((g.x - g.z) * (g.x - g.z) + 4.0 * g.y * g.y, 0.0));
  let lambda1 = 0.5 * (g.x + g.z + root);
  let lambda2 = 0.5 * (g.x + g.z - root);
  let v = vec2f(lambda1 - g.x, -g.y);
  var t = vec2f(0.0, 1.0);
  if (length(v) > 1e-9) {
    t = normalize(v);
  }
  let total = lambda1 + lambda2;
  let anisotropy = select(0.0, (lambda1 - lambda2) / total, total > 1e-9);
  return vec4f(t, anisotropy, sqrt(max(lambda1, 0.0)));
}

// ---------------------------------------------------------------------------
// Anisotropic Kuwahara filter with polynomial sector weights (Kyprianidis et
// al. 2009/2010). src: image, aux1: tangent field.
// a.x: radius, a.y: sharpness q, a.z: anisotropy alpha.
@fragment
fn akfMain(in: VertexOut) -> @location(0) vec4f {
  let texel = texelOf(src);
  let radius = max(p.a.x, 1.0);
  let q = p.a.y;
  let alpha = max(p.a.z, 0.01);
  let tfm = textureSampleLevel(aux1, samp, in.uv, 0.0);
  let anisotropy = tfm.z;
  let a = radius * clamp((alpha + anisotropy) / alpha, 0.1, 2.0);
  let b = radius * clamp(alpha / (alpha + anisotropy), 0.1, 2.0);
  let phi = -atan2(tfm.y, tfm.x);
  let cosPhi = cos(phi);
  let sinPhi = sin(phi);
  let row0 = vec2f(cosPhi, -sinPhi) * (0.5 / a);
  let row1 = vec2f(sinPhi, cosPhi) * (0.5 / b);
  let maxX = i32(sqrt(a * a * cosPhi * cosPhi + b * b * sinPhi * sinPhi));
  let maxY = i32(sqrt(a * a * sinPhi * sinPhi + b * b * cosPhi * cosPhi));
  let zeta = 2.0 / radius;
  let zeroCross = 0.58;
  let sinZero = sin(zeroCross);
  let eta = (zeta + cos(zeroCross)) / (sinZero * sinZero);

  var m: array<vec4f, 8>;
  var s: array<vec3f, 8>;

  for (var y = -maxY; y <= maxY; y++) {
    for (var x = -maxX; x <= maxX; x++) {
      let offset = vec2f(f32(x), f32(y));
      var v = vec2f(dot(row0, offset), dot(row1, offset));
      if (dot(v, v) > 0.25) {
        continue;
      }
      let c = sampleSrc(in.uv + offset * texel);
      var w: array<f32, 8>;
      var sum = 0.0;
      var vxx = zeta - eta * v.x * v.x;
      var vyy = zeta - eta * v.y * v.y;
      var z = max(0.0, v.y + vxx);
      w[0] = z * z;
      z = max(0.0, -v.x + vyy);
      w[2] = z * z;
      z = max(0.0, -v.y + vxx);
      w[4] = z * z;
      z = max(0.0, v.x + vyy);
      w[6] = z * z;
      v = 0.70710678 * vec2f(v.x - v.y, v.x + v.y);
      vxx = zeta - eta * v.x * v.x;
      vyy = zeta - eta * v.y * v.y;
      z = max(0.0, v.y + vxx);
      w[1] = z * z;
      z = max(0.0, -v.x + vyy);
      w[3] = z * z;
      z = max(0.0, -v.y + vxx);
      w[5] = z * z;
      z = max(0.0, v.x + vyy);
      w[7] = z * z;
      for (var k = 0; k < 8; k++) {
        sum += w[k];
      }
      let g = exp(-3.125 * dot(v, v)) / max(sum, 1e-6);
      for (var k = 0; k < 8; k++) {
        let wk = w[k] * g;
        m[k] += vec4f(c * wk, wk);
        s[k] += c * c * wk;
      }
    }
  }

  var result = vec4f(0.0);
  for (var k = 0; k < 8; k++) {
    if (m[k].w <= 0.0) {
      continue;
    }
    let mean = m[k].rgb / m[k].w;
    let variance = abs(s[k] / m[k].w - mean * mean);
    let sigma2 = variance.r + variance.g + variance.b;
    let weight = 1.0 / (1.0 + pow(max(1000.0 * sigma2, 1e-6), 0.5 * q));
    result += vec4f(mean * weight, weight);
  }
  if (result.w <= 0.0) {
    return vec4f(sampleSrc(in.uv), 1.0);
  }
  return vec4f(clamp(result.rgb / result.w, vec3f(0.0), vec3f(1.0)), 1.0);
}

// ---------------------------------------------------------------------------
// Flow-based DoG, step 1: difference of Gaussians across the edge direction.
// src: image, aux1: tangent field. a.x: sigmaE, a.y: tau.
@fragment
fn dogMain(in: VertexOut) -> @location(0) vec4f {
  let texel = texelOf(src);
  let t = textureSampleLevel(aux1, samp, in.uv, 0.0).xy;
  let n = vec2f(t.y, -t.x);
  let sigmaE = max(p.a.x, 0.3);
  let sigmaR = sigmaE * 1.6;
  let halfWidth = min(ceil(2.0 * sigmaR), 32.0);
  let center = luma(sampleSrc(in.uv));
  var sum = vec2f(center);
  var norm = vec2f(1.0);
  for (var d = 1.0; d <= halfWidth; d += 1.0) {
    let k = vec2f(exp(-(d * d) / (2.0 * sigmaE * sigmaE)), exp(-(d * d) / (2.0 * sigmaR * sigmaR)));
    let l1 = luma(sampleSrc(in.uv + n * d * texel));
    let l2 = luma(sampleSrc(in.uv - n * d * texel));
    sum += (l1 + l2) * k;
    norm += 2.0 * k;
  }
  sum = sum / norm;
  return vec4f(sum.x - p.a.y * sum.y, 0.0, 0.0, 1.0);
}

// Flow-based DoG, step 2: smooth along the tangent streamline, then apply a
// soft XDoG threshold. src: DoG response, aux1: tangent field.
// a.x: sigmaM, a.y: phi (sharpness). Output: 1 = paper, 0 = line.
@fragment
fn licMain(in: VertexOut) -> @location(0) vec4f {
  let texel = texelOf(src);
  let sigmaM = max(p.a.x, 0.3);
  let steps = i32(min(ceil(2.0 * sigmaM), 24.0));
  var sum = textureSampleLevel(src, samp, in.uv, 0.0).r;
  var total = 1.0;
  for (var direction = 0; direction < 2; direction++) {
    var pos = in.uv;
    var t = textureSampleLevel(aux1, samp, in.uv, 0.0).xy;
    if (direction == 1) {
      t = -t;
    }
    for (var i = 1; i <= steps; i++) {
      pos += t * texel;
      var next = textureSampleLevel(aux1, samp, pos, 0.0).xy;
      if (dot(next, t) < 0.0) {
        next = -next;
      }
      t = next;
      let fi = f32(i);
      let k = exp(-(fi * fi) / (2.0 * sigmaM * sigmaM));
      sum += textureSampleLevel(src, samp, pos, 0.0).r * k;
      total += k;
    }
  }
  let h = sum / total;
  let edge = select(1.0 + tanh(p.a.y * h), 1.0, h >= 0.0);
  return vec4f(vec3f(edge), 1.0);
}

// Edges filter output. src: image, aux1: lines.
// a.x: strength, a.y: contrast, a.z: lines only.
@fragment
fn edgesMain(in: VertexOut) -> @location(0) vec4f {
  let line = textureSampleLevel(aux1, samp, in.uv, 0.0).r;
  var color = applyContrast(sampleSrc(in.uv), p.a.y);
  if (p.a.z > 0.5) {
    color = vec3f(1.0);
  }
  return vec4f(color * mix(1.0, line, p.a.x), 1.0);
}

// ---------------------------------------------------------------------------
// Value study: posterized OKLab lightness. src: (squinted) image.
// a.x: levels, a.y: contrast, a.z: keep color.
@fragment
fn valueMain(in: VertexOut) -> @location(0) vec4f {
  let lab = rgbToOklab(sampleSrc(in.uv));
  let levels = max(p.a.x, 2.0);
  let lightness = clamp((lab.x - 0.5) * p.a.y + 0.5, 0.0, 1.0);
  let band = min(floor(lightness * levels), levels - 1.0);
  let quantized = band / (levels - 1.0);
  let chroma = select(vec2f(0.0), lab.yz, p.a.z > 0.5);
  return vec4f(oklabToRgb(vec3f(quantized, chroma)), 1.0);
}

fn heatRamp(value: f32) -> vec3f {
  let x = clamp(value, 0.0, 1.0) * 4.0;
  let c0 = vec3f(0.04, 0.04, 0.07);
  let c1 = vec3f(0.33, 0.1, 0.52);
  let c2 = vec3f(0.84, 0.2, 0.26);
  let c3 = vec3f(1.0, 0.64, 0.1);
  let c4 = vec3f(1.0, 0.98, 0.84);
  if (x < 1.0) {
    return mix(c0, c1, x);
  } else if (x < 2.0) {
    return mix(c1, c2, x - 1.0);
  } else if (x < 3.0) {
    return mix(c2, c3, x - 2.0);
  }
  return mix(c3, c4, x - 3.0);
}

// Chroma map. a.x: gain, a.y: 0 heat / 1 gray.
@fragment
fn chromaMain(in: VertexOut) -> @location(0) vec4f {
  let lab = rgbToOklab(sampleSrc(in.uv));
  let value = clamp(length(lab.yz) * p.a.x / 0.3, 0.0, 1.0);
  if (p.a.y > 0.5) {
    return vec4f(vec3f(value), 1.0);
  }
  return vec4f(heatRamp(value), 1.0);
}

// Warm/cool map along the OKLab orange (warm) - blue (cool) axis.
// a.x: gain, a.y: neutral zone, a.z: keep value.
@fragment
fn temperatureMain(in: VertexOut) -> @location(0) vec4f {
  let lab = rgbToOklab(sampleSrc(in.uv));
  let axis = vec2f(0.5, 0.8660254);
  let t = dot(lab.yz, axis);
  let magnitude = max(abs(t) - p.a.y, 0.0) * p.a.x;
  let lightness = select(0.72, lab.x, p.a.z > 0.5);
  let ab = axis * clamp(sign(t) * magnitude, -0.16, 0.16);
  return vec4f(oklabToRgb(vec3f(lightness, ab)), 1.0);
}

// ---------------------------------------------------------------------------
// Flow brush: paint smeared along the smoothed flow field (line integral
// convolution). Each pixel averages color along its streamline; the trace
// stops once the color departs from the starting pixel, so strokes end at
// object boundaries instead of dragging across them. Brush texture is noise
// integrated along the same streamline, which turns isotropic blobs into
// coherent streaks of stroke width and length.
// src: image blurred to stroke width, aux1: tangent field, aux2: original.
// a: (length, width, unused, colorMerge), b: (edgePreserve, edgeThreshold,
// strength, contrast).
fn hash21(value: vec2f) -> f32 {
  return fract(sin(dot(value, vec2f(127.1, 311.7))) * 43758.5453123);
}

fn valueNoise(position: vec2f) -> f32 {
  let cell = floor(position);
  let f = fract(position);
  let u = f * f * (3.0 - 2.0 * f);
  let top = mix(hash21(cell), hash21(cell + vec2f(1.0, 0.0)), u.x);
  let bottom = mix(hash21(cell + vec2f(0.0, 1.0)), hash21(cell + vec2f(1.0, 1.0)), u.x);
  return mix(top, bottom, u.y);
}

fn quantizeColor(color: vec3f, merge: f32) -> vec3f {
  let amount = clamp(merge, 0.0, 1.0);
  let levels = mix(48.0, 7.0, amount);
  return mix(color, floor(color * levels + vec3f(0.5)) / levels, amount);
}

const FLOW_STEPS = 14;
// Where the image is flat the structure tensor has no reliable direction and
// its tangents wander into marbled swirls. There the stroke falls back to one
// consistent diagonal, the way a painter lays in a quiet passage.
const HAND_DIRECTION = vec2f(0.8660254, -0.5);

fn brushTangent(uv: vec2f) -> vec2f {
  let field = textureSampleLevel(aux1, samp, uv, 0.0);
  let confidence = smoothstep(0.02, 0.25, field.z) * smoothstep(0.008, 0.04, field.w);
  let tangent = select(-field.xy, field.xy, dot(field.xy, HAND_DIRECTION) >= 0.0);
  let blended = mix(HAND_DIRECTION, tangent, confidence);
  return blended / max(length(blended), 1e-5);
}

@fragment
fn flowBrushMain(in: VertexOut) -> @location(0) vec4f {
  let size = vec2f(textureDimensions(src));
  // src is pre-blurred to the stroke width, so a stroke carries one color
  // across its width and fine texture does not stop it.
  let center = sampleSrc(in.uv);
  let original = textureSampleLevel(aux2, samp, in.uv, 0.0).rgb;
  let halfLength = max(p.a.x, 2.0);
  let halfWidth = max(p.a.y, 1.0);
  let edgePreserve = clamp(p.b.x, 0.0, 1.0);
  let threshold = max(p.b.y, 0.01);
  let strength = clamp(p.b.z, 0.0, 1.0);
  let stepLength = halfLength / f32(FLOW_STEPS);
  let pixel = in.uv * size;

  // Coarse noise sets per-stroke value shifts, fine noise the bristle lines.
  let coarseScale = 1.0 / halfWidth;
  let fineScale = 3.5 / halfWidth;
  var colorSum = center;
  var weightSum = 1.0;
  var coarseSum = valueNoise(pixel * coarseScale);
  var fineSum = valueNoise(pixel * fineScale + vec2f(41.0, 17.0));

  for (var direction = 0; direction < 2; direction++) {
    var position = pixel;
    var tangent = brushTangent(in.uv);
    if (direction == 1) {
      tangent = -tangent;
    }
    var alive = 1.0;
    for (var i = 1; i <= FLOW_STEPS; i++) {
      position += tangent * stepLength;
      let uv = position / size;
      var next = brushTangent(uv);
      if (dot(next, tangent) < 0.0) {
        next = -next;
      }
      tangent = next;
      let sampleColor = sampleSrc(uv);
      if (length(sampleColor - center) > threshold) {
        alive *= 1.0 - edgePreserve;
      }
      let along = f32(i) / f32(FLOW_STEPS);
      let weight = exp(-along * along * 1.5) * alive;
      colorSum += sampleColor * weight;
      weightSum += weight;
      coarseSum += valueNoise(position * coarseScale) * weight;
      fineSum += valueNoise(position * fineScale + vec2f(41.0, 17.0)) * weight;
    }
  }

  let smeared = quantizeColor(colorSum / weightSum, p.a.w);
  // Averaging flattens the noise toward 0.5; rescale so streaks stay visible.
  let coarse = (coarseSum / weightSum - 0.5) * 2.2;
  let fine = (fineSum / weightSum - 0.5) * 2.2;
  let shade = 1.0 + (coarse * 0.16 + fine * 0.1) * strength;
  let color = mix(original, smeared * shade, strength);
  return vec4f(applyContrast(color, p.b.w), 1.0);
}

// ---------------------------------------------------------------------------
// Lost & found edges. src: simplified base, aux1: softened base,
// aux2: lightness tangent field of the base (w = edge strength), aux3:
// original image. a: (found threshold, accent threshold, accent strength,
// show mask), b.x: lost blur radius in pixels.
fn edgeImportance(uv: vec2f) -> f32 {
  return clamp(textureSampleLevel(aux2, samp, uv, 0.0).w * 6.0, 0.0, 1.0);
}

@fragment
fn lostFoundMain(in: VertexOut) -> @location(0) vec4f {
  let base = sampleSrc(in.uv);
  let lost = textureSampleLevel(aux1, samp, in.uv, 0.0).rgb;
  let importance = edgeImportance(in.uv);
  let original = textureSampleLevel(aux3, samp, in.uv, 0.0).rgb;
  // The lost blur reaches about its radius past an edge, so a found edge
  // keeps the crisp base over that whole reach; otherwise the blur smears a
  // halo just outside every found contour.
  var reach = importance;
  let radius = p.b.x;
  if (radius >= 1.0) {
    let texel = texelOf(src);
    for (var i = 0; i < 12; i++) {
      let angle = f32(i) * 0.5235988;
      let direction = vec2f(cos(angle), sin(angle)) * texel;
      reach = max(reach, edgeImportance(in.uv + direction * radius));
      reach = max(reach, edgeImportance(in.uv + direction * radius * 0.5));
    }
  }
  let found = smoothstep(p.a.x - 0.06, p.a.x + 0.06, reach);
  let accentMask = smoothstep(p.a.y - 0.05, p.a.y + 0.05, importance);
  var color = mix(lost, base, found);
  color = mix(color, original, accentMask * p.a.z);
  if (p.a.w > 0.5) {
    let context = vec3f(luma(base) * 0.35);
    var mask = mix(vec3f(0.1, 0.16, 0.32), vec3f(0.95, 0.55, 0.15), found);
    mask = mix(mask, vec3f(1.0), accentMask);
    color = mix(context, mask, 0.75);
  }
  return vec4f(color, 1.0);
}

// ---------------------------------------------------------------------------
// Final composite onto the canvas or export target. src: filtered image,
// aux1: original image, aux2: edge overlay.
// a: crop offset.xy, crop scale.zw
// b: mirror, grayscale, split enabled, split position
// c: grid divisions, diagonals, overlay strength, line width (px)
// d: output size.xy
@fragment
fn compositeMain(in: VertexOut) -> @location(0) vec4f {
  let screen = in.uv;
  var imageUv = screen;
  if (p.b.x > 0.5) {
    imageUv.x = 1.0 - imageUv.x;
  }
  let uv = p.a.xy + imageUv * p.a.zw;
  var color = sampleSrc(uv);
  if (p.b.z > 0.5 && screen.x < p.b.w) {
    color = textureSampleLevel(aux1, samp, uv, 0.0).rgb;
  }
  if (p.c.z > 0.0) {
    color *= mix(1.0, textureSampleLevel(aux2, samp, uv, 0.0).r, p.c.z);
  }
  if (p.b.y > 0.5) {
    color = oklabToRgb(vec3f(rgbToOklab(color).x, 0.0, 0.0));
  }

  let outSize = p.d.xy;
  let pixel = screen * outSize;
  let halfLine = p.c.w * 0.5;
  let lineColor = select(vec3f(1.0), vec3f(0.06), luma(color) > 0.55);
  if (p.c.x > 0.5) {
    let cell = outSize / p.c.x;
    let gridDistance = abs(fract(pixel / cell + 0.5) - 0.5) * cell;
    if (min(gridDistance.x, gridDistance.y) < halfLine) {
      color = mix(color, lineColor, 0.8);
    }
  }
  if (p.c.y > 0.5) {
    let diagonal = length(outSize);
    let d1 = abs(outSize.y * pixel.x - outSize.x * pixel.y) / diagonal;
    let d2 = abs(outSize.y * pixel.x + outSize.x * pixel.y - outSize.x * outSize.y) / diagonal;
    if (min(d1, d2) < halfLine) {
      color = mix(color, lineColor, 0.8);
    }
  }
  if (p.b.z > 0.5 && abs(pixel.x - p.b.w * outSize.x) < max(1.0, halfLine)) {
    color = vec3f(1.0);
  }
  return vec4f(clamp(color, vec3f(0.0), vec3f(1.0)), 1.0);
}
`

// SLIC superpixels in OKLab (scaled x100 to match the usual CIELAB
// compactness range). Clustering only; connectivity clean-up and coloring run
// on the CPU from the read-back labels.
export const SLIC_SHADER = /* wgsl */ `
struct Params {
  imageSize: vec2u,
  gridWidth: u32,
  centerCount: u32,
  step: f32,
  compactness: f32,
  pad0: f32,
  pad1: f32,
};

struct Center {
  x: f32,
  y: f32,
  l: f32,
  a: f32,
  b: f32,
  pad0: f32,
  pad1: f32,
  pad2: f32,
};

@group(0) @binding(0) var imageTexture: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> centers: array<Center>;
@group(0) @binding(2) var<storage, read_write> labels: array<u32>;
@group(0) @binding(3) var<storage, read_write> sums: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> params: Params;

${COLOR_WGSL}

const SUM_STRIDE = 6u;
// Fixed-point scale for atomic sums of OKLab*100 values.
const FIXED = 64.0;

fn labAt(id: vec2u) -> vec3f {
  return rgbToOklab(textureLoad(imageTexture, id, 0).rgb) * 100.0;
}

@compute @workgroup_size(64)
fn clearSums(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= params.centerCount * SUM_STRIDE) {
    return;
  }
  atomicStore(&sums[id.x], 0u);
}

@compute @workgroup_size(16, 16)
fn assignPixels(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= params.imageSize.x || id.y >= params.imageSize.y) {
    return;
  }

  let lab = labAt(id.xy);
  let cellX = i32(floor(f32(id.x) / params.step));
  let cellY = i32(floor(f32(id.y) / params.step));
  let spatialWeight = (params.compactness * params.compactness) / (params.step * params.step);
  let gridHeight = i32((params.centerCount + params.gridWidth - 1u) / params.gridWidth);

  var bestLabel = 0u;
  var bestDistance = 3.402823e38;
  for (var y = cellY - 2; y <= cellY + 2; y++) {
    for (var x = cellX - 2; x <= cellX + 2; x++) {
      if (x < 0 || y < 0 || x >= i32(params.gridWidth) || y >= gridHeight) {
        continue;
      }
      let label = u32(x) + u32(y) * params.gridWidth;
      if (label >= params.centerCount) {
        continue;
      }
      let center = centers[label];
      let dl = lab.x - center.l;
      let da = lab.y - center.a;
      let db = lab.z - center.b;
      let dx = f32(id.x) - center.x;
      let dy = f32(id.y) - center.y;
      let clusterDistance = dl * dl + da * da + db * db + (dx * dx + dy * dy) * spatialWeight;
      if (clusterDistance < bestDistance) {
        bestDistance = clusterDistance;
        bestLabel = label;
      }
    }
  }

  labels[id.x + id.y * params.imageSize.x] = bestLabel;
}

@compute @workgroup_size(16, 16)
fn accumulateCenters(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= params.imageSize.x || id.y >= params.imageSize.y) {
    return;
  }

  let label = labels[id.x + id.y * params.imageSize.x];
  let lab = labAt(id.xy);
  let offset = label * SUM_STRIDE;
  atomicAdd(&sums[offset], id.x);
  atomicAdd(&sums[offset + 1u], id.y);
  atomicAdd(&sums[offset + 2u], u32(round(clamp(lab.x, 0.0, 100.0) * FIXED)));
  atomicAdd(&sums[offset + 3u], u32(round(clamp(lab.y + 50.0, 0.0, 100.0) * FIXED)));
  atomicAdd(&sums[offset + 4u], u32(round(clamp(lab.z + 50.0, 0.0, 100.0) * FIXED)));
  atomicAdd(&sums[offset + 5u], 1u);
}

@compute @workgroup_size(64)
fn updateCenters(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= params.centerCount) {
    return;
  }

  let offset = id.x * SUM_STRIDE;
  let count = atomicLoad(&sums[offset + 5u]);
  if (count == 0u) {
    return;
  }

  let countF = f32(count);
  centers[id.x].x = f32(atomicLoad(&sums[offset])) / countF;
  centers[id.x].y = f32(atomicLoad(&sums[offset + 1u])) / countF;
  centers[id.x].l = f32(atomicLoad(&sums[offset + 2u])) / (countF * FIXED);
  centers[id.x].a = f32(atomicLoad(&sums[offset + 3u])) / (countF * FIXED) - 50.0;
  centers[id.x].b = f32(atomicLoad(&sums[offset + 4u])) / (countF * FIXED) - 50.0;
}
`
