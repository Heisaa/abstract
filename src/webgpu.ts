export type FilterId =
  | 'original'
  | 'kuwahara'
  | 'posterize'
  | 'edges'
  | 'flowBrush'
  | 'painter'
  | 'slic'
  | 'adaptiveBrush'
  | 'colorCheck'

export type PainterUnderpainting = 'blocks' | 'smooth'
export type PainterDetailSource = 'edges' | 'color'

export type FilterSettings = {
  filter: FilterId
  radius: number
  levels: number
  grayscale: boolean
  contrast: number
  edgeThreshold: number
  edgeStrength: number
  slicSize: number
  slicCompactness: number
  slicSaturationPreserve: number
  colorCheckScale: number
  colorCheckMinSize: number
  colorCheckSaturation: number
  colorCheckSeparator: number
  colorCheckPerceptual: boolean
  adaptiveLargeBrush: number
  adaptiveMediumBrush: number
  adaptiveSmallBrush: number
  adaptiveDetailThreshold: number
  adaptiveDetailStrength: number
  flowBrushLength: number
  flowBrushWidth: number
  flowSmoothing: number
  flowColorMerge: number
  flowEdgePreserve: number
  flowStrokeStrength: number
  painterAbstraction: number
  painterSize: number
  painterLength: number
  painterDetail: number
  painterEdgeFidelity: number
  painterColorSimplification: number
  painterTexture: number
  painterLayers: number
  painterUnderpainting: PainterUnderpainting
  painterDetailSource: PainterDetailSource
}

export type LoadedImage = {
  bitmap: ImageBitmap
  name: string
  width: number
  height: number
  pixels: Uint8ClampedArray
}

type SlicResources = {
  initLabels: GPUComputePipeline
  clearSums: GPUComputePipeline
  assignPixels: GPUComputePipeline
  accumulateCenters: GPUComputePipeline
  updateCenters: GPUComputePipeline
  colorize: GPUComputePipeline
}

type FelzenszwalbResources = {
  computeWeights: GPUComputePipeline
}

const FILTER_INDEX: Record<FilterId, number> = {
  original: 0,
  kuwahara: 1,
  posterize: 2,
  edges: 3,
  flowBrush: 4,
  painter: 5,
  slic: 6,
  adaptiveBrush: 7,
  colorCheck: 8,
}

const MAX_WORKING_IMAGE_DIMENSION = 1280

const SHADER = /* wgsl */ `
struct Uniforms {
  imageSize: vec2f,
  radius: f32,
  filterMode: f32,
  levels: f32,
  grayscale: f32,
  contrast: f32,
  edgeThreshold: f32,
  edgeStrength: f32,
  slicSize: f32,
  flowBrushLength: f32,
  flowBrushWidth: f32,
  flowSmoothing: f32,
  flowColorMerge: f32,
  flowEdgePreserve: f32,
  flowStrokeStrength: f32,
  pad2: f32,
};

struct VertexOut {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

@group(0) @binding(0) var imageTexture: texture_2d<f32>;
@group(0) @binding(1) var imageSampler: sampler;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;
@group(0) @binding(3) var segmentTexture: texture_2d<f32>;

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOut {
  var positions = array<vec2f, 3>(
    vec2f(-1.0, -1.0),
    vec2f(3.0, -1.0),
    vec2f(-1.0, 3.0)
  );

  var out: VertexOut;
  let pos = positions[vertexIndex];
  out.position = vec4f(pos, 0.0, 1.0);
  out.uv = vec2f(pos.x * 0.5 + 0.5, 0.5 - pos.y * 0.5);
  return out;
}

fn sampleImage(uv: vec2f) -> vec3f {
  let clampedUv = clamp(uv, vec2f(0.0), vec2f(1.0));
  return textureSampleLevel(imageTexture, imageSampler, clampedUv, 0.0).rgb;
}

fn luminance(color: vec3f) -> f32 {
  return dot(color, vec3f(0.299, 0.587, 0.114));
}

fn applyContrast(color: vec3f, contrast: f32) -> vec3f {
  return clamp((color - vec3f(0.5)) * contrast + vec3f(0.5), vec3f(0.0), vec3f(1.0));
}

fn posterize(uv: vec2f) -> vec3f {
  var color = applyContrast(sampleImage(uv), uniforms.contrast);
  if (uniforms.grayscale > 0.5) {
    let value = luminance(color);
    color = vec3f(value);
  }

  let levels = max(uniforms.levels, 2.0);
  return floor(color * levels) / (levels - 1.0);
}

fn kuwahara(uv: vec2f) -> vec3f {
  let texel = 1.0 / uniforms.imageSize;
  let radius = i32(clamp(uniforms.radius, 1.0, 16.0));
  var bestMean = sampleImage(uv);
  var bestVariance = 99999.0;

  for (var quadrant = 0; quadrant < 4; quadrant = quadrant + 1) {
    var signX = -1.0;
    var signY = -1.0;
    if (quadrant == 1) {
      signX = 1.0;
    } else if (quadrant == 2) {
      signY = 1.0;
    } else if (quadrant == 3) {
      signX = 1.0;
      signY = 1.0;
    }

    var sum = vec3f(0.0);
    var sumSq = vec3f(0.0);
    var count = 0.0;

    for (var y = 0; y <= radius; y = y + 1) {
      for (var x = 0; x <= radius; x = x + 1) {
        let offset = vec2f(f32(x) * signX, f32(y) * signY);
        let color = sampleImage(uv + offset * texel);
        sum = sum + color;
        sumSq = sumSq + color * color;
        count = count + 1.0;
      }
    }

    let mean = sum / count;
    let variance = sumSq / count - mean * mean;
    let totalVariance = variance.r + variance.g + variance.b;
    if (totalVariance < bestVariance) {
      bestVariance = totalVariance;
      bestMean = mean;
    }
  }

  return bestMean;
}

fn edges(uv: vec2f) -> vec3f {
  let texel = 1.0 / uniforms.imageSize;
  let c00 = luminance(sampleImage(uv + vec2f(-1.0, -1.0) * texel));
  let c10 = luminance(sampleImage(uv + vec2f(0.0, -1.0) * texel));
  let c20 = luminance(sampleImage(uv + vec2f(1.0, -1.0) * texel));
  let c01 = luminance(sampleImage(uv + vec2f(-1.0, 0.0) * texel));
  let c21 = luminance(sampleImage(uv + vec2f(1.0, 0.0) * texel));
  let c02 = luminance(sampleImage(uv + vec2f(-1.0, 1.0) * texel));
  let c12 = luminance(sampleImage(uv + vec2f(0.0, 1.0) * texel));
  let c22 = luminance(sampleImage(uv + vec2f(1.0, 1.0) * texel));

  let gx = -c00 - 2.0 * c01 - c02 + c20 + 2.0 * c21 + c22;
  let gy = -c00 - 2.0 * c10 - c20 + c02 + 2.0 * c12 + c22;
  let edge = smoothstep(uniforms.edgeThreshold, uniforms.edgeThreshold + 0.18, length(vec2f(gx, gy)));
  let color = applyContrast(sampleImage(uv), uniforms.contrast);
  return mix(color, color * (1.0 - uniforms.edgeStrength), edge);
}

fn sobelGradient(uv: vec2f, sampleStep: f32) -> vec2f {
  let texel = sampleStep / uniforms.imageSize;
  let c00 = luminance(sampleImage(uv + vec2f(-1.0, -1.0) * texel));
  let c10 = luminance(sampleImage(uv + vec2f(0.0, -1.0) * texel));
  let c20 = luminance(sampleImage(uv + vec2f(1.0, -1.0) * texel));
  let c01 = luminance(sampleImage(uv + vec2f(-1.0, 0.0) * texel));
  let c21 = luminance(sampleImage(uv + vec2f(1.0, 0.0) * texel));
  let c02 = luminance(sampleImage(uv + vec2f(-1.0, 1.0) * texel));
  let c12 = luminance(sampleImage(uv + vec2f(0.0, 1.0) * texel));
  let c22 = luminance(sampleImage(uv + vec2f(1.0, 1.0) * texel));

  let gx = -c00 - 2.0 * c01 - c02 + c20 + 2.0 * c21 + c22;
  let gy = -c00 - 2.0 * c10 - c20 + c02 + 2.0 * c12 + c22;
  return vec2f(gx, gy);
}

fn hash21(value: vec2f) -> f32 {
  return fract(sin(dot(value, vec2f(127.1, 311.7))) * 43758.5453123);
}

fn quantizeColor(color: vec3f, merge: f32) -> vec3f {
  let levels = mix(48.0, 7.0, clamp(merge, 0.0, 1.0));
  let quantized = floor(color * levels + vec3f(0.5)) / levels;
  return mix(color, quantized, clamp(merge, 0.0, 1.0));
}

fn flowAverage(uv: vec2f, tangent: vec2f, normal: vec2f, halfLength: f32, halfWidth: f32, centerColor: vec3f) -> vec3f {
  let texel = 1.0 / uniforms.imageSize;
  var sum = vec3f(0.0);
  var totalWeight = 0.0;

  for (var y = -2; y <= 2; y = y + 1) {
    let across = f32(y) / 2.0;
    let widthWeight = exp(-across * across * 2.25);
    for (var x = -6; x <= 6; x = x + 1) {
      let along = f32(x) / 6.0;
      let lengthWeight = exp(-along * along * 1.65);
      let offset = tangent * along * halfLength + normal * across * halfWidth;
      let sampleColor = sampleImage(uv + offset * texel);
      let colorDelta = length(sampleColor - centerColor);
      let colorWeight = 1.0 / (1.0 + colorDelta * colorDelta * 18.0 * clamp(uniforms.flowEdgePreserve, 0.0, 1.0));
      let weight = lengthWeight * widthWeight * colorWeight;
      sum = sum + sampleColor * weight;
      totalWeight = totalWeight + weight;
    }
  }

  return sum / max(totalWeight, 0.0001);
}

fn strokeMask(pixel: vec2f, tangent: vec2f, normal: vec2f, halfLength: f32, halfWidth: f32) -> f32 {
  let lengthSpacing = max(8.0, halfLength * 1.55);
  let widthSpacing = max(2.0, halfWidth * 2.4);
  let flowCoord = vec2f(dot(pixel, tangent), dot(pixel, normal));
  let cell = floor(flowCoord / vec2f(lengthSpacing, widthSpacing));
  let stagger = hash21(cell);
  let shifted = flowCoord + vec2f(stagger * lengthSpacing * 0.5, hash21(cell + vec2f(19.7, 3.1)) * widthSpacing * 0.52);
  let local = fract(shifted / vec2f(lengthSpacing, widthSpacing)) - vec2f(0.5);
  let raggedWidth = 0.2 + hash21(cell + vec2f(4.3, 8.9)) * 0.22;
  let raggedLength = 0.32 + hash21(cell + vec2f(13.7, 2.4)) * 0.16;
  let capsule = max(abs(local.x) / raggedLength, abs(local.y) / raggedWidth);
  let mask = 1.0 - smoothstep(0.78, 1.04, capsule);
  let dryBreak = smoothstep(0.18, 0.72, hash21(cell + floor(local * vec2f(17.0, 5.0))));
  return mask * mix(0.58, 1.0, dryBreak);
}

fn flowBrush(uv: vec2f) -> vec3f {
  let base = sampleImage(uv);
  let gradient = sobelGradient(uv, max(1.0, uniforms.flowSmoothing));
  let edgeMagnitude = length(gradient);
  let gradientTangent = vec2f(-gradient.y, gradient.x) / max(edgeMagnitude, 0.0001);
  let tangent = select(vec2f(1.0, 0.0), gradientTangent, edgeMagnitude > 0.0001);
  let normal = vec2f(-tangent.y, tangent.x);
  let edge = smoothstep(uniforms.edgeThreshold, uniforms.edgeThreshold + 0.22, edgeMagnitude);
  let edgeLimiter = mix(1.0, 0.34, edge * clamp(uniforms.flowEdgePreserve, 0.0, 1.0));
  let halfLength = clamp(uniforms.flowBrushLength, 2.0, 72.0) * edgeLimiter;
  let halfWidth = clamp(uniforms.flowBrushWidth, 1.0, 18.0);

  let pixel = uv * uniforms.imageSize;
  let broad = quantizeColor(flowAverage(uv, tangent, normal, halfLength * 1.35, halfWidth * 1.8, base), uniforms.flowColorMerge);
  let strokeColor = quantizeColor(flowAverage(uv, tangent, normal, halfLength, halfWidth, base), uniforms.flowColorMerge);
  let mask = strokeMask(pixel, tangent, normal, halfLength, halfWidth);
  var color = mix(broad, strokeColor, mask);

  let along = dot(pixel, tangent);
  let across = dot(pixel, normal);
  let bristle = hash21(vec2f(floor(along / max(2.0, halfLength * 0.18)), floor(across / max(1.0, halfWidth * 0.38)))) - 0.5;
  let ridge = sin(along * 0.42 + hash21(floor(vec2f(across / max(1.0, halfWidth), along / max(4.0, halfLength)))) * 6.28318);
  color = color * (1.0 + (bristle * 0.2 + ridge * 0.045 * mask) * clamp(uniforms.flowStrokeStrength, 0.0, 1.0));

  let detailAccent = clamp(edge * uniforms.flowEdgePreserve, 0.0, 1.0);
  color = mix(color, base, detailAccent * 0.22);
  color = mix(base, color, clamp(uniforms.flowStrokeStrength, 0.0, 1.0));
  return applyContrast(color, uniforms.contrast);
}

fn sampleSegments(uv: vec2f) -> vec3f {
  let pixel = vec2u(clamp(uv * uniforms.imageSize, vec2f(0.0), uniforms.imageSize - vec2f(1.0)));
  return textureLoad(segmentTexture, pixel, 0).rgb;
}

@fragment
fn fragmentMain(input: VertexOut) -> @location(0) vec4f {
  var color = sampleImage(input.uv);

  if (uniforms.filterMode > 0.5 && uniforms.filterMode < 1.5) {
    color = kuwahara(input.uv);
  } else if (uniforms.filterMode > 1.5 && uniforms.filterMode < 2.5) {
    color = posterize(input.uv);
  } else if (uniforms.filterMode > 2.5 && uniforms.filterMode < 3.5) {
    color = edges(input.uv);
  } else if (uniforms.filterMode > 3.5 && uniforms.filterMode < 4.5) {
    color = flowBrush(input.uv);
  } else if (uniforms.filterMode > 4.5 && uniforms.filterMode < 10.5) {
    color = sampleSegments(input.uv);
  }

  return vec4f(clamp(color, vec3f(0.0), vec3f(1.0)), 1.0);
}
`

const SLIC_SHADER = /* wgsl */ `
struct Params {
  imageSize: vec2u,
  gridWidth: u32,
  centerCount: u32,
  step: f32,
  compactness: f32,
  saturationPreserve: f32,
  pad1: f32,
};

struct Center {
  x: f32,
  y: f32,
  r: f32,
  g: f32,
  b: f32,
  saturation: f32,
  pad1: f32,
  pad2: f32,
};

@group(0) @binding(0) var imageTexture: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> centers: array<Center>;
@group(0) @binding(2) var<storage, read_write> labels: array<u32>;
@group(0) @binding(3) var<storage, read_write> distances: array<f32>;
@group(0) @binding(4) var<storage, read_write> sums: array<atomic<u32>>;
@group(0) @binding(5) var outputTexture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(6) var<uniform> params: Params;

fn pixelIndex(x: u32, y: u32) -> u32 {
  return x + y * params.imageSize.x;
}

fn centerIndex(cellX: i32, cellY: i32) -> u32 {
  return u32(cellX) + u32(cellY) * params.gridWidth;
}

fn rgbToHsv(color: vec3f) -> vec3f {
  let maxChannel = max(max(color.r, color.g), color.b);
  let minChannel = min(min(color.r, color.g), color.b);
  let chroma = maxChannel - minChannel;
  var hue = 0.0;

  if (chroma > 0.0001) {
    if (maxChannel == color.r) {
      let redHue = (color.g - color.b) / chroma;
      hue = redHue - floor(redHue / 6.0) * 6.0;
    } else if (maxChannel == color.g) {
      hue = (color.b - color.r) / chroma + 2.0;
    } else {
      hue = (color.r - color.g) / chroma + 4.0;
    }
    hue = fract(hue / 6.0);
  }

  let saturation = select(0.0, chroma / maxChannel, maxChannel > 0.0001);
  return vec3f(hue, saturation, maxChannel);
}

fn hsvToRgb(hsv: vec3f) -> vec3f {
  let hue = fract(hsv.x) * 6.0;
  let chroma = hsv.z * hsv.y;
  let hueRemainder = hue - floor(hue / 2.0) * 2.0;
  let secondary = chroma * (1.0 - abs(hueRemainder - 1.0));
  let matchValue = hsv.z - chroma;
  var color = vec3f(0.0);

  if (hue < 1.0) {
    color = vec3f(chroma, secondary, 0.0);
  } else if (hue < 2.0) {
    color = vec3f(secondary, chroma, 0.0);
  } else if (hue < 3.0) {
    color = vec3f(0.0, chroma, secondary);
  } else if (hue < 4.0) {
    color = vec3f(0.0, secondary, chroma);
  } else if (hue < 5.0) {
    color = vec3f(secondary, 0.0, chroma);
  } else {
    color = vec3f(chroma, 0.0, secondary);
  }

  return color + vec3f(matchValue);
}

@compute @workgroup_size(16, 16)
fn initLabels(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= params.imageSize.x || id.y >= params.imageSize.y) {
    return;
  }

  let index = pixelIndex(id.x, id.y);
  labels[index] = 0u;
  distances[index] = 3.402823e38;
}

@compute @workgroup_size(64)
fn clearSums(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= params.centerCount * 7u) {
    return;
  }

  atomicStore(&sums[id.x], 0u);
}

@compute @workgroup_size(16, 16)
fn assignPixels(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= params.imageSize.x || id.y >= params.imageSize.y) {
    return;
  }

  let index = pixelIndex(id.x, id.y);
  let color = textureLoad(imageTexture, vec2u(id.xy), 0).rgb * 255.0;
  let cellX = i32(floor(f32(id.x) / params.step));
  let cellY = i32(floor(f32(id.y) / params.step));
  let spatialWeight = (params.compactness * params.compactness) / (params.step * params.step);

  var bestLabel = 0u;
  var bestDistance = 3.402823e38;
  for (var y = cellY - 2; y <= cellY + 2; y = y + 1) {
    for (var x = cellX - 2; x <= cellX + 2; x = x + 1) {
      if (x < 0 || y < 0) {
        continue;
      }

      let label = centerIndex(x, y);
      if (label >= params.centerCount) {
        continue;
      }

      let center = centers[label];
      let dr = color.r - center.r;
      let dg = color.g - center.g;
      let db = color.b - center.b;
      let dx = f32(id.x) - center.x;
      let dy = f32(id.y) - center.y;
      let distance = dr * dr + dg * dg + db * db + (dx * dx + dy * dy) * spatialWeight;
      if (distance < bestDistance) {
        bestDistance = distance;
        bestLabel = label;
      }
    }
  }

  labels[index] = bestLabel;
  distances[index] = bestDistance;
}

@compute @workgroup_size(16, 16)
fn accumulateCenters(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= params.imageSize.x || id.y >= params.imageSize.y) {
    return;
  }

  let index = pixelIndex(id.x, id.y);
  let label = labels[index];
  let color = textureLoad(imageTexture, vec2u(id.xy), 0).rgb * 255.0;
  let saturation = rgbToHsv(color / 255.0).y;
  let offset = label * 7u;

  atomicAdd(&sums[offset], id.x);
  atomicAdd(&sums[offset + 1u], id.y);
  atomicAdd(&sums[offset + 2u], u32(round(color.r)));
  atomicAdd(&sums[offset + 3u], u32(round(color.g)));
  atomicAdd(&sums[offset + 4u], u32(round(color.b)));
  atomicAdd(&sums[offset + 5u], u32(round(saturation * 255.0)));
  atomicAdd(&sums[offset + 6u], 1u);
}

@compute @workgroup_size(64)
fn updateCenters(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= params.centerCount) {
    return;
  }

  let offset = id.x * 7u;
  let count = atomicLoad(&sums[offset + 6u]);
  if (count == 0u) {
    return;
  }

  let countF = f32(count);
  centers[id.x].x = f32(atomicLoad(&sums[offset])) / countF;
  centers[id.x].y = f32(atomicLoad(&sums[offset + 1u])) / countF;
  centers[id.x].r = f32(atomicLoad(&sums[offset + 2u])) / countF;
  centers[id.x].g = f32(atomicLoad(&sums[offset + 3u])) / countF;
  centers[id.x].b = f32(atomicLoad(&sums[offset + 4u])) / countF;
  centers[id.x].saturation = f32(atomicLoad(&sums[offset + 5u])) / (countF * 255.0);
}

@compute @workgroup_size(16, 16)
fn colorize(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= params.imageSize.x || id.y >= params.imageSize.y) {
    return;
  }

  let index = pixelIndex(id.x, id.y);
  let center = centers[labels[index]];
  var hsv = rgbToHsv(vec3f(center.r, center.g, center.b) / 255.0);
  hsv.y = clamp(mix(hsv.y, center.saturation, clamp(params.saturationPreserve, 0.0, 1.0)), 0.0, 1.0);
  textureStore(outputTexture, vec2u(id.xy), vec4f(hsvToRgb(hsv), 1.0));
}
`

const FELZENSZWALB_SHADER = /* wgsl */ `
struct Params {
  imageSize: vec2u,
  horizontalOffset: u32,
  verticalOffset: u32,
  diagonalDownOffset: u32,
  diagonalUpOffset: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
};

@group(0) @binding(0) var imageTexture: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> weights: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

fn edgeWeight(a: vec3f, b: vec3f) -> f32 {
  let delta = (a - b) * 255.0;
  return length(delta);
}

@compute @workgroup_size(16, 16)
fn computeWeights(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= params.imageSize.x || id.y >= params.imageSize.y) {
    return;
  }

  let color = textureLoad(imageTexture, vec2u(id.xy), 0).rgb;
  let horizontalCount = params.imageSize.x - 1u;

  if (id.x + 1u < params.imageSize.x) {
    let right = textureLoad(imageTexture, vec2u(id.x + 1u, id.y), 0).rgb;
    weights[params.horizontalOffset + id.x + id.y * horizontalCount] = edgeWeight(color, right);
  }

  if (id.y + 1u < params.imageSize.y) {
    let down = textureLoad(imageTexture, vec2u(id.x, id.y + 1u), 0).rgb;
    weights[params.verticalOffset + id.x + id.y * params.imageSize.x] = edgeWeight(color, down);
  }

  if (id.x + 1u < params.imageSize.x && id.y + 1u < params.imageSize.y) {
    let diagonalDown = textureLoad(imageTexture, vec2u(id.x + 1u, id.y + 1u), 0).rgb;
    weights[params.diagonalDownOffset + id.x + id.y * horizontalCount] = edgeWeight(color, diagonalDown);
  }

  if (id.x + 1u < params.imageSize.x && id.y > 0u) {
    let diagonalUp = textureLoad(imageTexture, vec2u(id.x + 1u, id.y - 1u), 0).rgb;
    weights[params.diagonalUpOffset + id.x + (id.y - 1u) * horizontalCount] = edgeWeight(color, diagonalUp);
  }
}
`

export class WebGpuImageRenderer {
  private readonly canvas: HTMLCanvasElement
  private adapter?: GPUAdapter
  private device?: GPUDevice
  private context?: GPUCanvasContext
  private pipeline?: GPURenderPipeline
  private slicPipelines?: SlicResources
  private slicBindGroupLayout?: GPUBindGroupLayout
  private felzenszwalbPipelines?: FelzenszwalbResources
  private felzenszwalbBindGroupLayout?: GPUBindGroupLayout
  private bindGroup?: GPUBindGroup
  private texture?: GPUTexture
  private segmentTexture?: GPUTexture
  private sampler?: GPUSampler
  private uniformBuffer?: GPUBuffer
  private format?: GPUTextureFormat
  private onError?: (message: string) => void
  private image?: LoadedImage
  private segmentCacheKey = ''

  constructor(canvas: HTMLCanvasElement, onError?: (message: string) => void) {
    this.canvas = canvas
    this.onError = onError
  }

  async init() {
    if (!navigator.gpu) {
      throw new Error('WebGPU is not available in this browser.')
    }

    const adapter = await navigator.gpu.requestAdapter()
    if (!adapter) {
      throw new Error('No compatible GPU adapter was found.')
    }
    this.adapter = adapter

    this.device = await this.adapter.requestDevice()
    this.device.addEventListener('uncapturederror', (event) => {
      this.onError?.(event.error.message)
    })

    this.context = (this.canvas.getContext('webgpu') as GPUCanvasContext | null) ?? undefined
    if (!this.context) {
      throw new Error('Could not create a WebGPU canvas context.')
    }

    this.format = navigator.gpu.getPreferredCanvasFormat()
    this.configureContext()

    this.uniformBuffer = this.device.createBuffer({
      size: 80,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })

    this.sampler = this.device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    })

    const module = this.device.createShaderModule({ code: SHADER })
    const compilation = await module.getCompilationInfo()
    const shaderErrors = compilation.messages
      .filter((message) => message.type === 'error')
      .map((message) => message.message)
    if (shaderErrors.length > 0) {
      throw new Error(shaderErrors.join('\n'))
    }

    this.pipeline = await this.device.createRenderPipelineAsync({
      layout: 'auto',
      vertex: {
        module,
        entryPoint: 'vertexMain',
      },
      fragment: {
        module,
        entryPoint: 'fragmentMain',
        targets: [{ format: this.format }],
      },
      primitive: {
        topology: 'triangle-list',
      },
    })

    const slicModule = this.device.createShaderModule({ code: SLIC_SHADER })
    const slicCompilation = await slicModule.getCompilationInfo()
    const slicShaderErrors = slicCompilation.messages
      .filter((message) => message.type === 'error')
      .map((message) => message.message)
    if (slicShaderErrors.length > 0) {
      throw new Error(slicShaderErrors.join('\n'))
    }

    this.slicBindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        {
          binding: 5,
          visibility: GPUShaderStage.COMPUTE,
          storageTexture: { access: 'write-only', format: 'rgba8unorm' },
        },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    })
    const slicPipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts: [this.slicBindGroupLayout],
    })

    this.slicPipelines = {
      initLabels: await this.device.createComputePipelineAsync({
        layout: slicPipelineLayout,
        compute: { module: slicModule, entryPoint: 'initLabels' },
      }),
      clearSums: await this.device.createComputePipelineAsync({
        layout: slicPipelineLayout,
        compute: { module: slicModule, entryPoint: 'clearSums' },
      }),
      assignPixels: await this.device.createComputePipelineAsync({
        layout: slicPipelineLayout,
        compute: { module: slicModule, entryPoint: 'assignPixels' },
      }),
      accumulateCenters: await this.device.createComputePipelineAsync({
        layout: slicPipelineLayout,
        compute: { module: slicModule, entryPoint: 'accumulateCenters' },
      }),
      updateCenters: await this.device.createComputePipelineAsync({
        layout: slicPipelineLayout,
        compute: { module: slicModule, entryPoint: 'updateCenters' },
      }),
      colorize: await this.device.createComputePipelineAsync({
        layout: slicPipelineLayout,
        compute: { module: slicModule, entryPoint: 'colorize' },
      }),
    }

    const felzenszwalbModule = this.device.createShaderModule({ code: FELZENSZWALB_SHADER })
    const felzenszwalbCompilation = await felzenszwalbModule.getCompilationInfo()
    const felzenszwalbShaderErrors = felzenszwalbCompilation.messages
      .filter((message) => message.type === 'error')
      .map((message) => message.message)
    if (felzenszwalbShaderErrors.length > 0) {
      throw new Error(felzenszwalbShaderErrors.join('\n'))
    }

    this.felzenszwalbBindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    })
    const felzenszwalbPipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts: [this.felzenszwalbBindGroupLayout],
    })
    this.felzenszwalbPipelines = {
      computeWeights: await this.device.createComputePipelineAsync({
        layout: felzenszwalbPipelineLayout,
        compute: { module: felzenszwalbModule, entryPoint: 'computeWeights' },
      }),
    }
  }

  loadImage(image: LoadedImage) {
    const device = this.requireDevice()
    this.texture?.destroy()
    this.segmentTexture?.destroy()
    this.image = image
    this.segmentCacheKey = ''
    this.texture = device.createTexture({
      size: [image.width, image.height],
      format: 'rgba8unorm',
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT |
        GPUTextureUsage.STORAGE_BINDING,
    })

    device.queue.copyExternalImageToTexture(
      { source: image.bitmap },
      { texture: this.texture },
      [image.width, image.height],
    )

    this.segmentTexture = this.createTextureFromPixels(image.pixels, image.width, image.height)
    this.canvas.width = image.width
    this.canvas.height = image.height
    this.canvas.style.aspectRatio = `${image.width} / ${image.height}`
    this.configureContext()
    this.createBindGroup()
  }

  async render(settings: FilterSettings) {
    const device = this.requireDevice()
    if (!this.context || !this.pipeline || !this.bindGroup || !this.uniformBuffer) {
      throw new Error('The image renderer is not ready yet.')
    }

    device.pushErrorScope('validation')
    await this.updateSegments(settings)
    this.writeUniforms(settings)
    const encoder = device.createCommandEncoder()
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.context.getCurrentTexture().createView(),
          clearValue: { r: 0.07, g: 0.07, b: 0.07, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    })

    pass.setPipeline(this.pipeline)
    pass.setBindGroup(0, this.bindGroup)
    pass.draw(3)
    pass.end()
    device.queue.submit([encoder.finish()])
    const validationError = await device.popErrorScope()
    if (validationError) {
      throw new Error(validationError.message)
    }
  }

  async exportBlob(type: 'image/png' | 'image/jpeg', quality?: number) {
    await this.device?.queue.onSubmittedWorkDone()

    return new Promise<Blob>((resolve, reject) => {
      this.canvas.toBlob(
        (blob) => {
          if (blob) {
            resolve(blob)
          } else {
            reject(new Error('Could not export the canvas image.'))
          }
        },
        type,
        quality,
      )
    })
  }

  private createBindGroup() {
    const device = this.requireDevice()
    if (!this.pipeline || !this.texture || !this.segmentTexture || !this.sampler || !this.uniformBuffer) {
      return
    }

    this.bindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.texture.createView() },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: { buffer: this.uniformBuffer } },
        { binding: 3, resource: this.segmentTexture.createView() },
      ],
    })
  }

  private createTextureFromPixels(pixels: Uint8ClampedArray, width: number, height: number) {
    const device = this.requireDevice()
    const texture = device.createTexture({
      size: [width, height],
      format: 'rgba8unorm',
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT |
        GPUTextureUsage.STORAGE_BINDING,
    })

    device.queue.writeTexture(
      { texture },
      pixels,
      { bytesPerRow: width * 4, rowsPerImage: height },
      [width, height],
    )

    return texture
  }

  private async updateSegments(settings: FilterSettings) {
    if (
      !this.image ||
      (settings.filter !== 'slic' &&
        settings.filter !== 'adaptiveBrush' &&
        settings.filter !== 'painter' &&
        settings.filter !== 'colorCheck')
    ) {
      return
    }

    const key = this.getSegmentCacheKey(settings)
    if (key === this.segmentCacheKey) {
      return
    }

    if (settings.filter === 'slic') {
      await this.createSlicSegmentsGpu(settings.slicSize, settings.slicCompactness, settings.slicSaturationPreserve)
    } else if (settings.filter === 'colorCheck') {
      const edgeWeights = await this.createFelzenszwalbWeightsGpu()
      const pixels = createColorCheckPixels(this.image, settings, edgeWeights)
      this.segmentTexture?.destroy()
      this.segmentTexture = this.createTextureFromPixels(pixels, this.image.width, this.image.height)
    } else if (settings.filter === 'adaptiveBrush') {
      const pixels = createAdaptiveBrushSegments(
        this.image,
        settings.adaptiveLargeBrush,
        settings.adaptiveMediumBrush,
        settings.adaptiveSmallBrush,
        settings.adaptiveDetailThreshold,
        settings.adaptiveDetailStrength,
      )
      this.segmentTexture?.destroy()
      this.segmentTexture = this.createTextureFromPixels(pixels, this.image.width, this.image.height)
    } else {
      const pixels = createPainterPixels(this.image, settings)
      this.segmentTexture?.destroy()
      this.segmentTexture = this.createTextureFromPixels(pixels, this.image.width, this.image.height)
    }

    this.segmentCacheKey = key
    this.createBindGroup()
  }

  private getSegmentCacheKey(settings: FilterSettings) {
    if (settings.filter === 'slic') {
      return `slic:${settings.slicSize}:${settings.slicCompactness}:${settings.slicSaturationPreserve}`
    }

    if (settings.filter === 'colorCheck') {
      return `colorCheck:${settings.colorCheckScale}:${settings.colorCheckMinSize}:${settings.colorCheckSaturation}:${settings.colorCheckSeparator}:${settings.colorCheckPerceptual}`
    }

    if (settings.filter === 'painter') {
      return [
        'painter',
        settings.painterAbstraction,
        settings.painterSize,
        settings.painterLength,
        settings.painterDetail,
        settings.painterEdgeFidelity,
        settings.painterColorSimplification,
        settings.painterTexture,
        settings.painterLayers,
        settings.painterUnderpainting,
        settings.painterDetailSource,
      ].join(':')
    }

    return [
      'adaptiveBrush',
      settings.adaptiveLargeBrush,
      settings.adaptiveMediumBrush,
      settings.adaptiveSmallBrush,
      settings.adaptiveDetailThreshold,
      settings.adaptiveDetailStrength,
    ].join(':')
  }

  private async createSlicSegmentsGpu(requestedSize: number, requestedCompactness: number, requestedSaturationPreserve: number) {
    const device = this.requireDevice()
    if (!this.image || !this.texture || !this.segmentTexture || !this.slicPipelines || !this.slicBindGroupLayout) {
      return
    }

    const { width, height, pixels } = this.image
    const step = Math.max(8, Math.round(requestedSize))
    const compactness = Math.max(1, requestedCompactness)
    const saturationPreserve = clamp(requestedSaturationPreserve, 0, 1)
    const gridWidth = Math.max(1, Math.ceil(width / step))
    const gridHeight = Math.max(1, Math.ceil(height / step))
    const centerCount = gridWidth * gridHeight
    const centers = new Float32Array(centerCount * 8)

    for (let centerY = 0; centerY < gridHeight; centerY += 1) {
      for (let centerX = 0; centerX < gridWidth; centerX += 1) {
        const centerIndex = centerX + centerY * gridWidth
        const x = Math.min(width - 1, Math.round(centerX * step + step * 0.5))
        const y = Math.min(height - 1, Math.round(centerY * step + step * 0.5))
        const pixelOffset = (x + y * width) * 4
        const offset = centerIndex * 8
        centers[offset] = x
        centers[offset + 1] = y
        centers[offset + 2] = pixels[pixelOffset]
        centers[offset + 3] = pixels[pixelOffset + 1]
        centers[offset + 4] = pixels[pixelOffset + 2]
      }
    }

    const centerBuffer = device.createBuffer({
      size: centers.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
    const labelBuffer = device.createBuffer({
      size: width * height * 4,
      usage: GPUBufferUsage.STORAGE,
    })
    const distanceBuffer = device.createBuffer({
      size: width * height * 4,
      usage: GPUBufferUsage.STORAGE,
    })
    const sumBuffer = device.createBuffer({
      size: centerCount * 7 * 4,
      usage: GPUBufferUsage.STORAGE,
    })
    const paramsBuffer = device.createBuffer({
      size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })

    const paramsData = new ArrayBuffer(48)
    const paramsU32 = new Uint32Array(paramsData)
    const paramsF32 = new Float32Array(paramsData)
    paramsU32[0] = width
    paramsU32[1] = height
    paramsU32[2] = gridWidth
    paramsU32[3] = centerCount
    paramsF32[4] = step
    paramsF32[5] = compactness
    paramsF32[6] = saturationPreserve

    device.queue.writeBuffer(centerBuffer, 0, centers)
    device.queue.writeBuffer(paramsBuffer, 0, paramsData)

    const bindGroup = device.createBindGroup({
      layout: this.slicBindGroupLayout,
      entries: [
        { binding: 0, resource: this.texture.createView() },
        { binding: 1, resource: { buffer: centerBuffer } },
        { binding: 2, resource: { buffer: labelBuffer } },
        { binding: 3, resource: { buffer: distanceBuffer } },
        { binding: 4, resource: { buffer: sumBuffer } },
        { binding: 5, resource: this.segmentTexture.createView() },
        { binding: 6, resource: { buffer: paramsBuffer } },
      ],
    })

    const pixelWorkgroupsX = Math.ceil(width / 16)
    const pixelWorkgroupsY = Math.ceil(height / 16)
    const centerWorkgroups = Math.ceil(centerCount / 64)
    const sumWorkgroups = Math.ceil((centerCount * 7) / 64)
    const encoder = device.createCommandEncoder()

    for (let iteration = 0; iteration < 6; iteration += 1) {
      this.dispatchSlicPass(encoder, this.slicPipelines.initLabels, bindGroup, pixelWorkgroupsX, pixelWorkgroupsY)
      this.dispatchSlicPass(encoder, this.slicPipelines.clearSums, bindGroup, sumWorkgroups, 1)
      this.dispatchSlicPass(encoder, this.slicPipelines.assignPixels, bindGroup, pixelWorkgroupsX, pixelWorkgroupsY)
      this.dispatchSlicPass(encoder, this.slicPipelines.accumulateCenters, bindGroup, pixelWorkgroupsX, pixelWorkgroupsY)
      this.dispatchSlicPass(encoder, this.slicPipelines.updateCenters, bindGroup, centerWorkgroups, 1)
    }

    this.dispatchSlicPass(encoder, this.slicPipelines.colorize, bindGroup, pixelWorkgroupsX, pixelWorkgroupsY)
    device.queue.submit([encoder.finish()])
    await device.queue.onSubmittedWorkDone()

    centerBuffer.destroy()
    labelBuffer.destroy()
    distanceBuffer.destroy()
    sumBuffer.destroy()
    paramsBuffer.destroy()
  }

  private dispatchSlicPass(
    encoder: GPUCommandEncoder,
    pipeline: GPUComputePipeline,
    bindGroup: GPUBindGroup,
    workgroupsX: number,
    workgroupsY: number,
  ) {
    const pass = encoder.beginComputePass()
    pass.setPipeline(pipeline)
    pass.setBindGroup(0, bindGroup)
    pass.dispatchWorkgroups(workgroupsX, workgroupsY)
    pass.end()
  }

  private async createFelzenszwalbWeightsGpu() {
    const device = this.requireDevice()
    if (!this.image || !this.texture || !this.felzenszwalbPipelines || !this.felzenszwalbBindGroupLayout) {
      return undefined
    }

    const { width, height } = this.image
    const horizontalCount = Math.max(0, (width - 1) * height)
    const verticalCount = Math.max(0, width * (height - 1))
    const diagonalCount = Math.max(0, (width - 1) * (height - 1))
    const edgeCount = horizontalCount + verticalCount + 2 * diagonalCount
    if (edgeCount === 0) {
      return new Float32Array()
    }

    const weightBuffer = device.createBuffer({
      size: edgeCount * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    })
    const readBuffer = device.createBuffer({
      size: edgeCount * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    })
    const paramsBuffer = device.createBuffer({
      size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    const paramsData = new ArrayBuffer(48)
    const paramsU32 = new Uint32Array(paramsData)
    paramsU32[0] = width
    paramsU32[1] = height
    paramsU32[2] = 0
    paramsU32[3] = horizontalCount
    paramsU32[4] = horizontalCount + verticalCount
    paramsU32[5] = horizontalCount + verticalCount + diagonalCount
    device.queue.writeBuffer(paramsBuffer, 0, paramsData)

    const bindGroup = device.createBindGroup({
      layout: this.felzenszwalbBindGroupLayout,
      entries: [
        { binding: 0, resource: this.texture.createView() },
        { binding: 1, resource: { buffer: weightBuffer } },
        { binding: 2, resource: { buffer: paramsBuffer } },
      ],
    })

    const encoder = device.createCommandEncoder()
    const pass = encoder.beginComputePass()
    pass.setPipeline(this.felzenszwalbPipelines.computeWeights)
    pass.setBindGroup(0, bindGroup)
    pass.dispatchWorkgroups(Math.ceil(width / 16), Math.ceil(height / 16))
    pass.end()
    encoder.copyBufferToBuffer(weightBuffer, 0, readBuffer, 0, edgeCount * 4)
    device.queue.submit([encoder.finish()])

    await readBuffer.mapAsync(GPUMapMode.READ)
    const weights = new Float32Array(readBuffer.getMappedRange()).slice()
    readBuffer.unmap()
    weightBuffer.destroy()
    readBuffer.destroy()
    paramsBuffer.destroy()

    return weights
  }

  private configureContext() {
    if (!this.context || !this.device || !this.format) {
      return
    }

    this.context.configure({
      device: this.device,
      format: this.format,
      alphaMode: 'opaque',
    })
  }

  private writeUniforms(settings: FilterSettings) {
    if (!this.device || !this.uniformBuffer) {
      return
    }

    const data = new Float32Array([
      this.canvas.width,
      this.canvas.height,
      settings.radius,
      FILTER_INDEX[settings.filter],
      settings.levels,
      settings.grayscale ? 1 : 0,
      settings.contrast,
      settings.edgeThreshold,
      settings.edgeStrength,
      settings.slicSize,
      settings.flowBrushLength,
      settings.flowBrushWidth,
      settings.flowSmoothing,
      settings.flowColorMerge,
      settings.flowEdgePreserve,
      settings.flowStrokeStrength,
      0,
      0,
      0,
      0,
    ])

    this.device.queue.writeBuffer(this.uniformBuffer, 0, data)
  }

  private requireDevice() {
    if (!this.device) {
      throw new Error('WebGPU renderer has not been initialized.')
    }

    return this.device
  }
}

export async function loadImageFile(file: File): Promise<LoadedImage> {
  const sourceBitmap = await createImageBitmap(file, {
    imageOrientation: 'from-image',
    premultiplyAlpha: 'none',
  })

  const maxDimension = Math.max(sourceBitmap.width, sourceBitmap.height)
  if (maxDimension <= MAX_WORKING_IMAGE_DIMENSION) {
    const pixels = getImagePixels(sourceBitmap)
    return {
      bitmap: sourceBitmap,
      name: file.name,
      width: sourceBitmap.width,
      height: sourceBitmap.height,
      pixels,
    }
  }

  const scale = MAX_WORKING_IMAGE_DIMENSION / maxDimension
  const resizeWidth = Math.max(1, Math.round(sourceBitmap.width * scale))
  const resizeHeight = Math.max(1, Math.round(sourceBitmap.height * scale))
  sourceBitmap.close()

  const bitmap = await createImageBitmap(file, {
    imageOrientation: 'from-image',
    premultiplyAlpha: 'none',
    resizeWidth,
    resizeHeight,
    resizeQuality: 'high',
  })

  return {
    bitmap,
    name: file.name,
    width: bitmap.width,
    height: bitmap.height,
    pixels: getImagePixels(bitmap),
  }
}

function getImagePixels(bitmap: ImageBitmap) {
  const canvas = document.createElement('canvas')
  canvas.width = bitmap.width
  canvas.height = bitmap.height
  const context = canvas.getContext('2d', { willReadFrequently: true })
  if (!context) {
    throw new Error('Could not read image pixels.')
  }

  context.drawImage(bitmap, 0, 0)
  return context.getImageData(0, 0, bitmap.width, bitmap.height).data
}

function computeFelzenszwalbLabels(
  image: LoadedImage,
  requestedScale: number,
  requestedMinSize: number,
  computedWeights?: Float32Array,
) {
  const { width, height, pixels } = image
  const count = width * height
  const horizontalCount = Math.max(0, (width - 1) * height)
  const verticalCount = Math.max(0, width * (height - 1))
  const diagonalCount = Math.max(0, (width - 1) * (height - 1))
  const edgeCount = horizontalCount + verticalCount + 2 * diagonalCount
  const edgeA = new Uint32Array(edgeCount)
  const edgeB = new Uint32Array(edgeCount)
  const weights = new Float32Array(edgeCount)
  const order = new Uint32Array(edgeCount)
  let edgeIndex = 0

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = x + y * width
      if (x + 1 < width) {
        addEdge(edgeA, edgeB, weights, order, edgeIndex, index, index + 1, pixels, computedWeights?.[x + y * (width - 1)])
        edgeIndex += 1
      }
      if (y + 1 < height) {
        addEdge(
          edgeA,
          edgeB,
          weights,
          order,
          edgeIndex,
          index,
          index + width,
          pixels,
          computedWeights?.[horizontalCount + x + y * width],
        )
        edgeIndex += 1
      }
      if (x + 1 < width && y + 1 < height) {
        addEdge(
          edgeA,
          edgeB,
          weights,
          order,
          edgeIndex,
          index,
          index + width + 1,
          pixels,
          computedWeights?.[horizontalCount + verticalCount + x + y * (width - 1)],
        )
        edgeIndex += 1
      }
      if (x + 1 < width && y > 0) {
        addEdge(
          edgeA,
          edgeB,
          weights,
          order,
          edgeIndex,
          index,
          index - width + 1,
          pixels,
          computedWeights?.[horizontalCount + verticalCount + diagonalCount + x + (y - 1) * (width - 1)],
        )
        edgeIndex += 1
      }
    }
  }

  order.sort((left, right) => weights[left] - weights[right])

  const scale = Math.max(10, requestedScale)
  const minSize = Math.max(2, Math.round(requestedMinSize))
  const forest = new SegmentForest(count)

  for (let i = 0; i < edgeCount; i += 1) {
    const edge = order[i]
    const a = forest.find(edgeA[edge])
    const b = forest.find(edgeB[edge])
    if (a === b) {
      continue
    }

    const weight = weights[edge]
    if (weight <= forest.internal[a] + scale / forest.size[a] && weight <= forest.internal[b] + scale / forest.size[b]) {
      forest.union(a, b, weight)
    }
  }

  for (let i = 0; i < edgeCount; i += 1) {
    const edge = order[i]
    const a = forest.find(edgeA[edge])
    const b = forest.find(edgeB[edge])
    if (a !== b && (forest.size[a] < minSize || forest.size[b] < minSize)) {
      forest.union(a, b, weights[edge])
    }
  }

  const labels = new Int32Array(count)
  const rootToLabel = new Map<number, number>()
  for (let index = 0; index < count; index += 1) {
    const root = forest.find(index)
    let label = rootToLabel.get(root)
    if (label === undefined) {
      label = rootToLabel.size
      rootToLabel.set(root, label)
    }
    labels[index] = label
  }

  return { labels, labelCount: rootToLabel.size }
}

function addEdge(
  edgeA: Uint32Array,
  edgeB: Uint32Array,
  weights: Float32Array,
  order: Uint32Array,
  edgeIndex: number,
  a: number,
  b: number,
  pixels: Uint8ClampedArray,
  computedWeight?: number,
) {
  const offsetA = a * 4
  const offsetB = b * 4
  const dr = pixels[offsetA] - pixels[offsetB]
  const dg = pixels[offsetA + 1] - pixels[offsetB + 1]
  const db = pixels[offsetA + 2] - pixels[offsetB + 2]

  edgeA[edgeIndex] = a
  edgeB[edgeIndex] = b
  weights[edgeIndex] = computedWeight ?? Math.sqrt(dr * dr + dg * dg + db * db)
  order[edgeIndex] = edgeIndex
}

function createAdaptiveBrushSegments(
  image: LoadedImage,
  requestedLargeBrush: number,
  requestedMediumBrush: number,
  requestedSmallBrush: number,
  requestedDetailThreshold: number,
  requestedDetailStrength: number,
) {
  const { width, height, pixels } = image
  const largeBrush = Math.max(2, Math.round(requestedLargeBrush))
  const mediumBrush = Math.max(2, Math.min(largeBrush, Math.round(requestedMediumBrush)))
  const smallBrush = Math.max(2, Math.min(mediumBrush, Math.round(requestedSmallBrush)))
  const detailThreshold = clamp(requestedDetailThreshold, 0, 1)
  const detailStrength = clamp(requestedDetailStrength, 0, 1)
  const largeField = createBrushField(image, largeBrush, 0x15a4)
  const mediumField = createBrushField(image, mediumBrush, 0x89f1)
  const smallField = createBrushField(image, smallBrush, 0x4c2d)
  const largeCellDetail = createCellDetailMap(pixels, width, largeField.labels, largeField.labelCount)
  const mediumCellDetail = createCellDetailMap(pixels, width, mediumField.labels, mediumField.labelCount)
  const output = new Uint8ClampedArray(pixels.length)

  for (let index = 0; index < width * height; index += 1) {
    const pixelOffset = index * 4
    const x = index % width
    const y = Math.floor(index / width)
    const largeDetail = largeCellDetail[largeField.labels[index]]
    const mediumDetail = mediumCellDetail[mediumField.labels[index]]
    const localDetail = pixelContrast(pixels, width, x, y)
    const useMediumBrush = largeDetail > detailThreshold
    const useSmallBrush = useMediumBrush && mediumDetail > Math.min(1, detailThreshold + 0.12)
    const source = useSmallBrush ? smallField.pixels : useMediumBrush ? mediumField.pixels : largeField.pixels
    const accentMix = useSmallBrush ? Math.min(0.5, smoothstep(0.03, 0.18, localDetail) * detailStrength) : 0

    output[pixelOffset] = mixChannel(source[pixelOffset], pixels[pixelOffset], accentMix)
    output[pixelOffset + 1] = mixChannel(source[pixelOffset + 1], pixels[pixelOffset + 1], accentMix)
    output[pixelOffset + 2] = mixChannel(source[pixelOffset + 2], pixels[pixelOffset + 2], accentMix)
    output[pixelOffset + 3] = 255
  }

  return output
}

// Unified painter: lays an underpainting base, then layered brush strokes whose
// density and size are driven either by structural edges or by local color
// activity. Replaces the former Stroke Painter and Detail Painter, which were
// the same algorithm differing only in underpainting style and detail source.
function createPainterPixels(image: LoadedImage, settings: FilterSettings) {
  const { width, height, pixels } = image
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const context = canvas.getContext('2d', { willReadFrequently: true })
  if (!context) {
    throw new Error('Could not create the painting canvas.')
  }

  const abstraction = clamp(settings.painterAbstraction, 0, 1)
  const detailAmount = clamp(settings.painterDetail, 0, 1)
  const edgeFidelity = clamp(settings.painterEdgeFidelity, 0, 1)
  const simplification = clamp(settings.painterColorSimplification, 0, 1)
  const textureStrength = clamp(settings.painterTexture, 0, 1)
  const baseSize = Math.max(3, Math.round(settings.painterSize))
  const lengthScale = clamp(settings.painterLength, 1, 5)
  const layers = Math.max(1, Math.min(4, Math.round(settings.painterLayers)))
  const useColorDetail = settings.painterDetailSource === 'color'

  const edgeMap = createPixelDetailMap(pixels, width, height)
  // Color mode drives finer passes toward shifting color; edge mode reuses the
  // structural edge map for that role.
  const activityMap = useColorDetail ? createColorActivityMap(pixels, width, height, baseSize) : edgeMap
  const flowField = createStrokeFlowField(pixels, width, height, Math.max(8, Math.round(baseSize * 2.2)))

  // Smooth base avoids square underpainting tiles showing through short strokes;
  // blocks give a more visibly chunky block-in.
  if (settings.painterUnderpainting === 'smooth') {
    paintSmoothUnderpainting(context, image, baseSize, abstraction, simplification)
  } else {
    paintUnderpainting(context, image, baseSize, abstraction, simplification)
  }

  const layerScale = [3.8, 2.05, 1.02, 0.52]
  for (let layer = 0; layer < layers; layer += 1) {
    const size = Math.max(2, baseSize * layerScale[layer])
    // Color mode tightens the grid as detail rises so busy areas get more marks.
    const spacing = useColorDetail
      ? Math.max(1.5, size * (layer === 0 ? 1.05 : 0.74 - detailAmount * 0.3))
      : Math.max(2, size * (layer === 0 ? 1.05 : 0.76))
    const fineLayer = layer / Math.max(1, layers - 1)
    const alpha = 0.82 - fineLayer * 0.18

    for (let y = -spacing; y < height + spacing; y += spacing) {
      for (let x = -spacing; x < width + spacing; x += spacing) {
        const jitterSeed = layer * 0x9e37 + Math.floor(x / spacing) * 193 + Math.floor(y / spacing) * 977
        const jitter = layer < 2 ? 0.42 : 0.58
        const px = Math.round(x + (hashToUnit(jitterSeed, layer, 0x2471) - 0.5) * spacing * jitter)
        const py = Math.round(y + (hashToUnit(jitterSeed, layer, 0x71b5) - 0.5) * spacing * jitter)
        if (px < 0 || py < 0 || px >= width || py >= height) {
          continue
        }

        const index = px + py * width
        const edgeValue = edgeMap[index]
        const activity = activityMap[index]
        const detail = useColorDetail ? clamp(Math.max(activity, edgeValue * 0.7), 0, 1) : edgeValue

        const broadProbability = layer === 0 ? 1 : 0.74 - abstraction * 0.22
        const detailProbability = useColorDetail
          ? smoothstep(0.12, 0.7, activity) * (0.3 + detailAmount * 0.7)
          : smoothstep(0.14 + abstraction * 0.22, 0.72, detail) * detailAmount
        const probability = layer < 2
          ? broadProbability
          : clamp((useColorDetail ? 0.05 : 0.04) + detailProbability * (useColorDetail ? 1 : 0.82), 0, 1)
        if (hashToUnit(px, py, layer * 0x5151) > probability) {
          continue
        }

        const gradient = luminanceGradient(pixels, width, height, px, py)
        const flow = sampleStrokeFlow(flowField, px, py)
        const edge = clamp(gradient.strength * 3.8, 0, 1)
        const localAngle = Math.atan2(gradient.y, gradient.x) + Math.PI * 0.5
        const coherentFallback = coherentFlowAngle(px, py, width, height)
        const flowAngle = flow.confidence > 0.02 ? flow.angle : coherentFallback
        const localInfluence = smoothstep(0.42, 0.92, detail) * edgeFidelity
        const randomWobble = (hashToUnit(px, py, layer * 0xa53d) - 0.5) * (0.18 + fineLayer * 0.22)
        const angle = blendAngles(flowAngle, localAngle, localInfluence) + randomWobble
        // Shrink strokes in busy areas (color mode) for finer detail.
        const detailShrink = useColorDetail ? 1 - activity * detailAmount * 0.72 : 1
        const edgeLimiter = 1 - edge * edgeFidelity * 0.64
        const quietAreaBoost = 1 + (1 - detail) * abstraction * (layer < 2 ? 0.95 : 0.25)
        const strokeWidth = Math.max(
          1.2,
          size * (0.64 - fineLayer * 0.21) * (1 + (1 - detail) * abstraction * 0.31) * detailShrink,
        )
        // Keep strokes elongated so short marks never read as squares.
        const strokeLength = Math.max(
          size * lengthScale * (1.25 - fineLayer * 0.33) * edgeLimiter * quietAreaBoost * detailShrink,
          strokeWidth * 1.7,
        )
        const color = quantizedPixelColor(pixels, width, height, px, py, simplification)
        const valueShift = (hashToUnit(px, py, layer * 0x345f) - 0.5) * 18 * textureStrength

        drawTaperedStroke(context, {
          x: px,
          y: py,
          angle,
          length: strokeLength,
          width: strokeWidth,
          color: shiftColor(color, valueShift),
          alpha: alpha * (0.86 + hashToUnit(py, px, layer * 0x8849) * 0.2),
          textureStrength,
          seed: px * 17 + py * 31 + layer * 101,
        })
      }
    }
  }

  return context.getImageData(0, 0, width, height).data
}

// Regional measure of how much the color is shifting: per-pixel RGB gradient
// magnitude, box-blurred so it reflects busy neighborhoods rather than single
// edges, then normalized to 0..1.
function createColorActivityMap(pixels: Uint8ClampedArray, width: number, height: number, baseSize: number) {
  const raw = new Float32Array(width * height)
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const gx = pixelColorDistance(pixels, width, height, x - 1, y, x + 1, y)
      const gy = pixelColorDistance(pixels, width, height, x, y - 1, x, y + 1)
      raw[x + y * width] = Math.sqrt(gx * gx + gy * gy)
    }
  }

  const radius = Math.max(2, Math.round(baseSize * 0.9))
  const blurred = boxBlurFloat(raw, width, height, radius)
  normalizePixelDetailMap(blurred)
  return blurred
}

function pixelColorDistance(
  pixels: Uint8ClampedArray,
  width: number,
  height: number,
  ax: number,
  ay: number,
  bx: number,
  by: number,
) {
  const aClampedX = Math.max(0, Math.min(width - 1, ax))
  const aClampedY = Math.max(0, Math.min(height - 1, ay))
  const bClampedX = Math.max(0, Math.min(width - 1, bx))
  const bClampedY = Math.max(0, Math.min(height - 1, by))
  const aOffset = (aClampedX + aClampedY * width) * 4
  const bOffset = (bClampedX + bClampedY * width) * 4
  const dr = pixels[aOffset] - pixels[bOffset]
  const dg = pixels[aOffset + 1] - pixels[bOffset + 1]
  const db = pixels[aOffset + 2] - pixels[bOffset + 2]
  return Math.sqrt(dr * dr + dg * dg + db * db)
}

function boxBlurFloat(src: Float32Array, width: number, height: number, radius: number) {
  const window = radius * 2 + 1
  const temp = new Float32Array(width * height)
  const out = new Float32Array(width * height)

  for (let y = 0; y < height; y += 1) {
    const row = y * width
    let sum = 0
    for (let k = -radius; k <= radius; k += 1) {
      sum += src[row + Math.max(0, Math.min(width - 1, k))]
    }
    for (let x = 0; x < width; x += 1) {
      temp[row + x] = sum / window
      const addX = Math.max(0, Math.min(width - 1, x + radius + 1))
      const subX = Math.max(0, Math.min(width - 1, x - radius))
      sum += src[row + addX] - src[row + subX]
    }
  }

  for (let x = 0; x < width; x += 1) {
    let sum = 0
    for (let k = -radius; k <= radius; k += 1) {
      sum += temp[x + Math.max(0, Math.min(height - 1, k)) * width]
    }
    for (let y = 0; y < height; y += 1) {
      out[x + y * width] = sum / window
      const addY = Math.max(0, Math.min(height - 1, y + radius + 1))
      const subY = Math.max(0, Math.min(height - 1, y - radius))
      sum += temp[x + addY * width] - temp[x + subY * width]
    }
  }

  return out
}

function paintSmoothUnderpainting(
  context: CanvasRenderingContext2D,
  image: LoadedImage,
  baseSize: number,
  abstraction: number,
  simplification: number,
) {
  const { width, height, pixels } = image
  const blockSize = Math.max(6, Math.round(baseSize * (2.4 + abstraction * 4.0)))
  const lowWidth = Math.max(1, Math.ceil(width / blockSize))
  const lowHeight = Math.max(1, Math.ceil(height / blockSize))

  const lowCanvas = document.createElement('canvas')
  lowCanvas.width = lowWidth
  lowCanvas.height = lowHeight
  const lowContext = lowCanvas.getContext('2d')
  if (!lowContext) {
    // Fall back to the blocky base if the low-res canvas is unavailable.
    paintUnderpainting(context, image, baseSize, abstraction, simplification)
    return
  }

  const lowData = lowContext.createImageData(lowWidth, lowHeight)
  for (let cellY = 0; cellY < lowHeight; cellY += 1) {
    for (let cellX = 0; cellX < lowWidth; cellX += 1) {
      const color = quantizeRgb(
        averageRectColor(pixels, width, height, cellX * blockSize, cellY * blockSize, blockSize, blockSize),
        simplification * 0.85,
      )
      const offset = (cellX + cellY * lowWidth) * 4
      lowData.data[offset] = color[0]
      lowData.data[offset + 1] = color[1]
      lowData.data[offset + 2] = color[2]
      lowData.data[offset + 3] = 255
    }
  }
  lowContext.putImageData(lowData, 0, 0)

  // Bilinear upscale turns the averaged tiles into a smooth color field.
  context.imageSmoothingEnabled = true
  context.imageSmoothingQuality = 'high'
  context.drawImage(lowCanvas, 0, 0, lowWidth, lowHeight, 0, 0, width, height)
}

// Capsule-shaped brush mark with round caps; always elongated, so it never
// collapses into the square/diamond blob that the original brush produced at
// low stroke length.
function drawTaperedStroke(context: CanvasRenderingContext2D, stroke: StrokeSpec) {
  const halfLength = stroke.length * 0.5
  const halfWidth = stroke.width * 0.5
  const [red, green, blue] = stroke.color
  const capInset = Math.min(halfWidth, halfLength)

  context.save()
  context.translate(stroke.x, stroke.y)
  context.rotate(stroke.angle)
  context.globalAlpha = clamp(stroke.alpha, 0, 1)
  context.lineCap = 'round'
  context.lineJoin = 'round'
  context.strokeStyle = rgbCss(stroke.color)
  context.lineWidth = Math.max(1, stroke.width)
  context.beginPath()
  context.moveTo(-halfLength + capInset, 0)
  context.lineTo(halfLength - capInset, 0)
  context.stroke()

  const bristleCount = Math.max(2, Math.round(stroke.width * 0.42))
  for (let index = 0; index < bristleCount; index += 1) {
    const row = (index / Math.max(1, bristleCount - 1) - 0.5) * stroke.width
    const start = -halfLength * (0.84 + hashToUnit(stroke.seed, index, 0x1789) * 0.12)
    const end = halfLength * (0.72 + hashToUnit(stroke.seed, index, 0x8921) * 0.22)
    const amount = (hashToUnit(stroke.seed, index, 0x493f) - 0.5) * 38 * stroke.textureStrength
    context.globalAlpha = clamp(stroke.alpha * (0.18 + stroke.textureStrength * 0.28), 0, 1)
    context.strokeStyle = `rgb(${clamp(Math.round(red + amount), 0, 255)} ${clamp(Math.round(green + amount), 0, 255)} ${clamp(Math.round(blue + amount), 0, 255)})`
    context.lineWidth = Math.max(0.75, stroke.width * (0.04 + hashToUnit(stroke.seed, index, 0xa711) * 0.08))
    context.beginPath()
    context.moveTo(start, row)
    context.bezierCurveTo(start * 0.25, row + (hashToUnit(index, stroke.seed, 0x7f4a) - 0.5) * halfWidth, end * 0.55, row, end, row)
    context.stroke()
  }

  context.restore()
}

function paintUnderpainting(
  context: CanvasRenderingContext2D,
  image: LoadedImage,
  baseSize: number,
  abstraction: number,
  simplification: number,
) {
  const blockSize = Math.max(8, Math.round(baseSize * (3.2 + abstraction * 5.4)))
  const { width, height, pixels } = image

  for (let y = 0; y < height; y += blockSize) {
    for (let x = 0; x < width; x += blockSize) {
      const color = averageRectColor(pixels, width, height, x, y, blockSize, blockSize)
      context.fillStyle = rgbCss(quantizeRgb(color, simplification * 0.85))
      context.fillRect(x - 1, y - 1, blockSize + 2, blockSize + 2)
    }
  }
}

type StrokeSpec = {
  x: number
  y: number
  angle: number
  length: number
  width: number
  color: [number, number, number]
  alpha: number
  textureStrength: number
  seed: number
}

function createPixelDetailMap(pixels: Uint8ClampedArray, width: number, height: number) {
  const detail = new Float32Array(width * height)
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      detail[x + y * width] = luminanceGradient(pixels, width, height, x, y).strength
    }
  }
  normalizePixelDetailMap(detail)
  return detail
}

type StrokeFlowField = {
  cellSize: number
  width: number
  height: number
  angles: Float32Array
  confidence: Float32Array
}

function createStrokeFlowField(pixels: Uint8ClampedArray, width: number, height: number, requestedCellSize: number): StrokeFlowField {
  const cellSize = Math.max(4, requestedCellSize)
  const fieldWidth = Math.max(1, Math.ceil(width / cellSize))
  const fieldHeight = Math.max(1, Math.ceil(height / cellSize))
  const tensorA = new Float64Array(fieldWidth * fieldHeight)
  const tensorB = new Float64Array(fieldWidth * fieldHeight)
  const tensorC = new Float64Array(fieldWidth * fieldHeight)
  const angles = new Float32Array(fieldWidth * fieldHeight)
  const confidence = new Float32Array(fieldWidth * fieldHeight)

  for (let y = 0; y < height; y += 2) {
    for (let x = 0; x < width; x += 2) {
      const gradient = luminanceGradient(pixels, width, height, x, y)
      const cellX = Math.min(fieldWidth - 1, Math.floor(x / cellSize))
      const cellY = Math.min(fieldHeight - 1, Math.floor(y / cellSize))
      const index = cellX + cellY * fieldWidth
      tensorA[index] += gradient.x * gradient.x
      tensorB[index] += gradient.x * gradient.y
      tensorC[index] += gradient.y * gradient.y
    }
  }

  for (let cellY = 0; cellY < fieldHeight; cellY += 1) {
    for (let cellX = 0; cellX < fieldWidth; cellX += 1) {
      let a = 0
      let b = 0
      let c = 0
      let weightTotal = 0

      for (let offsetY = -2; offsetY <= 2; offsetY += 1) {
        for (let offsetX = -2; offsetX <= 2; offsetX += 1) {
          const sampleX = cellX + offsetX
          const sampleY = cellY + offsetY
          if (sampleX < 0 || sampleY < 0 || sampleX >= fieldWidth || sampleY >= fieldHeight) {
            continue
          }

          const distance = offsetX * offsetX + offsetY * offsetY
          const weight = distance === 0 ? 1 : 1 / (1 + distance)
          const index = sampleX + sampleY * fieldWidth
          a += tensorA[index] * weight
          b += tensorB[index] * weight
          c += tensorC[index] * weight
          weightTotal += weight
        }
      }

      const index = cellX + cellY * fieldWidth
      a /= Math.max(0.0001, weightTotal)
      b /= Math.max(0.0001, weightTotal)
      c /= Math.max(0.0001, weightTotal)
      const normalAngle = 0.5 * Math.atan2(2 * b, a - c)
      const trace = a + c
      const anisotropy = Math.sqrt((a - c) * (a - c) + 4 * b * b)
      angles[index] = normalAngle + Math.PI * 0.5
      confidence[index] = trace > 0 ? clamp(anisotropy / trace, 0, 1) : 0
    }
  }

  return {
    cellSize,
    width: fieldWidth,
    height: fieldHeight,
    angles,
    confidence,
  }
}

function sampleStrokeFlow(field: StrokeFlowField, x: number, y: number) {
  const fieldX = clamp(x / field.cellSize - 0.5, 0, field.width - 1)
  const fieldY = clamp(y / field.cellSize - 0.5, 0, field.height - 1)
  const x0 = Math.floor(fieldX)
  const y0 = Math.floor(fieldY)
  const x1 = Math.min(field.width - 1, x0 + 1)
  const y1 = Math.min(field.height - 1, y0 + 1)
  const tx = fieldX - x0
  const ty = fieldY - y0
  const top = blendFlowSamples(field, x0, y0, x1, y0, tx)
  const bottom = blendFlowSamples(field, x0, y1, x1, y1, tx)

  return {
    angle: blendAngles(top.angle, bottom.angle, ty),
    confidence: top.confidence * (1 - ty) + bottom.confidence * ty,
  }
}

function blendFlowSamples(field: StrokeFlowField, x0: number, y0: number, x1: number, y1: number, amount: number) {
  const left = x0 + y0 * field.width
  const right = x1 + y1 * field.width

  return {
    angle: blendAngles(field.angles[left], field.angles[right], amount),
    confidence: field.confidence[left] * (1 - amount) + field.confidence[right] * amount,
  }
}

function coherentFlowAngle(x: number, y: number, width: number, height: number) {
  const normalizedX = x / Math.max(1, width)
  const normalizedY = y / Math.max(1, height)
  const lowFrequency = hashToUnit(Math.floor(normalizedX * 5), Math.floor(normalizedY * 5), 0x6c31)
  const sweep = (normalizedX - 0.5) * 0.45 + (normalizedY - 0.5) * 0.28
  return (lowFrequency - 0.5) * Math.PI * 0.38 + sweep
}

function blendAngles(a: number, b: number, amount: number) {
  const clampedAmount = clamp(amount, 0, 1)
  let delta = ((b - a + Math.PI * 1.5) % Math.PI) - Math.PI * 0.5
  if (delta < -Math.PI * 0.5) {
    delta += Math.PI
  }

  return a + delta * clampedAmount
}

function normalizePixelDetailMap(detail: Float32Array) {
  const buckets = new Uint32Array(256)
  let maxValue = 0
  for (let index = 0; index < detail.length; index += 1) {
    maxValue = Math.max(maxValue, detail[index])
  }

  if (maxValue <= 0) {
    return
  }

  for (let index = 0; index < detail.length; index += 1) {
    const bucket = Math.min(255, Math.floor((detail[index] / maxValue) * 255))
    buckets[bucket] += 1
  }

  const target = detail.length * 0.96
  let cumulative = 0
  let percentileBucket = 255
  for (let bucket = 0; bucket < buckets.length; bucket += 1) {
    cumulative += buckets[bucket]
    if (cumulative >= target) {
      percentileBucket = bucket
      break
    }
  }

  const scale = Math.max(0.0001, (percentileBucket / 255) * maxValue)
  for (let index = 0; index < detail.length; index += 1) {
    detail[index] = clamp(detail[index] / scale, 0, 1)
  }
}

function luminanceGradient(pixels: Uint8ClampedArray, width: number, height: number, x: number, y: number) {
  const left = pixelLuminance(pixels, width, height, x - 1, y)
  const right = pixelLuminance(pixels, width, height, x + 1, y)
  const up = pixelLuminance(pixels, width, height, x, y - 1)
  const down = pixelLuminance(pixels, width, height, x, y + 1)
  const gx = right - left
  const gy = down - up

  return {
    x: gx,
    y: gy,
    strength: Math.sqrt(gx * gx + gy * gy) / 255,
  }
}

function pixelLuminance(pixels: Uint8ClampedArray, width: number, height: number, x: number, y: number) {
  const clampedX = Math.max(0, Math.min(width - 1, x))
  const clampedY = Math.max(0, Math.min(height - 1, y))
  const offset = (clampedX + clampedY * width) * 4
  return pixels[offset] * 0.299 + pixels[offset + 1] * 0.587 + pixels[offset + 2] * 0.114
}

function quantizedPixelColor(
  pixels: Uint8ClampedArray,
  width: number,
  height: number,
  x: number,
  y: number,
  simplification: number,
): [number, number, number] {
  const radius = Math.max(1, Math.round(1 + simplification * 5))
  return quantizeRgb(averageRectColor(pixels, width, height, x - radius, y - radius, radius * 2 + 1, radius * 2 + 1), simplification)
}

function averageRectColor(
  pixels: Uint8ClampedArray,
  width: number,
  height: number,
  x: number,
  y: number,
  rectWidth: number,
  rectHeight: number,
): [number, number, number] {
  const startX = Math.max(0, Math.floor(x))
  const startY = Math.max(0, Math.floor(y))
  const endX = Math.min(width, Math.ceil(x + rectWidth))
  const endY = Math.min(height, Math.ceil(y + rectHeight))
  let red = 0
  let green = 0
  let blue = 0
  let count = 0

  for (let py = startY; py < endY; py += 1) {
    for (let px = startX; px < endX; px += 1) {
      const offset = (px + py * width) * 4
      red += pixels[offset]
      green += pixels[offset + 1]
      blue += pixels[offset + 2]
      count += 1
    }
  }

  const divisor = Math.max(1, count)
  return [red / divisor, green / divisor, blue / divisor]
}

function quantizeRgb(color: [number, number, number], simplification: number): [number, number, number] {
  const levels = Math.round(36 - clamp(simplification, 0, 1) * 26)
  return [
    Math.round(Math.round((color[0] / 255) * levels) * (255 / levels)),
    Math.round(Math.round((color[1] / 255) * levels) * (255 / levels)),
    Math.round(Math.round((color[2] / 255) * levels) * (255 / levels)),
  ]
}

function shiftColor(color: [number, number, number], amount: number): [number, number, number] {
  return [
    clamp(Math.round(color[0] + amount), 0, 255),
    clamp(Math.round(color[1] + amount), 0, 255),
    clamp(Math.round(color[2] + amount), 0, 255),
  ]
}

function rgbCss(color: [number, number, number]) {
  return `rgb(${Math.round(color[0])} ${Math.round(color[1])} ${Math.round(color[2])})`
}

function createBrushField(image: LoadedImage, requestedBrushSize: number, seed: number) {
  const { width, height, pixels } = image
  const brushSize = Math.max(2, requestedBrushSize)
  const gridWidth = Math.ceil(width / brushSize) + 2
  const gridHeight = Math.ceil(height / brushSize) + 2
  const labelCount = gridWidth * gridHeight
  const sums = new Float64Array(labelCount * 3)
  const counts = new Uint32Array(labelCount)
  const labels = new Uint32Array(width * height)
  const output = new Uint8ClampedArray(pixels.length)

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const cellX = Math.floor(x / brushSize) + 1
      const cellY = Math.floor(y / brushSize) + 1
      let bestLabel = 0
      let bestDistance = Number.POSITIVE_INFINITY

      for (let offsetY = -1; offsetY <= 1; offsetY += 1) {
        for (let offsetX = -1; offsetX <= 1; offsetX += 1) {
          const candidateX = cellX + offsetX
          const candidateY = cellY + offsetY
          const center = brushCenter(candidateX, candidateY, brushSize, seed)
          const dx = x - center.x
          const dy = y - center.y
          const distance = dx * dx + dy * dy
          if (distance < bestDistance) {
            bestDistance = distance
            bestLabel = candidateX + candidateY * gridWidth
          }
        }
      }

      const index = x + y * width
      const pixelOffset = index * 4
      const sumOffset = bestLabel * 3
      labels[index] = bestLabel
      sums[sumOffset] += pixels[pixelOffset]
      sums[sumOffset + 1] += pixels[pixelOffset + 1]
      sums[sumOffset + 2] += pixels[pixelOffset + 2]
      counts[bestLabel] += 1
    }
  }

  for (let index = 0; index < labels.length; index += 1) {
    const pixelOffset = index * 4
    const label = labels[index]
    const sumOffset = label * 3
    const count = Math.max(1, counts[label])
    output[pixelOffset] = sums[sumOffset] / count
    output[pixelOffset + 1] = sums[sumOffset + 1] / count
    output[pixelOffset + 2] = sums[sumOffset + 2] / count
    output[pixelOffset + 3] = 255
  }

  return {
    pixels: output,
    labels,
    labelCount,
  }
}

function createCellDetailMap(pixels: Uint8ClampedArray, width: number, labels: Uint32Array, labelCount: number) {
  const totals = new Float32Array(labelCount)
  const counts = new Uint32Array(labelCount)

  for (let index = 0; index < labels.length; index += 1) {
    const x = index % width
    const y = Math.floor(index / width)
    const label = labels[index]
    totals[label] += pixelContrast(pixels, width, x, y)
    counts[label] += 1
  }

  for (let index = 0; index < totals.length; index += 1) {
    totals[index] = counts[index] > 0 ? totals[index] / counts[index] : 0
  }

  normalizeDetailValues(totals)
  return totals
}

function normalizeDetailValues(detailValues: Float32Array) {
  const ordered = Array.from(detailValues).sort((left, right) => left - right)
  if (ordered.length === 0) {
    return
  }

  let firstDifferent = ordered.length - 1
  for (let index = 1; index < ordered.length; index += 1) {
    if (ordered[index] > ordered[0]) {
      firstDifferent = index
      break
    }
  }

  if (firstDifferent === ordered.length - 1 && ordered[0] === ordered[ordered.length - 1]) {
    detailValues.fill(0)
    return
  }

  const denominator = Math.max(1, ordered.length - firstDifferent)
  for (let index = 0; index < detailValues.length; index += 1) {
    const rank = upperBound(ordered, detailValues[index])
    detailValues[index] = rank <= firstDifferent ? 0 : clamp((rank - firstDifferent) / denominator, 0, 1)
  }
}

function upperBound(values: number[], target: number) {
  let low = 0
  let high = values.length

  while (low < high) {
    const middle = Math.floor((low + high) / 2)
    if (values[middle] <= target) {
      low = middle + 1
    } else {
      high = middle
    }
  }

  return low
}

function brushCenter(cellX: number, cellY: number, brushSize: number, seed: number) {
  const jitterX = hashToUnit(cellX, cellY, seed) - 0.5
  const jitterY = hashToUnit(cellX, cellY, seed ^ 0x6d2b79f5) - 0.5

  return {
    x: (cellX - 0.5 + jitterX * 0.72) * brushSize,
    y: (cellY - 0.5 + jitterY * 0.72) * brushSize,
  }
}

function hashToUnit(x: number, y: number, seed: number) {
  let value = Math.imul(x + seed, 374761393) + Math.imul(y ^ seed, 668265263)
  value = Math.imul(value ^ (value >>> 13), 1274126177)
  return ((value ^ (value >>> 16)) >>> 0) / 4294967295
}

function pixelContrast(pixels: Uint8ClampedArray, width: number, x: number, y: number) {
  if (x === 0 || y === 0) {
    return 0
  }

  const offset = (x + y * width) * 4
  const leftOffset = offset - 4
  const upOffset = offset - width * 4
  const dx = colorDistance(pixels, offset, leftOffset)
  const dy = colorDistance(pixels, offset, upOffset)

  return Math.min(1, Math.sqrt(dx * dx + dy * dy) / 255)
}

function colorDistance(pixels: Uint8ClampedArray, offsetA: number, offsetB: number) {
  const red = pixels[offsetA] - pixels[offsetB]
  const green = pixels[offsetA + 1] - pixels[offsetB + 1]
  const blue = pixels[offsetA + 2] - pixels[offsetB + 2]

  return Math.sqrt(red * red + green * green + blue * blue)
}

function mixChannel(a: number, b: number, amount: number) {
  return a * (1 - amount) + b * amount
}

function smoothstep(edge0: number, edge1: number, value: number) {
  const amount = clamp((value - edge0) / (edge1 - edge0), 0, 1)
  return amount * amount * (3 - 2 * amount)
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value))
}

class SegmentForest {
  readonly parent: Int32Array
  readonly size: Int32Array
  readonly internal: Float32Array

  constructor(count: number) {
    this.parent = new Int32Array(count)
    this.size = new Int32Array(count)
    this.internal = new Float32Array(count)
    for (let index = 0; index < count; index += 1) {
      this.parent[index] = index
      this.size[index] = 1
    }
  }

  find(index: number) {
    let root = index
    while (this.parent[root] !== root) {
      root = this.parent[root]
    }

    while (this.parent[index] !== index) {
      const parent = this.parent[index]
      this.parent[index] = root
      index = parent
    }

    return root
  }

  union(a: number, b: number, weight: number) {
    let rootA = this.find(a)
    let rootB = this.find(b)
    if (rootA === rootB) {
      return rootA
    }

    if (this.size[rootA] < this.size[rootB]) {
      const swap = rootA
      rootA = rootB
      rootB = swap
    }

    this.parent[rootB] = rootA
    this.size[rootA] += this.size[rootB]
    this.internal[rootA] = weight
    return rootA
  }
}

function labelsToAveragePixels(pixels: Uint8ClampedArray, labels: Int32Array, labelCount: number) {
  const sums = new Float64Array(labelCount * 3)
  const counts = new Uint32Array(labelCount)
  const output = new Uint8ClampedArray(pixels.length)

  for (let index = 0; index < labels.length; index += 1) {
    const label = labels[index]
    if (label < 0) {
      continue
    }

    const pixelOffset = index * 4
    const labelOffset = label * 3
    sums[labelOffset] += pixels[pixelOffset]
    sums[labelOffset + 1] += pixels[pixelOffset + 1]
    sums[labelOffset + 2] += pixels[pixelOffset + 2]
    counts[label] += 1
  }

  for (let index = 0; index < labels.length; index += 1) {
    const pixelOffset = index * 4
    const label = labels[index]
    if (label < 0 || counts[label] === 0) {
      output[pixelOffset] = pixels[pixelOffset]
      output[pixelOffset + 1] = pixels[pixelOffset + 1]
      output[pixelOffset + 2] = pixels[pixelOffset + 2]
      output[pixelOffset + 3] = 255
      continue
    }

    const labelOffset = label * 3
    output[pixelOffset] = sums[labelOffset] / counts[label]
    output[pixelOffset + 1] = sums[labelOffset + 1] / counts[label]
    output[pixelOffset + 2] = sums[labelOffset + 2] / counts[label]
    output[pixelOffset + 3] = 255
  }

  return output
}

function createColorCheckPixels(image: LoadedImage, settings: FilterSettings, computedWeights?: Float32Array) {
  const { width, height, pixels } = image
  const { labels, labelCount } = computeFelzenszwalbLabels(
    image,
    settings.colorCheckScale,
    settings.colorCheckMinSize,
    computedWeights,
  )
  // Perceptual (OKLab) averaging keeps regions clean; the raw sRGB average is
  // the simpler, muddier block-in offered as a toggle.
  const output = settings.colorCheckPerceptual
    ? labelsToPerceptualPixels(pixels, labels, labelCount, settings.colorCheckSaturation)
    : labelsToAveragePixels(pixels, labels, labelCount)
  const separator = Math.round(settings.colorCheckSeparator)
  if (separator > 0) {
    applySeparator(output, labels, width, height, separator)
  }
  return output
}

// Averages each region in OKLab (perceptually accurate, avoids the muddy,
// desaturated result of averaging in sRGB) then restores chroma toward the
// region's true saturation, which shrinks when slightly varying hues are mixed.
function labelsToPerceptualPixels(
  pixels: Uint8ClampedArray,
  labels: Int32Array,
  labelCount: number,
  saturation: number,
) {
  const sums = new Float64Array(labelCount * 3)
  const counts = new Uint32Array(labelCount)
  const output = new Uint8ClampedArray(pixels.length)

  for (let index = 0; index < labels.length; index += 1) {
    const label = labels[index]
    if (label < 0) {
      continue
    }

    const pixelOffset = index * 4
    const [l, a, b] = srgbToOklab(pixels[pixelOffset], pixels[pixelOffset + 1], pixels[pixelOffset + 2])
    const labelOffset = label * 3
    sums[labelOffset] += l
    sums[labelOffset + 1] += a
    sums[labelOffset + 2] += b
    counts[label] += 1
  }

  const chromaFactor = 1 + clamp(saturation, 0, 1) * 0.6
  const labelColors = new Uint8ClampedArray(labelCount * 3)
  for (let label = 0; label < labelCount; label += 1) {
    const count = counts[label]
    if (count === 0) {
      continue
    }
    const labelOffset = label * 3
    const meanL = sums[labelOffset] / count
    const meanA = (sums[labelOffset + 1] / count) * chromaFactor
    const meanB = (sums[labelOffset + 2] / count) * chromaFactor
    const [r, g, b] = oklabToSrgb(meanL, meanA, meanB)
    labelColors[labelOffset] = r
    labelColors[labelOffset + 1] = g
    labelColors[labelOffset + 2] = b
  }

  for (let index = 0; index < labels.length; index += 1) {
    const pixelOffset = index * 4
    const label = labels[index]
    if (label < 0 || counts[label] === 0) {
      output[pixelOffset] = pixels[pixelOffset]
      output[pixelOffset + 1] = pixels[pixelOffset + 1]
      output[pixelOffset + 2] = pixels[pixelOffset + 2]
      output[pixelOffset + 3] = 255
      continue
    }

    const labelOffset = label * 3
    output[pixelOffset] = labelColors[labelOffset]
    output[pixelOffset + 1] = labelColors[labelOffset + 1]
    output[pixelOffset + 2] = labelColors[labelOffset + 2]
    output[pixelOffset + 3] = 255
  }

  return output
}

// Paints a neutral-gray line of the given thickness along region boundaries.
function applySeparator(
  output: Uint8ClampedArray,
  labels: Int32Array,
  width: number,
  height: number,
  thickness: number,
) {
  const count = width * height
  let boundary = new Uint8Array(count)
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = x + y * width
      const label = labels[index]
      if (
        (x + 1 < width && labels[index + 1] !== label) ||
        (y + 1 < height && labels[index + width] !== label)
      ) {
        boundary[index] = 1
        if (x + 1 < width) boundary[index + 1] = 1
        if (y + 1 < height) boundary[index + width] = 1
      }
    }
  }

  for (let pass = 1; pass < thickness; pass += 1) {
    const dilated = new Uint8Array(boundary)
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const index = x + y * width
        if (boundary[index]) {
          continue
        }
        if (
          (x > 0 && boundary[index - 1]) ||
          (x + 1 < width && boundary[index + 1]) ||
          (y > 0 && boundary[index - width]) ||
          (y + 1 < height && boundary[index + width])
        ) {
          dilated[index] = 1
        }
      }
    }
    boundary = dilated
  }

  for (let index = 0; index < count; index += 1) {
    if (!boundary[index]) {
      continue
    }
    const pixelOffset = index * 4
    output[pixelOffset] = 128
    output[pixelOffset + 1] = 128
    output[pixelOffset + 2] = 128
  }
}

function srgbToOklab(r8: number, g8: number, b8: number): [number, number, number] {
  const r = srgbChannelToLinear(r8 / 255)
  const g = srgbChannelToLinear(g8 / 255)
  const b = srgbChannelToLinear(b8 / 255)

  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)

  return [
    0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
  ]
}

function oklabToSrgb(L: number, a: number, b: number): [number, number, number] {
  const l_ = L + 0.3963377774 * a + 0.2158037573 * b
  const m_ = L - 0.1055613458 * a - 0.0638541728 * b
  const s_ = L - 0.0894841775 * a - 1.291485548 * b

  const l = l_ * l_ * l_
  const m = m_ * m_ * m_
  const s = s_ * s_ * s_

  const r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
  const g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
  const bb = -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s

  return [
    linearChannelToSrgb(r) * 255,
    linearChannelToSrgb(g) * 255,
    linearChannelToSrgb(bb) * 255,
  ]
}

function srgbChannelToLinear(c: number) {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4)
}

function linearChannelToSrgb(c: number) {
  const clamped = clamp(c, 0, 1)
  return clamped <= 0.0031308 ? clamped * 12.92 : 1.055 * Math.pow(clamped, 1 / 2.4) - 0.055
}
