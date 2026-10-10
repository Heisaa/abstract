export type Rgb = [number, number, number]

export function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value))
}

export function smoothstep(edge0: number, edge1: number, value: number) {
  const amount = clamp((value - edge0) / (edge1 - edge0), 0, 1)
  return amount * amount * (3 - 2 * amount)
}

export function mixChannel(a: number, b: number, amount: number) {
  return a * (1 - amount) + b * amount
}

export function hashToUnit(x: number, y: number, seed: number) {
  let value = Math.imul(x + seed, 374761393) + Math.imul(y ^ seed, 668265263)
  value = Math.imul(value ^ (value >>> 13), 1274126177)
  return ((value ^ (value >>> 16)) >>> 0) / 4294967295
}

// Small deterministic PRNG so CPU filters give identical output for identical
// settings (important for preview/export parity).
export function createRandom(seed: number) {
  let state = seed >>> 0 || 0x9e3779b9
  return () => {
    state = (state + 0x6d2b79f5) >>> 0
    let t = state
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

// Blends two orientations (period PI) along the shortest path.
export function blendAngles(a: number, b: number, amount: number) {
  const clampedAmount = clamp(amount, 0, 1)
  let delta = ((b - a + Math.PI * 1.5) % Math.PI) - Math.PI * 0.5
  if (delta < -Math.PI * 0.5) {
    delta += Math.PI
  }

  return a + delta * clampedAmount
}

export function boxBlurFloat(src: Float32Array, width: number, height: number, radius: number) {
  if (radius < 1) {
    return new Float32Array(src)
  }

  const window = radius * 2 + 1
  const temp = new Float32Array(width * height)
  const out = new Float32Array(width * height)

  for (let y = 0; y < height; y += 1) {
    const row = y * width
    let sum = 0
    for (let k = -radius; k <= radius; k += 1) {
      sum += src[row + clamp(k, 0, width - 1)]
    }
    for (let x = 0; x < width; x += 1) {
      temp[row + x] = sum / window
      sum += src[row + Math.min(width - 1, x + radius + 1)] - src[row + Math.max(0, x - radius)]
    }
  }

  for (let x = 0; x < width; x += 1) {
    let sum = 0
    for (let k = -radius; k <= radius; k += 1) {
      sum += temp[x + clamp(k, 0, height - 1) * width]
    }
    for (let y = 0; y < height; y += 1) {
      out[x + y * width] = sum / window
      sum += temp[x + Math.min(height - 1, y + radius + 1) * width] - temp[x + Math.max(0, y - radius) * width]
    }
  }

  return out
}

// Exact separable Gaussian for small sigmas; three box passes (which
// approximate a Gaussian) for large ones, where an exact kernel gets slow.
export function gaussianBlurFloat(src: Float32Array, width: number, height: number, sigma: number) {
  if (sigma <= 0.25) {
    return new Float32Array(src)
  }

  if (sigma > 3) {
    const radius = Math.max(1, Math.round((Math.sqrt(4 * sigma * sigma + 1) - 1) / 2))
    let out = boxBlurFloat(src, width, height, radius)
    out = boxBlurFloat(out, width, height, radius)
    return boxBlurFloat(out, width, height, radius)
  }

  const radius = Math.ceil(sigma * 3)
  const kernel = new Float32Array(radius * 2 + 1)
  let total = 0
  for (let k = -radius; k <= radius; k += 1) {
    kernel[k + radius] = Math.exp(-(k * k) / (2 * sigma * sigma))
    total += kernel[k + radius]
  }
  for (let k = 0; k < kernel.length; k += 1) {
    kernel[k] /= total
  }

  const temp = new Float32Array(width * height)
  const out = new Float32Array(width * height)
  for (let y = 0; y < height; y += 1) {
    const row = y * width
    for (let x = 0; x < width; x += 1) {
      let sum = 0
      for (let k = -radius; k <= radius; k += 1) {
        sum += src[row + clamp(x + k, 0, width - 1)] * kernel[k + radius]
      }
      temp[row + x] = sum
    }
  }
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let sum = 0
      for (let k = -radius; k <= radius; k += 1) {
        sum += temp[x + clamp(y + k, 0, height - 1) * width] * kernel[k + radius]
      }
      out[x + y * width] = sum
    }
  }

  return out
}

// Gaussian blur applied to the RGB channels of an RGBA8 image, returning
// float planes (r, g, b) so later stages avoid re-quantization.
export function blurRgbPlanes(pixels: Uint8ClampedArray, width: number, height: number, sigma: number) {
  const count = width * height
  const planes = [new Float32Array(count), new Float32Array(count), new Float32Array(count)]
  for (let index = 0; index < count; index += 1) {
    planes[0][index] = pixels[index * 4]
    planes[1][index] = pixels[index * 4 + 1]
    planes[2][index] = pixels[index * 4 + 2]
  }

  return planes.map((plane) => gaussianBlurFloat(plane, width, height, sigma)) as [Float32Array, Float32Array, Float32Array]
}

// Normalizes a non-negative map so its 96th percentile maps to 1.
export function normalizePixelDetailMap(detail: Float32Array) {
  const buckets = new Uint32Array(256)
  let maxValue = 0
  for (let index = 0; index < detail.length; index += 1) {
    maxValue = Math.max(maxValue, detail[index])
  }

  if (maxValue <= 0) {
    return
  }

  for (let index = 0; index < detail.length; index += 1) {
    buckets[Math.min(255, Math.floor((detail[index] / maxValue) * 255))] += 1
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

export function pixelLuminance(pixels: Uint8ClampedArray, width: number, height: number, x: number, y: number) {
  const offset = (clamp(x, 0, width - 1) + clamp(y, 0, height - 1) * width) * 4
  return pixels[offset] * 0.2126 + pixels[offset + 1] * 0.7152 + pixels[offset + 2] * 0.0722
}

export function luminanceGradient(pixels: Uint8ClampedArray, width: number, height: number, x: number, y: number) {
  const gx = pixelLuminance(pixels, width, height, x + 1, y) - pixelLuminance(pixels, width, height, x - 1, y)
  const gy = pixelLuminance(pixels, width, height, x, y + 1) - pixelLuminance(pixels, width, height, x, y - 1)

  return {
    x: gx,
    y: gy,
    strength: Math.sqrt(gx * gx + gy * gy) / 255,
  }
}

export function averageRectColor(
  pixels: Uint8ClampedArray,
  width: number,
  height: number,
  x: number,
  y: number,
  rectWidth: number,
  rectHeight: number,
): Rgb {
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

export function quantizeRgb(color: Rgb, simplification: number): Rgb {
  const levels = Math.round(36 - clamp(simplification, 0, 1) * 26)
  return [
    Math.round(Math.round((color[0] / 255) * levels) * (255 / levels)),
    Math.round(Math.round((color[1] / 255) * levels) * (255 / levels)),
    Math.round(Math.round((color[2] / 255) * levels) * (255 / levels)),
  ]
}

export function rgbCss(color: Rgb) {
  return `rgb(${Math.round(color[0])} ${Math.round(color[1])} ${Math.round(color[2])})`
}

export function colorDistance(pixels: Uint8ClampedArray, offsetA: number, offsetB: number) {
  const red = pixels[offsetA] - pixels[offsetB]
  const green = pixels[offsetA + 1] - pixels[offsetB + 1]
  const blue = pixels[offsetA + 2] - pixels[offsetB + 2]
  return Math.sqrt(red * red + green * green + blue * blue)
}

// Downsamples an RGBA8 image by an integer factor using box averaging.
export function downsamplePixels(pixels: Uint8ClampedArray, width: number, height: number, factor: number) {
  const outWidth = Math.max(1, Math.floor(width / factor))
  const outHeight = Math.max(1, Math.floor(height / factor))
  const out = new Uint8ClampedArray(outWidth * outHeight * 4)
  for (let y = 0; y < outHeight; y += 1) {
    for (let x = 0; x < outWidth; x += 1) {
      const [r, g, b] = averageRectColor(pixels, width, height, x * factor, y * factor, factor, factor)
      const offset = (x + y * outWidth) * 4
      out[offset] = r
      out[offset + 1] = g
      out[offset + 2] = b
      out[offset + 3] = 255
    }
  }

  return { pixels: out, width: outWidth, height: outHeight }
}
