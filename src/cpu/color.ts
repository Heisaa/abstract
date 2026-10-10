import { clamp, type Rgb } from './common'

export type Oklab = [number, number, number]

const SRGB_TO_LINEAR = new Float32Array(256)
for (let value = 0; value < 256; value += 1) {
  const c = value / 255
  SRGB_TO_LINEAR[value] = c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4)
}

export function srgbChannelToLinear(c: number) {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4)
}

export function linearChannelToSrgb(c: number) {
  const clamped = clamp(c, 0, 1)
  return clamped <= 0.0031308 ? clamped * 12.92 : 1.055 * Math.pow(clamped, 1 / 2.4) - 0.055
}

export function linearRgbToOklab(r: number, g: number, b: number): Oklab {
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)

  return [
    0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
  ]
}

export function oklabToLinearRgb(L: number, a: number, b: number): Rgb {
  const l_ = L + 0.3963377774 * a + 0.2158037573 * b
  const m_ = L - 0.1055613458 * a - 0.0638541728 * b
  const s_ = L - 0.0894841775 * a - 1.291485548 * b
  const l = l_ * l_ * l_
  const m = m_ * m_ * m_
  const s = s_ * s_ * s_

  return [
    4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s,
  ]
}

// 8-bit sRGB in, OKLab out.
export function srgbToOklab(r8: number, g8: number, b8: number): Oklab {
  const ri = clamp(Math.round(r8), 0, 255)
  const gi = clamp(Math.round(g8), 0, 255)
  const bi = clamp(Math.round(b8), 0, 255)
  return linearRgbToOklab(SRGB_TO_LINEAR[ri], SRGB_TO_LINEAR[gi], SRGB_TO_LINEAR[bi])
}

// OKLab in, 8-bit-range sRGB (unrounded floats, clamped to gamut per channel).
export function oklabToSrgb(L: number, a: number, b: number): Rgb {
  const [r, g, bb] = oklabToLinearRgb(L, a, b)
  return [linearChannelToSrgb(r) * 255, linearChannelToSrgb(g) * 255, linearChannelToSrgb(bb) * 255]
}

// Converts an RGBA8 image into three OKLab planes.
export function pixelsToOklabPlanes(pixels: Uint8ClampedArray, count: number) {
  const L = new Float32Array(count)
  const A = new Float32Array(count)
  const B = new Float32Array(count)
  for (let index = 0; index < count; index += 1) {
    const offset = index * 4
    const lab = linearRgbToOklab(SRGB_TO_LINEAR[pixels[offset]], SRGB_TO_LINEAR[pixels[offset + 1]], SRGB_TO_LINEAR[pixels[offset + 2]])
    L[index] = lab[0]
    A[index] = lab[1]
    B[index] = lab[2]
  }

  return { L, A, B }
}

export function oklabChroma(a: number, b: number) {
  return Math.sqrt(a * a + b * b)
}

// Hue angle in degrees, 0..360.
export function oklabHue(a: number, b: number) {
  const degrees = (Math.atan2(b, a) * 180) / Math.PI
  return degrees < 0 ? degrees + 360 : degrees
}

// Coarse painter-friendly hue names for OKLCh hue angles.
export function hueName(hue: number, chroma: number) {
  if (chroma < 0.02) {
    return 'neutral'
  }

  const names: Array<[number, string]> = [
    [20, 'red-violet'],
    [45, 'red'],
    [70, 'orange'],
    [100, 'yellow-orange'],
    [125, 'yellow'],
    [150, 'yellow-green'],
    [190, 'green'],
    [230, 'blue-green'],
    [275, 'blue'],
    [315, 'blue-violet'],
    [350, 'violet'],
    [360, 'red-violet'],
  ]

  return names.find(([limit]) => hue < limit)?.[1] ?? 'red-violet'
}

export function rgbToHex(color: Rgb) {
  return `#${color.map((channel) => clamp(Math.round(channel), 0, 255).toString(16).padStart(2, '0')).join('')}`
}
