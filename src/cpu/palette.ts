import { createRandom, gaussianBlurFloat, type Rgb } from './common'
import { linearRgbToOklab, oklabToSrgb, pixelsToOklabPlanes, srgbChannelToLinear, type Oklab } from './color'
import { num, str, type ParamValues } from '../filters/registry'
import { connectedComponents, mergeSmallRegions, type RegionMap } from './regions'
import { paintRegions } from './regionColor'
import type { CpuFilter, CpuImage } from './types'

type PaletteAnalysis = {
  regions: RegionMap
  // Palette entries as 8-bit sRGB, indexed by region class.
  colors: Rgb[]
  // Per-pixel OKLab distance to the nearest gamut color (Zorn mode only).
  gamutDistance?: Float32Array
  swatches: Rgb[]
}

function squintedOklab(image: CpuImage, squint: number) {
  const count = image.width * image.height
  const { L, A, B } = pixelsToOklabPlanes(image.pixels, count)
  const sigma = squint / 2
  return {
    L: gaussianBlurFloat(L, image.width, image.height, sigma),
    A: gaussianBlurFloat(A, image.width, image.height, sigma),
    B: gaussianBlurFloat(B, image.width, image.height, sigma),
  }
}

// OKLab lightness spans 0..1 while chroma rarely exceeds 0.25, so in raw
// OKLab a value step dwarfs any hue change and small saturated accents (lips,
// flowers, signs) collapse into a neighboring neutral. Weighting a/b up makes a
// clear hue or chroma shift count about as much as a value step.
const CHROMA_WEIGHT = 2.5

function labDistanceSq(l1: number, a1: number, b1: number, l2: number, a2: number, b2: number) {
  const dl = l1 - l2
  const da = (a1 - a2) * CHROMA_WEIGHT
  const db = (b1 - b2) * CHROMA_WEIGHT
  return dl * dl + da * da + db * db
}

// Zorn gamut matching keeps a value-led metric: the question there is which
// mixture a painter would reach for, and the tolerance is calibrated to it.
const ZORN_LIGHTNESS_WEIGHT = 1.5

function gamutDistanceSq(l1: number, a1: number, b1: number, l2: number, a2: number, b2: number) {
  const dl = (l1 - l2) * ZORN_LIGHTNESS_WEIGHT
  const da = a1 - a2
  const db = b1 - b2
  return dl * dl + da * da + db * db
}

// Saturated samples pull their cluster harder, so an accent that covers a
// small area can still claim a palette entry instead of being averaged away.
function sampleWeight(a: number, b: number) {
  return 1 + 20 * Math.sqrt(a * a + b * b)
}

// Chroma-weighted k-means++ seeded, Lloyd-iterated clustering of OKLab colors
// on a fixed subsample, deterministic for a given image and k.
export function kmeansOklab(L: Float32Array, A: Float32Array, B: Float32Array, k: number, iterations = 16) {
  const count = L.length
  const stride = Math.max(1, Math.floor(count / 40000))
  const samples: number[] = []
  for (let index = 0; index < count; index += stride) {
    samples.push(index)
  }

  const random = createRandom(0x51ed + k)
  const centers: Oklab[] = []
  const first = samples[Math.floor(random() * samples.length)]
  centers.push([L[first], A[first], B[first]])
  const nearest = new Float64Array(samples.length).fill(Number.POSITIVE_INFINITY)
  const weights = new Float64Array(samples.length)
  for (let s = 0; s < samples.length; s += 1) {
    weights[s] = sampleWeight(A[samples[s]], B[samples[s]])
  }

  while (centers.length < k) {
    const last = centers[centers.length - 1]
    let total = 0
    for (let s = 0; s < samples.length; s += 1) {
      const index = samples[s]
      nearest[s] = Math.min(nearest[s], labDistanceSq(L[index], A[index], B[index], last[0], last[1], last[2]))
      total += nearest[s]
    }
    if (total <= 0) {
      break
    }
    let target = random() * total
    let chosen = samples.length - 1
    for (let s = 0; s < samples.length; s += 1) {
      target -= nearest[s] * weights[s]
      if (target <= 0) {
        chosen = s
        break
      }
    }
    const index = samples[chosen]
    centers.push([L[index], A[index], B[index]])
  }

  const sums = new Float64Array(centers.length * 4)
  for (let iteration = 0; iteration < iterations; iteration += 1) {
    sums.fill(0)
    for (let s = 0; s < samples.length; s += 1) {
      const index = samples[s]
      const weight = weights[s]
      const center = nearestCenter(centers, L[index], A[index], B[index])
      sums[center * 4] += L[index] * weight
      sums[center * 4 + 1] += A[index] * weight
      sums[center * 4 + 2] += B[index] * weight
      sums[center * 4 + 3] += weight
    }
    let moved = 0
    for (let c = 0; c < centers.length; c += 1) {
      const size = sums[c * 4 + 3]
      if (size === 0) {
        continue
      }
      const next: Oklab = [sums[c * 4] / size, sums[c * 4 + 1] / size, sums[c * 4 + 2] / size]
      moved = Math.max(moved, labDistanceSq(...next, ...centers[c]))
      centers[c] = next
    }
    if (moved < 1e-8) {
      break
    }
  }

  // Order dark to light so swatches read as a value scale.
  centers.sort((left, right) => left[0] - right[0])
  return centers
}

function nearestCenter(centers: Oklab[], l: number, a: number, b: number, distanceSq = labDistanceSq) {
  let best = 0
  let bestDistance = Number.POSITIVE_INFINITY
  for (let c = 0; c < centers.length; c += 1) {
    const distance = distanceSq(l, a, b, centers[c][0], centers[c][1], centers[c][2])
    if (distance < bestDistance) {
      bestDistance = distance
      best = c
    }
  }
  return best
}

// Approximate Zorn palette pigments (sRGB): yellow ochre, vermilion/cadmium
// red light, ivory black, titanium white.
export const ZORN_PIGMENTS: Rgb[] = [
  [198, 148, 62],
  [213, 52, 38],
  [34, 32, 33],
  [246, 244, 238],
]

type Gamut = {
  colors: Oklab[]
  rgb: Rgb[]
  lut: Uint16Array
}

const LUT_SIZE = 40
const LUT_AB_RANGE = 0.32
let zornGamut: Gamut | undefined

// Mixtures of the pigments in 1/16 steps. Paint mixing is subtractive, so
// mixtures use a weighted geometric mean of linear reflectance per channel,
// a cheap stand-in for Kubelka-Munk that darkens and shifts hue plausibly.
export function zornMixtures(steps = 16) {
  const reflectance = ZORN_PIGMENTS.map((pigment) => pigment.map((channel) => Math.max(0.002, srgbChannelToLinear(channel / 255))))
  const colors: Oklab[] = []
  for (let w0 = 0; w0 <= steps; w0 += 1) {
    for (let w1 = 0; w1 <= steps - w0; w1 += 1) {
      for (let w2 = 0; w2 <= steps - w0 - w1; w2 += 1) {
        const weights = [w0, w1, w2, steps - w0 - w1 - w2].map((weight) => weight / steps)
        const linear = [0, 1, 2].map((channel) =>
          Math.exp(weights.reduce((sum, weight, pigment) => sum + weight * Math.log(reflectance[pigment][channel]), 0)),
        )
        colors.push(linearRgbToOklab(linear[0], linear[1], linear[2]))
      }
    }
  }
  return colors
}

function lutCell(l: number, a: number, b: number) {
  const li = Math.min(LUT_SIZE - 1, Math.max(0, Math.floor(l * LUT_SIZE)))
  const ai = Math.min(LUT_SIZE - 1, Math.max(0, Math.floor(((a + LUT_AB_RANGE) / (2 * LUT_AB_RANGE)) * LUT_SIZE)))
  const bi = Math.min(LUT_SIZE - 1, Math.max(0, Math.floor(((b + LUT_AB_RANGE) / (2 * LUT_AB_RANGE)) * LUT_SIZE)))
  return li + ai * LUT_SIZE + bi * LUT_SIZE * LUT_SIZE
}

// Nearest-mixture lookup table over OKLab, built once per session.
function getZornGamut(): Gamut {
  if (zornGamut) {
    return zornGamut
  }

  const colors = zornMixtures()
  const lut = new Uint16Array(LUT_SIZE * LUT_SIZE * LUT_SIZE)
  for (let bi = 0; bi < LUT_SIZE; bi += 1) {
    for (let ai = 0; ai < LUT_SIZE; ai += 1) {
      for (let li = 0; li < LUT_SIZE; li += 1) {
        const l = (li + 0.5) / LUT_SIZE
        const a = ((ai + 0.5) / LUT_SIZE) * 2 * LUT_AB_RANGE - LUT_AB_RANGE
        const b = ((bi + 0.5) / LUT_SIZE) * 2 * LUT_AB_RANGE - LUT_AB_RANGE
        lut[li + ai * LUT_SIZE + bi * LUT_SIZE * LUT_SIZE] = nearestCenter(colors, l, a, b, gamutDistanceSq)
      }
    }
  }

  zornGamut = { colors, lut, rgb: colors.map((color) => oklabToSrgb(...color)) }
  return zornGamut
}

function analyzePalette(image: CpuImage, params: ParamValues): PaletteAnalysis {
  const { width, height } = image
  const count = width * height
  const { L, A, B } = squintedOklab(image, num(params, 'squint'))
  const classes = new Int32Array(count)
  let colors: Rgb[]
  let gamutDistance: Float32Array | undefined
  let swatches: Rgb[]

  if (str(params, 'mode') === 'zorn') {
    const gamut = getZornGamut()
    gamutDistance = new Float32Array(count)
    const original = pixelsToOklabPlanes(image.pixels, count)
    for (let index = 0; index < count; index += 1) {
      const mixture = gamut.lut[lutCell(L[index], A[index], B[index])]
      classes[index] = mixture
      const target = gamut.colors[gamut.lut[lutCell(original.L[index], original.A[index], original.B[index])]]
      gamutDistance[index] = Math.sqrt(gamutDistanceSq(original.L[index], original.A[index], original.B[index], ...target))
    }
    colors = gamut.rgb
    swatches = ZORN_PIGMENTS
  } else {
    const k = Math.max(2, Math.min(16, Math.round(num(params, 'colors'))))
    const centers = kmeansOklab(L, A, B, k)
    for (let index = 0; index < count; index += 1) {
      classes[index] = nearestCenter(centers, L[index], A[index], B[index])
    }
    colors = centers.map((center) => oklabToSrgb(...center))
    swatches = colors
  }

  const regions = mergeSmallRegions(connectedComponents(classes, width, height), width, height, Math.round(num(params, 'minArea')))
  return { regions, colors, gamutDistance, swatches }
}

export const paletteFilter: CpuFilter<PaletteAnalysis> = {
  analyze: analyzePalette,
  render(image, analysis, params) {
    const palette = analysis.swatches.map((color) => color.map(Math.round) as Rgb)
    if (str(params, 'mode') === 'zorn' && str(params, 'zornView') === 'check' && analysis.gamutDistance) {
      return { pixels: flagOutOfGamut(image, analysis.gamutDistance, num(params, 'tolerance')), palette }
    }

    const regionColors = new Uint8ClampedArray(analysis.regions.count * 3)
    for (let region = 0; region < analysis.regions.count; region += 1) {
      regionColors.set(analysis.colors[analysis.regions.classOf[region]], region * 3)
    }
    return { pixels: paintRegions(analysis.regions.labels, regionColors), palette }
  },
}

// Shows the original image with magenta hatching wherever the color cannot be
// mixed from the palette within the tolerance.
function flagOutOfGamut(image: CpuImage, distance: Float32Array, tolerance: number) {
  const { width, pixels } = image
  const output = new Uint8ClampedArray(pixels)
  const period = Math.max(4, Math.round(8 * image.scale))
  for (let index = 0; index < distance.length; index += 1) {
    if (distance[index] <= tolerance) {
      continue
    }
    const x = index % width
    const y = Math.floor(index / width)
    if ((x + y) % period < period * 0.45) {
      output[index * 4] = 236
      output[index * 4 + 1] = 32
      output[index * 4 + 2] = 196
    }
  }
  return output
}
