import { clamp } from './common'
import { oklabToSrgb, srgbToOklab } from './color'

export type RegionStats = {
  count: number
  meanL: Float32Array
  meanA: Float32Array
  meanB: Float32Array
  // Average per-pixel chroma; exceeds the chroma of the mean color whenever
  // the region mixes slightly different hues.
  meanChroma: Float32Array
  meanRgb: Float32Array
  sizes: Uint32Array
}

export function regionStats(pixels: Uint8ClampedArray, labels: Int32Array, count: number): RegionStats {
  const sumL = new Float64Array(count)
  const sumA = new Float64Array(count)
  const sumB = new Float64Array(count)
  const sumChroma = new Float64Array(count)
  const sumRgb = new Float64Array(count * 3)
  const sizes = new Uint32Array(count)

  for (let index = 0; index < labels.length; index += 1) {
    const label = labels[index]
    const offset = index * 4
    const [l, a, b] = srgbToOklab(pixels[offset], pixels[offset + 1], pixels[offset + 2])
    sumL[label] += l
    sumA[label] += a
    sumB[label] += b
    sumChroma[label] += Math.sqrt(a * a + b * b)
    sumRgb[label * 3] += pixels[offset]
    sumRgb[label * 3 + 1] += pixels[offset + 1]
    sumRgb[label * 3 + 2] += pixels[offset + 2]
    sizes[label] += 1
  }

  const meanL = new Float32Array(count)
  const meanA = new Float32Array(count)
  const meanB = new Float32Array(count)
  const meanChroma = new Float32Array(count)
  const meanRgb = new Float32Array(count * 3)
  for (let label = 0; label < count; label += 1) {
    const size = Math.max(1, sizes[label])
    meanL[label] = sumL[label] / size
    meanA[label] = sumA[label] / size
    meanB[label] = sumB[label] / size
    meanChroma[label] = sumChroma[label] / size
    meanRgb[label * 3] = sumRgb[label * 3] / size
    meanRgb[label * 3 + 1] = sumRgb[label * 3 + 1] / size
    meanRgb[label * 3 + 2] = sumRgb[label * 3 + 2] / size
  }

  return { count, meanL, meanA, meanB, meanChroma, meanRgb, sizes }
}

// Region color from OKLab means, with chroma pushed back toward the region's
// average per-pixel chroma by `preserve` (0..1) and scaled by `boost`.
export function perceptualRegionColors(stats: RegionStats, preserve: number, boost = 1) {
  const colors = new Uint8ClampedArray(stats.count * 3)
  for (let label = 0; label < stats.count; label += 1) {
    const a = stats.meanA[label]
    const b = stats.meanB[label]
    const chroma = Math.sqrt(a * a + b * b)
    const restore = chroma > 1e-5 ? clamp(stats.meanChroma[label] / chroma, 1, 3) : 1
    const factor = (1 + (restore - 1) * clamp(preserve, 0, 1)) * boost
    const [r, g, bb] = oklabToSrgb(stats.meanL[label], a * factor, b * factor)
    colors[label * 3] = r
    colors[label * 3 + 1] = g
    colors[label * 3 + 2] = bb
  }

  return colors
}

export function averageRegionColors(stats: RegionStats) {
  return Uint8ClampedArray.from(stats.meanRgb)
}

export function paintRegions(labels: Int32Array, colors: Uint8ClampedArray) {
  const output = new Uint8ClampedArray(labels.length * 4)
  for (let index = 0; index < labels.length; index += 1) {
    const offset = index * 4
    const colorOffset = labels[index] * 3
    output[offset] = colors[colorOffset]
    output[offset + 1] = colors[colorOffset + 1]
    output[offset + 2] = colors[colorOffset + 2]
    output[offset + 3] = 255
  }

  return output
}

export function paintMask(output: Uint8ClampedArray, mask: Uint8Array, color: [number, number, number]) {
  for (let index = 0; index < mask.length; index += 1) {
    if (mask[index]) {
      output[index * 4] = color[0]
      output[index * 4 + 1] = color[1]
      output[index * 4 + 2] = color[2]
    }
  }
}
