import { gaussianBlurFloat } from './common'
import { oklabToSrgb, pixelsToOklabPlanes } from './color'
import { bool, num, str, type ParamValues } from '../filters/registry'
import { boundaryMask, connectedComponents, mergeSmallRegions, type RegionMap } from './regions'
import { paintMask, paintRegions, regionStats } from './regionColor'
import type { CpuFilter, CpuImage } from './types'

export type NotanAnalysis = {
  regions: RegionMap
  bands: number
}

export function notanThresholds(params: ParamValues) {
  const bands = Math.max(2, Math.min(5, Math.round(Number(params.bands) || 3)))
  const thresholds: number[] = []
  for (let index = 1; index < bands; index += 1) {
    thresholds.push(num(params, `t${index}`))
  }
  thresholds.sort((a, b) => a - b)
  return { bands, thresholds }
}

export function bandForLightness(lightness: number, thresholds: number[]) {
  let band = 0
  while (band < thresholds.length && lightness >= thresholds[band]) {
    band += 1
  }
  return band
}

const OTSU_BINS = 64

// Multi-level Otsu: the thresholds that split the lightness histogram into
// `classes` groups with maximal between-class variance, i.e. where this
// image's own values separate into masses. Exhaustive over a 64-bin
// histogram (at most 4 thresholds), using prefix sums for O(1) class terms.
export function otsuThresholds(lightness: Float32Array, classes: number) {
  const histogram = new Float64Array(OTSU_BINS)
  for (let index = 0; index < lightness.length; index += 1) {
    histogram[Math.min(OTSU_BINS - 1, Math.max(0, Math.floor(lightness[index] * OTSU_BINS)))] += 1
  }
  const weight = new Float64Array(OTSU_BINS + 1)
  const moment = new Float64Array(OTSU_BINS + 1)
  for (let bin = 0; bin < OTSU_BINS; bin += 1) {
    weight[bin + 1] = weight[bin] + histogram[bin]
    moment[bin + 1] = moment[bin] + histogram[bin] * (bin + 0.5)
  }
  // Between-class term for bins [from, to).
  const term = (from: number, to: number) => {
    const w = weight[to] - weight[from]
    return w > 0 ? ((moment[to] - moment[from]) * (moment[to] - moment[from])) / w : 0
  }

  const cuts = Math.max(1, Math.min(4, classes - 1))
  let best = -1
  let bestCuts: number[] = []
  const current: number[] = []
  const search = (start: number, score: number) => {
    if (current.length === cuts) {
      const total = score + term(start, OTSU_BINS)
      if (total > best) {
        best = total
        bestCuts = [...current]
      }
      return
    }
    for (let cut = start + 1; cut <= OTSU_BINS - (cuts - current.length); cut += 1) {
      current.push(cut)
      search(cut, score + term(start, cut))
      current.pop()
    }
  }
  search(0, 0)
  return bestCuts.map((cut) => cut / OTSU_BINS)
}

// Display lightness for a band: evenly spaced from black to white so a 2-value
// notan is pure black and white.
export function bandLightness(band: number, bands: number) {
  return bands <= 1 ? 0.5 : band / (bands - 1)
}

export function analyzeNotan(image: CpuImage, params: ParamValues): NotanAnalysis {
  const { width, height, pixels } = image
  const count = width * height
  const { L } = pixelsToOklabPlanes(pixels, count)
  const squinted = gaussianBlurFloat(L, width, height, num(params, 'squint') / 2)
  const manual = notanThresholds(params)
  const bands = manual.bands
  const thresholds = bool(params, 'auto') ? otsuThresholds(squinted, bands) : manual.thresholds

  const classes = new Int32Array(count)
  for (let index = 0; index < count; index += 1) {
    classes[index] = bandForLightness(squinted[index], thresholds)
  }

  const components = connectedComponents(classes, width, height)
  const regions = mergeSmallRegions(components, width, height, Math.round(num(params, 'minArea')))
  return { regions, bands }
}

export const notanFilter: CpuFilter<NotanAnalysis> = {
  analyze: analyzeNotan,
  render(image, { regions, bands }, params) {
    const output = str(params, 'output')
    const colors = new Uint8ClampedArray(regions.count * 3)
    const stats = output === 'gray' ? undefined : regionStats(image.pixels, regions.labels, regions.count)

    for (let region = 0; region < regions.count; region += 1) {
      const lightness = bandLightness(regions.classOf[region], bands)
      let rgb
      if (!stats) {
        rgb = oklabToSrgb(lightness, 0, 0)
      } else if (output === 'snapped') {
        // Region hue and chroma at the band's value: a color study that still
        // reads as a clean value pattern.
        rgb = oklabToSrgb(lightness * 0.9 + 0.05, stats.meanA[region], stats.meanB[region])
      } else {
        rgb = oklabToSrgb(stats.meanL[region], stats.meanA[region], stats.meanB[region])
      }
      colors.set(rgb, region * 3)
    }

    const pixels = paintRegions(regions.labels, colors)
    if (params.outline === true) {
      paintMask(pixels, boundaryMask(regions.labels, image.width, image.height, 1), [214, 64, 48])
    }

    return { pixels }
  },
}
