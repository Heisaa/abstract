import { describe, expect, it, vi } from 'vitest'
import { adaptiveBrushFilter, rankNormalize } from './adaptiveBrush'
import { colorCheckFilter } from './colorCheck'
import { bandForLightness, bandLightness, notanFilter, notanThresholds, otsuThresholds } from './notan'
import { kmeansOklab, paletteFilter, ZORN_PIGMENTS, zornMixtures } from './palette'
import { CpuFilterRunner } from './run'
import { defaultParams } from '../filters/registry'
import { pixelsToOklabPlanes, srgbToOklab } from './color'
import type { CpuImage } from './types'

function gradientImage(width: number, height: number): CpuImage {
  const pixels = new Uint8ClampedArray(width * height * 4)
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const value = Math.round((x / (width - 1)) * 255)
      pixels.set([value, value, value, 255], (x + y * width) * 4)
    }
  }
  return { pixels, width, height, scale: 1 }
}

function distinctColors(width: number, height: number, colors: Array<[number, number, number]>): CpuImage {
  const pixels = new Uint8ClampedArray(width * height * 4)
  for (let index = 0; index < width * height; index += 1) {
    const band = Math.min(colors.length - 1, Math.floor(((index % width) / width) * colors.length))
    pixels.set([...colors[band], 255], index * 4)
  }
  return { pixels, width, height, scale: 1 }
}

describe('notan', () => {
  it('sorts thresholds and assigns bands', () => {
    const { bands, thresholds } = notanThresholds({ bands: '3', t1: 0.7, t2: 0.3, t3: 0.9, t4: 0.95 })
    expect(bands).toBe(3)
    expect(thresholds).toEqual([0.3, 0.7])
    expect(bandForLightness(0.1, thresholds)).toBe(0)
    expect(bandForLightness(0.5, thresholds)).toBe(1)
    expect(bandForLightness(0.9, thresholds)).toBe(2)
    expect(bandLightness(0, 2)).toBe(0)
    expect(bandLightness(1, 2)).toBe(1)
  })

  it('places automatic thresholds between clusters of values', () => {
    // Values clustered in [0.17, 0.23], [0.47, 0.53] and [0.82, 0.88].
    const values = Float32Array.from({ length: 3000 }, (_, index) => [0.2, 0.5, 0.85][index % 3] + ((index % 7) - 3) * 0.01)
    const thresholds = otsuThresholds(values, 3)
    expect(thresholds).toHaveLength(2)
    expect(thresholds[0]).toBeGreaterThan(0.23)
    expect(thresholds[0]).toBeLessThan(0.47)
    expect(thresholds[1]).toBeGreaterThan(0.53)
    expect(thresholds[1]).toBeLessThan(0.82)
  })

  it('groups a gradient into one connected mass per band', () => {
    const image = gradientImage(120, 40)
    const params = { ...defaultParams('notan'), bands: '3', auto: false, t1: 0.4, t2: 0.75, squint: 0, minArea: 50 }
    const analysis = notanFilter.analyze(image, params, {})
    expect(analysis.regions.count).toBe(3)
    const output = notanFilter.render(image, analysis, params)
    // Two-tone ends: darkest band is black, lightest is white.
    expect(output.pixels[0]).toBe(0)
    expect(output.pixels[(119 + 0) * 4]).toBe(255)
  })
})

describe('palette', () => {
  const colors: Array<[number, number, number]> = [
    [200, 40, 40],
    [40, 160, 60],
    [30, 50, 200],
  ]

  it('k-means recovers distinct colors deterministically', () => {
    const image = distinctColors(90, 30, colors)
    const { L, A, B } = pixelsToOklabPlanes(image.pixels, image.width * image.height)
    const first = kmeansOklab(L, A, B, 3)
    const second = kmeansOklab(L, A, B, 3)
    expect(first).toEqual(second)
    for (const color of colors) {
      const [l, a, b] = srgbToOklab(...color)
      const nearest = Math.min(...first.map((center) => Math.hypot(center[0] - l, center[1] - a, center[2] - b)))
      expect(nearest).toBeLessThan(0.01)
    }
  })

  it('keeps a small saturated accent as its own palette color', () => {
    // A large field of neutral value steps with a small red patch, the kind
    // of accent (lips, a flower) plain k-means averages away.
    const width = 120
    const height = 60
    const pixels = new Uint8ClampedArray(width * height * 4)
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const shade = 40 + Math.floor(x / 20) * 36
        const accent = x >= 54 && x < 62 && y >= 26 && y < 34
        pixels.set(accent ? [210, 40, 40, 255] : [shade, shade, shade, 255], (x + y * width) * 4)
      }
    }
    const { L, A, B } = pixelsToOklabPlanes(pixels, width * height)
    const centers = kmeansOklab(L, A, B, 6)
    const [, ra, rb] = srgbToOklab(210, 40, 40)
    expect(Math.min(...centers.map((center) => Math.hypot(center[1] - ra, center[2] - rb)))).toBeLessThan(0.03)
  })

  it('reports swatches and maps every pixel to a palette color', () => {
    const image = distinctColors(90, 30, colors)
    const params = { ...defaultParams('palette'), colors: 3, squint: 0, minArea: 0 }
    const analysis = paletteFilter.analyze(image, params, {})
    const output = paletteFilter.render(image, analysis, params)
    expect(output.palette).toHaveLength(3)
    expect(output.pixels.length).toBe(image.pixels.length)
  })

  it('builds the Zorn mixture gamut from four pigments', () => {
    const mixtures = zornMixtures(16)
    expect(mixtures).toHaveLength(969)
    // Pure white and pure black are among the mixtures.
    const lightness = mixtures.map((color) => color[0])
    expect(Math.max(...lightness)).toBeGreaterThan(0.95)
    expect(Math.min(...lightness)).toBeLessThan(0.25)
    expect(ZORN_PIGMENTS).toHaveLength(4)
  })

  it('flags saturated blue as outside the Zorn gamut', () => {
    const image = distinctColors(40, 20, [[20, 40, 230]])
    const params = { ...defaultParams('palette'), mode: 'zorn', zornView: 'check', tolerance: 0.05, squint: 0 }
    const analysis = paletteFilter.analyze(image, params, {})
    expect(Math.min(...analysis.gamutDistance!)).toBeGreaterThan(0.05)
  })
})

describe('adaptive brush', () => {
  it('produces a full image and is deterministic', () => {
    const image = gradientImage(64, 48)
    const params = defaultParams('adaptiveBrush')
    const first = adaptiveBrushFilter.render(image, adaptiveBrushFilter.analyze(image, params, {}), params)
    const second = adaptiveBrushFilter.render(image, adaptiveBrushFilter.analyze(image, params, {}), params)
    expect(first.pixels).toEqual(second.pixels)
    expect(first.pixels.length).toBe(64 * 48 * 4)
  })

  it('rank-normalizes detail values', () => {
    const values = Float32Array.from([0, 0, 1, 2, 3])
    rankNormalize(values)
    expect(values[0]).toBe(0)
    expect(values[4]).toBe(1)
  })
})

describe('CpuFilterRunner', () => {
  it('reuses cached analysis when only render-only params change', () => {
    const runner = new CpuFilterRunner()
    const image = distinctColors(60, 40, [
      [200, 40, 40],
      [40, 40, 200],
    ])
    const analyze = vi.spyOn(colorCheckFilter, 'analyze')
    const params = defaultParams('colorCheck')
    runner.run(image, 'img', 'colorCheck', params)
    runner.run(image, 'img', 'colorCheck', { ...params, separator: 2, perceptual: false })
    expect(analyze).toHaveBeenCalledTimes(1)
    runner.run(image, 'img', 'colorCheck', { ...params, scale: 900 })
    expect(analyze).toHaveBeenCalledTimes(2)
    runner.run(image, 'other-image', 'colorCheck', params)
    expect(analyze).toHaveBeenCalledTimes(3)
    analyze.mockRestore()
  })

  it('rejects filters without a CPU implementation', () => {
    const runner = new CpuFilterRunner()
    expect(() => runner.run(gradientImage(4, 4), 'img', 'kuwahara', {})).toThrow(/No CPU implementation/)
  })
})
