import { describe, expect, it } from 'vitest'
import { blendAngles, createRandom, downsamplePixels, gaussianBlurFloat, normalizePixelDetailMap } from './common'

describe('blendAngles', () => {
  it('blends orientations along the shortest path modulo PI', () => {
    expect(blendAngles(0, Math.PI / 3, 0.5)).toBeCloseTo(Math.PI / 6)
    // 170 degrees is only 10 degrees away from 0 as an orientation.
    const near = blendAngles(0, (170 * Math.PI) / 180, 1)
    expect(Math.abs(near)).toBeCloseTo((10 * Math.PI) / 180, 5)
  })
})

describe('gaussianBlurFloat', () => {
  it('keeps constant images constant for small and large sigmas', () => {
    const src = new Float32Array(40 * 30).fill(0.7)
    for (const sigma of [0.8, 2, 6]) {
      const out = gaussianBlurFloat(src, 40, 30, sigma)
      expect(Math.max(...out)).toBeCloseTo(0.7, 5)
      expect(Math.min(...out)).toBeCloseTo(0.7, 5)
    }
  })

  it('spreads an impulse while preserving its mass', () => {
    const src = new Float32Array(31 * 31)
    src[15 + 15 * 31] = 1
    const out = gaussianBlurFloat(src, 31, 31, 2)
    const total = out.reduce((sum, value) => sum + value, 0)
    expect(total).toBeCloseTo(1, 3)
    expect(out[15 + 15 * 31]).toBeLessThan(0.1)
  })
})

describe('normalizePixelDetailMap', () => {
  it('maps the 96th percentile to 1', () => {
    const detail = Float32Array.from({ length: 100 }, (_, index) => index)
    normalizePixelDetailMap(detail)
    expect(detail[0]).toBe(0)
    expect(detail[99]).toBe(1)
    expect(detail[50]).toBeGreaterThan(0.45)
    expect(detail[50]).toBeLessThan(0.6)
  })
})

describe('createRandom', () => {
  it('is deterministic per seed', () => {
    const a = createRandom(42)
    const b = createRandom(42)
    expect([a(), a(), a()]).toEqual([b(), b(), b()])
  })
})

describe('downsamplePixels', () => {
  it('box-averages by the factor', () => {
    const pixels = new Uint8ClampedArray([0, 0, 0, 255, 200, 200, 200, 255, 100, 100, 100, 255, 100, 100, 100, 255])
    const out = downsamplePixels(pixels, 2, 2, 2)
    expect(out.width).toBe(1)
    expect(out.pixels[0]).toBe(100)
  })
})
