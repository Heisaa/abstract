import { describe, expect, it } from 'vitest'
import { halfToFloat } from './gpu/renderer'
import { computeCrop, DEFAULT_VIEW, outputToImageUv, sanitizeView } from './view'

describe('computeCrop', () => {
  it('returns the full image without a crop', () => {
    expect(computeCrop(DEFAULT_VIEW, 1200, 800)).toMatchObject({ offsetX: 0, scaleX: 1, width: 1200, height: 800 })
  })

  it('matches panel ratio to the image orientation and centers it', () => {
    const landscape = computeCrop({ ...DEFAULT_VIEW, crop: '8x10' }, 1200, 800)
    expect(landscape.width / landscape.height).toBeCloseTo(10 / 8, 2)
    expect(landscape.height).toBe(800)
    expect(landscape.offsetX).toBeCloseTo((1200 - landscape.width) / 2 / 1200, 5)

    const rotated = computeCrop({ ...DEFAULT_VIEW, crop: '8x10', cropRotate: true }, 1200, 800)
    expect(rotated.width / rotated.height).toBeCloseTo(8 / 10, 2)
  })

  it('shifts the crop window to the edges', () => {
    const left = computeCrop({ ...DEFAULT_VIEW, crop: '1x1', cropX: -1 }, 1200, 800)
    const right = computeCrop({ ...DEFAULT_VIEW, crop: '1x1', cropX: 1 }, 1200, 800)
    expect(left.offsetX).toBe(0)
    expect(right.offsetX + right.scaleX).toBeCloseTo(1, 5)
  })
})

describe('outputToImageUv', () => {
  it('inverts mirroring and crop', () => {
    const view = { ...DEFAULT_VIEW, crop: '1x1', mirror: true }
    const crop = computeCrop(view, 1000, 500)
    const point = outputToImageUv(view, crop, 0, 0.5)
    expect(point.u).toBeCloseTo(crop.offsetX + crop.scaleX, 5)
    expect(point.v).toBeCloseTo(0.5, 5)
  })
})

describe('sanitizeView', () => {
  it('drops invalid values', () => {
    const view = sanitizeView({ crop: 'poster', gridDivisions: 40, mirror: 'yes', cropX: 5 })
    expect(view.crop).toBe('image')
    expect(view.gridDivisions).toBe(12)
    expect(view.mirror).toBe(false)
    expect(view.cropX).toBe(1)
  })
})

describe('halfToFloat', () => {
  it('decodes IEEE half floats', () => {
    expect(halfToFloat(0x3c00)).toBe(1)
    expect(halfToFloat(0x3800)).toBe(0.5)
    expect(halfToFloat(0xc000)).toBe(-2)
    expect(halfToFloat(0)).toBe(0)
  })
})
