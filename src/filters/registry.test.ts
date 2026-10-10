import { describe, expect, it } from 'vitest'
import { analysisKey, defaultParams, FILTERS, fullKey, getFilter, isFilterId, sanitizeParams, scaleParams } from './registry'

describe('filter registry', () => {
  it('has unique filter ids and unique param keys per filter', () => {
    const ids = FILTERS.map((filter) => filter.id)
    expect(new Set(ids).size).toBe(ids.length)
    for (const filter of FILTERS) {
      const keys = filter.params.map((param) => param.key)
      expect(new Set(keys).size, filter.id).toBe(keys.length)
    }
  })

  it('has defaults within range and on valid options', () => {
    for (const filter of FILTERS) {
      for (const param of filter.params) {
        if (param.type === 'range') {
          expect(param.default, `${filter.id}.${param.key}`).toBeGreaterThanOrEqual(param.min)
          expect(param.default, `${filter.id}.${param.key}`).toBeLessThanOrEqual(param.max)
        } else if (param.type === 'choice') {
          expect(param.options.map((option) => option.value)).toContain(param.default)
        }
      }
    }
  })

  it('sanitizes untrusted values', () => {
    const values = sanitizeParams('kuwahara', { radius: 999, sharpness: 'loud', passes: Number.NaN, extra: 1 })
    expect(values.radius).toBe(12)
    expect(values.sharpness).toBe(getFilter('kuwahara').params.find((param) => param.key === 'sharpness')!.default)
    expect(values).not.toHaveProperty('extra')
    expect(sanitizeParams('painter', { underpainting: 'watercolor' }).underpainting).toBe('smooth')
    expect(sanitizeParams('slic', null)).toEqual(defaultParams('slic'))
  })

  it('scales pixel params linearly and area params quadratically', () => {
    const base = defaultParams('colorCheck')
    const scaled = scaleParams('colorCheck', base, 2)
    expect(scaled.scale).toBe((base.scale as number) * 4)
    expect(scaled.separator).toBe((base.separator as number) * 2)
    expect(scaled.saturation).toBe(base.saturation)
    expect(scaleParams('colorCheck', base, 1)).toBe(base)
  })

  it('keeps render-only params out of the analysis key', () => {
    const base = defaultParams('notan')
    expect(analysisKey('notan', base)).toBe(analysisKey('notan', { ...base, output: 'mean', outline: true }))
    expect(analysisKey('notan', base)).not.toBe(analysisKey('notan', { ...base, minArea: 10 }))
    expect(fullKey('notan', base)).not.toBe(fullKey('notan', { ...base, output: 'mean' }))
  })

  it('resets value thresholds evenly when the band count changes', () => {
    const notan = getFilter('notan')
    const patch = notan.onParamChange?.('bands', { ...defaultParams('notan'), bands: '4' })
    expect(patch).toMatchObject({ t1: 0.25, t2: 0.5, t3: 0.75 })
    expect(notan.onParamChange?.('squint', defaultParams('notan'))).toBeUndefined()
  })

  it('recognizes filter ids', () => {
    expect(isFilterId('original')).toBe(true)
    expect(isFilterId('kuwahara')).toBe(true)
    expect(isFilterId('posterize')).toBe(false)
  })
})
