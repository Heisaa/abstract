import { describe, expect, it } from 'vitest'
import { hueName, oklabChroma, oklabHue, oklabToSrgb, rgbToHex, srgbToOklab } from './color'

describe('OKLab conversion', () => {
  it('maps black and white to the ends of the lightness axis', () => {
    expect(srgbToOklab(0, 0, 0)[0]).toBeCloseTo(0, 5)
    const [l, a, b] = srgbToOklab(255, 255, 255)
    expect(l).toBeCloseTo(1, 3)
    expect(a).toBeCloseTo(0, 3)
    expect(b).toBeCloseTo(0, 3)
  })

  it('round-trips sRGB colors within rounding error', () => {
    const samples: Array<[number, number, number]> = [
      [12, 200, 90],
      [255, 0, 0],
      [33, 66, 240],
      [128, 128, 128],
      [250, 230, 10],
    ]
    for (const color of samples) {
      const back = oklabToSrgb(...srgbToOklab(...color))
      back.forEach((channel, index) => expect(Math.abs(channel - color[index])).toBeLessThan(0.6))
    }
  })

  it('treats grays as neutral and names saturated hues', () => {
    const [, ga, gb] = srgbToOklab(120, 120, 120)
    expect(oklabChroma(ga, gb)).toBeLessThan(1e-4)
    expect(hueName(oklabHue(ga, gb), oklabChroma(ga, gb))).toBe('neutral')

    const [, ra, rb] = srgbToOklab(220, 30, 30)
    expect(hueName(oklabHue(ra, rb), oklabChroma(ra, rb))).toBe('red')
    const [, ba, bb] = srgbToOklab(30, 60, 230)
    expect(hueName(oklabHue(ba, bb), oklabChroma(ba, bb))).toBe('blue')
  })

  it('formats hex', () => {
    expect(rgbToHex([255, 128.4, 0])).toBe('#ff8000')
  })
})
