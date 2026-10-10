import { describe, expect, it } from 'vitest'
import { segmentFelzenszwalb } from './felzenszwalb'
import { blurRgbPlanes } from './common'
import { boundaryMask, connectedComponents, DisjointSet, mergeSmallRegions } from './regions'
import { slicFilter } from './slic'
import { defaultParams } from '../filters/registry'

function makeImage(width: number, height: number, color: (x: number, y: number) => [number, number, number]) {
  const pixels = new Uint8ClampedArray(width * height * 4)
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      pixels.set([...color(x, y), 255], (x + y * width) * 4)
    }
  }
  return pixels
}

describe('connectedComponents', () => {
  it('separates equal classes that do not touch', () => {
    // 0 0 1 0
    // 0 0 1 0
    const classes = [0, 0, 1, 0, 0, 0, 1, 0]
    const regions = connectedComponents(classes, 4, 2)
    expect(regions.count).toBe(3)
    expect(Array.from(regions.sizes).sort()).toEqual([2, 2, 4])
  })
})

describe('mergeSmallRegions', () => {
  it('absorbs specks into their surroundings and keeps the surrounding class', () => {
    const width = 20
    const height = 20
    const classes = new Int32Array(width * height)
    classes[5 + 5 * width] = 1
    classes[12 + 14 * width] = 2
    const merged = mergeSmallRegions(connectedComponents(classes, width, height), width, height, 4)
    expect(merged.count).toBe(1)
    expect(merged.classOf[0]).toBe(0)
    expect(merged.sizes[0]).toBe(width * height)
  })

  it('leaves regions at or above the minimum alone', () => {
    const width = 10
    const height = 10
    const classes = new Int32Array(width * height).map((_, index) => (index % width < 5 ? 0 : 1))
    const merged = mergeSmallRegions(connectedComponents(classes, width, height), width, height, 20)
    expect(merged.count).toBe(2)
  })
})

describe('boundaryMask', () => {
  it('marks pixels where labels change and dilates by thickness', () => {
    const labels = new Int32Array(10 * 1).map((_, index) => (index < 5 ? 0 : 1))
    expect(Array.from(boundaryMask(labels, 10, 1, 1)).reduce((a, b) => a + b, 0)).toBe(1)
    expect(Array.from(boundaryMask(labels, 10, 1, 3)).reduce((a, b) => a + b, 0)).toBe(5)
  })
})

// Straightforward reference: explicit edge list and a comparison sort.
function referenceFelzenszwalb(pixels: Uint8ClampedArray, width: number, height: number, scale: number, minSize: number) {
  const [r, g, b] = blurRgbPlanes(pixels, width, height, 0.8)
  const edges: Array<[number, number, number]> = []
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const i = x + y * width
      for (const [dx, dy] of [
        [1, 0],
        [0, 1],
        [1, 1],
        [1, -1],
      ]) {
        const nx = x + dx
        const ny = y + dy
        if (nx < width && ny >= 0 && ny < height) {
          const j = nx + ny * width
          edges.push([Math.hypot(r[i] - r[j], g[i] - g[j], b[i] - b[j]), i, j])
        }
      }
    }
  }
  edges.sort((p, q) => p[0] - q[0])
  const set = new DisjointSet(width * height)
  const internal = new Float32Array(width * height)
  for (const [weight, i, j] of edges) {
    const a = set.find(i)
    const c = set.find(j)
    if (a !== c && weight <= internal[a] + scale / set.size[a] && weight <= internal[c] + scale / set.size[c]) {
      internal[set.unionRoots(a, c)] = weight
    }
  }
  for (const [, i, j] of edges) {
    const a = set.find(i)
    const c = set.find(j)
    if (a !== c && (set.size[a] < minSize || set.size[c] < minSize)) {
      set.unionRoots(a, c)
    }
  }
  return Array.from({ length: width * height }, (_, index) => set.find(index))
}

// True when two labelings describe the same partition up to renaming.
function samePartition(a: ArrayLike<number>, b: ArrayLike<number>) {
  const forward = new Map<number, number>()
  const backward = new Map<number, number>()
  for (let index = 0; index < a.length; index += 1) {
    if ((forward.get(a[index]) ?? b[index]) !== b[index] || (backward.get(b[index]) ?? a[index]) !== a[index]) {
      return false
    }
    forward.set(a[index], b[index])
    backward.set(b[index], a[index])
  }
  return true
}

describe('segmentFelzenszwalb', () => {
  it('splits two flat color fields', () => {
    const width = 80
    const height = 60
    const pixels = makeImage(width, height, (x) => (x < 40 ? [200, 60, 50] : [40, 80, 190]))
    // The pre-blur turns the hard edge into thin transition columns; the
    // minimum size absorbs them.
    const { labels, labelCount } = segmentFelzenszwalb(pixels, width, height, { scale: 500, minSize: 200 })
    expect(labelCount).toBe(2)
    expect(labels[0]).not.toBe(labels[width - 1])
    expect(labels[0]).toBe(labels[39 + 59 * width])
  })

  it('matches the comparison-sort reference despite bucketed weights', () => {
    const width = 90
    const height = 70
    const pixels = makeImage(width, height, (x, y) => {
      const noise = ((x * 7 + y * 13) % 5) - 2
      const shade = Math.round(60 + 120 * Math.sin(x / 9) * Math.cos(y / 11))
      return x + y < 80 ? [200 + noise, shade, 50] : [40, 80 + noise, shade + 60]
    })
    for (const [scale, minSize] of [
      [300, 20],
      [700, 120],
    ]) {
      const ours = segmentFelzenszwalb(pixels, width, height, { scale, minSize })
      const reference = referenceFelzenszwalb(pixels, width, height, scale, minSize)
      expect(new Set(reference).size).toBe(ours.labelCount)
      expect(samePartition(ours.labels, reference)).toBe(true)
    }
  })

  it('handles a preview-sized image quickly thanks to bucket sorting', () => {
    const width = 1280
    const height = 960
    const pixels = makeImage(width, height, (x, y) => [(x * 3 + y) % 256, (y * 5) % 256, (x ^ y) % 256])
    const started = performance.now()
    const { labelCount } = segmentFelzenszwalb(pixels, width, height, { scale: 700, minSize: 120 })
    const elapsed = performance.now() - started
    expect(labelCount).toBeGreaterThan(0)
    // The previous comparison sort alone took over 2 s for this size.
    expect(elapsed).toBeLessThan(4000)
  })
})

describe('SLIC clean-up', () => {
  it('merges disconnected label fragments into neighbors', () => {
    const width = 32
    const height = 32
    const pixels = makeImage(width, height, () => [120, 120, 120])
    // Label 0 on the left half, label 1 on the right, plus a stray
    // two-pixel fragment of label 0 inside label 1.
    const labels = new Int32Array(width * height).map((_, index) => (index % width < 16 ? 0 : 1))
    labels[24 + 10 * width] = 0
    labels[25 + 10 * width] = 0
    // Both halves share one color, so similar-color merging is off here.
    const params = { ...defaultParams('slic'), size: 16, cleanup: true, merge: 0 }
    const analysis = slicFilter.analyze({ pixels, width, height, scale: 1 }, params, { labels })
    expect(analysis.stats.count).toBe(2)
    const output = slicFilter.render({ pixels, width, height, scale: 1 }, analysis, params)
    expect(output.pixels.length).toBe(width * height * 4)
  })

  it('merges neighboring superpixels of nearly the same color only', () => {
    const width = 48
    const height = 16
    // Three vertical stripes: two near-identical grays, then a red.
    const pixels = makeImage(width, height, (x) => (x < 16 ? [120, 120, 120] : x < 32 ? [123, 121, 120] : [200, 40, 40]))
    const labels = new Int32Array(width * height).map((_, index) => Math.floor((index % width) / 16))
    const params = { ...defaultParams('slic'), size: 16, cleanup: false, merge: 0.3 }
    const analysis = slicFilter.analyze({ pixels, width, height, scale: 1 }, params, { labels })
    expect(analysis.stats.count).toBe(2)
    expect(analysis.labels[0]).toBe(analysis.labels[20])
    expect(analysis.labels[0]).not.toBe(analysis.labels[40])
  })
})
