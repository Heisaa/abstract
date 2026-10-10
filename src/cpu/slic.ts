import { bool, num } from '../filters/registry'
import { boundaryMask, connectedComponents, DisjointSet, mergeSmallRegions } from './regions'
import { paintMask, paintRegions, perceptualRegionColors, regionStats, type RegionStats } from './regionColor'
import type { CpuFilter } from './types'

type SlicAnalysis = {
  labels: Int32Array
  stats: RegionStats
}

// Post-processing for GPU SLIC: the clustering itself runs on the GPU, but
// k-means labels can be split into disconnected fragments. Connectivity
// enforcement (merging fragments smaller than a quarter of a superpixel into
// their longest-border neighbor) and OKLab coloring happen here.
export const slicFilter: CpuFilter<SlicAnalysis> = {
  analyze(image, params, extra) {
    if (!extra.labels || extra.labels.length !== image.width * image.height) {
      throw new Error('SLIC labels are missing or do not match the image size.')
    }

    let labels = extra.labels
    let count = 0
    if (bool(params, 'cleanup')) {
      const size = num(params, 'size')
      const components = connectedComponents(labels, image.width, image.height)
      const merged = mergeSmallRegions(components, image.width, image.height, Math.round((size * size) / 4))
      labels = merged.labels
      count = merged.count
    } else {
      for (let index = 0; index < labels.length; index += 1) {
        count = Math.max(count, labels[index] + 1)
      }
    }

    const tolerance = num(params, 'merge') * MAX_MERGE_DISTANCE
    if (tolerance > 0) {
      const merged = mergeSimilarRegions(labels, count, image.width, image.height, regionStats(image.pixels, labels, count), tolerance)
      labels = merged.labels
      count = merged.count
    }

    return { labels, stats: regionStats(image.pixels, labels, count) }
  },
  render(image, { labels, stats }, params) {
    const pixels = paintRegions(labels, perceptualRegionColors(stats, num(params, 'saturation')))
    if (bool(params, 'outline')) {
      paintMask(pixels, boundaryMask(labels, image.width, image.height, 1), [40, 40, 40])
    }
    return { pixels }
  },
}

// OKLab distance that "Merge similar" = 1 corresponds to.
const MAX_MERGE_DISTANCE = 0.08

// Joins neighboring superpixels whose mean OKLab colors are within
// `tolerance`, cheapest borders first, comparing the running means of the
// merged groups. Flat passages (sky, wall, skin in light) become one large
// patch instead of a grid of near-identical tiles, while busy areas keep
// their small superpixels: a brush-size map.
export function mergeSimilarRegions(labels: Int32Array, count: number, width: number, height: number, stats: RegionStats, tolerance: number) {
  const pairs = new Set<number>()
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const label = labels[x + y * width]
      if (x + 1 < width && labels[x + 1 + y * width] !== label) {
        const other = labels[x + 1 + y * width]
        pairs.add(Math.min(label, other) * count + Math.max(label, other))
      }
      if (y + 1 < height && labels[x + (y + 1) * width] !== label) {
        const other = labels[x + (y + 1) * width]
        pairs.add(Math.min(label, other) * count + Math.max(label, other))
      }
    }
  }

  const L = Float64Array.from(stats.meanL)
  const A = Float64Array.from(stats.meanA)
  const B = Float64Array.from(stats.meanB)
  const sizes = Float64Array.from(stats.sizes)
  const distance = (a: number, b: number) => Math.hypot(L[a] - L[b], A[a] - A[b], B[a] - B[b])
  const edges = Array.from(pairs, (pair) => {
    const a = Math.floor(pair / count)
    const b = pair % count
    return { a, b, weight: distance(a, b) }
  }).filter((edge) => edge.weight <= tolerance)
  edges.sort((left, right) => left.weight - right.weight)

  const sets = new DisjointSet(count)
  for (const edge of edges) {
    const a = sets.find(edge.a)
    const b = sets.find(edge.b)
    if (a === b || distance(a, b) > tolerance) {
      continue
    }
    const total = sizes[a] + sizes[b]
    const root = sets.unionRoots(a, b)
    L[root] = (L[a] * sizes[a] + L[b] * sizes[b]) / total
    A[root] = (A[a] * sizes[a] + A[b] * sizes[b]) / total
    B[root] = (B[a] * sizes[a] + B[b] * sizes[b]) / total
    sizes[root] = total
  }

  const remap = new Int32Array(count).fill(-1)
  let next = 0
  const merged = new Int32Array(labels.length)
  for (let index = 0; index < labels.length; index += 1) {
    const root = sets.find(labels[index])
    if (remap[root] === -1) {
      remap[root] = next++
    }
    merged[index] = remap[root]
  }
  return { labels: merged, count: next }
}
