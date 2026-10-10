import { blurRgbPlanes, gaussianBlurFloat } from './common'
import { pixelsToOklabPlanes } from './color'
import { DisjointSet } from './regions'

// Felzenszwalb-Huttenlocher graph segmentation on an 8-connected pixel grid.
//
// Weights are color distances in RGB, or in chroma-weighted OKLab.
//
// Edges are implicit (pixel index * 4 + direction) and are ordered with a
// counting sort over quantized weights rather than a comparison sort: weights
// are bounded RGB distances, so 65535 buckets (~0.007 units wide) order ~5M
// edges in tens of milliseconds instead of seconds. Merge decisions use the
// exact weight, so only ties within one bucket can order differently.

const WEIGHT_BUCKETS = 0xffff
const INVALID_EDGE = 0xffff

// OKLab planes are scaled so black-to-white spans about the same distance as
// in RGB (keeping `scale` comparable), with a/b doubled so a hue or chroma
// shift separates fields as readily as a value step. In plain RGB a red
// accent on skin or a tan head against foliage are close enough to merge.
const OKLAB_LIGHTNESS_SCALE = 400
const OKLAB_CHROMA_SCALE = 800
// sRGB spans less than 0.52 along OKLab a and b.
const MAX_WEIGHT = {
  rgb: Math.sqrt(3 * 255 * 255),
  oklab: Math.hypot(OKLAB_LIGHTNESS_SCALE, 0.52 * OKLAB_CHROMA_SCALE, 0.52 * OKLAB_CHROMA_SCALE),
}

// dx, dy for directions: right, down, down-right, up-right.
const DIRECTION_DX = [1, 0, 1, 1]
const DIRECTION_DY = [0, 1, 1, -1]

export type FelzenszwalbOptions = {
  scale: number
  minSize: number
  // Pre-smoothing sigma from the original paper (0.8) to suppress noise.
  sigma?: number
  colorSpace?: 'rgb' | 'oklab'
}

function colorPlanes(pixels: Uint8ClampedArray, width: number, height: number, sigma: number, colorSpace: 'rgb' | 'oklab') {
  if (colorSpace === 'rgb') {
    return blurRgbPlanes(pixels, width, height, sigma)
  }
  const { L, A, B } = pixelsToOklabPlanes(pixels, width * height)
  for (let index = 0; index < L.length; index += 1) {
    L[index] *= OKLAB_LIGHTNESS_SCALE
    A[index] *= OKLAB_CHROMA_SCALE
    B[index] *= OKLAB_CHROMA_SCALE
  }
  return [L, A, B].map((plane) => gaussianBlurFloat(plane, width, height, sigma)) as [Float32Array, Float32Array, Float32Array]
}

export function segmentFelzenszwalb(
  pixels: Uint8ClampedArray,
  width: number,
  height: number,
  { scale, minSize, sigma = 0.8, colorSpace = 'rgb' }: FelzenszwalbOptions,
) {
  const count = width * height
  const [red, green, blue] = colorPlanes(pixels, width, height, sigma, colorSpace)
  const bucketScale = WEIGHT_BUCKETS / (MAX_WEIGHT[colorSpace] + 1e-3)
  const keys = new Uint16Array(count * 4)
  const bucketCounts = new Uint32Array(WEIGHT_BUCKETS + 1)
  let edgeCount = 0

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = x + y * width
      for (let direction = 0; direction < 4; direction += 1) {
        const nx = x + DIRECTION_DX[direction]
        const ny = y + DIRECTION_DY[direction]
        if (nx >= width || ny < 0 || ny >= height) {
          keys[index * 4 + direction] = INVALID_EDGE
          continue
        }

        const neighbor = nx + ny * width
        const dr = red[index] - red[neighbor]
        const dg = green[index] - green[neighbor]
        const db = blue[index] - blue[neighbor]
        const key = Math.min(WEIGHT_BUCKETS - 1, Math.floor(Math.sqrt(dr * dr + dg * dg + db * db) * bucketScale))
        keys[index * 4 + direction] = key
        bucketCounts[key + 1] += 1
        edgeCount += 1
      }
    }
  }

  for (let bucket = 1; bucket <= WEIGHT_BUCKETS; bucket += 1) {
    bucketCounts[bucket] += bucketCounts[bucket - 1]
  }

  const order = new Uint32Array(edgeCount)
  for (let edge = 0; edge < keys.length; edge += 1) {
    const key = keys[edge]
    if (key !== INVALID_EDGE) {
      order[bucketCounts[key]++] = edge
    }
  }

  const forest = new DisjointSet(count)
  const internal = new Float32Array(count)
  const k = Math.max(1, scale)
  const minimum = Math.max(1, Math.round(minSize))

  for (let i = 0; i < edgeCount; i += 1) {
    const edge = order[i]
    const index = edge >>> 2
    const direction = edge & 3
    const neighbor = index + DIRECTION_DX[direction] + DIRECTION_DY[direction] * width
    const a = forest.find(index)
    const b = forest.find(neighbor)
    if (a === b) {
      continue
    }

    const dr = red[index] - red[neighbor]
    const dg = green[index] - green[neighbor]
    const db = blue[index] - blue[neighbor]
    const weight = Math.sqrt(dr * dr + dg * dg + db * db)
    if (weight <= internal[a] + k / forest.size[a] && weight <= internal[b] + k / forest.size[b]) {
      const root = forest.unionRoots(a, b)
      internal[root] = weight
    }
  }

  for (let i = 0; i < edgeCount; i += 1) {
    const edge = order[i]
    const index = edge >>> 2
    const direction = edge & 3
    const a = forest.find(index)
    const b = forest.find(index + DIRECTION_DX[direction] + DIRECTION_DY[direction] * width)
    if (a !== b && (forest.size[a] < minimum || forest.size[b] < minimum)) {
      forest.unionRoots(a, b)
    }
  }

  const labels = new Int32Array(count)
  const rootToLabel = new Int32Array(count).fill(-1)
  let labelCount = 0
  for (let index = 0; index < count; index += 1) {
    const root = forest.find(index)
    if (rootToLabel[root] === -1) {
      rootToLabel[root] = labelCount++
    }
    labels[index] = rootToLabel[root]
  }

  return { labels, labelCount }
}
