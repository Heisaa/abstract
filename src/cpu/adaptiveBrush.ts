import { blurRgbPlanes, clamp, colorDistance, hashToUnit, mixChannel, smoothstep } from './common'
import { num } from '../filters/registry'
import type { CpuFilter, CpuImage } from './types'

type BrushField = {
  pixels: Uint8ClampedArray
  labels: Uint32Array
  labelCount: number
}

type AdaptiveAnalysis = {
  large: BrushField
  medium: BrushField
  small: BrushField
  largeDetail: Float32Array
  mediumDetail: Float32Array
  localDetail: Float32Array
}

export const adaptiveBrushFilter: CpuFilter<AdaptiveAnalysis> = {
  analyze(image, params) {
    const largeBrush = Math.max(2, Math.round(num(params, 'large')))
    const mediumBrush = Math.max(2, Math.min(largeBrush, Math.round(num(params, 'medium'))))
    const smallBrush = Math.max(2, Math.min(mediumBrush, Math.round(num(params, 'small'))))
    const large = createBrushField(image, largeBrush, 0x15a4)
    const medium = createBrushField(image, mediumBrush, 0x89f1)
    const small = createBrushField(image, smallBrush, 0x4c2d)
    const localDetail = createLocalContrast(image)

    return {
      large,
      medium,
      small,
      largeDetail: createCellDetailMap(localDetail, large.labels, large.labelCount),
      mediumDetail: createCellDetailMap(localDetail, medium.labels, medium.labelCount),
      localDetail,
    }
  },
  render(image, analysis, params) {
    const { pixels } = image
    const detailThreshold = clamp(num(params, 'threshold'), 0, 1)
    const detailStrength = clamp(num(params, 'accent'), 0, 1)
    const output = new Uint8ClampedArray(pixels.length)

    for (let index = 0; index < image.width * image.height; index += 1) {
      const offset = index * 4
      const useMediumBrush = analysis.largeDetail[analysis.large.labels[index]] > detailThreshold
      const useSmallBrush =
        useMediumBrush && analysis.mediumDetail[analysis.medium.labels[index]] > Math.min(1, detailThreshold + 0.12)
      const source = useSmallBrush ? analysis.small.pixels : useMediumBrush ? analysis.medium.pixels : analysis.large.pixels
      const accentMix = useSmallBrush ? Math.min(0.5, smoothstep(0.03, 0.18, analysis.localDetail[index]) * detailStrength) : 0

      output[offset] = mixChannel(source[offset], pixels[offset], accentMix)
      output[offset + 1] = mixChannel(source[offset + 1], pixels[offset + 1], accentMix)
      output[offset + 2] = mixChannel(source[offset + 2], pixels[offset + 2], accentMix)
      output[offset + 3] = 255
    }

    return { pixels: output }
  },
}

function createLocalContrast({ pixels, width, height }: CpuImage) {
  const detail = new Float32Array(width * height)
  for (let y = 1; y < height; y += 1) {
    for (let x = 1; x < width; x += 1) {
      const offset = (x + y * width) * 4
      const dx = colorDistance(pixels, offset, offset - 4)
      const dy = colorDistance(pixels, offset, offset - width * 4)
      detail[x + y * width] = Math.min(1, Math.sqrt(dx * dx + dy * dy) / 255)
    }
  }
  return detail
}

// Color distance (RGB units) at which a pixel prefers a neighboring cell
// over the nearest one a full brush width away.
const CELL_COLOR_TOLERANCE = 52
const REFINE_ITERATIONS = 2

// Brush cells of roughly brushSize, each filled with its mean color. Seeds
// are a jittered grid; a few SLIC-style iterations then reassign pixels by
// distance plus color difference, so a cell stops at an object boundary
// instead of averaging both sides into mud.
function createBrushField({ pixels, width, height }: CpuImage, brushSize: number, seed: number): BrushField {
  const gridWidth = Math.ceil(width / brushSize) + 2
  const gridHeight = Math.ceil(height / brushSize) + 2
  const labelCount = gridWidth * gridHeight
  const centersX = new Float32Array(labelCount)
  const centersY = new Float32Array(labelCount)
  const centerColors = new Float32Array(labelCount * 3)
  for (let cellY = 0; cellY < gridHeight; cellY += 1) {
    for (let cellX = 0; cellX < gridWidth; cellX += 1) {
      const label = cellX + cellY * gridWidth
      centersX[label] = (cellX - 0.5 + (hashToUnit(cellX, cellY, seed) - 0.5) * 0.72) * brushSize
      centersY[label] = (cellY - 0.5 + (hashToUnit(cellX, cellY, seed ^ 0x6d2b79f5) - 0.5) * 0.72) * brushSize
    }
  }

  // Cells are assigned on a copy blurred to a fraction of the brush, so fine
  // texture (knit, foliage, grain) cannot shred a cell into its own pattern.
  const [blurRed, blurGreen, blurBlue] = blurRgbPlanes(pixels, width, height, Math.max(1, brushSize * 0.08))
  const sums = new Float64Array(labelCount * 5)
  const counts = new Uint32Array(labelCount)
  const labels = new Uint32Array(width * height)
  const spatialWeight = 1 / (brushSize * brushSize)
  const colorWeight = 1 / (CELL_COLOR_TOLERANCE * CELL_COLOR_TOLERANCE)

  for (let iteration = 0; iteration <= REFINE_ITERATIONS; iteration += 1) {
    // The first pass is plain Voronoi; later passes also weigh color.
    const useColor = iteration > 0
    sums.fill(0)
    counts.fill(0)
    for (let y = 0; y < height; y += 1) {
      const cellY = Math.floor(y / brushSize) + 1
      for (let x = 0; x < width; x += 1) {
        const cellX = Math.floor(x / brushSize) + 1
        const index = x + y * width
        const red = blurRed[index]
        const green = blurGreen[index]
        const blue = blurBlue[index]
        let bestLabel = 0
        let bestDistance = Number.POSITIVE_INFINITY
        for (let offsetY = -1; offsetY <= 1; offsetY += 1) {
          for (let offsetX = -1; offsetX <= 1; offsetX += 1) {
            const label = cellX + offsetX + (cellY + offsetY) * gridWidth
            const dx = x - centersX[label]
            const dy = y - centersY[label]
            let distance = (dx * dx + dy * dy) * spatialWeight
            if (useColor) {
              const dr = red - centerColors[label * 3]
              const dg = green - centerColors[label * 3 + 1]
              const db = blue - centerColors[label * 3 + 2]
              distance += (dr * dr + dg * dg + db * db) * colorWeight
            }
            if (distance < bestDistance) {
              bestDistance = distance
              bestLabel = label
            }
          }
        }

        labels[index] = bestLabel
        sums[bestLabel * 5] += pixels[index * 4]
        sums[bestLabel * 5 + 1] += pixels[index * 4 + 1]
        sums[bestLabel * 5 + 2] += pixels[index * 4 + 2]
        sums[bestLabel * 5 + 3] += x
        sums[bestLabel * 5 + 4] += y
        counts[bestLabel] += 1
      }
    }

    for (let label = 0; label < labelCount; label += 1) {
      const count = counts[label]
      if (count === 0) {
        // Empty cells keep their seed but can no longer win on color.
        centerColors.fill(-1e4, label * 3, label * 3 + 3)
        continue
      }
      centerColors[label * 3] = sums[label * 5] / count
      centerColors[label * 3 + 1] = sums[label * 5 + 1] / count
      centerColors[label * 3 + 2] = sums[label * 5 + 2] / count
      centersX[label] = sums[label * 5 + 3] / count
      centersY[label] = sums[label * 5 + 4] / count
    }
  }

  const output = new Uint8ClampedArray(pixels.length)
  for (let index = 0; index < labels.length; index += 1) {
    const label = labels[index]
    output[index * 4] = centerColors[label * 3]
    output[index * 4 + 1] = centerColors[label * 3 + 1]
    output[index * 4 + 2] = centerColors[label * 3 + 2]
    output[index * 4 + 3] = 255
  }

  return { pixels: output, labels, labelCount }
}

// Mean local contrast per cell, rank-normalized to 0..1 across cells.
function createCellDetailMap(localDetail: Float32Array, labels: Uint32Array, labelCount: number) {
  const totals = new Float32Array(labelCount)
  const counts = new Uint32Array(labelCount)
  for (let index = 0; index < labels.length; index += 1) {
    totals[labels[index]] += localDetail[index]
    counts[labels[index]] += 1
  }
  for (let label = 0; label < labelCount; label += 1) {
    totals[label] = counts[label] > 0 ? totals[label] / counts[label] : 0
  }
  rankNormalize(totals)
  return totals
}

export function rankNormalize(values: Float32Array) {
  const ordered = Float32Array.from(values).sort()
  if (ordered.length === 0) {
    return
  }
  if (ordered[0] === ordered[ordered.length - 1]) {
    values.fill(0)
    return
  }

  let firstDifferent = 1
  while (ordered[firstDifferent] <= ordered[0]) {
    firstDifferent += 1
  }

  const denominator = Math.max(1, ordered.length - firstDifferent)
  for (let index = 0; index < values.length; index += 1) {
    const rank = upperBound(ordered, values[index])
    values[index] = rank <= firstDifferent ? 0 : clamp((rank - firstDifferent) / denominator, 0, 1)
  }
}

function upperBound(values: Float32Array, target: number) {
  let low = 0
  let high = values.length
  while (low < high) {
    const middle = (low + high) >>> 1
    if (values[middle] <= target) {
      low = middle + 1
    } else {
      high = middle
    }
  }
  return low
}
