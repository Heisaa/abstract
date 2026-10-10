import {
  averageRectColor,
  blendAngles,
  blurRgbPlanes,
  boxBlurFloat,
  clamp,
  createRandom,
  hashToUnit,
  luminanceGradient,
  normalizePixelDetailMap,
  quantizeRgb,
  rgbCss,
  smoothstep,
  type Rgb,
} from './common'
import { num, str } from '../filters/registry'
import type { CpuFilter, CpuImage } from './types'

type StrokeFlowField = {
  cellSize: number
  width: number
  height: number
  angles: Float32Array
  confidence: Float32Array
}

type PainterAnalysis = {
  edgeMap: Float32Array
  activityMap: Float32Array
  flowField: StrokeFlowField
}

type Canvas2D = OffscreenCanvasRenderingContext2D

// Underpainting, then layered strokes from a large brush down to a small one
// (after Hertzmann's painterly rendering). Each layer paints against the
// source blurred to its brush size, and finer layers only add strokes where
// the canvas still differs from that reference, so detail goes where the
// image needs it rather than everywhere. Strokes follow the flow field and
// end where the image color departs from the stroke's color, so they stay
// inside the shape they describe instead of dragging across its contour.
export const painterFilter: CpuFilter<PainterAnalysis> = {
  analyze(image, params) {
    const { pixels, width, height } = image
    const baseSize = Math.max(3, Math.round(num(params, 'size')))
    const edgeMap = createPixelDetailMap(image)
    // Color mode biases finer passes toward shifting color; edge mode toward
    // structural edges.
    const activityMap = str(params, 'detailSource') === 'color' ? createColorActivityMap(image, baseSize) : edgeMap
    const flowField = createStrokeFlowField(pixels, width, height, Math.max(8, Math.round(baseSize * 2.2)))
    return { edgeMap, activityMap, flowField }
  },
  render(image, { edgeMap, activityMap, flowField }, params) {
    const { width, height } = image
    const canvas = new OffscreenCanvas(width, height)
    const context = canvas.getContext('2d', { willReadFrequently: true })
    if (!context) {
      throw new Error('Could not create the painting canvas.')
    }

    const settings: PaintSettings = {
      abstraction: clamp(num(params, 'abstraction'), 0, 1),
      detail: clamp(num(params, 'detail'), 0, 1),
      edgeFidelity: clamp(num(params, 'edgeFidelity'), 0, 1),
      simplification: clamp(num(params, 'colorSimplify'), 0, 1),
      texture: clamp(num(params, 'texture'), 0, 1),
      lengthScale: clamp(num(params, 'length'), 1, 5),
    }
    const baseSize = Math.max(3, Math.round(num(params, 'size')))
    const layers = Math.max(1, Math.min(4, Math.round(num(params, 'layers'))))
    const detailBias = str(params, 'detailSource') === 'color' ? activityMap : edgeMap

    // Smooth base avoids square underpainting tiles showing through gaps
    // between strokes; blocks give a more visibly chunky block-in.
    if (str(params, 'underpainting') === 'smooth') {
      paintSmoothUnderpainting(context, image, baseSize, settings.abstraction, settings.simplification)
    } else {
      paintBlockUnderpainting(context, image, baseSize, settings.abstraction, settings.simplification)
    }

    const layerScale = [3.8, 2.05, 1.02, 0.52]
    const random = createRandom(0x7a1e + baseSize)
    for (let layer = 0; layer < layers; layer += 1) {
      const size = Math.max(2, baseSize * layerScale[layer])
      const reference = createReference(image, size)
      const painted = context.getImageData(0, 0, width, height).data
      const strokes = planLayer({ layer, layers, size, reference, painted, detailBias, flowField, settings, random })
      for (const stroke of strokes) {
        drawStroke(context, stroke)
      }
    }

    return { pixels: context.getImageData(0, 0, width, height).data }
  },
}

type PaintSettings = {
  abstraction: number
  detail: number
  edgeFidelity: number
  simplification: number
  texture: number
  lengthScale: number
}

// Source blurred to a brush size, as float RGB planes.
type Reference = {
  width: number
  height: number
  planes: [Float32Array, Float32Array, Float32Array]
}

type Stroke = {
  // Flat x, y pairs; always at least two points.
  points: number[]
  width: number
  color: Rgb
  alpha: number
  texture: number
  seed: number
}

function createReference(image: CpuImage, size: number): Reference {
  return { width: image.width, height: image.height, planes: blurRgbPlanes(image.pixels, image.width, image.height, size * 0.3) }
}

function referenceColor(reference: Reference, x: number, y: number): Rgb {
  const index = clamp(Math.round(x), 0, reference.width - 1) + clamp(Math.round(y), 0, reference.height - 1) * reference.width
  return [reference.planes[0][index], reference.planes[1][index], reference.planes[2][index]]
}

function rgbDistance(a: Rgb, b: Rgb) {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])
}

function canvasDistance(painted: Uint8ClampedArray, reference: Reference, index: number) {
  const dr = painted[index * 4] - reference.planes[0][index]
  const dg = painted[index * 4 + 1] - reference.planes[1][index]
  const db = painted[index * 4 + 2] - reference.planes[2][index]
  return Math.sqrt(dr * dr + dg * dg + db * db)
}

type LayerPlan = {
  layer: number
  layers: number
  size: number
  reference: Reference
  painted: Uint8ClampedArray
  detailBias: Float32Array
  flowField: StrokeFlowField
  settings: PaintSettings
  random: () => number
}

// Places one layer's strokes on a jittered grid. The first layer covers the
// whole canvas; later layers only start strokes in cells whose mean error
// against the reference exceeds the detail threshold, at the worst pixel.
function planLayer(plan: LayerPlan) {
  const { layer, layers, size, reference, painted, detailBias, settings, random } = plan
  const { width, height } = reference
  const fineLayer = layer / Math.max(1, layers - 1)
  const spacing = Math.max(2, size * (layer === 0 ? 0.9 : 0.72))
  // Mean RGB error a cell may keep before it gets another stroke.
  const threshold = (4 + (1 - settings.detail) * 26) * (0.7 + settings.abstraction * 0.8)
  const sampleStep = Math.max(1, Math.floor(spacing / 6))
  const strokes: Stroke[] = []

  for (let cellY = 0; cellY * spacing < height; cellY += 1) {
    for (let cellX = 0; cellX * spacing < width; cellX += 1) {
      const x0 = Math.floor(cellX * spacing)
      const y0 = Math.floor(cellY * spacing)
      const x1 = Math.min(width, Math.floor((cellX + 1) * spacing))
      const y1 = Math.min(height, Math.floor((cellY + 1) * spacing))
      let startX: number
      let startY: number
      if (layer === 0) {
        startX = x0 + (x1 - x0) * (0.2 + hashToUnit(cellX, cellY, 0x2471) * 0.6)
        startY = y0 + (y1 - y0) * (0.2 + hashToUnit(cellX, cellY, 0x71b5) * 0.6)
      } else {
        let total = 0
        let count = 0
        let worst = -1
        startX = x0
        startY = y0
        for (let y = y0; y < y1; y += sampleStep) {
          for (let x = x0; x < x1; x += sampleStep) {
            const error = canvasDistance(painted, reference, x + y * width)
            total += error
            count += 1
            if (error > worst) {
              worst = error
              startX = x
              startY = y
            }
          }
        }
        const bias = detailBias[Math.round(startX) + Math.round(startY) * width]
        if (count === 0 || total / count < threshold * (1.25 - bias * 0.6)) {
          continue
        }
      }
      strokes.push(traceStroke(plan, startX, startY, fineLayer))
    }
  }

  // Random order so later strokes do not systematically overlap earlier ones
  // in grid order.
  for (let index = strokes.length - 1; index > 0; index -= 1) {
    const swap = Math.floor(random() * (index + 1))
    const held = strokes[index]
    strokes[index] = strokes[swap]
    strokes[swap] = held
  }
  return strokes
}

// Grows a stroke both ways from (x, y) along the flow. It ends at the length
// limit, where the reference color departs from the stroke's color (an
// edge), or, on refining layers, where the canvas is already closer to the
// reference than this stroke would be.
function traceStroke(plan: LayerPlan, x: number, y: number, fineLayer: number): Stroke {
  const { layer, size, reference, painted, flowField, settings } = plan
  const strokeWidth = Math.max(1.2, size * (0.64 - fineLayer * 0.14))
  const color = quantizeRgb(referenceColor(reference, x, y), settings.simplification)
  const quiet = 1 - plan.detailBias[Math.round(x) + Math.round(y) * reference.width]
  const maxLength = size * settings.lengthScale * (layer < 2 ? 0.9 + quiet * settings.abstraction * 0.7 : 0.75)
  const stepLength = Math.max(1, strokeWidth * 0.5)
  // RGB distance from the stroke color at which the brush lifts.
  const stopDistance = 30 + (1 - settings.edgeFidelity) * 70
  const wobble = (hashToUnit(Math.round(x), Math.round(y), layer * 0xa53d) - 0.5) * (0.16 + fineLayer * 0.2)

  const trace = (sign: number) => {
    const points: number[] = []
    let px = x
    let py = y
    let directionX = 0
    let directionY = 0
    for (let travelled = 0; travelled < maxLength * 0.5; travelled += stepLength) {
      const angle = strokeAngle(reference, flowField, px, py, size, settings.edgeFidelity) + wobble
      let dx = Math.cos(angle) * sign
      let dy = Math.sin(angle) * sign
      if (travelled > 0) {
        // Orientation has no sign; keep heading the same way, and damp
        // sharp turns so strokes curve rather than kink.
        if (dx * directionX + dy * directionY < 0) {
          dx = -dx
          dy = -dy
        }
        dx = dx * 0.6 + directionX * 0.4
        dy = dy * 0.6 + directionY * 0.4
        const norm = Math.hypot(dx, dy) || 1
        dx /= norm
        dy /= norm
      }
      directionX = dx
      directionY = dy
      const nextX = px + dx * stepLength
      const nextY = py + dy * stepLength
      if (nextX < 0 || nextY < 0 || nextX >= reference.width || nextY >= reference.height) {
        break
      }
      const target = referenceColor(reference, nextX, nextY)
      const strokeError = rgbDistance(target, color)
      if (strokeError > stopDistance) {
        break
      }
      if (layer > 0 && travelled > strokeWidth) {
        const index = Math.round(nextX) + Math.round(nextY) * reference.width
        if (canvasDistance(painted, reference, index) < strokeError) {
          break
        }
      }
      px = nextX
      py = nextY
      points.push(px, py)
    }
    return points
  }

  const forward = trace(1)
  const backward = trace(-1)
  const points: number[] = []
  for (let index = backward.length - 2; index >= 0; index -= 2) {
    points.push(backward[index], backward[index + 1])
  }
  points.push(x, y)
  points.push(...forward)
  if (points.length < 4) {
    // Nowhere to go: a short dab along the local direction.
    const angle = strokeAngle(reference, flowField, x, y, size, settings.edgeFidelity) + wobble
    const reach = strokeWidth * 0.45
    points.splice(0, points.length, x - Math.cos(angle) * reach, y - Math.sin(angle) * reach, x + Math.cos(angle) * reach, y + Math.sin(angle) * reach)
  }

  const seed = Math.round(x) * 17 + Math.round(y) * 31 + layer * 101
  const valueShift = (hashToUnit(Math.round(x), Math.round(y), layer * 0x345f) - 0.5) * 18 * settings.texture
  return {
    points,
    width: strokeWidth,
    color: shiftColor(color, valueShift),
    alpha: (0.9 - fineLayer * 0.1) * (0.9 + hashToUnit(Math.round(y), Math.round(x), layer * 0x8849) * 0.1),
    texture: settings.texture,
    seed,
  }
}

// Stroke orientation: along the smoothed flow field, turning toward the local
// contour (perpendicular to the reference's gradient) where an edge is
// strong, by edge fidelity.
function strokeAngle(reference: Reference, flowField: StrokeFlowField, x: number, y: number, size: number, edgeFidelity: number) {
  const flow = sampleStrokeFlow(flowField, x, y)
  const flowAngle = flow.confidence > 0.02 ? flow.angle : coherentFlowAngle(x, y, reference.width, reference.height)
  const reach = Math.max(1, Math.round(size * 0.25))
  const lum = (sx: number, sy: number) => {
    const [r, g, b] = referenceColor(reference, sx, sy)
    return r * 0.2126 + g * 0.7152 + b * 0.0722
  }
  const gx = lum(x + reach, y) - lum(x - reach, y)
  const gy = lum(x, y + reach) - lum(x, y - reach)
  const strength = Math.hypot(gx, gy) / 255
  const localAngle = Math.atan2(gy, gx) + Math.PI * 0.5
  return blendAngles(flowAngle, localAngle, smoothstep(0.04, 0.3, strength) * edgeFidelity)
}

// Round-capped polyline in the stroke color, with faint bristle streaks
// following the same path.
function drawStroke(context: Canvas2D, stroke: Stroke) {
  const { points, width: strokeWidth, color } = stroke
  const count = points.length / 2
  context.lineCap = 'round'
  context.lineJoin = 'round'
  context.globalAlpha = clamp(stroke.alpha, 0, 1)
  context.strokeStyle = rgbCss(color)
  context.lineWidth = Math.max(1, strokeWidth)
  context.beginPath()
  context.moveTo(points[0], points[1])
  for (let index = 1; index < count; index += 1) {
    context.lineTo(points[index * 2], points[index * 2 + 1])
  }
  context.stroke()

  if (stroke.texture <= 0) {
    return
  }
  const bristleCount = Math.max(2, Math.round(strokeWidth * 0.42))
  for (let bristle = 0; bristle < bristleCount; bristle += 1) {
    const offset = (bristle / Math.max(1, bristleCount - 1) - 0.5) * strokeWidth * 0.8
    const amount = (hashToUnit(stroke.seed, bristle, 0x493f) - 0.5) * 38 * stroke.texture
    context.globalAlpha = clamp(stroke.alpha * (0.18 + stroke.texture * 0.28), 0, 1)
    context.strokeStyle = rgbCss([clamp(color[0] + amount, 0, 255), clamp(color[1] + amount, 0, 255), clamp(color[2] + amount, 0, 255)])
    context.lineWidth = Math.max(0.75, strokeWidth * (0.04 + hashToUnit(stroke.seed, bristle, 0xa711) * 0.08))
    context.beginPath()
    for (let index = 0; index < count; index += 1) {
      const previous = Math.max(0, index - 1)
      const next = Math.min(count - 1, index + 1)
      const tx = points[next * 2] - points[previous * 2]
      const ty = points[next * 2 + 1] - points[previous * 2 + 1]
      const norm = Math.hypot(tx, ty) || 1
      const bx = points[index * 2] - (ty / norm) * offset
      const by = points[index * 2 + 1] + (tx / norm) * offset
      if (index === 0) {
        context.moveTo(bx, by)
      } else {
        context.lineTo(bx, by)
      }
    }
    context.stroke()
  }
}

// Regional measure of how much the color is shifting: per-pixel RGB gradient
// magnitude, box-blurred so it reflects busy neighborhoods rather than single
// edges, then normalized to 0..1.
function createColorActivityMap({ pixels, width, height }: CpuImage, baseSize: number) {
  const raw = new Float32Array(width * height)
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const gx = pixelColorDistance(pixels, width, height, x - 1, y, x + 1, y)
      const gy = pixelColorDistance(pixels, width, height, x, y - 1, x, y + 1)
      raw[x + y * width] = Math.sqrt(gx * gx + gy * gy)
    }
  }

  const blurred = boxBlurFloat(raw, width, height, Math.max(2, Math.round(baseSize * 0.9)))
  normalizePixelDetailMap(blurred)
  return blurred
}

function pixelColorDistance(pixels: Uint8ClampedArray, width: number, height: number, ax: number, ay: number, bx: number, by: number) {
  const aOffset = (clamp(ax, 0, width - 1) + clamp(ay, 0, height - 1) * width) * 4
  const bOffset = (clamp(bx, 0, width - 1) + clamp(by, 0, height - 1) * width) * 4
  const dr = pixels[aOffset] - pixels[bOffset]
  const dg = pixels[aOffset + 1] - pixels[bOffset + 1]
  const db = pixels[aOffset + 2] - pixels[bOffset + 2]
  return Math.sqrt(dr * dr + dg * dg + db * db)
}

function paintSmoothUnderpainting(context: Canvas2D, image: CpuImage, baseSize: number, abstraction: number, simplification: number) {
  const { width, height, pixels } = image
  const blockSize = Math.max(6, Math.round(baseSize * (2.4 + abstraction * 4.0)))
  const lowWidth = Math.max(1, Math.ceil(width / blockSize))
  const lowHeight = Math.max(1, Math.ceil(height / blockSize))
  const lowCanvas = new OffscreenCanvas(lowWidth, lowHeight)
  const lowContext = lowCanvas.getContext('2d')
  if (!lowContext) {
    paintBlockUnderpainting(context, image, baseSize, abstraction, simplification)
    return
  }

  const lowData = lowContext.createImageData(lowWidth, lowHeight)
  for (let cellY = 0; cellY < lowHeight; cellY += 1) {
    for (let cellX = 0; cellX < lowWidth; cellX += 1) {
      const color = quantizeRgb(
        averageRectColor(pixels, width, height, cellX * blockSize, cellY * blockSize, blockSize, blockSize),
        simplification * 0.85,
      )
      const offset = (cellX + cellY * lowWidth) * 4
      lowData.data[offset] = color[0]
      lowData.data[offset + 1] = color[1]
      lowData.data[offset + 2] = color[2]
      lowData.data[offset + 3] = 255
    }
  }
  lowContext.putImageData(lowData, 0, 0)

  // Bilinear upscale turns the averaged tiles into a smooth color field.
  context.imageSmoothingEnabled = true
  context.imageSmoothingQuality = 'high'
  context.drawImage(lowCanvas, 0, 0, lowWidth, lowHeight, 0, 0, width, height)
}

function paintBlockUnderpainting(context: Canvas2D, image: CpuImage, baseSize: number, abstraction: number, simplification: number) {
  const blockSize = Math.max(8, Math.round(baseSize * (3.2 + abstraction * 5.4)))
  const { width, height, pixels } = image
  for (let y = 0; y < height; y += blockSize) {
    for (let x = 0; x < width; x += blockSize) {
      const color = averageRectColor(pixels, width, height, x, y, blockSize, blockSize)
      context.fillStyle = rgbCss(quantizeRgb(color, simplification * 0.85))
      context.fillRect(x - 1, y - 1, blockSize + 2, blockSize + 2)
    }
  }
}

function createPixelDetailMap({ pixels, width, height }: CpuImage) {
  const detail = new Float32Array(width * height)
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      detail[x + y * width] = luminanceGradient(pixels, width, height, x, y).strength
    }
  }
  normalizePixelDetailMap(detail)
  return detail
}

// Smoothed structure tensor per cell, giving a coherent stroke direction and
// a confidence (anisotropy) value.
function createStrokeFlowField(pixels: Uint8ClampedArray, width: number, height: number, requestedCellSize: number): StrokeFlowField {
  const cellSize = Math.max(4, requestedCellSize)
  const fieldWidth = Math.max(1, Math.ceil(width / cellSize))
  const fieldHeight = Math.max(1, Math.ceil(height / cellSize))
  const tensorA = new Float64Array(fieldWidth * fieldHeight)
  const tensorB = new Float64Array(fieldWidth * fieldHeight)
  const tensorC = new Float64Array(fieldWidth * fieldHeight)
  const angles = new Float32Array(fieldWidth * fieldHeight)
  const confidence = new Float32Array(fieldWidth * fieldHeight)

  for (let y = 0; y < height; y += 2) {
    for (let x = 0; x < width; x += 2) {
      const gradient = luminanceGradient(pixels, width, height, x, y)
      const index = Math.min(fieldWidth - 1, Math.floor(x / cellSize)) + Math.min(fieldHeight - 1, Math.floor(y / cellSize)) * fieldWidth
      tensorA[index] += gradient.x * gradient.x
      tensorB[index] += gradient.x * gradient.y
      tensorC[index] += gradient.y * gradient.y
    }
  }

  for (let cellY = 0; cellY < fieldHeight; cellY += 1) {
    for (let cellX = 0; cellX < fieldWidth; cellX += 1) {
      let a = 0
      let b = 0
      let c = 0
      let weightTotal = 0
      for (let offsetY = -2; offsetY <= 2; offsetY += 1) {
        for (let offsetX = -2; offsetX <= 2; offsetX += 1) {
          const sampleX = cellX + offsetX
          const sampleY = cellY + offsetY
          if (sampleX < 0 || sampleY < 0 || sampleX >= fieldWidth || sampleY >= fieldHeight) {
            continue
          }
          const distance = offsetX * offsetX + offsetY * offsetY
          const weight = distance === 0 ? 1 : 1 / (1 + distance)
          const index = sampleX + sampleY * fieldWidth
          a += tensorA[index] * weight
          b += tensorB[index] * weight
          c += tensorC[index] * weight
          weightTotal += weight
        }
      }

      const index = cellX + cellY * fieldWidth
      a /= Math.max(0.0001, weightTotal)
      b /= Math.max(0.0001, weightTotal)
      c /= Math.max(0.0001, weightTotal)
      const trace = a + c
      angles[index] = 0.5 * Math.atan2(2 * b, a - c) + Math.PI * 0.5
      confidence[index] = trace > 0 ? clamp(Math.sqrt((a - c) * (a - c) + 4 * b * b) / trace, 0, 1) : 0
    }
  }

  return { cellSize, width: fieldWidth, height: fieldHeight, angles, confidence }
}

function sampleStrokeFlow(field: StrokeFlowField, x: number, y: number) {
  const fieldX = clamp(x / field.cellSize - 0.5, 0, field.width - 1)
  const fieldY = clamp(y / field.cellSize - 0.5, 0, field.height - 1)
  const x0 = Math.floor(fieldX)
  const y0 = Math.floor(fieldY)
  const x1 = Math.min(field.width - 1, x0 + 1)
  const y1 = Math.min(field.height - 1, y0 + 1)
  const tx = fieldX - x0
  const ty = fieldY - y0
  const top = blendFlowSamples(field, x0 + y0 * field.width, x1 + y0 * field.width, tx)
  const bottom = blendFlowSamples(field, x0 + y1 * field.width, x1 + y1 * field.width, tx)

  return {
    angle: blendAngles(top.angle, bottom.angle, ty),
    confidence: top.confidence * (1 - ty) + bottom.confidence * ty,
  }
}

function blendFlowSamples(field: StrokeFlowField, left: number, right: number, amount: number) {
  return {
    angle: blendAngles(field.angles[left], field.angles[right], amount),
    confidence: field.confidence[left] * (1 - amount) + field.confidence[right] * amount,
  }
}

function coherentFlowAngle(x: number, y: number, width: number, height: number) {
  const normalizedX = x / Math.max(1, width)
  const normalizedY = y / Math.max(1, height)
  const lowFrequency = hashToUnit(Math.floor(normalizedX * 5), Math.floor(normalizedY * 5), 0x6c31)
  const sweep = (normalizedX - 0.5) * 0.45 + (normalizedY - 0.5) * 0.28
  return (lowFrequency - 0.5) * Math.PI * 0.38 + sweep
}

function shiftColor(color: Rgb, amount: number): Rgb {
  return [clamp(Math.round(color[0] + amount), 0, 255), clamp(Math.round(color[1] + amount), 0, 255), clamp(Math.round(color[2] + amount), 0, 255)]
}
