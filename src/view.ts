// View settings apply on top of any filter in the final composite pass, so
// they never invalidate cached filter results.

export type CropPreset = { id: string; label: string; ratio: number }

// Ratios are long side / short side; orientation follows the image unless
// rotated. 0 means the full image.
export const CROP_PRESETS: CropPreset[] = [
  { id: 'image', label: 'Full image', ratio: 0 },
  { id: '1x1', label: '1:1', ratio: 1 },
  { id: '8x10', label: '8×10 · 16×20', ratio: 10 / 8 },
  { id: '9x12', label: '9×12 · 12×16', ratio: 12 / 9 },
  { id: '11x14', label: '11×14', ratio: 14 / 11 },
  { id: '5x7', label: '5×7', ratio: 7 / 5 },
  { id: '12x18', label: '12×18 · 2:3', ratio: 1.5 },
  { id: '12x24', label: '12×24 · 1:2', ratio: 2 },
]

export type ViewSettings = {
  grayscale: boolean
  mirror: boolean
  crop: string
  cropRotate: boolean
  cropX: number
  cropY: number
  gridDivisions: number
  gridDiagonals: boolean
  overlay: boolean
  overlayStrength: number
  overlayDetail: number
  overlaySensitivity: number
  // Include grid and mirroring in exported files.
  exportViewAids: boolean
}

export const DEFAULT_VIEW: ViewSettings = {
  grayscale: false,
  mirror: false,
  crop: 'image',
  cropRotate: false,
  cropX: 0,
  cropY: 0,
  gridDivisions: 0,
  gridDiagonals: false,
  overlay: false,
  overlayStrength: 0.7,
  overlayDetail: 1.2,
  overlaySensitivity: 0.55,
  exportViewAids: false,
}

export function sanitizeView(input: unknown): ViewSettings {
  const source = input && typeof input === 'object' ? (input as Record<string, unknown>) : {}
  const view = { ...DEFAULT_VIEW }
  for (const key of Object.keys(DEFAULT_VIEW) as Array<keyof ViewSettings>) {
    const raw = source[key]
    if (typeof raw === typeof DEFAULT_VIEW[key] && (typeof raw !== 'number' || Number.isFinite(raw))) {
      ;(view as Record<string, unknown>)[key] = raw
    }
  }
  if (!CROP_PRESETS.some((preset) => preset.id === view.crop)) {
    view.crop = DEFAULT_VIEW.crop
  }
  view.cropX = Math.max(-1, Math.min(1, view.cropX))
  view.cropY = Math.max(-1, Math.min(1, view.cropY))
  view.gridDivisions = Math.max(0, Math.min(12, Math.round(view.gridDivisions)))
  return view
}

export type CropRect = {
  // Offset and size in normalized image coordinates.
  offsetX: number
  offsetY: number
  scaleX: number
  scaleY: number
  // Output size in pixels.
  width: number
  height: number
}

export function computeCrop(view: Pick<ViewSettings, 'crop' | 'cropRotate' | 'cropX' | 'cropY'>, width: number, height: number): CropRect {
  const preset = CROP_PRESETS.find((candidate) => candidate.id === view.crop)
  if (!preset || preset.ratio === 0) {
    return { offsetX: 0, offsetY: 0, scaleX: 1, scaleY: 1, width, height }
  }

  const landscape = width >= height
  const orientLandscape = view.cropRotate ? !landscape : landscape
  const targetRatio = orientLandscape ? preset.ratio : 1 / preset.ratio
  let cropWidth = width
  let cropHeight = height
  if (width / height > targetRatio) {
    cropWidth = height * targetRatio
  } else {
    cropHeight = width / targetRatio
  }

  const outWidth = Math.max(1, Math.round(cropWidth))
  const outHeight = Math.max(1, Math.round(cropHeight))
  const freeX = (width - outWidth) / 2
  const freeY = (height - outHeight) / 2
  return {
    offsetX: (freeX * (1 + view.cropX)) / width,
    offsetY: (freeY * (1 + view.cropY)) / height,
    scaleX: outWidth / width,
    scaleY: outHeight / height,
    width: outWidth,
    height: outHeight,
  }
}

// Maps a point on the displayed output (0..1) back to image coordinates
// (0..1), matching the composite shader's crop and mirror handling.
export function outputToImageUv(view: ViewSettings, crop: CropRect, u: number, v: number) {
  const x = view.mirror ? 1 - u : u
  return { u: crop.offsetX + x * crop.scaleX, v: crop.offsetY + v * crop.scaleY }
}
