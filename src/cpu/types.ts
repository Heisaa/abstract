import type { ParamValues } from '../filters/registry'
import type { Rgb } from './common'

export type CpuImage = {
  pixels: Uint8ClampedArray
  width: number
  height: number
  // Resolution relative to the preview image (0.5 draft, 1 preview, >1
  // high-res export) for algorithms with fixed pixel-sized neighborhoods.
  scale: number
}

export type CpuOutput = {
  pixels: Uint8ClampedArray
  palette?: Rgb[]
}

export type CpuExtra = {
  // Superpixel labels produced on the GPU (SLIC only).
  labels?: Int32Array
}

// Two-stage CPU filter. `analyze` holds the expensive work (segmentation,
// flow fields) and is cached on its parameters; `render` turns the analysis
// into pixels and reruns for render-only parameter changes.
export type CpuFilter<Analysis> = {
  analyze: (image: CpuImage, params: ParamValues, extra: CpuExtra) => Analysis
  render: (image: CpuImage, analysis: Analysis, params: ParamValues) => CpuOutput
}
