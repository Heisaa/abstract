import type { FilterId, ParamValues } from '../filters/registry'
import type { Rgb } from './common'

export type ImageSlot = 'preview' | 'export'

export type WorkerRequest =
  | { type: 'setImage'; slot: ImageSlot; version: number; width: number; height: number; pixels: ArrayBuffer }
  | { type: 'dropImage'; slot: ImageSlot }
  | {
      type: 'run'
      jobId: number
      slot: ImageSlot
      // Use the half-resolution copy of the preview image.
      draft: boolean
      // Resolution relative to the preview image; params are already scaled.
      scale: number
      filter: FilterId
      params: ParamValues
      labels?: ArrayBuffer
    }

export type WorkerResponse =
  | { type: 'result'; jobId: number; width: number; height: number; pixels: ArrayBuffer; palette?: Rgb[] }
  | { type: 'error'; jobId: number; message: string }
