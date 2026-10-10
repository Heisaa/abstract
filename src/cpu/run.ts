import { analysisKey, type FilterId, type ParamValues } from '../filters/registry'
import { adaptiveBrushFilter } from './adaptiveBrush'
import { colorCheckFilter } from './colorCheck'
import { notanFilter } from './notan'
import { painterFilter } from './painter'
import { paletteFilter } from './palette'
import { slicFilter } from './slic'
import type { CpuExtra, CpuFilter, CpuImage, CpuOutput } from './types'

const CPU_FILTERS: Partial<Record<FilterId, CpuFilter<unknown>>> = {
  notan: notanFilter as CpuFilter<unknown>,
  colorCheck: colorCheckFilter as CpuFilter<unknown>,
  palette: paletteFilter as CpuFilter<unknown>,
  adaptiveBrush: adaptiveBrushFilter as CpuFilter<unknown>,
  painter: painterFilter as CpuFilter<unknown>,
  slic: slicFilter as CpuFilter<unknown>,
}

const MAX_CACHED_ANALYSES = 4

// Runs CPU filters with a small LRU cache of analysis results, so changing a
// render-only parameter (colors, outlines, stroke texture...) skips the
// expensive segmentation or field computation.
export class CpuFilterRunner {
  private readonly cache = new Map<string, unknown>()

  run(image: CpuImage, imageKey: string, filter: FilterId, params: ParamValues, extra: CpuExtra = {}): CpuOutput {
    const implementation = CPU_FILTERS[filter]
    if (!implementation) {
      throw new Error(`No CPU implementation for filter "${filter}".`)
    }

    const key = `${imageKey}|${analysisKey(filter, params)}`
    let analysis = this.cache.get(key)
    if (analysis === undefined) {
      analysis = implementation.analyze(image, params, extra)
      this.cache.set(key, analysis)
      while (this.cache.size > MAX_CACHED_ANALYSES) {
        this.cache.delete(this.cache.keys().next().value as string)
      }
    } else {
      // Refresh LRU position.
      this.cache.delete(key)
      this.cache.set(key, analysis)
    }

    return implementation.render(image, analysis, params)
  }

  clear(imageKeyPrefix?: string) {
    for (const key of Array.from(this.cache.keys())) {
      if (!imageKeyPrefix || key.startsWith(imageKeyPrefix)) {
        this.cache.delete(key)
      }
    }
  }
}
