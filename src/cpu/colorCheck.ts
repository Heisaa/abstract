import { segmentFelzenszwalb } from './felzenszwalb'
import { bool, num } from '../filters/registry'
import { boundaryMask } from './regions'
import { averageRegionColors, paintMask, paintRegions, perceptualRegionColors, regionStats, type RegionStats } from './regionColor'
import type { CpuFilter } from './types'

type ColorCheckAnalysis = {
  labels: Int32Array
  stats: RegionStats
}

export const colorCheckFilter: CpuFilter<ColorCheckAnalysis> = {
  analyze(image, params) {
    const { labels, labelCount } = segmentFelzenszwalb(image.pixels, image.width, image.height, {
      scale: num(params, 'scale'),
      minSize: num(params, 'minSize'),
      colorSpace: 'oklab',
    })
    return { labels, stats: regionStats(image.pixels, labels, labelCount) }
  },
  render(image, { labels, stats }, params) {
    // Perceptual (OKLab) averaging keeps regions clean; the raw sRGB average
    // is the simpler, muddier block-in offered as a toggle.
    const colors = bool(params, 'perceptual')
      ? perceptualRegionColors(stats, 0, 1 + num(params, 'saturation') * 0.6)
      : averageRegionColors(stats)
    const pixels = paintRegions(labels, colors)
    const separator = Math.round(num(params, 'separator'))
    if (separator > 0) {
      paintMask(pixels, boundaryMask(labels, image.width, image.height, separator), [128, 128, 128])
    }
    return { pixels }
  },
}
