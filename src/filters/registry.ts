// Single source of truth for every filter: its id, sidebar group, which engine
// renders it, and its parameters. The sidebar controls, cache keys, preset
// validation, and resolution scaling (for draft previews and high-res export)
// are all derived from these definitions, so adding a filter means adding one
// entry here plus its implementation.

export type FilterId =
  | 'original'
  | 'value'
  | 'notan'
  | 'edges'
  | 'chroma'
  | 'temperature'
  | 'kuwahara'
  | 'slic'
  | 'colorCheck'
  | 'palette'
  | 'lostFound'
  | 'flowBrush'
  | 'adaptiveBrush'
  | 'painter'

// gpu: WGSL passes. cpu: worker computation uploaded as a texture.
// slic: GPU clustering followed by worker clean-up and coloring.
export type FilterEngine = 'none' | 'gpu' | 'cpu' | 'slic'

export type FilterGroup = 'Analyze' | 'Simplify' | 'Render'

export type ParamValue = number | boolean | string
export type ParamValues = Record<string, ParamValue>

// How a parameter scales when the image is rendered at another resolution:
// px values scale linearly, area values quadratically.
export type ParamUnit = 'px' | 'area'

type ParamBase = {
  key: string
  label: string
  hint?: string
  unit?: ParamUnit
  // Only affects the final coloring/drawing stage, so cached analysis
  // (segmentation, flow fields...) can be reused when it changes.
  renderOnly?: boolean
  visible?: (values: ParamValues) => boolean
}

export type RangeParam = ParamBase & { type: 'range'; min: number; max: number; step: number; default: number }
export type ToggleParam = ParamBase & { type: 'toggle'; default: boolean }
export type ChoiceParam = ParamBase & {
  type: 'choice'
  options: Array<{ value: string; label: string }>
  default: string
}
export type ParamDef = RangeParam | ToggleParam | ChoiceParam

// A built-in starting point: named overrides on top of the defaults.
export type FilterStyle = {
  name: string
  params: ParamValues
}

export type FilterDef = {
  id: FilterId
  label: string
  group: FilterGroup
  engine: FilterEngine
  description: string
  params: ParamDef[]
  styles?: FilterStyle[]
  // Lets one parameter adjust others (e.g. reset thresholds when the band
  // count changes). Returns the patch to apply.
  onParamChange?: (key: string, values: ParamValues) => ParamValues | undefined
}

const contrastParam = (defaultValue = 1): RangeParam => ({
  key: 'contrast',
  label: 'Contrast',
  type: 'range',
  min: 0.5,
  max: 2,
  step: 0.05,
  default: defaultValue,
})

function evenThresholds(bands: number) {
  const patch: ParamValues = {}
  for (let index = 1; index <= 4; index += 1) {
    patch[`t${index}`] = Number(Math.min(0.95, index / bands).toFixed(2))
  }
  return patch
}

const thresholdParam = (index: number, defaultValue: number): RangeParam => ({
  key: `t${index}`,
  label: `Threshold ${index}`,
  hint: 'OKLab lightness boundary between value bands',
  type: 'range',
  min: 0.02,
  max: 0.98,
  step: 0.01,
  default: defaultValue,
  visible: (values) => values.auto !== true && Number(values.bands) > index,
})

export const FILTERS: FilterDef[] = [
  {
    id: 'value',
    label: 'Value',
    group: 'Analyze',
    engine: 'gpu',
    description: 'Posterizes perceptual lightness into equal value steps. Squint first to see masses, not texture.',
    params: [
      { key: 'levels', label: 'Levels', type: 'range', min: 2, max: 10, step: 1, default: 5 },
      { key: 'squint', label: 'Squint', hint: 'Blur before quantizing', type: 'range', min: 0, max: 12, step: 0.5, default: 3, unit: 'px' },
      contrastParam(1),
      {
        key: 'output',
        label: 'Output',
        type: 'choice',
        options: [
          { value: 'gray', label: 'Gray' },
          { value: 'color', label: 'Keep color' },
        ],
        default: 'gray',
      },
    ],
  },
  {
    id: 'notan',
    label: 'Value Masses',
    group: 'Analyze',
    engine: 'cpu',
    description: 'Groups the image into 2–5 connected value masses (notan), absorbing small specks into their surroundings.',
    params: [
      {
        key: 'bands',
        label: 'Values',
        type: 'choice',
        options: [
          { value: '2', label: '2' },
          { value: '3', label: '3' },
          { value: '4', label: '4' },
          { value: '5', label: '5' },
        ],
        default: '3',
      },
      {
        key: 'auto',
        label: 'Auto thresholds',
        hint: "Split where this image's values naturally separate (multi-level Otsu)",
        type: 'toggle',
        default: true,
      },
      thresholdParam(1, 0.33),
      thresholdParam(2, 0.67),
      thresholdParam(3, 0.75),
      thresholdParam(4, 0.8),
      { key: 'squint', label: 'Squint', type: 'range', min: 0, max: 24, step: 0.5, default: 4, unit: 'px' },
      { key: 'minArea', label: 'Min Mass Size', type: 'range', min: 0, max: 6000, step: 20, default: 600, unit: 'area' },
      {
        key: 'output',
        label: 'Output',
        type: 'choice',
        options: [
          { value: 'gray', label: 'Gray' },
          { value: 'snapped', label: 'Color at value' },
          { value: 'mean', label: 'Mean color' },
        ],
        default: 'gray',
        renderOnly: true,
      },
      { key: 'outline', label: 'Outline masses', type: 'toggle', default: false, renderOnly: true },
    ],
    onParamChange: (key, values) => (key === 'bands' ? evenThresholds(Number(values.bands)) : undefined),
  },
  {
    id: 'edges',
    label: 'Edges',
    group: 'Analyze',
    engine: 'gpu',
    description: 'Flow-guided difference-of-Gaussians lines that follow form instead of noise.',
    params: [
      { key: 'detail', label: 'Line Scale', type: 'range', min: 0.5, max: 4, step: 0.1, default: 1.2, unit: 'px' },
      { key: 'sensitivity', label: 'Sensitivity', type: 'range', min: 0, max: 1, step: 0.01, default: 0.6 },
      { key: 'smoothing', label: 'Flow Smoothing', type: 'range', min: 0.5, max: 8, step: 0.1, default: 3, unit: 'px' },
      { key: 'sharpness', label: 'Line Sharpness', type: 'range', min: 0, max: 1, step: 0.01, default: 0.5 },
      { key: 'strength', label: 'Strength', type: 'range', min: 0.1, max: 1, step: 0.01, default: 0.85 },
      contrastParam(1),
      { key: 'linesOnly', label: 'Lines only', type: 'toggle', default: false },
    ],
  },
  {
    id: 'chroma',
    label: 'Chroma',
    group: 'Analyze',
    engine: 'gpu',
    description: 'Shows color intensity alone, to find where saturated accents sit among grays.',
    params: [
      { key: 'gain', label: 'Gain', type: 'range', min: 0.5, max: 4, step: 0.05, default: 1.5 },
      {
        key: 'display',
        label: 'Display',
        type: 'choice',
        options: [
          { value: 'heat', label: 'Heat' },
          { value: 'gray', label: 'Gray' },
        ],
        default: 'heat',
      },
    ],
  },
  {
    id: 'temperature',
    label: 'Warm / Cool',
    group: 'Analyze',
    engine: 'gpu',
    description: 'Classifies each area as warm or cool relative to the hue axis orange–blue.',
    params: [
      { key: 'gain', label: 'Gain', type: 'range', min: 0.5, max: 6, step: 0.05, default: 2.5 },
      { key: 'neutral', label: 'Neutral Zone', type: 'range', min: 0, max: 0.1, step: 0.002, default: 0.02 },
      { key: 'keepValue', label: 'Keep value', type: 'toggle', default: true },
    ],
  },
  {
    id: 'kuwahara',
    label: 'Kuwahara',
    group: 'Simplify',
    engine: 'gpu',
    description: 'Anisotropic Kuwahara: flattens passages while bending along edges like brushwork.',
    params: [
      { key: 'radius', label: 'Radius', type: 'range', min: 2, max: 12, step: 1, default: 6, unit: 'px' },
      { key: 'sharpness', label: 'Sharpness', type: 'range', min: 1, max: 18, step: 0.5, default: 8 },
      { key: 'elongation', label: 'Elongation', type: 'range', min: 0, max: 1, step: 0.01, default: 0.5 },
      { key: 'smoothing', label: 'Flow Smoothing', type: 'range', min: 0.5, max: 6, step: 0.1, default: 2, unit: 'px' },
      { key: 'passes', label: 'Passes', type: 'range', min: 1, max: 3, step: 1, default: 1 },
    ],
  },
  {
    id: 'slic',
    label: 'SLIC',
    group: 'Simplify',
    engine: 'slic',
    description: 'Superpixels clustered in OKLab, cleaned so every region is one connected brush patch.',
    params: [
      { key: 'size', label: 'Region Size', type: 'range', min: 8, max: 96, step: 1, default: 36, unit: 'px' },
      { key: 'compactness', label: 'Compactness', type: 'range', min: 1, max: 40, step: 1, default: 12 },
      { key: 'cleanup', label: 'Merge fragments', type: 'toggle', default: true },
      {
        key: 'merge',
        label: 'Merge Similar',
        hint: 'Join neighboring patches of nearly the same color, so quiet areas become one large patch',
        type: 'range',
        min: 0,
        max: 1,
        step: 0.01,
        default: 0.3,
      },
      { key: 'saturation', label: 'Saturation Preserve', type: 'range', min: 0, max: 1, step: 0.01, default: 0.55, renderOnly: true },
      { key: 'outline', label: 'Outline regions', type: 'toggle', default: false, renderOnly: true },
    ],
  },
  {
    id: 'colorCheck',
    label: 'Color Check',
    group: 'Simplify',
    engine: 'cpu',
    description: 'Graph segmentation into perceptual color fields for checking color relationships.',
    params: [
      { key: 'scale', label: 'Field Size', type: 'range', min: 50, max: 1500, step: 10, default: 700, unit: 'area' },
      { key: 'minSize', label: 'Min Field Size', type: 'range', min: 16, max: 600, step: 4, default: 120, unit: 'area' },
      { key: 'saturation', label: 'Saturation', type: 'range', min: 0, max: 1, step: 0.01, default: 0.4, renderOnly: true },
      { key: 'separator', label: 'Separator', type: 'range', min: 0, max: 3, step: 1, default: 0, unit: 'px', renderOnly: true },
      { key: 'perceptual', label: 'Perceptual color', type: 'toggle', default: true, renderOnly: true },
    ],
  },
  {
    id: 'palette',
    label: 'Palette',
    group: 'Simplify',
    engine: 'cpu',
    description: 'Reduces the image to a limited palette (k-means in OKLab), or tests it against the Zorn palette gamut.',
    params: [
      {
        key: 'mode',
        label: 'Palette',
        type: 'choice',
        options: [
          { value: 'kmeans', label: 'Extract N colors' },
          { value: 'zorn', label: 'Zorn gamut' },
        ],
        default: 'kmeans',
      },
      {
        key: 'colors',
        label: 'Colors',
        type: 'range',
        min: 2,
        max: 16,
        step: 1,
        default: 8,
        visible: (values) => values.mode === 'kmeans',
      },
      { key: 'squint', label: 'Squint', type: 'range', min: 0, max: 12, step: 0.5, default: 1.5, unit: 'px' },
      { key: 'minArea', label: 'Min Patch Size', type: 'range', min: 0, max: 2000, step: 10, default: 80, unit: 'area' },
      {
        key: 'zornView',
        label: 'View',
        type: 'choice',
        options: [
          { value: 'mapped', label: 'Map to gamut' },
          { value: 'check', label: 'Flag outside' },
        ],
        default: 'mapped',
        renderOnly: true,
        visible: (values) => values.mode === 'zorn',
      },
      {
        key: 'tolerance',
        label: 'Gamut Tolerance',
        type: 'range',
        min: 0.01,
        max: 0.2,
        step: 0.005,
        default: 0.05,
        renderOnly: true,
        visible: (values) => values.mode === 'zorn' && values.zornView === 'check',
      },
    ],
  },
  {
    id: 'lostFound',
    label: 'Lost & Found',
    group: 'Render',
    engine: 'gpu',
    description:
      'Broad simplified passages: edges with strong value contrast stay found, weak ones are lost, and only the strongest keep crisp accents.',
    params: [
      { key: 'radius', label: 'Simplify', type: 'range', min: 2, max: 12, step: 1, default: 6, unit: 'px' },
      { key: 'lost', label: 'Lost Softness', type: 'range', min: 0, max: 16, step: 0.5, default: 5, unit: 'px' },
      { key: 'scale', label: 'Edge Scale', type: 'range', min: 0.5, max: 8, step: 0.1, default: 2.5, unit: 'px' },
      { key: 'found', label: 'Found Threshold', type: 'range', min: 0, max: 1, step: 0.01, default: 0.3 },
      { key: 'accentThreshold', label: 'Accent Threshold', type: 'range', min: 0, max: 1, step: 0.01, default: 0.65 },
      { key: 'accents', label: 'Accent Strength', type: 'range', min: 0, max: 1, step: 0.01, default: 0.4 },
      { key: 'showMask', label: 'Show edge map', type: 'toggle', default: false },
    ],
  },
  {
    id: 'flowBrush',
    label: 'Flow Brush',
    group: 'Render',
    engine: 'gpu',
    description: 'Strokes smeared along the local edge direction.',
    params: [
      { key: 'length', label: 'Stroke Length', type: 'range', min: 4, max: 72, step: 1, default: 44, unit: 'px' },
      { key: 'width', label: 'Stroke Width', type: 'range', min: 1, max: 18, step: 1, default: 7, unit: 'px' },
      { key: 'smoothing', label: 'Flow Smoothing', type: 'range', min: 1, max: 6, step: 0.5, default: 4, unit: 'px' },
      { key: 'colorMerge', label: 'Color Merge', type: 'range', min: 0, max: 1, step: 0.01, default: 0.62 },
      { key: 'edgePreserve', label: 'Edge Preserve', type: 'range', min: 0, max: 1, step: 0.01, default: 0.55 },
      { key: 'edgeThreshold', label: 'Edge Threshold', type: 'range', min: 0.05, max: 0.8, step: 0.01, default: 0.28 },
      { key: 'strength', label: 'Stroke Strength', type: 'range', min: 0, max: 1, step: 0.01, default: 0.95 },
      contrastParam(1.15),
    ],
    styles: [
      { name: 'Fine strokes', params: { length: 22, width: 4, colorMerge: 0.35, edgePreserve: 0.8 } },
      { name: 'Bold strokes', params: { length: 64, width: 12, colorMerge: 0.78, edgePreserve: 0.7 } },
    ],
  },
  {
    id: 'adaptiveBrush',
    label: 'Adaptive Brush',
    group: 'Render',
    engine: 'cpu',
    description: 'Large brush in quiet areas, smaller brushes only where detail demands it.',
    params: [
      { key: 'large', label: 'Large Brush', type: 'range', min: 16, max: 128, step: 1, default: 56, unit: 'px' },
      { key: 'medium', label: 'Medium Brush', type: 'range', min: 8, max: 80, step: 1, default: 28, unit: 'px' },
      { key: 'small', label: 'Small Brush', type: 'range', min: 4, max: 48, step: 1, default: 14, unit: 'px' },
      { key: 'threshold', label: 'Detail Threshold', type: 'range', min: 0, max: 1, step: 0.01, default: 0.45, renderOnly: true },
      { key: 'accent', label: 'Accent Strength', type: 'range', min: 0, max: 1, step: 0.01, default: 0.45, renderOnly: true },
    ],
  },
  {
    id: 'painter',
    label: 'Painter',
    group: 'Render',
    engine: 'cpu',
    description: 'Underpainting plus layered tapered strokes, finer where color or edges are busy.',
    params: [
      {
        key: 'underpainting',
        label: 'Base',
        type: 'choice',
        options: [
          { value: 'smooth', label: 'Smooth base' },
          { value: 'blocks', label: 'Block base' },
        ],
        default: 'smooth',
        renderOnly: true,
      },
      {
        key: 'detailSource',
        label: 'Detail from',
        type: 'choice',
        options: [
          { value: 'color', label: 'Color' },
          { value: 'edges', label: 'Edges' },
        ],
        default: 'color',
      },
      { key: 'abstraction', label: 'Abstraction', type: 'range', min: 0, max: 1, step: 0.01, default: 0.64, renderOnly: true },
      { key: 'size', label: 'Stroke Size', type: 'range', min: 4, max: 32, step: 1, default: 14, unit: 'px' },
      { key: 'length', label: 'Stroke Length', type: 'range', min: 1, max: 5, step: 0.1, default: 3, renderOnly: true },
      { key: 'detail', label: 'Detail', type: 'range', min: 0, max: 1, step: 0.01, default: 0.45, renderOnly: true },
      { key: 'edgeFidelity', label: 'Edge Fidelity', type: 'range', min: 0, max: 1, step: 0.01, default: 0.66, renderOnly: true },
      { key: 'colorSimplify', label: 'Color Simplify', type: 'range', min: 0, max: 1, step: 0.01, default: 0.5, renderOnly: true },
      { key: 'texture', label: 'Texture', type: 'range', min: 0, max: 1, step: 0.01, default: 0.55, renderOnly: true },
      { key: 'layers', label: 'Layers', type: 'range', min: 1, max: 4, step: 1, default: 3, renderOnly: true },
    ],
    styles: [
      // Reads as realism from a distance, loose strokes up close.
      { name: 'Sargent: tight', params: { abstraction: 0.3, detail: 0.75, size: 10, edgeFidelity: 0.8 } },
      // Big flat blocks that still keep the main shapes readable.
      {
        name: 'Bold blocks',
        params: { abstraction: 0.85, detail: 0.45, size: 24, layers: 3, length: 2, edgeFidelity: 0.85, colorSimplify: 0.7, texture: 0.3 },
      },
    ],
  },
]

export const FILTER_GROUPS: FilterGroup[] = ['Analyze', 'Simplify', 'Render']

const FILTER_MAP = new Map<FilterId, FilterDef>(FILTERS.map((filter) => [filter.id, filter]))

const ORIGINAL_FILTER: FilterDef = {
  id: 'original',
  label: 'Original',
  group: 'Analyze',
  engine: 'none',
  description: 'The unfiltered image.',
  params: [],
}

export function getFilter(id: FilterId): FilterDef {
  return id === 'original' ? ORIGINAL_FILTER : (FILTER_MAP.get(id) ?? ORIGINAL_FILTER)
}

export function isFilterId(value: unknown): value is FilterId {
  return typeof value === 'string' && (value === 'original' || FILTER_MAP.has(value as FilterId))
}

export function styleParams(id: FilterId, style: FilterStyle): ParamValues {
  return sanitizeParams(id, { ...defaultParams(id), ...style.params })
}

export function defaultParams(id: FilterId): ParamValues {
  const values: ParamValues = {}
  for (const param of getFilter(id).params) {
    values[param.key] = param.default
  }
  return values
}

export type AllParams = Record<FilterId, ParamValues>

export function defaultAllParams(): AllParams {
  const all = { original: {} } as AllParams
  for (const filter of FILTERS) {
    all[filter.id] = defaultParams(filter.id)
  }
  return all
}

// Coerces untrusted values (localStorage, presets) into valid parameters,
// falling back to defaults for anything missing or malformed.
export function sanitizeParams(id: FilterId, input: unknown): ParamValues {
  const source = input && typeof input === 'object' ? (input as Record<string, unknown>) : {}
  const values: ParamValues = {}
  for (const param of getFilter(id).params) {
    const raw = source[param.key]
    if (param.type === 'range') {
      const numeric = typeof raw === 'number' && Number.isFinite(raw) ? raw : param.default
      values[param.key] = Math.min(param.max, Math.max(param.min, numeric))
    } else if (param.type === 'toggle') {
      values[param.key] = typeof raw === 'boolean' ? raw : param.default
    } else {
      values[param.key] = param.options.some((option) => option.value === raw) ? (raw as string) : param.default
    }
  }
  return values
}

// Returns parameters adjusted for rendering at `factor` times the preview
// resolution (0.5 for drafts, >1 for high-res export).
export function scaleParams(id: FilterId, values: ParamValues, factor: number): ParamValues {
  if (factor === 1) {
    return values
  }

  const scaled: ParamValues = { ...values }
  for (const param of getFilter(id).params) {
    const value = values[param.key]
    if (typeof value !== 'number') {
      continue
    }
    if (param.unit === 'px') {
      scaled[param.key] = value * factor
    } else if (param.unit === 'area') {
      scaled[param.key] = value * factor * factor
    }
  }
  return scaled
}

function stableValues(id: FilterId, values: ParamValues, include: (param: ParamDef) => boolean) {
  return getFilter(id)
    .params.filter(include)
    .map((param) => `${param.key}=${String(values[param.key])}`)
    .join('&')
}

// Key covering parameters that invalidate cached analysis.
export function analysisKey(id: FilterId, values: ParamValues) {
  return `${id}?${stableValues(id, values, (param) => !param.renderOnly)}`
}

export function fullKey(id: FilterId, values: ParamValues) {
  return `${id}?${stableValues(id, values, () => true)}`
}

export function num(values: ParamValues, key: string) {
  const value = values[key]
  return typeof value === 'number' ? value : Number(value) || 0
}

export function bool(values: ParamValues, key: string) {
  return values[key] === true
}

export function str(values: ParamValues, key: string) {
  return String(values[key] ?? '')
}
