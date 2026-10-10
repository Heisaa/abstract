import { defaultAllParams, FILTERS, isFilterId, sanitizeParams, type AllParams, type FilterId, type ParamValues } from './filters/registry'
import { DEFAULT_VIEW, sanitizeView, type ViewSettings } from './view'

// Settings persistence and named presets in localStorage. Everything read
// back is validated against the filter registry, so stale or hand-edited data
// can never produce invalid parameters.

const SETTINGS_KEY = 'painting-reference:settings:v1'
const PRESETS_KEY = 'painting-reference:presets:v1'

export type ExportFormat = 'png' | 'jpeg'
export type ExportSize = 'preview' | 'high'

export type StoredSettings = {
  filter: FilterId
  params: AllParams
  view: ViewSettings
  exportFormat: ExportFormat
  exportSize: ExportSize
}

export type Preset = {
  id: string
  name: string
  filter: FilterId
  params: ParamValues
}

export const DEFAULT_SETTINGS: StoredSettings = {
  filter: 'kuwahara',
  params: defaultAllParams(),
  view: DEFAULT_VIEW,
  exportFormat: 'png',
  exportSize: 'high',
}

function readJson(key: string): unknown {
  try {
    const raw = localStorage.getItem(key)
    return raw ? JSON.parse(raw) : undefined
  } catch {
    return undefined
  }
}

function writeJson(key: string, value: unknown) {
  try {
    localStorage.setItem(key, JSON.stringify(value))
  } catch {
    // Storage full or unavailable (private mode): settings just won't persist.
  }
}

export function loadSettings(): StoredSettings {
  const stored = readJson(SETTINGS_KEY)
  if (!stored || typeof stored !== 'object') {
    return DEFAULT_SETTINGS
  }

  const source = stored as Record<string, unknown>
  const storedParams = (source.params && typeof source.params === 'object' ? source.params : {}) as Record<string, unknown>
  const params = defaultAllParams()
  for (const filter of FILTERS) {
    params[filter.id] = sanitizeParams(filter.id, storedParams[filter.id])
  }

  return {
    filter: isFilterId(source.filter) && source.filter !== 'original' ? source.filter : DEFAULT_SETTINGS.filter,
    params,
    view: sanitizeView(source.view),
    exportFormat: source.exportFormat === 'jpeg' ? 'jpeg' : 'png',
    exportSize: source.exportSize === 'preview' ? 'preview' : 'high',
  }
}

export function saveSettings(settings: StoredSettings) {
  writeJson(SETTINGS_KEY, settings)
}

export function loadPresets(): Preset[] {
  const stored = readJson(PRESETS_KEY)
  if (!Array.isArray(stored)) {
    return []
  }

  return stored.flatMap((entry): Preset[] => {
    if (!entry || typeof entry !== 'object') {
      return []
    }
    const { id, name, filter, params } = entry as Record<string, unknown>
    if (typeof id !== 'string' || typeof name !== 'string' || !isFilterId(filter) || filter === 'original') {
      return []
    }
    return [{ id, name: name.slice(0, 60), filter, params: sanitizeParams(filter, params) }]
  })
}

export function savePresets(presets: Preset[]) {
  writeJson(PRESETS_KEY, presets)
}

export function createPresetId() {
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`
}
