import { useState } from 'react'
import { hueName, oklabChroma, oklabHue, rgbToHex, srgbToOklab } from '../cpu/color'
import type { Rgb } from '../cpu/common'
import { styleParams, type FilterId, type FilterStyle, type ParamValues } from '../filters/registry'
import type { Preset } from '../storage'
import { CROP_PRESETS, type ViewSettings } from '../view'
import { Section, Slider, Toggle } from './controls'

type ViewPanelProps = {
  view: ViewSettings
  onChange: (patch: Partial<ViewSettings>) => void
  split: { enabled: boolean; position: number }
  onSplitToggle: (enabled: boolean) => void
}

export function ViewPanel({ view, onChange, split, onSplitToggle }: ViewPanelProps) {
  const cropped = view.crop !== 'image'
  return (
    <Section title="View">
      <div className="toggle-grid">
        <Toggle label="Grayscale (G)" checked={view.grayscale} onChange={(grayscale) => onChange({ grayscale })} />
        <Toggle label="Mirror (M)" checked={view.mirror} onChange={(mirror) => onChange({ mirror })} />
        <Toggle label="Split compare (S)" checked={split.enabled} onChange={onSplitToggle} />
        <Toggle label="Edge overlay (E)" checked={view.overlay} onChange={(overlay) => onChange({ overlay })} />
      </div>

      {view.overlay ? (
        <div className="subfields">
          <Slider label="Overlay Strength" min={0.1} max={1} step={0.01} value={view.overlayStrength} onChange={(overlayStrength) => onChange({ overlayStrength })} />
          <Slider label="Overlay Line Scale" min={0.5} max={4} step={0.1} value={view.overlayDetail} onChange={(overlayDetail) => onChange({ overlayDetail })} />
          <Slider
            label="Overlay Sensitivity"
            min={0}
            max={1}
            step={0.01}
            value={view.overlaySensitivity}
            onChange={(overlaySensitivity) => onChange({ overlaySensitivity })}
          />
        </div>
      ) : null}

      <label className="field">
        <span className="field-label">Crop to panel</span>
        <select value={view.crop} onChange={(event) => onChange({ crop: event.currentTarget.value, cropX: 0, cropY: 0 })}>
          {CROP_PRESETS.map((preset) => (
            <option key={preset.id} value={preset.id}>
              {preset.label}
            </option>
          ))}
        </select>
      </label>
      {cropped ? (
        <div className="subfields">
          <Toggle label="Rotate crop" checked={view.cropRotate} onChange={(cropRotate) => onChange({ cropRotate, cropX: 0, cropY: 0 })} />
          <Slider label="Crop Horizontal" min={-1} max={1} step={0.01} value={view.cropX} onChange={(cropX) => onChange({ cropX })} />
          <Slider label="Crop Vertical" min={-1} max={1} step={0.01} value={view.cropY} onChange={(cropY) => onChange({ cropY })} />
        </div>
      ) : null}

      <Slider
        label="Grid"
        min={0}
        max={12}
        step={1}
        value={view.gridDivisions}
        format={(value) => (value === 0 ? 'off' : `${value} × ${value}`)}
        onChange={(gridDivisions) => onChange({ gridDivisions })}
      />
      <Toggle label="Diagonals" checked={view.gridDiagonals} onChange={(gridDiagonals) => onChange({ gridDiagonals })} />
      <Toggle
        label="Include grid and mirror in export"
        checked={view.exportViewAids}
        onChange={(exportViewAids) => onChange({ exportViewAids })}
      />
    </Section>
  )
}

type PresetPanelProps = {
  filter: FilterId
  filterLabel: string
  styles?: FilterStyle[]
  presets: Preset[]
  onApply: (params: ParamValues) => void
  onSave: (name: string) => void
  onDelete: (id: string) => void
  onReset: () => void
}

export function PresetPanel({ filter, filterLabel, styles = [], presets, onApply, onSave, onDelete, onReset }: PresetPanelProps) {
  const [name, setName] = useState('')
  const filterPresets = presets.filter((preset) => preset.filter === filter)
  const save = () => {
    const trimmed = name.trim()
    if (trimmed) {
      onSave(trimmed)
      setName('')
    }
  }

  return (
    <Section
      title={`${filterLabel} presets`}
      actions={
        <button type="button" className="link-button" onClick={onReset}>
          Reset to defaults
        </button>
      }
    >
      {styles.length > 0 ? (
        <ul className="preset-list">
          {styles.map((style) => (
            <li key={style.name}>
              <button type="button" className="preset-apply" onClick={() => onApply(styleParams(filter, style))}>
                {style.name}
              </button>
            </li>
          ))}
        </ul>
      ) : null}
      {filterPresets.length > 0 ? (
        <ul className="preset-list">
          {filterPresets.map((preset) => (
            <li key={preset.id}>
              <button type="button" className="preset-apply" onClick={() => onApply(preset.params)}>
                {preset.name}
              </button>
              <button type="button" className="preset-delete" aria-label={`Delete preset ${preset.name}`} onClick={() => onDelete(preset.id)}>
                ×
              </button>
            </li>
          ))}
        </ul>
      ) : (
        <p className="muted">No saved presets for this filter.</p>
      )}
      <form
        className="preset-form"
        onSubmit={(event) => {
          event.preventDefault()
          save()
        }}
      >
        <input value={name} maxLength={60} placeholder="Preset name" aria-label="Preset name" onChange={(event) => setName(event.currentTarget.value)} />
        <button type="submit" disabled={!name.trim()}>
          Save
        </button>
      </form>
    </Section>
  )
}

export function PaletteSwatches({ palette }: { palette: Rgb[] }) {
  return (
    <div className="field">
      <span className="field-label">Palette</span>
      <div className="swatches">
        {palette.map((color, index) => (
          <span key={index} className="swatch" style={{ background: rgbToHex(color) }} title={describeColor(color)} />
        ))}
      </div>
    </div>
  )
}

function describeColor(color: Rgb) {
  const [l, a, b] = srgbToOklab(...color)
  const chroma = oklabChroma(a, b)
  return `${rgbToHex(color)} · value ${(l * 10).toFixed(1)}/10 · chroma ${chroma.toFixed(3)} · ${hueName(oklabHue(a, b), chroma)}`
}

export type PickedColor = { original: Rgb; filtered?: Rgb }

function ColorRow({ label, color }: { label: string; color: Rgb }) {
  const [l, a, b] = srgbToOklab(...color)
  const chroma = oklabChroma(a, b)
  const hue = oklabHue(a, b)
  return (
    <div className="color-row">
      <span className="swatch large" style={{ background: rgbToHex(color) }} />
      <dl>
        <dt>{label}</dt>
        <dd>{rgbToHex(color)}</dd>
        <dt>Value</dt>
        <dd>{(l * 10).toFixed(1)} / 10</dd>
        <dt>Chroma</dt>
        <dd>{chroma.toFixed(3)}</dd>
        <dt>Hue</dt>
        <dd>
          {hueName(hue, chroma)} {chroma >= 0.02 ? `(${Math.round(hue)}°)` : ''}
        </dd>
      </dl>
    </div>
  )
}

export function ColorReadout({ picked, eyedropper, onToggle }: { picked?: PickedColor; eyedropper: boolean; onToggle: () => void }) {
  return (
    <Section
      title="Eyedropper"
      actions={
        <button type="button" className={`link-button ${eyedropper ? 'active' : ''}`} onClick={onToggle}>
          {eyedropper ? 'Picking… (I)' : 'Pick color (I)'}
        </button>
      }
    >
      {picked ? (
        <>
          <ColorRow label="Original" color={picked.original} />
          {picked.filtered ? <ColorRow label="Filtered" color={picked.filtered} /> : null}
        </>
      ) : (
        <p className="muted">Turn on picking, then click the image to read value, chroma, and hue.</p>
      )}
    </Section>
  )
}
