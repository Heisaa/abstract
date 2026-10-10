import type { ReactNode } from 'react'

type SliderProps = {
  label: string
  min: number
  max: number
  step: number
  value: number
  hint?: string
  format?: (value: number) => string
  onChange: (value: number) => void
}

function formatValue(value: number, step: number) {
  if (Number.isInteger(step) && Number.isInteger(value)) {
    return String(value)
  }
  return value.toFixed(step < 0.01 ? 3 : 2)
}

export function Slider({ label, min, max, step, value, hint, format, onChange }: SliderProps) {
  const safeValue = Number.isFinite(value) ? value : min
  return (
    <label className="field" title={hint}>
      <span className="field-label">
        {label}
        <output>{format ? format(safeValue) : formatValue(safeValue, step)}</output>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={safeValue}
        data-draft-control=""
        onChange={(event) => onChange(Number(event.currentTarget.value))}
      />
    </label>
  )
}

export function Toggle({ label, checked, hint, onChange }: { label: string; checked: boolean; hint?: string; onChange: (checked: boolean) => void }) {
  return (
    <label className="check-row" title={hint}>
      <input type="checkbox" checked={checked} onChange={(event) => onChange(event.currentTarget.checked)} />
      {label}
    </label>
  )
}

export function Segmented<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label?: string
  value: T
  options: Array<{ value: T; label: string }>
  onChange: (value: T) => void
}) {
  return (
    <div className="field">
      {label ? <span className="field-label">{label}</span> : null}
      <div className={`segmented ${options.length > 3 ? 'segmented-many' : ''}`} role="radiogroup" aria-label={label}>
        {options.map((option) => (
          <button
            key={option.value}
            type="button"
            role="radio"
            aria-checked={value === option.value}
            className={value === option.value ? 'active' : ''}
            onClick={() => onChange(option.value)}
          >
            {option.label}
          </button>
        ))}
      </div>
    </div>
  )
}

export function Section({ title, children, actions }: { title: string; children: ReactNode; actions?: ReactNode }) {
  return (
    <section className="panel-section">
      <header className="panel-section-header">
        <h2>{title}</h2>
        {actions}
      </header>
      {children}
    </section>
  )
}
