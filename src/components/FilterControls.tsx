import { FILTER_GROUPS, FILTERS, getFilter, type FilterId, type ParamDef, type ParamValue, type ParamValues } from '../filters/registry'
import { Segmented, Slider, Toggle } from './controls'

export function FilterPicker({ value, onChange }: { value: FilterId; onChange: (filter: FilterId) => void }) {
  return (
    <div className="field">
      {FILTER_GROUPS.map((group) => (
        <div key={group} className="filter-group">
          <span className="filter-group-title">{group}</span>
          <div className="segmented">
            {FILTERS.filter((filter) => filter.group === group).map((filter) => (
              <button
                key={filter.id}
                type="button"
                title={filter.description}
                className={value === filter.id ? 'active' : ''}
                onClick={() => onChange(filter.id)}
              >
                {filter.label}
              </button>
            ))}
          </div>
        </div>
      ))}
    </div>
  )
}

function ParamControl({ param, value, onChange }: { param: ParamDef; value: ParamValue; onChange: (value: ParamValue) => void }) {
  if (param.type === 'range') {
    return (
      <Slider
        label={param.label}
        hint={param.hint}
        min={param.min}
        max={param.max}
        step={param.step}
        value={typeof value === 'number' ? value : param.default}
        format={param.unit === 'px' ? (amount) => `${amount.toFixed(param.step < 1 ? 1 : 0)} px` : undefined}
        onChange={onChange}
      />
    )
  }

  if (param.type === 'toggle') {
    return <Toggle label={param.label} hint={param.hint} checked={value === true} onChange={onChange} />
  }

  return <Segmented label={param.label} value={String(value)} options={param.options} onChange={onChange} />
}

// Controls generated from the filter's registry definition.
export function FilterParams({ filter, values, onChange }: { filter: FilterId; values: ParamValues; onChange: (key: string, value: ParamValue) => void }) {
  const definition = getFilter(filter)
  return (
    <>
      <p className="filter-description">{definition.description}</p>
      {definition.params
        .filter((param) => !param.visible || param.visible(values))
        .map((param) => (
          <ParamControl key={param.key} param={param} value={values[param.key]} onChange={(value) => onChange(param.key, value)} />
        ))}
    </>
  )
}
