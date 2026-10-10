import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import './App.css'
import { Section, Segmented } from './components/controls'
import { FilterParams, FilterPicker } from './components/FilterControls'
import { ColorReadout, PaletteSwatches, PresetPanel, ViewPanel, type PickedColor } from './components/Panels'
import { Preview } from './components/Preview'
import type { Rgb } from './cpu/common'
import { CpuWorkerClient } from './cpu/workerClient'
import { getFilter, sanitizeParams, defaultParams, type FilterId, type ParamValue, type ParamValues } from './filters/registry'
import { WebGpuImageRenderer, type PreviewInfo, type RenderState } from './gpu/renderer'
import {
  decodeAtMaxDimension,
  downloadBlob,
  encodePixels,
  EXPORT_MAX_DIMENSION,
  highResExportSize,
  IMAGE_FILE_EXTENSIONS,
  isSupportedImageFile,
  loadImageFile,
  type LoadedImage,
} from './image'
import {
  createPresetId,
  loadPresets,
  loadSettings,
  savePresets,
  saveSettings,
  type ExportFormat,
  type ExportSize,
  type Preset,
  type StoredSettings,
} from './storage'
import { computeCrop, outputToImageUv, type ViewSettings } from './view'

type ImagePickerOptions = {
  multiple?: boolean
  types?: Array<{ description?: string; accept: Record<string, string[]> }>
}

declare global {
  interface Window {
    showOpenFilePicker?: (options?: ImagePickerOptions) => Promise<Array<{ getFile: () => Promise<File> }>>
  }
}

const IMAGE_PICKER_OPTIONS: ImagePickerOptions = {
  multiple: false,
  types: [{ description: 'Images', accept: { 'image/*': IMAGE_FILE_EXTENSIONS } }],
}

function errorMessage(reason: unknown, fallback: string) {
  return reason instanceof Error ? reason.message : fallback
}

function isTypingTarget(target: EventTarget | null) {
  if (!(target instanceof HTMLElement)) {
    return false
  }
  if (target.isContentEditable || target.tagName === 'TEXTAREA' || target.tagName === 'SELECT') {
    return true
  }
  return target.tagName === 'INPUT' && !['range', 'checkbox', 'radio', 'button'].includes((target as HTMLInputElement).type)
}

function App() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const rendererRef = useRef<WebGpuImageRenderer | null>(null)
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const [rendererReady, setRendererReady] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [image, setImage] = useState<LoadedImage | null>(null)
  const [imageToken, setImageToken] = useState(0)
  const [settings, setSettings] = useState<StoredSettings>(loadSettings)
  const [presets, setPresets] = useState<Preset[]>(loadPresets)
  const [showOriginal, setShowOriginal] = useState(false)
  const [split, setSplit] = useState({ enabled: false, position: 0.5 })
  const [interacting, setInteracting] = useState(false)
  const [dragging, setDragging] = useState(false)
  const [exporting, setExporting] = useState(false)
  const [busy, setBusy] = useState(false)
  const [status, setStatus] = useState('Ready')
  const [previewInfo, setPreviewInfo] = useState<PreviewInfo | undefined>()
  const [eyedropper, setEyedropper] = useState(false)
  const [picked, setPicked] = useState<PickedColor | undefined>()

  const filterDefinition = getFilter(settings.filter)
  const effectiveFilter: FilterId = showOriginal ? 'original' : settings.filter

  useEffect(() => {
    const timer = window.setTimeout(() => saveSettings(settings), 250)
    return () => window.clearTimeout(timer)
  }, [settings])

  useEffect(() => {
    savePresets(presets)
  }, [presets])

  useEffect(() => {
    let cancelled = false
    let renderer: WebGpuImageRenderer | undefined
    const worker = new CpuWorkerClient()
    const canvas = canvasRef.current
    if (!canvas) {
      return
    }

    WebGpuImageRenderer.create(canvas, worker, {
      onError: (message) => setError(message),
      onBusyChange: (value) => setBusy(value),
    })
      .then((created) => {
        if (cancelled) {
          created.destroy()
          return
        }
        renderer = created
        rendererRef.current = created
        setRendererReady(true)
      })
      .catch((reason) => {
        if (!cancelled) {
          setError(errorMessage(reason, 'WebGPU could not be initialized.'))
        }
      })

    return () => {
      cancelled = true
      renderer?.destroy()
      worker.terminate()
      rendererRef.current = null
    }
  }, [])

  const renderState = useMemo<RenderState>(
    () => ({
      filter: effectiveFilter,
      params: settings.params[effectiveFilter] ?? {},
      view: settings.view,
      split,
      draft: interacting,
    }),
    [effectiveFilter, interacting, settings.params, settings.view, split],
  )

  useEffect(() => {
    const renderer = rendererRef.current
    if (!renderer || !rendererReady || !image) {
      return
    }
    renderer
      .requestPreview(renderState)
      .then((info) => {
        if (info) {
          setPreviewInfo(info)
        }
      })
      .catch((reason) => setError(errorMessage(reason, 'The preview could not be rendered.')))
  }, [image, renderState, rendererReady])

  // Sliders render CPU filters at draft resolution while being dragged.
  useEffect(() => {
    const onPointerDown = (event: PointerEvent) => {
      if (event.target instanceof HTMLElement && event.target.closest('[data-draft-control]')) {
        setInteracting(true)
      }
    }
    const onPointerUp = () => setInteracting(false)
    window.addEventListener('pointerdown', onPointerDown)
    window.addEventListener('pointerup', onPointerUp)
    window.addEventListener('pointercancel', onPointerUp)
    return () => {
      window.removeEventListener('pointerdown', onPointerDown)
      window.removeEventListener('pointerup', onPointerUp)
      window.removeEventListener('pointercancel', onPointerUp)
    }
  }, [])

  const updateView = useCallback((patch: Partial<ViewSettings>) => {
    setSettings((current) => ({ ...current, view: { ...current.view, ...patch } }))
  }, [])

  const importImageFile = useCallback(async (file?: File) => {
    if (!file) {
      return
    }
    if (!isSupportedImageFile(file)) {
      setError('Choose a PNG, JPG, or other browser-supported image file.')
      return
    }

    try {
      setError(null)
      setStatus(`Loading ${file.name || 'image'}…`)
      const renderer = rendererRef.current
      if (!renderer) {
        throw new Error('The WebGPU renderer is still starting. Try importing the image again.')
      }
      const loaded = await loadImageFile(file)
      await renderer.setImage(loaded)
      setImage(loaded)
      setImageToken((token) => token + 1)
      setPicked(undefined)
      const reduced = loaded.sourceWidth > loaded.width ? ` (from ${loaded.sourceWidth} × ${loaded.sourceHeight})` : ''
      setStatus(`Loaded ${loaded.width} × ${loaded.height}px${reduced}`)
    } catch (reason) {
      setError(errorMessage(reason, 'The image could not be loaded.'))
      setStatus('Import failed')
    }
  }, [])

  const openImagePicker = useCallback(async () => {
    if (!window.showOpenFilePicker) {
      fileInputRef.current?.click()
      return
    }

    try {
      setError(null)
      const [handle] = await window.showOpenFilePicker(IMAGE_PICKER_OPTIONS)
      await importImageFile(await handle.getFile())
    } catch (reason) {
      if (reason instanceof DOMException && reason.name === 'AbortError') {
        return
      }
      // Some contexts expose the API but refuse it; fall back to the input.
      fileInputRef.current?.click()
    }
  }, [importImageFile])

  const pasteFromClipboard = useCallback(async () => {
    try {
      if (!navigator.clipboard?.read) {
        setError('Clipboard image reading is not available here. Press Ctrl+V (⌘V) to paste instead.')
        return
      }
      const items = await navigator.clipboard.read()
      for (const item of items) {
        const imageType = item.types.find((type) => type.startsWith('image/'))
        if (imageType) {
          const blob = await item.getType(imageType)
          await importImageFile(new File([blob], `pasted-image.${imageType.split('/')[1] || 'png'}`, { type: imageType }))
          return
        }
      }
      setError('No image was found on the clipboard.')
    } catch (reason) {
      setError(errorMessage(reason, 'The clipboard image could not be imported.'))
    }
  }, [importImageFile])

  // Ctrl+V / ⌘V anywhere pastes an image without clipboard permissions.
  useEffect(() => {
    const onPaste = (event: ClipboardEvent) => {
      if (isTypingTarget(event.target)) {
        return
      }
      const file = Array.from(event.clipboardData?.files ?? []).find((candidate) => candidate.type.startsWith('image/'))
      if (file) {
        event.preventDefault()
        void importImageFile(file)
      }
    }
    document.addEventListener('paste', onPaste)
    return () => document.removeEventListener('paste', onPaste)
  }, [importImageFile])

  const loadSampleImage = useCallback(async () => {
    const canvas = new OffscreenCanvas(1200, 800)
    const context = canvas.getContext('2d')
    if (!context) {
      setError('Could not create a sample image.')
      return
    }
    const gradient = context.createLinearGradient(0, 0, 1200, 800)
    gradient.addColorStop(0, '#d7b56d')
    gradient.addColorStop(0.45, '#6d7f54')
    gradient.addColorStop(1, '#343536')
    context.fillStyle = gradient
    context.fillRect(0, 0, 1200, 800)
    context.fillStyle = '#f1ece0'
    context.beginPath()
    context.arc(360, 310, 190, 0, Math.PI * 2)
    context.fill()
    context.fillStyle = '#883d2c'
    context.fillRect(610, 190, 340, 430)
    context.fillStyle = '#1f2527'
    context.beginPath()
    context.moveTo(180, 680)
    context.lineTo(1040, 680)
    context.lineTo(860, 520)
    context.lineTo(310, 545)
    context.closePath()
    context.fill()
    const blob = await canvas.convertToBlob({ type: 'image/png' })
    await importImageFile(new File([blob], 'sample-study.png', { type: 'image/png' }))
  }, [importImageFile])

  const exportImage = useCallback(async () => {
    const renderer = rendererRef.current
    if (!renderer || !image) {
      return
    }

    try {
      setExporting(true)
      setError(null)
      const useHighRes = settings.exportSize === 'high' && highResExportSize(image).width > image.width
      setStatus(useHighRes ? 'Decoding full-resolution image…' : 'Exporting…')
      const highRes = useHighRes ? await decodeAtMaxDimension(image.file, EXPORT_MAX_DIMENSION) : undefined
      if (useHighRes) {
        setStatus('Rendering high-resolution export…')
      }
      const result = await renderer.exportImage(
        { filter: settings.filter, params: settings.params[settings.filter], view: settings.view, split: { enabled: false, position: 0 }, draft: false },
        { highRes, includeViewAids: settings.view.exportViewAids },
      )
      const type = settings.exportFormat === 'png' ? 'image/png' : 'image/jpeg'
      const blob = await encodePixels(result, type, type === 'image/jpeg' ? 0.94 : undefined)
      const baseName = image.name.replace(/\.[^.]+$/, '') || 'painting-reference'
      downloadBlob(blob, `${baseName}-${settings.filter}.${settings.exportFormat === 'png' ? 'png' : 'jpg'}`)
      setStatus(`Exported ${result.width} × ${result.height}px`)
    } catch (reason) {
      setError(errorMessage(reason, 'The filtered image could not be exported.'))
      setStatus('Export failed')
    } finally {
      setExporting(false)
    }
  }, [image, settings])

  const pickColor = useCallback(
    async (u: number, v: number) => {
      const renderer = rendererRef.current
      if (!renderer || !image) {
        return
      }
      const crop = computeCrop(settings.view, image.width, image.height)
      const point = outputToImageUv(settings.view, crop, u, v)
      const x = Math.min(image.width - 1, Math.max(0, Math.floor(point.u * image.width)))
      const y = Math.min(image.height - 1, Math.max(0, Math.floor(point.v * image.height)))
      const offset = (x + y * image.width) * 4
      const original: Rgb = [image.pixels[offset], image.pixels[offset + 1], image.pixels[offset + 2]]
      try {
        const filtered = await renderer.readFilteredPixel(point.u, point.v)
        setPicked({ original, filtered: effectiveFilter === 'original' ? undefined : filtered })
      } catch (reason) {
        setError(errorMessage(reason, 'Could not read the color.'))
      }
    },
    [effectiveFilter, image, settings.view],
  )

  // Keyboard shortcuts: hold Space for the original; G, M, S, E, I toggles.
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (isTypingTarget(event.target) || event.ctrlKey || event.metaKey || event.altKey) {
        return
      }
      if (event.code === 'Space') {
        event.preventDefault()
        setShowOriginal(true)
        return
      }
      if (event.repeat) {
        return
      }
      const key = event.key.toLowerCase()
      if (key === 'g') {
        setSettings((current) => ({ ...current, view: { ...current.view, grayscale: !current.view.grayscale } }))
      } else if (key === 'm') {
        setSettings((current) => ({ ...current, view: { ...current.view, mirror: !current.view.mirror } }))
      } else if (key === 'e') {
        setSettings((current) => ({ ...current, view: { ...current.view, overlay: !current.view.overlay } }))
      } else if (key === 's') {
        setSplit((current) => ({ ...current, enabled: !current.enabled }))
      } else if (key === 'i') {
        setEyedropper((current) => !current)
      } else if (key === 'escape') {
        setEyedropper(false)
      }
    }
    const onKeyUp = (event: KeyboardEvent) => {
      if (event.code === 'Space') {
        setShowOriginal(false)
      }
    }
    const onBlur = () => setShowOriginal(false)
    window.addEventListener('keydown', onKeyDown)
    window.addEventListener('keyup', onKeyUp)
    window.addEventListener('blur', onBlur)
    return () => {
      window.removeEventListener('keydown', onKeyDown)
      window.removeEventListener('keyup', onKeyUp)
      window.removeEventListener('blur', onBlur)
    }
  }, [])

  const selectFilter = (filter: FilterId) => {
    setSettings((current) => ({ ...current, filter }))
    setShowOriginal(false)
  }

  const updateParam = (key: string, value: ParamValue) => {
    setSettings((current) => {
      const filter = current.filter
      let next: ParamValues = { ...current.params[filter], [key]: value }
      const patch = getFilter(filter).onParamChange?.(key, next)
      if (patch) {
        next = { ...next, ...patch }
      }
      return { ...current, params: { ...current.params, [filter]: next } }
    })
    setShowOriginal(false)
  }

  const replaceParams = (params: ParamValues) => {
    setSettings((current) => ({ ...current, params: { ...current.params, [current.filter]: sanitizeParams(current.filter, params) } }))
  }

  const crop = image ? computeCrop(settings.view, image.width, image.height) : undefined
  const highRes = image ? highResExportSize(image) : undefined
  const canExportHigher = Boolean(image && highRes && highRes.width > image.width)

  return (
    <main className="app-shell">
      <header className="top-bar">
        <div>
          <h1>Painting Reference</h1>
          <p>Simplify a photo into paintable masses, values, and edges, then export the study.</p>
        </div>
        <div className="actions">
          <Segmented<ExportFormat>
            value={settings.exportFormat}
            options={[
              { value: 'png', label: 'PNG' },
              { value: 'jpeg', label: 'JPG' },
            ]}
            onChange={(exportFormat) => setSettings((current) => ({ ...current, exportFormat }))}
          />
          <select
            aria-label="Export size"
            value={settings.exportSize}
            onChange={(event) => setSettings((current) => ({ ...current, exportSize: event.currentTarget.value as ExportSize }))}
          >
            <option value="preview">Preview size{crop ? ` (${crop.width} × ${crop.height})` : ''}</option>
            <option value="high" disabled={Boolean(image) && !canExportHigher}>
              High-res{highRes && canExportHigher ? ` (up to ${highRes.width} × ${highRes.height})` : ''}
            </option>
          </select>
          <button type="button" className="primary" disabled={!image || exporting} onClick={() => void exportImage()}>
            {exporting ? 'Exporting…' : 'Export'}
          </button>
        </div>
      </header>

      <section className="workspace">
        <aside className="controls">
          <div
            className={`drop-zone ${dragging ? 'is-dragging' : ''}`}
            onDragEnter={(event) => {
              event.preventDefault()
              setDragging(true)
            }}
            onDragOver={(event) => event.preventDefault()}
            onDragLeave={() => setDragging(false)}
            onDrop={(event) => {
              event.preventDefault()
              setDragging(false)
              void importImageFile(event.dataTransfer.files[0])
            }}
          >
            <span>{image ? image.name : 'Choose an image'}</span>
            <small>{image ? `${image.width} × ${image.height}px preview` : 'PNG, JPG, WebP, and more'}</small>
            <div className="import-actions">
              <button type="button" onClick={() => void openImagePicker()}>
                Open
              </button>
              <button type="button" onClick={() => void pasteFromClipboard()}>
                Paste
              </button>
              <button type="button" onClick={() => void loadSampleImage()}>
                Sample
              </button>
            </div>
            <small>or drop a file here · Ctrl+V pastes</small>
            <input
              ref={fileInputRef}
              type="file"
              accept={['image/*', ...IMAGE_FILE_EXTENSIONS].join(',')}
              hidden
              onChange={(event) => {
                void importImageFile(event.currentTarget.files?.[0])
                event.currentTarget.value = ''
              }}
            />
          </div>
          <p className="status">{status}</p>
          {error ? (
            <p className="error" role="alert">
              {error}
              <button type="button" className="link-button" onClick={() => setError(null)}>
                Dismiss
              </button>
            </p>
          ) : null}

          <Section title="Filter">
            <FilterPicker value={settings.filter} onChange={selectFilter} />
          </Section>

          <Section title={filterDefinition.label}>
            <FilterParams filter={settings.filter} values={settings.params[settings.filter]} onChange={updateParam} />
            {previewInfo?.palette && !showOriginal ? <PaletteSwatches palette={previewInfo.palette} /> : null}
          </Section>

          <PresetPanel
            filter={settings.filter}
            filterLabel={filterDefinition.label}
            styles={filterDefinition.styles}
            presets={presets}
            onApply={replaceParams}
            onSave={(name) =>
              setPresets((current) => [...current, { id: createPresetId(), name, filter: settings.filter, params: settings.params[settings.filter] }])
            }
            onDelete={(id) => setPresets((current) => current.filter((preset) => preset.id !== id))}
            onReset={() => replaceParams(defaultParams(settings.filter))}
          />
        </aside>

        <section className="preview-area">
          <Preview
            canvasRef={canvasRef}
            hasImage={Boolean(image)}
            resetToken={imageToken}
            outputWidth={crop?.width ?? 0}
            outputHeight={crop?.height ?? 0}
            split={split}
            onSplitChange={(position) => setSplit((current) => ({ ...current, position }))}
            eyedropper={eyedropper}
            onPick={(u, v) => void pickColor(u, v)}
            busy={busy}
          />
          <footer className="preview-status">
            <span>
              {showOriginal ? 'Original' : filterDefinition.label}
              {previewInfo ? ` · ${previewInfo.width} × ${previewInfo.height}px · ${Math.round(previewInfo.ms)} ms` : ''}
              {previewInfo?.draft ? ' · draft' : ''}
            </span>
            <span className="shortcuts">Hold Space: original · S split · G gray · M mirror · E edges · I pick · wheel zoom · drag pan</span>
          </footer>
        </section>

        <aside className="tools">
          <ViewPanel
            view={settings.view}
            onChange={updateView}
            split={split}
            onSplitToggle={(enabled) => setSplit((current) => ({ ...current, enabled }))}
          />
          <ColorReadout picked={picked} eyedropper={eyedropper} onToggle={() => setEyedropper((current) => !current)} />
          {!rendererReady && !error ? <p className="muted">Starting WebGPU…</p> : null}
        </aside>
      </section>
    </main>
  )
}

export default App
