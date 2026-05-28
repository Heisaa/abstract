import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import './App.css'
import {
  type FilterId,
  type FilterSettings,
  type LoadedImage,
  WebGpuImageRenderer,
  loadImageFile,
} from './webgpu'

const DEFAULT_SETTINGS: FilterSettings = {
  filter: 'kuwahara',
  radius: 4,
  levels: 5,
  grayscale: false,
  contrast: 1.15,
  edgeThreshold: 0.28,
  edgeStrength: 0.82,
  slicSize: 36,
  slicCompactness: 12,
  slicSaturationPreserve: 0.55,
  felzenszwalbScale: 450,
  felzenszwalbMinSize: 48,
  adaptiveLargeBrush: 56,
  adaptiveMediumBrush: 28,
  adaptiveSmallBrush: 14,
  adaptiveDetailThreshold: 0.45,
  adaptiveDetailStrength: 0.45,
  flowBrushLength: 44,
  flowBrushWidth: 7,
  flowSmoothing: 4,
  flowColorMerge: 0.62,
  flowEdgePreserve: 0.55,
  flowStrokeStrength: 0.95,
  strokeAbstraction: 0.68,
  strokeSize: 14,
  strokeLength: 3.2,
  strokeDetail: 0.28,
  strokeEdgeFidelity: 0.66,
  strokeColorSimplification: 0.52,
  strokeTextureStrength: 0.58,
  strokeLayers: 3,
}

const FILTERS: Array<{ id: FilterId; label: string }> = [
  { id: 'kuwahara', label: 'Kuwahara' },
  { id: 'posterize', label: 'Value' },
  { id: 'edges', label: 'Edges' },
  { id: 'strokePainter', label: 'Stroke Painter' },
  { id: 'flowBrush', label: 'Flow Brush' },
  { id: 'slic', label: 'SLIC' },
  { id: 'adaptiveBrush', label: 'Adaptive Brush' },
  { id: 'felzenszwalb', label: 'Felzenszwalb' },
]

const IMAGE_FILE_EXTENSIONS = ['.avif', '.bmp', '.gif', '.jpeg', '.jpg', '.jfif', '.png', '.webp']

type ImagePickerOptions = {
  multiple?: boolean
  types?: Array<{
    description?: string
    accept: Record<string, string[]>
  }>
}

type FilePickerHandle = {
  getFile: () => Promise<File>
}

declare global {
  interface Window {
    showOpenFilePicker?: (options?: ImagePickerOptions) => Promise<FilePickerHandle[]>
  }
}

const IMAGE_PICKER_OPTIONS: ImagePickerOptions = {
  multiple: false,
  types: [
    {
      description: 'Images',
      accept: {
        'image/*': IMAGE_FILE_EXTENSIONS,
      },
    },
  ],
}

const DEBUG_IMPORTS = false

function debugImport(message: string, details?: unknown) {
  if (!DEBUG_IMPORTS) {
    return
  }

  if (details === undefined) {
    console.log(`[import] ${message}`)
  } else {
    console.log(`[import] ${message}`, details)
  }
}

function isSupportedImageFile(file: File) {
  if (file.type.startsWith('image/')) {
    return true
  }

  const lowerName = file.name.toLowerCase()
  return IMAGE_FILE_EXTENSIONS.some((extension) => lowerName.endsWith(extension))
}

function App() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const rendererRef = useRef<WebGpuImageRenderer | null>(null)
  const objectUrlRef = useRef<string | null>(null)
  const renderFrameRef = useRef<number | null>(null)
  const renderInFlightRef = useRef(false)
  const renderQueuedRef = useRef(false)
  const latestSettingsRef = useRef<FilterSettings>(DEFAULT_SETTINGS)
  const [rendererReady, setRendererReady] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [image, setImage] = useState<LoadedImage | null>(null)
  const [settings, setSettings] = useState<FilterSettings>(DEFAULT_SETTINGS)
  const [showOriginal, setShowOriginal] = useState(false)
  const [dragging, setDragging] = useState(false)
  const [exporting, setExporting] = useState(false)
  const [status, setStatus] = useState('Ready')

  const activeSettings = useMemo<FilterSettings>(
    () => ({
      ...settings,
      filter: showOriginal ? 'original' : settings.filter,
    }),
    [settings, showOriginal],
  )

  useEffect(() => {
    latestSettingsRef.current = activeSettings
  }, [activeSettings])

  const renderPreview = useCallback(async () => {
    if (!rendererRef.current) {
      return
    }

    if (renderInFlightRef.current) {
      renderQueuedRef.current = true
      return
    }

    renderInFlightRef.current = true
    try {
      do {
        renderQueuedRef.current = false
        await rendererRef.current.render(latestSettingsRef.current)
      } while (renderQueuedRef.current)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'The preview could not be rendered.')
    } finally {
      renderInFlightRef.current = false
    }
  }, [])

  useEffect(() => {
    let cancelled = false

    async function initRenderer() {
      if (!canvasRef.current) {
        return
      }

      try {
        const renderer = new WebGpuImageRenderer(canvasRef.current, setError)
        await renderer.init()
        if (!cancelled) {
          rendererRef.current = renderer
          setRendererReady(true)
          setError(null)
          debugImport('WebGPU renderer ready')
        }
      } catch (reason) {
        if (!cancelled) {
          debugImport('WebGPU renderer failed', reason)
          setError(reason instanceof Error ? reason.message : 'WebGPU could not be initialized.')
        }
      }
    }

    void initRenderer()

    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (!image || !rendererRef.current || !rendererReady) {
      return
    }

    if (renderFrameRef.current !== null) {
      cancelAnimationFrame(renderFrameRef.current)
    }
    renderFrameRef.current = requestAnimationFrame(() => {
      renderFrameRef.current = null
      void renderPreview()
    })

    return () => {
      if (renderFrameRef.current !== null) {
        cancelAnimationFrame(renderFrameRef.current)
        renderFrameRef.current = null
      }
    }
  }, [activeSettings, image, renderPreview, rendererReady])

  useEffect(
    () => () => {
      if (objectUrlRef.current) {
        URL.revokeObjectURL(objectUrlRef.current)
      }
      image?.bitmap.close()
    },
    [image],
  )

  const importImageFile = useCallback(
    async (file?: File) => {
      debugImport('importImageFile called', {
        hasFile: Boolean(file),
        name: file?.name,
        type: file?.type,
        size: file?.size,
        rendererReady,
        hasRenderer: Boolean(rendererRef.current),
      })

      if (!file) {
        return
      }

      if (!isSupportedImageFile(file)) {
        debugImport('rejected non-image file', file.type)
        setError('Choose a PNG, JPG, or other browser-supported image file.')
        return
      }

      let loadedImage: LoadedImage | null = null
      let imageCommitted = false
      try {
        setError(null)
        setStatus(`Loading ${file.name || 'image'}...`)
        loadedImage = await loadImageFile(file)
        if (!rendererRef.current) {
          throw new Error('The WebGPU renderer is still starting. Try importing the image again.')
        }
        rendererRef.current.loadImage(loadedImage)
        await rendererRef.current.render(activeSettings)
        imageCommitted = true
        setImage(loadedImage)
        debugImport('image loaded and rendered', {
          name: loadedImage.name,
          width: loadedImage.width,
          height: loadedImage.height,
        })
        setStatus(`Loaded ${loadedImage.width} x ${loadedImage.height}px`)
      } catch (reason) {
        if (loadedImage && !imageCommitted) {
          loadedImage.bitmap.close()
        }
        debugImport('image import failed', reason)
        setError(reason instanceof Error ? reason.message : 'The image could not be loaded.')
        setStatus('Import failed')
      }
    },
    [activeSettings, rendererReady],
  )

  const openImagePicker = useCallback(async () => {
    debugImport('Open Image clicked', {
      hasOpenFilePicker: Boolean(window.showOpenFilePicker),
      isSecureContext: window.isSecureContext,
    })

    if (!window.showOpenFilePicker) {
      setError('The system image picker is not available in this browser.')
      setStatus('Image picker unavailable')
      return
    }

    try {
      setError(null)
      setStatus('Opening image picker...')
      const [handle] = await window.showOpenFilePicker(IMAGE_PICKER_OPTIONS)
      await importImageFile(await handle.getFile())
    } catch (reason) {
      if (reason instanceof DOMException && reason.name === 'AbortError') {
        setStatus(image ? `Loaded ${image.width} x ${image.height}px` : 'Ready')
        return
      }

      debugImport('image picker failed', reason)
      setError(reason instanceof Error ? reason.message : 'The image picker could not be opened.')
      setStatus('Image picker failed')
    }
  }, [image, importImageFile])

  const pasteImage = useCallback(async () => {
    try {
      debugImport('Paste clicked', {
        hasClipboardRead: Boolean(navigator.clipboard?.read),
        isSecureContext: window.isSecureContext,
      })
      setStatus('Reading clipboard...')
      if (!navigator.clipboard?.read) {
        setError('Clipboard image import is not available in this browser.')
        setStatus('Clipboard unavailable')
        return
      }

      const items = await navigator.clipboard.read()
      debugImport('clipboard items read', items.map((item) => item.types))
      for (const item of items) {
        const imageType = item.types.find((type) => type.startsWith('image/'))
        if (imageType) {
          const blob = await item.getType(imageType)
          const file = new File([blob], `pasted-image.${imageType.split('/')[1] || 'png'}`, {
            type: imageType,
          })
          void importImageFile(file)
          return
        }
      }

      setError('No image was found on the clipboard.')
      setStatus('No clipboard image found')
    } catch (reason) {
      debugImport('Paste failed', reason)
      setError(reason instanceof Error ? reason.message : 'The clipboard image could not be imported.')
      setStatus('Clipboard import failed')
    }
  }, [importImageFile])

  const loadSampleImage = useCallback(async () => {
    debugImport('Sample clicked')
    setStatus('Creating sample image...')
    const sampleCanvas = document.createElement('canvas')
    sampleCanvas.width = 1200
    sampleCanvas.height = 800
    const context = sampleCanvas.getContext('2d')
    if (!context) {
      setError('Could not create a sample image.')
      return
    }

    const gradient = context.createLinearGradient(0, 0, sampleCanvas.width, sampleCanvas.height)
    gradient.addColorStop(0, '#d7b56d')
    gradient.addColorStop(0.45, '#6d7f54')
    gradient.addColorStop(1, '#343536')
    context.fillStyle = gradient
    context.fillRect(0, 0, sampleCanvas.width, sampleCanvas.height)
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

    const blob = await new Promise<Blob>((resolve, reject) => {
      sampleCanvas.toBlob((createdBlob) => {
        if (createdBlob) {
          resolve(createdBlob)
        } else {
          reject(new Error('Could not create a sample image.'))
        }
      }, 'image/png')
    })
    void importImageFile(new File([blob], 'sample-study.png', { type: 'image/png' }))
  }, [importImageFile])

  const exportImage = useCallback(
    async (type: 'image/png' | 'image/jpeg') => {
      if (!rendererRef.current || !image) {
        return
      }

      try {
        setExporting(true)
        await rendererRef.current.render(activeSettings)
        const blob = await rendererRef.current.exportBlob(type, type === 'image/jpeg' ? 0.94 : undefined)
        const extension = type === 'image/png' ? 'png' : 'jpg'
        const baseName = image.name.replace(/\.[^.]+$/, '') || 'painting-reference'

        if (objectUrlRef.current) {
          URL.revokeObjectURL(objectUrlRef.current)
        }
        objectUrlRef.current = URL.createObjectURL(blob)

        const link = document.createElement('a')
        link.href = objectUrlRef.current
        link.download = `${baseName}-${settings.filter}.${extension}`
        link.click()
      } catch (reason) {
        setError(reason instanceof Error ? reason.message : 'The filtered image could not be exported.')
      } finally {
        setExporting(false)
      }
    },
    [activeSettings, image, settings.filter],
  )

  function updateSetting<K extends keyof FilterSettings>(key: K, value: FilterSettings[K]) {
    setSettings((current) => ({ ...current, [key]: value }))
    setShowOriginal(false)
  }

  return (
    <main className="app-shell">
      <header className="top-bar">
        <div>
          <h1>Painting Reference</h1>
          <p>Import a photo, simplify it with shader filters, then export a study image.</p>
        </div>
        <div className="actions">
          <button type="button" disabled={!image || exporting} onClick={() => void exportImage('image/png')}>
            PNG
          </button>
          <button type="button" disabled={!image || exporting} onClick={() => void exportImage('image/jpeg')}>
            JPG
          </button>
        </div>
      </header>

      <section className="workspace">
        <aside className="controls">
          <div
            className={`drop-zone ${dragging ? 'is-dragging' : ''}`}
            onDragEnter={(event) => {
              event.preventDefault()
              debugImport('drag enter')
              setDragging(true)
            }}
            onDragOver={(event) => {
              event.preventDefault()
              debugImport('drag over')
            }}
            onDragLeave={() => {
              debugImport('drag leave')
              setDragging(false)
            }}
            onDrop={(event) => {
              event.preventDefault()
              setDragging(false)
              debugImport('drop', {
                fileCount: event.dataTransfer.files.length,
                files: Array.from(event.dataTransfer.files).map((file) => ({
                  name: file.name,
                  type: file.type,
                  size: file.size,
                })),
              })
              void importImageFile(event.dataTransfer.files[0])
            }}
          >
            <span>{image ? image.name : 'Choose an image'}</span>
            {image ? <small>{image.width} x {image.height}px</small> : <small>PNG, JPG, WebP, and more</small>}
            <div className="import-actions">
              <button type="button" onClick={() => void openImagePicker()}>
                Open Image
              </button>
              <button type="button" onClick={() => void pasteImage()}>
                Paste
              </button>
              <button type="button" onClick={() => void loadSampleImage()}>
                Sample
              </button>
            </div>
            <small>or drop a file here</small>
          </div>
          <p className="status">{status}</p>

          <div className="field">
            <span className="field-label">Filter</span>
            <div className="segmented">
              {FILTERS.map((filter) => (
                <button
                  key={filter.id}
                  type="button"
                  className={settings.filter === filter.id ? 'active' : ''}
                  onClick={() => updateSetting('filter', filter.id)}
                >
                  {filter.label}
                </button>
              ))}
            </div>
          </div>

          {settings.filter === 'kuwahara' ? (
            <Slider label="Radius" min={1} max={16} step={1} value={settings.radius} onChange={(value) => updateSetting('radius', value)} />
          ) : null}

          {settings.filter === 'posterize' ? (
            <>
              <Slider label="Levels" min={2} max={12} step={1} value={settings.levels} onChange={(value) => updateSetting('levels', value)} />
              <Slider label="Contrast" min={0.5} max={2} step={0.05} value={settings.contrast} onChange={(value) => updateSetting('contrast', value)} />
              <label className="check-row">
                <input
                  type="checkbox"
                  checked={settings.grayscale}
                  onChange={(event) => updateSetting('grayscale', event.currentTarget.checked)}
                />
                Grayscale
              </label>
            </>
          ) : null}

          {settings.filter === 'edges' ? (
            <>
              <Slider label="Threshold" min={0.05} max={0.8} step={0.01} value={settings.edgeThreshold} onChange={(value) => updateSetting('edgeThreshold', value)} />
              <Slider label="Strength" min={0.1} max={1} step={0.01} value={settings.edgeStrength} onChange={(value) => updateSetting('edgeStrength', value)} />
              <Slider label="Contrast" min={0.5} max={2} step={0.05} value={settings.contrast} onChange={(value) => updateSetting('contrast', value)} />
            </>
          ) : null}

          {settings.filter === 'flowBrush' ? (
            <>
              <Slider label="Stroke Length" min={4} max={72} step={1} value={settings.flowBrushLength} onChange={(value) => updateSetting('flowBrushLength', value)} />
              <Slider label="Stroke Width" min={1} max={18} step={1} value={settings.flowBrushWidth} onChange={(value) => updateSetting('flowBrushWidth', value)} />
              <Slider label="Flow Smoothing" min={1} max={6} step={0.5} value={settings.flowSmoothing} onChange={(value) => updateSetting('flowSmoothing', value)} />
              <Slider label="Color Merge" min={0} max={1} step={0.01} value={settings.flowColorMerge} onChange={(value) => updateSetting('flowColorMerge', value)} />
              <Slider label="Edge Preserve" min={0} max={1} step={0.01} value={settings.flowEdgePreserve} onChange={(value) => updateSetting('flowEdgePreserve', value)} />
              <Slider label="Stroke Strength" min={0} max={1} step={0.01} value={settings.flowStrokeStrength} onChange={(value) => updateSetting('flowStrokeStrength', value)} />
              <Slider label="Contrast" min={0.5} max={2} step={0.05} value={settings.contrast} onChange={(value) => updateSetting('contrast', value)} />
            </>
          ) : null}

          {settings.filter === 'strokePainter' ? (
            <>
              <Slider label="Abstraction" min={0} max={1} step={0.01} value={settings.strokeAbstraction} onChange={(value) => updateSetting('strokeAbstraction', value)} />
              <Slider label="Stroke Size" min={4} max={32} step={1} value={settings.strokeSize} onChange={(value) => updateSetting('strokeSize', value)} />
              <Slider label="Stroke Length" min={1} max={5} step={0.1} value={settings.strokeLength} onChange={(value) => updateSetting('strokeLength', value)} />
              <Slider label="Detail Amount" min={0} max={1} step={0.01} value={settings.strokeDetail} onChange={(value) => updateSetting('strokeDetail', value)} />
              <Slider label="Edge Fidelity" min={0} max={1} step={0.01} value={settings.strokeEdgeFidelity} onChange={(value) => updateSetting('strokeEdgeFidelity', value)} />
              <Slider label="Color Simplify" min={0} max={1} step={0.01} value={settings.strokeColorSimplification} onChange={(value) => updateSetting('strokeColorSimplification', value)} />
              <Slider label="Texture" min={0} max={1} step={0.01} value={settings.strokeTextureStrength} onChange={(value) => updateSetting('strokeTextureStrength', value)} />
              <Slider label="Layers" min={1} max={4} step={1} value={settings.strokeLayers} onChange={(value) => updateSetting('strokeLayers', value)} />
            </>
          ) : null}

          {settings.filter === 'slic' ? (
            <>
              <Slider label="Region Size" min={8} max={96} step={1} value={settings.slicSize} onChange={(value) => updateSetting('slicSize', value)} />
              <Slider label="Compactness" min={1} max={40} step={1} value={settings.slicCompactness} onChange={(value) => updateSetting('slicCompactness', value)} />
              <Slider label="Saturation Preserve" min={0} max={1} step={0.01} value={settings.slicSaturationPreserve} onChange={(value) => updateSetting('slicSaturationPreserve', value)} />
            </>
          ) : null}

          {settings.filter === 'adaptiveBrush' ? (
            <>
              <Slider label="Large Brush" min={16} max={128} step={1} value={settings.adaptiveLargeBrush} onChange={(value) => updateSetting('adaptiveLargeBrush', value)} />
              <Slider label="Medium Brush" min={8} max={80} step={1} value={settings.adaptiveMediumBrush} onChange={(value) => updateSetting('adaptiveMediumBrush', value)} />
              <Slider label="Small Brush" min={4} max={48} step={1} value={settings.adaptiveSmallBrush} onChange={(value) => updateSetting('adaptiveSmallBrush', value)} />
              <Slider label="Detail Threshold" min={0} max={1} step={0.01} value={settings.adaptiveDetailThreshold} onChange={(value) => updateSetting('adaptiveDetailThreshold', value)} />
              <Slider label="Accent Strength" min={0} max={1} step={0.01} value={settings.adaptiveDetailStrength} onChange={(value) => updateSetting('adaptiveDetailStrength', value)} />
            </>
          ) : null}

          {settings.filter === 'felzenszwalb' ? (
            <>
              <Slider label="Scale" min={50} max={1500} step={10} value={settings.felzenszwalbScale} onChange={(value) => updateSetting('felzenszwalbScale', value)} />
              <Slider label="Min Size" min={4} max={300} step={1} value={settings.felzenszwalbMinSize} onChange={(value) => updateSetting('felzenszwalbMinSize', value)} />
            </>
          ) : null}

          <button
            type="button"
            className="compare"
            disabled={!image}
            onClick={() => setShowOriginal((current) => !current)}
          >
            {showOriginal ? 'Show Filter' : 'Show Original'}
          </button>

          {error ? <p className="error">{error}</p> : null}
        </aside>

        <section className="preview-area">
          <div className="canvas-frame">
            {!image ? <div className="empty-state">No image loaded</div> : null}
            <canvas ref={canvasRef} aria-label="Filtered image preview" />
          </div>
        </section>
      </section>
    </main>
  )
}

type SliderProps = {
  label: string
  min: number
  max: number
  step: number
  value?: number
  onChange: (value: number) => void
}

function Slider({ label, min, max, step, value, onChange }: SliderProps) {
  const candidateValue = value ?? min
  const safeValue = Number.isFinite(candidateValue) ? candidateValue : min

  return (
    <label className="field">
      <span className="field-label">
        {label}
        <output>{Number.isInteger(safeValue) ? safeValue : safeValue.toFixed(2)}</output>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={safeValue}
        onChange={(event) => onChange(Number(event.currentTarget.value))}
      />
    </label>
  )
}

export default App
