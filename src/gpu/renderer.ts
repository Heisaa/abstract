import { srgbToOklab } from '../cpu/color'
import type { Rgb } from '../cpu/common'
import type { ImageSlot } from '../cpu/protocol'
import type { CpuJobResult } from '../cpu/workerClient'
import { analysisKey, bool, fullKey, getFilter, num, scaleParams, str, type FilterId, type ParamValues } from '../filters/registry'
import type { DecodedPixels } from '../image'
import { computeCrop, type ViewSettings } from '../view'
import { PASS_SHADER, SLIC_SHADER } from './shaders'

// Rendering pipeline:
//   source texture -> filter stage (GPU passes, worker CPU job, or both)
//                  -> cached "filtered" texture
//                  -> composite pass (crop, mirror, split, grayscale, grid,
//                     edge overlay) -> canvas or offscreen export texture
//
// Preview and export run the same filter and composite code; export only
// differs in resolution (pixel-unit parameters are scaled) and target.

export type CpuBackend = {
  setImage(slot: ImageSlot, version: number, width: number, height: number, pixels: Uint8ClampedArray): void
  dropImage(slot: ImageSlot): void
  run(options: {
    slot: ImageSlot
    draft: boolean
    scale: number
    filter: FilterId
    params: ParamValues
    labels?: Int32Array
  }): Promise<CpuJobResult>
}

export type RenderState = {
  filter: FilterId
  params: ParamValues
  view: ViewSettings
  split: { enabled: boolean; position: number }
  // Render CPU filters at half resolution (while a slider is dragged).
  draft: boolean
}

export type PreviewInfo = {
  width: number
  height: number
  ms: number
  draft: boolean
  palette?: Rgb[]
}

export type ExportResult = DecodedPixels

const PASS_ENTRIES = {
  tensor: 'tensorMain',
  blur: 'blurMain',
  tfm: 'tfmMain',
  akf: 'akfMain',
  dog: 'dogMain',
  lic: 'licMain',
  edges: 'edgesMain',
  value: 'valueMain',
  chroma: 'chromaMain',
  temperature: 'temperatureMain',
  flowBrush: 'flowBrushMain',
  lostFound: 'lostFoundMain',
} as const

type PassName = keyof typeof PASS_ENTRIES

// Expensive passes are split into separately submitted tiles on large
// targets so a single submission cannot trip the GPU watchdog.
const HEAVY_PASSES = new Set<PassName>(['akf', 'flowBrush', 'lic'])
const TILE_SIZE = 1024
const TILE_PIXEL_THRESHOLD = 2_500_000

const SCRATCH_FORMAT: GPUTextureFormat = 'rgba16float'

type FilteredResult = {
  key: string
  texture: GPUTexture
  palette?: Rgb[]
  draft: boolean
}

type ImageResources = {
  slot: ImageSlot
  version: number
  width: number
  height: number
  pixels: Uint8ClampedArray
  source: GPUTexture
  scratch: Map<string, GPUTexture>
  cpuTexture?: GPUTexture
  filtered?: FilteredResult
  overlayKey?: string
  slicLabels?: { key: string; labels: Int32Array }
}

type PassBatch = {
  encoder: GPUCommandEncoder
  buffers: GPUBuffer[]
}

type SlicPipelines = {
  layout: GPUBindGroupLayout
  clearSums: GPUComputePipeline
  assignPixels: GPUComputePipeline
  accumulateCenters: GPUComputePipeline
  updateCenters: GPUComputePipeline
}

async function compileModule(device: GPUDevice, code: string, label: string) {
  const module = device.createShaderModule({ code, label })
  const info = await module.getCompilationInfo()
  const errors = info.messages.filter((message) => message.type === 'error')
  if (errors.length > 0) {
    throw new Error(`${label} failed to compile:\n${errors.map((message) => `${message.lineNum}:${message.linePos} ${message.message}`).join('\n')}`)
  }
  return module
}

export class WebGpuImageRenderer {
  private readonly canvas: HTMLCanvasElement
  private readonly cpu: CpuBackend
  private readonly device: GPUDevice
  private readonly context: GPUCanvasContext
  private readonly canvasFormat: GPUTextureFormat
  private readonly onBusyChange?: (busy: boolean) => void
  private contextConfigured = false
  private queuedTasks = 0
  private readonly sampler: GPUSampler
  private readonly passLayout: GPUBindGroupLayout
  private readonly passPipelines: Map<PassName, GPURenderPipeline>
  private readonly canvasComposite: GPURenderPipeline
  private readonly exportComposite: GPURenderPipeline
  private readonly slic: SlicPipelines
  private readonly dummyTexture: GPUTexture
  private preview?: ImageResources
  private versionCounter = 0
  private chain: Promise<unknown> = Promise.resolve()
  private pendingPreview?: RenderState
  private previewPromise?: Promise<PreviewInfo | undefined>
  private destroyed = false

  private constructor(options: {
    canvas: HTMLCanvasElement
    cpu: CpuBackend
    device: GPUDevice
    context: GPUCanvasContext
    canvasFormat: GPUTextureFormat
    onBusyChange?: (busy: boolean) => void
    passLayout: GPUBindGroupLayout
    passPipelines: Map<PassName, GPURenderPipeline>
    canvasComposite: GPURenderPipeline
    exportComposite: GPURenderPipeline
    slic: SlicPipelines
  }) {
    this.canvas = options.canvas
    this.cpu = options.cpu
    this.device = options.device
    this.context = options.context
    this.canvasFormat = options.canvasFormat
    this.onBusyChange = options.onBusyChange
    this.passLayout = options.passLayout
    this.passPipelines = options.passPipelines
    this.canvasComposite = options.canvasComposite
    this.exportComposite = options.exportComposite
    this.slic = options.slic
    this.sampler = this.device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    })
    this.dummyTexture = this.device.createTexture({
      size: [1, 1],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    })
  }

  static async create(
    canvas: HTMLCanvasElement,
    cpu: CpuBackend,
    callbacks: { onError: (message: string) => void; onBusyChange?: (busy: boolean) => void },
  ) {
    const { onError, onBusyChange } = callbacks
    if (!navigator.gpu) {
      throw new Error('WebGPU is not available in this browser.')
    }
    const adapter = await navigator.gpu.requestAdapter()
    if (!adapter) {
      throw new Error('No compatible GPU adapter was found.')
    }

    const device = await adapter.requestDevice()
    device.addEventListener('uncapturederror', (event) => onError((event as GPUUncapturedErrorEvent).error.message))
    void device.lost.then((info) => {
      if (info.reason !== 'destroyed') {
        onError(`The GPU device was lost (${info.message}). Reload the page to continue.`)
      }
    })

    const context = canvas.getContext('webgpu') as GPUCanvasContext | null
    if (!context) {
      throw new Error('Could not create a WebGPU canvas context.')
    }
    // The context is configured on first render, so an instance discarded
    // during startup (React StrictMode) never claims the canvas.
    const format = navigator.gpu.getPreferredCanvasFormat()

    const passModule = await compileModule(device, PASS_SHADER, 'Filter shaders')
    const slicModule = await compileModule(device, SLIC_SHADER, 'SLIC shader')

    const textureEntry = (binding: number): GPUBindGroupLayoutEntry => ({
      binding,
      visibility: GPUShaderStage.FRAGMENT,
      texture: { sampleType: 'float' },
    })
    const passLayout = device.createBindGroupLayout({
      label: 'pass layout',
      entries: [
        textureEntry(0),
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        textureEntry(3),
        textureEntry(4),
        textureEntry(5),
      ],
    })
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [passLayout] })
    const createPipeline = (entryPoint: string, targetFormat: GPUTextureFormat) =>
      device.createRenderPipelineAsync({
        label: entryPoint,
        layout: pipelineLayout,
        vertex: { module: passModule, entryPoint: 'vertexMain' },
        fragment: { module: passModule, entryPoint, targets: [{ format: targetFormat }] },
        primitive: { topology: 'triangle-list' },
      })

    const passPipelines = new Map<PassName, GPURenderPipeline>()
    await Promise.all(
      (Object.entries(PASS_ENTRIES) as Array<[PassName, string]>).map(async ([name, entryPoint]) => {
        passPipelines.set(name, await createPipeline(entryPoint, SCRATCH_FORMAT))
      }),
    )

    const slicLayout = device.createBindGroupLayout({
      label: 'slic layout',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    })
    const slicPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [slicLayout] })
    const createCompute = (entryPoint: string) =>
      device.createComputePipelineAsync({ label: entryPoint, layout: slicPipelineLayout, compute: { module: slicModule, entryPoint } })

    const [canvasComposite, exportComposite, clearSums, assignPixels, accumulateCenters, updateCenters] = await Promise.all([
      createPipeline('compositeMain', format),
      createPipeline('compositeMain', 'rgba8unorm'),
      createCompute('clearSums'),
      createCompute('assignPixels'),
      createCompute('accumulateCenters'),
      createCompute('updateCenters'),
    ])

    return new WebGpuImageRenderer({
      canvas,
      cpu,
      device,
      context,
      canvasFormat: format,
      onBusyChange,
      passLayout,
      passPipelines,
      canvasComposite,
      exportComposite,
      slic: { layout: slicLayout, clearSums, assignPixels, accumulateCenters, updateCenters },
    })
  }

  // ---------------------------------------------------------------------------
  // Public API. Every operation runs through one serialized queue so GPU
  // resources are never replaced mid-render and error scopes never interleave.

  setImage(image: DecodedPixels) {
    return this.enqueue(async () => {
      if (this.preview) {
        this.destroyResources(this.preview)
      }
      this.preview = this.createResources('preview', image)
    })
  }

  // Coalesces requests: while a render is running, only the newest pending
  // state is rendered next.
  requestPreview(state: RenderState) {
    this.pendingPreview = state
    if (!this.previewPromise) {
      this.previewPromise = this.enqueue(async () => {
        const latest = this.pendingPreview
        this.pendingPreview = undefined
        this.previewPromise = undefined
        return latest ? this.renderPreview(latest) : undefined
      })
    }
    return this.previewPromise
  }

  // Renders the current state offscreen. Pass `highRes` pixels to render at
  // a larger size than the preview.
  exportImage(state: RenderState, options: { highRes?: DecodedPixels; includeViewAids: boolean }) {
    return this.enqueue(() =>
      this.withErrorScope(async () => {
        const base = this.requirePreview()
        let resources = base
        let temporary: ImageResources | undefined
        if (options.highRes && options.highRes.width > base.width) {
          temporary = this.createResources('export', options.highRes)
          resources = temporary
        }
        const scale = resources.width / base.width
        let target: GPUTexture | undefined

        try {
          const filtered = await this.prepareFiltered(resources, state.filter, state.params, false, scale)
          const overlay = this.prepareOverlay(resources, state.view, scale)
          const crop = computeCrop(state.view, resources.width, resources.height)
          target = this.device.createTexture({
            size: [crop.width, crop.height],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
          })
          const encoder = this.device.createCommandEncoder()
          const buffer = this.composite(encoder, target.createView(), this.exportComposite, resources, filtered.texture, overlay, state.view, {
            crop,
            split: { enabled: false, position: 0 },
            includeViewAids: options.includeViewAids,
            lineWidth: Math.max(1.5, 1.5 * scale),
          })
          this.device.queue.submit([encoder.finish()])
          buffer.destroy()
          const pixels = await this.readTexture(target, 0, 0, crop.width, crop.height, 4)
          return { pixels: new Uint8ClampedArray(pixels.buffer), width: crop.width, height: crop.height }
        } finally {
          target?.destroy()
          if (temporary) {
            this.destroyResources(temporary)
          }
        }
      }),
    )
  }

  // Reads the filtered (pre-composite) color at normalized image coordinates.
  readFilteredPixel(u: number, v: number) {
    return this.enqueue(async (): Promise<Rgb | undefined> => {
      const filtered = this.preview?.filtered
      if (!filtered) {
        return undefined
      }
      const { texture } = filtered
      const x = Math.min(texture.width - 1, Math.max(0, Math.floor(u * texture.width)))
      const y = Math.min(texture.height - 1, Math.max(0, Math.floor(v * texture.height)))
      if (texture.format === 'rgba8unorm') {
        const data = await this.readTexture(texture, x, y, 1, 1, 4)
        return [data[0], data[1], data[2]]
      }
      const data = await this.readTexture(texture, x, y, 1, 1, 8)
      const halves = new Uint16Array(data.buffer, data.byteOffset, 4)
      return [0, 1, 2].map((channel) => Math.round(Math.min(1, Math.max(0, halfToFloat(halves[channel]))) * 255)) as Rgb
    })
  }

  destroy() {
    if (this.destroyed) {
      return
    }
    this.destroyed = true
    if (this.preview) {
      this.destroyResources(this.preview)
      this.preview = undefined
    }
    this.dummyTexture.destroy()
    if (this.contextConfigured) {
      this.context.unconfigure()
    }
    this.device.destroy()
  }

  // ---------------------------------------------------------------------------
  // Queue and error handling.

  private enqueue<T>(task: () => Promise<T>): Promise<T> {
    this.queuedTasks += 1
    if (this.queuedTasks === 1) {
      this.onBusyChange?.(true)
    }
    const run = this.chain
      .then(
        () => task(),
        () => task(),
      )
      .finally(() => {
        this.queuedTasks -= 1
        if (this.queuedTasks === 0) {
          this.onBusyChange?.(false)
        }
      })
    this.chain = run.catch(() => undefined)
    return run
  }

  private async withErrorScope<T>(work: () => Promise<T>): Promise<T> {
    this.device.pushErrorScope('out-of-memory')
    this.device.pushErrorScope('validation')
    let result: T | undefined
    let failure: unknown
    try {
      result = await work()
    } catch (reason) {
      failure = reason
    }
    const validation = await this.device.popErrorScope()
    const outOfMemory = await this.device.popErrorScope()
    if (failure) {
      throw failure
    }
    if (validation) {
      throw new Error(validation.message)
    }
    if (outOfMemory) {
      throw new Error(`The GPU ran out of memory: ${outOfMemory.message}`)
    }
    return result as T
  }

  private requirePreview() {
    if (!this.preview) {
      throw new Error('No image is loaded.')
    }
    return this.preview
  }

  // ---------------------------------------------------------------------------
  // Preview.

  private renderPreview(state: RenderState) {
    return this.withErrorScope(async (): Promise<PreviewInfo | undefined> => {
      const resources = this.preview
      if (!resources || this.destroyed) {
        return undefined
      }

      const started = performance.now()
      const filtered = await this.prepareFiltered(resources, state.filter, state.params, state.draft, 1)
      const overlay = this.prepareOverlay(resources, state.view, 1)
      const crop = computeCrop(state.view, resources.width, resources.height)
      if (!this.contextConfigured) {
        this.context.configure({ device: this.device, format: this.canvasFormat, alphaMode: 'opaque' })
        this.contextConfigured = true
      }
      if (this.canvas.width !== crop.width || this.canvas.height !== crop.height) {
        this.canvas.width = crop.width
        this.canvas.height = crop.height
      }

      const encoder = this.device.createCommandEncoder()
      const buffer = this.composite(
        encoder,
        this.context.getCurrentTexture().createView(),
        this.canvasComposite,
        resources,
        filtered.texture,
        overlay,
        state.view,
        { crop, split: state.split, includeViewAids: true, lineWidth: 1.5 },
      )
      this.device.queue.submit([encoder.finish()])
      buffer.destroy()

      return {
        width: crop.width,
        height: crop.height,
        ms: performance.now() - started,
        draft: filtered.draft,
        palette: filtered.palette,
      }
    })
  }

  // ---------------------------------------------------------------------------
  // Image resources.

  private createResources(slot: ImageSlot, image: DecodedPixels): ImageResources {
    const { width, height, pixels } = image
    const source = this.device.createTexture({
      label: `${slot} source`,
      size: [width, height],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC,
    })
    this.device.queue.writeTexture({ texture: source }, pixels, { bytesPerRow: width * 4, rowsPerImage: height }, [width, height])
    const version = ++this.versionCounter
    this.cpu.setImage(slot, version, width, height, pixels)
    return { slot, version, width, height, pixels, source, scratch: new Map() }
  }

  private destroyResources(resources: ImageResources) {
    resources.source.destroy()
    resources.cpuTexture?.destroy()
    for (const texture of resources.scratch.values()) {
      texture.destroy()
    }
    resources.scratch.clear()
    if (resources.slot === 'export') {
      this.cpu.dropImage('export')
    }
  }

  private scratch(resources: ImageResources, name: string) {
    let texture = resources.scratch.get(name)
    if (!texture) {
      texture = this.device.createTexture({
        label: `${resources.slot} ${name}`,
        size: [resources.width, resources.height],
        format: SCRATCH_FORMAT,
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
      })
      resources.scratch.set(name, texture)
    }
    return texture
  }

  // ---------------------------------------------------------------------------
  // Filter stage.

  private async prepareFiltered(
    resources: ImageResources,
    filter: FilterId,
    params: ParamValues,
    draft: boolean,
    scale: number,
  ): Promise<FilteredResult> {
    const definition = getFilter(filter)
    const useDraft = draft && definition.engine === 'cpu'
    const renderScale = useDraft ? scale * 0.5 : scale
    const scaled = scaleParams(filter, params, renderScale)
    const key = `${fullKey(filter, scaled)}|draft=${useDraft}`
    if (resources.filtered?.key === key) {
      return resources.filtered
    }
    // A full-resolution result for the same settings beats a fresh draft
    // (e.g. a slider was pressed but not moved).
    if (useDraft && resources.filtered?.key === `${fullKey(filter, scaleParams(filter, params, scale))}|draft=false`) {
      return resources.filtered
    }

    let result: FilteredResult
    if (definition.engine === 'none') {
      result = { key, texture: resources.source, draft: false }
    } else if (definition.engine === 'gpu') {
      const batch = this.beginBatch()
      const texture = this.encodeGpuFilter(batch, resources, filter, scaled, scale)
      this.submitBatch(batch)
      result = { key, texture, draft: false }
    } else {
      const labels = definition.engine === 'slic' ? await this.slicLabels(resources, scaled) : undefined
      const job = await this.cpu.run({ slot: resources.slot, draft: useDraft, scale: renderScale, filter, params: scaled, labels })
      result = { key, texture: this.uploadCpuResult(resources, job), palette: job.palette, draft: useDraft }
    }

    resources.filtered = result
    return result
  }

  private uploadCpuResult(resources: ImageResources, job: CpuJobResult) {
    let texture = resources.cpuTexture
    if (!texture || texture.width !== job.width || texture.height !== job.height) {
      texture?.destroy()
      texture = this.device.createTexture({
        label: `${resources.slot} cpu result`,
        size: [job.width, job.height],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC,
      })
      resources.cpuTexture = texture
    }
    this.device.queue.writeTexture(
      { texture },
      new Uint8Array(job.pixels),
      { bytesPerRow: job.width * 4, rowsPerImage: job.height },
      [job.width, job.height],
    )
    return texture
  }

  // Records the passes for a GPU filter; the result always lands in 'out'.
  private encodeGpuFilter(batch: PassBatch, resources: ImageResources, filter: FilterId, params: ParamValues, scale: number) {
    const source = resources.source
    const out = this.scratch(resources, 'out')

    switch (filter) {
      case 'value': {
        const squinted = this.blur(batch, resources, source, num(params, 'squint') / 2, 'a')
        this.pass(batch, 'value', out, [squinted], [num(params, 'levels'), num(params, 'contrast'), str(params, 'output') === 'color' ? 1 : 0])
        break
      }
      case 'edges': {
        const lines = this.encodeLines(batch, resources, {
          detail: num(params, 'detail'),
          sensitivity: num(params, 'sensitivity'),
          smoothing: num(params, 'smoothing'),
          sharpness: num(params, 'sharpness'),
          target: 'b',
        })
        this.pass(batch, 'edges', out, [source, lines], [num(params, 'strength'), num(params, 'contrast'), bool(params, 'linesOnly') ? 1 : 0])
        break
      }
      case 'chroma':
        this.pass(batch, 'chroma', out, [source], [num(params, 'gain'), str(params, 'display') === 'gray' ? 1 : 0])
        break
      case 'temperature':
        this.pass(batch, 'temperature', out, [source], [num(params, 'gain'), num(params, 'neutral'), bool(params, 'keepValue') ? 1 : 0])
        break
      case 'flowBrush': {
        const tfm = this.tangentField(batch, resources, source, false, num(params, 'smoothing'), 'tfm')
        const strokeSource = this.blur(batch, resources, source, num(params, 'width') * 0.35, 'a')
        this.pass(
          batch,
          'flowBrush',
          out,
          [strokeSource, tfm, source],
          [
            num(params, 'length'),
            num(params, 'width'),
            num(params, 'smoothing'),
            num(params, 'colorMerge'),
            num(params, 'edgePreserve'),
            num(params, 'edgeThreshold'),
            num(params, 'strength'),
            num(params, 'contrast'),
          ],
        )
        break
      }
      case 'kuwahara': {
        const tfm = this.tangentField(batch, resources, source, false, num(params, 'smoothing'), 'tfm')
        const alpha = Math.pow(4, 1 - 2 * num(params, 'elongation'))
        const passes = Math.max(1, Math.min(3, Math.round(num(params, 'passes'))))
        let input = source
        for (let index = 0; index < passes; index += 1) {
          const target = index === passes - 1 ? out : this.scratch(resources, index % 2 === 0 ? 'a' : 'b')
          this.pass(batch, 'akf', target, [input, tfm], [num(params, 'radius'), num(params, 'sharpness'), alpha])
          input = target
        }
        break
      }
      case 'lostFound': {
        const tfm = this.tangentField(batch, resources, source, false, 2 * scale, 'tfm')
        const base = this.scratch(resources, 'a')
        this.pass(batch, 'akf', base, [source, tfm], [num(params, 'radius'), 8, 1])
        const lost = this.blur(batch, resources, base, num(params, 'lost'), 'b')
        // Edge importance comes from the simplified base, so surface texture
        // (knit, foliage, grain) does not register as a found edge.
        const valueField = this.tangentField(batch, resources, base, true, num(params, 'scale'), 'tfmL')
        this.pass(
          batch,
          'lostFound',
          out,
          [base, lost, valueField, source],
          [
            num(params, 'found'),
            num(params, 'accentThreshold'),
            num(params, 'accents'),
            bool(params, 'showMask') ? 1 : 0,
            // The Gaussian's visible reach is about two sigma.
            num(params, 'lost') * 2,
          ],
        )
        break
      }
      default:
        throw new Error(`No GPU implementation for filter "${filter}".`)
    }

    return out
  }

  // Edge overlay for any filter, cached on its settings.
  private prepareOverlay(resources: ImageResources, view: ViewSettings, scale: number) {
    if (!view.overlay || view.overlayStrength <= 0) {
      return undefined
    }
    const key = `${view.overlayDetail}|${view.overlaySensitivity}|${scale}`
    if (resources.overlayKey !== key) {
      const batch = this.beginBatch()
      this.encodeLines(batch, resources, {
        detail: view.overlayDetail * scale,
        sensitivity: view.overlaySensitivity,
        smoothing: 3 * scale,
        sharpness: 0.5,
        target: 'overlay',
      })
      this.submitBatch(batch)
      resources.overlayKey = key
    }
    return this.scratch(resources, 'overlay')
  }

  // Flow-based DoG line drawing (Kang et al. / Winnemöller XDoG threshold).
  private encodeLines(
    batch: PassBatch,
    resources: ImageResources,
    options: { detail: number; sensitivity: number; smoothing: number; sharpness: number; target: string },
  ) {
    const tfm = this.tangentField(batch, resources, resources.source, false, options.smoothing, 'tfm')
    const dog = this.scratch(resources, 'a')
    const tau = 0.94 + 0.06 * Math.min(1, Math.max(0, options.sensitivity))
    this.pass(batch, 'dog', dog, [resources.source, tfm], [options.detail, tau])
    const lines = this.scratch(resources, options.target)
    this.pass(batch, 'lic', lines, [dog, tfm], [options.smoothing, 30 + 370 * options.sharpness])
    return lines
  }

  private tangentField(batch: PassBatch, resources: ImageResources, input: GPUTexture, lightness: boolean, sigma: number, outName: string) {
    const tensor = this.scratch(resources, 'tensor')
    this.pass(batch, 'tensor', tensor, [input], [lightness ? 1 : 0])
    const smoothed = this.blur(batch, resources, tensor, sigma, 'tensorBlur')
    const tfm = this.scratch(resources, outName)
    this.pass(batch, 'tfm', tfm, [smoothed])
    return tfm
  }

  // Separable Gaussian into `outName` (via the shared 'tmp' texture). Returns
  // the input unchanged for negligible sigmas.
  private blur(batch: PassBatch, resources: ImageResources, input: GPUTexture, sigma: number, outName: string) {
    if (sigma < 0.3) {
      return input
    }
    const temp = this.scratch(resources, 'tmp')
    const out = this.scratch(resources, outName)
    this.pass(batch, 'blur', temp, [input], [1, 0, sigma])
    this.pass(batch, 'blur', out, [temp], [0, 1, sigma])
    return out
  }

  // ---------------------------------------------------------------------------
  // Pass recording.

  private beginBatch(): PassBatch {
    return { encoder: this.device.createCommandEncoder(), buffers: [] }
  }

  private submitBatch(batch: PassBatch) {
    this.device.queue.submit([batch.encoder.finish()])
    // Destruction is deferred by the implementation until submitted work
    // using the buffers completes.
    for (const buffer of batch.buffers) {
      buffer.destroy()
    }
    batch.buffers = []
  }

  private uniformBuffer(values: number[]) {
    const buffer = this.device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM, mappedAtCreation: true })
    new Float32Array(buffer.getMappedRange()).set(values.slice(0, 16))
    buffer.unmap()
    return buffer
  }

  private bindGroup(inputs: Array<GPUTexture | undefined>, buffer: GPUBuffer) {
    const view = (texture: GPUTexture | undefined) => (texture ?? this.dummyTexture).createView()
    return this.device.createBindGroup({
      layout: this.passLayout,
      entries: [
        { binding: 0, resource: view(inputs[0]) },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: { buffer } },
        { binding: 3, resource: view(inputs[1]) },
        { binding: 4, resource: view(inputs[2]) },
        { binding: 5, resource: view(inputs[3]) },
      ],
    })
  }

  private pass(batch: PassBatch, name: PassName, target: GPUTexture, inputs: Array<GPUTexture | undefined>, params: number[] = []) {
    const pipeline = this.passPipelines.get(name)
    if (!pipeline) {
      throw new Error(`Missing pipeline for pass "${name}".`)
    }
    const buffer = this.uniformBuffer(params)
    batch.buffers.push(buffer)
    const bindGroup = this.bindGroup(inputs, buffer)
    const view = target.createView()
    const tiled = HEAVY_PASSES.has(name) && target.width * target.height > TILE_PIXEL_THRESHOLD

    if (!tiled) {
      const pass = batch.encoder.beginRenderPass({
        colorAttachments: [{ view, loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 1 }, storeOp: 'store' }],
      })
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, bindGroup)
      pass.draw(3)
      pass.end()
      return
    }

    this.device.queue.submit([batch.encoder.finish()])
    for (let y = 0; y < target.height; y += TILE_SIZE) {
      for (let x = 0; x < target.width; x += TILE_SIZE) {
        const encoder = this.device.createCommandEncoder()
        const pass = encoder.beginRenderPass({ colorAttachments: [{ view, loadOp: 'load', storeOp: 'store' }] })
        pass.setPipeline(pipeline)
        pass.setBindGroup(0, bindGroup)
        pass.setScissorRect(x, y, Math.min(TILE_SIZE, target.width - x), Math.min(TILE_SIZE, target.height - y))
        pass.draw(3)
        pass.end()
        this.device.queue.submit([encoder.finish()])
      }
    }
    batch.encoder = this.device.createCommandEncoder()
  }

  private composite(
    encoder: GPUCommandEncoder,
    targetView: GPUTextureView,
    pipeline: GPURenderPipeline,
    resources: ImageResources,
    filtered: GPUTexture,
    overlay: GPUTexture | undefined,
    view: ViewSettings,
    options: {
      crop: ReturnType<typeof computeCrop>
      split: { enabled: boolean; position: number }
      includeViewAids: boolean
      lineWidth: number
    },
  ) {
    const { crop, split, includeViewAids, lineWidth } = options
    const buffer = this.uniformBuffer([
      crop.offsetX,
      crop.offsetY,
      crop.scaleX,
      crop.scaleY,
      includeViewAids && view.mirror ? 1 : 0,
      view.grayscale ? 1 : 0,
      split.enabled ? 1 : 0,
      split.position,
      includeViewAids ? view.gridDivisions : 0,
      includeViewAids && view.gridDiagonals ? 1 : 0,
      overlay ? view.overlayStrength : 0,
      lineWidth,
      crop.width,
      crop.height,
      0,
      0,
    ])
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: targetView, loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 1 }, storeOp: 'store' }],
    })
    pass.setPipeline(pipeline)
    pass.setBindGroup(0, this.bindGroup([filtered, resources.source, overlay], buffer))
    pass.draw(3)
    pass.end()
    return buffer
  }

  // ---------------------------------------------------------------------------
  // SLIC clustering on the GPU; returns per-pixel labels.

  private async slicLabels(resources: ImageResources, params: ParamValues) {
    const key = analysisKey('slic', params)
    if (resources.slicLabels?.key === key) {
      return resources.slicLabels.labels
    }

    const { width, height, pixels } = resources
    const step = Math.max(4, Math.round(num(params, 'size')))
    const compactness = Math.max(1, num(params, 'compactness'))
    const gridWidth = Math.max(1, Math.ceil(width / step))
    const gridHeight = Math.max(1, Math.ceil(height / step))
    const centerCount = gridWidth * gridHeight
    const centers = new Float32Array(centerCount * 8)
    for (let centerY = 0; centerY < gridHeight; centerY += 1) {
      for (let centerX = 0; centerX < gridWidth; centerX += 1) {
        const x = Math.min(width - 1, Math.round(centerX * step + step * 0.5))
        const y = Math.min(height - 1, Math.round(centerY * step + step * 0.5))
        const pixelOffset = (x + y * width) * 4
        const [l, a, b] = srgbToOklab(pixels[pixelOffset], pixels[pixelOffset + 1], pixels[pixelOffset + 2])
        centers.set([x, y, l * 100, a * 100, b * 100], (centerX + centerY * gridWidth) * 8)
      }
    }

    const device = this.device
    const centerBuffer = device.createBuffer({ size: centers.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST })
    const labelBuffer = device.createBuffer({ size: width * height * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC })
    const sumBuffer = device.createBuffer({ size: centerCount * 6 * 4, usage: GPUBufferUsage.STORAGE })
    const readBuffer = device.createBuffer({ size: width * height * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
    const paramsData = new ArrayBuffer(32)
    new Uint32Array(paramsData, 0, 4).set([width, height, gridWidth, centerCount])
    new Float32Array(paramsData, 16, 2).set([step, compactness])
    const paramsBuffer = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
    device.queue.writeBuffer(centerBuffer, 0, centers)
    device.queue.writeBuffer(paramsBuffer, 0, paramsData)

    const bindGroup = device.createBindGroup({
      layout: this.slic.layout,
      entries: [
        { binding: 0, resource: resources.source.createView() },
        { binding: 1, resource: { buffer: centerBuffer } },
        { binding: 2, resource: { buffer: labelBuffer } },
        { binding: 3, resource: { buffer: sumBuffer } },
        { binding: 4, resource: { buffer: paramsBuffer } },
      ],
    })

    const pixelGroupsX = Math.ceil(width / 16)
    const pixelGroupsY = Math.ceil(height / 16)
    const dispatch = (encoder: GPUCommandEncoder, pipeline: GPUComputePipeline, x: number, y = 1) => {
      const pass = encoder.beginComputePass()
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, bindGroup)
      pass.dispatchWorkgroups(x, y)
      pass.end()
    }

    const encoder = device.createCommandEncoder()
    for (let iteration = 0; iteration < 8; iteration += 1) {
      dispatch(encoder, this.slic.clearSums, Math.ceil((centerCount * 6) / 64))
      dispatch(encoder, this.slic.assignPixels, pixelGroupsX, pixelGroupsY)
      dispatch(encoder, this.slic.accumulateCenters, pixelGroupsX, pixelGroupsY)
      dispatch(encoder, this.slic.updateCenters, Math.ceil(centerCount / 64))
    }
    // Final assignment against the converged centers.
    dispatch(encoder, this.slic.assignPixels, pixelGroupsX, pixelGroupsY)
    encoder.copyBufferToBuffer(labelBuffer, 0, readBuffer, 0, width * height * 4)
    device.queue.submit([encoder.finish()])

    try {
      await readBuffer.mapAsync(GPUMapMode.READ)
      const labels = new Int32Array(readBuffer.getMappedRange().slice(0))
      readBuffer.unmap()
      resources.slicLabels = { key, labels }
      return labels
    } finally {
      centerBuffer.destroy()
      labelBuffer.destroy()
      sumBuffer.destroy()
      readBuffer.destroy()
      paramsBuffer.destroy()
    }
  }

  // ---------------------------------------------------------------------------
  // Readback.

  private async readTexture(texture: GPUTexture, x: number, y: number, width: number, height: number, bytesPerPixel: number) {
    const rowBytes = width * bytesPerPixel
    const bytesPerRow = Math.ceil(rowBytes / 256) * 256
    const buffer = this.device.createBuffer({ size: bytesPerRow * height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
    const encoder = this.device.createCommandEncoder()
    encoder.copyTextureToBuffer({ texture, origin: { x, y } }, { buffer, bytesPerRow, rowsPerImage: height }, [width, height])
    this.device.queue.submit([encoder.finish()])
    try {
      await buffer.mapAsync(GPUMapMode.READ)
      const mapped = new Uint8Array(buffer.getMappedRange())
      const out = new Uint8Array(rowBytes * height)
      for (let row = 0; row < height; row += 1) {
        out.set(mapped.subarray(row * bytesPerRow, row * bytesPerRow + rowBytes), row * rowBytes)
      }
      buffer.unmap()
      return out
    } finally {
      buffer.destroy()
    }
  }
}

export function halfToFloat(half: number) {
  const sign = half & 0x8000 ? -1 : 1
  const exponent = (half >> 10) & 0x1f
  const fraction = half & 0x3ff
  if (exponent === 0) {
    return sign * 2 ** -14 * (fraction / 1024)
  }
  if (exponent === 31) {
    return fraction ? Number.NaN : sign * Number.POSITIVE_INFINITY
  }
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024)
}
