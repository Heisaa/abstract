export type FilterId = 'original' | 'kuwahara' | 'posterize' | 'edges' | 'slic' | 'felzenszwalb' | 'adaptiveBrush'

export type FilterSettings = {
  filter: FilterId
  radius: number
  levels: number
  grayscale: boolean
  contrast: number
  edgeThreshold: number
  edgeStrength: number
  slicSize: number
  slicCompactness: number
  felzenszwalbScale: number
  felzenszwalbMinSize: number
  adaptiveLargeBrush: number
  adaptiveMediumBrush: number
  adaptiveSmallBrush: number
  adaptiveDetailThreshold: number
  adaptiveDetailStrength: number
}

export type LoadedImage = {
  bitmap: ImageBitmap
  name: string
  width: number
  height: number
  pixels: Uint8ClampedArray
}

type SlicResources = {
  initLabels: GPUComputePipeline
  clearSums: GPUComputePipeline
  assignPixels: GPUComputePipeline
  accumulateCenters: GPUComputePipeline
  updateCenters: GPUComputePipeline
  colorize: GPUComputePipeline
}

type FelzenszwalbResources = {
  computeWeights: GPUComputePipeline
}

const FILTER_INDEX: Record<FilterId, number> = {
  original: 0,
  kuwahara: 1,
  posterize: 2,
  edges: 3,
  slic: 4,
  felzenszwalb: 5,
  adaptiveBrush: 6,
}

const MAX_WORKING_IMAGE_DIMENSION = 1280

const SHADER = /* wgsl */ `
struct Uniforms {
  imageSize: vec2f,
  radius: f32,
  filterMode: f32,
  levels: f32,
  grayscale: f32,
  contrast: f32,
  edgeThreshold: f32,
  edgeStrength: f32,
  slicSize: f32,
  pad1: f32,
  pad2: f32,
};

struct VertexOut {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

@group(0) @binding(0) var imageTexture: texture_2d<f32>;
@group(0) @binding(1) var imageSampler: sampler;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;
@group(0) @binding(3) var segmentTexture: texture_2d<f32>;

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOut {
  var positions = array<vec2f, 3>(
    vec2f(-1.0, -1.0),
    vec2f(3.0, -1.0),
    vec2f(-1.0, 3.0)
  );

  var out: VertexOut;
  let pos = positions[vertexIndex];
  out.position = vec4f(pos, 0.0, 1.0);
  out.uv = vec2f(pos.x * 0.5 + 0.5, 0.5 - pos.y * 0.5);
  return out;
}

fn sampleImage(uv: vec2f) -> vec3f {
  let clampedUv = clamp(uv, vec2f(0.0), vec2f(1.0));
  return textureSampleLevel(imageTexture, imageSampler, clampedUv, 0.0).rgb;
}

fn luminance(color: vec3f) -> f32 {
  return dot(color, vec3f(0.299, 0.587, 0.114));
}

fn applyContrast(color: vec3f, contrast: f32) -> vec3f {
  return clamp((color - vec3f(0.5)) * contrast + vec3f(0.5), vec3f(0.0), vec3f(1.0));
}

fn posterize(uv: vec2f) -> vec3f {
  var color = applyContrast(sampleImage(uv), uniforms.contrast);
  if (uniforms.grayscale > 0.5) {
    let value = luminance(color);
    color = vec3f(value);
  }

  let levels = max(uniforms.levels, 2.0);
  return floor(color * levels) / (levels - 1.0);
}

fn kuwahara(uv: vec2f) -> vec3f {
  let texel = 1.0 / uniforms.imageSize;
  let radius = i32(clamp(uniforms.radius, 1.0, 16.0));
  var bestMean = sampleImage(uv);
  var bestVariance = 99999.0;

  for (var quadrant = 0; quadrant < 4; quadrant = quadrant + 1) {
    var signX = -1.0;
    var signY = -1.0;
    if (quadrant == 1) {
      signX = 1.0;
    } else if (quadrant == 2) {
      signY = 1.0;
    } else if (quadrant == 3) {
      signX = 1.0;
      signY = 1.0;
    }

    var sum = vec3f(0.0);
    var sumSq = vec3f(0.0);
    var count = 0.0;

    for (var y = 0; y <= radius; y = y + 1) {
      for (var x = 0; x <= radius; x = x + 1) {
        let offset = vec2f(f32(x) * signX, f32(y) * signY);
        let color = sampleImage(uv + offset * texel);
        sum = sum + color;
        sumSq = sumSq + color * color;
        count = count + 1.0;
      }
    }

    let mean = sum / count;
    let variance = sumSq / count - mean * mean;
    let totalVariance = variance.r + variance.g + variance.b;
    if (totalVariance < bestVariance) {
      bestVariance = totalVariance;
      bestMean = mean;
    }
  }

  return bestMean;
}

fn edges(uv: vec2f) -> vec3f {
  let texel = 1.0 / uniforms.imageSize;
  let c00 = luminance(sampleImage(uv + vec2f(-1.0, -1.0) * texel));
  let c10 = luminance(sampleImage(uv + vec2f(0.0, -1.0) * texel));
  let c20 = luminance(sampleImage(uv + vec2f(1.0, -1.0) * texel));
  let c01 = luminance(sampleImage(uv + vec2f(-1.0, 0.0) * texel));
  let c21 = luminance(sampleImage(uv + vec2f(1.0, 0.0) * texel));
  let c02 = luminance(sampleImage(uv + vec2f(-1.0, 1.0) * texel));
  let c12 = luminance(sampleImage(uv + vec2f(0.0, 1.0) * texel));
  let c22 = luminance(sampleImage(uv + vec2f(1.0, 1.0) * texel));

  let gx = -c00 - 2.0 * c01 - c02 + c20 + 2.0 * c21 + c22;
  let gy = -c00 - 2.0 * c10 - c20 + c02 + 2.0 * c12 + c22;
  let edge = smoothstep(uniforms.edgeThreshold, uniforms.edgeThreshold + 0.18, length(vec2f(gx, gy)));
  let color = applyContrast(sampleImage(uv), uniforms.contrast);
  return mix(color, color * (1.0 - uniforms.edgeStrength), edge);
}

fn sampleSegments(uv: vec2f) -> vec3f {
  let pixel = vec2u(clamp(uv * uniforms.imageSize, vec2f(0.0), uniforms.imageSize - vec2f(1.0)));
  return textureLoad(segmentTexture, pixel, 0).rgb;
}

@fragment
fn fragmentMain(input: VertexOut) -> @location(0) vec4f {
  var color = sampleImage(input.uv);

  if (uniforms.filterMode > 0.5 && uniforms.filterMode < 1.5) {
    color = kuwahara(input.uv);
  } else if (uniforms.filterMode > 1.5 && uniforms.filterMode < 2.5) {
    color = posterize(input.uv);
  } else if (uniforms.filterMode > 2.5 && uniforms.filterMode < 3.5) {
    color = edges(input.uv);
  } else if (uniforms.filterMode > 3.5 && uniforms.filterMode < 6.5) {
    color = sampleSegments(input.uv);
  }

  return vec4f(clamp(color, vec3f(0.0), vec3f(1.0)), 1.0);
}
`

const SLIC_SHADER = /* wgsl */ `
struct Params {
  imageSize: vec2u,
  gridWidth: u32,
  centerCount: u32,
  step: f32,
  compactness: f32,
  pad0: f32,
  pad1: f32,
};

struct Center {
  x: f32,
  y: f32,
  r: f32,
  g: f32,
  b: f32,
  pad0: f32,
  pad1: f32,
  pad2: f32,
};

@group(0) @binding(0) var imageTexture: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> centers: array<Center>;
@group(0) @binding(2) var<storage, read_write> labels: array<u32>;
@group(0) @binding(3) var<storage, read_write> distances: array<f32>;
@group(0) @binding(4) var<storage, read_write> sums: array<atomic<u32>>;
@group(0) @binding(5) var outputTexture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(6) var<uniform> params: Params;

fn pixelIndex(x: u32, y: u32) -> u32 {
  return x + y * params.imageSize.x;
}

fn centerIndex(cellX: i32, cellY: i32) -> u32 {
  return u32(cellX) + u32(cellY) * params.gridWidth;
}

@compute @workgroup_size(16, 16)
fn initLabels(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= params.imageSize.x || id.y >= params.imageSize.y) {
    return;
  }

  let index = pixelIndex(id.x, id.y);
  labels[index] = 0u;
  distances[index] = 3.402823e38;
}

@compute @workgroup_size(64)
fn clearSums(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= params.centerCount * 6u) {
    return;
  }

  atomicStore(&sums[id.x], 0u);
}

@compute @workgroup_size(16, 16)
fn assignPixels(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= params.imageSize.x || id.y >= params.imageSize.y) {
    return;
  }

  let index = pixelIndex(id.x, id.y);
  let color = textureLoad(imageTexture, vec2u(id.xy), 0).rgb * 255.0;
  let cellX = i32(floor(f32(id.x) / params.step));
  let cellY = i32(floor(f32(id.y) / params.step));
  let spatialWeight = (params.compactness * params.compactness) / (params.step * params.step);

  var bestLabel = 0u;
  var bestDistance = 3.402823e38;
  for (var y = cellY - 2; y <= cellY + 2; y = y + 1) {
    for (var x = cellX - 2; x <= cellX + 2; x = x + 1) {
      if (x < 0 || y < 0) {
        continue;
      }

      let label = centerIndex(x, y);
      if (label >= params.centerCount) {
        continue;
      }

      let center = centers[label];
      let dr = color.r - center.r;
      let dg = color.g - center.g;
      let db = color.b - center.b;
      let dx = f32(id.x) - center.x;
      let dy = f32(id.y) - center.y;
      let distance = dr * dr + dg * dg + db * db + (dx * dx + dy * dy) * spatialWeight;
      if (distance < bestDistance) {
        bestDistance = distance;
        bestLabel = label;
      }
    }
  }

  labels[index] = bestLabel;
  distances[index] = bestDistance;
}

@compute @workgroup_size(16, 16)
fn accumulateCenters(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= params.imageSize.x || id.y >= params.imageSize.y) {
    return;
  }

  let index = pixelIndex(id.x, id.y);
  let label = labels[index];
  let color = textureLoad(imageTexture, vec2u(id.xy), 0).rgb * 255.0;
  let offset = label * 6u;

  atomicAdd(&sums[offset], id.x);
  atomicAdd(&sums[offset + 1u], id.y);
  atomicAdd(&sums[offset + 2u], u32(round(color.r)));
  atomicAdd(&sums[offset + 3u], u32(round(color.g)));
  atomicAdd(&sums[offset + 4u], u32(round(color.b)));
  atomicAdd(&sums[offset + 5u], 1u);
}

@compute @workgroup_size(64)
fn updateCenters(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= params.centerCount) {
    return;
  }

  let offset = id.x * 6u;
  let count = atomicLoad(&sums[offset + 5u]);
  if (count == 0u) {
    return;
  }

  let countF = f32(count);
  centers[id.x].x = f32(atomicLoad(&sums[offset])) / countF;
  centers[id.x].y = f32(atomicLoad(&sums[offset + 1u])) / countF;
  centers[id.x].r = f32(atomicLoad(&sums[offset + 2u])) / countF;
  centers[id.x].g = f32(atomicLoad(&sums[offset + 3u])) / countF;
  centers[id.x].b = f32(atomicLoad(&sums[offset + 4u])) / countF;
}

@compute @workgroup_size(16, 16)
fn colorize(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= params.imageSize.x || id.y >= params.imageSize.y) {
    return;
  }

  let index = pixelIndex(id.x, id.y);
  let center = centers[labels[index]];
  textureStore(outputTexture, vec2u(id.xy), vec4f(center.r, center.g, center.b, 255.0) / 255.0);
}
`

const FELZENSZWALB_SHADER = /* wgsl */ `
struct Params {
  imageSize: vec2u,
  horizontalOffset: u32,
  verticalOffset: u32,
  diagonalDownOffset: u32,
  diagonalUpOffset: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
};

@group(0) @binding(0) var imageTexture: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> weights: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

fn edgeWeight(a: vec3f, b: vec3f) -> f32 {
  let delta = (a - b) * 255.0;
  return length(delta);
}

@compute @workgroup_size(16, 16)
fn computeWeights(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= params.imageSize.x || id.y >= params.imageSize.y) {
    return;
  }

  let color = textureLoad(imageTexture, vec2u(id.xy), 0).rgb;
  let horizontalCount = params.imageSize.x - 1u;

  if (id.x + 1u < params.imageSize.x) {
    let right = textureLoad(imageTexture, vec2u(id.x + 1u, id.y), 0).rgb;
    weights[params.horizontalOffset + id.x + id.y * horizontalCount] = edgeWeight(color, right);
  }

  if (id.y + 1u < params.imageSize.y) {
    let down = textureLoad(imageTexture, vec2u(id.x, id.y + 1u), 0).rgb;
    weights[params.verticalOffset + id.x + id.y * params.imageSize.x] = edgeWeight(color, down);
  }

  if (id.x + 1u < params.imageSize.x && id.y + 1u < params.imageSize.y) {
    let diagonalDown = textureLoad(imageTexture, vec2u(id.x + 1u, id.y + 1u), 0).rgb;
    weights[params.diagonalDownOffset + id.x + id.y * horizontalCount] = edgeWeight(color, diagonalDown);
  }

  if (id.x + 1u < params.imageSize.x && id.y > 0u) {
    let diagonalUp = textureLoad(imageTexture, vec2u(id.x + 1u, id.y - 1u), 0).rgb;
    weights[params.diagonalUpOffset + id.x + (id.y - 1u) * horizontalCount] = edgeWeight(color, diagonalUp);
  }
}
`

export class WebGpuImageRenderer {
  private readonly canvas: HTMLCanvasElement
  private adapter?: GPUAdapter
  private device?: GPUDevice
  private context?: GPUCanvasContext
  private pipeline?: GPURenderPipeline
  private slicPipelines?: SlicResources
  private slicBindGroupLayout?: GPUBindGroupLayout
  private felzenszwalbPipelines?: FelzenszwalbResources
  private felzenszwalbBindGroupLayout?: GPUBindGroupLayout
  private bindGroup?: GPUBindGroup
  private texture?: GPUTexture
  private segmentTexture?: GPUTexture
  private sampler?: GPUSampler
  private uniformBuffer?: GPUBuffer
  private format?: GPUTextureFormat
  private onError?: (message: string) => void
  private image?: LoadedImage
  private segmentCacheKey = ''

  constructor(canvas: HTMLCanvasElement, onError?: (message: string) => void) {
    this.canvas = canvas
    this.onError = onError
  }

  async init() {
    if (!navigator.gpu) {
      throw new Error('WebGPU is not available in this browser.')
    }

    const adapter = await navigator.gpu.requestAdapter()
    if (!adapter) {
      throw new Error('No compatible GPU adapter was found.')
    }
    this.adapter = adapter

    this.device = await this.adapter.requestDevice()
    this.device.addEventListener('uncapturederror', (event) => {
      this.onError?.(event.error.message)
    })

    this.context = (this.canvas.getContext('webgpu') as GPUCanvasContext | null) ?? undefined
    if (!this.context) {
      throw new Error('Could not create a WebGPU canvas context.')
    }

    this.format = navigator.gpu.getPreferredCanvasFormat()
    this.configureContext()

    this.uniformBuffer = this.device.createBuffer({
      size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })

    this.sampler = this.device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    })

    const module = this.device.createShaderModule({ code: SHADER })
    const compilation = await module.getCompilationInfo()
    const shaderErrors = compilation.messages
      .filter((message) => message.type === 'error')
      .map((message) => message.message)
    if (shaderErrors.length > 0) {
      throw new Error(shaderErrors.join('\n'))
    }

    this.pipeline = await this.device.createRenderPipelineAsync({
      layout: 'auto',
      vertex: {
        module,
        entryPoint: 'vertexMain',
      },
      fragment: {
        module,
        entryPoint: 'fragmentMain',
        targets: [{ format: this.format }],
      },
      primitive: {
        topology: 'triangle-list',
      },
    })

    const slicModule = this.device.createShaderModule({ code: SLIC_SHADER })
    const slicCompilation = await slicModule.getCompilationInfo()
    const slicShaderErrors = slicCompilation.messages
      .filter((message) => message.type === 'error')
      .map((message) => message.message)
    if (slicShaderErrors.length > 0) {
      throw new Error(slicShaderErrors.join('\n'))
    }

    this.slicBindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        {
          binding: 5,
          visibility: GPUShaderStage.COMPUTE,
          storageTexture: { access: 'write-only', format: 'rgba8unorm' },
        },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    })
    const slicPipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts: [this.slicBindGroupLayout],
    })

    this.slicPipelines = {
      initLabels: await this.device.createComputePipelineAsync({
        layout: slicPipelineLayout,
        compute: { module: slicModule, entryPoint: 'initLabels' },
      }),
      clearSums: await this.device.createComputePipelineAsync({
        layout: slicPipelineLayout,
        compute: { module: slicModule, entryPoint: 'clearSums' },
      }),
      assignPixels: await this.device.createComputePipelineAsync({
        layout: slicPipelineLayout,
        compute: { module: slicModule, entryPoint: 'assignPixels' },
      }),
      accumulateCenters: await this.device.createComputePipelineAsync({
        layout: slicPipelineLayout,
        compute: { module: slicModule, entryPoint: 'accumulateCenters' },
      }),
      updateCenters: await this.device.createComputePipelineAsync({
        layout: slicPipelineLayout,
        compute: { module: slicModule, entryPoint: 'updateCenters' },
      }),
      colorize: await this.device.createComputePipelineAsync({
        layout: slicPipelineLayout,
        compute: { module: slicModule, entryPoint: 'colorize' },
      }),
    }

    const felzenszwalbModule = this.device.createShaderModule({ code: FELZENSZWALB_SHADER })
    const felzenszwalbCompilation = await felzenszwalbModule.getCompilationInfo()
    const felzenszwalbShaderErrors = felzenszwalbCompilation.messages
      .filter((message) => message.type === 'error')
      .map((message) => message.message)
    if (felzenszwalbShaderErrors.length > 0) {
      throw new Error(felzenszwalbShaderErrors.join('\n'))
    }

    this.felzenszwalbBindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    })
    const felzenszwalbPipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts: [this.felzenszwalbBindGroupLayout],
    })
    this.felzenszwalbPipelines = {
      computeWeights: await this.device.createComputePipelineAsync({
        layout: felzenszwalbPipelineLayout,
        compute: { module: felzenszwalbModule, entryPoint: 'computeWeights' },
      }),
    }
  }

  loadImage(image: LoadedImage) {
    const device = this.requireDevice()
    this.texture?.destroy()
    this.segmentTexture?.destroy()
    this.image = image
    this.segmentCacheKey = ''
    this.texture = device.createTexture({
      size: [image.width, image.height],
      format: 'rgba8unorm',
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT |
        GPUTextureUsage.STORAGE_BINDING,
    })

    device.queue.copyExternalImageToTexture(
      { source: image.bitmap },
      { texture: this.texture },
      [image.width, image.height],
    )

    this.segmentTexture = this.createTextureFromPixels(image.pixels, image.width, image.height)
    this.canvas.width = image.width
    this.canvas.height = image.height
    this.canvas.style.aspectRatio = `${image.width} / ${image.height}`
    this.configureContext()
    this.createBindGroup()
  }

  async render(settings: FilterSettings) {
    const device = this.requireDevice()
    if (!this.context || !this.pipeline || !this.bindGroup || !this.uniformBuffer) {
      throw new Error('The image renderer is not ready yet.')
    }

    device.pushErrorScope('validation')
    await this.updateSegments(settings)
    this.writeUniforms(settings)
    const encoder = device.createCommandEncoder()
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.context.getCurrentTexture().createView(),
          clearValue: { r: 0.07, g: 0.07, b: 0.07, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    })

    pass.setPipeline(this.pipeline)
    pass.setBindGroup(0, this.bindGroup)
    pass.draw(3)
    pass.end()
    device.queue.submit([encoder.finish()])
    const validationError = await device.popErrorScope()
    if (validationError) {
      throw new Error(validationError.message)
    }
  }

  async exportBlob(type: 'image/png' | 'image/jpeg', quality?: number) {
    await this.device?.queue.onSubmittedWorkDone()

    return new Promise<Blob>((resolve, reject) => {
      this.canvas.toBlob(
        (blob) => {
          if (blob) {
            resolve(blob)
          } else {
            reject(new Error('Could not export the canvas image.'))
          }
        },
        type,
        quality,
      )
    })
  }

  private createBindGroup() {
    const device = this.requireDevice()
    if (!this.pipeline || !this.texture || !this.segmentTexture || !this.sampler || !this.uniformBuffer) {
      return
    }

    this.bindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.texture.createView() },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: { buffer: this.uniformBuffer } },
        { binding: 3, resource: this.segmentTexture.createView() },
      ],
    })
  }

  private createTextureFromPixels(pixels: Uint8ClampedArray, width: number, height: number) {
    const device = this.requireDevice()
    const texture = device.createTexture({
      size: [width, height],
      format: 'rgba8unorm',
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT |
        GPUTextureUsage.STORAGE_BINDING,
    })

    device.queue.writeTexture(
      { texture },
      pixels,
      { bytesPerRow: width * 4, rowsPerImage: height },
      [width, height],
    )

    return texture
  }

  private async updateSegments(settings: FilterSettings) {
    if (!this.image || (settings.filter !== 'slic' && settings.filter !== 'felzenszwalb' && settings.filter !== 'adaptiveBrush')) {
      return
    }

    const key = this.getSegmentCacheKey(settings)
    if (key === this.segmentCacheKey) {
      return
    }

    if (settings.filter === 'slic') {
      await this.createSlicSegmentsGpu(settings.slicSize, settings.slicCompactness)
    } else if (settings.filter === 'felzenszwalb') {
      const edgeWeights = await this.createFelzenszwalbWeightsGpu()
      const pixels = createFelzenszwalbSegments(
        this.image,
        settings.felzenszwalbScale,
        settings.felzenszwalbMinSize,
        edgeWeights,
      )
      this.segmentTexture?.destroy()
      this.segmentTexture = this.createTextureFromPixels(pixels, this.image.width, this.image.height)
    } else {
      const pixels = createAdaptiveBrushSegments(
        this.image,
        settings.adaptiveLargeBrush,
        settings.adaptiveMediumBrush,
        settings.adaptiveSmallBrush,
        settings.adaptiveDetailThreshold,
        settings.adaptiveDetailStrength,
      )
      this.segmentTexture?.destroy()
      this.segmentTexture = this.createTextureFromPixels(pixels, this.image.width, this.image.height)
    }

    this.segmentCacheKey = key
    this.createBindGroup()
  }

  private getSegmentCacheKey(settings: FilterSettings) {
    if (settings.filter === 'slic') {
      return `slic:${settings.slicSize}:${settings.slicCompactness}`
    }

    if (settings.filter === 'felzenszwalb') {
      return `felzenszwalb:${settings.felzenszwalbScale}:${settings.felzenszwalbMinSize}`
    }

    return [
      'adaptiveBrush',
      settings.adaptiveLargeBrush,
      settings.adaptiveMediumBrush,
      settings.adaptiveSmallBrush,
      settings.adaptiveDetailThreshold,
      settings.adaptiveDetailStrength,
    ].join(':')
  }

  private async createSlicSegmentsGpu(requestedSize: number, requestedCompactness: number) {
    const device = this.requireDevice()
    if (!this.image || !this.texture || !this.segmentTexture || !this.slicPipelines || !this.slicBindGroupLayout) {
      return
    }

    const { width, height, pixels } = this.image
    const step = Math.max(8, Math.round(requestedSize))
    const compactness = Math.max(1, requestedCompactness)
    const gridWidth = Math.max(1, Math.ceil(width / step))
    const gridHeight = Math.max(1, Math.ceil(height / step))
    const centerCount = gridWidth * gridHeight
    const centers = new Float32Array(centerCount * 8)

    for (let centerY = 0; centerY < gridHeight; centerY += 1) {
      for (let centerX = 0; centerX < gridWidth; centerX += 1) {
        const centerIndex = centerX + centerY * gridWidth
        const x = Math.min(width - 1, Math.round(centerX * step + step * 0.5))
        const y = Math.min(height - 1, Math.round(centerY * step + step * 0.5))
        const pixelOffset = (x + y * width) * 4
        const offset = centerIndex * 8
        centers[offset] = x
        centers[offset + 1] = y
        centers[offset + 2] = pixels[pixelOffset]
        centers[offset + 3] = pixels[pixelOffset + 1]
        centers[offset + 4] = pixels[pixelOffset + 2]
      }
    }

    const centerBuffer = device.createBuffer({
      size: centers.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
    const labelBuffer = device.createBuffer({
      size: width * height * 4,
      usage: GPUBufferUsage.STORAGE,
    })
    const distanceBuffer = device.createBuffer({
      size: width * height * 4,
      usage: GPUBufferUsage.STORAGE,
    })
    const sumBuffer = device.createBuffer({
      size: centerCount * 6 * 4,
      usage: GPUBufferUsage.STORAGE,
    })
    const paramsBuffer = device.createBuffer({
      size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })

    const paramsData = new ArrayBuffer(48)
    const paramsU32 = new Uint32Array(paramsData)
    const paramsF32 = new Float32Array(paramsData)
    paramsU32[0] = width
    paramsU32[1] = height
    paramsU32[2] = gridWidth
    paramsU32[3] = centerCount
    paramsF32[4] = step
    paramsF32[5] = compactness

    device.queue.writeBuffer(centerBuffer, 0, centers)
    device.queue.writeBuffer(paramsBuffer, 0, paramsData)

    const bindGroup = device.createBindGroup({
      layout: this.slicBindGroupLayout,
      entries: [
        { binding: 0, resource: this.texture.createView() },
        { binding: 1, resource: { buffer: centerBuffer } },
        { binding: 2, resource: { buffer: labelBuffer } },
        { binding: 3, resource: { buffer: distanceBuffer } },
        { binding: 4, resource: { buffer: sumBuffer } },
        { binding: 5, resource: this.segmentTexture.createView() },
        { binding: 6, resource: { buffer: paramsBuffer } },
      ],
    })

    const pixelWorkgroupsX = Math.ceil(width / 16)
    const pixelWorkgroupsY = Math.ceil(height / 16)
    const centerWorkgroups = Math.ceil(centerCount / 64)
    const sumWorkgroups = Math.ceil((centerCount * 6) / 64)
    const encoder = device.createCommandEncoder()

    for (let iteration = 0; iteration < 6; iteration += 1) {
      this.dispatchSlicPass(encoder, this.slicPipelines.initLabels, bindGroup, pixelWorkgroupsX, pixelWorkgroupsY)
      this.dispatchSlicPass(encoder, this.slicPipelines.clearSums, bindGroup, sumWorkgroups, 1)
      this.dispatchSlicPass(encoder, this.slicPipelines.assignPixels, bindGroup, pixelWorkgroupsX, pixelWorkgroupsY)
      this.dispatchSlicPass(encoder, this.slicPipelines.accumulateCenters, bindGroup, pixelWorkgroupsX, pixelWorkgroupsY)
      this.dispatchSlicPass(encoder, this.slicPipelines.updateCenters, bindGroup, centerWorkgroups, 1)
    }

    this.dispatchSlicPass(encoder, this.slicPipelines.colorize, bindGroup, pixelWorkgroupsX, pixelWorkgroupsY)
    device.queue.submit([encoder.finish()])
    await device.queue.onSubmittedWorkDone()

    centerBuffer.destroy()
    labelBuffer.destroy()
    distanceBuffer.destroy()
    sumBuffer.destroy()
    paramsBuffer.destroy()
  }

  private dispatchSlicPass(
    encoder: GPUCommandEncoder,
    pipeline: GPUComputePipeline,
    bindGroup: GPUBindGroup,
    workgroupsX: number,
    workgroupsY: number,
  ) {
    const pass = encoder.beginComputePass()
    pass.setPipeline(pipeline)
    pass.setBindGroup(0, bindGroup)
    pass.dispatchWorkgroups(workgroupsX, workgroupsY)
    pass.end()
  }

  private async createFelzenszwalbWeightsGpu() {
    const device = this.requireDevice()
    if (!this.image || !this.texture || !this.felzenszwalbPipelines || !this.felzenszwalbBindGroupLayout) {
      return undefined
    }

    const { width, height } = this.image
    const horizontalCount = Math.max(0, (width - 1) * height)
    const verticalCount = Math.max(0, width * (height - 1))
    const diagonalCount = Math.max(0, (width - 1) * (height - 1))
    const edgeCount = horizontalCount + verticalCount + 2 * diagonalCount
    if (edgeCount === 0) {
      return new Float32Array()
    }

    const weightBuffer = device.createBuffer({
      size: edgeCount * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    })
    const readBuffer = device.createBuffer({
      size: edgeCount * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    })
    const paramsBuffer = device.createBuffer({
      size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    const paramsData = new ArrayBuffer(48)
    const paramsU32 = new Uint32Array(paramsData)
    paramsU32[0] = width
    paramsU32[1] = height
    paramsU32[2] = 0
    paramsU32[3] = horizontalCount
    paramsU32[4] = horizontalCount + verticalCount
    paramsU32[5] = horizontalCount + verticalCount + diagonalCount
    device.queue.writeBuffer(paramsBuffer, 0, paramsData)

    const bindGroup = device.createBindGroup({
      layout: this.felzenszwalbBindGroupLayout,
      entries: [
        { binding: 0, resource: this.texture.createView() },
        { binding: 1, resource: { buffer: weightBuffer } },
        { binding: 2, resource: { buffer: paramsBuffer } },
      ],
    })

    const encoder = device.createCommandEncoder()
    const pass = encoder.beginComputePass()
    pass.setPipeline(this.felzenszwalbPipelines.computeWeights)
    pass.setBindGroup(0, bindGroup)
    pass.dispatchWorkgroups(Math.ceil(width / 16), Math.ceil(height / 16))
    pass.end()
    encoder.copyBufferToBuffer(weightBuffer, 0, readBuffer, 0, edgeCount * 4)
    device.queue.submit([encoder.finish()])

    await readBuffer.mapAsync(GPUMapMode.READ)
    const weights = new Float32Array(readBuffer.getMappedRange()).slice()
    readBuffer.unmap()
    weightBuffer.destroy()
    readBuffer.destroy()
    paramsBuffer.destroy()

    return weights
  }

  private configureContext() {
    if (!this.context || !this.device || !this.format) {
      return
    }

    this.context.configure({
      device: this.device,
      format: this.format,
      alphaMode: 'opaque',
    })
  }

  private writeUniforms(settings: FilterSettings) {
    if (!this.device || !this.uniformBuffer) {
      return
    }

    const data = new Float32Array([
      this.canvas.width,
      this.canvas.height,
      settings.radius,
      FILTER_INDEX[settings.filter],
      settings.levels,
      settings.grayscale ? 1 : 0,
      settings.contrast,
      settings.edgeThreshold,
      settings.edgeStrength,
      settings.slicSize,
      0,
      0,
    ])

    this.device.queue.writeBuffer(this.uniformBuffer, 0, data)
  }

  private requireDevice() {
    if (!this.device) {
      throw new Error('WebGPU renderer has not been initialized.')
    }

    return this.device
  }
}

export async function loadImageFile(file: File): Promise<LoadedImage> {
  const sourceBitmap = await createImageBitmap(file, {
    imageOrientation: 'from-image',
    premultiplyAlpha: 'none',
  })

  const maxDimension = Math.max(sourceBitmap.width, sourceBitmap.height)
  if (maxDimension <= MAX_WORKING_IMAGE_DIMENSION) {
    const pixels = getImagePixels(sourceBitmap)
    return {
      bitmap: sourceBitmap,
      name: file.name,
      width: sourceBitmap.width,
      height: sourceBitmap.height,
      pixels,
    }
  }

  const scale = MAX_WORKING_IMAGE_DIMENSION / maxDimension
  const resizeWidth = Math.max(1, Math.round(sourceBitmap.width * scale))
  const resizeHeight = Math.max(1, Math.round(sourceBitmap.height * scale))
  sourceBitmap.close()

  const bitmap = await createImageBitmap(file, {
    imageOrientation: 'from-image',
    premultiplyAlpha: 'none',
    resizeWidth,
    resizeHeight,
    resizeQuality: 'high',
  })

  return {
    bitmap,
    name: file.name,
    width: bitmap.width,
    height: bitmap.height,
    pixels: getImagePixels(bitmap),
  }
}

function getImagePixels(bitmap: ImageBitmap) {
  const canvas = document.createElement('canvas')
  canvas.width = bitmap.width
  canvas.height = bitmap.height
  const context = canvas.getContext('2d', { willReadFrequently: true })
  if (!context) {
    throw new Error('Could not read image pixels.')
  }

  context.drawImage(bitmap, 0, 0)
  return context.getImageData(0, 0, bitmap.width, bitmap.height).data
}

function createFelzenszwalbSegments(
  image: LoadedImage,
  requestedScale: number,
  requestedMinSize: number,
  computedWeights?: Float32Array,
) {
  const { width, height, pixels } = image
  const count = width * height
  const horizontalCount = Math.max(0, (width - 1) * height)
  const verticalCount = Math.max(0, width * (height - 1))
  const diagonalCount = Math.max(0, (width - 1) * (height - 1))
  const edgeCount = horizontalCount + verticalCount + 2 * diagonalCount
  const edgeA = new Uint32Array(edgeCount)
  const edgeB = new Uint32Array(edgeCount)
  const weights = new Float32Array(edgeCount)
  const order = new Uint32Array(edgeCount)
  let edgeIndex = 0

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = x + y * width
      if (x + 1 < width) {
        addEdge(edgeA, edgeB, weights, order, edgeIndex, index, index + 1, pixels, computedWeights?.[x + y * (width - 1)])
        edgeIndex += 1
      }
      if (y + 1 < height) {
        addEdge(
          edgeA,
          edgeB,
          weights,
          order,
          edgeIndex,
          index,
          index + width,
          pixels,
          computedWeights?.[horizontalCount + x + y * width],
        )
        edgeIndex += 1
      }
      if (x + 1 < width && y + 1 < height) {
        addEdge(
          edgeA,
          edgeB,
          weights,
          order,
          edgeIndex,
          index,
          index + width + 1,
          pixels,
          computedWeights?.[horizontalCount + verticalCount + x + y * (width - 1)],
        )
        edgeIndex += 1
      }
      if (x + 1 < width && y > 0) {
        addEdge(
          edgeA,
          edgeB,
          weights,
          order,
          edgeIndex,
          index,
          index - width + 1,
          pixels,
          computedWeights?.[horizontalCount + verticalCount + diagonalCount + x + (y - 1) * (width - 1)],
        )
        edgeIndex += 1
      }
    }
  }

  order.sort((left, right) => weights[left] - weights[right])

  const scale = Math.max(10, requestedScale)
  const minSize = Math.max(2, Math.round(requestedMinSize))
  const forest = new SegmentForest(count)

  for (let i = 0; i < edgeCount; i += 1) {
    const edge = order[i]
    const a = forest.find(edgeA[edge])
    const b = forest.find(edgeB[edge])
    if (a === b) {
      continue
    }

    const weight = weights[edge]
    if (weight <= forest.internal[a] + scale / forest.size[a] && weight <= forest.internal[b] + scale / forest.size[b]) {
      forest.union(a, b, weight)
    }
  }

  for (let i = 0; i < edgeCount; i += 1) {
    const edge = order[i]
    const a = forest.find(edgeA[edge])
    const b = forest.find(edgeB[edge])
    if (a !== b && (forest.size[a] < minSize || forest.size[b] < minSize)) {
      forest.union(a, b, weights[edge])
    }
  }

  const labels = new Int32Array(count)
  const rootToLabel = new Map<number, number>()
  for (let index = 0; index < count; index += 1) {
    const root = forest.find(index)
    let label = rootToLabel.get(root)
    if (label === undefined) {
      label = rootToLabel.size
      rootToLabel.set(root, label)
    }
    labels[index] = label
  }

  return labelsToAveragePixels(pixels, labels, rootToLabel.size)
}

function addEdge(
  edgeA: Uint32Array,
  edgeB: Uint32Array,
  weights: Float32Array,
  order: Uint32Array,
  edgeIndex: number,
  a: number,
  b: number,
  pixels: Uint8ClampedArray,
  computedWeight?: number,
) {
  const offsetA = a * 4
  const offsetB = b * 4
  const dr = pixels[offsetA] - pixels[offsetB]
  const dg = pixels[offsetA + 1] - pixels[offsetB + 1]
  const db = pixels[offsetA + 2] - pixels[offsetB + 2]

  edgeA[edgeIndex] = a
  edgeB[edgeIndex] = b
  weights[edgeIndex] = computedWeight ?? Math.sqrt(dr * dr + dg * dg + db * db)
  order[edgeIndex] = edgeIndex
}

function createAdaptiveBrushSegments(
  image: LoadedImage,
  requestedLargeBrush: number,
  requestedMediumBrush: number,
  requestedSmallBrush: number,
  requestedDetailThreshold: number,
  requestedDetailStrength: number,
) {
  const { width, height, pixels } = image
  const largeBrush = Math.max(2, Math.round(requestedLargeBrush))
  const mediumBrush = Math.max(2, Math.min(largeBrush, Math.round(requestedMediumBrush)))
  const smallBrush = Math.max(2, Math.min(mediumBrush, Math.round(requestedSmallBrush)))
  const detailThreshold = clamp(requestedDetailThreshold, 0, 1)
  const detailStrength = clamp(requestedDetailStrength, 0, 1)
  const largeField = createBrushField(image, largeBrush, 0x15a4)
  const mediumField = createBrushField(image, mediumBrush, 0x89f1)
  const smallField = createBrushField(image, smallBrush, 0x4c2d)
  const largeCellDetail = createCellDetailMap(pixels, width, largeField.labels, largeField.labelCount)
  const mediumCellDetail = createCellDetailMap(pixels, width, mediumField.labels, mediumField.labelCount)
  const output = new Uint8ClampedArray(pixels.length)

  for (let index = 0; index < width * height; index += 1) {
    const pixelOffset = index * 4
    const x = index % width
    const y = Math.floor(index / width)
    const largeDetail = largeCellDetail[largeField.labels[index]]
    const mediumDetail = mediumCellDetail[mediumField.labels[index]]
    const localDetail = pixelContrast(pixels, width, x, y)
    const useMediumBrush = largeDetail > detailThreshold
    const useSmallBrush = useMediumBrush && mediumDetail > Math.min(1, detailThreshold + 0.12)
    const source = useSmallBrush ? smallField.pixels : useMediumBrush ? mediumField.pixels : largeField.pixels
    const accentMix = useSmallBrush ? Math.min(0.5, smoothstep(0.03, 0.18, localDetail) * detailStrength) : 0

    output[pixelOffset] = mixChannel(source[pixelOffset], pixels[pixelOffset], accentMix)
    output[pixelOffset + 1] = mixChannel(source[pixelOffset + 1], pixels[pixelOffset + 1], accentMix)
    output[pixelOffset + 2] = mixChannel(source[pixelOffset + 2], pixels[pixelOffset + 2], accentMix)
    output[pixelOffset + 3] = 255
  }

  return output
}

function createBrushField(image: LoadedImage, requestedBrushSize: number, seed: number) {
  const { width, height, pixels } = image
  const brushSize = Math.max(2, requestedBrushSize)
  const gridWidth = Math.ceil(width / brushSize) + 2
  const gridHeight = Math.ceil(height / brushSize) + 2
  const labelCount = gridWidth * gridHeight
  const sums = new Float64Array(labelCount * 3)
  const counts = new Uint32Array(labelCount)
  const labels = new Uint32Array(width * height)
  const output = new Uint8ClampedArray(pixels.length)

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const cellX = Math.floor(x / brushSize) + 1
      const cellY = Math.floor(y / brushSize) + 1
      let bestLabel = 0
      let bestDistance = Number.POSITIVE_INFINITY

      for (let offsetY = -1; offsetY <= 1; offsetY += 1) {
        for (let offsetX = -1; offsetX <= 1; offsetX += 1) {
          const candidateX = cellX + offsetX
          const candidateY = cellY + offsetY
          const center = brushCenter(candidateX, candidateY, brushSize, seed)
          const dx = x - center.x
          const dy = y - center.y
          const distance = dx * dx + dy * dy
          if (distance < bestDistance) {
            bestDistance = distance
            bestLabel = candidateX + candidateY * gridWidth
          }
        }
      }

      const index = x + y * width
      const pixelOffset = index * 4
      const sumOffset = bestLabel * 3
      labels[index] = bestLabel
      sums[sumOffset] += pixels[pixelOffset]
      sums[sumOffset + 1] += pixels[pixelOffset + 1]
      sums[sumOffset + 2] += pixels[pixelOffset + 2]
      counts[bestLabel] += 1
    }
  }

  for (let index = 0; index < labels.length; index += 1) {
    const pixelOffset = index * 4
    const label = labels[index]
    const sumOffset = label * 3
    const count = Math.max(1, counts[label])
    output[pixelOffset] = sums[sumOffset] / count
    output[pixelOffset + 1] = sums[sumOffset + 1] / count
    output[pixelOffset + 2] = sums[sumOffset + 2] / count
    output[pixelOffset + 3] = 255
  }

  return {
    pixels: output,
    labels,
    labelCount,
  }
}

function createCellDetailMap(pixels: Uint8ClampedArray, width: number, labels: Uint32Array, labelCount: number) {
  const totals = new Float32Array(labelCount)
  const counts = new Uint32Array(labelCount)

  for (let index = 0; index < labels.length; index += 1) {
    const x = index % width
    const y = Math.floor(index / width)
    const label = labels[index]
    totals[label] += pixelContrast(pixels, width, x, y)
    counts[label] += 1
  }

  for (let index = 0; index < totals.length; index += 1) {
    totals[index] = counts[index] > 0 ? totals[index] / counts[index] : 0
  }

  normalizeDetailValues(totals)
  return totals
}

function normalizeDetailValues(detailValues: Float32Array) {
  const ordered = Array.from(detailValues).sort((left, right) => left - right)
  if (ordered.length === 0) {
    return
  }

  let firstDifferent = ordered.length - 1
  for (let index = 1; index < ordered.length; index += 1) {
    if (ordered[index] > ordered[0]) {
      firstDifferent = index
      break
    }
  }

  if (firstDifferent === ordered.length - 1 && ordered[0] === ordered[ordered.length - 1]) {
    detailValues.fill(0)
    return
  }

  const denominator = Math.max(1, ordered.length - firstDifferent)
  for (let index = 0; index < detailValues.length; index += 1) {
    const rank = upperBound(ordered, detailValues[index])
    detailValues[index] = rank <= firstDifferent ? 0 : clamp((rank - firstDifferent) / denominator, 0, 1)
  }
}

function upperBound(values: number[], target: number) {
  let low = 0
  let high = values.length

  while (low < high) {
    const middle = Math.floor((low + high) / 2)
    if (values[middle] <= target) {
      low = middle + 1
    } else {
      high = middle
    }
  }

  return low
}

function brushCenter(cellX: number, cellY: number, brushSize: number, seed: number) {
  const jitterX = hashToUnit(cellX, cellY, seed) - 0.5
  const jitterY = hashToUnit(cellX, cellY, seed ^ 0x6d2b79f5) - 0.5

  return {
    x: (cellX - 0.5 + jitterX * 0.72) * brushSize,
    y: (cellY - 0.5 + jitterY * 0.72) * brushSize,
  }
}

function hashToUnit(x: number, y: number, seed: number) {
  let value = Math.imul(x + seed, 374761393) + Math.imul(y ^ seed, 668265263)
  value = Math.imul(value ^ (value >>> 13), 1274126177)
  return ((value ^ (value >>> 16)) >>> 0) / 4294967295
}

function pixelContrast(pixels: Uint8ClampedArray, width: number, x: number, y: number) {
  if (x === 0 || y === 0) {
    return 0
  }

  const offset = (x + y * width) * 4
  const leftOffset = offset - 4
  const upOffset = offset - width * 4
  const dx = colorDistance(pixels, offset, leftOffset)
  const dy = colorDistance(pixels, offset, upOffset)

  return Math.min(1, Math.sqrt(dx * dx + dy * dy) / 255)
}

function colorDistance(pixels: Uint8ClampedArray, offsetA: number, offsetB: number) {
  const red = pixels[offsetA] - pixels[offsetB]
  const green = pixels[offsetA + 1] - pixels[offsetB + 1]
  const blue = pixels[offsetA + 2] - pixels[offsetB + 2]

  return Math.sqrt(red * red + green * green + blue * blue)
}

function mixChannel(a: number, b: number, amount: number) {
  return a * (1 - amount) + b * amount
}

function smoothstep(edge0: number, edge1: number, value: number) {
  const amount = clamp((value - edge0) / (edge1 - edge0), 0, 1)
  return amount * amount * (3 - 2 * amount)
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value))
}

class SegmentForest {
  readonly parent: Int32Array
  readonly size: Int32Array
  readonly internal: Float32Array

  constructor(count: number) {
    this.parent = new Int32Array(count)
    this.size = new Int32Array(count)
    this.internal = new Float32Array(count)
    for (let index = 0; index < count; index += 1) {
      this.parent[index] = index
      this.size[index] = 1
    }
  }

  find(index: number) {
    let root = index
    while (this.parent[root] !== root) {
      root = this.parent[root]
    }

    while (this.parent[index] !== index) {
      const parent = this.parent[index]
      this.parent[index] = root
      index = parent
    }

    return root
  }

  union(a: number, b: number, weight: number) {
    let rootA = this.find(a)
    let rootB = this.find(b)
    if (rootA === rootB) {
      return rootA
    }

    if (this.size[rootA] < this.size[rootB]) {
      const swap = rootA
      rootA = rootB
      rootB = swap
    }

    this.parent[rootB] = rootA
    this.size[rootA] += this.size[rootB]
    this.internal[rootA] = weight
    return rootA
  }
}

function labelsToAveragePixels(pixels: Uint8ClampedArray, labels: Int32Array, labelCount: number) {
  const sums = new Float64Array(labelCount * 3)
  const counts = new Uint32Array(labelCount)
  const output = new Uint8ClampedArray(pixels.length)

  for (let index = 0; index < labels.length; index += 1) {
    const label = labels[index]
    if (label < 0) {
      continue
    }

    const pixelOffset = index * 4
    const labelOffset = label * 3
    sums[labelOffset] += pixels[pixelOffset]
    sums[labelOffset + 1] += pixels[pixelOffset + 1]
    sums[labelOffset + 2] += pixels[pixelOffset + 2]
    counts[label] += 1
  }

  for (let index = 0; index < labels.length; index += 1) {
    const pixelOffset = index * 4
    const label = labels[index]
    if (label < 0 || counts[label] === 0) {
      output[pixelOffset] = pixels[pixelOffset]
      output[pixelOffset + 1] = pixels[pixelOffset + 1]
      output[pixelOffset + 2] = pixels[pixelOffset + 2]
      output[pixelOffset + 3] = 255
      continue
    }

    const labelOffset = label * 3
    output[pixelOffset] = sums[labelOffset] / counts[label]
    output[pixelOffset + 1] = sums[labelOffset + 1] / counts[label]
    output[pixelOffset + 2] = sums[labelOffset + 2] / counts[label]
    output[pixelOffset + 3] = 255
  }

  return output
}
