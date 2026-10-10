import { downsamplePixels } from './common'
import type { ImageSlot, WorkerRequest, WorkerResponse } from './protocol'
import { CpuFilterRunner } from './run'
import type { CpuImage } from './types'

// Dedicated worker for the CPU filters, keeping segmentation and stroke
// painting off the main thread so the UI stays responsive.

const scope = self as unknown as Worker
const runner = new CpuFilterRunner()
const images = new Map<ImageSlot, { version: number; image: CpuImage; draft?: CpuImage }>()

function getImage(slot: ImageSlot, draft: boolean) {
  const entry = images.get(slot)
  if (!entry) {
    throw new Error('The worker has no image loaded.')
  }

  if (!draft) {
    return { image: entry.image, key: `${slot}:${entry.version}:full` }
  }

  if (!entry.draft) {
    const reduced = downsamplePixels(entry.image.pixels, entry.image.width, entry.image.height, 2)
    entry.draft = { ...reduced, scale: entry.image.scale / 2 }
  }
  return { image: entry.draft, key: `${slot}:${entry.version}:draft` }
}

scope.onmessage = (event: MessageEvent<WorkerRequest>) => {
  const request = event.data
  if (request.type === 'setImage') {
    runner.clear(`${request.slot}:`)
    images.set(request.slot, {
      version: request.version,
      image: { pixels: new Uint8ClampedArray(request.pixels), width: request.width, height: request.height, scale: 1 },
    })
    return
  }

  if (request.type === 'dropImage') {
    runner.clear(`${request.slot}:`)
    images.delete(request.slot)
    return
  }

  try {
    const { image, key } = getImage(request.slot, request.draft)
    const scaledImage = { ...image, scale: request.scale }
    const labels = request.labels ? new Int32Array(request.labels) : undefined
    const output = runner.run(scaledImage, `${key}:${request.scale}`, request.filter, request.params, { labels })
    // Copy into a standalone buffer: getImageData results may be views.
    const pixels = output.pixels.slice().buffer
    const response: WorkerResponse = {
      type: 'result',
      jobId: request.jobId,
      width: image.width,
      height: image.height,
      pixels,
      palette: output.palette,
    }
    scope.postMessage(response, [pixels])
  } catch (reason) {
    const response: WorkerResponse = {
      type: 'error',
      jobId: request.jobId,
      message: reason instanceof Error ? reason.message : String(reason),
    }
    scope.postMessage(response)
  }
}
