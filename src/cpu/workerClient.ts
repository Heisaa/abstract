import type { FilterId, ParamValues } from '../filters/registry'
import type { ImageSlot, WorkerRequest, WorkerResponse } from './protocol'

export type CpuJobResult = Extract<WorkerResponse, { type: 'result' }>

// Promise wrapper around the CPU filter worker.
export class CpuWorkerClient {
  private readonly worker: Worker
  private nextJobId = 1
  private readonly pending = new Map<number, { resolve: (result: CpuJobResult) => void; reject: (error: Error) => void }>()

  constructor() {
    this.worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' })
    this.worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      const response = event.data
      const job = this.pending.get(response.jobId)
      if (!job) {
        return
      }
      this.pending.delete(response.jobId)
      if (response.type === 'result') {
        job.resolve(response)
      } else {
        job.reject(new Error(response.message))
      }
    }
    this.worker.onerror = (event) => {
      const error = new Error(event.message || 'The filter worker crashed.')
      for (const job of this.pending.values()) {
        job.reject(error)
      }
      this.pending.clear()
    }
  }

  // Copies the pixels so the caller keeps its own buffer.
  setImage(slot: ImageSlot, version: number, width: number, height: number, pixels: Uint8ClampedArray) {
    const copy = pixels.slice().buffer
    this.post({ type: 'setImage', slot, version, width, height, pixels: copy }, [copy])
  }

  dropImage(slot: ImageSlot) {
    this.post({ type: 'dropImage', slot })
  }

  run(options: { slot: ImageSlot; draft: boolean; scale: number; filter: FilterId; params: ParamValues; labels?: Int32Array }) {
    const jobId = this.nextJobId++
    const labels = options.labels ? options.labels.slice().buffer : undefined
    return new Promise<CpuJobResult>((resolve, reject) => {
      this.pending.set(jobId, { resolve, reject })
      this.post(
        {
          type: 'run',
          jobId,
          slot: options.slot,
          draft: options.draft,
          scale: options.scale,
          filter: options.filter,
          params: options.params,
          labels,
        },
        labels ? [labels] : [],
      )
    })
  }

  terminate() {
    this.worker.terminate()
  }

  private post(message: WorkerRequest, transfer: Transferable[] = []) {
    this.worker.postMessage(message, transfer)
  }
}
