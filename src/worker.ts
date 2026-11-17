import type { ClusterResult, EMeansOptions, KMeansOptions } from './types'
import type { WorkerRequest } from './worker-entry'

export function runInWorker(
  algorithm: 'emeans' | 'kmeans',
  data: Float32Array,
  dim: number,
  k: number,
  opts?: EMeansOptions | KMeansOptions
): Promise<ClusterResult> {
  if (typeof Worker === 'undefined') {
    return Promise.reject(
      new Error(
        'runInWorker requires the Worker API (browser or a bundler that polyfills it); call fit()/kmeans() directly in Node.'
      )
    )
  }
  const worker = new Worker(new URL('./worker-entry.js', import.meta.url), {
    type: 'module',
  })
  return new Promise((resolve, reject) => {
    worker.onmessage = (event: MessageEvent<ClusterResult>) => {
      resolve(event.data)
      worker.terminate()
    }
    worker.onerror = (event) => {
      reject(event.error ?? new Error('e-means worker failed'))
      worker.terminate()
    }
    const request: WorkerRequest = { algorithm, data, dim, k, opts }
    worker.postMessage(request, { transfer: [data.buffer] })
  })
}
