/// <reference lib="webworker" />
import { fit } from './emeans'
import { kmeans } from './kmeans'
import type { EMeansOptions, KMeansOptions } from './types'

export interface WorkerRequest {
  algorithm: 'emeans' | 'kmeans'
  data: Float32Array
  dim: number
  k: number
  opts?: EMeansOptions | KMeansOptions
}

self.onmessage = (event: MessageEvent<WorkerRequest>) => {
  const { algorithm, data, dim, k, opts } = event.data
  const result = algorithm === 'emeans' ? fit(data, dim, k, opts) : kmeans(data, dim, k, opts)
  ;(self as unknown as Worker).postMessage(result, {
    transfer: [result.centroids.buffer, result.labels.buffer],
  })
}
