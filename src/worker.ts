import type { ClusterResult, EMeansOptions, KMeansOptions } from './types'
import type { WorkerRequest } from './worker-entry'

/**
 * Runs `fit` or `kmeans` in a Web Worker so browser callers don't block the
 * main thread. Requires a bundler that supports `new URL(..., import.meta.url)`
 * worker syntax (Vite, webpack 5+) -- see the README for CJS/Node scoping.
 *
 * Note: `data`'s underlying `ArrayBuffer` is transferred to the worker (not
 * copied) and becomes unusable in the caller afterward. If `data` is a
 * `.subarray()` view, the ENTIRE backing buffer is detached, not just the
 * view's portion. Pass a copy (`data.slice()`) if you need to reuse the
 * original array, e.g. for a later `predict()` call.
 */
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
  // import.meta.url is empty in the CJS build (esbuild/tsup can't resolve
  // "import.meta" outside an ES module), which would otherwise make `new
  // URL(..., import.meta.url)` below throw a cryptic "Invalid URL" instead of
  // a clear, actionable error. Guard it explicitly.
  if (typeof import.meta === 'undefined' || !import.meta.url) {
    return Promise.reject(
      new Error(
        'runInWorker requires an ESM-aware bundler (Vite, webpack 5+) or ESM entry point; the CJS build cannot resolve the worker URL. Import e-means as an ES module, or call fit()/kmeans() directly.'
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
