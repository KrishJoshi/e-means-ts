import { assignBatch } from './assign'
import { kmeansPlusPlusInit } from './kmeansPlusPlus'
import { seededPermutation } from './rng'
import type { ClusterResult, KMeansOptions } from './types'

function randomInit(
  data: Float32Array,
  dim: number,
  n: number,
  k: number,
  seed: number
): Float32Array {
  const order = seededPermutation(n, seed)
  const centroids = new Float32Array(k * dim)
  for (let j = 0; j < k; j++) {
    const row = order[j]
    centroids.set(data.subarray(row * dim, row * dim + dim), j * dim)
  }
  return centroids
}

function centroidsEqual(a: Float32Array, b: Float32Array): boolean {
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false
  }
  return true
}

export function kmeans(
  data: Float32Array,
  dim: number,
  k: number,
  opts: KMeansOptions = {}
): ClusterResult {
  const n = data.length / dim
  const seed = opts.seed ?? 1234
  const maxEpochs = opts.maxEpochs ?? 25
  const init = opts.init ?? 'kmeans++'

  let centroids =
    init === 'kmeans++'
      ? kmeansPlusPlusInit(data, dim, n, k, seed)
      : randomInit(data, dim, n, k, seed)

  const allIndices = new Int32Array(n)
  for (let i = 0; i < n; i++) allIndices[i] = i

  let labels = new Int32Array(n)
  let loss = Infinity
  let epoch = 0
  for (; epoch < maxEpochs; epoch++) {
    const result = assignBatch(data, dim, allIndices, centroids, k)
    labels = result.labels
    loss = result.objective
    const newCentroids = new Float32Array(k * dim)
    for (let j = 0; j < k; j++) {
      const count = result.counts[j]
      const off = j * dim
      if (count > 0) {
        for (let d = 0; d < dim; d++) {
          newCentroids[off + d] = result.sums[off + d] / count
        }
      } else {
        newCentroids.set(centroids.subarray(off, off + dim), off)
      }
    }
    const converged = centroidsEqual(centroids, newCentroids)
    centroids = newCentroids
    if (converged) {
      epoch += 1
      break
    }
  }
  return { centroids, labels, loss, epochs: epoch }
}
