import { mulberry32 } from './rng'

function squaredDistanceToCentroid(
  data: Float32Array,
  dim: number,
  row: number,
  centroids: Float32Array,
  c: number
): number {
  const rowOffset = row * dim
  const cOffset = c * dim
  let dist = 0
  for (let d = 0; d < dim; d++) {
    const diff = data[rowOffset + d] - centroids[cOffset + d]
    dist += diff * diff
  }
  return dist
}

export function kmeansPlusPlusInit(
  data: Float32Array,
  dim: number,
  n: number,
  k: number,
  seed: number
): Float32Array {
  const rand = mulberry32(seed)
  const centroids = new Float32Array(k * dim)
  const firstIdx = Math.floor(rand() * n)
  centroids.set(data.subarray(firstIdx * dim, firstIdx * dim + dim), 0)

  const closestDistSq = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    closestDistSq[i] = squaredDistanceToCentroid(data, dim, i, centroids, 0)
  }

  for (let c = 1; c < k; c++) {
    let total = 0
    for (let i = 0; i < n; i++) total += closestDistSq[i]
    let target = rand() * total
    let chosen = n - 1
    for (let i = 0; i < n; i++) {
      target -= closestDistSq[i]
      if (target <= 0) {
        chosen = i
        break
      }
    }
    centroids.set(data.subarray(chosen * dim, chosen * dim + dim), c * dim)
    for (let i = 0; i < n; i++) {
      const d = squaredDistanceToCentroid(data, dim, i, centroids, c)
      if (d < closestDistSq[i]) closestDistSq[i] = d
    }
  }
  return centroids
}
