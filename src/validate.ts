// Shared input guards for fit()/kmeans()/predict(). Centralized here so the
// three entry points stay consistent and callers get the same error wording
// regardless of which function they call.

export function assertValidClusteringInput(data: Float32Array, dim: number, k: number): number {
  if (!Number.isInteger(dim) || dim <= 0) {
    throw new Error(`dim must be a positive integer, got ${dim}`)
  }
  if (data.length % dim !== 0) {
    throw new Error(
      `data.length (${data.length}) is not divisible by dim (${dim}); data must be a flat, row-major array of n * dim values`
    )
  }
  const n = data.length / dim
  if (!Number.isInteger(k) || k <= 0) {
    throw new Error(`k must be a positive integer, got ${k}`)
  }
  if (k > n) {
    throw new Error(`k (${k}) cannot exceed the number of points (${n})`)
  }
  return n
}

export function assertValidPredictInput(
  data: Float32Array,
  dim: number,
  centroids: Float32Array
): void {
  if (!Number.isInteger(dim) || dim <= 0) {
    throw new Error(`dim must be a positive integer, got ${dim}`)
  }
  if (data.length % dim !== 0) {
    throw new Error(
      `data.length (${data.length}) is not divisible by dim (${dim}); data must be a flat, row-major array of n * dim values`
    )
  }
  if (centroids.length % dim !== 0) {
    throw new Error(
      `centroids.length (${centroids.length}) is not divisible by dim (${dim})`
    )
  }
}
