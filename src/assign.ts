export interface AssignResult {
  sums: Float64Array
  counts: Float64Array
  objective: number
  labels: Int32Array
}

export function assignBatch(
  data: Float32Array,
  dim: number,
  rowIndices: Int32Array,
  centroids: Float32Array,
  k: number
): AssignResult {
  const sums = new Float64Array(k * dim)
  const counts = new Float64Array(k)
  const labels = new Int32Array(rowIndices.length)
  let objective = 0
  for (let r = 0; r < rowIndices.length; r++) {
    const row = rowIndices[r]
    const rowOffset = row * dim
    let bestDist = Infinity
    let bestJ = 0
    for (let j = 0; j < k; j++) {
      const cOffset = j * dim
      let dist = 0
      for (let d = 0; d < dim; d++) {
        const diff = data[rowOffset + d] - centroids[cOffset + d]
        dist += diff * diff
      }
      if (dist < bestDist) {
        bestDist = dist
        bestJ = j
      }
    }
    labels[r] = bestJ
    objective += bestDist
    counts[bestJ] += 1
    const sOffset = bestJ * dim
    for (let d = 0; d < dim; d++) {
      sums[sOffset + d] += data[rowOffset + d]
    }
  }
  return { sums, counts, objective, labels }
}
