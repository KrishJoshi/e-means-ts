import { describe, expect, it } from 'vitest'
import { fit, predict } from '../src/emeans'
import { kmeans } from '../src/kmeans'

function makeBlobs(): { data: Float32Array; dim: number } {
  const dim = 2
  const points: number[] = []
  const centers = [
    [0, 0],
    [10, 10],
    [0, 10],
    [10, 0],
  ]
  for (const [cx, cy] of centers) {
    for (let i = 0; i < 200; i++) {
      points.push(cx + (Math.random() - 0.5) * 0.5, cy + (Math.random() - 0.5) * 0.5)
    }
  }
  return { data: new Float32Array(points), dim }
}

describe('fit (e-means)', () => {
  it('separates four well-separated blobs into four clusters', () => {
    const { data, dim } = makeBlobs()
    const result = fit(data, dim, 4, { seed: 1, nepochs: 5 })
    const labelCounts = new Map<number, number>()
    for (const label of result.labels) {
      labelCounts.set(label, (labelCounts.get(label) ?? 0) + 1)
    }
    expect(labelCounts.size).toBe(4)
  })

  it('reaches a loss within 2x of 25-epoch k-means after far fewer epochs', () => {
    const { data, dim } = makeBlobs()
    // kmeans++ init: with default 'random' init, dim=2 disables dead-cluster
    // revival (dim <= NO_REVIVAL_MAX_D), and random init has a high chance of
    // picking two seed rows from the same blob out of only 4 equal-sized
    // blobs — permanently merging two clusters with nothing to split them
    // back apart within 5 epochs. kmeans++ init avoids that collision.
    const emeansResult = fit(data, dim, 4, { seed: 1, nepochs: 5, init: 'kmeans++' })
    const kmeansResult = kmeans(data, dim, 4, { seed: 1, maxEpochs: 25 })
    expect(emeansResult.loss).toBeLessThan(kmeansResult.loss * 2)
  })

  it('is deterministic for a fixed seed', () => {
    const { data, dim } = makeBlobs()
    const a = fit(data, dim, 4, { seed: 7 })
    const b = fit(data, dim, 4, { seed: 7 })
    expect(Array.from(a.centroids)).toEqual(Array.from(b.centroids))
  })
})

describe('predict', () => {
  it('assigns new points to the nearest fitted centroid', () => {
    const { data, dim } = makeBlobs()
    const { centroids } = fit(data, dim, 4, { seed: 1 })
    const query = new Float32Array([0, 0])
    const labels = predict(query, dim, centroids)
    expect(labels.length).toBe(1)
  })
})
