import { describe, expect, it } from 'vitest'
import { kmeans } from '../src/kmeans'

function makeBlobs(): { data: Float32Array; dim: number } {
  const dim = 2
  const points: number[] = []
  const centers = [
    [0, 0],
    [10, 10],
    [0, 10],
  ]
  for (const [cx, cy] of centers) {
    for (let i = 0; i < 30; i++) {
      points.push(cx + (Math.random() - 0.5) * 0.5, cy + (Math.random() - 0.5) * 0.5)
    }
  }
  return { data: new Float32Array(points), dim }
}

describe('kmeans', () => {
  it('separates three well-separated blobs into three clusters', () => {
    const { data, dim } = makeBlobs()
    const result = kmeans(data, dim, 3, { seed: 1 })
    const labelCounts = new Map<number, number>()
    for (const label of result.labels) {
      labelCounts.set(label, (labelCounts.get(label) ?? 0) + 1)
    }
    expect(labelCounts.size).toBe(3)
    for (const count of labelCounts.values()) {
      expect(count).toBeGreaterThan(20)
    }
  })

  it('is deterministic for a fixed seed', () => {
    const { data, dim } = makeBlobs()
    const a = kmeans(data, dim, 3, { seed: 99 })
    const b = kmeans(data, dim, 3, { seed: 99 })
    expect(Array.from(a.labels)).toEqual(Array.from(b.labels))
  })
})

describe('kmeans input validation', () => {
  it('throws when data.length is not divisible by dim', () => {
    expect(() => kmeans(new Float32Array([1, 2, 3]), 2, 1)).toThrow()
  })

  it('throws when k <= 0', () => {
    const { data, dim } = makeBlobs()
    expect(() => kmeans(data, dim, 0)).toThrow()
    expect(() => kmeans(data, dim, -1)).toThrow()
  })

  it('throws when k > n', () => {
    const { data, dim } = makeBlobs()
    expect(() => kmeans(data, dim, 10_000)).toThrow()
  })
})
