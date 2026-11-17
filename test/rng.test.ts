import { describe, expect, it } from 'vitest'
import { mulberry32, seededPermutation } from '../src/rng'

describe('mulberry32', () => {
  it('is deterministic for a given seed', () => {
    const a = mulberry32(42)
    const b = mulberry32(42)
    expect(a()).toBe(b())
    expect(a()).toBe(b())
  })

  it('produces values in [0, 1)', () => {
    const rand = mulberry32(1)
    for (let i = 0; i < 100; i++) {
      const value = rand()
      expect(value).toBeGreaterThanOrEqual(0)
      expect(value).toBeLessThan(1)
    }
  })
})

describe('seededPermutation', () => {
  it('returns a permutation of 0..n-1', () => {
    const order = seededPermutation(10, 7)
    const sorted = Array.from(order).sort((a, b) => a - b)
    expect(sorted).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
  })

  it('is deterministic for a given seed', () => {
    expect(Array.from(seededPermutation(20, 5))).toEqual(
      Array.from(seededPermutation(20, 5))
    )
  })
})
