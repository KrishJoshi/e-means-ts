import { describe, expect, it } from 'vitest'
import {
  epochSeed,
  resolveBatchLaw,
  resolveUnusedThreshold,
  roundPowerOfTwo,
  scheduledAlpha,
} from '../src/emeansSchedule'

describe('roundPowerOfTwo', () => {
  it('rounds to the nearest power of two', () => {
    expect(roundPowerOfTwo(1)).toBe(1)
    expect(roundPowerOfTwo(5)).toBe(4)
    expect(roundPowerOfTwo(6)).toBe(8)
    expect(roundPowerOfTwo(100)).toBe(128)
  })
})

describe('resolveBatchLaw', () => {
  it('picks quality for low dimensions and throughput for high', () => {
    expect(resolveBatchLaw('auto', 8)).toBe('quality')
    expect(resolveBatchLaw('auto', 16)).toBe('quality')
    expect(resolveBatchLaw('auto', 17)).toBe('throughput')
  })

  it('passes through an explicit law unchanged', () => {
    expect(resolveBatchLaw('throughput', 2)).toBe('throughput')
  })
})

describe('resolveUnusedThreshold', () => {
  it('disables revival at or below 2 dimensions by default', () => {
    expect(resolveUnusedThreshold(undefined, 2)).toBe(0)
    expect(resolveUnusedThreshold(undefined, 3)).toBeCloseTo(0.1)
  })

  it('rejects a negative threshold', () => {
    expect(() => resolveUnusedThreshold(-1, 10)).toThrow()
  })
})

describe('scheduledAlpha', () => {
  it('starts at alpha0 and anneals to one tenth of it', () => {
    expect(scheduledAlpha(1.0, 0, 100)).toBeCloseTo(1.0)
    expect(scheduledAlpha(1.0, 100, 100)).toBeCloseTo(0.1)
  })
})

describe('epochSeed', () => {
  it('is deterministic and differs across epochs', () => {
    expect(epochSeed(1234, 0)).toBe(epochSeed(1234, 0))
    expect(epochSeed(1234, 0)).not.toBe(epochSeed(1234, 1))
  })
})
