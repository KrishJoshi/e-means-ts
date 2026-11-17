import { describe, expect, it } from 'vitest'
import { runInWorker } from '../src/worker'

describe('runInWorker', () => {
  it('throws a clear error when the Worker API is unavailable', async () => {
    await expect(
      runInWorker('kmeans', new Float32Array([0, 0, 1, 1]), 2, 2)
    ).rejects.toThrow(/Worker API/)
  })
})
