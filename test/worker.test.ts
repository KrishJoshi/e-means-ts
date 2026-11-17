import { describe, expect, it } from 'vitest'
import { runInWorker } from '../src/worker'

describe('runInWorker', () => {
  it('throws a clear error when the Worker API is unavailable', async () => {
    await expect(
      runInWorker('kmeans', new Float32Array([0, 0, 1, 1]), 2, 2)
    ).rejects.toThrow(/Worker API/)
  })

  // The import.meta.url guard (for the CJS build, where import.meta is
  // compiled to an empty object by esbuild/tsup) can't be unit-tested here:
  // vitest runs this file as real ESM, so import.meta.url is always a
  // genuine string and the guard never engages. That codepath is verified
  // by building the package and requiring dist/index.cjs directly -- see the
  // final-review-fix-report.md for the exact repro and output.
})
