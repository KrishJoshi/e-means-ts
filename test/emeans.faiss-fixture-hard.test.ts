import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'
import { fit } from '../src/emeans'

interface Fixture {
  data: number[][]
  dim: number
  k: number
  nepochs: number
  seed: number
  finalLoss: number
  lossPerEpoch: number[]
}

const fixturePath = fileURLToPath(
  new URL('./fixtures/emeans-faiss-reference-hard.json', import.meta.url)
)
const fixture: Fixture = JSON.parse(readFileSync(fixturePath, 'utf-8'))

// This fixture exists because the original easy fixture (emeans-faiss-reference.json,
// 4 Gaussian blobs 10 units apart vs. a 0.25 noise std, d=2) barely discriminates
// correct vs. wrong implementations: ~40-sigma cluster separation means almost any
// non-catastrophic clustering lands at nearly the faiss loss, and d=2 disables
// dead-cluster revival entirely (NO_REVIVAL_MAX_D), so reviveUnused()/split logic in
// src/emeans.ts is never exercised by the easy fixture in either direction.
//
// This harder fixture uses d=8 (revival active) and 6 overlapping Gaussian blobs
// (centers ~1.4-3.0 noise stds apart, not ~40) with k=8 (more clusters than true
// structure) so dead-cluster revival actually fires during training. The tolerance
// here is tight (+/-5%) and was chosen empirically: a correct fit() call lands at
// ratio ~1.0026, while deliberately replacing the EMA update with plain per-batch
// mean replacement (alpha0: 1.0) lands at ratio ~1.084 -- outside this band. See
// task-7-report.md for the full validation.
describe('fit vs faiss NumpyEmeans reference (hard, overlapping, revival-active)', () => {
  it('reaches a final loss within 5% of the real faiss e-means output', () => {
    const flat = new Float32Array(fixture.data.flat())
    const result = fit(flat, fixture.dim, fixture.k, {
      seed: fixture.seed,
      nepochs: fixture.nepochs,
    })
    const ratio = result.loss / fixture.finalLoss
    expect(ratio).toBeGreaterThan(0.95)
    expect(ratio).toBeLessThan(1.05)
  })

  it('rejects a deliberately-broken update rule (alpha0: 1.0 disables the EMA)', () => {
    const flat = new Float32Array(fixture.data.flat())
    const result = fit(flat, fixture.dim, fixture.k, {
      seed: fixture.seed,
      nepochs: fixture.nepochs,
      alpha0: 1.0,
    })
    const ratio = result.loss / fixture.finalLoss
    expect(ratio < 0.95 || ratio > 1.05).toBe(true)
  })
})
