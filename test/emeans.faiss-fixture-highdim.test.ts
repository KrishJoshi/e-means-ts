import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'
import { fit } from '../src/emeans'
import { resolveBatchLaw } from '../src/emeansSchedule'

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
  new URL('./fixtures/emeans-faiss-reference-highdim.json', import.meta.url)
)
const fixture: Fixture = JSON.parse(readFileSync(fixturePath, 'utf-8'))

// Neither of the other two faiss fixtures (dim=2, dim=8) ever exercises the
// 'throughput' batch-law branch: resolveBatchLaw('auto', dim) in
// src/emeansSchedule.ts picks 'quality' for dim <= AUTO_QUALITY_MAX_D (16)
// and 'throughput' above that. Every realistic embedding dimension the
// README advertises (50-3072) runs exclusively through 'throughput', so this
// fixture uses dim=32 to close that coverage gap.
//
// Same "meaningfully overlapping" cluster design as the hard fixture (6 true
// Gaussian blobs, k=8 so dead-cluster revival fires), but the tolerance here
// is tighter (+/-1%, vs. the hard fixture's +/-5%) because the gap between a
// correct and a deliberately-broken run is narrower at this dim/batch-size
// combination: a correct fit() call lands at ratio ~1.0003, while
// deliberately replacing the EMA update with plain per-batch mean
// replacement (alpha0: 1.0) lands at ratio ~1.0130 -- outside this band.
describe('fit vs faiss NumpyEmeans reference (high-dim, throughput batch law)', () => {
  it("resolves to the 'throughput' batch law at this dimension", () => {
    expect(resolveBatchLaw('auto', fixture.dim)).toBe('throughput')
  })

  it('reaches a final loss within 1% of the real faiss e-means output', () => {
    const flat = new Float32Array(fixture.data.flat())
    const result = fit(flat, fixture.dim, fixture.k, {
      seed: fixture.seed,
      nepochs: fixture.nepochs,
    })
    const ratio = result.loss / fixture.finalLoss
    expect(ratio).toBeGreaterThan(0.99)
    expect(ratio).toBeLessThan(1.01)
  })

  it('rejects a deliberately-broken update rule (alpha0: 1.0 disables the EMA)', () => {
    const flat = new Float32Array(fixture.data.flat())
    const result = fit(flat, fixture.dim, fixture.k, {
      seed: fixture.seed,
      nepochs: fixture.nepochs,
      alpha0: 1.0,
    })
    const ratio = result.loss / fixture.finalLoss
    expect(ratio < 0.99 || ratio > 1.01).toBe(true)
  })
})
