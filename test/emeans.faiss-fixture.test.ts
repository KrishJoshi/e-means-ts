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
  new URL('./fixtures/emeans-faiss-reference.json', import.meta.url)
)
const fixture: Fixture = JSON.parse(readFileSync(fixturePath, 'utf-8'))

describe('fit vs faiss NumpyEmeans reference', () => {
  it('reaches a final loss within 25% of the real faiss e-means output', () => {
    const flat = new Float32Array(fixture.data.flat())
    const result = fit(flat, fixture.dim, fixture.k, {
      seed: fixture.seed,
      nepochs: fixture.nepochs,
    })
    const ratio = result.loss / fixture.finalLoss
    expect(ratio).toBeGreaterThan(0.75)
    expect(ratio).toBeLessThan(1.25)
  })
})
