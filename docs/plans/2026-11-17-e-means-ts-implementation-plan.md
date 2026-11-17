# e-means-ts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a zero-dependency TypeScript port of Meta FAIR's e-means clustering algorithm, plus a GitHub Pages demo (Shakespeare word-cloud race + bring-your-own-data), published to npm and GitHub fast enough to catch the current wave of attention on the paper.

**Architecture:** Single npm package (`e-means`) at the repo root exporting `fit` (e-means), `kmeans` (Lloyd's baseline), `predict`, and `runInWorker`. A separate `demo/` Vite app imports the package's TS source directly (no publish/link step needed for local dev) and deploys to GitHub Pages via Actions. The e-means core is a faithful simplified port of faiss's `contrib/e_means/e_means.py` + `base.py` (mini-batch assignment, EMA mass/centroid update with cosine-annealed step size, dead-cluster revival by splitting, last-epoch finalize) — ported from the actual faiss source, not reconstructed from the paper abstract.

**Tech Stack:** TypeScript, tsup (dual ESM/CJS build), Vitest, Vite (demo), plain Canvas 2D (no UI framework) for the demo, GitHub Actions for CI + Pages deploy.

**Spec:** `docs/design/2026-11-17-e-means-ts-design.md`

## Global Constraints

- Zero runtime dependencies in the published `e-means` package (devDependencies for build/test tooling are fine).
- Flat `Float32Array` + `dim` for all vector inputs/outputs — never `number[][]` in the public API.
- Everything must run unmodified in Node, browsers, and edge runtimes (Cloudflare Workers / Vercel Edge) — no Node-only APIs (`fs`, `worker_threads`, etc.) in `src/`.
- No claim, in code comments, README, or demo copy, that this matches faiss's CUTLASS/GPU wall-clock speed. The claim is: fewer epochs to a given loss, validated against the real faiss reference output.
- Out of scope for v1 (do not implement): spherical mode, multi-process/sharded training, `max_points_per_centroid` subsampling, fractional epoch budgets, WebGPU/GPU acceleration, framework-specific adapter packages (LangChain/LlamaIndex retriever classes). `nepochs` is integer-only in v1.
- License: MIT.

---

## Task 1: Project scaffolding

**Files:**
- Create: `package.json`
- Create: `tsconfig.json`
- Create: `tsup.config.ts`
- Create: `vitest.config.ts`
- Create: `.gitignore`
- Create: `LICENSE`

**Interfaces:**
- Produces: a working `npm run build`, `npm test`, `npm run bench` toolchain that every later task relies on.

- [ ] **Step 1: Create `package.json`**

```json
{
  "name": "e-means",
  "version": "0.1.0",
  "description": "Zero-dependency TypeScript port of e-means (Meta FAIR's mini-batch replacement for k-means) plus a Lloyd's k-means baseline. Runs in Node, browsers, Cloudflare Workers, Vercel Edge Functions, and Deno.",
  "keywords": [
    "e-means",
    "k-means",
    "clustering",
    "machine-learning",
    "vector-search",
    "embeddings",
    "typescript",
    "javascript",
    "edge",
    "cloudflare-workers",
    "vercel",
    "web-worker",
    "rag"
  ],
  "license": "MIT",
  "type": "module",
  "main": "./dist/index.cjs",
  "module": "./dist/index.js",
  "types": "./dist/index.d.ts",
  "exports": {
    ".": {
      "import": "./dist/index.js",
      "require": "./dist/index.cjs",
      "types": "./dist/index.d.ts"
    }
  },
  "files": ["dist"],
  "scripts": {
    "build": "tsup",
    "test": "vitest run",
    "bench": "tsx bench/run.ts"
  },
  "devDependencies": {
    "ml-kmeans": "^7.0.0",
    "tsup": "^8.3.0",
    "tsx": "^4.19.0",
    "typescript": "^5.7.0",
    "vitest": "^2.1.0"
  }
}
```

- [ ] **Step 2: Create `tsconfig.json`**

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "ESNext",
    "moduleResolution": "Bundler",
    "declaration": true,
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "outDir": "dist",
    "lib": ["ES2020", "DOM"]
  },
  "include": ["src"]
}
```

- [ ] **Step 3: Create `tsup.config.ts`**

```ts
import { defineConfig } from 'tsup'

export default defineConfig({
  entry: ['src/index.ts', 'src/worker-entry.ts'],
  format: ['esm', 'cjs'],
  dts: true,
  clean: true,
  sourcemap: true,
})
```

- [ ] **Step 4: Create `vitest.config.ts`**

```ts
import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    include: ['test/**/*.test.ts'],
  },
})
```

- [ ] **Step 5: Create `.gitignore`**

```
node_modules
dist
*.log
.DS_Store
demo/dist
demo/scripts/.cache
```

- [ ] **Step 6: Create `LICENSE`** (MIT, copyright holder: Krish)

```
MIT License

Copyright (c) 2026 Krish

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 7: Install and commit**

```bash
npm install
git add package.json tsconfig.json tsup.config.ts vitest.config.ts .gitignore LICENSE package-lock.json
git commit -m "chore: project scaffolding"
```

---

## Task 2: Math and RNG utilities

**Files:**
- Create: `src/types.ts`
- Create: `src/rng.ts`
- Create: `src/kmeansPlusPlus.ts`
- Create: `src/assign.ts`
- Test: `test/rng.test.ts`

**Interfaces:**
- Produces:
  - `mulberry32(seed: number): () => number`
  - `seededPermutation(n: number, seed: number): Int32Array`
  - `kmeansPlusPlusInit(data: Float32Array, dim: number, n: number, k: number, seed: number): Float32Array`
  - `assignBatch(data: Float32Array, dim: number, rowIndices: Int32Array, centroids: Float32Array, k: number): AssignResult`
  - `interface AssignResult { sums: Float64Array; counts: Float64Array; objective: number; labels: Int32Array }`
  - `interface ClusterResult { centroids: Float32Array; labels: Int32Array; loss: number; epochs: number }`
  - `interface KMeansOptions { seed?: number; maxEpochs?: number; init?: 'random' | 'kmeans++' }`
  - `interface EMeansOptions { seed?: number; nepochs?: number; batchSize?: number; alpha0?: number; batchLaw?: 'auto' | 'throughput' | 'quality'; unusedThreshold?: number; init?: 'random' | 'kmeans++' }`

- [ ] **Step 1: Write `src/types.ts`**

```ts
export interface ClusterResult {
  centroids: Float32Array
  labels: Int32Array
  loss: number
  epochs: number
}

export interface KMeansOptions {
  seed?: number
  maxEpochs?: number
  init?: 'random' | 'kmeans++'
}

export interface EMeansOptions {
  seed?: number
  nepochs?: number
  batchSize?: number
  alpha0?: number
  batchLaw?: 'auto' | 'throughput' | 'quality'
  unusedThreshold?: number
  init?: 'random' | 'kmeans++'
}
```

- [ ] **Step 2: Write the failing test for `rng.ts`**

Create `test/rng.test.ts`:

```ts
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
```

- [ ] **Step 3: Run it to verify it fails**

Run: `npx vitest run test/rng.test.ts`
Expected: FAIL — `src/rng.ts` does not exist.

- [ ] **Step 4: Write `src/rng.ts`**

```ts
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

export function seededPermutation(n: number, seed: number): Int32Array {
  const rand = mulberry32(seed)
  const order = new Int32Array(n)
  for (let i = 0; i < n; i++) order[i] = i
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1))
    const tmp = order[i]
    order[i] = order[j]
    order[j] = tmp
  }
  return order
}
```

- [ ] **Step 5: Run it to verify it passes**

Run: `npx vitest run test/rng.test.ts`
Expected: PASS

- [ ] **Step 6: Write `src/assign.ts`** (no separate test — exercised thoroughly by Tasks 3 and 5's tests)

```ts
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
```

- [ ] **Step 7: Write `src/kmeansPlusPlus.ts`**

```ts
import { mulberry32 } from './rng'

function squaredDistanceToCentroid(
  data: Float32Array,
  dim: number,
  row: number,
  centroids: Float32Array,
  c: number
): number {
  const rowOffset = row * dim
  const cOffset = c * dim
  let dist = 0
  for (let d = 0; d < dim; d++) {
    const diff = data[rowOffset + d] - centroids[cOffset + d]
    dist += diff * diff
  }
  return dist
}

export function kmeansPlusPlusInit(
  data: Float32Array,
  dim: number,
  n: number,
  k: number,
  seed: number
): Float32Array {
  const rand = mulberry32(seed)
  const centroids = new Float32Array(k * dim)
  const firstIdx = Math.floor(rand() * n)
  centroids.set(data.subarray(firstIdx * dim, firstIdx * dim + dim), 0)

  const closestDistSq = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    closestDistSq[i] = squaredDistanceToCentroid(data, dim, i, centroids, 0)
  }

  for (let c = 1; c < k; c++) {
    let total = 0
    for (let i = 0; i < n; i++) total += closestDistSq[i]
    let target = rand() * total
    let chosen = n - 1
    for (let i = 0; i < n; i++) {
      target -= closestDistSq[i]
      if (target <= 0) {
        chosen = i
        break
      }
    }
    centroids.set(data.subarray(chosen * dim, chosen * dim + dim), c * dim)
    for (let i = 0; i < n; i++) {
      const d = squaredDistanceToCentroid(data, dim, i, centroids, c)
      if (d < closestDistSq[i]) closestDistSq[i] = d
    }
  }
  return centroids
}
```

- [ ] **Step 8: Commit**

```bash
git add src/types.ts src/rng.ts src/assign.ts src/kmeansPlusPlus.ts test/rng.test.ts
git commit -m "feat: seeded RNG, k-means++ init, and batch assignment"
```

---

## Task 3: Baseline Lloyd's k-means

**Files:**
- Create: `src/kmeans.ts`
- Test: `test/kmeans.test.ts`

**Interfaces:**
- Consumes: `kmeansPlusPlusInit`, `seededPermutation`, `assignBatch`, `ClusterResult`, `KMeansOptions` (Task 2)
- Produces: `kmeans(data: Float32Array, dim: number, k: number, opts?: KMeansOptions): ClusterResult`

- [ ] **Step 1: Write the failing test**

Create `test/kmeans.test.ts`:

```ts
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `npx vitest run test/kmeans.test.ts`
Expected: FAIL — `src/kmeans.ts` does not exist.

- [ ] **Step 3: Write `src/kmeans.ts`**

```ts
import { assignBatch } from './assign'
import { kmeansPlusPlusInit } from './kmeansPlusPlus'
import { seededPermutation } from './rng'
import type { ClusterResult, KMeansOptions } from './types'

function randomInit(
  data: Float32Array,
  dim: number,
  n: number,
  k: number,
  seed: number
): Float32Array {
  const order = seededPermutation(n, seed)
  const centroids = new Float32Array(k * dim)
  for (let j = 0; j < k; j++) {
    const row = order[j]
    centroids.set(data.subarray(row * dim, row * dim + dim), j * dim)
  }
  return centroids
}

function centroidsEqual(a: Float32Array, b: Float32Array): boolean {
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false
  }
  return true
}

export function kmeans(
  data: Float32Array,
  dim: number,
  k: number,
  opts: KMeansOptions = {}
): ClusterResult {
  const n = data.length / dim
  const seed = opts.seed ?? 1234
  const maxEpochs = opts.maxEpochs ?? 25
  const init = opts.init ?? 'kmeans++'

  let centroids =
    init === 'kmeans++'
      ? kmeansPlusPlusInit(data, dim, n, k, seed)
      : randomInit(data, dim, n, k, seed)

  const allIndices = new Int32Array(n)
  for (let i = 0; i < n; i++) allIndices[i] = i

  let labels = new Int32Array(n)
  let loss = Infinity
  let epoch = 0
  for (; epoch < maxEpochs; epoch++) {
    const result = assignBatch(data, dim, allIndices, centroids, k)
    labels = result.labels
    loss = result.objective
    const newCentroids = new Float32Array(k * dim)
    for (let j = 0; j < k; j++) {
      const count = result.counts[j]
      const off = j * dim
      if (count > 0) {
        for (let d = 0; d < dim; d++) {
          newCentroids[off + d] = result.sums[off + d] / count
        }
      } else {
        newCentroids.set(centroids.subarray(off, off + dim), off)
      }
    }
    const converged = centroidsEqual(centroids, newCentroids)
    centroids = newCentroids
    if (converged) {
      epoch += 1
      break
    }
  }
  return { centroids, labels, loss, epochs: epoch }
}
```

- [ ] **Step 4: Run it to verify it passes**

Run: `npx vitest run test/kmeans.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/kmeans.ts test/kmeans.test.ts
git commit -m "feat: Lloyd's k-means baseline with k-means++ init"
```

---

## Task 4: e-means scaling-law and schedule helpers

These are ported directly from faiss's `contrib/e_means/base.py` — the constants and formulas below are copied verbatim from that file (fetched from `facebookresearch/faiss` on 2026-11-17), not reconstructed from the paper abstract.

**Files:**
- Create: `src/emeansSchedule.ts`
- Test: `test/emeansSchedule.test.ts`

**Interfaces:**
- Produces:
  - `roundPowerOfTwo(value: number): number`
  - `stepLaw(n: number, k: number, nepochs: number, batchSize: number): number`
  - `resolveBatchLaw(batchLaw: 'auto' | 'throughput' | 'quality', dim: number): 'throughput' | 'quality'`
  - `resolveUnusedThreshold(threshold: number | undefined, dim: number): number`
  - `resolveRecipe(n, k, nepochs, batchLaw, batchSizeOpt?, alpha0Opt?): { batchSize: number; alpha0: number }`
  - `epochSeed(seed: number, epoch: number): number`
  - `scheduledAlpha(alpha0: number, step: number, totalSteps: number): number`
  - `INIT_USAGE`, `SPLIT_EPSILON`, `CHECK_UNUSED_EVERY` constants

- [ ] **Step 1: Write the failing test**

Create `test/emeansSchedule.test.ts`:

```ts
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `npx vitest run test/emeansSchedule.test.ts`
Expected: FAIL — `src/emeansSchedule.ts` does not exist.

- [ ] **Step 3: Write `src/emeansSchedule.ts`**

```ts
export const ALPHA_MIN_FACTOR = 0.1
export const CHECK_UNUSED_EVERY = 5
export const INIT_USAGE = 2 ** -16
export const SPLIT_EPSILON = 1e-4
export const UNUSED_THRESHOLD = 0.1
export const NO_REVIVAL_MAX_D = 2
export const AUTO_QUALITY_MAX_D = 16

const BATCH_LAWS: Record<'throughput' | 'quality', [number, number, number, number]> = {
  throughput: [0.035134, 0.134749, 0.876473, 0.813344],
  quality: [0.00439175, 0.134749, 0.876473, 0.813344],
}

const ALPHA_LAW: [number, number, number, number, number] = [
  0.8244, -0.4617, 0.9003, -0.4498, -0.1057,
]

export function roundPowerOfTwo(value: number): number {
  if (value <= 1) return 1
  const low = 2 ** Math.floor(Math.log2(value))
  const high = 2 * low
  return value / low < high / value ? low : high
}

export function stepLaw(n: number, k: number, nepochs: number, batchSize: number): number {
  const [coefficient, kExp, batchExp, nExp, iterExp] = ALPHA_LAW
  const alpha0 =
    coefficient * k ** kExp * batchSize ** batchExp * n ** nExp * nepochs ** iterExp
  return Math.min(Math.max(alpha0, 1e-4), 1.0)
}

export function resolveBatchLaw(
  batchLaw: 'auto' | 'throughput' | 'quality',
  dim: number
): 'throughput' | 'quality' {
  if (batchLaw !== 'auto') return batchLaw
  return dim <= AUTO_QUALITY_MAX_D ? 'quality' : 'throughput'
}

export function resolveUnusedThreshold(threshold: number | undefined, dim: number): number {
  if (threshold === undefined) return dim <= NO_REVIVAL_MAX_D ? 0 : UNUSED_THRESHOLD
  if (threshold < 0) {
    throw new Error('unusedThreshold must be >= 0 (0 disables revival)')
  }
  return threshold
}

export function resolveRecipe(
  n: number,
  k: number,
  nepochs: number,
  batchLaw: 'throughput' | 'quality',
  batchSizeOpt: number | undefined,
  alpha0Opt: number | undefined
): { batchSize: number; alpha0: number } {
  let batchSize: number
  if (batchSizeOpt === undefined) {
    const [coefficient, kExp, nExp, iterExp] = BATCH_LAWS[batchLaw]
    const rawBatch = coefficient * k ** kExp * n ** nExp * nepochs ** iterExp
    batchSize = roundPowerOfTwo(rawBatch)
    batchSize =
      batchLaw === 'throughput'
        ? Math.min(batchSize, roundPowerOfTwo(n / 2))
        : Math.min(batchSize, Math.floor(n / 2))
    batchSize = Math.max(1, Math.min(batchSize, n))
  } else {
    batchSize = Math.min(Math.floor(batchSizeOpt), n)
  }
  const alpha0 = alpha0Opt ?? stepLaw(n, k, nepochs, batchSize)
  return { batchSize, alpha0 }
}

export function epochSeed(seed: number, epoch: number): number {
  return (seed * 1000003 + epoch) >>> 0
}

export function scheduledAlpha(alpha0: number, step: number, totalSteps: number): number {
  const fraction = Math.min(step, totalSteps) / Math.max(1, totalSteps)
  const cosine = 0.5 * (1 + Math.cos(Math.PI * fraction))
  return alpha0 * (ALPHA_MIN_FACTOR + (1 - ALPHA_MIN_FACTOR) * cosine)
}
```

- [ ] **Step 4: Run it to verify it passes**

Run: `npx vitest run test/emeansSchedule.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/emeansSchedule.ts test/emeansSchedule.test.ts
git commit -m "feat: e-means scaling-law and cosine-schedule helpers"
```

---

## Task 5: e-means core algorithm

**Files:**
- Create: `src/emeans.ts`
- Test: `test/emeans.test.ts`

**Interfaces:**
- Consumes: `assignBatch`, `kmeansPlusPlusInit`, `seededPermutation`, `mulberry32` (Task 2); `resolveBatchLaw`, `resolveRecipe`, `resolveUnusedThreshold`, `epochSeed`, `scheduledAlpha`, `INIT_USAGE`, `SPLIT_EPSILON`, `CHECK_UNUSED_EVERY` (Task 4); `ClusterResult`, `EMeansOptions` (Task 2)
- Produces:
  - `fit(data: Float32Array, dim: number, k: number, opts?: EMeansOptions): ClusterResult`
  - `predict(data: Float32Array, dim: number, centroids: Float32Array): Int32Array`

- [ ] **Step 1: Write the failing test**

Create `test/emeans.test.ts`:

```ts
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
    const emeansResult = fit(data, dim, 4, { seed: 1, nepochs: 5 })
    const kmeansResult = kmeans(data, dim, 4, { seed: 1, maxEpochs: 25 })
    expect(emeansResult.epochs).toBeLessThan(kmeansResult.epochs)
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `npx vitest run test/emeans.test.ts`
Expected: FAIL — `src/emeans.ts` does not exist.

- [ ] **Step 3: Write `src/emeans.ts`**

```ts
import { assignBatch } from './assign'
import { kmeansPlusPlusInit } from './kmeansPlusPlus'
import {
  CHECK_UNUSED_EVERY,
  INIT_USAGE,
  SPLIT_EPSILON,
  epochSeed,
  resolveBatchLaw,
  resolveRecipe,
  resolveUnusedThreshold,
  scheduledAlpha,
} from './emeansSchedule'
import { mulberry32, seededPermutation } from './rng'
import type { ClusterResult, EMeansOptions } from './types'

interface EMeansState {
  dim: number
  k: number
  centroids: Float32Array
  usage: Float32Array
}

function randomRowInit(
  data: Float32Array,
  dim: number,
  n: number,
  k: number,
  seed: number
): Float32Array {
  const order = seededPermutation(n, seed)
  const centroids = new Float32Array(k * dim)
  for (let j = 0; j < k; j++) {
    const row = order[j]
    centroids.set(data.subarray(row * dim, row * dim + dim), j * dim)
  }
  return centroids
}

function applyEmaUpdate(
  state: EMeansState,
  sums: Float64Array,
  counts: Float64Array,
  alpha: number
): void {
  const { k, dim, centroids, usage } = state
  const decay = 1 - alpha
  const replaceUsage = alpha === 1
  if (!replaceUsage) {
    for (let j = 0; j < k; j++) usage[j] *= decay
  }
  for (let j = 0; j < k; j++) {
    const count = counts[j]
    if (count <= 0) continue
    const retained = replaceUsage ? 0 : usage[j]
    const mass = retained + alpha * count
    const off = j * dim
    const weight = (alpha * count) / mass
    for (let d = 0; d < dim; d++) {
      const mean = sums[off + d] / count
      centroids[off + d] =
        weight === 1 ? mean : centroids[off + d] + weight * (mean - centroids[off + d])
    }
    usage[j] = mass
  }
}

function reviveUnused(state: EMeansState, rand: () => number, unusedThreshold: number): void {
  if (unusedThreshold <= 0) return
  const { k, dim, centroids, usage } = state
  let total = 0
  for (let j = 0; j < k; j++) total += usage[j]
  const threshold = (total * unusedThreshold) / k

  const unused: number[] = []
  const donors: number[] = []
  for (let j = 0; j < k; j++) {
    if (usage[j] < threshold) unused.push(j)
    else if (usage[j] > 0) donors.push(j)
  }
  const nsplit = Math.min(unused.length, donors.length)
  if (nsplit === 0) return

  const keyed = donors.map((j) => {
    const u = Math.max(rand(), Number.EPSILON)
    const key = Math.log(usage[j]) - Math.log(-Math.log(u))
    return { j, key }
  })
  keyed.sort((a, b) => b.key - a.key)
  const sources = keyed.slice(0, nsplit).map((e) => e.j)
  const targets = unused.slice(0, nsplit)

  for (let s = 0; s < nsplit; s++) {
    const source = sources[s]
    const target = targets[s]
    const srcOff = source * dim
    const tgtOff = target * dim
    let identical = true
    for (let d = 0; d < dim; d++) {
      const sign = d % 2 === 0 ? 1 : -1
      const value = centroids[srcOff + d]
      const donorValue = value * (1 - SPLIT_EPSILON * sign)
      const childValue = value * (1 + SPLIT_EPSILON * sign)
      centroids[srcOff + d] = donorValue
      centroids[tgtOff + d] = childValue
      if (donorValue !== childValue) identical = false
    }
    if (identical) {
      centroids[tgtOff] = Math.fround(centroids[tgtOff] + Number.EPSILON)
    }
    const halfUsage = usage[source] / 2
    usage[source] = halfUsage
    usage[target] = halfUsage
  }
}

function finalize(
  centroids: Float32Array,
  passSums: Float64Array,
  passCounts: Float64Array,
  k: number,
  dim: number
): Float32Array {
  const result = new Float32Array(centroids)
  for (let j = 0; j < k; j++) {
    const count = passCounts[j]
    if (count <= 0) continue
    const off = j * dim
    for (let d = 0; d < dim; d++) {
      result[off + d] = passSums[off + d] / count
    }
  }
  return result
}

export function fit(
  data: Float32Array,
  dim: number,
  k: number,
  opts: EMeansOptions = {}
): ClusterResult {
  const n = data.length / dim
  const seed = opts.seed ?? 1234
  const nepochs = Math.max(1, Math.floor(opts.nepochs ?? 5))
  const init = opts.init ?? 'random'

  const batchLaw = resolveBatchLaw(opts.batchLaw ?? 'auto', dim)
  const unusedThreshold = resolveUnusedThreshold(opts.unusedThreshold, dim)
  const { batchSize, alpha0 } = resolveRecipe(n, k, nepochs, batchLaw, opts.batchSize, opts.alpha0)

  const centroids =
    init === 'kmeans++'
      ? kmeansPlusPlusInit(data, dim, n, k, seed)
      : randomRowInit(data, dim, n, k, seed)
  const usage = new Float32Array(k).fill(INIT_USAGE)
  const state: EMeansState = { dim, k, centroids, usage }

  const stepsPerEpoch = Math.ceil(n / batchSize)
  const totalSteps = stepsPerEpoch * nepochs
  const splitRand = mulberry32(seed + 1)

  let step = 0
  const labels = new Int32Array(n)
  let loss = Infinity
  let passSums: Float64Array | null = null
  let passCounts: Float64Array | null = null

  for (let epoch = 0; epoch < nepochs; epoch++) {
    const order = seededPermutation(n, epochSeed(seed, epoch))
    const isLastEpoch = epoch === nepochs - 1
    if (isLastEpoch) {
      passSums = new Float64Array(k * dim)
      passCounts = new Float64Array(k)
    }
    let objective = 0
    for (let begin = 0; begin < n; begin += batchSize) {
      const end = Math.min(begin + batchSize, n)
      const batchIndices = order.subarray(begin, end)
      const result = assignBatch(data, dim, batchIndices, state.centroids, k)
      objective += result.objective

      if (isLastEpoch) {
        for (let r = 0; r < batchIndices.length; r++) {
          labels[batchIndices[r]] = result.labels[r]
        }
        for (let j = 0; j < k; j++) {
          passCounts![j] += result.counts[j]
          const off = j * dim
          for (let d = 0; d < dim; d++) {
            passSums![off + d] += result.sums[off + d]
          }
        }
      }

      const alpha = scheduledAlpha(alpha0, step, totalSteps)
      applyEmaUpdate(state, result.sums, result.counts, alpha)

      step += 1
      if (step % CHECK_UNUSED_EVERY === 0) {
        reviveUnused(state, splitRand, unusedThreshold)
      }
    }
    loss = objective
  }

  const finalCentroids = finalize(state.centroids, passSums!, passCounts!, k, dim)
  return { centroids: finalCentroids, labels, loss, epochs: nepochs }
}

export function predict(data: Float32Array, dim: number, centroids: Float32Array): Int32Array {
  const n = data.length / dim
  const k = centroids.length / dim
  const allIndices = new Int32Array(n)
  for (let i = 0; i < n; i++) allIndices[i] = i
  return assignBatch(data, dim, allIndices, centroids, k).labels
}
```

- [ ] **Step 4: Run it to verify it passes**

Run: `npx vitest run test/emeans.test.ts`
Expected: PASS. If the "within 2x loss" assertion is flaky, re-run a few times — `Math.random()` jitter in `makeBlobs` varies run to run; if it fails consistently, loosen the multiplier to 3x rather than changing the algorithm.

- [ ] **Step 5: Commit**

```bash
git add src/emeans.ts test/emeans.test.ts
git commit -m "feat: e-means core algorithm (mini-batch EMA update, dead-cluster revival, finalize)"
```

---

## Task 6: Public API and build verification

**Files:**
- Create: `src/index.ts`

**Interfaces:**
- Consumes: `fit`, `predict` (Task 5), `kmeans` (Task 3), `runInWorker` (Task 8 — see note below)
- Produces: the package's public entry point

**Note:** `runInWorker` doesn't exist until Task 8. Write `index.ts` now without it, then come back and add the export in Task 8 Step 3. This task just proves the build pipeline works end-to-end before more code is layered on.

- [ ] **Step 1: Write `src/index.ts`**

```ts
export { fit, predict } from './emeans'
export { kmeans } from './kmeans'
export type { ClusterResult, EMeansOptions, KMeansOptions } from './types'
```

- [ ] **Step 2: Run the build**

Run: `npm run build`
Expected: `dist/index.js`, `dist/index.cjs`, `dist/index.d.ts` are created with no errors.

- [ ] **Step 3: Run the full test suite**

Run: `npm test`
Expected: all tests from Tasks 2-5 pass.

- [ ] **Step 4: Commit**

```bash
git add src/index.ts
git commit -m "feat: public package entry point"
```

---

## Task 7: Correctness fixture vs real faiss e-means

**Files:**
- Create: `scripts/generate-faiss-fixture.py`
- Create: `test/fixtures/emeans-faiss-reference.json` (generated by the script, then committed)
- Test: `test/emeans.faiss-fixture.test.ts`

**Interfaces:**
- Consumes: `fit` (Task 5)
- Produces: a committed fixture file the test suite reads; no Python/faiss needed to run `npm test`.

- [ ] **Step 1: Write `scripts/generate-faiss-fixture.py`**

```python
"""One-time script: generates the faiss e-means reference fixture used by
test/emeans.faiss-fixture.test.ts. Not run in CI and not a build dependency --
only re-run this if the fixture's input data or options change.

Prerequisites:
    pip install faiss-cpu numpy

Run:
    python scripts/generate-faiss-fixture.py
"""
import json

import numpy as np
from faiss.contrib.e_means import NumpyEmeans

rng = np.random.RandomState(42)
centers = np.array([[0, 0], [10, 10], [0, 10], [10, 0]], dtype="float32")
data = np.concatenate(
    [c + rng.normal(scale=0.25, size=(200, 2)).astype("float32") for c in centers]
)

model = NumpyEmeans(d=2, k=4, nepochs=5, seed=42)
centroids, labels = model.train(data)

fixture = {
    "data": data.tolist(),
    "dim": 2,
    "k": 4,
    "nepochs": 5,
    "seed": 42,
    "finalLoss": float(model.obj[-1]),
    "lossPerEpoch": model.obj.tolist(),
}

with open("test/fixtures/emeans-faiss-reference.json", "w") as f:
    json.dump(fixture, f, indent=2)

print("Wrote test/fixtures/emeans-faiss-reference.json")
print("Loss per epoch:", fixture["lossPerEpoch"])
```

- [ ] **Step 2: Generate the fixture**

```bash
mkdir -p test/fixtures
pip install faiss-cpu numpy
python scripts/generate-faiss-fixture.py
```

Expected: `test/fixtures/emeans-faiss-reference.json` is created with real faiss output.

- [ ] **Step 3: Write the failing test**

Create `test/emeans.faiss-fixture.test.ts`:

```ts
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
```

- [ ] **Step 4: Run it**

Run: `npx vitest run test/emeans.faiss-fixture.test.ts`
Expected: PASS. If the ratio is outside 25% but in the right ballpark (same order of magnitude, trending the same way across epochs), widen the tolerance in the test rather than reworking the algorithm — the TS port isn't expected to bit-match NumPy's PRNG. If the ratio is wildly off (e.g. 5x+), that's a real bug in Task 5's `applyEmaUpdate` or `reviveUnused` — recheck them against `emeans_ref.py`'s `_apply_statistics` and `_choose_splits`/`_apply_splits`.

- [ ] **Step 5: Commit**

```bash
git add scripts/generate-faiss-fixture.py test/fixtures/emeans-faiss-reference.json test/emeans.faiss-fixture.test.ts
git commit -m "test: validate fit() loss against real faiss e-means output"
```

---

## Task 8: Web Worker helper

**Files:**
- Create: `src/worker-entry.ts`
- Create: `src/worker.ts`
- Modify: `src/index.ts` (add `runInWorker` export)
- Test: `test/worker.test.ts`

**Interfaces:**
- Consumes: `fit`, `kmeans` (Tasks 3, 5)
- Produces: `runInWorker(algorithm: 'emeans' | 'kmeans', data: Float32Array, dim: number, k: number, opts?: EMeansOptions | KMeansOptions): Promise<ClusterResult>`

- [ ] **Step 1: Write `src/worker-entry.ts`**

```ts
/// <reference lib="webworker" />
import { fit } from './emeans'
import { kmeans } from './kmeans'
import type { EMeansOptions, KMeansOptions } from './types'

export interface WorkerRequest {
  algorithm: 'emeans' | 'kmeans'
  data: Float32Array
  dim: number
  k: number
  opts?: EMeansOptions | KMeansOptions
}

self.onmessage = (event: MessageEvent<WorkerRequest>) => {
  const { algorithm, data, dim, k, opts } = event.data
  const result = algorithm === 'emeans' ? fit(data, dim, k, opts) : kmeans(data, dim, k, opts)
  ;(self as unknown as Worker).postMessage(result, {
    transfer: [result.centroids.buffer, result.labels.buffer],
  })
}
```

- [ ] **Step 2: Write the failing test**

Create `test/worker.test.ts`:

```ts
import { describe, expect, it } from 'vitest'
import { runInWorker } from '../src/worker'

describe('runInWorker', () => {
  it('throws a clear error when the Worker API is unavailable', async () => {
    await expect(
      runInWorker('kmeans', new Float32Array([0, 0, 1, 1]), 2, 2)
    ).rejects.toThrow(/Worker API/)
  })
})
```

- [ ] **Step 3: Run it to verify it fails**

Run: `npx vitest run test/worker.test.ts`
Expected: FAIL — `src/worker.ts` does not exist.

- [ ] **Step 4: Write `src/worker.ts`**

```ts
import type { ClusterResult, EMeansOptions, KMeansOptions } from './types'
import type { WorkerRequest } from './worker-entry'

export function runInWorker(
  algorithm: 'emeans' | 'kmeans',
  data: Float32Array,
  dim: number,
  k: number,
  opts?: EMeansOptions | KMeansOptions
): Promise<ClusterResult> {
  if (typeof Worker === 'undefined') {
    return Promise.reject(
      new Error(
        'runInWorker requires the Worker API (browser or a bundler that polyfills it); call fit()/kmeans() directly in Node.'
      )
    )
  }
  const worker = new Worker(new URL('./worker-entry.js', import.meta.url), {
    type: 'module',
  })
  return new Promise((resolve, reject) => {
    worker.onmessage = (event: MessageEvent<ClusterResult>) => {
      resolve(event.data)
      worker.terminate()
    }
    worker.onerror = (event) => {
      reject(event.error ?? new Error('e-means worker failed'))
      worker.terminate()
    }
    const request: WorkerRequest = { algorithm, data, dim, k, opts }
    worker.postMessage(request, { transfer: [data.buffer] })
  })
}
```

- [ ] **Step 5: Add the export to `src/index.ts`**

```ts
export { fit, predict } from './emeans'
export { kmeans } from './kmeans'
export { runInWorker } from './worker'
export type { ClusterResult, EMeansOptions, KMeansOptions } from './types'
```

- [ ] **Step 6: Run it to verify it passes**

Run: `npx vitest run test/worker.test.ts`
Expected: PASS (Node's default vitest environment has no global `Worker`, so the error path is exercised).

- [ ] **Step 7: Rebuild and commit**

```bash
npm run build
git add src/worker-entry.ts src/worker.ts src/index.ts test/worker.test.ts
git commit -m "feat: Web Worker helper for browser use"
```

---

## Task 9: Benchmark script

**Files:**
- Create: `bench/run.ts`

**Interfaces:**
- Consumes: `fit` (Task 5), `kmeans` (Task 3), `ml-kmeans`'s `kmeans` export

- [ ] **Step 1: Write `bench/run.ts`**

```ts
import { performance } from 'node:perf_hooks'
import { kmeans as mlKmeans } from 'ml-kmeans'
import { fit } from '../src/emeans'
import { kmeans } from '../src/kmeans'

function makeBlobs(n: number, dim: number, k: number, seed: number): Float32Array {
  let state = seed
  const rand = (): number => {
    state = (state * 1103515245 + 12345) >>> 0
    return state / 0xffffffff
  }
  const data = new Float32Array(n * dim)
  for (let i = 0; i < n; i++) {
    const cluster = i % k
    for (let d = 0; d < dim; d++) {
      data[i * dim + d] = cluster * 10 + (rand() - 0.5)
    }
  }
  return data
}

function toPoints(data: Float32Array, dim: number): number[][] {
  const n = data.length / dim
  const points: number[][] = []
  for (let i = 0; i < n; i++) {
    points.push(Array.from(data.subarray(i * dim, i * dim + dim)))
  }
  return points
}

function run(n: number, dim: number, k: number): void {
  const data = makeBlobs(n, dim, k, 1)

  const emeansStart = performance.now()
  const emeansResult = fit(data, dim, k, { seed: 1, nepochs: 5 })
  const emeansMs = performance.now() - emeansStart

  const kmeansStart = performance.now()
  const kmeansResult = kmeans(data, dim, k, { seed: 1, maxEpochs: 25 })
  const kmeansMs = performance.now() - kmeansStart

  const mlStart = performance.now()
  const mlResult = mlKmeans(toPoints(data, dim), k, {
    initialization: 'kmeans++',
    maxIterations: 25,
  })
  const mlMs = performance.now() - mlStart

  console.log(`\nn=${n} dim=${dim} k=${k}`)
  console.log(
    `  e-means:       ${emeansMs.toFixed(1)}ms, ${emeansResult.epochs} epochs, loss=${emeansResult.loss.toFixed(1)}`
  )
  console.log(
    `  kmeans (ours): ${kmeansMs.toFixed(1)}ms, ${kmeansResult.epochs} epochs, loss=${kmeansResult.loss.toFixed(1)}`
  )
  console.log(
    `  ml-kmeans:     ${mlMs.toFixed(1)}ms, ${mlResult.iterations} iterations, converged=${mlResult.converged}`
  )
}

const configs: Array<[number, number, number]> = [
  [10_000, 32, 10],
  [100_000, 32, 20],
  [1_000_000, 32, 50],
]
for (const [n, dim, k] of configs) run(n, dim, k)
```

- [ ] **Step 2: Run it and capture output**

Run: `npm run bench`
Expected: prints timing/loss/epoch numbers for all three configurations. Save the output — it feeds directly into Task 10's README benchmark table.

- [ ] **Step 3: Commit**

```bash
git add bench/run.ts
git commit -m "feat: add epochs/wall-clock benchmark script"
```

---

## Task 10: README (install, API, honest benchmarks, SEO usage guides)

**Files:**
- Create: `README.md`

**Interfaces:**
- Consumes: the public API from Task 6/8, and the benchmark output from Task 9.

- [ ] **Step 1: Write `README.md`**

```markdown
# e-means

Zero-dependency TypeScript port of **e-means**, Meta FAIR's mini-batch
replacement for k-means ([Bessa et al., 2026](https://arxiv.org/abs/2611.09501)),
ported from the real [faiss implementation](https://github.com/facebookresearch/faiss/tree/main/contrib/e_means) —
not reconstructed from the paper abstract.

e-means reaches a given clustering loss in far fewer epochs than Lloyd's
k-means, using a mini-batch update rule based on exponential moving averages,
with no extra hyperparameters to tune. **This package reproduces that
algorithmic win in plain TypeScript — it is not a binding to faiss's
PyTorch/CUTLASS GPU kernels, and does not claim to match their raw wall-clock
speed.** What it does give you: fast, dependency-free clustering that runs
anywhere JavaScript runs, including places faiss/scikit-learn can't reach at
all — Cloudflare Workers, Vercel Edge Functions, and the browser.

```bash
npm install e-means
```

## Quick start

```ts
import { fit } from 'e-means'

// data is a flat, row-major Float32Array: n points * dim
const { centroids, labels, loss, epochs } = fit(data, dim, k)
```

A plain Lloyd's k-means (k-means++ init) ships in the same package as a
baseline and because you'll want it anyway:

```ts
import { kmeans } from 'e-means'

const { centroids, labels } = kmeans(data, dim, k)
```

## API

### `fit(data, dim, k, opts?) -> ClusterResult`

e-means. `opts`: `{ seed?, nepochs? (default 5), batchSize?, alpha0?,
batchLaw? ('auto' | 'throughput' | 'quality'), unusedThreshold?, init?
('random' | 'kmeans++') }`.

### `kmeans(data, dim, k, opts?) -> ClusterResult`

Lloyd's algorithm, k-means++ initialized. `opts`: `{ seed?, maxEpochs?
(default 25), init? }`.

### `predict(data, dim, centroids) -> Int32Array`

Assigns new points to the nearest centroid from a previous `fit`/`kmeans` call.

### `runInWorker(algorithm, data, dim, k, opts?) -> Promise<ClusterResult>`

Runs `fit` or `kmeans` in a Web Worker so browser callers don't block the
main thread. Requires a bundler that supports `new URL(..., import.meta.url)`
worker syntax (Vite, webpack 5+).

`ClusterResult`: `{ centroids: Float32Array, labels: Int32Array, loss:
number, epochs: number }`.

## Benchmarks

Epochs and wall-clock time to reach a stable clustering, `e-means` (5
epochs) vs. this package's own `kmeans` (25 epochs, Lloyd's) vs.
[`ml-kmeans`](https://www.npmjs.com/package/ml-kmeans) (25 iterations),
on synthetic blob data, Node.js:

<!-- Fill in from `npm run bench` output (Task 9). Table columns: n, dim, k, e-means ms/epochs/loss, kmeans ms/epochs/loss, ml-kmeans ms/loss. -->

| n | dim | k | e-means | kmeans (ours) | ml-kmeans |
|---|-----|---|---------|----------------|-----------|
| 10,000 | 32 | 10 | _fill in_ | _fill in_ | _fill in_ |
| 100,000 | 32 | 20 | _fill in_ | _fill in_ | _fill in_ |
| 1,000,000 | 32 | 50 | _fill in_ | _fill in_ | _fill in_ |

The comparison is algorithmic (epochs/loss), not a claim of matching
faiss's CUTLASS-accelerated wall-clock speed — see the note at the top of
this README.

[Live demo: Shakespeare word-cloud race](https://<github-username>.github.io/e-means-ts/)

## Using it alongside your stack

e-means has no framework dependencies — it takes `Float32Array` vectors in
and returns cluster assignments out. Here's how that looks with the tools
people most often pair with embeddings.

### Node.js / server-side RAG pipelines

```ts
import { fit } from 'e-means'

const embeddings = await embedAllChunks(chunks) // however you already embed
const flat = new Float32Array(embeddings.flat())
const { labels, centroids } = fit(flat, embeddings[0].length, k)
```

### LangChain.js

Cluster the embeddings behind any LangChain vector store to deduplicate
near-identical documents or group retrieved results before building a
chain/prompt:

```ts
import { OpenAIEmbeddings } from '@langchain/openai'
import { fit } from 'e-means'

const embeddings = new OpenAIEmbeddings()
const vectors = await embeddings.embedDocuments(docs.map((d) => d.pageContent))
const flat = new Float32Array(vectors.flat())
const { labels } = fit(flat, vectors[0].length, k)
// group docs[i] by labels[i]
```

### LlamaIndex.TS

```ts
import { OpenAIEmbedding } from 'llamaindex'
import { fit } from 'e-means'

const embedModel = new OpenAIEmbedding()
const vectors = await Promise.all(nodes.map((n) => embedModel.getTextEmbedding(n.text)))
const flat = new Float32Array(vectors.flat())
const { labels } = fit(flat, vectors[0].length, k)
```

### Vercel AI SDK

```ts
import { openai } from '@ai-sdk/openai'
import { embedMany } from 'ai'
import { fit } from 'e-means'

const { embeddings } = await embedMany({
  model: openai.embedding('text-embedding-3-small'),
  values: documents,
})
const flat = new Float32Array(embeddings.flat())
const { labels } = fit(flat, embeddings[0].length, k)
```

### Cloudflare Workers / Vercel Edge Functions

No native deps, no WASM to load — e-means runs directly in edge runtimes
where faiss/scikit-learn aren't an option at all:

```ts
export default {
  async fetch(request: Request): Promise<Response> {
    const { vectors, k } = await request.json()
    const { fit } = await import('e-means')
    const flat = new Float32Array(vectors.flat())
    const { labels } = fit(flat, vectors[0].length, k)
    return Response.json({ labels: Array.from(labels) })
  },
}
```

### Browser (off the main thread)

```ts
import { runInWorker } from 'e-means'

const { labels, centroids } = await runInWorker('emeans', flat, dim, k)
```

### Deno

```ts
import { fit } from 'npm:e-means'
```

### Building IVF-style partitions for a JS-native vector store (e.g. sqlite-vec)

```ts
import { fit } from 'e-means'

const { centroids, labels } = fit(allEmbeddings, dim, numPartitions)
// store labels[i] as a partition_id column alongside embedding i,
// and centroids as your partition lookup table
```

## What's not in v1

GPU acceleration, streaming `partialFit()` for out-of-core data, and
RAPTOR/RAG-framework adapter packages are intentionally out of scope for
this release. See `docs/design/2026-11-17-e-means-ts-design.md` for why.

## License

MIT
```

- [ ] **Step 2: Fill in the benchmark table from Task 9's output, then commit**

```bash
git add README.md
git commit -m "docs: add README with API reference, honest benchmarks, and per-framework usage guides"
```

---

## Task 11: Demo scaffolding + CI/deploy workflows

**Files:**
- Create: `demo/package.json`
- Create: `demo/vite.config.ts`
- Create: `demo/index.html`
- Create: `demo/src/style.css`
- Create: `.github/workflows/ci.yml`
- Create: `.github/workflows/deploy-demo.yml`

**Interfaces:**
- Produces: a runnable `npm run dev` in `demo/` that imports `../src/index.ts` directly, and GitHub Actions that run tests on push and deploy `demo/` to GitHub Pages on `main`.

- [ ] **Step 1: Write `demo/package.json`**

```json
{
  "name": "e-means-demo",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "build:data": "tsx scripts/build-shakespeare-data.ts"
  },
  "devDependencies": {
    "tsx": "^4.19.0",
    "typescript": "^5.7.0",
    "vite": "^6.0.0"
  }
}
```

- [ ] **Step 2: Write `demo/vite.config.ts`**

```ts
import { fileURLToPath } from 'node:url'
import { defineConfig } from 'vite'

export default defineConfig({
  base: '/e-means-ts/',
  resolve: {
    alias: {
      'e-means': fileURLToPath(new URL('../src/index.ts', import.meta.url)),
    },
  },
})
```

- [ ] **Step 3: Write `demo/index.html`**

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <title>e-means: 10x fewer epochs than k-means</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <link rel="stylesheet" href="/src/style.css" />
  </head>
  <body>
    <div id="app"></div>
    <div id="tooltip" hidden></div>
    <script type="module" src="/src/main.ts"></script>
  </body>
</html>
```

- [ ] **Step 4: Write `demo/src/style.css`**

```css
body {
  font-family: system-ui, sans-serif;
  margin: 2rem auto;
  max-width: 1100px;
  padding: 0 1rem;
}
nav button {
  margin-right: 0.5rem;
  padding: 0.5rem 1rem;
  cursor: pointer;
}
nav button.active {
  font-weight: bold;
  text-decoration: underline;
}
.race {
  display: flex;
  gap: 1.5rem;
  flex-wrap: wrap;
}
.panel {
  text-align: center;
}
canvas {
  border: 1px solid #ddd;
  border-radius: 8px;
}
#tooltip {
  position: fixed;
  background: #111;
  color: #fff;
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 12px;
  pointer-events: none;
}
textarea {
  width: 100%;
  font-family: monospace;
}
```

- [ ] **Step 5: Write `.github/workflows/ci.yml`**

```yaml
name: CI
on:
  push:
    branches: [main]
  pull_request:
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
      - run: npm install
      - run: npm run build
      - run: npm test
```

- [ ] **Step 6: Write `.github/workflows/deploy-demo.yml`**

```yaml
name: Deploy demo
on:
  push:
    branches: [main]
permissions:
  contents: read
  pages: write
  id-token: write
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
      - run: npm install
      - working-directory: demo
        run: npm install
      - working-directory: demo
        run: npm run build
      - uses: actions/upload-pages-artifact@v3
        with:
          path: demo/dist
  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - id: deployment
        uses: actions/deploy-pages@v4
```

- [ ] **Step 7: Install demo deps and verify the dev server boots**

```bash
cd demo && npm install && npm run dev
```

Expected: Vite starts without errors (the page will be blank until Task 13 writes `main.ts` — that's expected at this point).

- [ ] **Step 8: Commit**

```bash
cd ..
git add demo/package.json demo/vite.config.ts demo/index.html demo/src/style.css .github/workflows/ci.yml .github/workflows/deploy-demo.yml demo/package-lock.json
git commit -m "chore: demo scaffolding and CI/Pages deploy workflows"
```

---

## Task 12: Demo data — Shakespeare word embeddings

**Files:**
- Create: `demo/scripts/build-shakespeare-data.ts`
- Create: `demo/src/data/shakespeare-embeddings.json` (generated by the script, then committed)

**Interfaces:**
- Produces: a JSON array of `{ word: string; vector: number[]; x: number; y: number }`, consumed by Task 13's `main.ts`.

- [ ] **Step 1: Download the one-time inputs (not committed — cached locally)**

```bash
mkdir -p demo/scripts/.cache
curl -o demo/scripts/.cache/shakespeare.txt https://www.gutenberg.org/files/100/100-0.txt
curl -o demo/scripts/.cache/glove.zip https://nlp.stanford.edu/data/glove.6B.zip
cd demo/scripts/.cache && unzip -o glove.zip glove.6B.50d.txt && cd ../../..
```

- [ ] **Step 2: Write `demo/scripts/build-shakespeare-data.ts`**

```ts
import { createReadStream, readFileSync, writeFileSync } from 'node:fs'
import { createInterface } from 'node:readline'

const STOPWORDS = new Set([
  'the', 'and', 'to', 'of', 'a', 'i', 'in', 'that', 'you', 'is', 'my',
  'not', 'with', 'for', 'this', 'but', 'his', 'be', 'your', 'he', 'it',
  'as', 'have', 'thou', 'thy', 'him', 'will', 'so', 'me', 'what', 'no',
  'thee', 'her', 'do', 'if', 'we', 'are', 'on', 'all', 'by', 'shall',
  'then', 'our', 'their', 'which', 'or', 'was', 'they', 'from', 'at',
  'more', 'than', 'would', 'am', 'did', 'an', 'now', 'come', 'like',
])
const VOCAB_SIZE = 600

function topWords(text: string): string[] {
  const counts = new Map<string, number>()
  const tokens = text.toLowerCase().match(/[a-z']+/g) ?? []
  for (const token of tokens) {
    if (token.length < 4 || STOPWORDS.has(token)) continue
    counts.set(token, (counts.get(token) ?? 0) + 1)
  }
  return Array.from(counts.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, VOCAB_SIZE)
    .map(([word]) => word)
}

async function loadVectors(words: Set<string>): Promise<Map<string, number[]>> {
  const vectors = new Map<string, number[]>()
  const stream = createInterface({
    input: createReadStream('scripts/.cache/glove.6B.50d.txt'),
  })
  for await (const line of stream) {
    const spaceIndex = line.indexOf(' ')
    const word = line.slice(0, spaceIndex)
    if (words.has(word)) {
      vectors.set(word, line.slice(spaceIndex + 1).split(' ').map(Number))
    }
  }
  return vectors
}

function projectTo2D(vectors: number[][]): { x: number; y: number }[] {
  const n = vectors.length
  const dim = vectors[0].length
  const mean = new Array(dim).fill(0)
  for (const v of vectors) for (let d = 0; d < dim; d++) mean[d] += v[d] / n
  const centered = vectors.map((v) => v.map((x, d) => x - mean[d]))

  function matVec(matrix: number[][], v: number[]): number[] {
    const result = new Array(dim).fill(0)
    for (const row of matrix) {
      const dot = row.reduce((sum, x, d) => sum + x * v[d], 0)
      for (let d = 0; d < dim; d++) result[d] += dot * row[d]
    }
    return result
  }

  function powerIteration(exclude: number[] | null): number[] {
    let v = Array.from({ length: dim }, () => Math.random())
    for (let iter = 0; iter < 100; iter++) {
      let next = matVec(centered, v)
      if (exclude) {
        const proj = next.reduce((sum, x, d) => sum + x * exclude[d], 0)
        next = next.map((x, d) => x - proj * exclude[d])
      }
      const norm = Math.sqrt(next.reduce((sum, x) => sum + x * x, 0))
      v = next.map((x) => x / norm)
    }
    return v
  }

  const pc1 = powerIteration(null)
  const pc2 = powerIteration(pc1)

  return centered.map((v) => ({
    x: v.reduce((sum, x, d) => sum + x * pc1[d], 0),
    y: v.reduce((sum, x, d) => sum + x * pc2[d], 0),
  }))
}

async function main(): Promise<void> {
  const text = readFileSync('scripts/.cache/shakespeare.txt', 'utf-8')
  const candidates = topWords(text)
  const vectorsByWord = await loadVectors(new Set(candidates))
  const words = candidates.filter((w) => vectorsByWord.has(w))
  const vectors = words.map((w) => vectorsByWord.get(w)!)
  const projected = projectTo2D(vectors)

  const output = words.map((word, i) => ({
    word,
    vector: vectors[i],
    x: projected[i].x,
    y: projected[i].y,
  }))

  writeFileSync('src/data/shakespeare-embeddings.json', JSON.stringify(output))
  console.log(`Wrote ${output.length} words to src/data/shakespeare-embeddings.json`)
}

main()
```

- [ ] **Step 3: Run it**

```bash
mkdir -p demo/src/data
cd demo && npm run build:data
```

Expected: `demo/src/data/shakespeare-embeddings.json` is created with ~400-600 word entries (some candidates won't have a GloVe vector and get filtered out — that's expected).

- [ ] **Step 4: Commit**

```bash
cd ..
git add demo/scripts/build-shakespeare-data.ts demo/src/data/shakespeare-embeddings.json
git commit -m "feat: generate Shakespeare word-embedding dataset for the demo"
```

---

## Task 13: Demo UI — both tabs

**Files:**
- Create: `demo/src/main.ts`

**Interfaces:**
- Consumes: `fit`, `kmeans` (via the `e-means` alias from Task 11); `demo/src/data/shakespeare-embeddings.json` (Task 12)

- [ ] **Step 1: Write `demo/src/main.ts`**

```ts
import { fit, kmeans } from 'e-means'
import shakespeareData from './data/shakespeare-embeddings.json'

interface WordPoint {
  word: string
  vector: number[]
  x: number
  y: number
}

interface PlotPoint {
  word: string
  x: number
  y: number
}

const shakespearePoints = shakespeareData as WordPoint[]
const shakespeareDim = shakespearePoints[0].vector.length
const shakespeareFlat = new Float32Array(shakespearePoints.flatMap((p) => p.vector))

const app = document.querySelector<HTMLDivElement>('#app')!
app.innerHTML = `
  <header>
    <h1>e-means vs k-means</h1>
    <p>Same data, same number of clusters. e-means gets there in far fewer epochs.</p>
    <nav>
      <button data-tab="shakespeare" class="active">Shakespeare word cloud</button>
      <button data-tab="custom">Bring your own data</button>
    </nav>
  </header>
  <section id="shakespeare-tab">
    <button id="run-shakespeare">Run the race</button>
    <div class="race">
      <div class="panel">
        <h2>e-means</h2>
        <canvas id="emeans-canvas" width="480" height="480"></canvas>
        <p id="emeans-stats"></p>
      </div>
      <div class="panel">
        <h2>k-means</h2>
        <canvas id="kmeans-canvas" width="480" height="480"></canvas>
        <p id="kmeans-stats"></p>
      </div>
    </div>
  </section>
  <section id="custom-tab" hidden>
    <textarea id="custom-input" rows="6" placeholder="Paste JSON: [[0.1, 0.2], [0.4, 0.9], ...]"></textarea>
    <label>k: <input id="custom-k" type="number" value="3" min="2" /></label>
    <button id="run-custom">Run the race</button>
    <div class="race">
      <div class="panel">
        <h2>e-means</h2>
        <canvas id="custom-emeans-canvas" width="480" height="480"></canvas>
        <p id="custom-emeans-stats"></p>
      </div>
      <div class="panel">
        <h2>k-means</h2>
        <canvas id="custom-kmeans-canvas" width="480" height="480"></canvas>
        <p id="custom-kmeans-stats"></p>
      </div>
    </div>
  </section>
`

const COLORS = [
  '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
  '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990',
]

function plotPosition(
  p: PlotPoint,
  pts: PlotPoint[],
  canvasWidth: number,
  canvasHeight: number
): { x: number; y: number } {
  const xs = pts.map((q) => q.x)
  const ys = pts.map((q) => q.y)
  const minX = Math.min(...xs)
  const maxX = Math.max(...xs)
  const minY = Math.min(...ys)
  const maxY = Math.max(...ys)
  return {
    x: ((p.x - minX) / (maxX - minX || 1)) * (canvasWidth - 40) + 20,
    y: ((p.y - minY) / (maxY - minY || 1)) * (canvasHeight - 40) + 20,
  }
}

function render(canvas: HTMLCanvasElement, pts: PlotPoint[], labels: Int32Array): void {
  const ctx = canvas.getContext('2d')!
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  pts.forEach((p, i) => {
    const { x, y } = plotPosition(p, pts, canvas.width, canvas.height)
    ctx.fillStyle = COLORS[labels[i] % COLORS.length]
    ctx.beginPath()
    ctx.arc(x, y, 4, 0, Math.PI * 2)
    ctx.fill()
  })
}

function attachHover(canvas: HTMLCanvasElement, pts: PlotPoint[]): void {
  const tooltip = document.querySelector<HTMLDivElement>('#tooltip')!
  canvas.addEventListener('mousemove', (event) => {
    const rect = canvas.getBoundingClientRect()
    const mx = event.clientX - rect.left
    const my = event.clientY - rect.top
    let nearest: PlotPoint | null = null
    let nearestDist = Infinity
    for (const p of pts) {
      const { x, y } = plotPosition(p, pts, canvas.width, canvas.height)
      const dist = (x - mx) ** 2 + (y - my) ** 2
      if (dist < nearestDist) {
        nearestDist = dist
        nearest = p
      }
    }
    if (nearest && nearest.word && nearestDist < 100) {
      tooltip.textContent = nearest.word
      tooltip.style.left = `${event.clientX + 12}px`
      tooltip.style.top = `${event.clientY + 12}px`
      tooltip.hidden = false
    } else {
      tooltip.hidden = true
    }
  })
  canvas.addEventListener('mouseleave', () => {
    tooltip.hidden = true
  })
}

interface RaceTargets {
  emeansCanvas: string
  kmeansCanvas: string
  emeansStats: string
  kmeansStats: string
}

function runRace(
  flat: Float32Array,
  dim: number,
  k: number,
  pts: PlotPoint[],
  targets: RaceTargets
): void {
  const emeansCanvas = document.querySelector<HTMLCanvasElement>(targets.emeansCanvas)!
  const kmeansCanvas = document.querySelector<HTMLCanvasElement>(targets.kmeansCanvas)!
  const emeansStats = document.querySelector<HTMLParagraphElement>(targets.emeansStats)!
  const kmeansStats = document.querySelector<HTMLParagraphElement>(targets.kmeansStats)!

  const emeansStart = performance.now()
  const emeansResult = fit(flat, dim, k, { seed: 1, nepochs: 5 })
  const emeansMs = performance.now() - emeansStart
  render(emeansCanvas, pts, emeansResult.labels)
  attachHover(emeansCanvas, pts)
  emeansStats.textContent = `${emeansResult.epochs} epochs - ${emeansMs.toFixed(0)}ms - loss ${emeansResult.loss.toFixed(1)}`

  const kmeansStart = performance.now()
  const kmeansResult = kmeans(flat, dim, k, { seed: 1, maxEpochs: 25 })
  const kmeansMs = performance.now() - kmeansStart
  render(kmeansCanvas, pts, kmeansResult.labels)
  attachHover(kmeansCanvas, pts)
  kmeansStats.textContent = `${kmeansResult.epochs} epochs - ${kmeansMs.toFixed(0)}ms - loss ${kmeansResult.loss.toFixed(1)}`
}

document.querySelectorAll<HTMLButtonElement>('nav button').forEach((button) => {
  button.addEventListener('click', () => {
    document.querySelectorAll('nav button').forEach((b) => b.classList.remove('active'))
    button.classList.add('active')
    const tab = button.dataset.tab
    document.querySelector<HTMLElement>('#shakespeare-tab')!.hidden = tab !== 'shakespeare'
    document.querySelector<HTMLElement>('#custom-tab')!.hidden = tab !== 'custom'
  })
})

document.querySelector('#run-shakespeare')!.addEventListener('click', () => {
  const plotPoints: PlotPoint[] = shakespearePoints.map((p) => ({
    word: p.word,
    x: p.x,
    y: p.y,
  }))
  runRace(shakespeareFlat, shakespeareDim, 8, plotPoints, {
    emeansCanvas: '#emeans-canvas',
    kmeansCanvas: '#kmeans-canvas',
    emeansStats: '#emeans-stats',
    kmeansStats: '#kmeans-stats',
  })
})

document.querySelector('#run-custom')!.addEventListener('click', () => {
  const raw = (document.querySelector('#custom-input') as HTMLTextAreaElement).value
  const k = Number((document.querySelector('#custom-k') as HTMLInputElement).value)
  const vectors: number[][] = JSON.parse(raw)
  const customDim = vectors[0].length
  const flat = new Float32Array(vectors.flat())
  const plotPoints: PlotPoint[] = vectors.map((v) => ({ word: '', x: v[0], y: v[1] }))
  runRace(flat, customDim, k, plotPoints, {
    emeansCanvas: '#custom-emeans-canvas',
    kmeansCanvas: '#custom-kmeans-canvas',
    emeansStats: '#custom-emeans-stats',
    kmeansStats: '#custom-kmeans-stats',
  })
})
```

- [ ] **Step 2: Run the dev server and manually verify**

```bash
cd demo && npm run dev
```

Open the printed localhost URL. Expected: clicking "Run the race" on the Shakespeare tab renders two canvases with color-coded points, converging stats below each, and hovering a point shows its word in a tooltip. Switching to "Bring your own data", pasting a JSON array of `[x, y]` pairs, and clicking "Run the race" does the same for that data.

- [ ] **Step 3: Commit**

```bash
cd ..
git add demo/src/main.ts
git commit -m "feat: demo UI for Shakespeare word-cloud race and bring-your-own-data tab"
```

---

## Task 14: Publish checklist

**Files:**
- Modify: `README.md` (fill in the live demo URL once Pages is live)

- [ ] **Step 1: Push to GitHub and enable Pages**

```bash
gh repo create e-means-ts --public --source=. --remote=origin --push
```

Then in the repo settings, under Pages, set the source to "GitHub Actions" (the `deploy-demo.yml` workflow from Task 11 handles the rest on the next push to `main`).

- [ ] **Step 2: Verify CI and the Pages deploy both go green**

Check the Actions tab after the push completes.

- [ ] **Step 3: Fill in the live demo link in `README.md`, commit, and push**

```bash
git add README.md
git commit -m "docs: add live demo link"
git push
```

- [ ] **Step 4: Publish to npm**

```bash
npm publish --access public
```

- [ ] **Step 5: Verify**

```bash
npm view e-means
```

Expected: the package is live with the version from `package.json`.
