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
