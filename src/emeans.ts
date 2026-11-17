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
import { assertValidClusteringInput, assertValidPredictInput } from './validate'

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
  const n = assertValidClusteringInput(data, dim, k)
  const seed = opts.seed ?? 1234
  const nepochs = Math.max(1, Math.floor(opts.nepochs ?? 5))
  const init = opts.init ?? 'random'

  const batchLaw = resolveBatchLaw(opts.batchLaw ?? 'auto', dim)
  const unusedThreshold = resolveUnusedThreshold(opts.unusedThreshold, dim)
  const { batchSize, alpha0 } = resolveRecipe(n, k, nepochs, batchLaw, opts.batchSize, opts.alpha0)
  if (batchSize <= 0) {
    throw new Error(
      `batchSize must resolve to a positive integer, got ${batchSize}; check the batchSize option`
    )
  }

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
  assertValidPredictInput(data, dim, centroids)
  const n = data.length / dim
  const k = centroids.length / dim
  const allIndices = new Int32Array(n)
  for (let i = 0; i < n; i++) allIndices[i] = i
  return assignBatch(data, dim, allIndices, centroids, k).labels
}
