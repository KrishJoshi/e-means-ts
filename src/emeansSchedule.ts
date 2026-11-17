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
