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
