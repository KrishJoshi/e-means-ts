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
worker syntax (Vite, webpack 5+) -- it is not usable from a plain CJS
`require()`/Node entry point, since CJS has no `import.meta.url` to resolve
the worker file from. Call `fit()`/`kmeans()` directly for CJS/Node use.

**Note:** `runInWorker` transfers `data`'s underlying `ArrayBuffer` to the
worker; the array becomes unusable in the caller afterward (and if `data` is
a `.subarray()` view, the *entire* backing buffer is detached, not just the
view). Pass a copy (`data.slice()`) if you need the original array again,
e.g. for a later `predict()` call.

`ClusterResult`: `{ centroids: Float32Array, labels: Int32Array, loss:
number, epochs: number }`. Note that `labels` reflects each point's
assignment from the last epoch's streaming mini-batch pass against the
centroids *as they were during that epoch* (computed before the final
`finalize()` averaging step), so `predict(data, dim, result.centroids)` can
disagree with `result.labels` on a handful of points near cluster
boundaries. This is faithful to the real faiss e-means algorithm's behavior,
not a bug in this port.

## Benchmarks

### Real embeddings (the actual use case)

`e-means` (5 epochs) vs. this package's own `kmeans` (Lloyd's, run to
convergence) on the 18,396-word / 50-dimension Shakespeare embeddings
(the nearly-complete vocabulary of Shakespeare's complete works) used in
the live demo (linked below), both with `init: 'kmeans++'`, Node.js:

| k | e-means | kmeans (ours) | loss ratio | speedup |
|---|---------|----------------|------------|---------|
| 10 | 5 epochs, 51.9ms, loss=284,270.9 | 25 epochs, 191.1ms, loss=284,992.3 | 0.997 | 3.7x |
| 14 | 5 epochs, 62.8ms, loss=276,192.4 | 25 epochs, 253.5ms, loss=275,889.7 | 1.001 | 4.0x |
| 20 | 5 epochs, 85.5ms, loss=266,271.1 | 25 epochs, 393.2ms, loss=266,036.8 | 1.001 | 4.6x |

k-means uses its full 25-epoch budget here and still doesn't beat
e-means's loss — e-means matches it within 0.3% using a fifth of the
epochs and 3.7-4.6x less wall-clock time.

### Synthetic blobs (a harder case for e-means, included for transparency)

On well-separated synthetic blobs, Lloyd's k-means converges in as few as
2 epochs — there's no ambiguity for it to resolve, so there's little room
for a smarter update rule to win on *epochs*, though e-means still reaches
essentially the same loss:

| n | dim | k | e-means | kmeans (ours) | ml-kmeans |
|---|-----|---|---------|----------------|-----------|
| 10,000 | 32 | 10 | 26.5ms, 5 epochs, loss=26,825.0 | 10.1ms, 2 epochs, loss=26,760.8 | 78.2ms, 2 iterations, converged |
| 100,000 | 32 | 20 | 348.7ms, 5 epochs, loss=267,986.1 | 147.3ms, 2 epochs, loss=267,839.1 | 1,252.6ms, 2 iterations, converged |
| 1,000,000 | 32 | 50 | 7,796.8ms, 5 epochs, loss=2,678,994.1 | 3,654.6ms, 2 epochs, loss=2,678,557.0 | 49,033.0ms, 2 iterations, converged |

Both tables run `fit()` and `kmeans()` with the same `init: 'kmeans++'` for
a fair apples-to-apples comparison — an earlier version of the synthetic
table used `fit()`'s random-row-init default, which on well-separated
blobs can collide two centroids into the same blob and makes e-means look
far worse than it is. Neither table claims to match faiss's
CUTLASS-accelerated wall-clock speed — see the note at the top of this
README; the comparison here is algorithmic (epochs/loss) plus this
package's own honest wall-clock numbers.

[Live demo: Shakespeare word-cloud race](https://krishj.com/e-means-ts/)

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

MIT. The e-means scaling-law constants in `src/emeansSchedule.ts` are
derived from Meta Platforms, Inc.'s MIT-licensed
[faiss](https://github.com/facebookresearch/faiss) project -- see
[`NOTICE`](./NOTICE) for the full attribution.
