# e-means-ts — Design Spec

Date: 2026-11-17
Status: Draft, pending review

## 1. What this is

A zero-dependency, pure TypeScript implementation of **e-means**, a clustering
algorithm published by Meta FAIR in November 2026
([arXiv:2611.09501](https://arxiv.org/abs/2611.09501), shipped as
`faiss/contrib/e_means`). e-means is a drop-in replacement for k-means that
reaches a given loss in far fewer epochs than Lloyd's algorithm, using a
mini-batch update rule based on exponential moving averages. The reference
implementation is Python, and its wall-clock speed relies on PyTorch +
CUTLASS GPU kernels.

This project is an independent reimplementation of the algorithm's update
rule in TypeScript — **not a binding or port of the CUTLASS kernels**, and
not claiming to reproduce faiss's raw wall-clock performance. The claim is
narrower and more honest: the same algorithmic win (fewer epochs, lower
loss) is reproducible in plain TS, and that's useful in places the Python
version can't reach at all.

## 2. Why this exists / who it's for

No TypeScript implementation of e-means exists. More importantly, no
*practical* fast k-means exists for TS/JS environments that can't shell out
to Python or a native binary. Concrete use cases:

- **Edge/serverless runtimes** (Cloudflare Workers, Vercel Edge Functions) —
  no Python, no native addons, period. Today there's no good clustering
  option there at all.
- **Browser / client-side clustering** — local-first apps, browser
  extensions, note-taking tools doing semantic grouping without sending data
  to a server.
- **Training IVF-style index partitions for JS-native vector stores**
  (`sqlite-vec`, `vectra`, `hnswlib-node`) — these have no good in-ecosystem
  k-means for computing centroids today.
- **Embedding dedup in a Node pipeline** without standing up a Python
  microservice just for clustering.
- **Image/color quantization, log/event clustering for dashboards.**

The pitch is **"usable where the Python ecosystem can't go,"** not "faster
than faiss." This replaces an earlier direction (wiring e-means into a
RAPTOR-style RAG tree-builder) that was dropped after review: RAPTOR's
actual bottleneck is LLM summarization calls, not clustering, so pairing the
two invited an immediate "why does this need faster clustering?" objection.
RAPTOR/RAG integration may become a separate follow-on project later, built
on top of this package, only if there's real demand.

## 3. Scope (v1)

One package. No monorepo, no adapters, no optional accelerated backend.

### Package: `e-means`

- Zero runtime dependencies, pure TypeScript, ESM + CJS builds.
- Runs in Node, browsers, Workers, and Deno without modification.

**API:**

```ts
function fit(
  data: Float32Array,   // flat, row-major: n points * dim
  dim: number,
  k: number,
  opts?: {
    seed?: number
    maxEpochs?: number
    batchSize?: number   // e-means hyperparameter, sensible default
    emaDecay?: number    // e-means hyperparameter, sensible default
    init?: 'random' | 'kmeans++'
  }
): { centroids: Float32Array; labels: Int32Array; loss: number; epochs: number }

function predict(
  data: Float32Array,
  dim: number,
  centroids: Float32Array
): Int32Array

function kmeans(
  data: Float32Array,
  dim: number,
  k: number,
  opts?: { seed?: number; maxEpochs?: number; init?: 'random' | 'kmeans++' }
): { centroids: Float32Array; labels: Int32Array; loss: number; epochs: number }
```

- `fit` is e-means. `kmeans` is a plain Lloyd's implementation with
  k-means++ initialization, shipped in the same package as the baseline —
  needed for the benchmarks/demo, and people will want a plain
  implementation available regardless.
- Flat `Float32Array` + `dim` instead of `number[][]`: avoids per-row
  allocation/boxing overhead and matches how embeddings are already stored
  by most embedding APIs and typed-array-based consumers.
- `seed` makes both algorithms reproducible — required for the correctness
  fixtures (below) and for believable benchmarks.
- A `runInWorker()` helper wraps `fit`/`kmeans` in a Web Worker so browser
  callers don't block the main thread. Thin wrapper, not a separate package.

### Correctness validation

The credibility risk for this project is "is this actually e-means, or a
JS-flavored guess at the paper." To answer that directly: a fixed set of
small synthetic datasets are clustered with both this implementation and
the real faiss/Python `e_means` reference, and the resulting loss curves
(loss per epoch) are checked to land within a defined tolerance of each
other. These fixtures (input data + expected loss-curve ranges) are
generated once using the Python reference and checked into the test suite
as static fixtures — the test suite itself does not require Python or faiss
to run.

### Benchmarks (README)

Epochs-to-target-loss and wall-clock time, comparing `fit` (e-means) against
this package's own `kmeans` and against `ml-kmeans` (an existing JS k-means
package), at 10k / 100k / 1M points, run on Node and in Chrome. The README
explicitly states the comparison is algorithmic (epochs/loss), not a claim
of matching faiss's CUTLASS-accelerated wall-clock speed.

### Explicitly deferred (not in v1)

- GPU acceleration. If pursued later, a hand-written WebGPU compute shader
  for the distance/assignment step — not TensorFlow.js (development has
  stalled, `tfjs-node-gpu` has real install friction on current Node
  versions, and its WebGPU backend never matured; tensor-dispatch overhead
  would likely erase any gain at these data sizes anyway).
- `partialFit()` / streaming mini-batch support for out-of-core data.
- RAPTOR or any RAG-framework integration (LangChain.js, LlamaIndex.TS
  adapters, etc.) — a possible separate future project, not this one.

## 4. Demo (GitHub Pages)

Static site, no backend, no runtime model downloads — the earlier design
(full in-browser RAPTOR demo with WebLLM + transformers.js) was cut because
multi-GB model downloads and multi-minute run times would lose most
visitors before they saw anything. This demo loads and is interactive in
about a second.

**Tab 1 — Shakespeare word cloud.** Word embeddings for a curated vocabulary
from Shakespeare's complete works (content words, stopwords excluded,
roughly the 500–1000 most frequent, to keep the visualization legible) are
computed **offline at build time** using an existing pretrained word-vector
set, and shipped as a small static JSON asset containing, per word: the
word string, its embedding vector, and a precomputed 2D PCA projection used
only for plotting. In the browser, e-means and plain k-means cluster the
*actual embedding vectors* (not the 2D projection) side by side, live,
visualized on the precomputed 2D layout — color-coded by cluster, animated
as it converges, word labels on hover. Words that belong together
(character names, emotion words, etc.) visibly snap into the same color
group. A loss-curve + epoch/wall-clock readout runs alongside, same as the
README benchmark chart but as a live, pretty animation instead of a static
plot.

**Tab 2 — Bring your own data.** Paste or upload a JSON/CSV file of vectors;
run the same e-means vs. k-means side-by-side race on it, with the same
live loss-curve and timing display. No dimensionality assumptions beyond
what the pasted data provides; if it's higher-dimensional than 2D, project
with the same offline-computable PCA approach client-side for the plot only
(clustering still runs on the full vectors).

Both tabs run clustering inside the Web Worker helper so the page stays
responsive during the race.

## 5. Repository layout

```
e-means-ts/
  src/              # the e-means package source
  test/             # unit tests + faiss-correctness fixtures
  bench/            # benchmark scripts (README numbers generated from here)
  demo/             # static site source (Vite), deployed to GitHub Pages
  docs/
    superpowers/
      specs/        # design specs (this file)
README.md
LICENSE             # MIT
```

Single package at the repo root (not a monorepo) — v1 has exactly one
publishable package, so the workspace/changesets overhead of a monorepo
isn't justified yet. If a second package (e.g. a future RAPTOR integration)
materializes, it moves to its own repo rather than growing this one into a
monorepo.

## 6. v1 "done" criteria

- `e-means` package published to npm: `fit`, `predict`, `kmeans`,
  `runInWorker`, typed, documented, tested (including the faiss-correctness
  fixtures).
- README with honest epochs/wall-clock benchmarks vs. own `kmeans` and
  `ml-kmeans`, and a clear statement of what is and isn't being claimed
  relative to the Python/CUTLASS reference.
- Demo deployed and working on GitHub Pages: Shakespeare word-cloud tab
  fully working; bring-your-own-data tab fully working.
- MIT license.

## 7. Open parameters (not blocking, to settle during implementation)

- Exact pretrained word-vector set used for the Shakespeare demo (e.g. a
  small GloVe variant) — pick whichever keeps the offline build step and
  shipped asset size small.
- Exact e-means hyperparameter defaults (EMA decay, batch size) — taken
  from the paper's stated defaults, tuned only if the correctness fixtures
  don't land in range.
