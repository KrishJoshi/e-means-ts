import { fit, kmeans } from 'e-means'

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

interface Bounds {
  minX: number
  maxX: number
  minY: number
  maxY: number
}

interface Frame {
  labels: Int32Array
  loss: number
  epoch: number
  done: boolean
}

let shakespeareCache: { points: WordPoint[]; dim: number; flat: Float32Array } | null = null

async function loadShakespeareData(): Promise<{ points: WordPoint[]; dim: number; flat: Float32Array }> {
  if (shakespeareCache) return shakespeareCache
  const url = `${import.meta.env.BASE_URL}data/shakespeare-embeddings.json`
  const response = await fetch(url)
  const points = (await response.json()) as WordPoint[]
  const dim = points[0].vector.length
  const flat = new Float32Array(points.flatMap((p) => p.vector))
  shakespeareCache = { points, dim, flat }
  return shakespeareCache
}

const app = document.querySelector<HTMLDivElement>('#app')!
app.innerHTML = `
  <nav id="demo-nav">
    <button data-tab="shakespeare" class="active">Shakespeare word cloud</button>
    <button data-tab="custom">Bring your own data</button>
  </nav>
  <section id="shakespeare-tab">
    <button id="run-shakespeare">Run the race</button>
    <p id="shakespeare-loading" class="loading-note" hidden>Loading dataset&hellip;</p>
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
    <div class="epoch-controls" id="shakespeare-epoch-controls" hidden>
      <button id="shakespeare-play-btn" class="play-btn" type="button">&#9654; Play</button>
      <input type="range" id="shakespeare-epoch-slider" min="1" max="1" value="1" />
      <span id="shakespeare-epoch-label" class="epoch-label"></span>
    </div>
  </section>
  <section id="custom-tab" hidden>
    <textarea id="custom-input" rows="6" placeholder="Paste JSON: [[0.1, 0.2], [0.4, 0.9], ...]"></textarea>
    <label>k: <input id="custom-k" type="number" value="3" min="2" /></label>
    <button id="run-custom">Run the race</button>
    <p id="custom-loading" class="loading-note" hidden>Computing&hellip;</p>
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
    <div class="epoch-controls" id="custom-epoch-controls" hidden>
      <button id="custom-play-btn" class="play-btn" type="button">&#9654; Play</button>
      <input type="range" id="custom-epoch-slider" min="1" max="1" value="1" />
      <span id="custom-epoch-label" class="epoch-label"></span>
    </div>
  </section>
`

const COLORS = [
  '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
  '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990',
]

const EMEANS_EPOCHS = 5
const PLAY_INTERVAL_MS = 650
const VORONOI_GRID_STEP = 4

function computeBounds(pts: PlotPoint[]): Bounds {
  let minX = Infinity
  let maxX = -Infinity
  let minY = Infinity
  let maxY = -Infinity
  for (const p of pts) {
    if (p.x < minX) minX = p.x
    if (p.x > maxX) maxX = p.x
    if (p.y < minY) minY = p.y
    if (p.y > maxY) maxY = p.y
  }
  return { minX, maxX, minY, maxY }
}

function project(
  p: PlotPoint,
  bounds: Bounds,
  canvasWidth: number,
  canvasHeight: number
): { x: number; y: number } {
  return {
    x: ((p.x - bounds.minX) / (bounds.maxX - bounds.minX || 1)) * (canvasWidth - 40) + 20,
    y: ((p.y - bounds.minY) / (bounds.maxY - bounds.minY || 1)) * (canvasHeight - 40) + 20,
  }
}

// Mean position (in already-projected 2D canvas space) of each cluster's
// member points for the current frame -- a display-only stand-in for the
// real centroid, which lives in the original high-dimensional space.
function displayCentroids(
  positions: { x: number; y: number }[],
  labels: Int32Array
): Map<number, { x: number; y: number }> {
  const sums = new Map<number, { x: number; y: number; n: number }>()
  positions.forEach((pos, i) => {
    const c = labels[i]
    const s = sums.get(c)
    if (!s) {
      sums.set(c, { x: pos.x, y: pos.y, n: 1 })
    } else {
      s.x += pos.x
      s.y += pos.y
      s.n += 1
    }
  })
  const centroids = new Map<number, { x: number; y: number }>()
  for (const [cluster, s] of sums) centroids.set(cluster, { x: s.x / s.n, y: s.y / s.n })
  return centroids
}

// Draws Voronoi-style cell boundaries by rasterizing a coarse grid and
// marking cells where the nearest display-centroid changes between
// neighbors -- an approximation that's visually equivalent to an exact
// Voronoi diagram at canvas resolution, without a computational-geometry
// dependency.
function drawVoronoiBoundaries(
  ctx: CanvasRenderingContext2D,
  centroids: Map<number, { x: number; y: number }>,
  canvasWidth: number,
  canvasHeight: number
): void {
  const entries = Array.from(centroids.entries())
  if (entries.length < 2) return
  const cols = Math.ceil(canvasWidth / VORONOI_GRID_STEP) + 1
  const rows = Math.ceil(canvasHeight / VORONOI_GRID_STEP) + 1
  const owner = new Int32Array(cols * rows)

  function nearest(x: number, y: number): number {
    let bestDist = Infinity
    let bestCluster = entries[0][0]
    for (const [cluster, c] of entries) {
      const d = (c.x - x) ** 2 + (c.y - y) ** 2
      if (d < bestDist) {
        bestDist = d
        bestCluster = cluster
      }
    }
    return bestCluster
  }

  for (let gy = 0; gy < rows; gy++) {
    for (let gx = 0; gx < cols; gx++) {
      owner[gy * cols + gx] = nearest(gx * VORONOI_GRID_STEP, gy * VORONOI_GRID_STEP)
    }
  }

  ctx.fillStyle = 'rgba(20, 21, 26, 0.75)'
  const halfStep = VORONOI_GRID_STEP / 2
  for (let gy = 0; gy < rows; gy++) {
    for (let gx = 0; gx < cols; gx++) {
      const here = owner[gy * cols + gx]
      const right = gx + 1 < cols ? owner[gy * cols + gx + 1] : here
      const below = gy + 1 < rows ? owner[(gy + 1) * cols + gx] : here
      if (here !== right || here !== below) {
        ctx.fillRect(
          gx * VORONOI_GRID_STEP - halfStep / 2,
          gy * VORONOI_GRID_STEP - halfStep / 2,
          halfStep,
          halfStep
        )
      }
    }
  }
}

function render(
  canvas: HTMLCanvasElement,
  pts: PlotPoint[],
  positions: { x: number; y: number }[],
  labels: Int32Array
): void {
  const ctx = canvas.getContext('2d')!
  ctx.clearRect(0, 0, canvas.width, canvas.height)

  const centroids = displayCentroids(positions, labels)

  const dotRadius = pts.length > 5000 ? 1.4 : pts.length > 1000 ? 2.2 : 4
  ctx.globalAlpha = pts.length > 1000 ? 0.65 : 1
  positions.forEach(({ x, y }, i) => {
    ctx.fillStyle = COLORS[labels[i] % COLORS.length]
    ctx.beginPath()
    ctx.arc(x, y, dotRadius, 0, Math.PI * 2)
    ctx.fill()
  })
  ctx.globalAlpha = 1

  drawVoronoiBoundaries(ctx, centroids, canvas.width, canvas.height)

  for (const [cluster, c] of centroids) {
    ctx.fillStyle = COLORS[cluster % COLORS.length]
    ctx.strokeStyle = '#ffffff'
    ctx.lineWidth = 1.5
    ctx.beginPath()
    ctx.arc(c.x, c.y, 7, 0, Math.PI * 2)
    ctx.fill()
    ctx.stroke()
  }
}

function attachHover(
  canvas: HTMLCanvasElement,
  pts: PlotPoint[],
  positions: { x: number; y: number }[]
): void {
  const tooltip = document.querySelector<HTMLDivElement>('#tooltip')!
  canvas.addEventListener('mousemove', (event) => {
    const rect = canvas.getBoundingClientRect()
    const mx = event.clientX - rect.left
    const my = event.clientY - rect.top
    let nearestIdx = -1
    let nearestDist = Infinity
    positions.forEach((pos, i) => {
      const dist = (pos.x - mx) ** 2 + (pos.y - my) ** 2
      if (dist < nearestDist) {
        nearestDist = dist
        nearestIdx = i
      }
    })
    const nearest = nearestIdx >= 0 ? pts[nearestIdx] : null
    if (nearest && nearest.word && nearestDist < 40) {
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

function computeFrames(
  flat: Float32Array,
  dim: number,
  k: number
): {
  emeansFrames: Frame[]
  kmeansFrames: Frame[]
  maxFrame: number
  emeansMs: number
  kmeansMs: number
} {
  const kProbeStart = performance.now()
  const kFull = kmeans(flat, dim, k, { seed: 1, maxEpochs: 25 })
  const kConvergedAt = Math.max(1, kFull.epochs)
  const maxFrame = Math.max(EMEANS_EPOCHS, kConvergedAt)

  const emeansStart = performance.now()
  const emeansFrames: Frame[] = []
  for (let i = 1; i <= maxFrame; i++) {
    if (i <= EMEANS_EPOCHS) {
      const r = fit(flat, dim, k, { seed: 1, nepochs: i })
      emeansFrames.push({ labels: r.labels, loss: r.loss, epoch: r.epochs, done: i === EMEANS_EPOCHS })
    } else {
      emeansFrames.push({ ...emeansFrames[EMEANS_EPOCHS - 1], done: true })
    }
  }
  const emeansMs = performance.now() - emeansStart

  const kmeansStart = performance.now()
  const kmeansFrames: Frame[] = []
  for (let i = 1; i <= maxFrame; i++) {
    if (i <= kConvergedAt) {
      const r = i === kConvergedAt ? kFull : kmeans(flat, dim, k, { seed: 1, maxEpochs: i })
      kmeansFrames.push({ labels: r.labels, loss: r.loss, epoch: r.epochs, done: r.epochs >= kConvergedAt })
    } else {
      kmeansFrames.push({ ...kmeansFrames[kConvergedAt - 1], done: true })
    }
  }
  const kmeansMs = performance.now() - kmeansStart + (emeansStart - kProbeStart)

  return { emeansFrames, kmeansFrames, maxFrame, emeansMs, kmeansMs }
}

interface RaceConfig {
  runBtn: string
  loadingNote: string
  emeansCanvas: string
  kmeansCanvas: string
  emeansStats: string
  kmeansStats: string
  controls: string
  slider: string
  playBtn: string
  epochLabel: string
  getInput: () => Promise<{ flat: Float32Array; dim: number; k: number; pts: PlotPoint[] }>
}

function setupRace(cfg: RaceConfig): void {
  const runBtn = document.querySelector<HTMLButtonElement>(cfg.runBtn)!
  const loadingNote = document.querySelector<HTMLParagraphElement>(cfg.loadingNote)!
  const emeansCanvas = document.querySelector<HTMLCanvasElement>(cfg.emeansCanvas)!
  const kmeansCanvas = document.querySelector<HTMLCanvasElement>(cfg.kmeansCanvas)!
  const emeansStats = document.querySelector<HTMLParagraphElement>(cfg.emeansStats)!
  const kmeansStats = document.querySelector<HTMLParagraphElement>(cfg.kmeansStats)!
  const controls = document.querySelector<HTMLDivElement>(cfg.controls)!
  const slider = document.querySelector<HTMLInputElement>(cfg.slider)!
  const playBtn = document.querySelector<HTMLButtonElement>(cfg.playBtn)!
  const epochLabel = document.querySelector<HTMLSpanElement>(cfg.epochLabel)!

  let emeansFrames: Frame[] = []
  let kmeansFrames: Frame[] = []
  let positions: { x: number; y: number }[] = []
  let pts: PlotPoint[] = []
  let emeansMs = 0
  let kmeansMs = 0
  let playTimer: number | null = null

  function stopPlaying(): void {
    if (playTimer !== null) {
      window.clearInterval(playTimer)
      playTimer = null
    }
    playBtn.innerHTML = '&#9654; Play'
  }

  function showFrame(index: number): void {
    const e = emeansFrames[index]
    const k = kmeansFrames[index]
    if (!e || !k) return
    render(emeansCanvas, pts, positions, e.labels)
    render(kmeansCanvas, pts, positions, k.labels)
    const totalEmeansEpochs = emeansFrames[emeansFrames.length - 1].epoch
    const totalKmeansEpochs = kmeansFrames[kmeansFrames.length - 1].epoch
    emeansStats.textContent = e.done
      ? `e-means: done at epoch ${e.epoch} (${emeansMs.toFixed(0)}ms total) - loss ${e.loss.toFixed(1)}`
      : `e-means: epoch ${e.epoch} of ${totalEmeansEpochs} - training - loss ${e.loss.toFixed(1)}`
    kmeansStats.textContent = k.done
      ? `k-means: converged at epoch ${k.epoch} (${kmeansMs.toFixed(0)}ms total) - loss ${k.loss.toFixed(1)}`
      : `k-means: epoch ${k.epoch} of ${totalKmeansEpochs} - still refining - loss ${k.loss.toFixed(1)}`
    epochLabel.textContent = `Step ${index + 1} of ${emeansFrames.length}`
  }

  slider.addEventListener('input', () => {
    stopPlaying()
    showFrame(Number(slider.value) - 1)
  })

  playBtn.addEventListener('click', () => {
    if (playTimer !== null) {
      stopPlaying()
      return
    }
    playBtn.innerHTML = '&#10073;&#10073; Pause'
    playTimer = window.setInterval(() => {
      const next = Number(slider.value) + 1
      if (next > Number(slider.max)) {
        stopPlaying()
        return
      }
      slider.value = String(next)
      showFrame(next - 1)
    }, PLAY_INTERVAL_MS)
  })

  runBtn.addEventListener('click', async () => {
    stopPlaying()
    controls.hidden = true
    loadingNote.hidden = false
    runBtn.disabled = true

    const input = await cfg.getInput()

    loadingNote.textContent = `Computing ${input.pts.length.toLocaleString()} points…`
    // Let the "computing" note paint before the (possibly multi-hundred-ms) synchronous work.
    await new Promise((resolve) => window.setTimeout(resolve, 20))

    pts = input.pts
    const bounds = computeBounds(pts)
    positions = pts.map((p) => project(p, bounds, emeansCanvas.width, emeansCanvas.height))
    attachHover(emeansCanvas, pts, positions)
    attachHover(kmeansCanvas, pts, positions)

    const frames = computeFrames(input.flat, input.dim, input.k)
    emeansFrames = frames.emeansFrames
    kmeansFrames = frames.kmeansFrames
    emeansMs = frames.emeansMs
    kmeansMs = frames.kmeansMs

    slider.min = '1'
    slider.max = String(frames.maxFrame)
    slider.value = '1'
    loadingNote.hidden = true
    controls.hidden = false
    runBtn.disabled = false
    showFrame(0)
  })
}

document.querySelectorAll<HTMLButtonElement>('#demo-nav button').forEach((button) => {
  button.addEventListener('click', () => {
    document.querySelectorAll('#demo-nav button').forEach((b) => b.classList.remove('active'))
    button.classList.add('active')
    const tab = button.dataset.tab
    document.querySelector<HTMLElement>('#shakespeare-tab')!.hidden = tab !== 'shakespeare'
    document.querySelector<HTMLElement>('#custom-tab')!.hidden = tab !== 'custom'
  })
})

setupRace({
  runBtn: '#run-shakespeare',
  loadingNote: '#shakespeare-loading',
  emeansCanvas: '#emeans-canvas',
  kmeansCanvas: '#kmeans-canvas',
  emeansStats: '#emeans-stats',
  kmeansStats: '#kmeans-stats',
  controls: '#shakespeare-epoch-controls',
  slider: '#shakespeare-epoch-slider',
  playBtn: '#shakespeare-play-btn',
  epochLabel: '#shakespeare-epoch-label',
  getInput: async () => {
    const { points, dim, flat } = await loadShakespeareData()
    return {
      flat,
      dim,
      k: 14,
      pts: points.map((p) => ({ word: p.word, x: p.x, y: p.y })),
    }
  },
})

setupRace({
  runBtn: '#run-custom',
  loadingNote: '#custom-loading',
  emeansCanvas: '#custom-emeans-canvas',
  kmeansCanvas: '#custom-kmeans-canvas',
  emeansStats: '#custom-emeans-stats',
  kmeansStats: '#custom-kmeans-stats',
  controls: '#custom-epoch-controls',
  slider: '#custom-epoch-slider',
  playBtn: '#custom-play-btn',
  epochLabel: '#custom-epoch-label',
  getInput: async () => {
    const raw = (document.querySelector('#custom-input') as HTMLTextAreaElement).value
    const k = Number((document.querySelector('#custom-k') as HTMLInputElement).value)
    const vectors: number[][] = JSON.parse(raw)
    return {
      flat: new Float32Array(vectors.flat()),
      dim: vectors[0].length,
      k,
      pts: vectors.map((v) => ({ word: '', x: v[0], y: v[1] })),
    }
  },
})

document.querySelectorAll<HTMLButtonElement>('.copy-btn').forEach((button) => {
  button.addEventListener('click', () => {
    const text = button.dataset.copy ?? button.parentElement?.querySelector('code')?.textContent ?? ''
    navigator.clipboard.writeText(text).then(() => {
      const original = button.textContent
      button.textContent = 'Copied!'
      setTimeout(() => {
        button.textContent = original
      }, 1500)
    })
  })
})

document.querySelectorAll<HTMLButtonElement>('.code-tab-btn').forEach((button) => {
  button.addEventListener('click', () => {
    const target = button.dataset.codeTab
    document.querySelectorAll('.code-tab-btn').forEach((b) => b.classList.remove('active'))
    document.querySelectorAll('.code-tab-panel').forEach((p) => p.classList.remove('active'))
    button.classList.add('active')
    document.querySelector(`[data-code-panel="${target}"]`)?.classList.add('active')
  })
})
