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
