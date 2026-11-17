// One-time script: builds src/data/shakespeare-embeddings.json for the demo
// from two inputs that aren't committed to the repo (see .gitignore). Fetch
// them before running `npm run build:data`:
//
//   mkdir -p demo/scripts/.cache
//   curl -o demo/scripts/.cache/shakespeare.txt https://www.gutenberg.org/files/100/100-0.txt
//   curl -o demo/scripts/.cache/glove.zip https://nlp.stanford.edu/data/glove.6B.zip
//   cd demo/scripts/.cache && unzip -o glove.zip glove.6B.50d.txt && cd ../../..
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
