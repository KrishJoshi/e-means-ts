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
