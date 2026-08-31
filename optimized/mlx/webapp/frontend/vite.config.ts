import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'node:path'

// Two apps share one frontend project and one backend:
//   index.html -> the studio        (run_studio.sh)
//   live.html  -> the live loop app (run_live.sh)
// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        live: resolve(__dirname, 'live.html'),
      },
    },
  },
})
