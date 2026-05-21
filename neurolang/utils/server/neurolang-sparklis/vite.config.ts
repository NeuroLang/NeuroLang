import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3100,
    proxy: {
      '/v1': {
        target: 'http://localhost:8888',
        changeOrigin: true,
        ws: true,
      },
      '/v2': {
        target: 'http://localhost:8888',
        changeOrigin: true,
      },
    },
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test-setup.ts'],
  },
})
