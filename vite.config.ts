import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  // Relative asset URLs, so the build works from any path, such as a GitHub
  // Pages project site at https://<user>.github.io/<repo>/.
  base: './',
  plugins: [react()],
})
