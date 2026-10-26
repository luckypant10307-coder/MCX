import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development the FastAPI server runs on :8000 and Vite proxies /api to it, so the
// app always calls same-origin "/api/..." -- exactly as it does when FastAPI serves
// the built files in production.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
  build: {
    outDir: "dist",
    sourcemap: false,
    chunkSizeWarningLimit: 800,
  },
});
