/// <reference types="vitest" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: { "/api": { target: process.env.API_URL ?? "http://127.0.0.1:8000", changeOrigin: true } },
  },
  build: {
    chunkSizeWarningLimit: 4000,
    rollupOptions: {
      output: { manualChunks: { monaco: ["monaco-editor"], mermaid: ["mermaid"] } },
    },
  },
  test: { environment: "node", include: ["src/**/*.test.ts"] },
});
