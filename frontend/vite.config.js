import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Development: `npm run dev` (http://127.0.0.1:5173) forwards API and WebSocket calls to the gateway.
// Demo: `npm run build` -> dist/, served by the gateway itself at http://127.0.0.1:8200 (D-052).
export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8200",
      "/ws": { target: "ws://127.0.0.1:8200", ws: true },
    },
  },
  build: { outDir: "dist", sourcemap: false, chunkSizeWarningLimit: 1200 },
});
