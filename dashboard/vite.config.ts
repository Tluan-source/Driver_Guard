import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev: `npm run dev` on :5173, API calls proxied to the FastAPI backend on :8000.
// Prod: `npm run build` -> dist/, served by FastAPI at "/" (same origin, no proxy needed).
const API = "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/auth": API,
      "/me": API,
      "/state": API,
      "/health": API,
      "/events": API,
      "/trips": API,
      "/config": API,
      "/ws": { target: API.replace("http", "ws"), ws: true },
    },
  },
});
