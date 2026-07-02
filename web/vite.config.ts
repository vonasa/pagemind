import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      "/books": "http://localhost:8001",
      "/chapters": "http://localhost:8001",
      "/sections": "http://localhost:8001",
      "/health": "http://localhost:8001",
    },
  },
});
