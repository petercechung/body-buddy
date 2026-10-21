import { resolve } from "node:path";
import { defineConfig } from "vite";

// Two pages: the squat mirror (index) and the desk posture buddy.
export default defineConfig({
  build: {
    rollupOptions: {
      input: {
        main: resolve(import.meta.dirname, "index.html"),
        posture: resolve(import.meta.dirname, "posture.html"),
      },
    },
  },
});
