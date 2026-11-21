import fs from "node:fs";
import path from "node:path";
import { defineConfig } from "vite";

const PARITY_DIR = path.resolve("self_twin/evaluation/webllm_parity");

// Dev server only: the WebLLM parity page (self_twin/webllm/parity/) posts its
// run here, and self_twin/evaluation/webllm_parity.py compares it with
// Transformers.
function parityResults() {
  return {
    name: "webllm-parity-results",
    apply: "serve",
    configureServer(server) {
      server.middlewares.use("/__parity/results", (req, res) => {
        const name = new URL(req.url, "http://localhost").searchParams.get("name") || "";
        if (req.method !== "POST" || !/^[\w.-]+$/.test(name)) {
          res.statusCode = 400;
          res.end("POST a JSON body with ?name=<run name>");
          return;
        }
        let body = "";
        req.setEncoding("utf8");
        req.on("data", (chunk) => { body += chunk; });
        req.on("end", () => {
          try {
            const parsed = JSON.parse(body);
            fs.mkdirSync(PARITY_DIR, { recursive: true });
            const file = path.join(PARITY_DIR, `${name}.json`);
            fs.writeFileSync(file, `${JSON.stringify(parsed, null, 1)}\n`);
            res.end(path.relative(process.cwd(), file));
          } catch (error) {
            res.statusCode = 400;
            res.end(String(error));
          }
        });
      });
    },
  };
}

export default defineConfig({
  plugins: [parityResults()],
  build: {
    manifest: "build-manifest.json",
  },
});
