// Drives the built page with simulated landmarks and screenshots the avatar from the front and the side.
//   npm run build && npx vite preview --port 4173 &   then   node avatar/shoot.mjs [outdir] [url]
import { createRequire } from "node:module";
import { mkdirSync } from "node:fs";

const require = createRequire(import.meta.url);
let chromium;
try {
  ({ chromium } = require("playwright"));
} catch {
  ({ chromium } = require("/opt/node-tools/node_modules/playwright"));
}
const out = process.argv[2] ?? "avatar/shots";
const url = process.argv[3] ?? "http://127.0.0.1:4173/";
mkdirSync(out, { recursive: true });

// MediaPipe-style normalized landmarks (0..1); only nose, ears and shoulders are read.
function landmarks({ earY = 0.45, tiltEars = 0 } = {}) {
  const lm = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, z: 0, visibility: 1 }));
  lm[0] = { x: 0.5, y: earY + 0.02, z: -0.3 };
  lm[7] = { x: 0.54, y: earY - tiltEars, z: 0 };
  lm[8] = { x: 0.46, y: earY + tiltEars, z: 0 };
  lm[11] = { x: 0.62, y: 0.75, z: 0 };
  lm[12] = { x: 0.38, y: 0.75, z: 0 };
  return lm;
}

const browser = await chromium.launch({ args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"] });
const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
const errors = [];
page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
page.on("pageerror", (e) => errors.push(e.message));
await page.goto(url);
await page.waitForFunction(() => !!window.__posture && document.querySelector("#status").textContent.indexOf("失敗") < 0);
await page.waitForTimeout(3000);

let t = 1000;
const feed = (opts, n = 20) =>
  page.evaluate(([lm, n, t0]) => { for (let i = 0; i < n; i++) window.__posture.update(lm, t0 + i * 100); }, [landmarks(opts), n, t]).then(() => { t += n * 100; });

const canvas = page.locator("#three");
const box = await canvas.boundingBox();
async function orbit(dx) {
  const cx = box.x + box.width / 2, cy = box.y + box.height / 2;
  await page.mouse.move(cx, cy);
  await page.mouse.down();
  await page.mouse.move(cx + dx, cy, { steps: 10 });
  await page.mouse.up();
  await page.waitForTimeout(400);
}
const quarter = box.height / 4; // OrbitControls: 2π per canvas height
async function shoot(name) {
  await page.waitForTimeout(300);
  await canvas.screenshot({ path: `${out}/${name}-front.png` });
  await orbit(-quarter);
  await canvas.screenshot({ path: `${out}/${name}-side.png` });
  await orbit(quarter);
}

await feed({});
await page.evaluate(() => window.__posture.calibrate());
await feed({});
await shoot("upright");
await feed({ earY: 0.56 }, 30);
await shoot("slouch");
await feed({}, 30);
await page.evaluate(() => window.__posture.startStretch());
await feed({}, 3);
await shoot("headtilt");
// let steps 1 and 2 time out (15 s each of simulated time) to reach the shrug
await feed({}, 160);
await feed({}, 160);
await shoot("shrug");
const state = await page.evaluate(() => window.__posture.state());
console.log("stretch state", JSON.stringify(state));
console.log(errors.length ? "console errors:\n" + errors.join("\n") : "no console errors");
await browser.close();
process.exit(errors.length ? 1 : 0);
