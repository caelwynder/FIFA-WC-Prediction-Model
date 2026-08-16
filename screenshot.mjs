// Screenshot a page with Puppeteer.
// Usage: node screenshot.mjs http://localhost:3000 [label]
// Saves to ./temporary screenshots/screenshot-N[-label].png (auto-incremented).
// First-time setup: npm install puppeteer   (run in the repo root; downloads Chrome)
import { mkdirSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const url = process.argv[2];
const label = process.argv[3];

if (!url) {
  console.error("Usage: node screenshot.mjs <url> [label]");
  process.exit(1);
}

let puppeteer;
try {
  puppeteer = (await import("puppeteer")).default;
} catch {
  console.error("Puppeteer is not installed. Run: npm install puppeteer");
  process.exit(1);
}

const OUT_DIR = join(fileURLToPath(new URL(".", import.meta.url)), "temporary screenshots");
mkdirSync(OUT_DIR, { recursive: true });

const taken = readdirSync(OUT_DIR)
  .map((f) => f.match(/^screenshot-(\d+)/)?.[1])
  .filter(Boolean)
  .map(Number);
const n = taken.length ? Math.max(...taken) + 1 : 1;
const outPath = join(OUT_DIR, `screenshot-${n}${label ? `-${label}` : ""}.png`);

const browser = await puppeteer.launch();
const page = await browser.newPage();
await page.setViewport({ width: 1440, height: 900, deviceScaleFactor: 2 });
await page.goto(url, { waitUntil: "networkidle0", timeout: 30000 });
await page.screenshot({ path: outPath, fullPage: true });
await browser.close();

console.log(`Saved: ${outPath}`);
