# WEBSITE.md — Frontend Website Rules

## Always Do First
- **Invoke the `frontend-design` skill** before writing any frontend code, every session, no exceptions.

## Project Context
- This repo is a 2026 FIFA World Cup match-outcome prediction model (see `CLAUDE.md`). Any website
  built here is for presenting its predictions and data.
- Prediction data lives in `project-root/data/predictions/` — `03_all_upcoming_match_predictions.csv`
  (all matches × 2 models: Random Forest + XGBoost, with H/D/A probabilities) plus one CSV per match.
  Upcoming-fixture feature vectors are in `project-root/data/feature_vectors/`.
- The local server (below) serves the repo root, so pages can `fetch()` those CSVs directly, e.g.
  `fetch('/project-root/data/predictions/03_all_upcoming_match_predictions.csv')`.
- Put website files in `website/` at the repo root (create it on first use) with a single
  `index.html` entry point, reachable at `http://localhost:3000/website/`.

## Reference Images
- If a reference image is provided: match layout, spacing, typography, and color exactly. Swap in placeholder content (images via `https://placehold.co/`, generic copy). Do not improve or add to the design.
- If no reference image: design from scratch with high craft (see guardrails below).
- Screenshot your output, compare against reference, fix mismatches, re-screenshot. Do at least 2 comparison rounds. Stop only when no visible differences remain or user says so.

## Local Server
- **Always serve on localhost** — never screenshot a `file:///` URL.
- Start the dev server: `node serve.mjs` (serves the repo root at `http://localhost:3000`; zero dependencies)
- `serve.mjs` lives in the repo root (`/Users/Cael/Soccer_Prediction_Model/`). Start it in the background before taking any screenshots.
- If the server is already running, do not start a second instance.

## Screenshot Workflow
- **First-time setup:** Puppeteer is NOT installed yet on this machine. Run `npm install puppeteer`
  in the repo root once (downloads Chrome to `~/.cache/puppeteer/`). Node is at `/usr/local/bin/node` (v19).
- **Always screenshot from localhost:** `node screenshot.mjs http://localhost:3000/website/`
- Screenshots are saved automatically to `./temporary screenshots/screenshot-N.png` (auto-incremented, never overwritten).
- Optional label suffix: `node screenshot.mjs <url> label` → saves as `screenshot-N-label.png`
- `screenshot.mjs` lives in the repo root. Use it as-is.
- After screenshotting, read the PNG from `temporary screenshots/` with the Read tool — Claude can see and analyze the image directly.
- When comparing, be specific: "heading is 32px but reference shows ~24px", "card gap is 16px but should be 24px"
- Check: spacing/padding, font size/weight/line-height, colors (exact hex), alignment, border-radius, shadows, image sizing
- `node_modules/`, `package*.json` (from the puppeteer install), and `temporary screenshots/` are
  build/scratch artifacts — do not commit them.

## Output Defaults
- Single `index.html` file, all styles inline, unless user says otherwise
- Tailwind CSS via CDN: `<script src="https://cdn.tailwindcss.com"></script>`
- Placeholder images: `https://placehold.co/WIDTHxHEIGHT`
- Mobile-first responsive

## Brand Assets
- Always check the `brand_assets/` folder at the repo root before designing. It may contain logos, color guides, style guides, or images. (It does not exist yet — if it appears, use it.)
- If assets exist there, use them. Do not use placeholders where real assets are available.
- If a logo is present, use it. If a color palette is defined, use those exact values — do not invent brand colors.

## Anti-Generic Guardrails
- **Colors:** Never use default Tailwind palette (indigo-500, blue-600, etc.). Pick a custom brand color and derive from it.
- **Shadows:** Never use flat `shadow-md`. Use layered, color-tinted shadows with low opacity.
- **Typography:** Never use the same font for headings and body. Pair a display/serif with a clean sans. Apply tight tracking (`-0.03em`) on large headings, generous line-height (`1.7`) on body.
- **Gradients:** Layer multiple radial gradients. Add grain/texture via SVG noise filter for depth.
- **Animations:** Only animate `transform` and `opacity`. Never `transition-all`. Use spring-style easing.
- **Interactive states:** Every clickable element needs hover, focus-visible, and active states. No exceptions.
- **Images:** Add a gradient overlay (`bg-gradient-to-t from-black/60`) and a color treatment layer with `mix-blend-multiply`.
- **Spacing:** Use intentional, consistent spacing tokens — not random Tailwind steps.
- **Depth:** Surfaces should have a layering system (base → elevated → floating), not all sit at the same z-plane.

## Hard Rules
- Do not add sections, features, or content not in the reference
- Do not "improve" a reference design — match it
- Do not stop after one screenshot pass
- Do not use `transition-all`
- Do not use default Tailwind blue/indigo as primary color
