# Sentiment Flow Lab (Next.js + NPM)

Sentiment Flow Lab is a web-first sentiment analysis app built with Next.js (App Router).
It is designed for local npm development and direct Vercel deployment.

## What You Get

1. Modern responsive UI with custom typography and animated layout
2. Three interaction flows:
	- Single Analyze
	- Compare Two Inputs
	- Batch Scan (one line per input)
3. Multi-engine sentiment evaluation through a Vercel-compatible API route:
	- Lexicon Classic
	- Sentiment.js
	- VADER
	- Context Lens (heuristic)
	- Optional RoBERTa (Hugging Face API)
4. Threshold slider and live mode for rapid exploratory testing

## Tech Stack

1. Next.js 14
2. React 18
3. TypeScript
4. Node runtime API routes for server-side inference logic

## Run Locally

1. Install npm dependencies

```bash
npm install
```

2. (Optional) Add Hugging Face token for transformer model

```bash
cp .env.example .env.local
# set HF_TOKEN=your_token_here
```

3. Start dev server

```bash
npm run dev
```

Open `http://localhost:3000`.

## Deploy on Vercel

1. Push this repository to GitHub
2. Import the repo in Vercel
3. Add environment variable `HF_TOKEN` (optional)
4. Deploy

The API route at `app/api/analyze/route.ts` is serverless-ready for Vercel Node runtime.

## Project Structure

```
Sentiment_Analysis_CP/
|- app/
|  |- api/
|  |  |- analyze/
|  |     |- route.ts
|  |- globals.css
|  |- layout.tsx
|  |- page.tsx
|- .env.example
|- .gitignore
|- next.config.mjs
|- package.json
|- tsconfig.json
```
