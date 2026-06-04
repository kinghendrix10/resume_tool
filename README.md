# Resume intelligence (Next.js)

## Quick start

```bash
cp .env.example .env
# set GEMINI_API_KEY
npm install
npm run dev
```

## Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Development server |
| `npm run build` | Production build |
| `npm test` | Vitest unit tests |
| `npm run test:e2e` | Playwright (needs `npx playwright install chromium` once) |

API: `POST /api/parse` with `multipart/form-data` field `files` (repeatable).

## Source layout

| Path | Role |
|------|------|
| `src/app/` | Next.js App Router pages and `api/` routes |
| `src/components/resume/` | Dashboard UI (workspace, charts, compare) |
| `src/lib/resume/` | Zod schema, Gemini parse, PDF/DOCX extract, scoring, normalization, chart data |
| `src/components/AppErrorBoundary.tsx` | Global client error boundary |
