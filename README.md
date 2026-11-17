# ResearchCompress AI

A local, single-user scientific PDF analysis application. Next.js provides the research workspace; FastAPI persists documents and background jobs in SQLite. Microsoft's **actual LLMLingua-2** token classifier compresses chunks; **Ollama** generates structured, cited summaries and cross-paper synthesis. No paid inference API or API key is used.

## Requirements

- Python **3.12** recommended (avoid the system's Python 3.15 for ML wheel compatibility).
- Node.js **22 LTS** or newer, npm.
- [Ollama](https://ollama.com/download). On Windows, use the installer or the official portable ZIP described in [Ollama's Windows documentation](https://docs.ollama.com/windows).
- A practical starting point is 16 GB RAM and 10 GB free disk. CPU operation works but generation can be slow. GPU support depends on the installed PyTorch build and Ollama's supported hardware. These are setup guidelines, not benchmarked minimums.
- Internet access for package installation, the initial compression model/tokenizer download, and the Ollama model download. Once cached, inference is local.

## Install without Docker

Run from the repository root. On Windows:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend/requirements.txt
Copy-Item .env.example .env
cd frontend
npm ci
cd ..
ollama pull qwen2.5:3b
```

On macOS/Linux:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r backend/requirements.txt
cp .env.example .env
cd frontend && npm ci && cd ..
ollama pull qwen2.5:3b
```

Start Ollama (`ollama serve` if the desktop app is not already serving it). Then start these in **two separate terminals**:

On this prepared Windows checkout, the portable runtime and `qwen2.5:3b` weights are already under `storage/`. Run `./scripts/start-ollama.ps1` in a third terminal to reuse them; the script disables Ollama cloud access. An installation elsewhere can use the normal Ollama desktop app instead.

```powershell
./scripts/start-backend.ps1
```

```powershell
./scripts/start-frontend.ps1
```

macOS/Linux equivalents: `sh scripts/start-backend.sh` and `sh scripts/start-frontend.sh`.

Open **http://127.0.0.1:3000**. API documentation: **http://127.0.0.1:8000/docs**. Health: `/api/health`; model availability: `/api/models/status`.

The scripts set model caches under `storage/`. The backend lazily downloads and initializes `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank` once per process, reusing it for subsequent jobs. `llmlingua==0.2.2` is pinned; its installed `PromptCompressor` signature supports `use_llmlingua2`, `force_tokens`, and `force_reserve_digit`. No remote Hugging Face model code is trusted. A local tokenizer file used internally by LLMLingua also downloads on first use; this does not make an OpenAI API call.

`backend/requirements.lock.txt` records the exact Python packages used during validation; `frontend/package-lock.json` locks the frontend dependency tree. Baseline runs initialize only the compression tokenizer, not the classifier weights.

For a production frontend build, run `npm run build` then `npm run start` in `frontend`. The webpack build path avoids a Turbopack subprocess failure observed on this Windows environment. Keep the backend to **one process / one Uvicorn worker**.

## Docker Compose

With Docker Engine/Desktop running:

```sh
docker compose up --build -d
docker compose exec ollama ollama pull qwen2.5:3b
```

Open http://localhost:3000. Models and SQLite results persist in volumes/`storage`. The supplied Compose file uses CPU compression, binds published ports only to loopback, and leaves model pulling explicit. GPU-specific Docker configuration is not included. The Docker path requires separate validation on a host with a running Docker daemon.

## Use the workspace

1. Create a project and give it a research topic.
2. Drop one or more real scientific PDFs into **Papers & settings**. Each upload reports its byte progress, then extraction/section detection, then readiness or its own error. Repeated file hashes are not added twice.
3. Review warnings and metadata. Missing publication years and authors remain unknown; use **Edit metadata** to correct them.
4. Choose retention (75%, 50%, 30%, or custom), a downloaded Ollama model, and optional protected names/terms. These are retention targets, not promised savings.
5. Start analysis. The background worker records genuine stages and progress. First-time model initialization can take several minutes.
6. Open **Paper summaries** and **Cross-paper analysis**. Citation buttons show original passages and page references. One-paper projects do not run synthesis.
7. Inspect **Compression inspector** for actual original/compressed text, counts, ratios, timings, and safeguard fallbacks.
8. Review **Analytics**, select past runs, and download Markdown, JSON, or comparison CSV. Individual summaries have their own Markdown download.

Adding or removing a paper invalidates previous project runs, because their evidence collection has changed. Failed jobs can **Resume analysis** from completed chunk/paper checkpoints. After a server restart, unfinished jobs become failed with a resumable interruption message. There is no automatic unbounded retry loop.

For an interrupted development session, `./.venv/Scripts/python.exe scripts/inspect_run.py` reads the saved stage without a running server. Add `--diagnostics` to inspect the last rejected model responses locally. The UI and API never treat rejected answers as accepted findings.

## Configuration

Copy `.env.example` to `.env`; settings are validated on startup. Important controls:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_MODEL` | `qwen2.5:3b` | Default model; selectable in UI |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local inference server |
| `OLLAMA_CONTEXT` | `8192` | Request context budget |
| `OLLAMA_OUTPUT_TOKENS` | `1200` | Output reserve / generation limit |
| `OLLAMA_TIMEOUT` | `600` | Per-generation timeout in seconds |
| `COMPRESSION_DEVICE` | `auto` | CUDA if available, otherwise CPU; override `cpu` |
| `COMPRESSION_MODEL` | multilingual BERT LLMLingua-2 | Optional XLM-RoBERTa large alternative |
| `CHUNK_TARGET_TOKENS` | `400` | Compression-tokenizer chunk target, 64–450 |
| `MAX_UPLOAD_SIZE_MB` | `30` | Per-file size limit |
| `MAX_PAGES` | `500` | Per-PDF resource limit |
| `ENABLE_BASELINE_COMPARISON` | `false` | Opt in to additional uncompressed runs |
| `DATABASE_URL`, `STORAGE_PATH` | repository `storage/` | Persistence locations |

When changing the compression tokenizer or chunk size, create a new project/re-upload sources so existing immutable chunks are not reused with a different tokenizer. Keep the same source/model/generation configuration for comparisons. Alternative Ollama tokenizers are not assumed to match BERT.

## Tests

Fast tests mock inference only, not the production services. They use temporary databases and real generated PDF fixtures:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
cd frontend
npm run build
```

Real, opt-in integrations:

```powershell
$env:HF_HOME="$PWD\storage\model-cache"
$env:TIKTOKEN_CACHE_DIR="$PWD\storage\tiktoken"
$env:RUN_LLMLINGUA_INTEGRATION='1'
$env:RUN_OLLAMA_INTEGRATION='1'
.\.venv\Scripts\python.exe -m pytest backend/tests/test_integration.py -s -q
```

On POSIX prefix the pytest command with `RUN_LLMLINGUA_INTEGRATION=1 RUN_OLLAMA_INTEGRATION=1` and use `.venv/bin/python`. Actual tests require downloads / a running model and are intentionally skipped in the default suite. See `VALIDATION.md` for what was executed in the development environment.

## Reproducible end-to-end evaluation

Use two **real research PDFs that you have permission to process**. For example, the LLMLingua-2 paper and a related prompt compression paper from their original publication pages. The application does not ship fake summaries or benchmark outputs.

With all services running:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate.py path/to/first.pdf path/to/second.pdf
```

To compare an uncompressed run, enable `ENABLE_BASELINE_COMPARISON=true`, restart the backend, and append `--baseline`. The script uploads the same PDFs once, runs both configurations with the same model, temperature, seed, chunking and context settings, and saves real exports under `storage/evaluation/`. The dashboard matches runs only when their configuration and content hashes agree (except the baseline toggle).

Review a fixed set of source facts manually: numerical findings and their units/conditions, dataset/method names, limitations, and citations. Mark each as retained, missing, altered, or unsupported in each report. Numerical presence checks are **not** a scientific quality score. Compare total analysis time plus one shared extraction measurement; model loading, cache warmth, and hardware affect wall time. Compression does not necessarily improve latency. No paper-reported speedup is presented as a measured product result.

## Accuracy and operational limits

- Layout extraction uses PyMuPDF blocks with a two-column heuristic and repeated margin removal. Complex tables, equations, mixed layouts, and figure content may be imperfect. Image-only pages are flagged; scanned documents with no meaningful text are rejected. OCR is not bundled.
- Missing metadata is not inferred from bibliography years. Title falls back to the filename, visibly editable.
- Source pages/chunks are immutable in the API. Every model claim must cite existing chunk IDs; important numerical tokens must occur in those original cited passages. Cross-paper agreement/contradiction claims require at least two source papers.
- Generation uses short request-local labels constrained by the JSON schema. The application maps them back to full original evidence IDs before validation and persistence; this avoids small-model UUID copying errors. Paper identities travel with those labels during synthesis.
- These checks do **not** prove semantic entailment, correct units, causal validity, or absence of hallucinations. Review original evidence. Hierarchical reduction is lossy and can omit findings.
- Protected terms, digit reservation, less-aggressive retries, and original-text fallbacks favor preservation over savings. A zero-savings chunk is a legitimate measured outcome.
- Context packing uses a conservative UTF-8 byte bound including prompt/schema/output reserves. It never labels this estimate as generation token usage. Actual input/output counts come from Ollama, including rejected structured outputs. Extremely large single notes or nonconverging reductions fail clearly rather than truncate silently.
- Jobs are serialized in a bounded in-process worker; the database persists checkpoints. This is not a distributed queue. Run exactly one backend worker and do not launch two servers against the same SQLite file.
- No authentication or multi-tenant access control is included. This is a **local single-user MVP**. Do not expose it publicly without authentication, authorization, HTTPS, rate limiting, an upload proxy limit, and a production worker strategy. Local CORS restrictions are not authentication.
- Upload parsing occurs in FastAPI's worker thread before inference queueing. Multi-file upload UI reports extraction state but does not report per-page extraction percentages.

## Troubleshooting

- **Ollama offline / missing model:** run `ollama serve`, then `ollama pull qwen2.5:3b`; check `/api/models/status`. A model's exact installed tag must match the UI value.
- **RAM / out of memory:** close other model processes, select a smaller Ollama model, force `COMPRESSION_DEVICE=cpu`, and reduce context/output budgets together. Keep sufficient space for the schema and prompt. GPU wheels are installed separately according to PyTorch's instructions.
- **Model download failures:** allow access to Hugging Face and the tokenizer asset host; retry once network access is restored. The UI reports the actual failure, never substitutes a fake compressor.
- **Invalid generation / numerical validation failure:** review sources; a small model may fail strict output checks. Increase output/context capacity or select a more capable local model. Three failed responses stop the job; all returned usage remains counted.
- **Slow CPU:** initial loading and long papers take time. Progress follows completed work, not a timer. Leave one job running and inspect the logs.
- **Permission errors / OneDrive:** move the checkout to a regular writable directory if synchronization locks generated build/database files. Windows sandbox environments can block asyncio socket pairs and Next.js subprocesses; run development tools in a normal terminal.
- **Scanned PDFs:** run a local OCR tool on a copy before uploading. Check recognized equations and table values manually.

## Research and API references

- [LLMLingua-2, Findings of ACL 2024](https://arxiv.org/abs/2403.12968)
- [Microsoft LLMLingua source and model usage](https://github.com/microsoft/LLMLingua)
- [Ollama chat API: structured output and token counts](https://docs.ollama.com/api/chat)

See `ARCHITECTURE.md` for the persistence and processing design.
