# Architecture

```text
Next.js / React / Tailwind / shadcn-style Button / Recharts
                 | same-origin /api proxy
FastAPI ---------+---------- SQLite (SQLAlchemy, WAL, foreign keys)
   | upload worker threads          | persisted jobs/checkpoints/results
   v                                v
PyMuPDF -> pages/sections -> bounded single inference worker
                               |
                     compression tokenizer chunks
                               |
                     real LLMLingua-2 classifier
                               |
                     preservation safeguards
                               |
                     Ollama structured chunk notes
                               |
                     evidence/numeric validation
                               |
                     hierarchical paper reduction
                               |
                     thematic evidence aggregation
                               |
                     optional prose compression
                               |
                     hierarchical cross-paper synthesis
                               |
                     cited reports + analytics + exports
```

## Modules

`pdf_extractor.py` owns block order, repeated margin removal, heading detection, text/metadata extraction and scanned-page warnings. `chunker.py` respects section/paragraph/sentence boundaries and measures every prospective chunk with the actual compressor tokenizer. Oversized individual sentences split using tokenizer character offsets, with no overlapping duplicate evidence. Special tokens are reserved beneath the model's 512-position limit.

`compressor.py` lazily creates and reuses official `PromptCompressor(use_llmlingua2=True)` with CPU/CUDA selection, a one-item inference batch, and a lock. It measures using the compression tokenizer rather than the library's internal OpenAI-tokenizer statistics. Invalid outputs, missing numeric strings or protected terms, and extreme token loss cause a less-aggressive retry then original-text fallback. Model initialization failures fail the job.

`ollama_client.py` uses nonstreaming `/api/chat`, temperature zero, seed 42, a Pydantic JSON schema, timeouts, and up to three validation attempts. Actual response usage is persisted even for rejected answers. Source documents occupy the untrusted data portion of the user message and cannot modify the system prompt.

Each request aliases its evidence IDs to short labels (`E1`, `E2`, …) and records their paper identity (`P1`, `P2`, …) outside compressed prose. An enum in the constrained generation schema permits only supplied labels. Labels are mapped back to original chunk IDs before evidence validation. Decoder limits bound claim count and length, preventing repetitive valid-looking JSON from exhausting the small model's output budget. Rejected response/error diagnostics remain persisted separately from accepted reports.

`evidence.py` maps every claim to immutable original chunks and rejects unknown IDs and unsupported numeric strings. `summarizer.py` packs evidence within a conservative byte-based bound; it reduces batches hierarchically before requesting a paper summary. `synthesizer.py` groups notes by theme, compresses prose optionally (never citation IDs or source records), and uses the same bounded reduction to generate integrated findings. Cross-paper agreements/contradictions require multiple paper identities.

`workers/pipeline.py` coordinates one inference job at a time. It commits each completed chunk compression/summary and paper summary independently. Retry skips those completed stages. Intermediate reductions are regenerated after failure; their prior calls remain in usage accounting. Interruptions are marked failed on server startup and can be resumed explicitly. Settings/source fingerprints identify comparable runs. UI status is polled from persisted state.

## Persistence

Final synthesis is generated section by section from matching evidence themes, with bounded reduction if needed. Comparison claims still require multiple paper identities. After three responses fail specifically that comparison check, the section abstains with empty claims and an application-authored notice; rejected prose is never accepted. The UI, JSON and Markdown expose this notice. All other validation failures stop the job. Retry correction space is reserved during context packing, and synthesis compression records are reused on resume.

| Table / model | Relationships and purpose |
| --- | --- |
| `projects` / Project | Project identity and creation time |
| `papers` / Paper | Project FK, content hash unique per project, extracted/editable metadata, warnings, extraction duration |
| `pages` / Page | Paper FK, original extracted text, page number, section-labeled blocks |
| `chunks` / Chunk | Paper FK, immutable source text, section, page span, ordinal, token count; serves as the immutable EvidenceReference |
| `jobs` / Job | Project FK, immutable run configuration, stage/status/progress, errors, duration |
| `compressions` / Compression | Job + chunk FKs; actual output, tokens, safeguards, timing; one per job/chunk |
| `reports` / Report | Job FK plus optional paper/chunk FKs; discriminated `chunk`, `paper`, `synthesis`, `synthesis_compression` records |
| `usage` / Usage | Job FK, stage/model, actual Ollama input/output counts, wall time, validation acceptance |

The three summary types share a validated JSON representation rather than nearly identical tables. Original evidence is addressed by chunk primary key; report citation arrays reference those IDs and are application-validated. Relational parents use enforced foreign keys with cascade deletion. Changing the paper collection removes stale jobs/reports before another run. Source files use server-generated hex IDs, never user paths; content hashes detect duplicates within a project. Failed uploads remove their partial files.

## Accounting

Compression counts and ratios are measured with one named Hugging Face tokenizer. Generation counts are Ollama's `prompt_eval_count` and `eval_count`, never summed with compression counts. Stages distinguish chunk, paper/intermediate, and synthesis/intermediate requests. Second-stage compression is persisted separately with original/output prose and its own tokenizer metrics. Extraction time is measured during upload; analysis job time includes loading, compression, inference, orchestration, and retries but excludes upload extraction. No quality or speedup claims are inferred from token reduction alone.

## Boundaries

FastAPI validates request settings and enforces size, signature, page-count, encryption, content and duplicate checks. UUID lookup gates file downloads/deletion; no API accepts an arbitrary path. CSV formula prefixes are escaped. React renders source/generated text as text, never HTML. Documents are untrusted, and generated claims require review.

This deployment is intentionally local and unauthenticated. Its single-process mutation lock is sufficient for one local API process, not multiple replicas. Public deployment, automatic OCR, exact table reconstruction, GPU provisioning, automated semantic quality grading, distributed queues, and schema migrations are outside this MVP.
