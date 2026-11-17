export type Job = { id: string; status: string; stage: string; progress: number; error: string | null; seconds: number; config: { baseline: boolean; model: string; retention: number } };
export type Paper = { id: string; title: string; filename: string; warnings: string[]; metadata_json: { authors: string | null; year: number | null; doi: string | null; page_count: number; abstract: string | null } };
export type Project = { id: string; name: string; created_at: string; paper_count?: number; job?: Job; papers?: Paper[]; jobs?: Job[] };
export type Claim = { text: string; evidence_ids: string[]; kind: string };
export type Report = { id: string; data: { sections: { heading: string; claims: Claim[]; notice?: string | null }[] }; validation: string };
export type Compression = { compressed_text: string; original_tokens: number; compressed_tokens: number; ratio: number; reduction_percent: number; seconds: number; fallback: string | null; tokenizer: string };
export type Chunk = { id: string; ordinal: number; section: string; page_start: number; page_end: number; original_text: string; compression: Compression | null };
export type Evidence = { chunk_id: string; paper_id: string; section: string; page_start: number; page_end: number; text: string };
export type Analytics = { job_id: string | null; baseline_mode?: boolean; tokenizer?: string; generation_model?: string; seconds?: number; llm_input_tokens?: number; llm_output_tokens?: number; llm_seconds?: number; papers: { id: string; title: string; original: number; compressed: number; chunks: number; ratio: number; compression_seconds: number; extraction_seconds: number; fallbacks: number }[]; stages: { stage: string; input_tokens: number; output_tokens: number; seconds: number }[]; synthesis_compression?: (Compression & {original_text: string})[]; baseline: { job_id: string; is_baseline: boolean; seconds: number; input_tokens: number; quality: string } | null };
export type Models = { ollama: { available: boolean; models: string[]; error?: string }; default_model: string; compression_model: string; compressor_loaded: boolean };
export async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch('/api' + path, { ...options, headers: { ...(options?.body instanceof FormData ? {} : { 'Content-Type': 'application/json' }), ...options?.headers } });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail || `Request failed (${response.status})`));
  }
  return response.status === 204 ? undefined as T : response.json();
}
