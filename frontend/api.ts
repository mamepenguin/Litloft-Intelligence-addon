import { fetchJSON } from "@/lib/api";
import type { FileItem, FileType } from "@/types";

const API_BASE = "/api";

/**
 * Drive context header for every intelligence API call.
 *
 * Manifest scope is "drive" so the host's Generic Addon Proxy demands
 * X-Lit-Drive on every route. The header value must be ISO-8859-1 only,
 * so non-ASCII drive names (e.g. Japanese) are percent-encoded; the
 * intelligence backend decodes once via drive_context.require_drive.
 */
function driveHeaders(drive: string): HeadersInit {
  return { "X-Lit-Drive": encodeURIComponent(drive) };
}

// Semantic search types
export interface SemanticSearchMatch {
  type: "transcript" | "clip" | "metadata" | "content";
  text?: string;
  score: number;
  page?: number | null;
}

export interface SemanticSearchSegment {
  time_range: [number, number] | null;
  matches: SemanticSearchMatch[];
}

export interface SemanticSearchResult {
  file_id: string;
  drive: string;
  filename: string;
  file_type: string;
  score: number;
  match_types: string[];
  segments: SemanticSearchSegment[];
  /**
   * Hydrated FileItem-shaped metadata from core's
   * ``POST /api/internal/files/bulk``. Null when core is unreachable
   * or the file is missing/trashed — consumers should fall back to
   * the IndexedFile-snapshot fields above (filename, file_type) for
   * minimal display.
   */
  file: FileItem | null;
}

export interface SemanticSearchResponse {
  available: boolean;
  results: SemanticSearchResult[];
  total: number;
}

// One file currently being processed by a given task. Filename is null
// when the file has been purged from the search index between enqueue
// and the dashboard poll, or when it predates filename hydration.
export interface QueueProcessingFile {
  file_id: string;
  filename: string | null;
}

export interface QueueTaskBreakdown {
  waiting: number;
  processing: QueueProcessingFile[];
}

// Task kinds that may appear under queue.tasks. The four indexing
// types (metadata, clip, whisper, text_content) are always present;
// LLM-task entries (auto_tags, summaries, vision_describe,
// transcript_refine) are only present when their worker is running.
export type QueueTaskKind =
  | "metadata"
  | "clip"
  | "whisper"
  | "text_content"
  | "auto_tags"
  | "summaries"
  | "vision_describe"
  | "transcript_refine";

export interface SearchServiceStatus {
  available: boolean;
  status?: string;
  indexed?: {
    total: number;
    metadata: number;
    clip: number;
    whisper: number;
    text?: number;
  };
  pending?: {
    total: number;
    metadata?: number;
    clip: number;
    whisper: number;
    text?: number;
  };
  queue?: {
    processing: number;
    waiting: number;
    paused: boolean;
    tasks?: Partial<Record<QueueTaskKind, QueueTaskBreakdown>>;
  };
  models?: { whisper: string; clip: string; text_embedding: string };
}

export interface KeywordScore {
  word: string;
  score?: number;
  source_tfidf?: number;
  target_tfidf?: number;
  relevance?: number;
}

export interface SimilarFileItem {
  file_id: string;
  drive: string;
  filename: string;
  file_type: string;
  mime_type: string;
  score: number;
  match_type: string;
  primary_score: number | null;
  secondary_score: number | null;
  shared_keywords: KeywordScore[];
}

export interface SimilarFilesResponse {
  available: boolean;
  results: SimilarFileItem[];
  source_keywords: KeywordScore[];
}

export interface SearchSourceCounts {
  text_vector: number;
  clip_vector: number;
  keyword: number;
  transcript_keyword: number;
}

export interface SearchCompareResponse {
  available: boolean;
  rrf: SemanticSearchResponse;
  cosine: SemanticSearchResponse;
  rrf_no_cutoff: SemanticSearchResponse;
  cosine_no_cutoff: SemanticSearchResponse;
  source_counts: SearchSourceCounts;
}

export interface TranscriptChunkItem {
  index: number;
  text: string;
  start: number;
  end: number;
  // Populated when the chunk has been AI-refined. `refinedAt` is the
  // ISO timestamp of the refine run. Null / undefined for unrefined
  // chunks. Originals are not preserved — refine re-chunks the
  // transcript on punctuation boundaries, so per-chunk originals
  // would no longer align.
  refinedAt?: string | null;
}

export interface TranscriptResponse {
  available: boolean;
  file_id?: string;
  drive?: string;
  language?: string;
  chunks?: TranscriptChunkItem[];
}

export interface ClipTimestampItem {
  start: number;
  content_preview: string;
}

export interface ClipTimestampsResponse {
  available: boolean;
  file_id?: string;
  drive?: string;
  timestamps?: ClipTimestampItem[];
}

// Semantic search
export async function semanticSearch(
  query: string,
  drive: string,
  params?: { limit?: number; type?: FileType }
): Promise<SemanticSearchResponse> {
  const searchParams = new URLSearchParams({ q: query });
  if (params?.limit) searchParams.set("limit", String(params.limit));
  if (params?.type) searchParams.set("type", params.type);
  try {
    return await fetchJSON<SemanticSearchResponse>(
      `${API_BASE}/addons/intelligence/search?${searchParams.toString()}`,
      { headers: driveHeaders(drive) },
    );
  } catch {
    return { available: false, results: [], total: 0 };
  }
}

export async function searchCompare(
  query: string,
  drive: string,
  params?: { limit?: number; type?: FileType }
): Promise<SearchCompareResponse> {
  const searchParams = new URLSearchParams({ q: query });
  if (params?.limit) searchParams.set("limit", String(params.limit));
  if (params?.type) searchParams.set("type", params.type);
  try {
    return await fetchJSON<SearchCompareResponse>(
      `${API_BASE}/addons/intelligence/search/compare?${searchParams.toString()}`,
      { headers: driveHeaders(drive) },
    );
  } catch {
    return {
      available: false,
      rrf: { available: false, results: [], total: 0 },
      cosine: { available: false, results: [], total: 0 },
      rrf_no_cutoff: { available: false, results: [], total: 0 },
      cosine_no_cutoff: { available: false, results: [], total: 0 },
      source_counts: { text_vector: 0, clip_vector: 0, keyword: 0, transcript_keyword: 0 },
    };
  }
}

export async function getSearchStatus(
  drive?: string,
): Promise<SearchServiceStatus> {
  // /status returns process-global counters (queue, indexed totals
  // across all drives). The host marks it drive_optional so the admin
  // dashboard can call it without a drive context, while per-drive
  // pages still pass their drive for consistency.
  try {
    return await fetchJSON<SearchServiceStatus>(
      `${API_BASE}/addons/intelligence/status`,
      drive ? { headers: driveHeaders(drive) } : {},
    );
  } catch {
    return { available: false };
  }
}

/**
 * Fetch similar files. Throws on transport failure so callers can
 * distinguish "addon proxy timed out / 5xx" from a successful empty
 * response. The first call against a cold file commonly hits the
 * 15 s proxy timeout while CLIP / tf-idf / whisper similarity is
 * computed; the backend continues and caches the result, so a retry
 * a few seconds later succeeds instantly.
 */
export async function getSimilarFiles(
  fileId: string,
  drive: string,
  limit: number = 6
): Promise<SimilarFilesResponse> {
  const response = await fetchJSON<Omit<SimilarFilesResponse, "available">>(
    `${API_BASE}/addons/intelligence/similar/${fileId}?limit=${limit}`,
    { headers: driveHeaders(drive) },
  );
  return {
    available: true,
    results: response.results ?? [],
    source_keywords: response.source_keywords ?? [],
  };
}

// Queue control. The queue is a process-global resource owned by the
// addon container; per-drive context is meaningless for it. The host
// marks every /queue/* route drive_optional so admin tooling can call
// them without an active drive, while per-drive widgets that happen
// to surface a queue button can still pass a drive header for
// observability.
export async function searchQueuePause(drive?: string): Promise<void> {
  await fetchJSON(`${API_BASE}/addons/intelligence/queue/pause`, {
    method: "POST",
    headers: drive ? driveHeaders(drive) : undefined,
  });
}

export async function searchQueueResume(drive?: string): Promise<void> {
  await fetchJSON(`${API_BASE}/addons/intelligence/queue/resume`, {
    method: "POST",
    headers: drive ? driveHeaders(drive) : undefined,
  });
}

export async function searchQueuePrioritize(
  fileId: string,
  drive?: string,
): Promise<void> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (drive) Object.assign(headers, driveHeaders(drive));
  await fetchJSON(`${API_BASE}/addons/intelligence/queue/prioritize`, {
    method: "POST",
    headers,
    body: JSON.stringify({ file_id: fileId }),
  });
}

// Search inspection APIs
export async function getFileTranscript(
  fileId: string,
  drive: string,
): Promise<TranscriptResponse> {
  try {
    return await fetchJSON<TranscriptResponse>(
      `${API_BASE}/addons/intelligence/files/${fileId}/transcript`,
      { headers: driveHeaders(drive) },
    );
  } catch {
    return { available: false };
  }
}

// --- Transcript refine (AI correction of ASR output) ---

export interface RefineFileResponse {
  job_id: string;
  chunk_count: number;
}

export async function refineFileTranscript(
  fileId: string,
  drive: string,
): Promise<RefineFileResponse> {
  return fetchJSON<RefineFileResponse>(
    `${API_BASE}/addons/intelligence/refine/files/${fileId}`,
    { method: "POST", headers: driveHeaders(drive) },
  );
}

export interface RefineFolderResponse {
  queued: number;
  jobs: string[];
}

export async function refineFolderTranscripts(
  drive: string,
  fileIds: string[],
): Promise<RefineFolderResponse> {
  return fetchJSON<RefineFolderResponse>(
    `${API_BASE}/addons/intelligence/refine/folders`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json", ...driveHeaders(drive) },
      body: JSON.stringify({ drive, file_ids: fileIds }),
    },
  );
}

export async function getClipTimestamps(
  fileId: string,
  drive: string,
): Promise<ClipTimestampsResponse> {
  try {
    return await fetchJSON<ClipTimestampsResponse>(
      `${API_BASE}/addons/intelligence/files/${fileId}/clip-timestamps`,
      { headers: driveHeaders(drive) },
    );
  } catch {
    return { available: false };
  }
}

export function getFrameUrl(fileId: string, timestamp: number): string {
  // <img src> can't carry headers. The host proxy's file_access pre_check
  // already verifies the caller's session can read this file regardless
  // of the missing X-Lit-Drive — and the URL itself is unguessable enough
  // (UUIDs) that this is no worse than a thumbnail URL. If we ever need
  // strict drive context here we'll need a different approach (signed
  // URL or a hidden iframe with fetch+blob).
  return `${API_BASE}/addons/intelligence/files/${fileId}/frame?t=${timestamp}`;
}

// Suggested tags types and API
export interface SuggestedTagsResponse {
  available: boolean;
  file_id?: string;
  tags?: string[];
  model?: string;
  status?: string;
  created_at?: string;
}

export async function getSuggestedTags(
  fileId: string,
  drive: string,
): Promise<SuggestedTagsResponse> {
  try {
    return await fetchJSON<SuggestedTagsResponse>(
      `${API_BASE}/addons/intelligence/files/${fileId}/suggested-tags`,
      { headers: driveHeaders(drive) },
    );
  } catch {
    return { available: false };
  }
}

export async function dismissSuggestedTags(
  fileId: string,
  drive: string,
): Promise<void> {
  await fetchJSON(
    `${API_BASE}/addons/intelligence/files/${fileId}/suggested-tags/dismiss`,
    { method: "POST", headers: driveHeaders(drive) },
  );
}

export async function regenerateSuggestedTags(
  fileId: string,
  drive: string,
): Promise<void> {
  await fetchJSON(
    `${API_BASE}/addons/intelligence/files/${fileId}/suggested-tags/regenerate`,
    { method: "POST", headers: driveHeaders(drive) },
  );
}

export interface BatchSuggestedTagsResponse {
  queued: number;
  skipped: number;
}

export async function batchSuggestedTags(
  fileIds: string[],
  drive: string,
): Promise<BatchSuggestedTagsResponse> {
  return fetchJSON<BatchSuggestedTagsResponse>(
    `${API_BASE}/addons/intelligence/batch/suggested-tags`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json", ...driveHeaders(drive) },
      body: JSON.stringify({ file_ids: fileIds }),
    }
  );
}

// Summaries types and API
export type SummaryMissingReason =
  | "not_generated"
  | "insufficient_content"
  | "unsupported_type"
  | "file_not_found";

export interface SummaryResponse {
  available: boolean;
  file_id?: string;
  short_summary?: string;
  long_summary?: string;
  model?: string;
  context_type?: string;
  was_truncated?: boolean;
  status?: "generated" | "hidden" | string;
  created_at?: string;
  // ISO timestamp of the last user edit. Absent/null means the
  // displayed text is the raw AI output.
  edited_at?: string | null;
  // True when an AI snapshot is stored and revert is possible.
  has_original?: boolean;
  reason?: SummaryMissingReason | string;
}

export async function getSummary(
  fileId: string,
  drive: string,
): Promise<SummaryResponse> {
  try {
    return await fetchJSON<SummaryResponse>(
      `${API_BASE}/addons/intelligence/files/${fileId}/summary`,
      { headers: driveHeaders(drive) },
    );
  } catch {
    return { available: false };
  }
}

export async function regenerateSummary(
  fileId: string,
  drive: string,
): Promise<void> {
  await fetchJSON(
    `${API_BASE}/addons/intelligence/files/${fileId}/summary/regenerate`,
    { method: "POST", headers: driveHeaders(drive) },
  );
}

export async function editSummary(
  fileId: string,
  drive: string,
  payload: { short_summary: string; long_summary: string },
): Promise<SummaryResponse> {
  return fetchJSON<SummaryResponse>(
    `${API_BASE}/addons/intelligence/files/${fileId}/summary/edit`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json", ...driveHeaders(drive) },
      body: JSON.stringify(payload),
    },
  );
}

export async function revertSummary(
  fileId: string,
  drive: string,
): Promise<SummaryResponse> {
  return fetchJSON<SummaryResponse>(
    `${API_BASE}/addons/intelligence/files/${fileId}/summary/revert`,
    { method: "POST", headers: driveHeaders(drive) },
  );
}

export interface BatchSummariesResponse {
  queued: number;
  skipped: number;
}

export async function batchSummaries(
  fileIds: string[],
  drive: string,
): Promise<BatchSummariesResponse> {
  return fetchJSON<BatchSummariesResponse>(
    `${API_BASE}/addons/intelligence/batch/summaries`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json", ...driveHeaders(drive) },
      body: JSON.stringify({ file_ids: fileIds }),
    }
  );
}

// --- Detailed (long-form Markdown) summary ---

export type DetailedSummaryStatus =
  | "generating"
  | "generated"
  | "failed"
  | string;

export interface DetailedSummaryResponse {
  available: boolean;
  file_id?: string;
  detailed_summary?: string;
  status?: DetailedSummaryStatus;
  model?: string;
  generated_at?: string;
  context_chars?: number;
  was_truncated?: boolean;
  error?: string;
  reason?: SummaryMissingReason | string;
  // ISO timestamp of the last user edit to the detailed summary. When
  // non-null the UI surfaces the "edited" badge and the "revert to AI
  // version" button. Backend column is `detailed_edited_at`, but the
  // response field drops the `detailed_` prefix because the response
  // schema is detailed-only (matches short/long pattern).
  edited_at?: string | null;
  // True when the backend still has the original AI-generated text
  // available for revert. Matches `detailed_original IS NOT NULL`.
  has_original?: boolean;
}

// --- Detailed summary citations (Phase 1: embedding-based backlinks) ---

/**
 * A single citation row returned by
 * GET /files/{file_id}/summary/detailed/citations.
 *
 * `segment_text` is the literal bullet/paragraph text the backend
 * computed similarity for; the UI uses it to locate the corresponding
 * DOM node and wrap it with the hover-trigger icon. `has_citation`
 * false means the top-1 score fell below the configured threshold —
 * the UI surfaces an amber ⚠ marker to flag potential hallucinations.
 */
export interface DetailedSummaryCitation {
  section_path: string;
  segment_type: "bullet" | "paragraph";
  segment_text: string;
  chunk_ids: string[];
  top_score: number;
  has_citation: boolean;
}

export interface DetailedSummaryCitationsResponse {
  available: boolean;
  file_id?: string;
  citations?: DetailedSummaryCitation[];
}

/**
 * Fetch the latest citation backlinks for a detailed summary.
 *
 * The backend recomputes citations automatically whenever the summary
 * is generated, edited, or reverted; the frontend listens for the
 * `intelligence.detailed_summary.citations_ready` WS event and re-calls
 * this endpoint. Failure falls back to an empty list so a missing
 * backend (older intelligence build) doesn't crash the detail page.
 */
export async function getDetailedSummaryCitations(
  fileId: string,
  drive: string,
): Promise<DetailedSummaryCitationsResponse> {
  try {
    return await fetchJSON<DetailedSummaryCitationsResponse>(
      `${API_BASE}/addons/intelligence/files/${fileId}/summary/detailed/citations`,
      { headers: driveHeaders(drive) },
    );
  } catch {
    return { available: false, citations: [] };
  }
}

/**
 * Excerpt payload for a single chunk, fetched on hover.
 *
 * The excerpt is split into `prefix` / `target` / `suffix` so the UI
 * can highlight the actual cited chunk against muted neighbour
 * context. `target` is the chunk's own text; `prefix` / `suffix` each
 * carry up to ±100 chars of neighbour text (with their space
 * separator and a "… " marker when the neighbour was truncated).
 * Concatenating the three strings reproduces the flat single-line
 * rendering used before the split was introduced. `start_time` is
 * non-null for audio/video chunks; the UI uses it to seek the
 * <video>/<audio> element. `page` is non-null for document chunks
 * (pdf/epub) and drives the text-preview scroll.
 */
export interface CitationChunkExcerpt {
  chunk_id: string;
  file_id: string;
  prefix: string;
  target: string;
  suffix: string;
  start_time: number | null;
  end_time: number | null;
  page: number | null;
}

export async function getCitationChunkExcerpt(
  fileId: string,
  chunkId: string,
  drive: string,
): Promise<CitationChunkExcerpt | null> {
  try {
    return await fetchJSON<CitationChunkExcerpt>(
      `${API_BASE}/addons/intelligence/files/${fileId}/chunks/${chunkId}/excerpt`,
      { headers: driveHeaders(drive) },
    );
  } catch {
    return null;
  }
}

// --- Detailed summary editing (Phase 2) ---

/**
 * Splice a heading-anchored range of the detailed summary with a
 * user-edited Markdown fragment.
 *
 * `section_heading` is the H2 anchor; set `subsection_heading` to
 * narrow the edit to a single `### Heading` inside that H2. The
 * fragment is verbatim — it may include `##` / `###` lines, restructure
 * the hierarchy, or drop a heading entirely; the backend re-parses on
 * save so structural changes propagate to the next render.
 *
 * Response mirrors `getDetailedSummary` so the UI can rehydrate from a
 * single call.
 */
export async function editDetailedSummarySection(
  fileId: string,
  drive: string,
  payload: {
    section_heading: string;
    subsection_heading?: string | null;
    new_content: string;
  },
): Promise<DetailedSummaryResponse> {
  return fetchJSON<DetailedSummaryResponse>(
    `${API_BASE}/addons/intelligence/files/${fileId}/summary/detailed/section`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json", ...driveHeaders(drive) },
      body: JSON.stringify(payload),
    },
  );
}

/**
 * Restore the AI-generated snapshot, discarding all user edits and
 * clearing `detailed_edited_at`. Returns the refreshed summary so
 * the UI can rehydrate without a follow-up GET.
 */
export async function revertDetailedSummary(
  fileId: string,
  drive: string,
): Promise<DetailedSummaryResponse> {
  return fetchJSON<DetailedSummaryResponse>(
    `${API_BASE}/addons/intelligence/files/${fileId}/summary/detailed/revert`,
    { method: "POST", headers: driveHeaders(drive) },
  );
}

/**
 * Regenerate the detailed summary from scratch.
 *
 * When the current summary is user-edited (`detailed_edited_at !== null`)
 * the server returns 409 unless `force: true` is passed. The UI detects
 * the 409 in a prior step and calls this helper with `{ force: true }`
 * after confirming with the user. The regenerate path always bypasses
 * `detailed_original` — the edited body is simply overwritten.
 */
export async function regenerateDetailedSummary(
  fileId: string,
  drive: string,
  options?: { force?: boolean },
): Promise<{ status: string; message: string }> {
  return fetchJSON<{ status: string; message: string }>(
    `${API_BASE}/addons/intelligence/files/${fileId}/summary/detailed/regenerate`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json", ...driveHeaders(drive) },
      body: JSON.stringify({ force: options?.force ?? false }),
    },
  );
}

export async function getDetailedSummary(
  fileId: string,
  drive: string,
): Promise<DetailedSummaryResponse> {
  try {
    return await fetchJSON<DetailedSummaryResponse>(
      `${API_BASE}/addons/intelligence/files/${fileId}/summary/detailed`,
      { headers: driveHeaders(drive) },
    );
  } catch {
    return { available: false };
  }
}

/**
 * Kick off detailed-summary generation.
 *
 * Returns a 202-style ack immediately; the row flips to
 * ``status: "generating"`` in the DB and the caller polls
 * `getDetailedSummary` until `available` flips true (or a `failed`
 * status surfaces).
 *
 * Errors are rethrown so the UI can distinguish 400 (feature off /
 * unsupported file) from 409 (already exists — regenerate flow must
 * call `deleteDetailedSummary` first).
 */
export async function startDetailedSummary(
  fileId: string,
  drive: string,
): Promise<{ status: string; message: string }> {
  return fetchJSON<{ status: string; message: string }>(
    `${API_BASE}/addons/intelligence/files/${fileId}/summary/detailed`,
    { method: "POST", headers: driveHeaders(drive) },
  );
}

export async function deleteDetailedSummary(
  fileId: string,
  drive: string,
): Promise<void> {
  await fetchJSON(
    `${API_BASE}/addons/intelligence/files/${fileId}/summary/detailed`,
    { method: "DELETE", headers: driveHeaders(drive) },
  );
}

/**
 * Build the URL for the ``.md`` download endpoint.
 *
 * Returned as a string so the UI can use it directly in an ``<a
 * download>`` tag or ``window.location`` navigation. The host proxy
 * requires ``X-Lit-Drive`` for every intelligence route, so the caller
 * cannot use a plain anchor — instead, fetch the blob and create an
 * object URL:
 *
 * ```ts
 * const res = await fetch(getDetailedSummaryDownloadUrl(id), {
 *   credentials: "include",
 *   headers: driveHeaders(drive),
 * });
 * ```
 */
export function getDetailedSummaryDownloadUrl(fileId: string): string {
  return `${API_BASE}/addons/intelligence/files/${fileId}/summary/detailed.md`;
}

/**
 * Fetch the detailed-summary Markdown with the drive header and
 * trigger a browser download via a temporary object URL.
 *
 * Kept as an API-layer helper so the UI code doesn't have to repeat
 * the header plumbing. Extracts the ``filename`` from the server's
 * Content-Disposition header when present so the download lands with
 * a meaningful name.
 */
export async function downloadDetailedSummary(
  fileId: string,
  drive: string,
): Promise<void> {
  const res = await fetch(getDetailedSummaryDownloadUrl(fileId), {
    credentials: "include",
    headers: driveHeaders(drive),
  });
  if (!res.ok) {
    throw new Error(`Download failed: ${res.status} ${res.statusText}`);
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = extractFilenameFromDisposition(
    res.headers.get("content-disposition"),
  ) ?? `${fileId}_summary.md`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  // Hold the blob URL long enough for the browser's download manager
  // to finish reading. Revoking too early (even via setTimeout(0))
  // has been observed to leave Chrome stuck on "downloading" because
  // the in-flight read sees an invalidated URL; Safari likewise
  // cancels a synchronous revoke. 40s mirrors FileSaver.js and is
  // harmless — the blob is GC'd once the anchor and reference go away.
  setTimeout(() => URL.revokeObjectURL(url), 40_000);
}

/**
 * Extract the filename from a Content-Disposition header value.
 *
 * Prefers the RFC 5987 ``filename*=UTF-8''...`` form (needed for
 * non-ASCII filenames) and falls back to the ASCII ``filename="..."``
 * parameter. Returns null when neither is present.
 */
function extractFilenameFromDisposition(
  disposition: string | null,
): string | null {
  if (!disposition) return null;
  // RFC 5987 extended parameter: filename*=UTF-8''<percent-encoded>
  const starMatch = /filename\*\s*=\s*UTF-8''([^;]+)/i.exec(disposition);
  if (starMatch) {
    try {
      return decodeURIComponent(starMatch[1]);
    } catch {
      // fall through to the ASCII fallback
    }
  }
  const asciiMatch = /filename\s*=\s*"([^"]+)"/i.exec(disposition);
  return asciiMatch ? asciiMatch[1] : null;
}

// --- Vision description (image LLM description) ---

export type VisualDescriptionStatus =
  | "pending"
  | "success"
  | "failed"
  | "unsupported"
  | null;

export interface VisualDescriptionResponse {
  file_id: string;
  visual_description: string | null;
  status: VisualDescriptionStatus | string;
  model: string | null;
  generated_at: string | null;
}

/**
 * Fetch the current vision description state for a file.
 *
 * Returns `null` when the backend responds with 404 (feature off
 * globally, per-drive policy OFF, or file not found). UI treats null as
 * "feature unavailable — hide the section".
 */
export async function getVisualDescription(
  fileId: string,
  drive: string,
): Promise<VisualDescriptionResponse | null> {
  try {
    return await fetchJSON<VisualDescriptionResponse>(
      `${API_BASE}/addons/intelligence/files/${fileId}/visual_description`,
      { headers: driveHeaders(drive) },
    );
  } catch {
    return null;
  }
}

/**
 * Kick off vision description generation for one file.
 *
 * Returns `{ status: "accepted" }` on success. The UI should poll
 * `getVisualDescription` to observe `pending → success|failed`.
 */
export async function generateVisualDescription(
  fileId: string,
  drive: string,
): Promise<{ status: string; file_id: string }> {
  return fetchJSON<{ status: string; file_id: string }>(
    `${API_BASE}/addons/intelligence/files/${fileId}/visual_description/generate`,
    { method: "POST", headers: driveHeaders(drive) },
  );
}

/**
 * Clear the stored description + vision_description embeddings for a
 * file. Backend returns 404 when nothing to clear — callers that use
 * this as part of a regenerate flow should treat 404 as a success.
 */
export async function deleteVisualDescription(
  fileId: string,
  drive: string,
): Promise<void> {
  await fetchJSON(
    `${API_BASE}/addons/intelligence/files/${fileId}/visual_description`,
    { method: "DELETE", headers: driveHeaders(drive) },
  );
}

export interface FolderVisualDescriptionResponse {
  queued: number;
  file_ids: string[];
}

export interface FolderVisualDescriptionTooManyError {
  kind: "too_many_files";
  max: number;
  requested: number;
}

/**
 * Fan-out: enqueue every image file under ``drive/path`` for vision
 * description. Rejects with a `FolderVisualDescriptionTooManyError`
 * when the folder exceeds the backend's `MAX_BULK_ENQUEUE` cap (413).
 * Other failures surface as plain `Error`s with the server message.
 */
export async function generateFolderVisualDescription(
  drive: string,
  fileIds: string[],
): Promise<FolderVisualDescriptionResponse> {
  const res = await fetch(
    `${API_BASE}/addons/intelligence/folders/visual_description/generate`,
    {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json", ...driveHeaders(drive) },
      body: JSON.stringify({ drive, file_ids: fileIds }),
    },
  );
  if (res.status === 413) {
    let detail: unknown = null;
    try {
      const body = (await res.json()) as { detail?: unknown };
      detail = body?.detail;
    } catch {
      // ignore — fall through to generic shape
    }
    const parsed =
      detail && typeof detail === "object" && !Array.isArray(detail)
        ? (detail as Record<string, unknown>)
        : {};
    const err = new Error("too_many_files") as Error & {
      info: FolderVisualDescriptionTooManyError;
    };
    err.info = {
      kind: "too_many_files",
      max: typeof parsed.max === "number" ? parsed.max : 500,
      requested: typeof parsed.requested === "number" ? parsed.requested : 0,
    };
    throw err;
  }
  if (!res.ok) {
    throw new Error(`Folder visual description failed: ${res.status}`);
  }
  return (await res.json()) as FolderVisualDescriptionResponse;
}

// --- RAG (question answering) ---

export interface Citation {
  file_id: string;
  drive: string;
  filename: string;
  file_type: string;
  quote: string;
  relevance: number;
  segment_location: string | null;
}

export interface Source {
  file_id: string;
  drive: string;
  filename: string;
  file_type: string;
  score: number;
  match_types: string[];
}

export interface AnswerResponse {
  query: string;
  answer: string | null;
  citations: Citation[];
  sources: Source[];
  retrieved_count: number;
  took_ms: number;
}

export interface AskOptions {
  topK?: number;
  fileType?: FileType;
  signal?: AbortSignal;
}

// Intelligence /status shape extended with feature flags + llm enablement.
// This mirrors the backend FeaturesStatus + LLMStatus subset the AskSearchMode
// gate needs. The other endpoints use SearchServiceStatus which is a minimal
// projection — we keep both to avoid breaking existing callers.
export interface IntelligenceStatus {
  status?: string;
  features?: {
    indexing: boolean;
    search: boolean;
    auto_tags: string;
    summaries: string;
    rag: boolean;
  };
  llm?: {
    provider: string;
    model: string;
    enabled: boolean;
    output_language: string;
  };
}

export async function getIntelligenceStatus(
  drive: string,
  signal?: AbortSignal,
): Promise<IntelligenceStatus | null> {
  try {
    const res = await fetch(`${API_BASE}/addons/intelligence/status`, {
      credentials: "include",
      headers: driveHeaders(drive),
      signal,
    });
    if (!res.ok) return null;
    return (await res.json()) as IntelligenceStatus;
  } catch {
    return null;
  }
}

// --- Find mode (single-shot file-list output) ---

/**
 * Override values for the four decomposed slots, sent on chip × re-POST.
 *
 * Spec: ``2026-04-30-intelligence-find-mode.md`` §3.1.
 *
 * The frontend builds this object from the most-recent ``decomposed``
 * snapshot, replacing the cleared slot with ``"none"`` (or ``""`` for
 * ``semantic_query``). The backend treats overrides as the canonical
 * decomposition for the request, bypassing the LLM ``query_decomposer``.
 */
export interface FindOverrides {
  time_range?: string;
  personal_scope?: string;
  file_type_hint?: string;
  semantic_query?: string;
}

/**
 * Decomposed query slots returned alongside results. ``time_range`` is
 * a half-resolved object so the chip layer can show "先週 (4/23-4/30)"
 * even though the raw label is "last_week".
 */
export interface FindDecomposed {
  time_range: {
    kind: string;
    value?: string;
    after?: string | null;
    before?: string | null;
  };
  personal_scope: string;
  file_type_hint: string;
  semantic_query: string;
  category_expansion: string[];
}

export interface FindResultHit {
  kind: string;
  location: { start_seconds?: number; end_seconds?: number } | null;
  text: string;
}

export interface FindResultFile {
  name: string;
  file_type: string;
  thumbnail_url: string;
  viewed_at: string | null;
}

export interface FindResultEntry {
  file_id: string;
  score: number;
  hit: FindResultHit;
  file: FindResultFile;
}

export interface FindResponse {
  decomposed: FindDecomposed;
  results: FindResultEntry[];
  total: number;
  limit: number;
}

/**
 * POST a Find query to the intelligence addon and return a single-shot
 * JSON response (no SSE — Find has no LLM streaming).
 *
 * Spec: ``2026-04-30-intelligence-find-mode.md`` §3.2.
 *
 * Header conventions:
 *  - ``X-Lit-Drive``: required, percent-encoded for non-ASCII drives
 *    (matches ``driveHeaders`` everywhere else in this module).
 *  - ``X-HV-Viewer-Id``: forwarded from the ``hv_viewer`` cookie when
 *    present so the personal-history filter (Stage B) can engage.
 *
 * Throws on non-2xx so the page-level error surface can render a
 * graceful message.
 */
export async function findFiles(
  question: string,
  drive: string,
  options?: { limit?: number; overrides?: FindOverrides },
): Promise<FindResponse> {
  const body: Record<string, unknown> = { question };
  if (options?.limit != null) body.limit = options.limit;
  if (options?.overrides) body.overrides = options.overrides;

  // Viewer-id is injected by the host addon_proxy from the `lit_viewer`
  // cookie (mirrors /ask). The frontend MUST NOT read the cookie or set
  // X-Lit-Viewer-Id directly — the proxy strips client-supplied values
  // to prevent forgery.
  const headers: HeadersInit = {
    "Content-Type": "application/json",
    ...driveHeaders(drive),
  };

  const res = await fetch(`${API_BASE}/addons/intelligence/find`, {
    method: "POST",
    credentials: "include",
    headers,
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const errBody = (await res.json()) as { detail?: string };
      if (errBody?.detail) detail = errBody.detail;
    } catch {
      // ignore — fall back to status text
    }
    const err = new Error(detail) as Error & { status?: number };
    err.status = res.status;
    throw err;
  }

  return (await res.json()) as FindResponse;
}

// --- Streaming Ask (SSE) ---

/**
 * Events yielded by the `askQuestionStream` async generator.
 *
 * The backend emits them in this order (always):
 *   1. `keywords` — the transformed search-keyword string.
 *   2. `sources`  — the access-filtered retrieved file list.
 *   3. `answer_chunk` × N — token chunks, in LLM output order.
 *   4. `citations` — final anti-hallucination-filtered citations.
 *   5. `done`     — terminal event, optionally carrying `error`.
 *
 * Using a discriminated union keeps the consumer's switch exhaustive;
 * adding a new kind on the backend surfaces as a TS error in the UI.
 */
/**
 * Resolved time window from the personal-history Stage A decomposition.
 * Both ends are naive ISO-8601 strings (the host's WatchHistory column
 * is naive UTC) and either end may be null for "unbounded".
 */
export interface DecomposedTimeRange {
  label: string; // "today" | "yesterday" | "this_week" | ... | "none"
  after: string | null;
  before: string | null;
}

export interface DecomposedQueryPayload {
  time_range: DecomposedTimeRange;
  personal_scope: "viewed" | "not_viewed" | "none";
  file_type_hint: "video" | "audio" | "image" | "text" | "none";
  semantic_query: string;
}

export type AskStreamEvent =
  | { kind: "keywords"; keywords: string }
  // Stage A output (personal-history feature only). Surfaces the
  // structured form the LLM extracted from the user's natural-language
  // question — emitted before ``keywords`` so the UI can show "we
  // think you mean ..." while retrieval is still warming up.
  | { kind: "query_decomposed"; decomposed: DecomposedQueryPayload }
  // Stage B result. Emitted only when Stage B actually ran (i.e. the
  // decomposed query had a personal signal). ``matched_file_count``
  // is the size of the file_id_scope that the retriever will use.
  | {
      kind: "history_filter";
      drive: string | null;
      kind_label: "viewed" | "not_viewed";
      matched_file_count: number;
    }
  // Stage C semantic category expansion (e.g. "SF" → ["SF", "science
  // fiction", "宇宙船", ...]). Emitted only when category_expansion
  // is enabled, has a semantic_query to expand, and the LLM produced
  // more than one surface form.
  | {
      kind: "category_expanded";
      semantic_query: string;
      expanded: string[];
    }
  // Hierarchical RAG Stage 2 multi-query expansion — emitted only when
  // the hierarchical pipeline runs (config on, drive set, shortlist
  // confident, ≥1 shortlist file accessible). Bypassed paths skip it.
  // ``clues`` is the list of independent search queries the LLM
  // expanded from the user's question + shortlist summaries.
  | { kind: "clues"; clues: string[] }
  | { kind: "sources"; sources: Source[] }
  | { kind: "answer_chunk"; delta: string }
  // Single-citation event — emitted 0..N times between the first
  // `answer_chunk` and the terminal `citations`. `index` is 1-based so
  // it matches the inline `[N]` markers in the answer text.
  | { kind: "citation"; citation: Citation; index: number }
  | { kind: "citations"; citations: Citation[] }
  | {
      kind: "done";
      retrieved_count?: number;
      took_ms?: number;
      error?: string;
    };

/**
 * Parse a single Server-Sent Events frame into a typed `AskStreamEvent`.
 *
 * A valid frame is a sequence of `event:` / `data:` / `id:` / comment
 * lines terminated by a blank line. We tolerate LF / CRLF line endings
 * and skip unknown fields. Returns null when the frame is malformed
 * (missing event or unparseable JSON payload) so the caller can drop
 * it rather than crash the stream.
 */
// Exported so the addon's unit tests can assert per-event parsing
// without spinning up the full `askQuestionStream` generator. The
// production consumers only need `askQuestionStream`.
export function parseSseFrame(frame: string): AskStreamEvent | null {
  let eventName = "";
  let dataLine = "";
  const lines = frame.split(/\r?\n/);
  for (const line of lines) {
    if (line.startsWith("event:")) {
      eventName = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      // Spec allows multi-line data; concatenate subsequent data lines
      // with newlines. In practice the backend only emits single-line
      // JSON, but the spec-correct behavior is cheap.
      dataLine = dataLine ? `${dataLine}\n${line.slice(5).trim()}` : line.slice(5).trim();
    }
  }
  if (!eventName) return null;
  let data: Record<string, unknown> = {};
  if (dataLine) {
    try {
      data = JSON.parse(dataLine) as Record<string, unknown>;
    } catch {
      return null;
    }
  }
  switch (eventName) {
    case "keywords":
      return { kind: "keywords", keywords: String(data.keywords ?? "") };
    case "query_decomposed": {
      // Stage A payload — defensive coercion because the backend's
      // schema is enforced server-side but a misconfigured proxy /
      // older addon could still emit something off-spec. Bail out
      // entirely on a missing shape so the UI ignores the event
      // rather than rendering empty fields.
      const tr = data.time_range;
      if (!tr || typeof tr !== "object" || Array.isArray(tr)) return null;
      const trObj = tr as Record<string, unknown>;
      const label = typeof trObj.label === "string" ? trObj.label : "none";
      const after = typeof trObj.after === "string" ? trObj.after : null;
      const before = typeof trObj.before === "string" ? trObj.before : null;
      const personal = data.personal_scope;
      const fileType = data.file_type_hint;
      return {
        kind: "query_decomposed",
        decomposed: {
          time_range: { label, after, before },
          personal_scope:
            personal === "viewed" || personal === "not_viewed"
              ? personal
              : "none",
          file_type_hint:
            fileType === "video" ||
            fileType === "audio" ||
            fileType === "image" ||
            fileType === "text"
              ? fileType
              : "none",
          semantic_query:
            typeof data.semantic_query === "string"
              ? data.semantic_query
              : "",
        },
      };
    }
    case "history_filter": {
      const k = data.kind;
      const matched = data.matched_file_count;
      if (typeof matched !== "number") return null;
      return {
        kind: "history_filter",
        drive: typeof data.drive === "string" ? data.drive : null,
        kind_label: k === "not_viewed" ? "not_viewed" : "viewed",
        matched_file_count: matched,
      };
    }
    case "category_expanded": {
      const raw = data.expanded;
      if (!Array.isArray(raw)) return null;
      const expanded = raw
        .filter((t): t is string => typeof t === "string")
        .map((t) => t.trim())
        .filter((t) => t.length > 0);
      if (expanded.length === 0) return null;
      return {
        kind: "category_expanded",
        semantic_query:
          typeof data.semantic_query === "string"
            ? data.semantic_query
            : "",
        expanded,
      };
    }
    case "clues": {
      // ``clues`` is always an array of strings. Defensive validation
      // because an empty/garbled payload should fall through to the
      // null branch instead of rendering ``[undefined, undefined]``.
      const raw = data.clues;
      if (!Array.isArray(raw)) return null;
      const clues = raw
        .filter((c): c is string => typeof c === "string")
        .map((c) => c.trim())
        .filter((c) => c.length > 0);
      if (clues.length === 0) return null;
      return { kind: "clues", clues };
    }
    case "sources":
      return {
        kind: "sources",
        sources: Array.isArray(data.sources) ? (data.sources as Source[]) : [],
      };
    case "answer_chunk":
      return { kind: "answer_chunk", delta: String(data.delta ?? "") };
    case "citation": {
      // Single citation frame — { citation: <Citation>, index: <int> }.
      // Validate the shape: we are about to interpolate these strings
      // into JSX text nodes and a file-detail href, so keep the cast
      // gated on the fields we actually render.
      const citation = data.citation;
      const index = typeof data.index === "number" ? data.index : NaN;
      if (
        !citation ||
        typeof citation !== "object" ||
        Array.isArray(citation) ||
        typeof (citation as Record<string, unknown>).file_id !== "string" ||
        typeof (citation as Record<string, unknown>).filename !== "string" ||
        !Number.isFinite(index)
      ) {
        return null;
      }
      return {
        kind: "citation",
        citation: citation as Citation,
        index,
      };
    }
    case "citations":
      return {
        kind: "citations",
        citations: Array.isArray(data.citations)
          ? (data.citations as Citation[])
          : [],
      };
    case "done":
      return {
        kind: "done",
        retrieved_count:
          typeof data.retrieved_count === "number" ? data.retrieved_count : undefined,
        took_ms:
          typeof data.took_ms === "number" ? data.took_ms : undefined,
        error: typeof data.error === "string" ? data.error : undefined,
      };
    default:
      return null;
  }
}

/**
 * Ask a question via the intelligence RAG endpoint and stream events.
 *
 * Yields `AskStreamEvent`s as the server emits them. The caller should
 * consume the generator with `for await` and react to each event kind
 * (append `answer_chunk.delta` to the visible answer, render
 * `sources` / `citations` in the sidebar, etc.).
 *
 * Throws on non-2xx responses before the stream starts (feature
 * disabled, LLM off, query too short). Once the stream is open,
 * network errors terminate the iterator cleanly — the consumer sees
 * whichever events arrived before the break. Abort via
 * `options.signal` is propagated to `fetch` and causes an AbortError,
 * which the caller can distinguish from a real failure.
 */
export async function* askQuestionStream(
  query: string,
  drive: string,
  options?: AskOptions,
): AsyncGenerator<AskStreamEvent> {
  const body: Record<string, unknown> = { query };
  if (options?.topK != null) body.top_k = options.topK;
  if (options?.fileType) body.file_type = options.fileType;

  const res = await fetch(`${API_BASE}/addons/intelligence/ask`, {
    method: "POST",
    credentials: "include",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
      ...driveHeaders(drive),
    },
    body: JSON.stringify(body),
    signal: options?.signal,
  });

  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const errBody = (await res.json()) as { detail?: string };
      if (errBody?.detail) detail = errBody.detail;
    } catch {
      // ignore — fall back to status text
    }
    const err = new Error(detail) as Error & { status?: number };
    err.status = res.status;
    throw err;
  }

  const reader = res.body?.getReader();
  if (!reader) {
    // No body on a 2xx SSE response is anomalous but not fatal — emit
    // a synthetic done with no events and let the consumer handle it.
    yield { kind: "done" };
    return;
  }

  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      // SSE frames are delimited by a blank line (\n\n or \r\n\r\n).
      // We split on either to be proxy-tolerant.
      let separator = buffer.indexOf("\n\n");
      while (separator !== -1) {
        const frame = buffer.slice(0, separator);
        buffer = buffer.slice(separator + 2);
        const event = parseSseFrame(frame);
        if (event) yield event;
        separator = buffer.indexOf("\n\n");
      }
    }
    // Flush any trailing non-terminated frame. Rare, but possible if
    // the connection closes without a final blank line.
    if (buffer.trim()) {
      const event = parseSseFrame(buffer);
      if (event) yield event;
    }
  } finally {
    // Release the reader even if the consumer breaks out early so the
    // response body's underlying connection can be reclaimed.
    try {
      reader.releaseLock();
    } catch {
      // ignore
    }
  }
}

// --- Reindex controls (spec 2026-05-24-intelligence-reindex-controls) ---

/**
 * Per-task reindex flags surfaced through ``GET /files/{id}/index-details``.
 *
 * Each flag is True when the corresponding embedding/index slice exists
 * for the file. They drive the per-row "Regenerate" buttons in
 * ``IndexDetailsSection`` — when False the section can offer to kick
 * off the missing task without forcing a full reindex.
 */
export interface IndexDetailsStatus {
  metadata: boolean;
  clip: boolean;
  whisper: boolean;
  text: boolean;
}

export interface IndexDetailsResponse {
  file_id: string;
  drive: string;
  filename: string;
  status: IndexDetailsStatus;
  indexed_at: string;
  // The embeddings + provider_stats payload is opaque to the
  // IndexDetailsSection UI today — declared as ``unknown`` so callers
  // that introspect deeper can narrow it locally without forcing the
  // shared type to grow ahead of the consumer.
  embeddings?: Record<string, unknown>;
  provider_stats?: Record<string, unknown>;
}

export type ReindexTask = "metadata" | "clip" | "whisper" | "text";

export interface ReindexResponse {
  status: "accepted" | "already_queued";
  file_id: string;
  tasks_reset: string[];
}

/**
 * Fetch the per-task indexing status (and provider stats) for a file.
 *
 * Used by ``IndexDetailsSection`` to render the per-task table with
 * Regenerate buttons. 404s and network failures are propagated so the
 * caller can hide the section gracefully.
 */
export async function getIndexDetails(
  fileId: string,
  drive: string,
): Promise<IndexDetailsResponse> {
  return fetchJSON<IndexDetailsResponse>(
    `${API_BASE}/addons/intelligence/files/${fileId}/index-details`,
    { headers: driveHeaders(drive) },
  );
}

/**
 * Reset selected ``*_indexed`` flags for a single file and re-enqueue
 * the corresponding tasks.
 *
 * Backend spec ``2026-05-24-intelligence-reindex-controls.md`` §2.1.
 * The legacy ``POST /queue/reindex`` global reset has been removed; this
 * is the targeted replacement.
 *
 * Returns ``status: "accepted"`` on success or
 * ``status: "already_queued"`` when every requested task was already in
 * the queue (HTTP 202 — no flag flip, no double enqueue).
 *
 * Both callers must pass the owning drive explicitly:
 * ``IndexDetailsSection`` receives it from the file-detail slot, and
 * ``FailedJobsModal`` receives it on each failed-job row. Keeping this
 * required ensures the retry request carries ``X-Lit-Drive``, which the
 * host proxy requires before forwarding drive-scoped addon routes.
 */
export async function reindexFile(
  fileId: string,
  tasks: ReindexTask[] | string[],
  drive: string,
): Promise<ReindexResponse> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  Object.assign(headers, driveHeaders(drive));
  return fetchJSON<ReindexResponse>(
    `${API_BASE}/addons/intelligence/files/${fileId}/reindex`,
    {
      method: "POST",
      headers,
      body: JSON.stringify({ tasks }),
    },
  );
}

/**
 * A single failed-job aggregate row from
 * ``GET /admin/failed-jobs``.
 *
 * Aggregated by (file_id, job_kind, provider). ``attempts`` is the
 * number of consecutive failures since the last success — not lifetime
 * total — so an admin can tell "this just broke" from "this has been
 * failing for hours".
 */
export interface FailedJobItem {
  file_id: string;
  filename: string;
  drive: string;
  job_kind: string;
  provider: string | null;
  // ``error_class`` / ``error_message_excerpt`` are nullable on the
  // backend (``JobRecord`` columns are NULL-able) — be conservative on
  // the UI side so we never render the literal "null" string.
  error_class: string | null;
  error_message_excerpt: string | null;
  attempted_at: string;
  attempts: number;
}

export interface FailedJobsResponse {
  items: FailedJobItem[];
  total: number;
  limit: number;
  offset: number;
}

export interface ResolveFailedJobResponse {
  status: "resolved";
  file_id: string;
  task: ReindexTask;
}

/**
 * Fetch the global failed-jobs queue for the admin dashboard.
 *
 * ``limit`` defaults to 50 (spec §3.2). Drive header is omitted —
 * ``/admin/failed-jobs`` is marked ``drive_optional`` on the proxy.
 */
export async function getFailedJobs(
  limit: number = 50,
  offset: number = 0,
): Promise<FailedJobsResponse> {
  const params = new URLSearchParams({
    limit: String(limit),
    offset: String(offset),
  });
  return fetchJSON<FailedJobsResponse>(
    `${API_BASE}/addons/intelligence/admin/failed-jobs?${params.toString()}`,
  );
}

export async function resolveFailedJob(
  row: Pick<FailedJobItem, "file_id" | "job_kind" | "provider">,
): Promise<ResolveFailedJobResponse> {
  return fetchJSON<ResolveFailedJobResponse>(
    `${API_BASE}/addons/intelligence/admin/failed-jobs/resolve`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(row),
    },
  );
}
