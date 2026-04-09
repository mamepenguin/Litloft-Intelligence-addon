import { fetchJSON } from "@/lib/api";
import type { FileType } from "@/types";

const API_BASE = "/api";

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
}

export interface SemanticSearchResponse {
  available: boolean;
  results: SemanticSearchResult[];
  total: number;
}

export interface SearchServiceStatus {
  available: boolean;
  status?: string;
  indexed?: { total: number; metadata: number; clip: number; whisper: number };
  pending?: { total: number; clip: number; whisper: number };
  queue?: { processing: number; waiting: number; paused: boolean };
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
}

export interface TranscriptResponse {
  available: boolean;
  file_id?: string;
  drive?: string;
  language?: string;
  chunks?: TranscriptChunkItem[];
}

export interface IndexDetailEmbeddingItem {
  content_preview: string;
  start: number | null;
  end: number | null;
}

export interface IndexDetailType {
  count: number;
  items: IndexDetailEmbeddingItem[];
}

export interface IndexDetailsResponse {
  available: boolean;
  file_id?: string;
  drive?: string;
  filename?: string;
  status?: { metadata: boolean; clip: boolean; whisper: boolean; text: boolean };
  indexed_at?: string;
  embeddings?: Record<string, IndexDetailType>;
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
  params?: { limit?: number; type?: FileType; drive?: string }
): Promise<SemanticSearchResponse> {
  const searchParams = new URLSearchParams({ q: query });
  if (params?.limit) searchParams.set("limit", String(params.limit));
  if (params?.type) searchParams.set("type", params.type);
  if (params?.drive) searchParams.set("drive", params.drive);
  try {
    return await fetchJSON<SemanticSearchResponse>(
      `${API_BASE}/addons/intelligence/search?${searchParams.toString()}`
    );
  } catch {
    return { available: false, results: [], total: 0 };
  }
}

export async function searchCompare(
  query: string,
  params?: { limit?: number; type?: FileType }
): Promise<SearchCompareResponse> {
  const searchParams = new URLSearchParams({ q: query });
  if (params?.limit) searchParams.set("limit", String(params.limit));
  if (params?.type) searchParams.set("type", params.type);
  try {
    return await fetchJSON<SearchCompareResponse>(
      `${API_BASE}/addons/intelligence/search/compare?${searchParams.toString()}`
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

export async function getSearchStatus(): Promise<SearchServiceStatus> {
  try {
    return await fetchJSON<SearchServiceStatus>(`${API_BASE}/addons/intelligence/status`);
  } catch {
    return { available: false };
  }
}

export async function getSimilarFiles(
  fileId: string,
  limit: number = 6
): Promise<SimilarFilesResponse> {
  try {
    return await fetchJSON<SimilarFilesResponse>(
      `${API_BASE}/addons/intelligence/similar/${fileId}?limit=${limit}`
    );
  } catch {
    return { available: false, results: [], source_keywords: [] as KeywordScore[] };
  }
}

// Queue control
export async function searchQueuePause(): Promise<void> {
  await fetchJSON(`${API_BASE}/addons/intelligence/queue/pause`, { method: "POST" });
}

export async function searchQueueResume(): Promise<void> {
  await fetchJSON(`${API_BASE}/addons/intelligence/queue/resume`, { method: "POST" });
}

export async function searchQueueReindex(): Promise<void> {
  await fetchJSON(`${API_BASE}/addons/intelligence/queue/reindex`, { method: "POST" });
}

export async function searchQueuePrioritize(fileId: string): Promise<void> {
  await fetchJSON(`${API_BASE}/addons/intelligence/queue/prioritize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ file_id: fileId }),
  });
}

// Search inspection APIs
export async function getFileTranscript(fileId: string): Promise<TranscriptResponse> {
  try {
    return await fetchJSON<TranscriptResponse>(
      `${API_BASE}/addons/intelligence/files/${fileId}/transcript`
    );
  } catch {
    return { available: false };
  }
}

export async function getFileIndexDetails(fileId: string): Promise<IndexDetailsResponse> {
  try {
    return await fetchJSON<IndexDetailsResponse>(
      `${API_BASE}/addons/intelligence/files/${fileId}/index-details`
    );
  } catch {
    return { available: false };
  }
}

export async function getClipTimestamps(fileId: string): Promise<ClipTimestampsResponse> {
  try {
    return await fetchJSON<ClipTimestampsResponse>(
      `${API_BASE}/addons/intelligence/files/${fileId}/clip-timestamps`
    );
  } catch {
    return { available: false };
  }
}

export function getFrameUrl(fileId: string, timestamp: number): string {
  return `${API_BASE}/addons/intelligence/files/${fileId}/frame?t=${timestamp}`;
}
