/**
 * Thin client for the knowledge addon's endpoints, used by the
 * DetailedSummarySection to promote an edited detailed_summary into a
 * Knowledge ``.md``. Cross-addon calls go through the host's Generic Addon
 * Proxy so we reuse the same ``/api/addons/knowledge/...`` paths that
 * the knowledge frontend uses.
 *
 * Drive scope is enforced by the proxy via the ``X-Lit-Drive`` header.
 * Drive names may contain non-ASCII characters, so we percent-encode
 * the header value — the knowledge backend decodes once.
 */
const KNOWLEDGE_BASE = "/api/addons/knowledge";

function driveHeaders(drive: string): Record<string, string> {
  return { "X-Lit-Drive": encodeURIComponent(drive) };
}

export interface NoteOrigin {
  note_file_id: string;
  drive: string;
  path: string;
  origin: string | null;
  approved_at: string | null;
  health: string;
}

export async function getNotesBySourceFile(
  drive: string,
  sourceFileId: string,
): Promise<NoteOrigin[]> {
  const res = await fetch(
    `${KNOWLEDGE_BASE}/notes/by_source_file/${encodeURIComponent(sourceFileId)}`,
    { credentials: "include", headers: driveHeaders(drive) },
  );
  if (!res.ok) {
    if (res.status === 404) return [];
    const data = await res.json().catch(() => null);
    throw new Error(data?.detail ?? `Error: ${res.status}`);
  }
  return res.json();
}

export interface DistillRequest {
  source_file_id: string;
  folder: string;
  filename: string;
  title: string;
  content: string;
  origin: string;
}

export interface DistillResponse {
  note_file_id: string;
  note_path: string;
}

export interface NoteCreateRequest {
  folder: string;
  filename: string;
  content: string;
  source_file_ids: string[];
}

export async function saveAskToKnowledge(
  drive: string,
  body: NoteCreateRequest,
): Promise<DistillResponse> {
  const res = await fetch(`${KNOWLEDGE_BASE}/notes`, {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json", ...driveHeaders(drive) },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => null);
    throw new Error(data?.detail ?? `Error: ${res.status}`);
  }
  return res.json();
}

export async function distillToKnowledge(
  drive: string,
  body: DistillRequest,
): Promise<DistillResponse> {
  const res = await fetch(`${KNOWLEDGE_BASE}/distill`, {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json", ...driveHeaders(drive) },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => null);
    throw new Error(data?.detail ?? `Error: ${res.status}`);
  }
  return res.json();
}
