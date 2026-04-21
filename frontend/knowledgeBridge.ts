/**
 * Thin client for the knowledge addon's endpoints, used by the
 * DetailedSummarySection to promote an edited detailed_summary into a
 * Vault ``.md``. Cross-addon calls go through the host's Generic Addon
 * Proxy so we reuse the same ``/api/addons/knowledge/...`` paths that
 * the knowledge frontend uses.
 *
 * Drive scope is enforced by the proxy via the ``X-HV-Drive`` header.
 * Drive names may contain non-ASCII characters, so we percent-encode
 * the header value — the knowledge backend decodes once.
 */
const KNOWLEDGE_BASE = "/api/addons/knowledge";

function driveHeaders(drive: string): Record<string, string> {
  return { "X-HV-Drive": encodeURIComponent(drive) };
}

export interface KnowledgeVault {
  id: number;
  label: string;
  drive: string;
  path: string;
  is_active: boolean;
  created_at: string;
}

interface VaultListResponse {
  vaults: KnowledgeVault[];
  active_vault_id: number | null;
}

export async function listKnowledgeVaults(
  drive: string,
): Promise<VaultListResponse> {
  const res = await fetch(`${KNOWLEDGE_BASE}/vaults`, {
    credentials: "include",
    headers: driveHeaders(drive),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => null);
    throw new Error(data?.detail ?? `Error: ${res.status}`);
  }
  return res.json();
}

export async function createKnowledgeVault(
  drive: string,
  body: { label: string; path?: string },
): Promise<KnowledgeVault> {
  const res = await fetch(`${KNOWLEDGE_BASE}/vaults`, {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json", ...driveHeaders(drive) },
    body: JSON.stringify({ label: body.label, drive, path: body.path ?? "" }),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => null);
    throw new Error(data?.detail ?? `Error: ${res.status}`);
  }
  return res.json();
}

export interface NoteOrigin {
  note_file_id: string;
  vault_id: number;
  drive: string;
  path: string;
  origin: string | null;
  origin_ref: string | null;
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
  vault_id: number;
  folder: string;
  filename: string;
  title: string;
  content: string;
  origin: string;
  origin_ref?: string | null;
}

export interface DistillResponse {
  note_file_id: string;
  note_path: string;
  vault_id: number;
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
