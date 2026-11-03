import { LLMRequestError } from "./api";

const API_BASE = "/api";

/** Where a cited chunk sits inside the open file. `null` means no viewer can move there. */
export type CitationPosition =
  | { seconds: number }
  | { page: number }
  | { section: number; title: string | null }
  | null;

export interface FileCitation {
  n: number;
  position: CitationPosition;
}

export type FileAskEvent =
  | { kind: "answer_chunk"; delta: string }
  | { kind: "citations"; citations: FileCitation[] }
  | { kind: "meta"; partial: boolean }
  | { kind: "done"; error?: string };

export interface FileQuestionOptions {
  fileId: string;
  drive: string;
  question: string;
  profile?: string;
  signal?: AbortSignal;
}

function driveHeaders(drive: string): HeadersInit {
  return { "X-Lit-Drive": encodeURIComponent(drive) };
}

/**
 * Whether the file page should offer the tab at all. Anything but a 200
 * carrying `available: true` — a 404 from the proxy, a transport failure —
 * leaves the tab hidden.
 */
export async function fetchFileAskAvailability(
  fileId: string,
  drive: string,
  signal?: AbortSignal,
): Promise<boolean> {
  try {
    const res = await fetch(
      `${API_BASE}/addons/intelligence/files/${fileId}/ask`,
      { credentials: "include", headers: driveHeaders(drive), signal },
    );
    if (!res.ok) return false;
    const body = (await res.json()) as { available?: unknown };
    return body?.available === true;
  } catch {
    return false;
  }
}

async function readDetail(res: Response): Promise<unknown> {
  try {
    const body = (await res.json()) as { detail?: unknown };
    return body?.detail;
  } catch {
    return undefined;
  }
}

function parsePosition(raw: unknown): CitationPosition {
  if (!raw || typeof raw !== "object") return null;
  const value = raw as Record<string, unknown>;
  if (typeof value.seconds === "number") return { seconds: value.seconds };
  if (typeof value.page === "number") return { page: value.page };
  if (typeof value.section === "number") {
    return {
      section: value.section,
      title: typeof value.title === "string" ? value.title : null,
    };
  }
  return null;
}

function parseFrame(frame: string): FileAskEvent | null {
  let eventName = "";
  let dataLine = "";
  for (const line of frame.split(/\r?\n/)) {
    if (line.startsWith("event:")) {
      eventName = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      const value = line.slice(5).trim();
      dataLine = dataLine ? `${dataLine}\n${value}` : value;
    }
  }
  if (!eventName) return null;

  let payload: unknown = null;
  if (dataLine) {
    try {
      payload = JSON.parse(dataLine);
    } catch {
      return null;
    }
  }
  const data =
    payload && typeof payload === "object" && !Array.isArray(payload)
      ? (payload as Record<string, unknown>)
      : {};

  switch (eventName) {
    case "answer_chunk":
      return {
        kind: "answer_chunk",
        delta: typeof data.delta === "string" ? data.delta : "",
      };
    case "citations": {
      // The route writes this event's payload as the citation list itself,
      // not wrapped in a `citations` key.
      const raw = Array.isArray(payload)
        ? payload
        : Array.isArray(data.citations)
          ? data.citations
          : [];
      const citations: FileCitation[] = [];
      for (const item of raw) {
        if (!item || typeof item !== "object") continue;
        const row = item as Record<string, unknown>;
        if (typeof row.n !== "number") continue;
        citations.push({ n: row.n, position: parsePosition(row.position) });
      }
      return { kind: "citations", citations };
    }
    case "meta":
      return { kind: "meta", partial: data.partial === true };
    case "done":
      return {
        kind: "done",
        error: typeof data.error === "string" ? data.error : undefined,
      };
    default:
      return null;
  }
}

function nextSeparator(
  buffer: string,
): { index: number; length: number } | null {
  const lf = buffer.indexOf("\n\n");
  const crlf = buffer.indexOf("\r\n\r\n");
  if (lf === -1 && crlf === -1) return null;
  if (crlf !== -1 && (lf === -1 || crlf < lf)) return { index: crlf, length: 4 };
  return { index: lf, length: 2 };
}

/**
 * Ask the addon about one file and yield its `answer_chunk`, `citations`,
 * `meta` and `done` frames. A non-2xx answer throws an `LLMRequestError`
 * carrying the detail the tab needs to choose its failure line.
 */
export async function* streamFileQuestion(
  options: FileQuestionOptions,
): AsyncGenerator<FileAskEvent> {
  const { fileId, drive, question, profile, signal } = options;
  const res = await fetch(
    `${API_BASE}/addons/intelligence/files/${fileId}/ask`,
    {
      method: "POST",
      credentials: "include",
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream",
        ...driveHeaders(drive),
      },
      body: JSON.stringify({ question, profile: profile ?? null }),
      signal,
    },
  );

  if (!res.ok) {
    const detail = await readDetail(res);
    throw new LLMRequestError(
      typeof detail === "string" && detail
        ? detail
        : `${res.status} ${res.statusText}`,
      res.status,
      detail,
    );
  }

  const reader = res.body?.getReader();
  if (!reader) return;

  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let separator = nextSeparator(buffer);
      while (separator !== null) {
        const frame = buffer.slice(0, separator.index);
        buffer = buffer.slice(separator.index + separator.length);
        const event = parseFrame(frame);
        if (event) yield event;
        separator = nextSeparator(buffer);
      }
    }
    if (buffer.trim()) {
      const event = parseFrame(buffer);
      if (event) yield event;
    }
  } finally {
    try {
      reader.releaseLock();
    } catch {
      // The stream may already be released by an abort.
    }
  }
}
