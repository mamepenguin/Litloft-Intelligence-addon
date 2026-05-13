/**
 * Pure helpers for building a Markdown note from an Ask answer.
 *
 * Spec: `docs/superpowers/specs/2026-05-12-markdown-link-three-forms.md` §3.10
 * Phase E: when a citation is itself a `.md` note, emit a wiki-link
 * `[[basename]]` instead of `[filename](loft://file_id)`. The wiki-link
 * resolver (Phase B, basename rule) maps it back to the right file at
 * render time, and Phase D's rename rewrite keeps it stable through
 * renames. Non-`.md` citations stay on the `loft://` scheme so query
 * parameters (`?t=<sec>` / `?page=<N>`) keep working.
 */
import type { Citation } from "./api";

export interface ParsedSegmentLocation {
  label: string;
  seconds: number | null;
  page: number | null;
  verbatim: string | null;
}

export function parseSegmentLocation(
  loc: string | null,
): ParsedSegmentLocation | null {
  if (!loc) return null;
  const timeMatch = loc.match(/^(\d+):(\d{2})$/);
  if (timeMatch) {
    const m = parseInt(timeMatch[1], 10);
    const s = parseInt(timeMatch[2], 10);
    if (Number.isFinite(m) && Number.isFinite(s)) {
      return { label: loc, seconds: m * 60 + s, page: null, verbatim: null };
    }
  }
  const pageMatch = loc.match(/^page\s+(\d+)$/i);
  if (pageMatch) {
    const p = parseInt(pageMatch[1], 10);
    if (Number.isFinite(p) && p > 0) {
      return { label: loc, seconds: null, page: p, verbatim: null };
    }
  }
  if (/^chunk\s+\d+$/i.test(loc)) {
    return { label: loc, seconds: null, page: null, verbatim: null };
  }
  const verbatim = loc.trim().length >= 12 ? loc.trim() : null;
  return { label: loc, seconds: null, page: null, verbatim };
}

export function citationToLoftUrl(citation: Citation): string {
  const parsed = parseSegmentLocation(
    (citation as Citation & { segment_location?: string | null }).segment_location ?? null,
  );
  const base = `loft://${citation.file_id}`;
  if (parsed?.seconds != null) return `${base}?t=${parsed.seconds}`;
  if (parsed?.page != null) return `${base}?page=${parsed.page}`;
  return base;
}

function isMarkdownCitation(citation: Citation): boolean {
  if (citation.file_type === "markdown") return true;
  const lower = (citation.filename ?? "").toLowerCase();
  return lower.endsWith(".md");
}

function basenameWithoutMd(filename: string): string {
  return filename.replace(/\.md$/i, "");
}

/**
 * Format one citation as a Markdown list item.
 *
 * - `.md` citations → `- [[basename]] — locLabel` (wiki-link form so the
 *   note→note edge participates in the resolver / rename rewrite chain).
 *   Query parameters like `?t=` / `?page=` are dropped because they
 *   don't apply to text notes.
 * - Other citations → `- [filename](loft://id?t=NN) — locLabel` (loft://
 *   scheme keeps timestamp / page jumps working).
 */
export function formatCitationListItem(citation: Citation): string {
  const loc = parseSegmentLocation(
    (citation as Citation & { segment_location?: string | null }).segment_location ?? null,
  );
  const locLabel = loc?.label ? ` — ${loc.label}` : "";
  if (isMarkdownCitation(citation)) {
    const basename = basenameWithoutMd(citation.filename);
    return `- [[${basename}]]${locLabel}`;
  }
  const url = citationToLoftUrl(citation);
  return `- [${citation.filename}](${url})${locLabel}`;
}

export function buildAskNoteMarkdown(
  query: string,
  answer: string,
  citations: Citation[],
): string {
  const savedAt = new Date().toISOString();
  const sourceIds = [...new Set(citations.map((c) => c.file_id))];
  const fmLines = [
    "---",
    `origin: ask_answer`,
    `query: ${JSON.stringify(query)}`,
    `source_file_ids: [${sourceIds.map((id) => JSON.stringify(id)).join(", ")}]`,
    `saved_at: ${savedAt}`,
    "---",
    "",
  ];
  const bodyLines = [`# ${query}`, "", answer.trimEnd(), ""];
  if (citations.length > 0) {
    bodyLines.push("## 引用元", "");
    for (const c of citations) {
      bodyLines.push(formatCitationListItem(c));
      const quote = c.quote?.trim();
      if (quote) {
        for (const line of quote.split("\n")) {
          bodyLines.push(`  > ${line}`);
        }
      }
    }
    bodyLines.push("");
  }
  return fmLines.join("\n") + bodyLines.join("\n");
}

export function queryToFilename(query: string): string {
  const slug = query
    .trim()
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s-]/gu, "")
    .replace(/\s+/g, "-")
    .slice(0, 60)
    .replace(/-+$/, "");
  return `${slug || "ask-note"}.md`;
}
