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
import type { CitationPosition, FileCitation } from "./askStream";
import { citationLabel } from "./citationLabel";

export interface ParsedSegmentLocation {
  label: string;
  seconds: number | null;
  page: number | null;
  section: number | null;
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
      return { label: loc, seconds: m * 60 + s, page: null, section: null, verbatim: null };
    }
  }
  const pageMatch = loc.match(/^page\s+(\d+)$/i);
  if (pageMatch) {
    const p = parseInt(pageMatch[1], 10);
    if (Number.isFinite(p) && p > 0) {
      return { label: loc, seconds: null, page: p, section: null, verbatim: null };
    }
  }
  const sectionMatch = loc.match(/^section\s+(\d+)$/i);
  if (sectionMatch) {
    const n = parseInt(sectionMatch[1], 10);
    return { label: loc, seconds: null, page: null, section: n > 0 ? n : null, verbatim: null };
  }
  if (/^chunk\s+\d+$/i.test(loc)) {
    return { label: loc, seconds: null, page: null, section: null, verbatim: null };
  }
  const verbatim = loc.trim().length >= 12 ? loc.trim() : null;
  return { label: loc, seconds: null, page: null, section: null, verbatim };
}

function loftUrl(
  fileId: string,
  seconds: number | null,
  page: number | null,
  section: number | null,
): string {
  const base = `loft://${fileId}`;
  if (seconds != null) return `${base}?t=${Math.floor(seconds)}`;
  if (page != null) return `${base}?page=${page}`;
  if (section != null) return `${base}?section=${section}`;
  return base;
}

export function citationToLoftUrl(citation: Citation): string {
  const parsed = parseSegmentLocation(
    (citation as Citation & { segment_location?: string | null }).segment_location ?? null,
  );
  return loftUrl(
    citation.file_id,
    parsed?.seconds ?? null,
    parsed?.page ?? null,
    parsed?.section ?? null,
  );
}

/** The same URL as `citationToLoftUrl`, for a position the answer stream carries. */
export function citationPositionToLoftUrl(
  fileId: string,
  position: CitationPosition,
): string {
  if (!position) return loftUrl(fileId, null, null, null);
  if ("seconds" in position) return loftUrl(fileId, position.seconds, null, null);
  if ("page" in position) return loftUrl(fileId, null, position.page, null);
  return loftUrl(fileId, null, null, position.section);
}

function isMarkdownFile(
  filename: string,
  fileType: string | null | undefined,
): boolean {
  if (fileType === "markdown") return true;
  return filename.toLowerCase().endsWith(".md");
}

/** The two link forms a citation and a saved thread's source line share. */
function citationLink(
  filename: string,
  fileType: string | null | undefined,
  url: string,
): string {
  if (isMarkdownFile(filename, fileType)) {
    return `[[${filename.replace(/\.md$/i, "")}]]`;
  }
  return `[${filename}](${url})`;
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
  const label = (loc?.section != null && citation.section_title) || loc?.label;
  const locLabel = label ? ` — ${label}` : "";
  const link = citationLink(
    citation.filename,
    citation.file_type,
    citationToLoftUrl(citation),
  );
  return `- ${link}${locLabel}`;
}

export function buildAskNoteMarkdown(
  query: string,
  answer: string,
  citations: Citation[],
  scope?: string | null,
): string {
  const savedAt = new Date().toISOString();
  const sourceIds = [...new Set(citations.map((c) => c.file_id))];
  const trimmedScope = scope?.trim() ?? "";
  const fmLines = [
    "---",
    `origin: ask_answer`,
    `query: ${JSON.stringify(query)}`,
    ...(trimmedScope ? [`scope: ${JSON.stringify(trimmedScope)}`] : []),
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

export interface AskThreadNoteTurn {
  question: string;
  answer: string;
  citations: FileCitation[];
}

function collapseWhitespace(text: string): string {
  return text.replace(/\s+/g, " ");
}

function escapeLinkLabel(label: string): string {
  return label.replace(/\\/g, "\\\\").replace(/\[/g, "\\[").replace(/\]/g, "\\]");
}

const MARKER = /\[(\d{1,3})\]/g;

function rewriteMarkers(
  text: string,
  positions: Map<number, CitationPosition>,
  fileId: string,
): string {
  return text.replace(MARKER, (_match, digits: string) => {
    const n = Number(digits);
    if (!positions.has(n)) return "";
    const position = positions.get(n) ?? null;
    const label = citationLabel(position);
    if (label === null) return "";
    const url = citationPositionToLoftUrl(fileId, position);
    return `[${escapeLinkLabel(label)}](${url})`;
  });
}

function closingBacktickRun(
  line: string,
  from: number,
  length: number,
): number {
  let index = line.indexOf("`", from);
  while (index >= 0) {
    let run = 1;
    while (line[index + run] === "`") run += 1;
    if (run === length) return index;
    index = line.indexOf("`", index + run);
  }
  return -1;
}

function rewriteInline(
  line: string,
  positions: Map<number, CitationPosition>,
  fileId: string,
): string {
  let out = "";
  let i = 0;
  while (i < line.length) {
    if (line[i] === "`") {
      let run = 1;
      while (line[i + run] === "`") run += 1;
      const close = closingBacktickRun(line, i + run, run);
      if (close >= 0) {
        out += line.slice(i, close + run);
        i = close + run;
      } else {
        out += line.slice(i, i + run);
        i += run;
      }
      continue;
    }
    let next = line.indexOf("`", i);
    if (next < 0) next = line.length;
    out += rewriteMarkers(line.slice(i, next), positions, fileId);
    i = next;
  }
  return out;
}

function rewriteAnswer(
  answer: string,
  citations: FileCitation[],
  fileId: string,
): string {
  const positions = new Map(citations.map((citation) => [citation.n, citation.position]));
  const lines = answer.split("\n");
  const out: string[] = [];
  let fence: { char: string; length: number } | null = null;
  let indented = false;
  let previousBlank = true;

  for (const line of lines) {
    if (fence !== null) {
      out.push(line);
      const closing = /^ {0,3}(`{3,}|~{3,})\s*$/.exec(line);
      if (
        closing &&
        closing[1][0] === fence.char &&
        closing[1].length >= fence.length
      ) {
        fence = null;
      }
      previousBlank = line.trim() === "";
      continue;
    }

    const opening = /^ {0,3}(`{3,}|~{3,})/.exec(line);
    if (opening) {
      fence = { char: opening[1][0], length: opening[1].length };
      indented = false;
      out.push(line);
      previousBlank = false;
      continue;
    }

    const lineIsIndented = /^(?: {4}|\t)/.test(line);
    if (lineIsIndented && (indented || previousBlank)) {
      indented = true;
      out.push(line);
      previousBlank = false;
      continue;
    }
    if (indented && line.trim() === "") {
      out.push(line);
      previousBlank = true;
      continue;
    }

    indented = false;
    out.push(rewriteInline(line, positions, fileId));
    previousBlank = line.trim() === "";
  }

  return out.join("\n");
}

/**
 * The note a file's Ask thread saves: every answered turn in thread order,
 * with the markers outside code pointing at this file's cited positions.
 */
export function buildAskThreadMarkdown(options: {
  fileId: string;
  filename: string;
  fileType?: string | null;
  turns: AskThreadNoteTurn[];
  savedAt?: string;
}): string {
  const { fileId, filename, fileType, turns } = options;
  const savedAt = options.savedAt ?? new Date().toISOString();
  const firstQuestion = turns[0]?.question ?? "";
  const sourceLink = citationLink(
    filename,
    fileType,
    citationPositionToLoftUrl(fileId, null),
  );

  const lines = [
    "---",
    "origin: ask_answer",
    `query: ${JSON.stringify(firstQuestion)}`,
    `source_file_ids: [${JSON.stringify(fileId)}]`,
    `saved_at: ${savedAt}`,
    "---",
    "",
    `# ${collapseWhitespace(firstQuestion)}`,
    "",
    sourceLink,
    "",
  ];

  for (const turn of turns) {
    lines.push(
      `## ${collapseWhitespace(turn.question)}`,
      "",
      rewriteAnswer(turn.answer, turn.citations, fileId),
      "",
    );
  }

  return lines.join("\n");
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
