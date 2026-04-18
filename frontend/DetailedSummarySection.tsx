"use client";

/**
 * DetailedSummarySection — long-form Markdown summary with two
 * trust-reinforcement layers:
 *
 *   1. Citation markers (🔗 / ⚠) beside each bullet/paragraph that
 *      point back to the source chunks the summary was derived from.
 *      Spec: docs/superpowers/specs/2026-04-18-intelligence-detailed-
 *      summary-citations-edit.md Phase 1.
 *
 *   2. Per-section inline edit with a plain-text textarea, a
 *      "編集済み" badge, and an AI-version revert button. Regenerating
 *      an edited summary surfaces a confirm dialog so the user
 *      explicitly agrees to lose their edits.
 *
 * The component rolls its own lightweight Markdown parser instead of
 * leaning on MarkdownPreview — the parser needs to preserve section
 * and bullet/paragraph boundaries so the citation markers can be
 * anchored to the exact text the backend embedded. MarkdownPreview's
 * sanitize-then-render pipeline would wipe any `data-citation-*`
 * attributes we injected.
 */

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { useTranslations } from "next-intl";
import { MarkdownPreview } from "@/components/MarkdownPreview";
import {
  ChevronDown,
  ChevronRight,
  Download,
  FileText,
  Pencil,
  RefreshCw,
  RotateCcw,
  Save,
  X,
} from "lucide-react";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import { useWebSocket } from "@/hooks/useWebSocket";

import {
  deleteDetailedSummary,
  downloadDetailedSummary,
  editDetailedSummarySection,
  getDetailedSummary,
  getDetailedSummaryCitations,
  regenerateDetailedSummary,
  revertDetailedSummary,
  startDetailedSummary,
} from "./api";
import type {
  CitationChunkExcerpt,
  DetailedSummaryCitation,
  DetailedSummaryResponse,
} from "./api";
import { DetailedSummaryCitationPopover } from "./DetailedSummaryCitationPopover";

interface DetailedSummarySectionProps {
  fileId: string;
  drive: string;
  // Shared with the rest of the file-detail addon slots (passed in by
  // the main frontend in /files/[id]/page.tsx). Used for citation jump
  // on video / audio content. May be absent for non-media file types.
  videoRef?: React.RefObject<HTMLVideoElement | null>;
}

// Detailed summary generation is expensive: tens of seconds to a few
// minutes on local ollama. Poll every 3 seconds up to 10 minutes so
// the UI can surface progress without hammering the backend.
const POLL_INTERVAL_MS = 3000;
const POLL_MAX_ATTEMPTS = 200; // 200 * 3s = 10 minutes

// WS events this section reacts to. Keep them as literals so a typo
// in the backend schema surfaces as a lint-visible mismatch during
// integration.
const WS_SUMMARY_UPDATED = "intelligence.detailed_summary.updated";
const WS_CITATIONS_READY = "intelligence.detailed_summary.citations_ready";

export default function DetailedSummarySection({
  fileId,
  drive,
  videoRef,
}: DetailedSummarySectionProps) {
  const t = useTranslations("file");
  const td = useTranslations("detailedSummary");

  const [data, setData] = useState<DetailedSummaryResponse | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [working, setWorking] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [collapsed, setCollapsed] = useState(true);

  // Phase 1 — citations state. Lazy-loaded once the section body is
  // visible; refetched on the WS `citations_ready` event.
  const [citations, setCitations] = useState<DetailedSummaryCitation[]>([]);

  // Phase 2 — per-section edit state. Only one section is editable at
  // a time; opening another cancels the current draft (parallel edits
  // don't compose with the single-section PUT endpoint anyway).
  const [editingHeading, setEditingHeading] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [saving, setSaving] = useState(false);
  const [reverting, setReverting] = useState(false);
  const [confirmRevertOpen, setConfirmRevertOpen] = useState(false);
  const [confirmRegenerateOpen, setConfirmRegenerateOpen] = useState(false);

  const pollTokenRef = useRef(0);

  const fetchData = useCallback(async () => {
    const result = await getDetailedSummary(fileId, drive);
    setData(result);
    setLoaded(true);
    return result;
  }, [fileId, drive]);

  const fetchCitations = useCallback(async () => {
    const result = await getDetailedSummaryCitations(fileId, drive);
    setCitations(result.citations ?? []);
  }, [fileId, drive]);

  useEffect(() => {
    setData(null);
    setLoaded(false);
    setWorking(false);
    setDownloading(false);
    setCollapsed(true);
    setCitations([]);
    setEditingHeading(null);
    setDraft("");
    setSaving(false);
    setReverting(false);
    setConfirmRevertOpen(false);
    setConfirmRegenerateOpen(false);
    pollTokenRef.current += 1;
    void fetchData().then((result) => {
      if (result.status === "generating") {
        void pollUntilDone();
      } else if (result.available) {
        void fetchCitations();
      }
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fileId, drive]);

  // React to WS events for this file. `useWebSocket` returns the last
  // event globally; we filter on file_id so other files don't perturb
  // this component's state.
  const wsUpdated = useWebSocket(WS_SUMMARY_UPDATED);
  const wsCitations = useWebSocket(WS_CITATIONS_READY);
  useEffect(() => {
    if (!wsUpdated) return;
    if ((wsUpdated.data as { file_id?: string }).file_id !== fileId) return;
    void fetchData();
  }, [wsUpdated, fileId, fetchData]);
  useEffect(() => {
    if (!wsCitations) return;
    if ((wsCitations.data as { file_id?: string }).file_id !== fileId) return;
    void fetchCitations();
  }, [wsCitations, fileId, fetchCitations]);

  const pollUntilDone = useCallback(async () => {
    const token = ++pollTokenRef.current;
    setWorking(true);
    try {
      for (let i = 0; i < POLL_MAX_ATTEMPTS; i++) {
        await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
        if (token !== pollTokenRef.current) return;
        const result = await getDetailedSummary(fileId, drive);
        if (token !== pollTokenRef.current) return;
        setData(result);
        if (
          result.status === "generated"
          || result.status === "failed"
          || !result.status
        ) {
          if (result.available) void fetchCitations();
          return;
        }
      }
    } finally {
      if (token === pollTokenRef.current) setWorking(false);
    }
  }, [fileId, drive, fetchCitations]);

  const doRegenerate = useCallback(async (force: boolean) => {
    setWorking(true);
    setCollapsed(false);
    try {
      if (force) {
        await regenerateDetailedSummary(fileId, drive, { force: true });
      } else {
        // Legacy path: delete+POST keeps the previous behaviour for
        // un-edited summaries where there's nothing to preserve.
        try {
          await deleteDetailedSummary(fileId, drive);
        } catch {
          // No row to delete — proceed with generation.
        }
        await startDetailedSummary(fileId, drive);
      }
      await pollUntilDone();
    } catch {
      await fetchData();
    } finally {
      setWorking(false);
    }
  }, [fileId, drive, pollUntilDone, fetchData]);

  const handleGenerate = useCallback(() => {
    // Entering regenerate from the edited state surfaces a confirm
    // dialog first; once accepted the confirm handler calls
    // doRegenerate(true). Untouched summaries skip the dialog.
    if (data?.edited_at) {
      setConfirmRegenerateOpen(true);
      return;
    }
    void doRegenerate(false);
  }, [data?.edited_at, doRegenerate]);

  const handleConfirmRegenerate = useCallback(() => {
    setConfirmRegenerateOpen(false);
    void doRegenerate(true);
  }, [doRegenerate]);

  const handleDownload = useCallback(async () => {
    setDownloading(true);
    try {
      await downloadDetailedSummary(fileId, drive);
    } catch {
      // silently fail — user can retry
    } finally {
      setDownloading(false);
    }
  }, [fileId, drive]);

  const handleToggleCollapsed = useCallback(() => {
    setCollapsed((prev) => !prev);
  }, []);

  const sections = useMemo(
    () => parseSections(data?.detailed_summary ?? ""),
    [data?.detailed_summary],
  );

  // Index citations by section_path for O(1) lookup during render.
  const citationByPath = useMemo(() => {
    const map = new Map<string, DetailedSummaryCitation>();
    for (const c of citations) {
      map.set(c.section_path, c);
    }
    return map;
  }, [citations]);

  const handleStartEdit = useCallback(
    (sectionName: string) => {
      const section = sections.find((s) => s.heading === sectionName);
      if (!section) return;
      setDraft(section.body);
      setEditingHeading(sectionName);
    },
    [sections],
  );

  const handleCancelEdit = useCallback(() => {
    setEditingHeading(null);
    setDraft("");
  }, []);

  const handleSaveEdit = useCallback(async () => {
    if (!editingHeading) return;
    const trimmed = draft.trim();
    if (!trimmed) return;
    setSaving(true);
    try {
      const updated = await editDetailedSummarySection(fileId, drive, {
        section_heading: editingHeading,
        new_content: trimmed,
      });
      setData(updated);
      setEditingHeading(null);
      setDraft("");
      // Fire-and-forget: citations are recomputed server-side and the
      // WS event will trigger another fetch; in tests / WS-less
      // environments fetch explicitly so the UI still updates.
      void fetchCitations();
    } catch {
      // keep the editor open so the user can retry
    } finally {
      setSaving(false);
    }
  }, [editingHeading, draft, fileId, drive, fetchCitations]);

  const handleRevert = useCallback(async () => {
    setConfirmRevertOpen(false);
    setReverting(true);
    try {
      const updated = await revertDetailedSummary(fileId, drive);
      setData(updated);
      void fetchCitations();
    } catch {
      // silently fail
    } finally {
      setReverting(false);
    }
  }, [fileId, drive, fetchCitations]);

  if (!loaded) return null;

  const reason = data?.reason;
  const status = data?.status;

  if (!data?.available && reason === "unsupported_type") return null;
  if (!data?.available && !status && !reason) return null;

  if (!data?.available && reason === "insufficient_content") {
    return (
      <div className="flex items-center gap-2">
        <FileText size={14} className="text-text-muted/50" />
        <span className="text-xs text-text-muted/70">
          {t("detailedSummaryInsufficientContent", {
            defaultMessage: "Not enough content for a detailed summary",
          })}
        </span>
      </div>
    );
  }

  if (!data?.available && status === "generating") {
    return (
      <div className="flex items-center gap-2">
        <FileText size={14} className="text-text-muted" />
        <span className="flex items-center gap-1 text-xs text-text-muted">
          <RefreshCw size={11} className="animate-spin" />
          {t("detailedSummaryGenerating", {
            defaultMessage: "Generating detailed summary…",
          })}
        </span>
      </div>
    );
  }

  if (!data?.available && status === "failed") {
    return (
      <div>
        <div className="flex items-center gap-2">
          <FileText size={14} className="text-accent-red/70" />
          <span className="text-xs text-accent-red/80">
            {t("detailedSummaryFailed", {
              defaultMessage: "Detailed summary generation failed",
            })}
          </span>
        </div>
        {data?.error && (
          <p className="mt-1 pl-6 text-[11px] text-text-muted/80">
            {data.error}
          </p>
        )}
        <button
          onClick={handleGenerate}
          disabled={working}
          className="mt-2 flex items-center gap-1 rounded px-2 py-1 text-xs text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary disabled:opacity-50"
        >
          <RefreshCw size={11} className={working ? "animate-spin" : ""} />
          {t("detailedSummaryRetry", { defaultMessage: "Retry" })}
        </button>
      </div>
    );
  }

  if (!data?.available) {
    return (
      <div>
        <div className="flex items-center gap-2">
          <FileText size={14} className="text-text-muted" />
          <button
            onClick={handleGenerate}
            disabled={working}
            className="flex items-center gap-1 rounded px-2 py-1 text-xs text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary disabled:opacity-50"
          >
            <RefreshCw size={11} className={working ? "animate-spin" : ""} />
            {working
              ? t("detailedSummaryGenerating", {
                  defaultMessage: "Generating detailed summary…",
                })
              : t("detailedSummaryGenerate", {
                  defaultMessage: "Generate detailed summary",
                })}
          </button>
        </div>
      </div>
    );
  }

  const edited = Boolean(data.edited_at);
  const canRevert = edited && (data.has_original !== false);

  return (
    <div>
      <div className={`flex flex-wrap items-center gap-2 ${collapsed ? "" : "mb-2"}`}>
        <button
          onClick={handleToggleCollapsed}
          aria-expanded={!collapsed}
          aria-label={
            collapsed
              ? t("detailedSummaryShow", { defaultMessage: "Expand" })
              : t("detailedSummaryHide", { defaultMessage: "Collapse" })
          }
          className="flex items-center gap-2 rounded text-text-muted transition-colors hover:text-text-primary"
        >
          {collapsed ? (
            <ChevronRight size={14} className="text-text-muted" />
          ) : (
            <ChevronDown size={14} className="text-text-muted" />
          )}
          <FileText size={14} className="text-accent-teal" />
          <h2 className="text-sm font-semibold">
            {t("detailedSummaryTitle", {
              defaultMessage: "AI Detailed Summary",
            })}
          </h2>
        </button>
        {!collapsed && data.model && !edited && (
          <span className="text-[10px] text-text-muted/50">{data.model}</span>
        )}
        {!collapsed && data.was_truncated && !edited && (
          <span className="text-[10px] text-text-muted/70">
            {t("detailedSummaryTruncatedNote", {
              defaultMessage: "(excerpts from long content)",
            })}
          </span>
        )}
        {edited && (
          <span className="rounded bg-bg-elevated px-1.5 py-0.5 text-[10px] text-text-muted/80">
            {td("edit.badge", { defaultMessage: "Edited" })}
          </span>
        )}
      </div>

      {!collapsed && (
        <>
          <div className="space-y-4">
            {sections.map((section) => (
              <SectionView
                key={section.heading || "__root__"}
                section={section}
                editing={editingHeading === section.heading}
                draft={draft}
                saving={saving}
                onStartEdit={() => handleStartEdit(section.heading)}
                onCancelEdit={handleCancelEdit}
                onSaveEdit={handleSaveEdit}
                onDraftChange={setDraft}
                fileId={fileId}
                drive={drive}
                videoRef={videoRef}
                citationByPath={citationByPath}
                edited={edited}
              />
            ))}
          </div>

          <div className="mt-3 flex flex-wrap items-center gap-2">
            <button
              onClick={handleDownload}
              disabled={downloading}
              className="flex items-center gap-1 rounded px-2 py-1 text-[11px] text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary disabled:opacity-50"
            >
              <Download size={11} />
              {t("detailedSummaryDownload", {
                defaultMessage: "Download as Markdown",
              })}
            </button>
            {canRevert && (
              <button
                onClick={() => setConfirmRevertOpen(true)}
                disabled={reverting}
                className="flex items-center gap-1 rounded px-2 py-1 text-[11px] text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary disabled:opacity-50"
              >
                {reverting ? (
                  <RefreshCw size={11} className="animate-spin" />
                ) : (
                  <RotateCcw size={11} />
                )}
                {td("edit.revertButton", {
                  defaultMessage: "Revert to AI version",
                })}
              </button>
            )}
            <button
              onClick={handleGenerate}
              disabled={working}
              className="flex items-center gap-1 rounded px-2 py-1 text-[11px] text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary disabled:opacity-50"
            >
              <RefreshCw size={11} className={working ? "animate-spin" : ""} />
              {working
                ? t("detailedSummaryGenerating", {
                    defaultMessage: "Generating detailed summary…",
                  })
                : t("detailedSummaryRegenerate", {
                    defaultMessage: "Regenerate",
                  })}
            </button>
          </div>
        </>
      )}

      <ConfirmDialog
        open={confirmRevertOpen}
        title={td("edit.revertButton", {
          defaultMessage: "Revert to AI version",
        })}
        message={td("edit.revertConfirm", {
          defaultMessage:
            "Discard your edits and restore the AI-generated summary?",
        })}
        confirmLabel={td("edit.revertButton", {
          defaultMessage: "Revert to AI version",
        })}
        onConfirm={handleRevert}
        onCancel={() => setConfirmRevertOpen(false)}
      />
      <ConfirmDialog
        open={confirmRegenerateOpen}
        title={t("detailedSummaryRegenerate", { defaultMessage: "Regenerate" })}
        message={td("edit.regenerateConfirm", {
          defaultMessage:
            "Your edits will be lost. Continue and regenerate?",
        })}
        confirmLabel={t("detailedSummaryRegenerate", {
          defaultMessage: "Regenerate",
        })}
        onConfirm={handleConfirmRegenerate}
        onCancel={() => setConfirmRegenerateOpen(false)}
      />
    </div>
  );
}

// ---------------------------------------------------------------------
// Section rendering
// ---------------------------------------------------------------------

interface ParsedSection {
  heading: string; // "" for pre-heading preamble
  body: string;    // raw markdown body for this section
  segments: ParsedSegment[];
}

interface ParsedSegment {
  // section_path follows the backend convention:
  //   "<heading>/<index>" for paragraphs / bullets
  //   "<heading>/row/<index>" for table rows
  section_path: string;
  type: "paragraph" | "bullet" | "table-row";
  // The raw display text. For nested bullets the parent indentation is
  // trimmed so the popover hit-target sits at the text start.
  text: string;
  // Original line or lines (preserved verbatim for re-serialisation
  // when the user edits a different section).
  raw: string;
  // Indent depth in spaces (bullet nesting only, used for CSS padding).
  indent: number;
}

/**
 * Split the Markdown body into `## Heading` sections and parse each
 * section's inline structure into segments the citation layer can
 * anchor to. Mirrors the backend parser in
 * `docs/superpowers/specs/...Phase 1` — keep the two in sync or
 * citations won't align.
 */
export function parseSections(markdown: string): ParsedSection[] {
  if (!markdown) return [];
  const lines = markdown.split(/\r?\n/);
  const sections: ParsedSection[] = [];
  let current: { heading: string; lines: string[] } = {
    heading: "",
    lines: [],
  };

  const flush = () => {
    const body = current.lines.join("\n").replace(/\n+$/, "");
    const segments = parseSegments(current.heading, current.lines);
    // Drop empty preambles entirely (no heading, no body) — they just
    // create phantom sections in the UI.
    if (current.heading || body || segments.length > 0) {
      sections.push({ heading: current.heading, body, segments });
    }
  };

  for (const line of lines) {
    const h = /^##\s+(.+?)\s*$/.exec(line);
    if (h) {
      flush();
      current = { heading: h[1].trim(), lines: [] };
      continue;
    }
    current.lines.push(line);
  }
  flush();
  return sections;
}

function parseSegments(heading: string, lines: string[]): ParsedSegment[] {
  const segments: ParsedSegment[] = [];
  // Shared counter for paragraphs + bullets. Matches the backend's
  // ``plain_idx`` so ``section_path`` stays aligned between the two
  // parsers — otherwise citations land on the wrong DOM element.
  let plainIdx = 0;
  let rowIdx = 0;
  // Table state mirrors summary_parser.py: ``tableOpen`` means we've
  // seen the header row; ``tableHeaderConsumed`` means we've seen the
  // ``|---|---|`` separator. Data rows are emitted only after the
  // header is consumed.
  let tableOpen = false;
  let tableHeaderConsumed = false;

  let paragraphBuf: string[] = [];
  const flushParagraph = () => {
    if (paragraphBuf.length === 0) return;
    const text = paragraphBuf.join(" ").trim();
    if (text) {
      segments.push({
        section_path: `${heading}/${plainIdx++}`,
        type: "paragraph",
        text,
        raw: paragraphBuf.join("\n"),
        indent: 0,
      });
    }
    paragraphBuf = [];
  };

  // Separator row: ``|---|---|`` or ``| :--: |`` etc. Matches backend
  // ``_IS_SEPARATOR_RE``.
  const isSeparator = (line: string) =>
    /^\s*\|[\s\-:|]+\|\s*$/.test(line);

  for (const line of lines) {
    const bulletMatch = /^(\s*)[-*]\s+(.*)$/.exec(line);
    const tableMatch = /^\s*\|.+\|\s*$/.test(line);
    const isBlank = line.trim() === "";

    if (bulletMatch) {
      flushParagraph();
      tableOpen = false;
      tableHeaderConsumed = false;
      const indent = bulletMatch[1].length;
      const text = bulletMatch[2].trim();
      segments.push({
        section_path: `${heading}/${plainIdx++}`,
        type: "bullet",
        text,
        raw: line,
        indent,
      });
      continue;
    }

    if (tableMatch) {
      flushParagraph();
      if (isSeparator(line)) {
        // Separator row: mark header consumed and skip. Subsequent
        // rows are body.
        tableHeaderConsumed = true;
        tableOpen = true;
        continue;
      }
      if (!tableOpen) {
        // First row of a new table — treat as header, skip.
        tableOpen = true;
        tableHeaderConsumed = false;
        continue;
      }
      if (!tableHeaderConsumed) {
        // Malformed table without a separator row — once we've seen
        // the header, any further row is body. Fall through.
        tableHeaderConsumed = true;
      }
      segments.push({
        section_path: `${heading}/row/${rowIdx++}`,
        type: "table-row",
        text: line.replace(/^\s*\|/, "").replace(/\|\s*$/, "").trim(),
        raw: line,
        indent: 0,
      });
      continue;
    }

    if (isBlank) {
      flushParagraph();
      tableOpen = false;
      tableHeaderConsumed = false;
      continue;
    }

    // Plain prose line — accumulate into a paragraph buffer.
    if (!tableOpen) {
      paragraphBuf.push(line.trim());
    }
  }
  flushParagraph();
  return segments;
}

interface SectionViewProps {
  section: ParsedSection;
  editing: boolean;
  draft: string;
  saving: boolean;
  edited: boolean;
  fileId: string;
  drive: string;
  videoRef?: React.RefObject<HTMLVideoElement | null>;
  citationByPath: Map<string, DetailedSummaryCitation>;
  onStartEdit: () => void;
  onCancelEdit: () => void;
  onSaveEdit: () => void;
  onDraftChange: (value: string) => void;
}

function SectionView({
  section,
  editing,
  draft,
  saving,
  fileId,
  drive,
  videoRef,
  citationByPath,
  onStartEdit,
  onCancelEdit,
  onSaveEdit,
  onDraftChange,
}: SectionViewProps) {
  const td = useTranslations("detailedSummary");

  if (!section.heading) {
    // Preamble block before the first `##` — render as-is, no edit
    // button (there's nothing to name the section by on the backend).
    return (
      <div className="text-sm leading-relaxed text-text-muted">
        {section.segments.map((seg) =>
          renderSegmentLine(seg, citationByPath.get(seg.section_path), {
            fileId,
            drive,
            videoRef,
          }),
        )}
      </div>
    );
  }

  return (
    <section data-section-heading={section.heading}>
      <div className="mb-1 flex items-center gap-2">
        <h3 className="text-sm font-semibold text-text-primary">
          {section.heading}
        </h3>
        {!editing && (
          <button
            type="button"
            onClick={onStartEdit}
            aria-label={td("edit.button", { defaultMessage: "Edit" })}
            className="flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary"
          >
            <Pencil size={11} />
            {td("edit.button", { defaultMessage: "Edit" })}
          </button>
        )}
      </div>

      {editing ? (
        <div>
          <textarea
            aria-label={td("edit.textareaLabel", {
              defaultMessage: "Edit section content",
            })}
            value={draft}
            onChange={(e) => onDraftChange(e.target.value)}
            rows={Math.max(4, draft.split("\n").length + 1)}
            className="block w-full resize-y rounded border border-bg-border bg-bg-card px-2 py-2 font-mono text-xs text-text-primary outline-none focus:border-accent-teal"
          />
          <div className="mt-2 flex items-center gap-2">
            <button
              type="button"
              onClick={onSaveEdit}
              disabled={saving || draft.trim().length === 0}
              className="flex items-center gap-1 rounded bg-accent-teal px-2 py-1 text-[11px] text-white transition-opacity hover:opacity-90 disabled:opacity-50"
            >
              {saving ? (
                <RefreshCw size={11} className="animate-spin" />
              ) : (
                <Save size={11} />
              )}
              {td("edit.save", { defaultMessage: "Save" })}
            </button>
            <button
              type="button"
              onClick={onCancelEdit}
              disabled={saving}
              className="flex items-center gap-1 rounded px-2 py-1 text-[11px] text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary disabled:opacity-50"
            >
              <X size={11} />
              {td("edit.cancel", { defaultMessage: "Cancel" })}
            </button>
            {saving && (
              <span className="text-[11px] text-text-muted">
                {td("edit.recomputing", {
                  defaultMessage: "Recomputing citations…",
                })}
              </span>
            )}
          </div>
        </div>
      ) : (
        <div className="text-sm leading-relaxed text-text-muted">
          {section.segments.map((seg) =>
            renderSegmentLine(seg, citationByPath.get(seg.section_path), {
              fileId,
              drive,
              videoRef,
            }),
          )}
        </div>
      )}
    </section>
  );
}

function renderSegmentLine(
  segment: ParsedSegment,
  citation: DetailedSummaryCitation | undefined,
  ctx: {
    fileId: string;
    drive: string;
    videoRef?: React.RefObject<HTMLVideoElement | null>;
  },
): ReactNode {
  const marker = citation ? (
    <DetailedSummaryCitationPopover
      fileId={ctx.fileId}
      drive={ctx.drive}
      citation={citation}
      videoRef={ctx.videoRef ?? null}
    />
  ) : null;

  const text = segment.text;

  if (segment.type === "bullet") {
    const padding = segment.indent > 0 ? segment.indent * 8 : 0;
    return (
      <div
        key={segment.section_path}
        data-citation-section-path={segment.section_path}
        className="flex items-start gap-1"
        style={{ paddingLeft: padding }}
      >
        <span aria-hidden className="mt-[2px] shrink-0 text-text-muted/60">
          •
        </span>
        {marker}
        <div className="min-w-0 flex-1">
          <SegmentMarkdown source={text} />
        </div>
      </div>
    );
  }
  if (segment.type === "table-row") {
    return (
      <div
        key={segment.section_path}
        data-citation-section-path={segment.section_path}
        className="flex items-start gap-1 font-mono text-xs"
      >
        {marker}
        <div className="min-w-0 flex-1 whitespace-pre-wrap break-words">
          {text}
        </div>
      </div>
    );
  }
  // Paragraph + backend-folded ``###``+ heading lines. MarkdownPreview
  // returns block-level HTML (<p>, <h3>, …), so align the marker on
  // its own flex row rather than trying to nest it inside the prose.
  return (
    <div
      key={segment.section_path}
      data-citation-section-path={segment.section_path}
      className="mb-2 flex items-start gap-1"
    >
      {marker}
      <div className="min-w-0 flex-1">
        <SegmentMarkdown source={text} />
      </div>
    </div>
  );
}

// Segment-level Markdown rendering. We delegate to the project's
// shared MarkdownPreview for parity with the rest of the app (sanitize
// pipeline, typography, code/table/quote styles, link hardening, etc.)
// and drop the two chrome layers that only make sense for a whole
// document: the outer card and mermaid rendering. Citation anchoring
// stays stable because segment boundaries are still established by
// our own parser (parseSections / parseSegments) — MarkdownPreview only
// sees one segment's worth of text at a time and its output nests
// cleanly inside the ``data-citation-section-path`` wrapper. See the
// file header for the broader design rationale.
function SegmentMarkdown({ source }: { source: string }) {
  // Match the chrome=true MarkdownPreview look (text-sm + leading-relaxed
  // + text-text-primary — the same classes the full-document pipeline
  // applies) so the detailed summary is visually indistinguishable from
  // the Markdown viewer used elsewhere. ``markdown-segment`` only
  // strips the first/last-child outer margins so adjacent segments
  // stack tightly next to the citation marker.
  return (
    <MarkdownPreview
      source={source}
      chrome={false}
      mermaid={false}
      showFrontmatter={false}
      className="markdown-segment text-sm leading-relaxed text-text-primary"
    />
  );
}

// Re-export the popover excerpt type for consumers that want to wire a
// custom onJump handler without importing from the popover module.
export type { CitationChunkExcerpt };
