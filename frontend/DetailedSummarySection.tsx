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
  Fragment,
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
import { CitationInlinePanel } from "./CitationInlinePanel";
import { CitationRailProvider } from "./CitationRailContext";

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

  // Phase 2 — per-section edit state. Only one target is editable at a
  // time; opening another cancels the current draft (parallel edits
  // don't compose with the single-section PUT endpoint anyway).
  // ``subsectionHeading = null`` means the whole H2 block is the edit
  // range (heading line included). A non-null value narrows the range
  // to one ``### subsection``.
  const [editingTarget, setEditingTarget] = useState<{
    sectionHeading: string;
    subsectionHeading: string | null;
  } | null>(null);
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
    setEditingTarget(null);
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
    (sectionName: string, subsectionName: string | null) => {
      const section = sections.find((s) => s.heading === sectionName);
      if (!section) return;
      if (subsectionName !== null) {
        const sub = section.subsections.find(
          (s) => s.heading === subsectionName,
        );
        if (!sub) return;
        setDraft(sub.fullFragment);
      } else {
        // H2 edit: seed with the full H2 fragment (heading line +
        // body), so the user can rename or restructure everything
        // including nested ``###`` subsections.
        setDraft(section.fullFragment);
      }
      setEditingTarget({
        sectionHeading: sectionName,
        subsectionHeading: subsectionName,
      });
    },
    [sections],
  );

  const handleCancelEdit = useCallback(() => {
    setEditingTarget(null);
    setDraft("");
  }, []);

  const handleSaveEdit = useCallback(async () => {
    if (!editingTarget) return;
    const trimmed = draft.trim();
    if (!trimmed) return;
    setSaving(true);
    try {
      const updated = await editDetailedSummarySection(fileId, drive, {
        section_heading: editingTarget.sectionHeading,
        subsection_heading: editingTarget.subsectionHeading,
        new_content: trimmed,
      });
      setData(updated);
      setEditingTarget(null);
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
  }, [editingTarget, draft, fileId, drive, fetchCitations]);

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
    <CitationRailProvider fileId={fileId} drive={drive}>
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
                editingTarget={
                  editingTarget?.sectionHeading === section.heading
                    ? editingTarget
                    : null
                }
                draft={draft}
                saving={saving}
                onStartEdit={(subsectionName) =>
                  handleStartEdit(section.heading, subsectionName)
                }
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
    </CitationRailProvider>
  );
}

// ---------------------------------------------------------------------
// Section rendering
// ---------------------------------------------------------------------

interface ParsedSection {
  heading: string; // "" for pre-heading preamble
  body: string;    // raw markdown body for this section
  // Full fragment including the ``## heading`` line — used as the
  // draft seed when the user opens the H2-level editor so they can
  // rename the heading or restructure the whole section in one pass.
  fullFragment: string;
  segments: ParsedSegment[];
  // H3 subdivisions, in source order. Empty when the section has no
  // ``###`` lines, in which case only the H2-level edit button shows.
  subsections: ParsedSubsection[];
}

interface ParsedSubsection {
  heading: string;       // H3 title
  // Full fragment including the ``### heading`` line, used as the
  // draft seed when the user opens the H3-level editor.
  fullFragment: string;
}

interface ParsedSegment {
  // section_path follows the backend convention:
  //   "<heading>/<index>" for paragraphs / bullets / code blocks
  //   "<heading>/row/<index>" for table rows
  // The path is always H2-scoped — ``###`` lines do not bump the
  // counter, preserving compatibility with existing citations.
  // ``code-block`` consumes a plain index so it lines up with the
  // backend's merged-paragraph behaviour for fenced blocks surrounded
  // by blank lines (the common case): backend emits one paragraph per
  // flattened fence, so we emit one ``code-block`` per fence and the
  // indices match.
  section_path: string;
  type: "paragraph" | "bullet" | "table-row" | "code-block";
  // The raw display text. For nested bullets the parent indentation is
  // trimmed so the popover hit-target sits at the text start.
  text: string;
  // Original line or lines (preserved verbatim for re-serialisation
  // when the user edits a different section).
  raw: string;
  // Indent depth in spaces (bullet nesting only, used for CSS padding).
  indent: number;
  // H3 subsection this segment falls under, or ``null`` when it sits
  // in the preamble between ``##`` and the first ``###``. Used by the
  // renderer to group segments and interleave H3 edit controls.
  subHeading: string | null;
  // Only set when ``type === 'table-row'``. Cells parsed from the raw
  // row so the renderer can emit a proper ``<table>`` instead of a
  // pipe-laden one-line fallback.
  tableCells?: string[];
  // Only set on the first body row of each consecutive table group —
  // the header cells from the line preceding the ``|---|---|`` separator.
  // Used by ``TableGroup`` to build ``<thead>``; subsequent body rows
  // leave this undefined so we don't duplicate the header.
  tableHeader?: string[];
}

/**
 * Split the Markdown body into `## Heading` sections and parse each
 * section's inline structure into segments the citation layer can
 * anchor to. Mirrors the backend parser in
 * `docs/superpowers/specs/...Phase 1` — keep the two in sync or
 * citations won't align.
 *
 * ``###`` subheadings are captured as edit targets (``subsections``)
 * and attributed to each ``segment.subHeading``, but they do not
 * increment the H2-scoped ``plain_idx`` counter so existing
 * ``section_path`` citations remain stable.
 */
export function parseSections(markdown: string): ParsedSection[] {
  if (!markdown) return [];
  const lines = markdown.split(/\r?\n/);
  const sections: ParsedSection[] = [];
  let current: {
    heading: string;
    headingLine: string;
    lines: string[];
  } = {
    heading: "",
    headingLine: "",
    lines: [],
  };

  const flush = () => {
    const body = current.lines.join("\n").replace(/\n+$/, "");
    const { segments, subsections } = parseH2Body(
      current.heading,
      current.lines,
    );
    // Drop empty preambles entirely (no heading, no body) — they just
    // create phantom sections in the UI.
    if (current.heading || body || segments.length > 0) {
      const fullFragment = current.headingLine
        ? body
          ? `${current.headingLine}\n${body}`
          : current.headingLine
        : body;
      sections.push({
        heading: current.heading,
        body,
        fullFragment,
        segments,
        subsections,
      });
    }
  };

  for (const line of lines) {
    const h = /^##\s+(.+?)\s*$/.exec(line);
    if (h) {
      flush();
      current = {
        heading: h[1].trim(),
        headingLine: line,
        lines: [],
      };
      continue;
    }
    current.lines.push(line);
  }
  flush();
  return sections;
}

function parseH2Body(
  heading: string,
  lines: string[],
): { segments: ParsedSegment[]; subsections: ParsedSubsection[] } {
  const segments: ParsedSegment[] = [];
  const subsections: ParsedSubsection[] = [];
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

  // H3 tracking — capture the raw lines under each ``###`` so the
  // H3-level edit button can seed a draft that preserves the exact
  // user-visible formatting (blank lines, bullet layout, etc.).
  let currentSub: string | null = null;
  let currentSubHeadingLine = "";
  let currentSubBodyLines: string[] = [];

  const flushSubsection = () => {
    if (currentSub === null) return;
    const body = currentSubBodyLines.join("\n").replace(/\n+$/, "");
    const fullFragment = body
      ? `${currentSubHeadingLine}\n${body}`
      : currentSubHeadingLine;
    subsections.push({ heading: currentSub, fullFragment });
    currentSub = null;
    currentSubHeadingLine = "";
    currentSubBodyLines = [];
  };

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
        subHeading: currentSub,
      });
    }
    paragraphBuf = [];
  };

  // Separator row: ``|---|---|`` or ``| :--: |`` etc. Matches backend
  // ``_IS_SEPARATOR_RE``.
  const isSeparator = (line: string) =>
    /^\s*\|[\s\-:|]+\|\s*$/.test(line);

  const parseTableCells = (line: string): string[] => {
    const trimmed = line.trim().replace(/^\|/, "").replace(/\|$/, "");
    return trimmed.split("|").map((cell) => cell.trim());
  };

  // Header cells captured from the line that opened the current table.
  // Attached to the first emitted body row so ``TableGroup`` can build
  // ``<thead>``; subsequent body rows leave it ``undefined``.
  let pendingTableHeader: string[] | null = null;

  // Fenced code block state. While ``inCodeBlock`` is true we short-
  // circuit the normal bullet / table / blank / prose branches and
  // buffer the raw source verbatim so MarkdownPreview can recognise
  // the fence when we emit the segment.
  let inCodeBlock = false;
  let codeBlockBuf: string[] = [];
  let codeBlockFence = "";

  const flushCodeBlock = () => {
    if (codeBlockBuf.length === 0) {
      inCodeBlock = false;
      codeBlockFence = "";
      return;
    }
    const raw = codeBlockBuf.join("\n");
    segments.push({
      section_path: `${heading}/${plainIdx++}`,
      type: "code-block",
      text: raw,
      raw,
      indent: 0,
      subHeading: currentSub,
    });
    codeBlockBuf = [];
    inCodeBlock = false;
    codeBlockFence = "";
  };

  for (const line of lines) {
    const h3Match = /^###\s+(.+?)\s*$/.exec(line);
    if (h3Match) {
      // Unterminated fence before an H3 boundary — emit what we have
      // so the H3 subsection starts cleanly.
      if (inCodeBlock) flushCodeBlock();
      flushParagraph();
      flushSubsection();
      tableOpen = false;
      tableHeaderConsumed = false;
      pendingTableHeader = null;
      currentSub = h3Match[1].trim();
      currentSubHeadingLine = line;
      continue;
    }

    // Attribute every subsequent raw line to the active H3 for later
    // reconstruction of the edit draft. ``slice`` after the heading
    // line guarantees we don't include ``### Heading`` itself twice.
    if (currentSub !== null) {
      currentSubBodyLines.push(line);
    }

    // Fenced code blocks: detected BEFORE bullet / table / blank /
    // prose so their content is preserved verbatim (indentation, blank
    // lines, backticks) instead of being chewed by those branches and
    // flattened with ``join(" ")`` in the paragraph buffer.
    const codeFenceMatch = /^\s*(`{3,}|~{3,})/.exec(line);
    if (inCodeBlock) {
      codeBlockBuf.push(line);
      if (codeFenceMatch && line.trim() === codeBlockFence) {
        flushCodeBlock();
      }
      continue;
    }
    if (codeFenceMatch) {
      flushParagraph();
      tableOpen = false;
      tableHeaderConsumed = false;
      pendingTableHeader = null;
      inCodeBlock = true;
      codeBlockFence = codeFenceMatch[1];
      codeBlockBuf = [line];
      continue;
    }

    const bulletMatch = /^(\s*)[-*]\s+(.*)$/.exec(line);
    const tableMatch = /^\s*\|.+\|\s*$/.test(line);
    const isBlank = line.trim() === "";

    if (bulletMatch) {
      flushParagraph();
      tableOpen = false;
      tableHeaderConsumed = false;
      pendingTableHeader = null;
      const indent = bulletMatch[1].length;
      const text = bulletMatch[2].trim();
      segments.push({
        section_path: `${heading}/${plainIdx++}`,
        type: "bullet",
        text,
        raw: line,
        indent,
        subHeading: currentSub,
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
        // First row of a new table — stash cells as the pending header
        // so the first body row can carry them into the rendered
        // ``<thead>``. Still skipped from ``segments`` to preserve
        // existing ``row/N`` section_path indexing.
        pendingTableHeader = parseTableCells(line);
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
        subHeading: currentSub,
        tableCells: parseTableCells(line),
        tableHeader: pendingTableHeader ?? undefined,
      });
      pendingTableHeader = null;
      continue;
    }

    if (isBlank) {
      flushParagraph();
      tableOpen = false;
      tableHeaderConsumed = false;
      pendingTableHeader = null;
      continue;
    }

    // Plain prose line — accumulate into a paragraph buffer.
    if (!tableOpen) {
      paragraphBuf.push(line.trim());
    }
  }
  // EOF cleanup: unterminated fence still gets emitted so malformed
  // LLM output doesn't silently swallow everything after the open
  // fence. Order matters — fence first (so its plainIdx precedes any
  // trailing paragraph), then paragraph, then subsection.
  if (inCodeBlock) flushCodeBlock();
  flushParagraph();
  flushSubsection();
  return { segments, subsections };
}

interface SectionViewProps {
  section: ParsedSection;
  // Either null (nothing in this section is being edited) or the
  // target descriptor. ``subsectionHeading === null`` means the whole
  // H2 is being edited; otherwise a single ``### subsection``.
  editingTarget: {
    sectionHeading: string;
    subsectionHeading: string | null;
  } | null;
  draft: string;
  saving: boolean;
  edited: boolean;
  fileId: string;
  drive: string;
  videoRef?: React.RefObject<HTMLVideoElement | null>;
  citationByPath: Map<string, DetailedSummaryCitation>;
  onStartEdit: (subsectionHeading: string | null) => void;
  onCancelEdit: () => void;
  onSaveEdit: () => void;
  onDraftChange: (value: string) => void;
}

function SectionView({
  section,
  editingTarget,
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
        {renderSegments(section.segments, citationByPath, {
          fileId,
          drive,
          videoRef,
        })}
      </div>
    );
  }

  // Group segments by their H3 subsection so the renderer can
  // interleave per-H3 edit controls. Segments with ``subHeading: null``
  // form the preamble (content between ``## heading`` and the first
  // ``###``).
  const preambleSegments = section.segments.filter((s) => s.subHeading === null);
  const segmentsBySub = new Map<string, ParsedSegment[]>();
  for (const seg of section.segments) {
    if (seg.subHeading === null) continue;
    const existing = segmentsBySub.get(seg.subHeading) ?? [];
    existing.push(seg);
    segmentsBySub.set(seg.subHeading, existing);
  }

  const editingH2 =
    editingTarget !== null && editingTarget.subsectionHeading === null;

  return (
    <section data-section-heading={section.heading}>
      <div className="markdown-body markdown-segment mb-2 flex items-center gap-2">
        <h2 className="flex-1">{section.heading}</h2>
        {!editingTarget && (
          <button
            type="button"
            onClick={() => onStartEdit(null)}
            aria-label={td("edit.button", { defaultMessage: "Edit" })}
            className="shrink-0 flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary"
          >
            <Pencil size={11} />
            {td("edit.button", { defaultMessage: "Edit" })}
          </button>
        )}
      </div>

      {editingH2 ? (
        <EditTextarea
          draft={draft}
          saving={saving}
          onDraftChange={onDraftChange}
          onSaveEdit={onSaveEdit}
          onCancelEdit={onCancelEdit}
        />
      ) : (
        <div className="text-sm leading-relaxed text-text-muted">
          {renderSegments(preambleSegments, citationByPath, {
            fileId,
            drive,
            videoRef,
          })}
          {section.subsections.map((sub) => {
            const subEditing =
              editingTarget !== null
              && editingTarget.subsectionHeading === sub.heading;
            const subSegments = segmentsBySub.get(sub.heading) ?? [];
            return (
              <div key={sub.heading} data-subsection-heading={sub.heading}>
                <div className="markdown-body markdown-segment mt-3 mb-1 flex items-center gap-2">
                  <h3 className="flex-1">{sub.heading}</h3>
                  {!editingTarget && (
                    <button
                      type="button"
                      onClick={() => onStartEdit(sub.heading)}
                      aria-label={td("edit.button", {
                        defaultMessage: "Edit",
                      })}
                      className="shrink-0 flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary"
                    >
                      <Pencil size={11} />
                      {td("edit.button", { defaultMessage: "Edit" })}
                    </button>
                  )}
                </div>
                {subEditing ? (
                  <EditTextarea
                    draft={draft}
                    saving={saving}
                    onDraftChange={onDraftChange}
                    onSaveEdit={onSaveEdit}
                    onCancelEdit={onCancelEdit}
                  />
                ) : (
                  renderSegments(subSegments, citationByPath, {
                    fileId,
                    drive,
                    videoRef,
                  })
                )}
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}

// Shared textarea + save/cancel toolbar used by both H2 and H3 edits.
// Extracted so the outer SectionView stays legible and the two edit
// sites share identical styling / keyboard behaviour.
function EditTextarea({
  draft,
  saving,
  onDraftChange,
  onSaveEdit,
  onCancelEdit,
}: {
  draft: string;
  saving: boolean;
  onDraftChange: (value: string) => void;
  onSaveEdit: () => void;
  onCancelEdit: () => void;
}) {
  const td = useTranslations("detailedSummary");
  return (
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
  );
}

// Walk a segment list and emit React nodes, folding consecutive
// ``table-row`` segments into a single ``<table>`` and consecutive
// ``bullet`` segments into a single ``<ul>``. Grouping is required so
// the ``.markdown-body`` list-item typography (disc marker, per-li
// spacing, padding-left) applies — a flat ``<div>`` chain of bullets
// misses all of that. Non-grouped segments (paragraphs, code blocks)
// are forwarded to ``renderSegmentLine``.
function renderSegments(
  segments: ParsedSegment[],
  citationByPath: Map<string, DetailedSummaryCitation>,
  ctx: {
    fileId: string;
    drive: string;
    videoRef?: React.RefObject<HTMLVideoElement | null>;
  },
): ReactNode[] {
  const nodes: ReactNode[] = [];
  let tableBuf: ParsedSegment[] = [];
  let bulletBuf: ParsedSegment[] = [];
  let groupCounter = 0;
  const flushTable = () => {
    if (tableBuf.length === 0) return;
    nodes.push(
      <TableGroup
        key={`tbl-${groupCounter++}-${tableBuf[0].section_path}`}
        rows={tableBuf}
        citationByPath={citationByPath}
        ctx={ctx}
      />,
    );
    tableBuf = [];
  };
  const flushBullets = () => {
    if (bulletBuf.length === 0) return;
    nodes.push(
      <BulletGroup
        key={`ul-${groupCounter++}-${bulletBuf[0].section_path}`}
        bullets={bulletBuf}
        citationByPath={citationByPath}
        ctx={ctx}
      />,
    );
    bulletBuf = [];
  };
  for (const seg of segments) {
    if (seg.type === "table-row") {
      flushBullets();
      tableBuf.push(seg);
      continue;
    }
    if (seg.type === "bullet") {
      flushTable();
      bulletBuf.push(seg);
      continue;
    }
    flushTable();
    flushBullets();
    nodes.push(
      renderSegmentLine(seg, citationByPath.get(seg.section_path), ctx),
    );
  }
  flushTable();
  flushBullets();
  return nodes;
}

function BulletGroup({
  bullets,
  citationByPath,
  ctx,
}: {
  bullets: ParsedSegment[];
  citationByPath: Map<string, DetailedSummaryCitation>;
  ctx: {
    fileId: string;
    drive: string;
    videoRef?: React.RefObject<HTMLVideoElement | null>;
  };
}) {
  // The first bullet's indent anchors the group — anything deeper gets
  // proportional extra ``margin-left`` so nested items still read as
  // hierarchy. We avoid re-building a proper ``<ul>`` tree here because
  // the segment list is flat (backend doesn't preserve bullet depth)
  // and a single-level ``<ul>`` with visual indent matches what the
  // MarkdownPreview output looks like for simple nested lists.
  const minIndent = Math.min(...bullets.map((b) => b.indent));
  return (
    // ``mb-[1.15em]`` mirrors ``.markdown-body ul``'s bottom margin so
    // the bullet block has the same rhythm as MarkdownPreview output.
    // The Tailwind arbitrary variants on each ``<li>`` below collapse
    // the inner SegmentMarkdown ``<p>`` into the li line so the marker
    // and text flow next to the disc instead of wrapping to a new row.
    <ul className="markdown-body markdown-segment mb-[1.15em] list-disc pl-[1.75em]">
      {bullets.map((b) => {
        const citation = citationByPath.get(b.section_path);
        const marker = citation ? (
          <DetailedSummaryCitationPopover citation={citation} />
        ) : null;
        const extraPad = (b.indent - minIndent) * 8;
        return (
          <li
            key={b.section_path}
            data-citation-section-path={b.section_path}
            // ``relative`` anchors the absolutely-positioned overlay
            // panel to this <li>. The `[&>div]:inline` selector only
            // targets <div> children — the panel renders as an
            // <aside>, so it stays a block-level overlay instead of
            // being flattened onto the bullet line.
            className="relative [&>div]:inline [&>div>p]:m-0 [&>div>p]:inline"
            style={extraPad ? { marginLeft: `${extraPad}px` } : undefined}
          >
            {marker}
            {marker ? " " : null}
            <SegmentMarkdown source={b.text} />
            {citation && (
              <CitationInlinePanel
                sectionPath={b.section_path}
                videoRef={ctx.videoRef}
              />
            )}
          </li>
        );
      })}
    </ul>
  );
}

function TableGroup({
  rows,
  citationByPath,
  ctx,
}: {
  rows: ParsedSegment[];
  citationByPath: Map<string, DetailedSummaryCitation>;
  ctx: {
    fileId: string;
    drive: string;
    videoRef?: React.RefObject<HTMLVideoElement | null>;
  };
}) {
  const header = rows[0]?.tableHeader ?? [];
  const anyMarker = rows.some((r) => citationByPath.has(r.section_path));
  // The citation column sits outside the `.markdown-body` cell grid so
  // the popover trigger doesn't inherit the cell border / padding /
  // zebra striping. A zero-width column keeps the header alignment
  // honest when only some rows have citations.
  const markerCellStyle: React.CSSProperties = {
    border: "none",
    padding: "0 0.4em 0 0",
    width: 0,
    whiteSpace: "nowrap",
    background: "transparent",
    verticalAlign: "top",
  };
  // Override `.markdown-body tr:nth-child(even) { background: bg-elevated }`
  // inline. The tr-level stripe paints the full row width — including
  // the transparent citation ``<td>`` — so the citation column reads as
  // part of the table. We disable the tr stripe and re-apply it per
  // data cell below, preserving zebra striping while letting the
  // citation column stay visually outside the frame.
  const transparentRowStyle: React.CSSProperties = { background: "transparent" };
  const stripedCellStyle: React.CSSProperties = {
    background: "var(--bg-elevated)",
  };
  // We deliberately do NOT wrap the table in `overflow-x-auto` — that
  // caused the citation popover (which extends beyond the table's right
  // edge on hover) to trigger a horizontal scrollbar on the table alone.
  // Wide tables flow into the section's natural overflow instead.
  return (
    <div className="markdown-body markdown-segment my-2">
      <table>
        {header.length > 0 && (
          <thead>
            <tr style={transparentRowStyle}>
              {anyMarker && <th aria-hidden style={markerCellStyle} />}
              {header.map((cell, i) => (
                <th key={i}>{cell}</th>
              ))}
            </tr>
          </thead>
        )}
        <tbody>
          {rows.map((row, idx) => {
            const citation = citationByPath.get(row.section_path);
            const marker = citation ? (
              <DetailedSummaryCitationPopover citation={citation} />
            ) : null;
            const cells = row.tableCells ?? [];
            // Replicate `.markdown-body tr:nth-child(even)` striping on
            // the data cells only. idx is 0-based; the matching
            // 1-based even rows (2nd, 4th, …) are idx=1, 3, …
            const stripe = idx % 2 === 1;
            // Total column count for the inline-panel colSpan — header
            // + data cells + optional marker column. Tables without a
            // header fall back to the cell count.
            const totalCols =
              (cells.length || header.length || 1) + (anyMarker ? 1 : 0);
            return (
              <Fragment key={row.section_path}>
                <tr
                  data-citation-section-path={row.section_path}
                  style={transparentRowStyle}
                >
                  {anyMarker && (
                    <td aria-hidden style={markerCellStyle}>
                      {marker}
                    </td>
                  )}
                  {cells.map((cell, i) => (
                    <td key={i} style={stripe ? stripedCellStyle : undefined}>
                      {cell}
                    </td>
                  ))}
                </tr>
                {citation && (
                  // Expansion row hosts the inline panel spanning the
                  // full width of the table. Tables use the panel's
                  // in-flow (push-down) mode because an absolutely
                  // positioned child inside a <td> breaks the table
                  // layout grid.
                  <tr style={transparentRowStyle}>
                    <td
                      colSpan={totalCols}
                      style={{ border: "none", padding: 0, background: "transparent" }}
                    >
                      <CitationInlinePanel
                        sectionPath={row.section_path}
                        videoRef={ctx.videoRef}
                        overlay={false}
                      />
                    </td>
                  </tr>
                )}
              </Fragment>
            );
          })}
        </tbody>
      </table>
    </div>
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
    <DetailedSummaryCitationPopover citation={citation} />
  ) : null;

  const text = segment.text;

  // ``bullet`` and ``table-row`` never reach this function — they are
  // folded into ``BulletGroup`` / ``TableGroup`` by ``renderSegments``
  // so ``.markdown-body`` ul/table typography applies. Any other
  // segment type (paragraph, code-block, future additions) renders
  // through SegmentMarkdown below; MarkdownPreview knows how to turn
  // fenced text into ``<pre><code>`` and prose into ``<p>``, so we do
  // not need to branch on ``code-block`` here.
  // ``mb-[1.15em]`` matches ``.markdown-body p``'s bottom margin so the
  // spacing between consecutive paragraphs / code blocks lines up with
  // the MarkdownPreview rhythm. The outer block hosts the text row +
  // the inline citation panel that expands directly beneath it when a
  // citation is active.
  return (
    // ``relative`` anchors the absolutely-positioned overlay panel to
    // this segment so it drops directly beneath the citing line without
    // shifting the surrounding layout when it opens.
    <div
      key={segment.section_path}
      data-citation-section-path={segment.section_path}
      className="relative mb-[1.15em]"
    >
      <div className="flex items-start gap-1">
        {marker}
        <div className="min-w-0 flex-1">
          <SegmentMarkdown source={text} />
        </div>
      </div>
      {citation && (
        <CitationInlinePanel
          sectionPath={segment.section_path}
          videoRef={ctx.videoRef}
        />
      )}
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
