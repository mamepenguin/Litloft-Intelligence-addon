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
  BookmarkPlus,
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
import { useActiveSummary } from "@/hooks/useActiveSummary";
import { useAddonSlots } from "@/components/AddonSlotsProvider";
import { getFile } from "@/lib/api";
import type { MediaController } from "@/lib/mediaController";
import { KnowledgeSaveDialog } from "./KnowledgeSaveDialog";

import {
  downloadDetailedSummary,
  editDetailedSummarySection,
  getDetailedSummary,
  getDetailedSummaryCitations,
  regenerateDetailedSummary,
  revertDetailedSummary,
} from "./api";
import { useOfferFileAiAction } from "./fileAiActions";
import type {
  CitationChunkExcerpt,
  DetailedSummaryCitation,
  DetailedSummaryResponse,
} from "./api";
import { DetailedSummaryCitationPopover } from "./DetailedSummaryCitationPopover";
import { CitationInlinePanel } from "./CitationInlinePanel";
import { InlineMarkdown } from "./InlineMarkdown";
import {
  CitationRailProvider,
  useCitationRail,
  CITATION_STRONG_THRESHOLD,
} from "./CitationRailContext";

interface DetailedSummarySectionProps {
  fileId: string;
  drive: string;
  // Shared with the rest of the file-detail addon slots (passed in by
  // the main frontend in /files/[id]/page.tsx). Used for citation jump
  // on video / audio content. May be absent for non-media file types.
  videoRef?: React.RefObject<HTMLVideoElement | null>;
  mediaController?: MediaController | null;
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
  mediaController,
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
  const [saveDialogOpen, setSaveDialogOpen] = useState(false);
  const [sourceFilename, setSourceFilename] = useState<string>("");

  // Self-hide when a knowledge note is the active summary for this
  // file. `ActiveSummaryHost` renders the knowledge-provided section
  // instead; showing both would stack a user-approved note above the
  // AI draft that no longer represents the file.
  const { data: activeSummary } = useActiveSummary(fileId, drive);
  const hasActiveSummary = activeSummary?.has_active_summary === true;

  // Knowledge availability gate for the "save" button. The addon
  // catalogue is per-drive, so this flips when the user switches
  // drives mid-session.
  const { addons } = useAddonSlots();
  const knowledgeAvailable = Boolean(addons["knowledge"]);

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
      // Single endpoint for every regenerate case (edited / un-edited /
      // first-time generation). The backend superseded-not-deleted
      // history changes mean the prior DELETE+POST fallback would
      // have dropped the history we now want to keep.
      await regenerateDetailedSummary(fileId, drive, { force });
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
    // doRegenerate(true). Also confirm when a knowledge note is
    // currently the active summary — regenerating flips the file
    // detail page back to the AI view, which the user deserves to
    // opt into. Untouched summaries with no active note skip the
    // dialog.
    if (data?.edited_at || hasActiveSummary) {
      setConfirmRegenerateOpen(true);
      return;
    }
    void doRegenerate(false);
  }, [data?.edited_at, hasActiveSummary, doRegenerate]);

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

  const handleOpenSaveDialog = useCallback(async () => {
    // Grab the filename lazily — the component doesn't otherwise need
    // it, and fetching on dialog open keeps the list page cheap.
    if (!sourceFilename) {
      try {
        const file = await getFile(fileId);
        setSourceFilename(file.filename);
      } catch {
        setSourceFilename(fileId);
      }
    }
    setSaveDialogOpen(true);
  }, [fileId, sourceFilename]);

  const handleCloseSaveDialog = useCallback(() => {
    setSaveDialogOpen(false);
  }, []);

  const handleSaveSuccess = useCallback(() => {
    // The WS event `core.file_active_summary.changed` refreshes
    // `useActiveSummary`, which triggers self-hide above. No explicit
    // state mutation needed here.
    setSaveDialogOpen(false);
  }, []);

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

  const reason = data?.reason;
  const status = data?.status;

  // Everything below this line that is not a real state — a run in
  // flight, a failure, too little text — was a heading offering to make
  // something. That offer belongs in the action row's "AI" menu.
  useOfferFileAiAction({
    fileId,
    kind: "detailedSummary",
    labelKey: "detailedSummaryGenerate",
    active: loaded
      && !data?.available
      && reason !== "unsupported_type"
      && reason !== "insufficient_content"
      && status !== "generating"
      && status !== "failed",
    busy: working,
    run: handleGenerate,
  });

  if (!loaded) return null;

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
          <FileText size={14} className="text-danger/70" />
          <span className="text-xs text-danger/80">
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
          className="mt-2 flex items-center gap-1 rounded-lg px-2 py-1 text-xs text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary disabled:opacity-50"
        >
          <RefreshCw size={11} className={working ? "animate-spin" : ""} />
          {t("detailedSummaryRetry", { defaultMessage: "Retry" })}
        </button>
      </div>
    );
  }

  // Never generated: the "AI" menu carries the offer.
  if (!data?.available) return null;

  const edited = Boolean(data.edited_at);
  const canRevert = edited && (data.has_original !== false);

  const canSaveToKnowledge = knowledgeAvailable;

  return (
    <CitationRailProvider fileId={fileId} drive={drive}>
      <DetailedSummaryBody
        data={data}
        collapsed={collapsed}
        onToggleCollapsed={handleToggleCollapsed}
        sections={sections}
        citations={citations}
        citationByPath={citationByPath}
        editingTarget={editingTarget}
        draft={draft}
        saving={saving}
        edited={edited}
        canRevert={canRevert}
        reverting={reverting}
        working={working}
        downloading={downloading}
        canSaveToKnowledge={canSaveToKnowledge}
        hasActiveSummary={hasActiveSummary}
        fileId={fileId}
        drive={drive}
        videoRef={videoRef}
        mediaController={mediaController}
        confirmRevertOpen={confirmRevertOpen}
        confirmRegenerateOpen={confirmRegenerateOpen}
        onStartEdit={handleStartEdit}
        onCancelEdit={handleCancelEdit}
        onSaveEdit={handleSaveEdit}
        onDraftChange={setDraft}
        onDownload={handleDownload}
        onOpenRevert={() => setConfirmRevertOpen(true)}
        onCloseRevert={() => setConfirmRevertOpen(false)}
        onConfirmRevert={handleRevert}
        onGenerate={handleGenerate}
        onCloseRegenerate={() => setConfirmRegenerateOpen(false)}
        onConfirmRegenerate={handleConfirmRegenerate}
        onOpenSave={handleOpenSaveDialog}
      />
      <KnowledgeSaveDialog
        open={saveDialogOpen}
        fileId={fileId}
        drive={drive}
        content={data.detailed_summary ?? ""}
        sourceFilename={sourceFilename || fileId}
        onClose={handleCloseSaveDialog}
        onSaved={handleSaveSuccess}
      />
    </CitationRailProvider>
  );
}

interface DetailedSummaryBodyProps {
  data: DetailedSummaryResponse;
  collapsed: boolean;
  onToggleCollapsed: () => void;
  sections: ParsedSection[];
  citations: DetailedSummaryCitation[];
  citationByPath: Map<string, DetailedSummaryCitation>;
  editingTarget: {
    sectionHeading: string;
    subsectionHeading: string | null;
  } | null;
  draft: string;
  saving: boolean;
  edited: boolean;
  canRevert: boolean;
  reverting: boolean;
  working: boolean;
  downloading: boolean;
  canSaveToKnowledge: boolean;
  hasActiveSummary: boolean;
  fileId: string;
  drive: string;
  videoRef?: React.RefObject<HTMLVideoElement | null>;
  mediaController?: MediaController | null;
  confirmRevertOpen: boolean;
  confirmRegenerateOpen: boolean;
  onStartEdit: (sectionName: string, subsectionName: string | null) => void;
  onCancelEdit: () => void;
  onSaveEdit: () => void;
  onDraftChange: (value: string) => void;
  onDownload: () => void;
  onOpenRevert: () => void;
  onCloseRevert: () => void;
  onConfirmRevert: () => void;
  onGenerate: () => void;
  onCloseRegenerate: () => void;
  onConfirmRegenerate: () => void;
  onOpenSave: () => void;
}

function DetailedSummaryBody({
  data,
  collapsed,
  onToggleCollapsed,
  sections,
  citations,
  citationByPath,
  editingTarget,
  draft,
  saving,
  edited,
  canRevert,
  reverting,
  working,
  downloading,
  canSaveToKnowledge,
  hasActiveSummary,
  fileId,
  drive,
  videoRef,
  mediaController,
  confirmRevertOpen,
  confirmRegenerateOpen,
  onStartEdit,
  onCancelEdit,
  onSaveEdit,
  onDraftChange,
  onDownload,
  onOpenRevert,
  onCloseRevert,
  onConfirmRevert,
  onGenerate,
  onCloseRegenerate,
  onConfirmRegenerate,
  onOpenSave,
}: DetailedSummaryBodyProps) {
  const t = useTranslations("file");
  const td = useTranslations("detailedSummary");
  const {
    verify,
    setVerify,
    expanded,
    collapseAll,
    expandAll,
    expandWeakOnly,
  } = useCitationRail();

  // Weak citations = has_citation && top_score < strong threshold.
  // These drive the "Needs check {n}" badge.
  const weakCitations = useMemo(
    () =>
      citations.filter(
        (c) => c.has_citation && c.top_score < CITATION_STRONG_THRESHOLD,
      ),
    [citations],
  );
  const hasCitations = useMemo(
    () => citations.some((c) => c.has_citation),
    [citations],
  );
  const allCitable = useMemo(
    () => citations.filter((c) => c.has_citation),
    [citations],
  );
  const allExpanded =
    allCitable.length > 0 && expanded.size >= allCitable.length;

  const handleExpandAllToggle = useCallback(() => {
    if (allExpanded) collapseAll();
    else expandAll(allCitable);
  }, [allExpanded, collapseAll, expandAll, allCitable]);

  const handleExpandWeak = useCallback(() => {
    expandWeakOnly(weakCitations);
  }, [expandWeakOnly, weakCitations]);

  const containerRef = useRef<HTMLDivElement | null>(null);

  // Keyboard shortcuts on the whole section. Registered on window so
  // any focused citable segment inside it handles ↑/↓/Enter/j even
  // when focus is transient (focused spans inside `<p>` lose focus
  // when React re-renders). The input-focus guard stops shortcuts
  // from stealing keystrokes inside the textarea / edit input.
  useEffect(() => {
    if (collapsed) return;
    const host = containerRef.current;
    if (!host) return;

    const isTextInput = (el: EventTarget | null) => {
      if (!(el instanceof HTMLElement)) return false;
      const tag = el.tagName;
      if (tag === "TEXTAREA" || tag === "INPUT") return true;
      if (el.isContentEditable) return true;
      return false;
    };

    const handler = (e: KeyboardEvent) => {
      if (isTextInput(e.target)) return;
      if (!host.contains(document.activeElement) && !host.contains(e.target as Node)) {
        // Allow global shortcuts when the body itself has focus too.
        if (e.key !== "v") return;
      }

      if (e.key === "v" || e.key === "V") {
        e.preventDefault();
        setVerify(!verify);
        return;
      }
      if (e.key === "Escape") {
        if (expanded.size > 0) {
          e.preventDefault();
          collapseAll();
        }
        return;
      }
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        const nodes = Array.from(
          host.querySelectorAll<HTMLElement>("[data-citation-section-path]"),
        );
        if (nodes.length === 0) return;
        const active = document.activeElement as HTMLElement | null;
        let index = nodes.findIndex((n) => n === active || n.contains(active));
        if (index < 0) index = 0;
        else index = e.key === "ArrowDown" ? index + 1 : index - 1;
        index = Math.max(0, Math.min(nodes.length - 1, index));
        e.preventDefault();
        nodes[index].focus({ preventScroll: true });
        nodes[index].scrollIntoView({ block: "start", behavior: "smooth" });
        return;
      }
      if (e.key === "Enter") {
        const active = document.activeElement as HTMLElement | null;
        if (!active) return;
        const node = active.closest<HTMLElement>("[data-citation-section-path]");
        if (!node) return;
        const sp = node.getAttribute("data-citation-section-path");
        if (!sp) return;
        const citation = citationByPath.get(sp);
        if (!citation || !citation.has_citation) return;
        e.preventDefault();
        // Route through the rail so the behaviour matches a marker
        // click exactly — toggle + fetch + cache.
        const target = node.querySelector<HTMLButtonElement>(
          `[data-citation-marker]`,
        );
        target?.click();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [
    collapsed,
    verify,
    setVerify,
    expanded,
    collapseAll,
    citationByPath,
  ]);

  // Dormant state: a knowledge note is the active summary. The user-
  // approved `.md` (rendered by the active-summary-view slot) is the
  // single source of truth, so exposing the raw AI body here would
  // either show identical content (just after promotion) or stale
  // content (after the user edited the `.md`). Either way there is
  // nothing useful to display. We keep the regenerate entry point so
  // the user can deliberately switch back to the AI view; everything
  // else (body, edit, revert, download, save) is hidden.
  if (hasActiveSummary) {
    return (
      <div data-testid="detailed-summary-dormant" className="flex flex-wrap items-center gap-2">
        <FileText size={14} className="text-text-muted/60" />
        <h2 className="text-sm font-semibold text-text-muted">
          {t("detailedSummaryTitle", { defaultMessage: "AI Detailed Summary" })}
        </h2>
        <span
          className="rounded-lg bg-bg-elevated px-1.5 py-0.5 text-[10px] text-text-muted/70"
          title={td("dormant.hint", {
            defaultMessage:
              "Knowledge note is the active summary; the AI draft is dormant",
          })}
        >
          {td("dormant.badge", { defaultMessage: "Dormant" })}
        </span>
        <button
          onClick={onGenerate}
          disabled={working}
          className="ml-auto flex items-center gap-1 rounded-lg px-2 py-1 text-[11px] text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary disabled:opacity-50"
        >
          <RefreshCw size={11} className={working ? "animate-spin" : ""} />
          {working
            ? t("detailedSummaryGenerating", {
                defaultMessage: "Generating detailed summary…",
              })
            : t("detailedSummaryRegenerate", { defaultMessage: "Regenerate" })}
        </button>
        <ConfirmDialog
          open={confirmRegenerateOpen}
          title={t("detailedSummaryRegenerate", { defaultMessage: "Regenerate" })}
          message={td("edit.regenerateConfirmWithNote", {
            defaultMessage:
              "Regenerating the AI version. Your saved knowledge note will remain, but the file detail page will switch back to the AI summary. Continue?",
          })}
          confirmLabel={t("detailedSummaryRegenerate", {
            defaultMessage: "Regenerate",
          })}
          onConfirm={onConfirmRegenerate}
          onCancel={onCloseRegenerate}
        />
      </div>
    );
  }

  return (
    <div ref={containerRef} data-citation-host={verify ? "on" : "off"}>
      <div className={`flex flex-wrap items-center gap-2 ${collapsed ? "" : "mb-2"}`}>
        <button
          onClick={onToggleCollapsed}
          aria-expanded={!collapsed}
          aria-label={
            collapsed
              ? t("detailedSummaryShow", { defaultMessage: "Expand" })
              : t("detailedSummaryHide", { defaultMessage: "Collapse" })
          }
          className="flex items-center gap-2 rounded-lg text-text-muted transition-colors hover:text-text-primary"
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
          <span className="rounded-lg bg-bg-elevated px-1.5 py-0.5 text-[10px] text-text-muted/80">
            {td("edit.badge", { defaultMessage: "Edited" })}
          </span>
        )}
      </div>
      {!collapsed && hasCitations && (
        // Dedicated control bar (docs/citation-ui-mockup.html §2
        // `.controls`). Kept on its own row below the title so the
        // Verify toggle and the bulk actions get the breathing room
        // the mockup shows; the title row would otherwise crowd the
        // Verify pill against "AI 詳細要約" on narrow viewports.
        <div className="mb-3 flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-bg-border bg-bg-card px-3.5 py-2.5">
          <div className="inline-flex items-center gap-2.5">
            <button
              type="button"
              onClick={() => setVerify(!verify)}
              aria-pressed={verify}
              className={`inline-flex items-center gap-2 rounded-2xl border px-3.5 py-1.5 text-[13px] font-[650] transition-colors ${
                verify
                  ? "border-accent-teal bg-accent-teal text-white"
                  : "border-bg-border bg-bg-elevated text-text-muted hover:text-text-primary"
              }`}
              data-testid="verify-toggle"
            >
              <span
                aria-hidden
                className={`relative inline-block h-[14px] w-[26px] rounded-full transition-colors after:absolute after:top-[2px] after:left-[2px] after:h-[10px] after:w-[10px] after:rounded-full after:bg-white after:transition-transform ${
                  verify
                    ? "bg-white/35 after:translate-x-[12px]"
                    : "bg-warm-silver after:translate-x-0"
                }`}
              />
              {td("verify.toggle.label", { defaultMessage: "Verify" })}
            </button>
          </div>
          {verify && (
            <div className="inline-flex flex-wrap items-center gap-2">
              <button
                type="button"
                onClick={handleExpandAllToggle}
                className="inline-flex items-center gap-1.5 rounded-2xl border border-bg-border bg-transparent px-3.5 py-1.5 text-xs font-[650] text-text-muted transition-colors hover:border-accent hover:text-text-primary"
                data-testid="verify-expand-all"
              >
                {allExpanded
                  ? td("verify.collapseAll", {
                      defaultMessage: "Collapse all",
                    })
                  : td("verify.expandAll", {
                      defaultMessage: "All expanded",
                    })}
              </button>
              {weakCitations.length > 0 && (
                <button
                  type="button"
                  onClick={handleExpandWeak}
                  className="inline-flex items-center gap-1.5 rounded-2xl border border-dashed px-3.5 py-1.5 text-xs font-bold transition-colors"
                  style={{
                    backgroundColor:
                      "color-mix(in srgb, var(--accent-amber) 12%, transparent)",
                    color: "var(--accent-amber)",
                    borderColor:
                      "color-mix(in srgb, var(--accent-amber) 55%, transparent)",
                  }}
                  data-testid="verify-weak-only"
                >
                  {td("verify.weakOnly.label", {
                    defaultMessage: "Needs check",
                  })}
                  <span
                    className="inline-flex min-w-[18px] items-center justify-center rounded-full px-1.5 text-[11px] font-bold leading-[16px] tabular-nums text-white"
                    style={{ backgroundColor: "var(--accent-amber)" }}
                  >
                    {weakCitations.length}
                  </span>
                </button>
              )}
            </div>
          )}
        </div>
      )}
      {/* Keyboard hint footer — rendered at the bottom of the
          expanded summary body. Placed after the section map below so
          it sits below the content rather than between the header and
          the first segment (where it otherwise competed with the
          keyboard shortcut affordance for attention). */}

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
                  onStartEdit(section.heading, subsectionName)
                }
                onCancelEdit={onCancelEdit}
                onSaveEdit={onSaveEdit}
                onDraftChange={onDraftChange}
                fileId={fileId}
                drive={drive}
                videoRef={videoRef}
        mediaController={mediaController}
                citationByPath={citationByPath}
                edited={edited}
              />
            ))}
          </div>

          <div className="mt-3 flex flex-wrap items-center gap-2">
            <button
              onClick={onDownload}
              disabled={downloading}
              className="flex items-center gap-1 rounded-lg px-2 py-1 text-[11px] text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary disabled:opacity-50"
            >
              <Download size={11} />
              {t("detailedSummaryDownload", {
                defaultMessage: "Download as Markdown",
              })}
            </button>
            {canRevert && (
              <button
                onClick={onOpenRevert}
                disabled={reverting}
                className="flex items-center gap-1 rounded-lg px-2 py-1 text-[11px] text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary disabled:opacity-50"
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
            {canSaveToKnowledge && (
              <button
                onClick={onOpenSave}
                className="flex items-center gap-1 rounded-lg px-2 py-1 text-[11px] text-accent-teal transition-colors hover:bg-bg-elevated hover:text-text-primary"
              >
                <BookmarkPlus size={11} />
                {td("edit.saveToKnowledge", {
                  defaultMessage: "Save as file",
                })}
              </button>
            )}
            <button
              onClick={onGenerate}
              disabled={working}
              className="flex items-center gap-1 rounded-lg px-2 py-1 text-[11px] text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary disabled:opacity-50"
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
          {verify && (
            <KeyboardHintFooter
              text={td("verify.keyboardHints", {
                defaultMessage:
                  "v: ON/OFF | ↑↓: move | Enter: expand | Esc: close",
              })}
            />
          )}
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
        onConfirm={onConfirmRevert}
        onCancel={onCloseRevert}
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
        onConfirm={onConfirmRegenerate}
        onCancel={onCloseRegenerate}
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
  mediaController?: MediaController | null;
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
  mediaController,
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
      <div className="text-base leading-relaxed text-text-muted">
        {renderSegments(section.segments, citationByPath, {
          fileId,
          drive,
          videoRef,
          mediaController,
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
            className="shrink-0 flex items-center gap-1 rounded-lg px-1.5 py-0.5 text-[11px] text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary"
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
        <div className="text-base leading-relaxed text-text-muted">
          {renderSegments(preambleSegments, citationByPath, {
            fileId,
            drive,
            videoRef,
            mediaController,
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
                      className="shrink-0 flex items-center gap-1 rounded-lg px-1.5 py-0.5 text-[11px] text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary"
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
                    mediaController,
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
        spellCheck={false}
        className="block w-full resize-y rounded-lg border border-bg-border bg-bg-card px-3 py-2.5 font-mono text-[13.5px] leading-relaxed text-text-primary outline-none focus:border-focus-ring"
      />
      <div className="mt-2 flex items-center gap-2">
        <button
          type="button"
          onClick={onSaveEdit}
          disabled={saving || draft.trim().length === 0}
          className="flex items-center gap-1 rounded-lg bg-accent-teal px-2 py-1 text-[11px] text-white transition-colors hover:bg-accent-teal/90 disabled:opacity-50"
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
          className="flex items-center gap-1 rounded-lg px-2 py-1 text-[11px] text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary disabled:opacity-50"
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
  mediaController?: MediaController | null;
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
  mediaController?: MediaController | null;
  };
}) {
  // The first bullet's indent anchors the group — anything deeper gets
  // proportional extra ``margin-left`` so nested items still read as
  // hierarchy. We avoid re-building a proper ``<ul>`` tree here because
  // the segment list is flat (backend doesn't preserve bullet depth)
  // and a single-level ``<ul>`` with visual indent matches what the
  // MarkdownPreview output looks like for simple nested lists.
  const { isExpanded } = useCitationRail();
  const minIndent = Math.min(...bullets.map((b) => b.indent));
  return (
    // Bullet block inherits ``.markdown-body ul`` rhythm via the class
    // below. Each ``<li>`` collapses its inner MarkdownPreview ``<p>``
    // to inline so the marker dot sits alongside the text rather than
    // on a new row. Indented bullets receive proportional margin-left
    // so nested hierarchy still reads as hierarchy — even though the
    // backend segment list is flat.
    <ul className="markdown-body markdown-segment list-disc pl-[1.75em]">
      {bullets.map((b) => {
        const citation = citationByPath.get(b.section_path);
        const marker = citation ? (
          <DetailedSummaryCitationPopover citation={citation} />
        ) : null;
        const tier = citation
          ? citation.has_citation
            ? citation.top_score >= 0.9
              ? "strong"
              : "weak"
            : "missing"
          : undefined;
        const canExpand = Boolean(citation && citation.has_citation);
        const extraPad = (b.indent - minIndent) * 8;
        return (
          <li
            key={b.section_path}
            data-citation-section-path={b.section_path}
            data-citation-tier={tier}
            aria-expanded={canExpand ? isExpanded(b.section_path) : undefined}
            tabIndex={canExpand ? 0 : -1}
            className="[&>div]:inline [&>div>p]:m-0 [&>div>p]:inline"
            style={extraPad ? { marginLeft: `${extraPad}px` } : undefined}
          >
            <SegmentMarkdown source={b.text} />
            {marker}
            {citation && (
              <CitationInlinePanel
                sectionPath={b.section_path}
                citation={citation}
                segmentType="bullet"
                videoRef={ctx.videoRef}
          mediaController={ctx.mediaController}
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
  mediaController?: MediaController | null;
  };
}) {
  // Subscribe to expanded set so the accordion row only mounts when
  // the excerpt is actually open — a dormant row lets table-layout
  // peek at the panel's intrinsic width and drift column widths.
  const { isExpanded } = useCitationRail();
  const header = rows[0]?.tableHeader ?? [];
  // Per mockup H3 the dot sits inline at the end of the trailing data
  // cell, not in a separate marker column. The table reads as a quiet
  // editorial grid (thead un-coloured via globals, no zebra, no side
  // borders) and the weak rows carry their tier via a 3px amber
  // left-edge accent on the first data cell.
  return (
    <div className="markdown-body markdown-segment">
      <table>
        {header.length > 0 && (
          <thead>
            <tr>
              {header.map((cell, i) => (
                <th key={i}>
                  <InlineMarkdown source={cell} />
                </th>
              ))}
            </tr>
          </thead>
        )}
        <tbody>
          {rows.map((row) => {
            const citation = citationByPath.get(row.section_path);
            const marker = citation ? (
              <DetailedSummaryCitationPopover citation={citation} />
            ) : null;
            const cells = row.tableCells ?? [];
            // colSpan of the accordion row matches whatever width the
            // data cells occupy. Header count is the authoritative
            // answer when cells were trimmed; otherwise fall back to
            // the cell count, then a minimum of 1 for safety.
            const totalCols = header.length || cells.length || 1;
            const tier = citation
              ? citation.has_citation
                ? citation.top_score >= 0.9
                  ? "strong"
                  : "weak"
                : "missing"
              : undefined;
            const canExpand = Boolean(citation && citation.has_citation);
            const expanded = canExpand && isExpanded(row.section_path);
            const lastIdx = cells.length - 1;
            return (
              <Fragment key={row.section_path}>
                <tr
                  data-citation-section-path={row.section_path}
                  data-citation-tier={tier}
                  aria-expanded={canExpand ? expanded : undefined}
                  tabIndex={canExpand ? 0 : -1}
                >
                  {cells.map((cell, i) => (
                    <td key={i}>
                      <InlineMarkdown source={cell} />
                      {i === lastIdx && marker && (
                        // Non-breaking wrapper keeps the trailing
                        // punctuation / text visually paired with the
                        // dot at line-end — matches the mockup's
                        // ``<span class="segment-endcap">`` idiom.
                        <span style={{ whiteSpace: "nowrap" }}>{marker}</span>
                      )}
                    </td>
                  ))}
                </tr>
                {citation && expanded && (
                  <tr data-citation-acc-row>
                    <td
                      colSpan={totalCols}
                      style={{
                        border: "none",
                        padding: 0,
                        background: "transparent",
                      }}
                    >
                      <CitationInlinePanel
                        sectionPath={row.section_path}
                        citation={citation}
                        segmentType="table"
                        videoRef={ctx.videoRef}
          mediaController={ctx.mediaController}
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
  mediaController?: MediaController | null;
  },
): ReactNode {
  // ``bullet`` and ``table-row`` never reach this function — they are
  // folded into ``BulletGroup`` / ``TableGroup`` by ``renderSegments``
  // so ``.markdown-body`` ul/table typography applies. Any other
  // segment type (paragraph, code-block, future additions) renders
  // through SegmentMarkdown below; MarkdownPreview turns fenced text
  // into ``<pre><code>`` and prose into ``<p>``, so we do not need to
  // branch on ``code-block`` here.
  return (
    <ParagraphSegment
      key={segment.section_path}
      segment={segment}
      citation={citation}
      ctx={ctx}
    />
  );
}

function ParagraphSegment({
  segment,
  citation,
  ctx,
}: {
  segment: ParsedSegment;
  citation: DetailedSummaryCitation | undefined;
  ctx: {
    fileId: string;
    drive: string;
    videoRef?: React.RefObject<HTMLVideoElement | null>;
  mediaController?: MediaController | null;
  };
}) {
  const { isExpanded } = useCitationRail();
  const marker = citation ? (
    <DetailedSummaryCitationPopover citation={citation} />
  ) : null;
  const tier = citation
    ? citation.has_citation
      ? citation.top_score >= 0.9
        ? "strong"
        : "weak"
      : "missing"
    : undefined;
  const canExpand = Boolean(citation && citation.has_citation);
  return (
    <div
      data-citation-section-path={segment.section_path}
      data-citation-tier={tier}
      aria-expanded={canExpand ? isExpanded(segment.section_path) : undefined}
      tabIndex={canExpand ? 0 : -1}
      className="[&>*>div>p]:inline [&>*>div>p]:m-0"
    >
      <SegmentMarkdown source={segment.text} />
      {marker}
      {citation && (
        <CitationInlinePanel
          sectionPath={segment.section_path}
          citation={citation}
          segmentType="paragraph"
          videoRef={ctx.videoRef}
          mediaController={ctx.mediaController}
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
      className="markdown-segment text-base leading-relaxed text-text-primary"
    />
  );
}

// Sticky keyboard hint footer for Verify mode. Renders each shortcut
// token (pipe-separated input) as a `<kbd>` + label pair so the
// shortcut keys stand out without competing with the summary body.
// The sticky positioning keeps the legend visible as the reader
// scrolls through long summaries.
function KeyboardHintFooter({ text }: { text: string }) {
  // Split the i18n string "v: ON/OFF | ↑↓: move | Enter: ..." into
  // (key, label) tuples. Falls back to the plain string when parsing
  // fails so the feature can't break the layout.
  const tokens = text
    .split("|")
    .map((chunk) => chunk.trim())
    .filter(Boolean)
    .map((chunk) => {
      const idx = chunk.indexOf(":");
      if (idx < 0) return { key: chunk, label: "" };
      return {
        key: chunk.slice(0, idx).trim(),
        label: chunk.slice(idx + 1).trim(),
      };
    });

  return (
    <div
      aria-label="Keyboard shortcuts"
      className="sticky bottom-3 z-10 mx-auto mt-8 flex w-fit flex-wrap items-center justify-center gap-3 rounded-2xl border border-bg-border px-4 py-2 text-xs text-text-muted backdrop-blur"
      style={{
        backgroundColor: "color-mix(in srgb, var(--bg-card) 92%, transparent)",
      }}
    >
      {tokens.map((tok, i) => (
        <span key={i} className="inline-flex items-center gap-1.5">
          <kbd
            className="inline-block rounded-lg border border-bg-border bg-bg-elevated px-1.5 py-[1px] font-mono text-[11px] text-text-primary"
            style={{
              boxShadow: "inset 0 -1px 0 rgba(0,0,0,0.08)",
            }}
          >
            {tok.key}
          </kbd>
          {tok.label && <span>{tok.label}</span>}
        </span>
      ))}
    </div>
  );
}

// Re-export the popover excerpt type for consumers that want to wire a
// custom onJump handler without importing from the popover module.
export type { CitationChunkExcerpt };
