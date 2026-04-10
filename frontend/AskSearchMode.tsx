"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { useTranslations } from "next-intl";
import {
  AlertCircle,
  Info,
  RefreshCw,
  SearchX,
  Sparkles,
  X,
} from "lucide-react";

import {
  askQuestion,
  getIntelligenceStatus,
  semanticSearch,
} from "./api";
import type { AnswerResponse, Citation } from "./api";

interface AskSearchModeProps {
  query: string;
  drive: string;
  filter: string;
  onSelect: (url: string) => void;
}

type ErrorReason = "disabled" | "generate" | "no-retrieval";

type AskState =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "answered"; data: AnswerResponse }
  | { kind: "error"; reason: ErrorReason; message?: string };

const MIN_QUERY_LENGTH = 3;

// Parse backend segment_location into {label, seconds}.
// Format can be:
//   - "m:ss" (video/audio timestamp) → seconds recoverable
//   - "page 3" (document) → no seconds
//   - null → no segment
function parseSegmentLocation(
  loc: string | null,
): { label: string; seconds: number | null } | null {
  if (!loc) return null;
  const timeMatch = loc.match(/^(\d+):(\d{2})$/);
  if (timeMatch) {
    const m = parseInt(timeMatch[1], 10);
    const s = parseInt(timeMatch[2], 10);
    if (Number.isFinite(m) && Number.isFinite(s)) {
      return { label: loc, seconds: m * 60 + s };
    }
  }
  return { label: loc, seconds: null };
}

function buildCitationUrl(citation: Citation): string {
  const parsed = parseSegmentLocation(citation.segment_location);
  if (parsed?.seconds != null) {
    return `/files/${citation.file_id}?t=${parsed.seconds}`;
  }
  return `/files/${citation.file_id}`;
}

function renderAnswerWithCitations(
  answer: string,
  citations: Citation[],
  onRefClick: (index: number) => void,
): ReactNode[] {
  const parts = answer.split(/(\[\d+\])/g);
  return parts.map((part, i) => {
    const match = part.match(/^\[(\d+)\]$/);
    if (!match) return <span key={i}>{part}</span>;
    const n = parseInt(match[1], 10);
    if (n < 1 || n > citations.length) return <span key={i}>{part}</span>;
    return (
      <button
        key={i}
        type="button"
        onClick={() => onRefClick(n)}
        className="mx-0.5 inline-flex items-center rounded px-1 py-0 align-super text-[10px] font-semibold text-accent bg-accent/10 hover:bg-accent/20 transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent"
        aria-label={`Jump to citation ${n}`}
      >
        {n}
      </button>
    );
  });
}

function CitationCard({
  index,
  citation,
  onSelect,
  ariaLabel,
}: {
  index: number;
  citation: Citation;
  onSelect: (url: string) => void;
  ariaLabel: string;
}) {
  const parsed = parseSegmentLocation(citation.segment_location);
  return (
    <button
      id={`ask-citation-${index}`}
      type="button"
      onClick={() => onSelect(buildCitationUrl(citation))}
      aria-label={ariaLabel}
      className="group flex w-full items-start gap-2 rounded-md border-l-2 border-accent/30 bg-bg-card px-3 py-2 text-left transition-colors hover:bg-bg-elevated focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent"
    >
      <span className="mt-0.5 inline-flex h-4 min-w-[1rem] items-center justify-center rounded px-1 text-[10px] font-semibold text-accent bg-accent/10">
        {index}
      </span>
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-1.5">
          <span className="truncate text-xs font-medium text-text-primary">
            {citation.filename}
          </span>
          {parsed && (
            <span className="flex-shrink-0 rounded px-1 py-0.5 text-[10px] font-medium text-accent">
              {parsed.label}
            </span>
          )}
        </div>
        {citation.quote && (
          <p className="mt-0.5 line-clamp-2 text-[11px] italic text-text-muted">
            “{citation.quote}”
          </p>
        )}
      </div>
    </button>
  );
}

function LoadingShimmer({ label, hint }: { label: string; hint: string }) {
  return (
    <div
      role="status"
      aria-live="polite"
      className="space-y-2 px-4 py-3"
    >
      <div className="flex items-center gap-2">
        <Sparkles size={14} className="text-accent-teal animate-pulse" />
        <span className="text-xs font-medium text-text-muted">{label}</span>
        <span className="text-[10px] text-text-muted/70">{hint}</span>
      </div>
      <div className="h-1 w-full overflow-hidden rounded-full bg-bg-elevated">
        <div className="h-full w-1/3 animate-pulse rounded-full bg-accent/50" />
      </div>
    </div>
  );
}

function ErrorBanner({
  icon,
  message,
  onRetry,
  onClose,
  retryLabel,
  closeLabel,
}: {
  icon: ReactNode;
  message: string;
  onRetry?: () => void;
  onClose: () => void;
  retryLabel?: string;
  closeLabel: string;
}) {
  return (
    <div
      role="alert"
      className="mx-4 my-2 flex items-start gap-2 rounded-md border border-bg-border bg-bg-card px-3 py-2"
    >
      <span className="mt-0.5 flex-shrink-0">{icon}</span>
      <div className="min-w-0 flex-1">
        <p className="text-xs text-text-muted">{message}</p>
        {onRetry && (
          <div className="mt-1.5 flex items-center gap-2">
            <button
              type="button"
              onClick={onRetry}
              className="flex items-center gap-1 rounded px-2 py-1 text-[11px] text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary"
            >
              <RefreshCw size={11} />
              {retryLabel}
            </button>
            <button
              type="button"
              onClick={onClose}
              className="flex items-center gap-1 rounded px-2 py-1 text-[11px] text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary"
            >
              <X size={11} />
              {closeLabel}
            </button>
          </div>
        )}
      </div>
      {!onRetry && (
        <button
          type="button"
          onClick={onClose}
          className="flex-shrink-0 rounded p-1 text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary"
          aria-label={closeLabel}
        >
          <X size={11} />
        </button>
      )}
    </div>
  );
}

export default function AskSearchMode({
  query,
  drive,
  filter,
  onSelect,
}: AskSearchModeProps) {
  const t = useTranslations("askSearch");
  const [ragAvailable, setRagAvailable] = useState<boolean | null>(null);
  const [state, setState] = useState<AskState>({ kind: "idle" });
  const abortRef = useRef<AbortController | null>(null);

  const trimmedQuery = query.trim();
  const queryLongEnough = trimmedQuery.length >= MIN_QUERY_LENGTH;

  // --- Guard 1: probe backend availability once per mount ---
  useEffect(() => {
    const controller = new AbortController();
    let cancelled = false;
    getIntelligenceStatus(controller.signal).then((status) => {
      if (cancelled) return;
      const enabled =
        status?.features?.rag === true && status?.llm?.enabled === true;
      setRagAvailable(enabled);
    });
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, []);

  // --- Reset expanded state when the query changes so stale answers
  //     never bleed across unrelated searches. Also cancels an inflight
  //     request bound to the previous query. ---
  useEffect(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setState({ kind: "idle" });
  }, [trimmedQuery, drive, filter]);

  // --- Unmount cancellation ---
  useEffect(() => {
    return () => {
      abortRef.current?.abort();
      abortRef.current = null;
    };
  }, []);

  // --- Escape closes the panel without tearing down the search modal.
  //     We attach in capture phase so GlobalSearch's own Escape listener
  //     (bubble phase) never fires. ---
  useEffect(() => {
    if (state.kind === "idle") return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      e.stopPropagation();
      abortRef.current?.abort();
      abortRef.current = null;
      setState({ kind: "idle" });
    };
    document.addEventListener("keydown", onKey, { capture: true });
    return () =>
      document.removeEventListener("keydown", onKey, { capture: true });
  }, [state.kind]);

  const runAsk = useCallback(async () => {
    if (!queryLongEnough) return;

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setState({ kind: "loading" });

    try {
      // Pre-flight retrieval check: if semantic search has 0 results,
      // the LLM would receive an empty context and either refuse or
      // hallucinate. Skip the LLM call entirely and surface a dedicated
      // "no retrieval" state. semanticSearch swallows errors and returns
      // `{ available: false, results: [], total: 0 }` — we treat that
      // as 0 hits which is fine: if semantic search is down, asking is
      // pointless too.
      const filterType =
        filter === "all" || !filter
          ? undefined
          : (filter as "video" | "image" | "audio" | "document");
      const probe = await semanticSearch(trimmedQuery, {
        limit: 1,
        type: filterType,
        drive: drive || undefined,
      });
      if (controller.signal.aborted) return;
      if (probe.results.length === 0) {
        setState({ kind: "error", reason: "no-retrieval" });
        return;
      }

      const data = await askQuestion(trimmedQuery, {
        drive: drive || undefined,
        fileType: filterType,
        signal: controller.signal,
      });
      if (controller.signal.aborted) return;
      setState({ kind: "answered", data });
    } catch (err) {
      if (controller.signal.aborted) return;
      if (err instanceof Error && err.name === "AbortError") return;
      const status = (err as { status?: number } | null)?.status;
      if (status === 400) {
        setState({
          kind: "error",
          reason: "disabled",
          message: err instanceof Error ? err.message : undefined,
        });
      } else {
        setState({
          kind: "error",
          reason: "generate",
          message: err instanceof Error ? err.message : undefined,
        });
      }
    }
  }, [queryLongEnough, trimmedQuery, drive, filter]);

  const handleClose = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setState({ kind: "idle" });
  }, []);

  const handleRefClick = useCallback((index: number) => {
    const el = document.getElementById(`ask-citation-${index}`);
    if (el) {
      el.scrollIntoView({ block: "nearest", behavior: "smooth" });
      el.focus({ preventScroll: true });
    }
  }, []);

  const isLoading = state.kind === "loading";

  // Memoize rendered citations nodes so re-renders (e.g. on focus) don't
  // rebuild the button array unnecessarily.
  const answerNodes = useMemo(() => {
    if (state.kind !== "answered" || !state.data.answer) return null;
    return renderAnswerWithCitations(
      state.data.answer,
      state.data.citations,
      handleRefClick,
    );
  }, [state, handleRefClick]);

  // --- Guard 1 + 2: hide entirely when RAG is unavailable or query is too short ---
  if (ragAvailable !== true) return null;
  if (!queryLongEnough) return null;

  // --- Idle: trigger button ---
  if (state.kind === "idle") {
    return (
      <div className="px-4 py-2">
        <button
          type="button"
          onClick={runAsk}
          className="flex w-full items-center gap-2 rounded-md border border-bg-border bg-bg-card px-3 py-2 text-left text-xs text-text-muted transition-colors hover:border-accent/40 hover:bg-bg-elevated hover:text-text-primary focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent"
        >
          <Sparkles size={14} className="flex-shrink-0 text-accent-teal" />
          <span className="truncate">
            {t("button", { query: trimmedQuery })}
          </span>
        </button>
      </div>
    );
  }

  // --- Loading ---
  if (isLoading) {
    return (
      <div className="px-4">
        <LoadingShimmer label={t("loading")} hint={t("loadingHint")} />
      </div>
    );
  }

  // --- Error states ---
  if (state.kind === "error") {
    if (state.reason === "disabled") {
      return (
        <ErrorBanner
          icon={<Info size={14} className="text-text-muted" />}
          message={t("llmDisabled")}
          onClose={handleClose}
          closeLabel={t("close")}
        />
      );
    }
    if (state.reason === "no-retrieval") {
      return (
        <ErrorBanner
          icon={<SearchX size={14} className="text-text-muted" />}
          message={t("noRetrieval")}
          onClose={handleClose}
          closeLabel={t("close")}
        />
      );
    }
    // generate failure → show retry
    return (
      <ErrorBanner
        icon={<AlertCircle size={14} className="text-red-400/70" />}
        message={t("errorGenerate")}
        onRetry={runAsk}
        onClose={handleClose}
        retryLabel={t("retryHint")}
        closeLabel={t("close")}
      />
    );
  }

  // --- Answered ---
  const { data } = state;
  const hasAnswer = data.answer != null && data.answer.length > 0;

  return (
    <div className="px-4 py-2">
      <div className="rounded-md border border-bg-border bg-bg-card">
        <div className="flex items-center gap-2 border-b border-bg-border px-3 py-2">
          <Sparkles size={14} className="flex-shrink-0 text-accent-teal" />
          <h3 className="flex-1 text-xs font-semibold text-text-muted">
            {t("answerTitle")}
          </h3>
          <span className="text-[10px] text-text-muted/70">
            {t("takenMs", { ms: data.took_ms })}
          </span>
          <button
            type="button"
            onClick={handleClose}
            className="flex-shrink-0 rounded p-1 text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary"
            aria-label={t("close")}
          >
            <X size={11} />
          </button>
        </div>

        <div className="px-3 py-2">
          {hasAnswer ? (
            <p className="whitespace-pre-wrap text-sm leading-relaxed text-text-primary">
              {answerNodes}
            </p>
          ) : (
            <p className="flex items-start gap-2 text-xs text-text-muted">
              <Info size={14} className="mt-0.5 flex-shrink-0" />
              <span>{t("noAnswer")}</span>
            </p>
          )}
        </div>

        {data.citations.length > 0 && (
          <div className="border-t border-bg-border px-3 py-2">
            <h4 className="mb-1.5 text-[10px] font-medium uppercase tracking-wider text-text-muted">
              {t("citationsTitle")}
            </h4>
            <div className="space-y-1.5">
              {data.citations.map((citation, idx) => (
                <CitationCard
                  key={`${citation.file_id}-${idx}`}
                  index={idx + 1}
                  citation={citation}
                  onSelect={onSelect}
                  ariaLabel={`${t("citationsTitle")} ${idx + 1}: ${citation.filename}`}
                />
              ))}
            </div>
          </div>
        )}

        <div className="flex items-center gap-2 border-t border-bg-border px-3 py-2">
          <button
            type="button"
            onClick={runAsk}
            className="flex items-center gap-1 rounded px-2 py-1 text-[11px] text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary"
          >
            <RefreshCw size={11} />
            {t("regenerate")}
          </button>
          <button
            type="button"
            onClick={handleClose}
            className="flex items-center gap-1 rounded px-2 py-1 text-[11px] text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary"
          >
            <X size={11} />
            {t("close")}
          </button>
        </div>
      </div>
    </div>
  );
}
