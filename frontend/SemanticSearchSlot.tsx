"use client";

import { useEffect, useRef, useState } from "react";
import { useTranslations } from "next-intl";
import { Sparkles } from "lucide-react";

import { semanticSearch, getSearchStatus } from "./api";
import type { SemanticSearchResult, SemanticSearchSegment } from "./api";
import { formatDuration } from "@/lib/format";
import { FileCard } from "@/components/FileCard";
import { FileContextMenu } from "@/components/FileContextMenu";
import { useContextMenu } from "@/hooks/useContextMenu";
import type { FileItem } from "@/types";
import { MatchOverlay } from "./MatchOverlay";

type SlotContext = "popup" | "page";

interface SemanticSearchSlotProps {
  query: string;
  drive: string;
  filter: string;
  onSelect: (url: string) => void;
  /** Layout mode. "popup" (default) = compact list for the search modal,
   *  "page" = full-page grid for /drive/<name>/search. */
  context?: SlotContext;
}

// Popup-only badge palette. The page layout uses MatchOverlay (warm
// palette per DESIGN.md §2.2). The popup view sits inside the search
// modal which has its own visual context, so we keep the existing
// compact badge style there to limit Phase 2 scope.
const POPUP_MATCH_TYPE_STYLES: Record<string, string> = {
  transcript: "bg-blue-500/15 text-blue-400",
  transcript_keyword: "bg-cyan-500/15 text-cyan-400",
  clip: "bg-emerald-500/15 text-emerald-400",
  metadata: "bg-zinc-500/15 text-zinc-400",
  content: "bg-purple-500/15 text-purple-400",
  text_content_keyword: "bg-violet-500/15 text-violet-400",
};

function PopupMatchBadge({ type, label }: { type: string; label: string }) {
  const style =
    POPUP_MATCH_TYPE_STYLES[type] ?? "bg-zinc-500/15 text-zinc-400";
  return (
    <span
      className={`inline-flex rounded px-1.5 py-0.5 text-[10px] font-medium ${style}`}
    >
      {label}
    </span>
  );
}

function PopupTimestampLink({
  seconds,
  fileId,
  onClick,
}: {
  seconds: number;
  fileId: string;
  onClick: (url: string) => void;
}) {
  return (
    <button
      onClick={(e) => {
        e.stopPropagation();
        onClick(`/files/${fileId}?t=${Math.floor(seconds)}`);
      }}
      className="rounded px-1 py-0.5 text-[10px] font-medium text-accent transition-colors hover:bg-accent/10"
    >
      {formatDuration(seconds)}
    </button>
  );
}

function SemanticResultItem({
  result,
  onSelect,
  t,
}: {
  result: SemanticSearchResult;
  onSelect: (url: string) => void;
  t: (key: string, values?: Record<string, string | number>) => string;
}) {
  const matchLabels: Record<string, string> = {
    transcript: t("matchTranscript"),
    transcript_keyword: t("matchTranscriptKeyword"),
    clip: t("matchClip"),
    metadata: t("matchMetadata"),
    content: t("matchContent"),
    text_content_keyword: t("matchTextContentKeyword"),
  };

  const matchedPages = [
    ...new Set(
      result.segments
        .flatMap((seg) => seg.matches)
        .filter((m) => m.page != null)
        .map((m) => m.page as number),
    ),
  ].sort((a, b) => a - b);

  const timestamps = result.segments
    .filter(
      (seg): seg is SemanticSearchSegment & { time_range: [number, number] } =>
        seg.time_range != null && seg.time_range[0] > 0,
    )
    .slice(0, 5);

  return (
    <button
      onClick={() => onSelect(`/files/${result.file_id}`)}
      className="flex w-full items-start gap-3 px-4 py-2.5 text-left transition-colors hover:bg-bg-elevated"
    >
      <img
        src={`/api/files/${result.file_id}/thumbnail`}
        alt=""
        className="h-10 w-16 flex-shrink-0 rounded bg-bg-elevated object-cover"
        onError={(e) => {
          e.currentTarget.style.display = "none";
        }}
      />
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm text-text-primary">{result.filename}</p>
        <div className="mt-0.5 flex flex-wrap items-center gap-1">
          {result.match_types.map((type) => (
            <PopupMatchBadge
              key={type}
              type={type}
              label={matchLabels[type] ?? type}
            />
          ))}
        </div>
        {timestamps.length > 0 && (
          <div className="mt-1 flex flex-wrap gap-0.5">
            {timestamps.map((seg) => (
              <PopupTimestampLink
                key={seg.time_range[0]}
                seconds={seg.time_range[0]}
                fileId={result.file_id}
                onClick={onSelect}
              />
            ))}
          </div>
        )}
        {matchedPages.length > 0 && (
          <p className="mt-1 text-[11px] text-text-muted">
            {t("matchedPages", { pages: matchedPages.join(", ") })}
          </p>
        )}
      </div>
    </button>
  );
}

/**
 * Build a minimal FileItem from a SemanticSearchResult when core's
 * bulk hydrate failed (``result.file === null``). The card stays
 * functional with reduced fidelity — favorite toggle and tag display
 * are unavailable, but title, thumbnail, and click-through still work.
 *
 * Should be rare: only fires when the core service is briefly
 * unreachable mid-search. The IndexedFile snapshot guarantees at least
 * filename / file_type / drive are present on every search hit.
 */
function fileItemFromSnapshot(result: SemanticSearchResult): FileItem {
  return {
    id: result.file_id,
    filename: result.filename,
    title: result.filename,
    description: "",
    drive: result.drive,
    folder_path: "",
    file_type: result.file_type as FileItem["file_type"],
    mime_type: "",
    thumbnail_url: `/api/files/${result.file_id}/thumbnail`,
    has_thumbnail: true,
    file_size: 0,
    duration: null,
    likes: 0,
    is_favorite: false,
    tags: [],
    subtitles: [],
    deleted_at: null,
    missing_since: null,
    created_at: "",
    updated_at: "",
  };
}

function PopupLayout({
  results,
  loading,
  query,
  drive,
  onSelect,
  t,
  askT,
}: {
  results: SemanticSearchResult[];
  loading: boolean;
  query: string;
  drive: string;
  onSelect: (url: string) => void;
  t: (key: string, values?: Record<string, string | number>) => string;
  askT: (key: string, values?: Record<string, string | number>) => string;
}) {
  const askHref = `/drive/${encodeURIComponent(drive)}/addons/intelligence?q=${encodeURIComponent(query)}`;

  return (
    <>
      {results.map((result) => (
        <SemanticResultItem
          key={result.file_id}
          result={result}
          onSelect={onSelect}
          t={t}
        />
      ))}
      {loading && (
        <div className="flex items-center justify-center border-t border-bg-border py-3">
          <div className="h-4 w-4 animate-spin rounded-full border-2 border-accent border-t-transparent" />
        </div>
      )}
      <button
        type="button"
        onClick={() => onSelect(askHref)}
        className="flex w-full items-center gap-2 border-t border-bg-border px-4 py-2.5 text-left text-xs text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary"
      >
        <Sparkles size={12} className="flex-shrink-0 text-accent-teal" />
        <span className="truncate">{askT("button", { query })}</span>
      </button>
    </>
  );
}

function PageLayout({
  results,
  loading,
  query,
  drive,
  onSelect,
  onResultsChange,
  t,
  askT,
}: {
  results: SemanticSearchResult[];
  loading: boolean;
  query: string;
  drive: string;
  onSelect: (url: string) => void;
  onResultsChange: (next: SemanticSearchResult[]) => void;
  t: (key: string, values?: Record<string, string | number>) => string;
  askT: (key: string, values?: Record<string, string | number>) => string;
}) {
  const askHref = `/drive/${encodeURIComponent(drive)}/addons/intelligence?q=${encodeURIComponent(query)}`;

  const matchLabels: Record<string, string> = {
    transcript: t("matchTranscript"),
    transcript_keyword: t("matchTranscriptKeyword"),
    clip: t("matchClip"),
    metadata: t("matchMetadata"),
    content: t("matchContent"),
    text_content_keyword: t("matchTextContentKeyword"),
  };

  // Per-section context menu — semantic results have their own
  // selection scope; cross-section selection (semantic + filename) is
  // intentionally out of scope for Phase 2 to avoid mode confusion.
  const { menuState, close, handlers } = useContextMenu();
  const [menuTarget, setMenuTarget] = useState<FileItem | null>(null);

  const handleFavoriteToggle = (updated: FileItem) => {
    onResultsChange(
      results.map((r) =>
        r.file_id === updated.id ? { ...r, file: updated } : r,
      ),
    );
  };

  return (
    <section
      aria-labelledby="semantic-search-heading"
      className="space-y-3"
    >
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h2
            id="semantic-search-heading"
            className="flex items-center gap-2 text-base font-semibold text-text-primary"
          >
            <Sparkles size={16} className="flex-shrink-0 text-accent-teal" />
            {t("semanticResults")}
          </h2>
          <p className="mt-1 text-xs text-text-muted">
            {t("semanticResultsDescription")}
          </p>
        </div>
        <button
          type="button"
          onClick={() => onSelect(askHref)}
          className="inline-flex flex-shrink-0 items-center gap-1.5 rounded-full bg-accent-teal/10 px-3 py-1.5 text-xs font-medium text-accent-teal transition-colors hover:bg-accent-teal/20"
        >
          <Sparkles size={12} className="flex-shrink-0" />
          <span className="truncate">{askT("button", { query })}</span>
        </button>
      </header>
      <div
        role="list"
        className="grid grid-cols-1 gap-x-4 gap-y-6 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4"
      >
        {results.map((result) => {
          const fileItem = result.file ?? fileItemFromSnapshot(result);
          return (
            <div role="listitem" key={result.file_id}>
              <FileCard
                file={fileItem}
                onFavoriteToggle={
                  result.file ? handleFavoriteToggle : undefined
                }
                onContextMenu={(e) => {
                  setMenuTarget(fileItem);
                  handlers.onContextMenu(e);
                }}
                onTouchStart={(e) => {
                  setMenuTarget(fileItem);
                  handlers.onTouchStart(e);
                }}
                onTouchEnd={handlers.onTouchEnd}
                onTouchMove={handlers.onTouchMove}
                matchOverlay={
                  <MatchOverlay
                    result={result}
                    onSelect={onSelect}
                    matchLabels={matchLabels}
                    matchedPagesLabel={(pages) =>
                      t("matchedPages", { pages: pages.join(", ") })
                    }
                  />
                }
              />
            </div>
          );
        })}
      </div>
      <FileContextMenu
        open={menuState.open}
        position={menuState.position}
        target={menuTarget}
        onClose={close}
      />
      {loading && (
        <div className="flex items-center justify-center py-3">
          <div className="h-4 w-4 animate-spin rounded-full border-2 border-accent border-t-transparent" />
        </div>
      )}
    </section>
  );
}

export default function SemanticSearchSlot({
  query,
  drive,
  filter,
  onSelect,
  context = "popup",
}: SemanticSearchSlotProps) {
  const t = useTranslations("search");
  // Separate namespace for the Ask link label: reuses the already-
  // translated "「{query}」について AI で質問応答" string from the
  // askSearch namespace (formerly owned by the deleted AskSearchMode
  // slot) so we don't add a duplicate translation just for a button.
  const askT = useTranslations("askSearch");
  const [available, setAvailable] = useState<boolean | null>(null);
  const [results, setResults] = useState<SemanticSearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Check availability on mount (rechecked when the active drive
  // changes, since /status is now drive-scoped).
  useEffect(() => {
    if (!drive) {
      setAvailable(false);
      return;
    }
    let cancelled = false;
    getSearchStatus(drive).then((res) => {
      if (!cancelled) {
        setAvailable(res.available);
      }
    });
    return () => {
      cancelled = true;
    };
  }, [drive]);

  // Debounced search
  useEffect(() => {
    if (!available || !query.trim() || !drive) {
      setResults([]);
      return;
    }

    if (debounceRef.current) clearTimeout(debounceRef.current);

    debounceRef.current = setTimeout(async () => {
      const filterType =
        filter === "all"
          ? undefined
          : (filter as "video" | "image" | "audio" | "document");
      setLoading(true);
      try {
        const res = await semanticSearch(query.trim(), drive, {
          limit: 20,
          type: filterType,
        });
        setResults(res.results);
      } catch {
        setResults([]);
      }
      setLoading(false);
    }, 300);

    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [query, drive, filter, available]);

  // Not available or not checked yet - render nothing
  if (available === null || !available) return null;

  // No query - nothing to show
  if (!query.trim()) return null;

  // Loading with no results yet
  if (loading && results.length === 0) {
    return (
      <div className="flex items-center justify-center py-4">
        <div className="h-4 w-4 animate-spin rounded-full border-2 border-accent border-t-transparent" />
      </div>
    );
  }

  if (results.length === 0) return null;

  const trimmedQuery = query.trim();

  if (context === "page") {
    return (
      <PageLayout
        results={results}
        loading={loading}
        query={trimmedQuery}
        drive={drive}
        onSelect={onSelect}
        onResultsChange={setResults}
        t={t}
        askT={askT}
      />
    );
  }

  return (
    <PopupLayout
      results={results}
      loading={loading}
      query={trimmedQuery}
      drive={drive}
      onSelect={onSelect}
      t={t}
      askT={askT}
    />
  );
}
