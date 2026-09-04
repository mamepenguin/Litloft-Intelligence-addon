"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { ChevronDown, ChevronUp } from "lucide-react";
import { useTranslations } from "next-intl";
import { getSimilarFiles, SIMILAR_FILES_LIMIT } from "./api";
import type { SimilarFileItem } from "./api";

interface SimilarFilesSectionProps {
  fileId: string;
  drive: string;
}

type Status = "idle" | "loading" | "loaded" | "unavailable";

// Heavy similarity computations frequently exceed the 15 s addon
// proxy timeout on the first request for a cold file. The backend
// keeps computing and caches the result, so a follow-up request a
// few seconds later hits the cache. We auto-retry up to twice with
// growing delays before surrendering to the unavailable state.
const RETRY_DELAYS_MS = [6000, 12000];

// One ghost per neighbour the request asks for, so a full result set
// swaps in at exactly the height the ghosts reserved. A shorter set
// shrinks the box, which reads as the answer arriving; a set that grew
// past the reservation would push everything below it down, which is
// the jump this exists to prevent. Taken from the request rather than
// written out again — the two agreeing is the whole mechanism.
const GHOST_CARDS = SIMILAR_FILES_LIMIT;

export default function SimilarFilesSection({ fileId, drive }: SimilarFilesSectionProps) {
  const t = useTranslations("file");
  const [results, setResults] = useState<SimilarFileItem[]>([]);
  const [status, setStatus] = useState<Status>("idle");
  const [isOpen, setIsOpen] = useState(false);
  const requestIdRef = useRef(0);

  // Reset to idle whenever we navigate to a different file. Detection
  // is heavy on the backend (CLIP / tf-idf / whisper similarity), so it
  // never runs on mount — opening the disclosure is what asks for it.
  useEffect(() => {
    requestIdRef.current += 1;
    setResults([]);
    setStatus("idle");
    setIsOpen(false);
  }, [fileId]);

  const fetchSimilar = useCallback(async () => {
    const reqId = ++requestIdRef.current;
    setStatus("loading");

    for (let attempt = 0; attempt <= RETRY_DELAYS_MS.length; attempt += 1) {
      try {
        const data = await getSimilarFiles(fileId, drive);
        if (reqId !== requestIdRef.current) return; // stale (file changed / re-clicked)
        setResults(data.results);
        setStatus("loaded");
        return;
      } catch {
        if (attempt === RETRY_DELAYS_MS.length) {
          if (reqId !== requestIdRef.current) return;
          setResults([]);
          setStatus("unavailable");
          return;
        }
        // Wait for the backend to finish + populate the cache, then retry.
        await new Promise((resolve) =>
          setTimeout(resolve, RETRY_DELAYS_MS[attempt]),
        );
        if (reqId !== requestIdRef.current) return;
      }
    }
  }, [fileId, drive]);

  // Opening is the trigger. That works under either reading of how long
  // detection takes: on a warm file the answer lands before the ghosts
  // have said much, and on a cold one the retries above cover the addon
  // proxy timeout — while a file nobody opens this on never computes at
  // all. Only the first open fetches; `unavailable` keeps its own retry
  // button rather than re-firing on every open/close.
  const handleToggle = useCallback(() => {
    const next = !isOpen;
    setIsOpen(next);
    if (next && status === "idle") void fetchSimilar();
  }, [isOpen, status, fetchSimilar]);

  const countLabel = status === "loaded" ? ` (${results.length})` : "";

  return (
    // The result grid sizes its columns against this element, not the
    // viewport: the section renders both full-width and inside the
    // ~384px inspector, which a viewport breakpoint cannot tell apart.
    // A containment context is safe here only because the subtree holds
    // nothing but thumbnails — one wrapped around a <video> or a
    // cross-origin iframe renders the subtree rotated and spinning on
    // iOS Safari. hako 7bFYOh3vFZP9EEuf9Ym_5.
    <div className="@container">
      {/* Collapsed by default, and the same disclosure shape as the
          other two derived views. */}
      <button
        type="button"
        onClick={handleToggle}
        aria-expanded={isOpen}
        aria-controls={`similar-files-${fileId}`}
        className="flex w-full cursor-pointer items-center gap-2 text-sm font-semibold text-text-muted transition-colors hover:text-text-primary"
      >
        <span>{t("similarFiles")}{countLabel}</span>
        <span className="ml-auto" aria-hidden>
          {isOpen ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </span>
      </button>

      {isOpen && (
        <div id={`similar-files-${fileId}`} className="mt-2">
          {status === "loading" && (
            <div
              data-testid="similar-files-ghosts"
              aria-hidden
              className="grid grid-cols-2 gap-3 @lg:grid-cols-3"
            >
              {Array.from({ length: GHOST_CARDS }, (_, i) => (
                <div key={i} className="overflow-hidden rounded-lg bg-bg-card">
                  <div className="aspect-video w-full animate-pulse bg-bg-elevated" />
                  {/* Sized against the real card's text block below, not
                      guessed: a filename line (`text-xs`, 16px) plus the
                      keyword chips (`mt-1`, then ~15px). Ghosts that are
                      merely present do not do the job — the point is that
                      the swap moves nothing. Reserving the taller of the
                      two real shapes means a card without keywords
                      shrinks the box rather than growing it, which pulls
                      content up instead of shoving it down. */}
                  <div className="px-2 py-1.5">
                    <div className="h-4 w-4/5 animate-pulse rounded-lg bg-bg-elevated" />
                    <div className="mt-1 h-[15px] w-1/2 animate-pulse rounded-lg bg-bg-elevated" />
                  </div>
                </div>
              ))}
            </div>
          )}

          {status === "unavailable" && (
            <div className="flex flex-wrap items-center gap-2">
              <p className="text-xs text-text-muted">
                {t("similarFilesUnavailable")}
              </p>
              <button
                type="button"
                onClick={() => void fetchSimilar()}
                className="rounded-lg bg-bg-elevated px-2.5 py-1 text-xs text-text-primary transition-colors hover:bg-bg-card"
              >
                {t("similarFilesRetry")}
              </button>
            </div>
          )}

          {status === "loaded" && results.length === 0 && (
            <p className="text-xs text-text-muted">{t("similarFilesEmpty")}</p>
          )}

          {status === "loaded" && results.length > 0 && (
            // @lg = 32rem, the narrowest host where a third column still
            // leaves each card ~160px of thumbnail.
            <div className="grid grid-cols-2 gap-3 @lg:grid-cols-3">
              {results.map((item: SimilarFileItem) => (
                <Link
                  key={item.file_id}
                  href={`/files/${item.file_id}`}
                  className="group overflow-hidden rounded-lg bg-bg-card transition-colors hover:bg-bg-elevated"
                >
                  <div className="relative aspect-video w-full overflow-hidden bg-bg-elevated">
                    <img
                      src={`/api/files/${item.file_id}/thumbnail`}
                      alt=""
                      className="h-full w-full object-cover transition-transform group-hover:scale-105"
                      onError={(e) => {
                        e.currentTarget.style.display = "none";
                      }}
                    />
                  </div>
                  <div className="px-2 py-1.5">
                    <p className="truncate text-xs text-text-primary">
                      {item.filename}
                    </p>
                    {item.shared_keywords.length > 0 && (
                      <div className="mt-1 flex flex-wrap gap-0.5">
                        {item.shared_keywords.slice(0, 3).map((kw) => (
                          <span
                            key={kw.word}
                            className="rounded-lg bg-bg-elevated px-1.5 py-px text-[11px] text-text-muted"
                          >
                            {kw.word}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                </Link>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
