"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { ChevronDown, ChevronUp } from "lucide-react";
import { useTranslations } from "next-intl";
import { getSimilarFiles, SIMILAR_FILES_LIMIT } from "./api";
import type { SimilarFileItem } from "./api";

interface SimilarFilesSectionProps {
  fileId: string;
  drive: string;
}

type Status = "loading" | "loaded" | "unavailable";

// A request to an addon that is still starting can exceed the 15 s addon
// proxy timeout; the backend keeps computing and caches the result, so a
// later attempt hits the cache.
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
  const pair = `${drive}\u0000${fileId}`;
  const [shownPair, setShownPair] = useState(pair);
  const [results, setResults] = useState<SimilarFileItem[]>([]);
  const [status, setStatus] = useState<Status>("loading");
  const [isOpen, setIsOpen] = useState(true);
  const [attempt, setAttempt] = useState(0);

  // Reset during render rather than in an effect: an effect runs after
  // the paint, so the previous file's results would flash for a frame
  // under the new file.
  if (shownPair !== pair) {
    setShownPair(pair);
    setResults([]);
    setStatus("loading");
    setIsOpen(true);
  }

  useEffect(() => {
    const controller = new AbortController();
    let backoff: ReturnType<typeof setTimeout> | undefined;

    // Whether a request was aborted is read from its own signal, not from
    // what it settled with: an aborted fetch may still resolve, or reject
    // with an error that is not an AbortError.
    const run = async (retry: number) => {
      try {
        const data = await getSimilarFiles(fileId, drive, { signal: controller.signal });
        if (controller.signal.aborted) return;
        setResults(data.results);
        setStatus("loaded");
      } catch {
        if (controller.signal.aborted) return;
        if (retry === RETRY_DELAYS_MS.length) {
          setResults([]);
          setStatus("unavailable");
          return;
        }
        backoff = setTimeout(() => void run(retry + 1), RETRY_DELAYS_MS[retry]);
      }
    };
    void run(0);

    return () => {
      controller.abort();
      clearTimeout(backoff);
    };
  }, [fileId, drive, attempt]);

  const retry = () => {
    setStatus("loading");
    setAttempt((n) => n + 1);
  };

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
      {/* The same disclosure shape as the other two derived views.

          The weight is the group's, not a section's: this entry is in
          core's `file-relations` slot, which draws it under one
          "Related" heading beside core's own relations. At section
          weight it matched the heading above it, so the two read as
          sibling lists rather than as a group and its member.
          `DESIGN.md` §The Related group holds the values — the other
          member is in the core repository, so the table is the only
          thing keeping the pair in step. */}
      <button
        type="button"
        onClick={() => setIsOpen((open) => !open)}
        aria-expanded={isOpen}
        aria-controls={`similar-files-${fileId}`}
        className="flex w-full cursor-pointer items-center gap-2 text-xs font-medium text-text-muted transition-colors hover:text-text-primary"
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
                  {/* The same elements as the real card's text block,
                      with the text made transparent and the box filled —
                      not bars of a height computed from the type scale.
                      Ghosts that are merely present do not do the job:
                      the point is that the swap moves nothing, and a
                      height worked out by hand is a second copy of the
                      real card's typography that nothing keeps in step.
                      `text-[11px]` in particular sets font size and
                      inherits its line height, so the arithmetic depends
                      on a value neither file states. Sharing the classes
                      makes the two heights equal by construction.

                      The keyword row is drawn always, so a card that
                      turns out to have no keywords shrinks the box
                      rather than growing it — pulling the content below
                      up, which is the direction that does not shove
                      away what someone is already reading. */}
                  <div className="px-2 py-1.5">
                    <p className="w-4/5 animate-pulse truncate rounded-lg bg-bg-elevated text-xs text-transparent">
                      &nbsp;
                    </p>
                    <div className="mt-1 flex flex-wrap gap-0.5">
                      <span className="animate-pulse rounded-lg bg-bg-elevated px-1.5 py-px text-[11px] text-transparent">
                        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
                      </span>
                    </div>
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
                onClick={retry}
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
                      loading="lazy"
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
