"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { ChevronDown, ChevronUp } from "lucide-react";
import { useTranslations } from "next-intl";
import { getSimilarFiles } from "./api";
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

export default function SimilarFilesSection({ fileId, drive }: SimilarFilesSectionProps) {
  const t = useTranslations("file");
  const [results, setResults] = useState<SimilarFileItem[]>([]);
  const [status, setStatus] = useState<Status>("idle");
  const [isOpen, setIsOpen] = useState(false);
  const requestIdRef = useRef(0);

  // Reset to idle whenever we navigate to a different file. Detection
  // is heavy on the backend (CLIP / tf-idf / whisper similarity), so we
  // never auto-trigger — the user must explicitly press the button.
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
        setIsOpen(true);
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

  const countLabel = status === "loaded" ? ` (${results.length})` : "";

  return (
    // The result grid sizes its columns against this element, not the
    // viewport: the section renders both full-width and inside the
    // ~300px inspector, which a viewport breakpoint cannot tell apart.
    // A containment context is safe here only because the subtree holds
    // nothing but thumbnails — one wrapped around a <video> or a
    // cross-origin iframe renders the subtree rotated and spinning on
    // iOS Safari. hako 7bFYOh3vFZP9EEuf9Ym_5.
    <div className="@container">
      {/* Collapsed by default, and the same disclosure shape as the
          other two derived views. Detection is heavy enough that it has
          always been opt-in, which made this a permanent heading over a
          single button — the header now carries that weight itself and
          the button waits inside for someone who opened the drawer. */}
      <button
        type="button"
        onClick={() => setIsOpen((v) => !v)}
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
          {status === "idle" && (
            <button
              type="button"
              onClick={fetchSimilar}
              className="rounded-lg bg-bg-elevated px-2.5 py-1 text-xs text-text-primary transition-colors hover:bg-bg-card"
            >
              {t("similarFilesDetect")}
            </button>
          )}

          {status === "loading" && (
            <span className="text-xs text-text-muted">
              {t("similarFilesDetecting")}
            </span>
          )}

          {status === "unavailable" && (
            <p className="text-xs text-text-muted">
              {t("similarFilesUnavailable")}
            </p>
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
