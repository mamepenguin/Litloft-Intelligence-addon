"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { ChevronRight } from "lucide-react";
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

  return (
    <div>
      <div className="mb-3 flex items-center justify-between gap-2">
        {status === "loaded" ? (
          <button
            type="button"
            onClick={() => setIsOpen((v) => !v)}
            className="flex items-center gap-1 text-sm font-semibold text-text-muted"
          >
            <ChevronRight
              size={14}
              className={`transition-transform ${isOpen ? "rotate-90" : ""}`}
            />
            {t("similarFiles")}
          </button>
        ) : (
          <h2 className="text-sm font-semibold text-text-muted">
            {t("similarFiles")}
          </h2>
        )}
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
      </div>

      {status === "unavailable" && (
        <p className="text-xs text-text-muted">
          {t("similarFilesUnavailable")}
        </p>
      )}

      {status === "loaded" && isOpen && results.length === 0 && (
        <p className="text-xs text-text-muted">{t("similarFilesEmpty")}</p>
      )}

      {status === "loaded" && isOpen && results.length > 0 && (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
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
                        className="rounded-lg bg-bg-elevated px-1 py-px text-[9px] text-text-muted"
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
  );
}
