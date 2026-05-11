"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useTranslations } from "next-intl";
import { getSimilarFiles } from "./api";
import type { SimilarFileItem, KeywordScore } from "./api";

interface SimilarFilesSectionProps {
  fileId: string;
  drive: string;
}

function matchTypeLabel(matchType: string): string {
  const labels: Record<string, string> = {
    clip: "visual",
    tfidf: "topic",
    "clip+tfidf": "visual+topic",
    whisper: "audio",
    text_content: "text",
    metadata: "metadata",
  };
  return labels[matchType] ?? matchType;
}

function ScoreBadge({ item }: { item: SimilarFileItem }) {
  const parts: string[] = [];
  if (item.primary_score != null) {
    parts.push(`V:${item.primary_score.toFixed(2)}`);
  }
  if (item.secondary_score != null) {
    parts.push(`T:${item.secondary_score.toFixed(2)}`);
  }

  return (
    <span className="inline-flex items-center gap-1 text-[10px] text-text-muted">
      <span className="rounded-lg bg-bg-elevated px-1 py-0.5">
        {matchTypeLabel(item.match_type)}
      </span>
      {parts.length > 0 && (
        <span className="tabular-nums">{parts.join(" ")}</span>
      )}
    </span>
  );
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
  const [sourceKeywords, setSourceKeywords] = useState<KeywordScore[]>([]);
  const [status, setStatus] = useState<Status>("idle");
  const requestIdRef = useRef(0);

  // Reset to idle whenever we navigate to a different file. Detection
  // is heavy on the backend (CLIP / tf-idf / whisper similarity), so we
  // never auto-trigger — the user must explicitly press the button.
  useEffect(() => {
    requestIdRef.current += 1;
    setResults([]);
    setSourceKeywords([]);
    setStatus("idle");
  }, [fileId]);

  const fetchSimilar = useCallback(async () => {
    const reqId = ++requestIdRef.current;
    setStatus("loading");

    for (let attempt = 0; attempt <= RETRY_DELAYS_MS.length; attempt += 1) {
      try {
        const data = await getSimilarFiles(fileId, drive);
        if (reqId !== requestIdRef.current) return; // stale (file changed / re-clicked)
        setResults(data.results);
        setSourceKeywords(data.source_keywords ?? []);
        setStatus("loaded");
        return;
      } catch {
        if (attempt === RETRY_DELAYS_MS.length) {
          if (reqId !== requestIdRef.current) return;
          setResults([]);
          setSourceKeywords([]);
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
        <h2 className="text-sm font-semibold text-text-muted">
          {t("similarFiles")}
        </h2>
        {status !== "loading" && (
          <button
            type="button"
            onClick={fetchSimilar}
            className="rounded-lg bg-bg-elevated px-2.5 py-1 text-xs text-text-primary transition-colors hover:bg-bg-card"
          >
            {status === "idle"
              ? t("similarFilesDetect")
              : t("similarFilesRetry")}
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

      {status === "loaded" && results.length === 0 && (
        <p className="text-xs text-text-muted">{t("similarFilesEmpty")}</p>
      )}

      {status === "loaded" && results.length > 0 && (
        <>
          {sourceKeywords.length > 0 && (
            <div className="mb-3 flex flex-wrap gap-1">
              {sourceKeywords.map((kw) => (
                <span
                  key={kw.word}
                  className="rounded-full bg-bg-elevated px-2 py-0.5 text-[10px] text-text-muted"
                  title={`TF-IDF: ${kw.score?.toFixed(4)}`}
                >
                  {kw.word}
                  <span className="ml-0.5 opacity-50">
                    {kw.score?.toFixed(3)}
                  </span>
                </span>
              ))}
            </div>
          )}
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
            {results.map((item) => (
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
                  <div className="mt-0.5 flex flex-col gap-0.5">
                    <ScoreBadge item={item} />
                    {item.shared_keywords.length > 0 && (
                      <div className="mt-0.5 flex flex-wrap gap-0.5">
                        {item.shared_keywords.map((kw) => (
                          <span
                            key={kw.word}
                            className="rounded-lg bg-bg-elevated px-1 py-px text-[9px] text-text-muted"
                            title={`src:${kw.source_tfidf?.toFixed(4)} tgt:${kw.target_tfidf?.toFixed(4)} rel:${kw.relevance?.toFixed(6)}`}
                          >
                            {kw.word}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </Link>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
