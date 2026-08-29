"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { ChevronRight } from "lucide-react";
import { useTranslations } from "next-intl";

import { getRelatedPassages } from "./api";
import type { PassageRef, RelatedPassageItem } from "./api";

interface RelatedPassagesSectionProps {
  fileId: string;
  drive: string;
  trustTier?: "verified" | "unverified";
  trustReviewedAt?: string | null;
}

type Status = "idle" | "loading" | "loaded" | "unavailable";

/**
 * Where a passage of this file meets a passage of something you vouched for.
 *
 * Shows **pointers, never generated prose**: both halves of every row are
 * the passage's own words, and the link lands on the passage — a page for
 * a document, a timestamp for a transcript — rather than merely on the
 * file. No LLM is called (hako ``DPcjrRgspKAXqHjHOkJ8L``).
 *
 * This used to live inside the promotion panel, which meant it vanished
 * the moment a viewer ruled on a clip and never applied to anything else.
 * Connections outlive that decision and are worth knowing for any file,
 * so they are their own section now, and the panel keeps only the
 * question it asks.
 *
 * Matching is deliberate rather than automatic: the work is a KNN plus a
 * matrix product, and most file opens will not consult it. The exception
 * is a source nobody has ruled on yet, where the promotion panel is also
 * on screen — making a viewer press a button to see their own evidence
 * would put friction exactly where the trust design cannot afford it
 * (spec ``2026-08-29-web-clip-promotion.md`` §11 risk 3).
 *
 * Spec ``2026-08-29-related-passages.md`` §5.4.
 */
export default function RelatedPassagesSection({
  fileId,
  drive,
  trustTier,
  trustReviewedAt,
}: RelatedPassagesSectionProps) {
  const t = useTranslations("file");
  const [results, setResults] = useState<RelatedPassageItem[]>([]);
  const [status, setStatus] = useState<Status>("idle");
  const [isOpen, setIsOpen] = useState(false);
  const requestIdRef = useRef(0);

  const awaitingRuling = trustTier === "unverified" && !trustReviewedAt;

  const fetchPassages = useCallback(async () => {
    const reqId = ++requestIdRef.current;
    setStatus("loading");
    try {
      const data = await getRelatedPassages(fileId, drive);
      if (reqId !== requestIdRef.current) return; // stale (file changed)
      setResults(data.results);
      setStatus("loaded");
      setIsOpen(true);
    } catch {
      if (reqId !== requestIdRef.current) return;
      setResults([]);
      setStatus("unavailable");
    }
  }, [fileId, drive]);

  useEffect(() => {
    requestIdRef.current += 1;
    setResults([]);
    setStatus("idle");
    setIsOpen(false);
  }, [fileId, drive]);

  useEffect(() => {
    if (!drive) return;
    if (!awaitingRuling) return;
    if (status !== "idle") return;
    void fetchPassages();
  }, [drive, awaitingRuling, status, fetchPassages]);

  // The `/files/{id}` route renders this before it knows the drive
  // (`drive={file?.drive ?? ""}` while its own getFile is in flight).
  // Every route here is drive-scoped and the host proxy rejects a
  // request without the header, so nothing may be fetched yet — and
  // the reset above re-arms the auto-fetch once the drive lands.
  if (!drive) return null;

  // Nothing to say and nobody asked: an empty shell would be noise on
  // every file that has no connections, which is most of them.
  if (awaitingRuling && status !== "loaded") return null;
  if (awaitingRuling && results.length === 0) return null;

  return (
    <div>
      <div className="mb-3 flex items-center justify-between gap-2">
        {status === "loaded" && results.length > 0 ? (
          <button
            type="button"
            onClick={() => setIsOpen((v) => !v)}
            className="flex items-center gap-1 text-sm font-semibold text-text-muted"
          >
            <ChevronRight
              size={14}
              className={`transition-transform ${isOpen ? "rotate-90" : ""}`}
            />
            {t("relatedPassages")}
          </button>
        ) : (
          <h2 className="text-sm font-semibold text-text-muted">
            {t("relatedPassages")}
          </h2>
        )}
        {status === "idle" && (
          <button
            type="button"
            onClick={fetchPassages}
            className="rounded-lg bg-bg-elevated px-2.5 py-1 text-xs text-text-primary transition-colors hover:bg-bg-card"
          >
            {t("relatedPassagesFind")}
          </button>
        )}
        {status === "loading" && (
          <span className="text-xs text-text-muted">
            {t("relatedPassagesSearching")}
          </span>
        )}
      </div>

      {status === "unavailable" && (
        <p className="text-xs text-text-muted">
          {t("relatedPassagesUnavailable")}
        </p>
      )}

      {status === "loaded" && results.length === 0 && (
        <p className="text-xs text-text-muted">{t("relatedPassagesEmpty")}</p>
      )}

      {status === "loaded" && isOpen && results.length > 0 && (
        <div className="space-y-4">
          {results.map((item) => (
            // One row per other file, so the file id is a stable key.
            <div key={item.file_id}>
              {/* Selectable on purpose: selecting a passage is how it
                  reaches Knowledge's quotation basket. */}
              <blockquote
                data-testid="source-passage"
                className="border-l-2 border-bg-border pl-3 text-sm text-text-primary break-words"
              >
                {item.source.text}
              </blockquote>
              <Link
                href={passageHref(item.file_id, item.match)}
                className="mt-1.5 inline-block break-all text-xs text-text-muted underline-offset-2 hover:underline"
              >
                {item.filename}
                {locatorLabel(item.match)}
              </Link>
              <blockquote
                data-testid="match-passage"
                className="mt-1 border-l-2 border-bg-border pl-3 text-sm text-text-muted break-words"
              >
                {item.match.text}
              </blockquote>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/** Deep-link to the passage: ``?t=`` for media, ``?page=`` for documents. */
export function passageHref(fileId: string, ref: PassageRef): string {
  if (ref.timestamp !== null && ref.timestamp !== undefined) {
    return `/files/${fileId}?t=${Math.floor(ref.timestamp)}`;
  }
  if (ref.page !== null && ref.page !== undefined) {
    return `/files/${fileId}?page=${ref.page}`;
  }
  return `/files/${fileId}`;
}

function locatorLabel(ref: PassageRef): string {
  if (ref.timestamp !== null && ref.timestamp !== undefined) {
    const total = Math.floor(ref.timestamp);
    const minutes = Math.floor(total / 60);
    const seconds = String(total % 60).padStart(2, "0");
    return ` \u00b7 ${minutes}:${seconds}`;
  }
  if (ref.page !== null && ref.page !== undefined) {
    return ` \u00b7 p.${ref.page}`;
  }
  return "";
}
