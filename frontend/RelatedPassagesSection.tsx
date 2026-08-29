"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { ChevronDown, ChevronRight } from "lucide-react";
import { useTranslations } from "next-intl";

import { getRelatedPassages } from "./api";
import type { PassageRef, RelatedPassageItem } from "./api";

interface RelatedPassagesSectionProps {
  fileId: string;
  drive: string;
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
 * It runs on open and **renders nothing at all unless it found
 * something**, which is what makes it worth having on the page: the
 * section being there is itself the signal. Measured on a real drive,
 * about half of files produce no pair, and a viewer who scrolls down to
 * a "no connections" line has been made to work for nothing. The
 * earlier shape — a button, and a message when it came back empty — put
 * that cost on every file.
 *
 * A failed lookup still says so. It is rare, it is actionable, and a
 * silent failure here once hid a 15-second timeout.
 *
 * Spec ``2026-08-29-related-passages.md`` §5.4.
 */
export default function RelatedPassagesSection({
  fileId,
  drive,
}: RelatedPassagesSectionProps) {
  const t = useTranslations("file");
  const [results, setResults] = useState<RelatedPassageItem[]>([]);
  const [status, setStatus] = useState<Status>("idle");
  const [isOpen, setIsOpen] = useState(false);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const requestIdRef = useRef(0);

  const toggle = useCallback((fileId: string) => {
    setExpanded((current) => {
      const next = new Set(current);
      if (!next.delete(fileId)) next.add(fileId);
      return next;
    });
  }, []);

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
    setExpanded(new Set());
  }, [fileId, drive]);

  useEffect(() => {
    if (!drive) return;
    if (status !== "idle") return;
    void fetchPassages();
  }, [drive, status, fetchPassages]);

  // The `/files/{id}` route renders this before it knows the drive
  // (`drive={file?.drive ?? ""}` while its own getFile is in flight).
  // Every route here is drive-scoped and the host proxy rejects a
  // request without the header, so nothing may be fetched yet — and
  // the reset above re-arms the auto-fetch once the drive lands.
  if (!drive) return null;

  // Present only when it has something. See the note on the component.
  if (status === "idle" || status === "loading") return null;
  if (status === "loaded" && results.length === 0) return null;

  return (
    <div>
      <div className="mb-3 flex items-center justify-between gap-2">
        {results.length > 0 ? (
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
      </div>

      {status === "unavailable" && (
        <p className="text-xs text-text-muted">
          {t("relatedPassagesUnavailable")}
        </p>
      )}

      {isOpen && results.length > 0 && (
        <div className="space-y-2">
          {results.map((item) => (
            // One row per other file, so the file id is a stable key.
            <PassageCard
              key={item.file_id}
              item={item}
              expanded={expanded.has(item.file_id)}
              onToggle={() => toggle(item.file_id)}
            />
          ))}
        </div>
      )}
    </div>
  );
}

/**
 * One pair: the other file as the card's heading, the two passages below it.
 *
 * The card is what tells one row from the next — the two passages used to
 * be distinguished only by text colour, which is nearly invisible in dark
 * mode, and the filename sat between them so it read as a caption for
 * whichever one you looked at first. A fixed label column settles which
 * side is which without asking anyone to compare shades.
 *
 * Both passages are clamped to two lines. Five pairs of full transcript
 * chunks is four thousand characters of unpunctuated prose, which buries
 * the rest of the page.
 */
function PassageCard({
  item,
  expanded,
  onToggle,
}: {
  item: RelatedPassageItem;
  expanded: boolean;
  onToggle: () => void;
}) {
  const t = useTranslations("file");
  const locator = locatorLabel(item.match);

  return (
    <article className="rounded-xl bg-bg-card p-3">
      <div className="flex items-start justify-between gap-2">
        <Link
          href={passageHref(item.file_id, item.match)}
          className="flex min-w-0 flex-1 items-baseline gap-1 text-xs text-text-muted underline-offset-2 hover:underline"
        >
          <span className="truncate">{item.filename}</span>
          {/* Never truncated: the locator is where the link goes. */}
          {locator && <span className="shrink-0">{locator}</span>}
        </Link>
        <button
          type="button"
          onClick={onToggle}
          aria-expanded={expanded}
          // The name carries the filename because a result set holds
          // several of these buttons, and "expand" alone tells a screen
          // reader or a voice-control user nothing about which one.
          aria-label={
            expanded
              ? t("relatedPassagesCollapse", { filename: item.filename })
              : t("relatedPassagesExpand", { filename: item.filename })
          }
          // The passages are deliberately not clickable, so this is the
          // only way to expand — the negative margin buys a real touch
          // target without moving the icon or growing the card.
          className="-m-2 shrink-0 rounded-full p-2 text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary"
        >
          <ChevronDown
            size={14}
            className={`transition-transform ${expanded ? "rotate-180" : ""}`}
          />
        </button>
      </div>

      <dl className="mt-2 space-y-1.5">
        <PassageRow
          label={t("relatedPassagesThisFile")}
          text={item.source.text}
          expanded={expanded}
          testId="source-passage"
        />
        <PassageRow
          label={t("relatedPassagesMatch")}
          text={item.match.text}
          expanded={expanded}
          testId="match-passage"
          muted
        />
      </dl>
    </article>
  );
}

function PassageRow({
  label,
  text,
  expanded,
  testId,
  muted = false,
}: {
  label: string;
  text: string;
  expanded: boolean;
  testId: string;
  muted?: boolean;
}) {
  return (
    <div className="flex gap-3">
      <dt className="w-20 shrink-0 text-xs leading-relaxed text-text-muted">
        {label}
      </dt>
      {/* Selectable on purpose, and never a click target: selecting a
          passage is how it reaches Knowledge's quotation basket, so the
          chevron owns expansion instead. */}
      <dd
        data-testid={testId}
        data-expanded={expanded ? "true" : "false"}
        className={`m-0 min-w-0 text-sm leading-relaxed ${
          muted ? "text-text-muted" : "text-text-primary"
        } ${expanded ? "" : "line-clamp-2"}`}
      >
        {text}
      </dd>
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
