"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import {
  ArrowUpRight,
  ChevronDown,
  ChevronRight,
  Clock,
  FileText,
} from "lucide-react";
import { useTranslations } from "next-intl";
import type { MediaController } from "@/lib/mediaController";

import { getRelatedPassages } from "./api";
import type { PassageRef, RelatedPassageItem } from "./api";
import { passageWindow } from "./passageText";

interface RelatedPassagesSectionProps {
  fileId: string;
  drive: string;
  /** Playback handle for the file being read, so a row can move in place. */
  mediaController?: MediaController | null;
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
 * It runs on open and **renders nothing at all unless it found
 * something**, which is what makes it worth having on the page: the
 * section being there is itself the signal. Measured on the real
 * library, only about a third of files produce a pair, and a viewer who
 * scrolls down to a "no connections" line has been made to work for
 * nothing.
 *
 * A failed lookup still says so. It is rare, it is actionable, and a
 * silent failure here once hid a 15-second timeout.
 *
 * Spec ``2026-08-30-related-passages-recognition-ui.md``.
 */
export default function RelatedPassagesSection({
  fileId,
  drive,
  mediaController,
}: RelatedPassagesSectionProps) {
  const t = useTranslations("file");
  const [results, setResults] = useState<RelatedPassageItem[]>([]);
  const [status, setStatus] = useState<Status>("idle");
  const [isOpen, setIsOpen] = useState(false);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const requestIdRef = useRef(0);

  const toggle = useCallback((id: string) => {
    setExpanded((current) => {
      const next = new Set(current);
      if (!next.delete(id)) next.add(id);
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
            <span className="ml-0.5 font-normal text-text-muted">
              {results.length}
            </span>
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
              mediaController={mediaController}
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
 * One pair, read top to bottom: where this connection sits in what you
 * are reading, who else covers it, and their words.
 *
 * The two passages used to sit side by side under a fixed label column,
 * which asked the reader to do the comparison and spent the widest part
 * of the card on text they already had — they are reading this file.
 * Only the other passage is shown by default now; this file's own is
 * behind the toggle, reached more often by the anchor above, which moves
 * the player rather than leaving the page.
 *
 * Two locators end up on screen and they must not read alike. The
 * anchor keeps this file's, in the link colour with an icon and no
 * filename; the line under it leads with the other file's name and ends
 * in an arrow. One stays here, one takes you away.
 */
function PassageCard({
  item,
  mediaController,
  expanded,
  onToggle,
}: {
  item: RelatedPassageItem;
  mediaController?: MediaController | null;
  expanded: boolean;
  onToggle: () => void;
}) {
  const t = useTranslations("file");
  const locator = locatorLabel(item.match);
  // No terms yet — the excerpt is still moved off the severed word a
  // chunk boundary leaves behind, which is the half of the windowing
  // that does not need to know what the words mean.
  const excerpt = useMemo(
    () => passageWindow(item.match.text, []),
    [item.match.text],
  );

  return (
    <article className="rounded-xl bg-bg-card p-3">
      <SourceAnchor source={item.source} mediaController={mediaController} />

      <Link
        href={passageHref(item.file_id, item.match)}
        className="mt-1 flex items-baseline gap-1 text-xs text-text-muted underline-offset-2 hover:underline"
      >
        <span className="truncate">{item.filename}</span>
        {/* Never truncated: the locator is where the link goes. */}
        {locator && <span className="shrink-0">{locator}</span>}
        <ArrowUpRight size={12} className="shrink-0 self-center" />
      </Link>

      {expanded ? (
        <dl className="mt-2 space-y-2">
          <FullPassage
            label={t("relatedPassagesThisFile")}
            text={item.source.text}
            testId="source-passage"
          />
          <FullPassage
            label={t("relatedPassagesMatch")}
            text={item.match.text}
            testId="match-passage"
          />
        </dl>
      ) : (
        <p className="mt-1.5 line-clamp-2 text-sm leading-relaxed text-text-primary">
          {/* Outside the passage element and unselectable, so neither the
              rendered text nor a selection sent to the quotation basket
              picks up a character the author never wrote. */}
          {excerpt.truncatedStart && (
            <span aria-hidden className="select-none text-text-muted">
              {"…"}
            </span>
          )}
          {/* Selectable on purpose, and never a click target: selecting a
              passage is how it reaches Knowledge's quotation basket, so the
              toggle below owns expansion instead. */}
          <span data-testid="match-passage">{excerpt.text}</span>
        </p>
      )}

      <div className="mt-1.5 flex justify-end">
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
          className="flex items-center gap-1 rounded-lg px-1.5 py-0.5 text-xs text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary"
        >
          {expanded
            ? t("relatedPassagesHideFull")
            : t("relatedPassagesShowFull")}
          <ChevronDown
            size={12}
            className={`transition-transform ${expanded ? "rotate-180" : ""}`}
          />
        </button>
      </div>
    </article>
  );
}

/**
 * Where this connection sits in the file being read.
 *
 * Moves in place when it can — the player is already on the page, and
 * the point of the section is the connection, not a trip to another
 * route. When it cannot (a plain text file, or media whose player has
 * published no controller), the locator is rendered as text rather than
 * as a button that does nothing. DESIGN.md §2.5: an affordance that will
 * never act must read as the prose around it, not as a dimmed link.
 */
function SourceAnchor({
  source,
  mediaController,
}: {
  source: PassageRef;
  mediaController?: MediaController | null;
}) {
  const t = useTranslations("file");
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  const hasTime = source.timestamp !== null && source.timestamp !== undefined;
  const hasPage = source.page !== null && source.page !== undefined;
  if (!hasTime && !hasPage) return null;

  const label = locatorLabel(source, { bare: true });
  const Icon = hasTime ? Clock : FileText;
  const body = (
    <>
      <Icon size={12} className="shrink-0" />
      {label}
    </>
  );

  const seek = () => {
    if (hasTime && mediaController) {
      mediaController.seek(source.timestamp as number);
      mediaController.play();
      return;
    }
    if (hasPage) {
      const params = new URLSearchParams(searchParams?.toString() ?? "");
      params.set("page", String(source.page));
      router.replace(`${pathname}?${params.toString()}`, { scroll: false });
    }
  };

  const canAct = (hasTime && !!mediaController) || hasPage;
  if (!canAct) {
    return (
      <p
        data-testid="source-anchor"
        className="flex items-center gap-1 text-xs text-text-muted"
      >
        {body}
      </p>
    );
  }

  return (
    <button
      type="button"
      data-testid="source-anchor"
      onClick={seek}
      aria-label={t("relatedPassagesJumpHere", { locator: label })}
      className="flex items-center gap-1 text-xs text-accent transition-colors hover:text-accent-hover"
    >
      {body}
    </button>
  );
}

function FullPassage({
  label,
  text,
  testId,
}: {
  label: string;
  text: string;
  testId: string;
}) {
  return (
    <div>
      <dt className="text-[11px] font-medium text-text-muted">{label}</dt>
      <dd
        data-testid={testId}
        className="m-0 mt-0.5 text-sm leading-relaxed text-text-primary"
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

function locatorLabel(ref: PassageRef, opts?: { bare?: boolean }): string {
  const prefix = opts?.bare ? "" : " · ";
  if (ref.timestamp !== null && ref.timestamp !== undefined) {
    const total = Math.floor(ref.timestamp);
    const minutes = Math.floor(total / 60);
    const seconds = String(total % 60).padStart(2, "0");
    return `${prefix}${minutes}:${seconds}`;
  }
  if (ref.page !== null && ref.page !== undefined) {
    return `${prefix}p.${ref.page}`;
  }
  return "";
}
