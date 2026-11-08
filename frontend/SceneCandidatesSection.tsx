"use client";

import { useEffect, useId, useState, type MouseEvent } from "react";
import { useRouter } from "next/navigation";
import { useTranslations } from "next-intl";

import { cardGridTemplate, useCardColumns } from "@/lib/cardGrid";

import { fetchSceneSegments, getFrameUrl, getSceneTileUrl } from "./api";
import type { SceneSegment } from "./api";

interface SceneCandidatesSectionProps {
  query: string;
  drive: string;
  includeSceneClip: boolean;
  typeFilter: string | null;
  smartFolderId: string | null;
}

type View =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "failed" }
  | { status: "loaded"; segments: SceneSegment[]; partial: boolean };

function clock(seconds: number): string {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = seconds % 60;
  const ss = String(secs).padStart(2, "0");
  return hours > 0
    ? `${hours}:${String(minutes).padStart(2, "0")}:${ss}`
    : `${minutes}:${ss}`;
}

function formatRange(start: number, end: number): string {
  return `${clock(Math.floor(start))}-${clock(Math.ceil(end))}`;
}

function SceneCard({ segment }: { segment: SceneSegment }) {
  const router = useRouter();
  const [frameFailed, setFrameFailed] = useState(false);
  // Rounds down, as `?t=` links do everywhere: a link at the rounded-up
  // second would open after the first frame of the scene.
  const href = `/files/${segment.file_id}?t=${Math.floor(segment.start)}`;
  // Scene rows carry no mime type, so the extension is the only signal that a
  // candidate is a `.loft` reference.
  const isLoft = segment.filename.toLowerCase().endsWith(".loft");

  const open = (e: MouseEvent<HTMLAnchorElement>) => {
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey || e.button !== 0) return;
    e.preventDefault();
    router.push(href);
  };

  return (
    <a
      href={href}
      onClick={open}
      className="group min-w-0 overflow-hidden rounded-lg bg-bg-card focus-visible:outline-2 focus-visible:outline-accent hover:outline-2 hover:outline-accent"
    >
      {frameFailed ? (
        <div className="aspect-video w-full bg-bg-elevated" />
      ) : (
        <img
          src={
            isLoft
              ? getSceneTileUrl(segment.file_id, segment.peak)
              : getFrameUrl(segment.file_id, segment.peak)
          }
          alt=""
          loading="lazy"
          onError={() => setFrameFailed(true)}
          className="aspect-video w-full bg-bg-elevated object-cover"
        />
      )}
      <span className="block px-1.5 pb-1 pt-1.5">
        <span className="block text-xs tabular-nums text-text-muted group-hover:text-accent group-focus-visible:text-accent">
          {formatRange(segment.start, segment.end)}
        </span>
        <span className="line-clamp-2 wrap-anywhere text-xs leading-snug">{segment.filename}</span>
      </span>
    </a>
  );
}

export default function SceneCandidatesSection({
  query,
  drive,
  includeSceneClip,
  typeFilter,
  smartFolderId,
}: SceneCandidatesSectionProps) {
  const t = useTranslations("sceneCandidates");
  const { ref, columns } = useCardColumns();
  const gridId = useId();
  const [expanded, setExpanded] = useState(false);

  // Scene candidates are videos of the whole drive, so a narrower listing
  // would contradict them.
  const eligible =
    includeSceneClip &&
    query !== "" &&
    (typeFilter === null || typeFilter === "video") &&
    smartFolderId === null;

  const inputs = JSON.stringify([query, drive, includeSceneClip, typeFilter, smartFolderId]);
  const [seenInputs, setSeenInputs] = useState(inputs);
  const [view, setView] = useState<View>({ status: eligible ? "loading" : "idle" });

  // Adjusted during render so no frame shows the previous inputs' answer.
  if (seenInputs !== inputs) {
    setSeenInputs(inputs);
    setExpanded(false);
    setView({ status: eligible ? "loading" : "idle" });
  }

  useEffect(() => {
    if (!eligible) return;
    const controller = new AbortController();
    let current = true;
    fetchSceneSegments(query, drive, controller.signal)
      .then((res) => {
        if (!current) return;
        setView(
          res === null
            ? { status: "idle" }
            : { status: "loaded", segments: res.segments, partial: res.partial },
        );
      })
      .catch(() => {
        if (current) setView({ status: "failed" });
      });
    return () => {
      current = false;
      controller.abort();
    };
  }, [eligible, query, drive, typeFilter, smartFolderId]);

  if (!eligible || view.status === "idle") return null;

  const partialLine = view.status === "loaded" && view.partial && (
    <p className="text-xs text-text-muted">{t("partial")}</p>
  );
  const hasRow =
    view.status === "loading" || (view.status === "loaded" && view.segments.length > 0);

  return (
    <section className="min-w-0 px-3 pb-3 pt-2">
      <div className="flex flex-wrap items-baseline gap-x-2.5 gap-y-0.5 px-1">
        <h3 className="text-sm font-semibold">{t("heading")}</h3>
        {view.status === "loading" && (
          <p className="text-xs text-text-muted">{t("loading")}</p>
        )}
        {partialLine}
      </div>
      {hasRow && (
        <div
          ref={ref}
          id={gridId}
          aria-label={t("heading")}
          aria-busy={view.status === "loading"}
          className="grid items-start gap-3 px-1 pb-1 pt-2"
          style={{ gridTemplateColumns: cardGridTemplate(columns) }}
        >
          {view.status === "loading"
            ? [0, 1, 2, 3, 4, 5, 6, 7].map((i) => (
                <div
                  key={i}
                  aria-hidden
                  className="aspect-video min-w-0 animate-pulse rounded-lg bg-bg-elevated motion-reduce:animate-none"
                />
              ))
            : view.status === "loaded" &&
              view.segments.slice(0, expanded ? undefined : 8).map((segment) => (
                <SceneCard key={`${segment.file_id}:${segment.start}`} segment={segment} />
              ))}
        </div>
      )}
      {view.status === "loaded" && view.segments.length > 8 && (
        <button
          type="button"
          aria-expanded={expanded}
          aria-controls={gridId}
          onClick={() => setExpanded(!expanded)}
          className="mt-2 rounded-lg bg-sand px-3 py-2 text-sm hover:bg-sand-hover focus-visible:outline-2 focus-visible:outline-accent"
        >
          {expanded ? t("collapse") : t("showAll", { count: view.segments.length })}
        </button>
      )}
      {view.status === "failed" && (
        <p className="mt-1.5 px-1 text-xs text-text-muted">{t("failed")}</p>
      )}
      {view.status === "loaded" && view.segments.length === 0 && !view.partial && (
        <p className="mt-1.5 px-1 text-xs text-text-muted">{t("noIndex")}</p>
      )}
    </section>
  );
}
