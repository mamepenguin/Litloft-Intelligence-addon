"use client";

import type {
  FocusEvent,
  MouseEvent,
  PointerEvent as ReactPointerEvent,
  RefObject,
} from "react";
import { ChevronDown, ChevronUp, Film, Loader2 } from "lucide-react";
import { useTranslations } from "next-intl";

import { formatDuration } from "@/lib/format";
import type { ClipTimestampItem, VideoVisualSceneItem } from "../api";
import {
  estimateLabelWidth,
  frameIndexAt,
  rulerFraction,
  timeLabels,
  windowStartFor,
} from "./scenesModel";

export type ScenesView = "collapsed" | "loading" | "expanded";

export type ScenesActionKind = "generate" | "retry";

export interface ScenesAction {
  kind: ScenesActionKind;
  label: string;
  filled: boolean;
}

export interface ScenesVisualView {
  count: number | null;
  progress: { text: string; updating: boolean } | null;
  spinner: boolean;
  marks: VideoVisualSceneItem[];
  dots: number[];
  scene: { time: number; label: string; text: string | null } | null;
  stale: boolean;
  note: string | null;
  actions: ScenesAction[];
}

export interface ScenesSectionPresenterProps {
  view: ScenesView;
  panelId: string;
  rows: ClipTimestampItem[];
  length: number;
  point: number;
  playingTime: number;
  hasPlayback: boolean;
  browse: number | null;
  requestedTimes: number[];
  failedTimes: ReadonlySet<number>;
  pictureRatio: number | null;
  isLoft: boolean;
  visual: ScenesVisualView | null;
  awaitingVisual: boolean;
  error: string | null;
  working: boolean;
  pictureUrl: (time: number) => string;
  rulerRef: RefObject<HTMLDivElement | null>;
  rulerWidth: number;
  onToggle: () => void;
  onRulerClick: (time: number, pointerType: string | null) => void;
  onRulerPointerDown: (time: number, pointerType: string, clientX: number) => void;
  onRulerPointerMove: (time: number, pointerType: string, clientX: number) => void;
  onRulerPointerUp: (time: number, pointerType: string, clientX: number) => void;
  onRulerPointerCancel: () => void;
  onSectionPointerLeave: (pointerType: string) => void;
  onSectionBlur: (event: FocusEvent<HTMLDivElement>) => void;
  onSeek: (time: number) => void;
  onAction: (kind: ScenesActionKind) => void;
  onPictureLoad: (width: number, height: number) => void;
  onPictureError: (time: number) => void;
}

function percent(fraction: number): string {
  return `${(fraction * 100).toFixed(3)}%`;
}

export default function ScenesSectionPresenter({
  view,
  panelId,
  rows,
  length,
  point,
  playingTime,
  hasPlayback,
  browse,
  requestedTimes,
  failedTimes,
  pictureRatio,
  isLoft,
  visual,
  awaitingVisual,
  error,
  working,
  pictureUrl,
  rulerRef,
  rulerWidth,
  onToggle,
  onRulerClick,
  onRulerPointerDown,
  onRulerPointerMove,
  onRulerPointerUp,
  onRulerPointerCancel,
  onSectionPointerLeave,
  onSectionBlur,
  onSeek,
  onAction,
  onPictureLoad,
  onPictureError,
}: ScenesSectionPresenterProps) {
  const t = useTranslations("file");

  const index = frameIndexAt(rows, point);
  const windowStart = windowStartFor(index, rows.length);
  const windowRows = rows.slice(windowStart, windowStart + 9);
  const playingIndex = frameIndexAt(rows, playingTime);
  const browseIndex = browse === null ? null : frameIndexAt(rows, browse);
  const requested = new Set(requestedTimes);
  const expanded = view !== "collapsed";
  const ready = view === "expanded";
  const aspectRatio = pictureRatio ?? 16 / 9;

  const timeAt = (clientX: number, target: HTMLElement): number => {
    const rect = target.getBoundingClientRect();
    const width = rect.width > 0 ? rect.width : 1;
    const fraction = Math.min(1, Math.max(0, (clientX - rect.left) / width));
    return fraction * length;
  };

  const handlePointer = (event: ReactPointerEvent<HTMLDivElement>) => {
    const time = timeAt(event.clientX, event.currentTarget);
    switch (event.type) {
      case "pointerdown":
        onRulerPointerDown(time, event.pointerType, event.clientX);
        break;
      case "pointermove":
        onRulerPointerMove(time, event.pointerType, event.clientX);
        break;
      case "pointerup":
        onRulerPointerUp(time, event.pointerType, event.clientX);
        break;
      case "pointercancel":
        onRulerPointerCancel();
        break;
      default:
        break;
    }
  };

  const handleClick = (event: MouseEvent<HTMLDivElement>) => {
    onRulerClick(
      timeAt(event.clientX, event.currentTarget),
      (event.nativeEvent as PointerEvent).pointerType || null,
    );
  };

  return (
    <div
      className="flex w-full max-w-96 flex-col gap-3.5"
      onPointerLeave={(event) => {
        if (event.target === event.currentTarget) {
          onSectionPointerLeave(event.pointerType);
        }
      }}
      onPointerOut={(event) => {
        const next = event.relatedTarget as Node | null;
        if (next && !event.currentTarget.contains(next)) {
          onSectionPointerLeave(event.pointerType);
        }
      }}
      onBlur={onSectionBlur}
    >
      <button
        type="button"
        onClick={onToggle}
        aria-expanded={expanded}
        aria-controls={panelId}
        className="flex w-full cursor-pointer items-center gap-2 text-sm text-text-muted transition-colors hover:text-text-primary"
      >
        <Film size={14} aria-hidden="true" />
        <span>{t("scenesTitle")}</span>
        {expanded && rows.length > 0 && (
          <span className="tabular-nums">{rows.length}</span>
        )}
        {ready && visual?.count != null && (
          <span
            aria-label={t("scenesCount", { count: visual.count })}
            className="inline-flex items-center gap-1 tabular-nums"
          >
            <span
              aria-hidden="true"
              className="h-1.5 w-1.5 rounded-full bg-accent-teal"
            />
            {visual.count}
          </span>
        )}
        {ready && visual?.progress && (
          <span
            aria-label={
              visual.progress.updating
                ? t("scenesUpdating", { progress: visual.progress.text })
                : t("scenesProcessing", { progress: visual.progress.text })
            }
            className="tabular-nums"
          >
            {visual.progress.text}
          </span>
        )}
        {ready && visual?.spinner && (
          <Loader2
            size={12}
            aria-hidden="true"
            className="animate-spin motion-reduce:animate-none"
          />
        )}
        <span className="ml-auto" aria-hidden="true">
          {expanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </span>
      </button>

      {expanded && (
        <div id={panelId} className="flex flex-col gap-3.5">
          <Ruler
            rows={rows}
            length={length}
            point={point}
            playingTime={playingTime}
            hasPlayback={hasPlayback}
            browse={browse}
            marks={view === "expanded" ? visual?.marks : undefined}
            disabled={view === "loading"}
            rulerRef={rulerRef}
            rulerWidth={rulerWidth}
            onPointer={handlePointer}
            onClick={handleClick}
          />

          {view === "loading" ? (
            <LoadingSkeleton />
          ) : (
          <>
          <div className="grid grid-cols-3 gap-x-2 gap-y-3">
            {windowRows.map((row, offset) => {
              const absoluteIndex = windowStart + offset;
              const isPlaying = hasPlayback && absoluteIndex === playingIndex;
              const isBrowsed =
                !isPlaying && browseIndex !== null && absoluteIndex === browseIndex;
              const hasDot =
                visual != null &&
                visual.dots.some((time) => Math.abs(time - row.start) <= 0.05);
              const failed = failedTimes.has(row.start);
              const outline = isPlaying
                ? "outline-2 outline-offset-2 outline-solid outline-accent"
                : isBrowsed
                  ? "outline-2 outline-offset-2 outline-solid outline-text-primary"
                  : "";
              return (
                <button
                  key={row.start}
                  type="button"
                  aria-label={t("scenesPlayFrom", {
                    time: formatDuration(row.start),
                  })}
                  onClick={() => onSeek(row.start)}
                  className="group relative flex min-w-0 cursor-pointer flex-col gap-1"
                >
                  <span
                    className={`relative block w-full overflow-hidden rounded-lg bg-bg-elevated ${outline}`}
                    style={{ aspectRatio }}
                  >
                    {isLoft && failed ? null : requested.has(row.start) ? (
                      <img
                        src={pictureUrl(row.start)}
                        alt=""
                        loading="lazy"
                        onLoad={(event) =>
                          onPictureLoad(
                            event.currentTarget.naturalWidth,
                            event.currentTarget.naturalHeight,
                          )
                        }
                        onError={
                          isLoft ? () => onPictureError(row.start) : undefined
                        }
                        className="h-full w-full object-contain"
                      />
                    ) : null}
                    {hasDot && (
                      <span
                        aria-hidden="true"
                        className="absolute left-1.5 top-1.5 h-2 w-2 rounded-full bg-accent-teal ring-2 ring-bg-primary/80"
                      />
                    )}
                  </span>
                  <span
                    className={`text-center text-xs tabular-nums ${
                      isPlaying ? "text-text-primary" : "text-text-muted"
                    }`}
                  >
                    {formatDuration(row.start)}
                  </span>
                </button>
              );
            })}
          </div>

          {awaitingVisual && (
            <div aria-hidden="true" className="flex flex-col gap-3">
              <div className="h-[7.75rem] rounded-xl bg-bg-card" />
              <div className="h-9" />
            </div>
          )}

          {visual?.scene && (
            <SceneCard scene={visual.scene} onSeek={onSeek} />
          )}

          {visual?.stale && (
            <p className="text-xs text-accent-amber">{t("visualIndexStale")}</p>
          )}

          {(visual?.note != null ||
            error != null ||
            (visual?.actions.length ?? 0) > 0) && (
            <div className="flex min-h-9 items-center justify-between gap-2">
              <span className="text-xs tabular-nums text-text-muted">
                {error ?? visual?.note ?? ""}
              </span>
              <div className="flex items-center gap-2">
                {visual?.actions.map((action) => (
                  <button
                    key={action.kind}
                    type="button"
                    onClick={() => onAction(action.kind)}
                    disabled={working}
                    className={`cursor-pointer rounded-2xl transition-colors disabled:cursor-not-allowed disabled:bg-transparent disabled:text-warm-silver ${
                      action.filled
                        ? "bg-bg-elevated px-3.5 py-2 text-[13px] text-text-primary enabled:hover:bg-sand"
                        : "px-2.5 py-1.5 text-xs text-text-muted enabled:hover:bg-(--bg-elevated) enabled:hover:text-text-primary"
                    }`}
                  >
                    {action.label}
                  </button>
                ))}
              </div>
            </div>
          )}
          </>
          )}
        </div>
      )}
    </div>
  );
}

function markTone(scene: VideoVisualSceneItem): { dot: string; line: string } {
  if (scene.status === "succeeded") {
    return { dot: "bg-accent-teal", line: "bg-accent-teal" };
  }
  if (scene.status === "failed") {
    return {
      dot: "border-2 border-danger bg-transparent",
      line: "bg-danger/45",
    };
  }
  return {
    dot: "border-2 border-accent-teal bg-transparent",
    line: "bg-accent-teal/35",
  };
}

interface RulerProps {
  rows: ClipTimestampItem[];
  length: number;
  point: number;
  playingTime: number;
  hasPlayback: boolean;
  browse: number | null;
  marks: VideoVisualSceneItem[] | undefined;
  disabled: boolean;
  rulerRef: RefObject<HTMLDivElement | null>;
  rulerWidth: number;
  onPointer: (event: ReactPointerEvent<HTMLDivElement>) => void;
  onClick: (event: MouseEvent<HTMLDivElement>) => void;
}

function Ruler({
  rows,
  length,
  point,
  playingTime,
  hasPlayback,
  browse,
  marks,
  disabled,
  rulerRef,
  rulerWidth,
  onPointer,
  onClick,
}: RulerProps) {
  const t = useTranslations("file");
  const index = frameIndexAt(rows, point);
  const windowStart = windowStartFor(index, rows.length);
  const windowRows = rows.slice(windowStart, windowStart + 9);
  const nextRow = rows[windowStart + windowRows.length];
  const windowLeft = windowRows.length > 0
    ? rulerFraction(windowRows[0].start, length)
    : 0;
  const windowRight = nextRow ? rulerFraction(nextRow.start, length) : 1;
  const ticks = rows
    .map((row) => rulerFraction(row.start, length))
    .map((fraction) => `M${(fraction * 100).toFixed(3)} 16V36`)
    .join("");
  const showLabels = length > 0;

  return (
    <div className="flex flex-col gap-1.5">
      <div
        ref={rulerRef}
        role="slider"
        aria-label={t("scenesRuler")}
        aria-valuemin={0}
        aria-valuemax={Math.round(length)}
        aria-valuenow={Math.round(Math.min(point, length))}
        aria-valuetext={formatDuration(point)}
        aria-disabled={disabled || undefined}
        tabIndex={disabled ? -1 : 0}
        onPointerDown={disabled ? undefined : onPointer}
        onPointerMove={disabled ? undefined : onPointer}
        onPointerUp={disabled ? undefined : onPointer}
        onPointerCancel={disabled ? undefined : onPointer}
        onClick={disabled ? undefined : onClick}
        className="relative h-11 w-full cursor-pointer touch-pan-y overflow-hidden rounded-lg bg-bg-elevated"
      >
        {!disabled && (
          <div
            aria-hidden="true"
            className="absolute inset-y-0 bg-text-primary/10"
            style={{
              left: percent(windowLeft),
              width: percent(Math.max(0, windowRight - windowLeft)),
            }}
          />
        )}
        <svg
          aria-hidden="true"
          className="pointer-events-none absolute inset-0 h-full w-full text-text-muted/45"
          viewBox="0 0 100 44"
          preserveAspectRatio="none"
        >
          <path
            d={ticks}
            stroke="currentColor"
            strokeWidth={1}
            vectorEffect="non-scaling-stroke"
          />
        </svg>

        {marks?.map((scene) => {
          const left = percent(rulerFraction(scene.start_time, length));
          const tone = markTone(scene);
          return (
            <span key={`mark-${scene.ordering}`} aria-hidden="true">
              <span
                className={`absolute top-[5px] -ml-[3px] box-border h-[7px] w-[7px] rounded-full ${tone.dot}`}
                style={{ left }}
              />
              <span
                className={`absolute bottom-1.5 top-3.5 -ml-px w-0.5 ${tone.line}`}
                style={{ left }}
              />
            </span>
          );
        })}

        {browse !== null && (
          <span
            aria-hidden="true"
            className="absolute inset-y-0 -ml-px w-px bg-text-primary"
            style={{ left: percent(rulerFraction(browse, length)) }}
          />
        )}
        {hasPlayback && !disabled && (
          <span
            aria-hidden="true"
            className="absolute inset-y-0 -ml-px w-0.5 bg-accent"
            style={{ left: percent(rulerFraction(playingTime, length)) }}
          />
        )}
      </div>

      {showLabels && (
        <div className="relative h-3.5 text-[11px] tabular-nums text-text-muted">
          <span className="absolute left-0">0:00</span>
          {timeLabels(length, rulerWidth, estimateLabelWidth).map((label) => (
            <span
              key={label.time}
              className="absolute -translate-x-1/2"
              style={{ left: percent(label.fraction) }}
            >
              {formatDuration(label.time)}
            </span>
          ))}
          <span className="absolute right-0">{formatDuration(length)}</span>
        </div>
      )}
    </div>
  );
}

function SceneCard({
  scene,
  onSeek,
}: {
  scene: { time: number; label: string; text: string | null };
  onSeek: (time: number) => void;
}) {
  return (
    <button
      type="button"
      onClick={() => onSeek(scene.time)}
      className="flex w-full cursor-pointer flex-col gap-1 rounded-xl bg-bg-elevated px-3 py-2.5 text-left transition-colors hover:bg-sand"
    >
      <span className="flex items-center gap-1.5 text-xs tabular-nums text-text-muted">
        <span
          aria-hidden="true"
          className="h-1.5 w-1.5 rounded-full bg-accent-teal"
        />
        {formatDuration(scene.time)}
      </span>
      <span className="line-clamp-2 min-h-[2.4rem] text-sm leading-snug text-text-primary">
        {scene.label}
      </span>
      <span className="line-clamp-2 min-h-[2.45rem] whitespace-pre-line text-xs leading-relaxed text-text-muted">
        {scene.text ?? ""}
      </span>
    </button>
  );
}

function LoadingSkeleton() {
  return (
    <div className="flex flex-col gap-3.5">
      <div className="h-3.5" />
      <div className="grid grid-cols-3 gap-x-2 gap-y-3">
        {Array.from({ length: 9 }, (_, index) => (
          <div key={index} className="flex flex-col gap-1">
            <div
              className="w-full rounded-lg bg-bg-elevated"
              style={{ aspectRatio: 4 / 3 }}
            />
            <div className="mx-auto h-3.5 w-2/5 rounded bg-bg-elevated" />
          </div>
        ))}
      </div>
    </div>
  );
}
