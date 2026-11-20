"use client";

import {
  forwardRef,
  memo,
  type PointerEvent as ReactPointerEvent,
  type RefObject,
} from "react";
import { ChevronDown, ChevronUp, Film, Loader2 } from "lucide-react";
import { useTranslations } from "next-intl";

import { formatDuration } from "@/lib/format";
import type { ClipTimestampItem, VideoVisualSceneItem } from "../api";
import {
  estimateLabelWidth,
  frameIndexAt,
  rulerFraction,
  seekTime,
  timeLabels,
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
  labelledByHost?: boolean;
  panelId: string;
  rows: ClipTimestampItem[];
  length: number;
  point: number;
  playingTime: number;
  hasPlayback: boolean;
  browse: number | null;
  hover: number | null;
  requestedTimes: number[];
  columns: number;
  flow: boolean;
  oneLine: boolean;
  /** The scroller's bottom padding, which a pinned card has to cover. */
  cardBottomInset: number;
  listMinHeight: number;
  band: { left: number; right: number };
  failedTimes: ReadonlySet<number>;
  pictureRatio: number | null;
  visual: ScenesVisualView | null;
  awaitingVisual: boolean;
  error: string | null;
  working: boolean;
  showChip: boolean;
  pictureUrl: (time: number) => string;
  sectionRef: RefObject<HTMLDivElement | null>;
  headerRef: RefObject<HTMLButtonElement | null>;
  listRef: RefObject<HTMLDivElement | null>;
  listBoxRef: RefObject<HTMLDivElement | null>;
  rulerBoxRef: RefObject<HTMLDivElement | null>;
  cardBoxRef: RefObject<HTMLDivElement | null>;
  gridRef: RefObject<HTMLDivElement | null>;
  tileRef: RefObject<HTMLButtonElement | null>;
  rulerRef: RefObject<HTMLDivElement | null>;
  rulerWidth: number;
  onToggle: () => void;
  onRulerHover: (time: number | null) => void;
  onRulerPointerDown: (
    time: number,
    pointerType: string,
    clientX: number,
    button: number,
  ) => void;
  onRulerPointerMove: (
    time: number,
    pointerType: string,
    clientX: number,
    buttons: number,
  ) => void;
  onRulerPointerUp: (time: number, pointerType: string) => void;
  onRulerPointerCancel: () => void;
  onResume: () => void;
  onSeek: (time: number) => void;
  onAction: (kind: ScenesActionKind) => void;
  onPictureLoad: (width: number, height: number) => void;
  onPictureError: (time: number) => void;
}

function percent(fraction: number): string {
  return `${(fraction * 100).toFixed(3)}%`;
}

/** The box a pinned part paints so tiles do not show through it. */
function ground(flow: boolean): string {
  return flow ? "bg-bg-card" : "bg-bg-elevated";
}

export default function ScenesSectionPresenter({
  view,
  labelledByHost = false,
  panelId,
  rows,
  length,
  point,
  playingTime,
  hasPlayback,
  browse,
  hover,
  requestedTimes,
  columns,
  flow,
  oneLine,
  cardBottomInset,
  listMinHeight,
  band,
  failedTimes,
  pictureRatio,
  visual,
  awaitingVisual,
  error,
  working,
  showChip,
  pictureUrl,
  sectionRef,
  headerRef,
  listRef,
  listBoxRef,
  rulerBoxRef,
  cardBoxRef,
  gridRef,
  tileRef,
  rulerRef,
  rulerWidth,
  onToggle,
  onRulerHover,
  onRulerPointerDown,
  onRulerPointerMove,
  onRulerPointerUp,
  onRulerPointerCancel,
  onResume,
  onSeek,
  onAction,
  onPictureLoad,
  onPictureError,
}: ScenesSectionPresenterProps) {
  const t = useTranslations("file");

  const playingIndex = frameIndexAt(rows, playingTime);
  const browseIndex = browse === null ? null : frameIndexAt(rows, browse);
  const requested = new Set(requestedTimes);
  const expanded = view !== "collapsed";
  const ready = view === "expanded";
  const aspectRatio = pictureRatio ?? 16 / 9;
  const actionRowDrawn =
    visual?.note != null ||
    error != null ||
    (visual?.actions.length ?? 0) > 0;
  const cardDrawn = awaitingVisual || visual?.scene != null;

  const timeAt = (clientX: number, target: HTMLElement): number => {
    const rect = target.getBoundingClientRect();
    const width = rect.width > 0 ? rect.width : 1;
    const fraction = Math.min(1, Math.max(0, (clientX - rect.left) / width));
    return fraction * length;
  };

  const handlePointer = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.type === "pointerleave") {
      onRulerHover(null);
    }
    const time = timeAt(event.clientX, event.currentTarget);
    switch (event.type) {
      case "pointerdown":
        onRulerPointerDown(time, event.pointerType, event.clientX, event.button);
        break;
      case "pointermove":
        onRulerPointerMove(
          time,
          event.pointerType,
          event.clientX,
          event.buttons,
        );
        break;
      case "pointerup":
        onRulerPointerUp(time, event.pointerType);
        break;
      case "pointercancel":
        onRulerPointerCancel();
        break;
      default:
        break;
    }
  };

  const tileElements = rows.map((row, index) => {
    const isPlaying = hasPlayback && index === playingIndex;
    const isBrowsed =
      !isPlaying && browseIndex !== null && index === browseIndex;
    return (
      <SceneTile
        key={row.start}
        ref={index === 0 ? tileRef : undefined}
        start={row.start}
        label={t("scenesPlayFrom", { time: formatDuration(row.start) })}
        timeLabel={formatDuration(row.start)}
        requested={requested.has(row.start)}
        failed={failedTimes.has(row.start)}
        playing={isPlaying}
        browsed={isBrowsed}
        dot={visual != null && visual.dots.includes(row.start)}
        aspectRatio={aspectRatio}
        pictureUrl={requested.has(row.start) ? pictureUrl(row.start) : ""}
        onSeek={onSeek}
        onPictureLoad={onPictureLoad}
        onPictureError={onPictureError}
      />
    );
  });

  return (
    <div
      ref={sectionRef}
      className={
        flow
          ? "flex w-full flex-col gap-3.5"
          : "flex min-h-0 w-full flex-1 flex-col gap-3.5"
      }
    >
      {!labelledByHost && (
        <button
          ref={headerRef}
          type="button"
          onClick={onToggle}
          aria-expanded={expanded}
          aria-controls={panelId}
          className="flex w-full cursor-pointer items-center gap-2 text-sm text-text-muted transition-colors hover:text-text-primary"
        >
          <Film size={14} aria-hidden="true" />
          <span>{t("scenesTitle")}</span>{" "}
          {expanded && rows.length > 0 && (
            <span className="tabular-nums">{rows.length}</span>
          )}{" "}
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
      )}

      {expanded && (
        <div
          id={panelId}
          className={
            flow
              ? "flex flex-col gap-3.5"
              : "flex min-h-0 flex-1 flex-col gap-3.5"
          }
        >
          <div
            ref={rulerBoxRef}
            className={`relative flex-shrink-0 ${flow ? `sticky z-[2] pb-3.5 pt-1 ${ground(flow)}` : ""}`}
            style={flow ? { top: "var(--inspector-sticky-top, 0px)" } : undefined}
          >
            <Ruler
              rows={rows}
              length={length}
              point={point}
              playingTime={playingTime}
              hasPlayback={hasPlayback}
              browse={browse}
              hover={hover}
              band={band}
              marks={ready ? visual?.marks : undefined}
              disabled={view === "loading"}
              rulerRef={rulerRef}
              rulerWidth={rulerWidth}
              onPointer={handlePointer}
            />
            {showChip && (
              <div
                className="pointer-events-none absolute inset-x-0 z-[5] flex justify-center"
                style={{ top: "calc(100% + 0.875rem)" }}
              >
                <button
                  type="button"
                  onClick={onResume}
                  className="pointer-events-auto mt-1 h-fit w-fit rounded-full bg-accent px-3 py-1 text-xs font-medium text-white shadow-card hover:bg-accent-hover focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus-ring"
                >
                  {t("scenesCurrentPosition")}
                </button>
              </div>
            )}
          </div>

          <div
            ref={listBoxRef}
            className={
              flow ? "relative" : "relative flex min-h-0 flex-1 flex-col"
            }
            style={flow ? undefined : { minHeight: listMinHeight || undefined }}
          >
            <div
              ref={listRef}
              data-scene-list=""
              className={
                flow
                  ? "px-1 py-1"
                  : "h-full overflow-y-auto overscroll-contain px-1 py-1 [scrollbar-gutter:stable]"
              }
            >
              {view === "loading" ? (
                <LoadingSkeleton columns={columns} />
              ) : (
                <div
                  ref={gridRef}
                  className="grid gap-x-2 gap-y-3"
                  style={{
                    gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))`,
                  }}
                >
                  {tileElements}
                </div>
              )}
            </div>
          </div>

          {view === "expanded" && (
            <>
              <div
                ref={cardBoxRef}
                style={flow && cardDrawn ? { bottom: -cardBottomInset } : undefined}
                className={
                  flow
                    ? cardDrawn
                      ? `sticky z-[2] flex-shrink-0 pt-3.5 pb-[calc(env(safe-area-inset-bottom)+0.5rem)] ${ground(flow)}`
                      : "hidden"
                    : "flex-shrink-0"
                }
              >
                {awaitingVisual && (
                  <div
                    aria-hidden="true"
                    data-testid="scenes-visual-placeholder"
                    style={{
                      height: oneLine
                        ? "calc(2.75rem + 2.25rem)"
                        : "calc(7.75rem + 2.25rem)",
                    }}
                    className="flex flex-col gap-3"
                  >
                    <div
                      className="rounded-xl bg-bg-card"
                      style={{ height: oneLine ? "2.75rem" : "7.75rem" }}
                    />
                    <div className="h-9" />
                  </div>
                )}

                {visual?.scene && (
                  <SceneCard
                    scene={visual.scene}
                    oneLine={oneLine}
                    onSeek={onSeek}
                  />
                )}
              </div>

              {visual?.stale && (
                <p className="flex-shrink-0 text-xs text-accent-amber">
                  {t("visualIndexStale")}
                </p>
              )}

              {actionRowDrawn && (
                <div className="flex min-h-9 flex-shrink-0 items-center justify-between gap-2">
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
  hover: number | null;
  band: { left: number; right: number };
  marks: VideoVisualSceneItem[] | undefined;
  disabled: boolean;
  rulerRef: RefObject<HTMLDivElement | null>;
  rulerWidth: number;
  onPointer: (event: ReactPointerEvent<HTMLDivElement>) => void;
}

function Ruler({
  rows,
  length,
  point,
  playingTime,
  hasPlayback,
  browse,
  hover,
  band,
  marks,
  disabled,
  rulerRef,
  rulerWidth,
  onPointer,
}: RulerProps) {
  const t = useTranslations("file");
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
        onPointerLeave={disabled ? undefined : onPointer}
        className="relative h-11 w-full cursor-pointer touch-pan-y overflow-hidden rounded-lg bg-bg-elevated"
      >
        {!disabled && (
          <div
            aria-hidden="true"
            className="absolute inset-y-0 bg-text-primary/10"
            style={{
              left: percent(band.left),
              width: percent(Math.max(0, band.right - band.left)),
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
          const left = percent(rulerFraction(seekTime(scene), length));
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

        {hover !== null && (
          <span
            aria-hidden="true"
            className="absolute inset-y-0 -ml-px w-px bg-text-primary"
            style={{ left: percent(rulerFraction(hover, length)) }}
          />
        )}
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
  oneLine,
  onSeek,
}: {
  scene: { time: number; label: string; text: string | null };
  oneLine: boolean;
  onSeek: (time: number) => void;
}) {
  const time = formatDuration(scene.time);
  if (oneLine) {
    return (
      <button
        type="button"
        onClick={() => onSeek(scene.time)}
        aria-label={`${time} ${scene.label}`}
        className="flex h-11 w-full cursor-pointer items-center gap-2 rounded-xl bg-bg-elevated px-3 text-left transition-colors hover:bg-sand"
      >
        <span
          aria-hidden="true"
          className="h-1.5 w-1.5 flex-shrink-0 rounded-full bg-accent-teal"
        />
        <span className="flex-shrink-0 text-xs tabular-nums text-text-muted">
          {time}
        </span>
        <span className="min-w-0 truncate text-[13px] text-text-primary">
          {scene.label}
        </span>
      </button>
    );
  }
  return (
    <button
      type="button"
      onClick={() => onSeek(scene.time)}
      aria-label={`${time} ${scene.label}`}
      className="flex min-h-[7.6rem] w-full cursor-pointer flex-col justify-start gap-1 rounded-xl bg-bg-elevated px-3 py-2.5 text-left transition-colors hover:bg-sand"
    >
      <span className="flex items-center gap-1.5 text-xs tabular-nums text-text-muted">
        <span
          aria-hidden="true"
          className="h-1.5 w-1.5 rounded-full bg-accent-teal"
        />
        {time}
      </span>
      <span className="line-clamp-2 text-sm leading-snug text-text-primary">
        {scene.label}
      </span>
      <span className="line-clamp-2 whitespace-pre-line text-xs leading-relaxed text-text-muted">
        {scene.text ?? ""}
      </span>
    </button>
  );
}

interface SceneTileProps {
  start: number;
  label: string;
  timeLabel: string;
  requested: boolean;
  failed: boolean;
  playing: boolean;
  browsed: boolean;
  dot: boolean;
  aspectRatio: number;
  pictureUrl: string;
  onSeek: (time: number) => void;
  onPictureLoad: (width: number, height: number) => void;
  onPictureError: (time: number) => void;
}

/**
 * One tile. Memoised so a clock tick re-renders only the tiles whose playing
 * outline changes, not every frame in the list.
 */
const SceneTile = memo(
  forwardRef<HTMLButtonElement, SceneTileProps>(function SceneTile(
    {
      start,
      label,
      timeLabel,
      requested,
      failed,
      playing,
      browsed,
      dot,
      aspectRatio,
      pictureUrl,
      onSeek,
      onPictureLoad,
      onPictureError,
    },
    ref,
  ) {
    const outline = playing
      ? "outline-2 outline-offset-2 outline-solid outline-accent"
      : browsed
        ? "outline-2 outline-offset-2 outline-solid outline-text-primary"
        : "";
    return (
      <button
        ref={ref}
        type="button"
        data-scene-tile=""
        aria-label={label}
        onClick={() => onSeek(start)}
        className="group relative flex min-w-0 cursor-pointer flex-col gap-1"
      >
        <span
          className={`relative block w-full overflow-hidden rounded-lg bg-bg-elevated ${outline}`}
          style={{ aspectRatio }}
        >
          {failed ? null : requested ? (
            <img
              src={pictureUrl}
              alt=""
              onLoad={(event) =>
                onPictureLoad(
                  event.currentTarget.naturalWidth,
                  event.currentTarget.naturalHeight,
                )
              }
              onError={() => onPictureError(start)}
              className="h-full w-full object-contain"
            />
          ) : null}
          {dot && (
            <span
              aria-hidden="true"
              className="absolute left-1.5 top-1.5 h-2 w-2 rounded-full bg-accent-teal ring-2 ring-bg-primary/80"
            />
          )}
        </span>
        <span
          className={`text-center text-xs tabular-nums ${
            playing ? "text-text-primary" : "text-text-muted"
          }`}
        >
          {timeLabel}
        </span>
      </button>
    );
  }),
);

function LoadingSkeleton({ columns }: { columns: number }) {
  return (
    <div
      className="grid gap-x-2 gap-y-3"
      style={{
        gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))`,
      }}
    >
      {Array.from({ length: 3 * columns }, (_, index) => (
        <div key={index} className="flex flex-col gap-1">
          <div
            className="w-full rounded-lg bg-bg-elevated"
            style={{ aspectRatio: 4 / 3 }}
          />
          <div className="mx-auto h-3.5 w-2/5 rounded bg-bg-elevated" />
        </div>
      ))}
    </div>
  );
}
