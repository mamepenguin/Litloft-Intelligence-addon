"use client";

import type { FocusEvent } from "react";
import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { flushSync } from "react-dom";
import { useTranslations } from "next-intl";

import { usePolicy } from "@/hooks/usePolicy";
import { useWebSocket } from "@/hooks/useWebSocket";
import type { MediaController } from "@/lib/mediaController";
import { normalizeKey } from "@/lib/shortcuts";

import {
  generateVideoVisualIndex,
  getClipTimestamps,
  getFrameUrl,
  getSceneTileUrl,
  getVideoVisualIndex,
  retryVideoVisualIndex,
  type ClipTimestampItem,
  type VideoVisualIndexResponse,
  type VideoVisualSceneItem,
} from "../api";
import {
  type BrowseClearEvent,
  type BrowseSource,
  browseClearedBy,
} from "./browseLifetime";
import {
  LOFT_MIME_TYPE,
  frameIndexAt,
  rulerLength,
  sceneColumns,
  windowStartFor,
} from "./scenesModel";
import { usePlaybackClock } from "./hooks/usePlaybackClock";
import { useRulerKeys } from "./hooks/useRulerKeys";
import { useRulerWidth } from "./hooks/useRulerWidth";
import { useSceneRows } from "./hooks/useSceneRows";
import { useWindowPictures } from "./hooks/useWindowPictures";
import ScenesSectionPresenter, {
  type ScenesAction,
  type ScenesActionKind,
  type ScenesView,
  type ScenesVisualView,
} from "./ScenesSectionPresenter";

const EVENT_PROGRESS = "intelligence.video_visual.progress";
const EVENT_SUCCEEDED = "intelligence.video_visual.succeeded";
const EVENT_PARTIAL = "intelligence.video_visual.partial";
const EVENT_FAILED = "intelligence.video_visual.failed";

const DRAG_THRESHOLD_PX = 8;

const NO_TIMES: number[] = [];

interface ScenesSectionProps {
  fileId: string;
  drive: string;
  fileType?: string;
  mimeType?: string;
  videoRef?: React.RefObject<HTMLVideoElement | null>;
  mediaController?: MediaController | null;
  /**
   * Whether the host is showing this panel. Absent where the file page
   * draws the section itself and the reader opens it with the header.
   */
  active?: boolean;
  /** The host already writes this panel's name above it. */
  labelledByHost?: boolean;
  /**
   * Whether this file has any frames to show. Silence means available,
   * so a native video has to answer before it knows.
   */
  onAvailability?: (available: boolean) => void;
}

type NativeState = "idle" | "loading" | "ready" | "hidden";

type ActionError = "waiting" | "generic" | null;

export default function ScenesSectionContainer({
  fileId,
  drive,
  fileType,
  mimeType,
  videoRef,
  mediaController,
  active,
  labelledByHost = false,
  onAvailability,
}: ScenesSectionProps) {
  const t = useTranslations("file");
  const isVideo = fileType === "video";
  const isLoft = mimeType === LOFT_MIME_TYPE;
  const policy = usePolicy(drive, "intelligence", "video_visual_index");
  const hostDriven = active !== undefined;
  const panel = hostDriven || onAvailability !== undefined;
  const hostShows = active === true;

  const [expanded, setExpanded] = useState(false);
  const [nativeState, setNativeState] = useState<NativeState>("idle");
  const [rows, setRows] = useState<ClipTimestampItem[] | null>(null);
  const [browse, setBrowse] = useState<number | null>(null);
  const [visual, setVisual] = useState<VideoVisualIndexResponse | null>(null);
  const [visualLoaded, setVisualLoaded] = useState(false);
  const [working, setWorking] = useState<ScenesActionKind | null>(null);
  const [actionError, setActionError] = useState<ActionError>(null);
  const [failedTimes, setFailedTimes] = useState<number[]>([]);
  const [pictureRatio, setPictureRatio] = useState<number | null>(null);

  const shown = hostDriven ? hostShows : expanded;

  const onAvailabilityRef = useRef(onAvailability);
  useEffect(() => {
    onAvailabilityRef.current = onAvailability;
  });

  // A `false` hides the panel in the host, which turns `active` false, so an
  // availability request must not be tied to the lifetime of one showing.
  const availabilityGeneration = useRef(0);

  useLayoutEffect(() => {
    availabilityGeneration.current += 1;
    onAvailabilityRef.current?.(false);
    return () => {
      availabilityGeneration.current += 1;
    };
  }, [fileId]);

  const requestGeneration = useRef(0);
  const visualGeneration = useRef(0);
  const visualRequestedFor = useRef(-1);
  const browseRef = useRef<number | null>(null);
  const browseSource = useRef<BrowseSource | null>(null);
  const expandedRef = useRef(false);
  const lastPointerType = useRef("mouse");
  const dragStart = useRef<{
    x: number;
    previous: number | null;
    previousSource: BrowseSource | null;
  } | null>(null);
  const rulerRef = useRef<HTMLDivElement>(null);
  const sectionRef = useRef<HTMLDivElement>(null);
  const headerRef = useRef<HTMLButtonElement>(null);
  const gridRef = useRef<HTMLDivElement>(null);
  const tileRef = useRef<HTMLButtonElement>(null);
  const lastPartRef = useRef<HTMLElement>(null);
  const setLastPart = useCallback((element: HTMLElement | null) => {
    lastPartRef.current = element;
  }, []);

  const setBrowsePoint = useCallback(
    (value: number | null, source: BrowseSource | null = null) => {
      browseRef.current = value;
      browseSource.current = source;
      setBrowse(value);
    },
    [],
  );

  const clearBrowseOn = useCallback(
    (event: BrowseClearEvent) => {
      if (!browseClearedBy(browseSource.current, event)) return;
      setBrowsePoint(null);
    },
    [setBrowsePoint],
  );

  const handleJump = useCallback(() => {
    clearBrowseOn("jump");
  }, [clearBrowseOn]);

  const playback = usePlaybackClock(mediaController ?? null, handleJump);

  const resetVisual = useCallback(() => {
    visualGeneration.current += 1;
    visualRequestedFor.current = -1;
    setVisual(null);
    setVisualLoaded(false);
    setWorking(null);
    setActionError(null);
  }, []);

  // A new file starts collapsed with nothing loaded, and a `.loft` reads its
  // rows on mount instead of on expand.
  useEffect(() => {
    const availability = availabilityGeneration.current;
    requestGeneration.current += 1;
    expandedRef.current = false;
    dragStart.current = null;
    setExpanded(false);
    setNativeState("idle");
    setRows(null);
    clearBrowseOn("filechange");
    setFailedTimes([]);
    setPictureRatio(null);
    resetVisual();
    if (!isVideo) return;
    if (!isLoft && !panel) return;

    void getClipTimestamps(fileId, drive)
      .then((result) => {
        if (availabilityGeneration.current !== availability) return;
        const next = result?.timestamps ?? [];
        if (isLoft) {
          setRows(next);
        } else {
          setRows(next);
          setNativeState(next.length > 0 ? "ready" : "hidden");
        }
        onAvailabilityRef.current?.(next.length > 0);
      })
      .catch(() => {
        if (availabilityGeneration.current !== availability) return;
        onAvailabilityRef.current?.(false);
      });
  }, [fileId, drive, isVideo, isLoft, panel, clearBrowseOn, resetVisual]);

  const loadNative = useCallback(
    (generation: number) => {
      setNativeState("loading");
      setRows(null);
      void getClipTimestamps(fileId, drive)
        .then((result) => {
          if (requestGeneration.current !== generation) return;
          const next = result?.timestamps ?? [];
          // The rows land outside a React event; flush so the placeholder ruler
          // and the nine tiles swap in one commit instead of a scheduled later one.
          flushSync(() => {
            if (next.length === 0) {
              setNativeState("hidden");
              setExpanded(false);
              return;
            }
            setRows(next);
            setNativeState("ready");
          });
          onAvailabilityRef.current?.(next.length > 0);
        })
        .catch(() => {
          if (requestGeneration.current !== generation) return;
          onAvailabilityRef.current?.(false);
        });
    },
    [fileId, drive],
  );

  // A panel turned on long after the page loaded must not draw the ruler from
  // rows that went stale while it was hidden.
  useEffect(() => {
    if (!hostDriven || !hostShows || isLoft || !isVideo) return;
    const generation = (requestGeneration.current += 1);
    loadNative(generation);
  }, [hostDriven, hostShows, isLoft, isVideo, fileId, loadNative]);

  const wasShown = useRef(false);
  useEffect(() => {
    if (!hostDriven) return;
    if (hostShows) {
      wasShown.current = true;
      return;
    }
    if (!wasShown.current) return;
    wasShown.current = false;
    requestGeneration.current += 1;
    clearBrowseOn("collapse");
    resetVisual();
  }, [hostDriven, hostShows, clearBrowseOn, resetVisual]);

  const expand = useCallback(() => {
    const generation = (requestGeneration.current += 1);
    expandedRef.current = true;
    dragStart.current = null;
    setExpanded(true);
    resetVisual();
    if (isLoft) return;

    loadNative(generation);
  }, [isLoft, loadNative, resetVisual]);

  const collapse = useCallback(() => {
    requestGeneration.current += 1;
    clearBrowseOn("collapse");
    expandedRef.current = false;
    dragStart.current = null;
    setExpanded(false);
    resetVisual();
    if (isLoft) return;
    setNativeState("idle");
    setRows(null);
  }, [isLoft, clearBrowseOn, resetVisual]);

  const toggle = useCallback(() => {
    if (expanded) collapse();
    else expand();
  }, [expanded, expand, collapse]);

  const loadVisual = useCallback(
    async (keepOnFailure: boolean) => {
      const generation = (visualGeneration.current += 1);
      setActionError(null);
      const result = await getVideoVisualIndex(fileId, drive);
      if (visualGeneration.current !== generation) return;
      if (result === null) {
        if (!keepOnFailure) setVisual(null);
      } else {
        setVisual(result);
      }
      setVisualLoaded(true);
    },
    [fileId, drive],
  );

  // The first answer for this showing, once the policy has resolved. Policy-off
  // drives never ask for it at all.
  useEffect(() => {
    if (!isVideo || isLoft || !shown) return;
    if (!hostDriven && !expandedRef.current) return;
    if (policy.isLoading || !policy.enabled) return;
    if (visualRequestedFor.current === requestGeneration.current) return;
    visualRequestedFor.current = requestGeneration.current;
    void loadVisual(false);
  }, [
    isVideo,
    isLoft,
    shown,
    hostDriven,
    policy.isLoading,
    policy.enabled,
    loadVisual,
  ]);

  const progressEvent = useWebSocket(EVENT_PROGRESS);
  const succeededEvent = useWebSocket(EVENT_SUCCEEDED);
  const partialEvent = useWebSocket(EVENT_PARTIAL);
  const failedEvent = useWebSocket(EVENT_FAILED);

  useEffect(() => {
    if (!isVideo || isLoft || !shown) return;
    if (!hostDriven && !expandedRef.current) return;
    if (policy.isLoading || !policy.enabled) return;
    const event = progressEvent ?? succeededEvent ?? partialEvent ?? failedEvent;
    if (!event) return;
    if ((event.data as { file_id?: unknown }).file_id !== fileId) return;
    void loadVisual(true);
  }, [
    progressEvent,
    succeededEvent,
    partialEvent,
    failedEvent,
    isVideo,
    isLoft,
    shown,
    hostDriven,
    policy.isLoading,
    policy.enabled,
    fileId,
    loadVisual,
  ]);

  const runAction = useCallback(
    async (kind: ScenesActionKind) => {
      const generation = requestGeneration.current;
      setActionError(null);
      setWorking(kind);
      visualRequestedFor.current = generation;
      try {
        if (kind === "generate") {
          await generateVideoVisualIndex(fileId, drive);
        } else {
          await retryVideoVisualIndex(fileId, drive);
        }
        if (requestGeneration.current !== generation) return;
        await loadVisual(true);
      } catch (error) {
        if (requestGeneration.current !== generation) return;
        const info = (error as { info?: { kind?: string; reason?: string } })
          .info;
        const waiting =
          info?.kind === "not_queued" && info.reason === "waiting_clip";
        setActionError(waiting ? "waiting" : "generic");
      } finally {
        setWorking(null);
      }
    },
    [fileId, drive, loadVisual],
  );

  const activate = useCallback(
    (time: number) => {
      clearBrowseOn("seek");
      if (mediaController) {
        mediaController.seek(time);
        mediaController.play();
        return;
      }
      const video = videoRef?.current;
      if (!video) return;
      video.currentTime = time;
      const started = video.play?.();
      if (started && typeof started.catch === "function") {
        void started.catch(() => {});
      }
    },
    [mediaController, videoRef, clearBrowseOn],
  );

  const handleRulerClick = useCallback(
    (time: number, pointerType: string | null) => {
      if ((pointerType ?? lastPointerType.current) !== "mouse") return;
      activate(time);
    },
    [activate],
  );

  const handleRulerKeyDown = useCallback(
    (event: KeyboardEvent) => {
      const list = rows ?? [];
      const key = normalizeKey(event);
      if (
        key !== "arrowleft" &&
        key !== "arrowright" &&
        key !== "home" &&
        key !== "end" &&
        key !== "enter"
      ) {
        return;
      }
      event.preventDefault();
      event.stopPropagation();
      if (list.length === 0) return;

      if (key === "enter") {
        if (browseRef.current !== null) activate(browseRef.current);
        return;
      }

      const playingIndex = frameIndexAt(list, playback.time);
      const currentIndex =
        browseRef.current === null
          ? playingIndex
          : frameIndexAt(list, browseRef.current);
      let nextIndex = currentIndex;
      if (key === "home") nextIndex = 0;
      else if (key === "end") nextIndex = list.length - 1;
      else if (key === "arrowleft") nextIndex = Math.max(0, currentIndex - 1);
      else nextIndex = Math.min(list.length - 1, currentIndex + 1);
      setBrowsePoint(list[nextIndex].start, "keyboard");
    },
    [rows, playback.time, activate, setBrowsePoint],
  );

  const handlePointerDown = useCallback(
    (_time: number, pointerType: string, clientX: number) => {
      lastPointerType.current = pointerType;
      if (pointerType === "mouse") {
        dragStart.current = null;
        return;
      }
      dragStart.current = {
        x: clientX,
        previous: browseRef.current,
        previousSource: browseSource.current,
      };
    },
    [],
  );

  const handlePointerMove = useCallback(
    (time: number, pointerType: string, clientX: number) => {
      if (pointerType === "mouse") {
        setBrowsePoint(time, "mouse");
        return;
      }
      const start = dragStart.current;
      if (!start) return;
      if (Math.abs(clientX - start.x) >= DRAG_THRESHOLD_PX) {
        setBrowsePoint(time, "touch");
      }
    },
    [setBrowsePoint],
  );

  const handlePointerUp = useCallback(
    (time: number, pointerType: string) => {
      if (pointerType === "mouse") return;
      if (dragStart.current) setBrowsePoint(time, "touch");
      dragStart.current = null;
    },
    [setBrowsePoint],
  );

  const handlePointerCancel = useCallback(() => {
    if (dragStart.current) {
      setBrowsePoint(
        dragStart.current.previous,
        dragStart.current.previousSource,
      );
    }
    dragStart.current = null;
  }, [setBrowsePoint]);

  const handleSectionPointerLeave = useCallback(() => {
    clearBrowseOn("pointerleave");
  }, [clearBrowseOn]);

  const handleSectionBlur = useCallback(
    (event: FocusEvent<HTMLDivElement>) => {
      const next = event.relatedTarget as Node | null;
      if (next && event.currentTarget.contains(next)) return;
      clearBrowseOn("focusleave");
    },
    [clearBrowseOn],
  );

  const handlePictureLoad = useCallback((width: number, height: number) => {
    if (width <= 0 || height <= 0) return;
    setPictureRatio((current) => current ?? width / height);
  }, []);

  const handlePictureError = useCallback((time: number) => {
    setFailedTimes((current) =>
      current.includes(time) ? current : [...current, time],
    );
  }, []);

  const point = browse ?? playback.time;
  const list = rows ?? [];
  const length = rulerLength(list, playback.duration);

  const view: ScenesView = !shown
    ? "collapsed"
    : isLoft || nativeState === "ready"
      ? "expanded"
      : "loading";

  const rulerWidth = useRulerWidth(rulerRef, view !== "collapsed");
  const {
    rows: rowCount,
    fullWidthColumns,
    scrollbarColumns,
    hasScrollbar,
  } = useSceneRows(
    {
      root: sectionRef,
      header: headerRef,
      ruler: rulerRef,
      grid: gridRef,
      tile: tileRef,
      lastPart: lastPartRef,
    },
    view === "expanded",
  );
  // Beside a scrollbar at 3 rows the grid keeps the narrower width, as it
  // always has. Everywhere else it draws the columns the rows were measured
  // at, so a scrollbar cannot redraw wider, taller tiles under a row count
  // chosen without one.
  const columns = useMemo(() => {
    if (rowCount <= 3 && hasScrollbar && scrollbarColumns > 0) {
      return scrollbarColumns;
    }
    return fullWidthColumns > 0 ? fullWidthColumns : sceneColumns(rulerWidth);
  }, [rowCount, hasScrollbar, scrollbarColumns, fullWidthColumns, rulerWidth]);

  // The one window of this render: the pictures requested, the drawn grid and
  // the ruler's band all read it, so they cannot disagree about which frames
  // are on screen.
  const windowStart = useMemo(
    () =>
      windowStartFor(
        frameIndexAt(list, point),
        list.length,
        columns,
        rowCount,
      ),
    [list, point, columns, rowCount],
  );

  const windowTimes = useMemo(() => {
    if (list.length === 0) return NO_TIMES;
    return list
      .slice(windowStart, windowStart + rowCount * columns)
      .map((row) => row.start);
  }, [list, windowStart, columns, rowCount]);

  const requestedTimes = useWindowPictures(windowTimes, view === "expanded");

  useRulerKeys(rulerRef, handleRulerKeyDown, view === "expanded");

  const visualView = buildVisualView(
    visual,
    visualLoaded,
    point,
    working,
    t,
  );

  const awaitingVisual =
    shown && !isLoft && !visualLoaded && policy.enabled;
  const errorMessage =
    actionError === "waiting"
      ? t("visualIndexWaitingPrerequisite")
      : actionError === "generic"
        ? t("visualIndexActionError")
        : null;

  const pictureUrl = useCallback(
    (time: number) =>
      isLoft ? getSceneTileUrl(fileId, time) : getFrameUrl(fileId, time),
    [isLoft, fileId],
  );

  if (!isVideo) return null;
  if (isLoft && list.length === 0) return null;
  if (!isLoft && nativeState === "hidden") return null;

  return (
    <ScenesSectionPresenter
      view={view}
      labelledByHost={labelledByHost}
      panelId={`scenes-${fileId}`}
      rows={list}
      length={length}
      point={point}
      playingTime={playback.time}
      hasPlayback={mediaController != null}
      browse={browse}
      requestedTimes={requestedTimes}
      columns={columns}
      rowCount={rowCount}
      windowStart={windowStart}
      failedTimes={new Set(failedTimes)}
      pictureRatio={pictureRatio}
      isLoft={isLoft}
      visual={visualView}
      awaitingVisual={awaitingVisual}
      error={errorMessage}
      working={working !== null}
      pictureUrl={pictureUrl}
      sectionRef={sectionRef}
      headerRef={headerRef}
      gridRef={gridRef}
      tileRef={tileRef}
      lastPartRef={setLastPart}
      rulerRef={rulerRef}
      rulerWidth={rulerWidth}
      onToggle={toggle}
      onRulerClick={handleRulerClick}
      onRulerPointerDown={handlePointerDown}
      onRulerPointerMove={handlePointerMove}
      onRulerPointerUp={handlePointerUp}
      onRulerPointerCancel={handlePointerCancel}
      onSectionPointerLeave={handleSectionPointerLeave}
      onSectionBlur={handleSectionBlur}
      onSeek={activate}
      onAction={runAction}
      onPictureLoad={handlePictureLoad}
      onPictureError={handlePictureError}
    />
  );
}

type Translate = (
  key: string,
  values?: Record<string, string | number | Date>,
) => string;

function buildVisualView(
  visual: VideoVisualIndexResponse | null,
  loaded: boolean,
  point: number,
  working: ScenesActionKind | null,
  t: Translate,
): ScenesVisualView | null {
  if (!loaded || !visual || !visual.eligible || !visual.available) return null;

  const active = visual.active_run;
  const staged = visual.staged_run;
  const stagedInFlight =
    staged !== null && (staged.status === "queued" || staged.status === "running");
  const stagedRetryable =
    staged !== null &&
    (staged.status === "partial" || staged.status === "failed");
  const activePartial = active !== null && active.status === "partial";
  const failedCount = staged && stagedRetryable
    ? staged.failed_count
    : activePartial && active
      ? active.failed_count
      : 0;

  let row = 5;
  if (stagedInFlight && !active) row = 1;
  else if (stagedInFlight && active) row = 2;
  else if (failedCount > 0) row = 3;
  else if (active) row = 4;

  const labelled = visual.scenes
    .filter((scene) => scene.status === "succeeded" && scene.scene_label)
    .sort((a, b) => a.start_time - b.start_time);
  let chosen: VideoVisualSceneItem | null = labelled[0] ?? null;
  for (const scene of labelled) {
    if (scene.start_time <= point) chosen = scene;
  }

  const marks = row >= 2 ? visual.scenes : [];
  const dots = row >= 2
    ? visual.scenes
        .filter((scene) => scene.status === "succeeded")
        .map((scene) => scene.start_time)
    : [];

  const progressText = staged
    ? staged.selected_count > 0
      ? `${staged.completed_count} / ${staged.selected_count}`
      : "…"
    : null;

  const actionLabel = (kind: ScenesActionKind, label: string) =>
    working === kind ? t("visualIndexGenerating") : label;

  const actions: ScenesAction[] = [];
  if (row === 3) {
    actions.push({
      kind: "retry",
      label: actionLabel("retry", t("scenesRetry", { count: failedCount })),
      filled: true,
    });
    actions.push({
      kind: "generate",
      label: actionLabel(
        "generate",
        active ? t("scenesRegenerate") : t("scenesCreate"),
      ),
      filled: !active,
    });
  } else if (row === 4) {
    actions.push({
      kind: "generate",
      label: actionLabel("generate", t("scenesRegenerate")),
      filled: visual.stale,
    });
  } else if (row === 5) {
    actions.push({
      kind: "generate",
      label: actionLabel("generate", t("scenesCreate")),
      filled: true,
    });
  }

  return {
    count: row >= 2 && active ? active.succeeded_count : null,
    progress:
      (row === 1 || row === 2) && progressText
        ? { text: progressText, updating: active !== null }
        : null,
    spinner: row === 1 || row === 2,
    marks,
    dots,
    scene:
      row >= 2 && chosen
        ? {
            time: chosen.start_time,
            label: chosen.scene_label ?? "",
            text: chosen.visible_text,
          }
        : null,
    stale: (row === 3 || row === 4) && visual.stale,
    note: row === 3 ? t("scenesFailedNote", { count: failedCount }) : null,
    actions,
  };
}
