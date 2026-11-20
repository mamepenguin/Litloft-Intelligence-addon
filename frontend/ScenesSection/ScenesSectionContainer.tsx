"use client";

import {
  useCallback,
  useEffect,
  useLayoutEffect,
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
  LOFT_MIME_TYPE,
  dotTileStarts,
  frameIndexAt,
  rulerFraction,
  rulerLength,
  seekTime,
} from "./scenesModel";
import { scrollerOf } from "./sceneList";
import { usePlaybackClock } from "./hooks/usePlaybackClock";
import { useRulerKeys } from "./hooks/useRulerKeys";
import { useRulerWidth } from "./hooks/useRulerWidth";
import { useSceneList } from "./hooks/useSceneList";
import { useScenePictures } from "./hooks/useScenePictures";
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

const NO_FRAMES: ClipTimestampItem[] = [];

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

interface DragStart {
  pointerType: string;
  x: number;
  dragged: boolean;
  previousBrowse: number | null;
  previousFollowing: boolean;
  previousScroll: number;
  /** A touch drag the browser cancelled, held until its gesture ends. */
  cancelled?: boolean;
}

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
  const [hover, setHover] = useState<number | null>(null);
  const [followTarget, setFollowTarget] = useState<number | null>(null);
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
  // A `.loft`'s Visual Index is requested once per file at mount, not once per
  // showing, so this records the file asked for rather than a request
  // generation.
  const visualRequestedFile = useRef<string | null>(null);
  const browseRef = useRef<number | null>(null);
  const expandedRef = useRef(false);
  const dragStart = useRef<DragStart | null>(null);
  // A mouse is not captured by the ruler, so a drag released off it never
  // reaches the ruler's pointerup.
  useEffect(() => {
    const endMouseDrag = (event: PointerEvent) => {
      if (event.pointerType === "mouse") dragStart.current = null;
    };
    // A cancelled touch drag still owns the rest of its gesture's touchmoves.
    const endCancelledTouch = () => {
      if (dragStart.current?.cancelled) dragStart.current = null;
    };
    window.addEventListener("pointerup", endMouseDrag);
    window.addEventListener("touchend", endCancelledTouch);
    window.addEventListener("touchcancel", endCancelledTouch);
    return () => {
      window.removeEventListener("pointerup", endMouseDrag);
      window.removeEventListener("touchend", endCancelledTouch);
      window.removeEventListener("touchcancel", endCancelledTouch);
    };
  }, []);
  const rulerRef = useRef<HTMLDivElement>(null);
  const sectionRef = useRef<HTMLDivElement>(null);
  const headerRef = useRef<HTMLButtonElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const listBoxRef = useRef<HTMLDivElement>(null);
  const rulerBoxRef = useRef<HTMLDivElement>(null);
  const cardBoxRef = useRef<HTMLDivElement>(null);
  const gridRef = useRef<HTMLDivElement>(null);
  const tileRef = useRef<HTMLButtonElement>(null);

  const setBrowsePoint = useCallback((value: number | null) => {
    browseRef.current = value;
    setBrowse(value);
  }, []);

  const playback = usePlaybackClock(mediaController ?? null);

  const list = rows ?? NO_FRAMES;
  // With no candidate frames the player's duration is the only length left
  // for the ruler to draw its scene marks and seeks against.
  const length =
    rows === null
      ? 0
      : list.length === 0
        ? positiveDuration(playback.duration)
        : rulerLength(list, playback.duration);

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
    setBrowsePoint(null);
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
  }, [fileId, drive, isVideo, isLoft, panel, setBrowsePoint, resetVisual]);

  const loadNative = useCallback(
    (generation: number) => {
      setNativeState("loading");
      setRows(null);
      void getClipTimestamps(fileId, drive)
        .then((result) => {
          if (requestGeneration.current !== generation) return;
          const next = result?.timestamps ?? [];
          // The rows land outside a React event; flush so the placeholder ruler
          // and every tile swap in one commit instead of a scheduled later one.
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
    setBrowsePoint(null);
    // A hidden `.loft` keeps its Visual Index: showing it again asks nothing,
    // so clearing it here would leave the section blank.
    if (isLoft) return;
    resetVisual();
  }, [hostDriven, hostShows, isLoft, setBrowsePoint, resetVisual]);

  const expand = useCallback(() => {
    const generation = (requestGeneration.current += 1);
    expandedRef.current = true;
    dragStart.current = null;
    setExpanded(true);
    if (isLoft) return;
    resetVisual();

    loadNative(generation);
  }, [isLoft, loadNative, resetVisual]);

  const collapse = useCallback(() => {
    requestGeneration.current += 1;
    setBrowsePoint(null);
    expandedRef.current = false;
    dragStart.current = null;
    setExpanded(false);
    if (isLoft) return;
    resetVisual();
    setNativeState("idle");
    setRows(null);
  }, [isLoft, setBrowsePoint, resetVisual]);

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

  // A `.loft` with no storyboard rows is still a section when its Visual Index
  // has something to draw, and the host hides the panel on `false`.
  useEffect(() => {
    if (!isLoft || rows === null) return;
    onAvailabilityRef.current?.(
      rows.length > 0 ||
        (visual != null && visual.eligible && visual.available),
    );
  }, [isLoft, rows, visual]);

  // The first answer for this showing, once the policy has resolved. Policy-off
  // drives never ask for it at all. A `.loft` asks at mount so its Visual Index
  // is there before the section is opened, and asks once per file.
  useEffect(() => {
    if (!isVideo) return;
    if (policy.isLoading || !policy.enabled) return;
    if (isLoft) {
      if (visualRequestedFile.current === fileId) return;
      visualRequestedFile.current = fileId;
      void loadVisual(false);
      return;
    }
    if (!shown) return;
    if (!hostDriven && !expandedRef.current) return;
    if (visualRequestedFor.current === requestGeneration.current) return;
    visualRequestedFor.current = requestGeneration.current;
    void loadVisual(false);
  }, [
    isVideo,
    isLoft,
    fileId,
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
  // The hook keeps returning its last event on every render, so the request is
  // keyed on the event object: a policy reload must not replay it.
  const appliedEvent = useRef<unknown>(null);

  useEffect(() => {
    const event = progressEvent ?? succeededEvent ?? partialEvent ?? failedEvent;
    if (!event || appliedEvent.current === event) return;
    appliedEvent.current = event;
    if (!isVideo || !shown) return;
    if (!hostDriven && !expandedRef.current) return;
    if (policy.isLoading || !policy.enabled) return;
    if ((event.data as { file_id?: unknown }).file_id !== fileId) return;
    void loadVisual(true);
  }, [
    progressEvent,
    succeededEvent,
    partialEvent,
    failedEvent,
    isVideo,
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

  const onFollowTargetConsumed = useCallback(() => {
    setFollowTarget(null);
  }, []);

  const onReaderTakeOver = useCallback(() => {
    setBrowsePoint(null);
    setFollowTarget(null);
  }, [setBrowsePoint]);

  const awaitingVisual =
    shown && !isLoft && !visualLoaded && policy.enabled;

  const sceneList = useSceneList({
    shown,
    fileId,
    rows: list,
    hasPlayback: mediaController != null,
    layoutSettled: !awaitingVisual,
    playbackTime: playback.time,
    followTarget,
    rulerDrag: dragStart,
    onFollowTargetConsumed,
    onReaderTakeOver,
    refs: {
      section: sectionRef,
      list: listRef,
      rulerBox: rulerBoxRef,
      cardBox: cardBoxRef,
    },
  });
  const {
    following,
    setFollowing,
    columns,
    step,
    flow,
    oneLine,
    cardBottomInset,
    visible,
    opening,
    ready,
    resetKey,
    endOpening,
    placeIndex,
  } = sceneList;

  const activate = useCallback(
    (time: number) => {
      setBrowsePoint(null);
      setFollowTarget(time);
      setFollowing(true);
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
    [mediaController, videoRef, setBrowsePoint, setFollowing],
  );

  const applyRulerPoint = useCallback(
    (time: number) => {
      if (list.length === 0) return;
      const index = frameIndexAt(list, time);
      setBrowsePoint(list[index].start);
      placeIndex(index, true);
      if (visible.canScroll) {
        setFollowing(false);
        setFollowTarget(null);
      }
    },
    [list, placeIndex, setBrowsePoint, setFollowing, visible.canScroll],
  );

  const handlePointerDown = useCallback(
    (time: number, pointerType: string, clientX: number, button: number) => {
      endOpening();
      if (list.length === 0) return;
      const primary = pointerType !== "mouse" || button === 0;
      const scroller = listRef.current ? scrollerOf(listRef.current) : null;
      dragStart.current = primary
        ? {
            pointerType,
            x: clientX,
            dragged: false,
            previousBrowse: browseRef.current,
            previousFollowing: following,
            previousScroll: scroller?.scrollTop ?? 0,
          }
        : null;
      if (pointerType !== "mouse" || button !== 0) return;
      applyRulerPoint(time);
    },
    [applyRulerPoint, endOpening, following, list.length],
  );

  const handlePointerMove = useCallback(
    (time: number, pointerType: string, clientX: number, buttons: number) => {
      if (list.length === 0) return;
      if (pointerType === "mouse") {
        setHover(time);
        const start = dragStart.current;
        if ((buttons & 1) !== 0 && start?.pointerType === "mouse") {
          applyRulerPoint(time);
        }
        return;
      }
      const start = dragStart.current;
      if (!start || start.cancelled) return;
      if (Math.abs(clientX - start.x) >= DRAG_THRESHOLD_PX) {
        start.dragged = true;
        applyRulerPoint(time);
      }
    },
    [applyRulerPoint, list.length],
  );

  const handlePointerUp = useCallback(
    (time: number, pointerType: string) => {
      if (pointerType === "mouse") {
        dragStart.current = null;
        return;
      }
      const start = dragStart.current;
      dragStart.current = null;
      if (!start || start.dragged) return;
      applyRulerPoint(time);
    },
    [applyRulerPoint],
  );

  const handlePointerCancel = useCallback(() => {
    const start = dragStart.current;
    dragStart.current = null;
    if (!start || start.pointerType === "mouse" || !start.dragged) return;
    dragStart.current = { ...start, cancelled: true };
    setBrowsePoint(start.previousBrowse);
    setFollowTarget(null);
    setFollowing(start.previousFollowing);
    const listEl = listRef.current;
    if (listEl) scrollerOf(listEl).scrollTop = start.previousScroll;
  }, [setBrowsePoint, setFollowing]);

  const handleRulerHover = useCallback((time: number | null) => {
    setHover(time);
  }, []);

  const handleRulerKeyDown = useCallback(
    (event: KeyboardEvent) => {
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
      endOpening();
      if (list.length === 0) return;

      if (key === "enter") {
        if (browseRef.current !== null) activate(browseRef.current);
        return;
      }

      let currentIndex: number;
      if (browseRef.current !== null) {
        currentIndex = frameIndexAt(list, browseRef.current);
      } else if (following && mediaController) {
        currentIndex = frameIndexAt(list, playback.time);
      } else {
        currentIndex = Math.min(visible.first * columns, list.length - 1);
      }
      let nextIndex = currentIndex;
      if (key === "home") nextIndex = 0;
      else if (key === "end") nextIndex = list.length - 1;
      else if (key === "arrowleft") nextIndex = Math.max(0, currentIndex - 1);
      else nextIndex = Math.min(list.length - 1, currentIndex + 1);
      setBrowsePoint(list[nextIndex].start);
      placeIndex(nextIndex, true);
      if (visible.canScroll) {
        setFollowing(false);
        setFollowTarget(null);
      }
    },
    [
      list,
      following,
      mediaController,
      playback.time,
      visible.first,
      visible.canScroll,
      columns,
      activate,
      endOpening,
      setBrowsePoint,
      placeIndex,
      setFollowing,
    ],
  );

  const handleResume = useCallback(() => {
    setBrowsePoint(null);
    setFollowTarget(null);
    setFollowing(true);
  }, [setBrowsePoint, setFollowing]);

  // A showing, a file and the frames arriving all start from the top with
  // following set, whatever the reader left behind.
  useLayoutEffect(() => {
    if (!shown) return;
    setBrowsePoint(null);
    setFollowTarget(null);
    setFailedTimes([]);
  }, [shown, fileId, rows, setBrowsePoint]);

  const handlePictureLoad = useCallback((width: number, height: number) => {
    if (width <= 0 || height <= 0) return;
    setPictureRatio((current) => current ?? width / height);
  }, []);

  const handlePictureError = useCallback((time: number) => {
    setFailedTimes((current) =>
      current.includes(time) ? current : [...current, time],
    );
  }, []);

  const count = list.length;
  const topFirstIndex =
    count > 0 ? Math.min(visible.first * columns, count - 1) : -1;
  const followsPlayback = following && mediaController != null;
  const point =
    browse ??
    (followsPlayback
      ? playback.time
      : count > 0 && topFirstIndex >= 0
        ? list[topFirstIndex].start
        : mediaController
          ? playback.time
          : 0);

  const view: ScenesView = !shown
    ? "collapsed"
    : isLoft || nativeState === "ready"
      ? "expanded"
      : "loading";

  const rulerWidth = useRulerWidth(rulerRef, view !== "collapsed");

  const bandFirst =
    count > 0 ? Math.min(visible.first * columns, count - 1) : 0;
  const bandEndIndex = Math.min(
    count,
    (Math.max(visible.first, visible.last) + 1) * columns,
  );
  const band =
    count > 0 && length > 0
      ? {
          left: rulerFraction(list[bandFirst].start, length),
          right:
            bandEndIndex < count
              ? rulerFraction(list[bandEndIndex].start, length)
              : 1,
        }
      : { left: 0, right: 0 };

  const rowTotal = count > 0 ? Math.ceil(count / columns) : 0;
  const pictureFirstRow = visible.measured ? Math.max(0, visible.first - 1) : 0;
  const pictureLastRow = visible.measured
    ? visible.last + 1
    : Math.min(rowTotal - 1, 2);
  const pictureTimes: number[] = [];
  for (let row = pictureFirstRow; row <= pictureLastRow; row += 1) {
    const from = row * columns;
    const to = Math.min(count, from + columns);
    for (let index = from; index < to; index += 1) {
      pictureTimes.push(list[index].start);
    }
  }
  const requestedTimes = useScenePictures({
    times: pictureTimes,
    active: view === "expanded",
    opening,
    ready,
    resetKey,
  });

  useRulerKeys(rulerRef, handleRulerKeyDown, view === "expanded");

  const visualView = buildVisualView(
    visual,
    visualLoaded,
    point,
    list,
    working,
    t,
  );

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
  if (isLoft && list.length === 0 && visualView === null) return null;
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
      hover={hover}
      requestedTimes={requestedTimes}
      columns={columns}
      flow={flow}
      oneLine={oneLine}
      cardBottomInset={cardBottomInset}
      listMinHeight={step}
      band={band}
      failedTimes={new Set(failedTimes)}
      pictureRatio={pictureRatio}
      visual={visualView}
      awaitingVisual={awaitingVisual}
      error={errorMessage}
      working={working !== null}
      showChip={view === "expanded" && !following && mediaController != null && visible.canScroll}
      pictureUrl={pictureUrl}
      sectionRef={sectionRef}
      headerRef={headerRef}
      listRef={listRef}
      listBoxRef={listBoxRef}
      rulerBoxRef={rulerBoxRef}
      cardBoxRef={cardBoxRef}
      gridRef={gridRef}
      tileRef={tileRef}
      rulerRef={rulerRef}
      rulerWidth={rulerWidth}
      onToggle={toggle}
      onRulerHover={handleRulerHover}
      onRulerPointerDown={handlePointerDown}
      onRulerPointerMove={handlePointerMove}
      onRulerPointerUp={handlePointerUp}
      onRulerPointerCancel={handlePointerCancel}
      onResume={handleResume}
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

function positiveDuration(duration: number): number {
  return Number.isFinite(duration) && duration > 0 ? duration : 0;
}

function buildVisualView(
  visual: VideoVisualIndexResponse | null,
  loaded: boolean,
  point: number,
  rows: ClipTimestampItem[],
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
    .sort((a, b) => seekTime(a) - seekTime(b));
  let chosen: VideoVisualSceneItem | null = labelled[0] ?? null;
  for (const scene of labelled) {
    if (seekTime(scene) <= point) chosen = scene;
  }

  const marks = row >= 2 ? visual.scenes : [];
  const dots = row >= 2
    ? dotTileStarts(visual.scenes, rows)
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
            time: seekTime(chosen),
            label: chosen.scene_label ?? "",
            text: chosen.visible_text,
          }
        : null,
    stale: (row === 3 || row === 4) && visual.stale,
    note: row === 3 ? t("scenesFailedNote", { count: failedCount }) : null,
    actions,
  };
}
