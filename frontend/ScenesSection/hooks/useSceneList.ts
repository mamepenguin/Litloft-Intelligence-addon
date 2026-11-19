import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type RefObject,
} from "react";
import { flushSync } from "react-dom";

import { frameIndexAt, sceneColumns } from "../scenesModel";
import type { ClipTimestampItem } from "../../api";
import {
  FULL_CARD_MIN_REM,
  canScroll as canScrollOf,
  contentOffsetTop,
  isFlowForm,
  isWhollyVisible,
  rootRemPx,
  rowAt,
  rowCountOf,
  rowStepOf,
  scrollerOf,
  stickyTopOf,
  tileElements,
  viewportOf,
  visibleRowsOf,
} from "../sceneList";

export interface SceneListGeometry {
  columns: number;
  step: number;
  flow: boolean;
}

export interface SceneListVisible {
  first: number;
  last: number;
  canScroll: boolean;
  measured: boolean;
}

export interface SceneListState {
  following: boolean;
  setFollowing: (value: boolean) => void;
  columns: number;
  step: number;
  flow: boolean;
  oneLine: boolean;
  cardBottomInset: number;
  visible: SceneListVisible;
  scrollVersion: number;
  placeIndex: (index: number, second: boolean) => void;
  placeRow: (row: number, second: boolean) => void;
}

export interface SceneListRefs {
  section: RefObject<HTMLElement | null>;
  list: RefObject<HTMLDivElement | null>;
  rulerBox: RefObject<HTMLDivElement | null>;
  cardBox: RefObject<HTMLDivElement | null>;
  frameStep?: number;
}

export interface SceneListOptions {
  shown: boolean;
  fileId: string;
  rows: ClipTimestampItem[];
  hasPlayback: boolean;
  playbackTime: number;
  followTarget: number | null;
  onFollowTargetConsumed: () => void;
  onReaderTakeOver: () => void;
  refs: SceneListRefs;
}

const INITIAL_VISIBLE: SceneListVisible = {
  first: 0,
  last: 0,
  canScroll: false,
  measured: false,
};

/** A row with at least one tile, so an empty list never claims a row. */
function clampRow(row: number, count: number): number {
  return count > 0 ? Math.max(0, Math.min(row, count - 1)) : 0;
}

function px(value: string): number {
  return Number.parseFloat(value) || 0;
}

/**
 * The list's height, column count and scroll position as one piece of state.
 * Placement reads the same measurements the render uses, so a column change
 * and the scroll that puts a row back land in one commit.
 */
export function useSceneList({
  shown,
  fileId,
  rows,
  hasPlayback,
  playbackTime,
  followTarget,
  onFollowTargetConsumed,
  onReaderTakeOver,
  refs,
}: SceneListOptions): SceneListState {
  const { section, list: listRef, rulerBox, cardBox } = refs;
  const [following, setFollowingState] = useState(true);
  const [columns, setColumns] = useState(3);
  const [step, setStep] = useState(0);
  const [flow, setFlow] = useState(false);
  const [oneLine, setOneLine] = useState(true);
  const [cardBottomInset, setCardBottomInset] = useState(0);
  const [visible, setVisible] = useState<SceneListVisible>(INITIAL_VISIBLE);
  const [scrollVersion, setScrollVersion] = useState(0);
  const [viewportVersion, setViewportVersion] = useState(0);

  const followingRef = useRef(true);
  const columnsRef = useRef(3);
  const stepRef = useRef(0);
  const flowRef = useRef(false);
  const visibleRef = useRef<SceneListVisible>(INITIAL_VISIBLE);
  const topFrameRef = useRef(0);
  const shownRef = useRef(shown);
  const hasPlaybackRef = useRef(hasPlayback);
  const previousColumns = useRef(0);
  const previousStep = useRef(0);

  shownRef.current = shown;
  hasPlaybackRef.current = hasPlayback;

  const setFollowing = useCallback((value: boolean) => {
    followingRef.current = value;
    setFollowingState(value);
  }, []);

  const measure = useCallback((): SceneListGeometry | null => {
    const list = listRef.current;
    if (!list) return null;
    const style = getComputedStyle(list);
    const width =
      list.clientWidth - px(style.paddingLeft) - px(style.paddingRight);
    const nextColumns = width > 0 ? sceneColumns(width) : 3;
    const nextStep = rowStepOf(list, nextColumns);
    const nextFlow = isFlowForm(list);
    if (nextColumns !== columnsRef.current) {
      columnsRef.current = nextColumns;
      setColumns(nextColumns);
    }
    if (nextStep !== stepRef.current) {
      stepRef.current = nextStep;
      setStep(nextStep);
    }
    if (nextFlow !== flowRef.current) {
      flowRef.current = nextFlow;
      setFlow(nextFlow);
    }
    const drawnCard = cardBox.current?.getBoundingClientRect().height ?? 0;
    const fullCardMin = FULL_CARD_MIN_REM * rootRemPx();
    const sparesFullCard =
      list.clientHeight + drawnCard - fullCardMin >= 2 * nextStep;
    const nextOneLine = nextFlow
      ? true
      : nextStep > 0
        ? !sparesFullCard
        : false;
    setOneLine((current) => (current === nextOneLine ? current : nextOneLine));
    const nextInset = nextFlow
      ? px(getComputedStyle(scrollerOf(list)).paddingBottom)
      : 0;
    setCardBottomInset((current) =>
      current === nextInset ? current : nextInset,
    );
    return { columns: nextColumns, step: nextStep, flow: nextFlow };
  }, [cardBox, listRef]);

  const syncVisible = useCallback(() => {
    const list = listRef.current;
    if (!list) return;
    const cols = columnsRef.current;
    const count = rowCountOf(list, cols);
    const view = viewportOf(list, rulerBox.current, cardBox.current);
    const rows = visibleRowsOf(list, cols, view);
    const isMeasured = view.bottom > view.top && count > 0;
    let first = 0;
    let last = 0;
    if (count === 0) {
      first = 0;
      last = 0;
    } else if (rows.length > 0) {
      first = rows[0];
      last = rows[rows.length - 1];
    } else if (isMeasured) {
      first = clampRow(rowAt(list, cols, view.top), count);
      last = first;
    } else {
      first = 0;
      last = Math.min(count - 1, 2);
    }
    const next: SceneListVisible = {
      first,
      last,
      canScroll: canScrollOf(scrollerOf(list)),
      measured: isMeasured,
    };
    const previous = visibleRef.current;
    if (
      previous.first !== next.first ||
      previous.last !== next.last ||
      previous.canScroll !== next.canScroll ||
      previous.measured !== next.measured
    ) {
      visibleRef.current = next;
      setVisible(next);
    }
    topFrameRef.current =
      count > 0 ? Math.min(first, count - 1) * cols : 0;
  }, [cardBox, listRef, rulerBox]);

  const placeListTop = useCallback(() => {
    const list = listRef.current;
    if (!list) return;
    const scroller = scrollerOf(list);
    if (scroller === list) {
      scroller.scrollTop = 0;
      return;
    }
    const host = section.current ?? list;
    const pinned =
      stickyTopOf(host) + (rulerBox.current?.getBoundingClientRect().height ?? 0);
    scroller.scrollTop = Math.max(
      0,
      contentOffsetTop(list, scroller) - pinned,
    );
  }, [listRef, rulerBox, section]);

  const placeRow = useCallback(
    (row: number, second: boolean) => {
      const list = listRef.current;
      if (!list) return;
      const cols = columnsRef.current;
      const tiles = tileElements(list);
      if (tiles.length === 0) return;
      const count = rowCountOf(list, cols);
      const view = viewportOf(list, rulerBox.current, cardBox.current);
      const roomForTwo =
        view.bottom - view.top >= 2 * rowStepOf(list, cols);
      const target =
        second && roomForTwo
          ? clampRow(row, count) - (row > 0 ? 1 : 0)
          : clampRow(row, count);
      if (target <= 0) {
        placeListTop();
        return;
      }
      const tile = tiles[Math.min(target * cols, tiles.length - 1)];
      const rect = tile.getBoundingClientRect();
      const scroller = scrollerOf(list);
      scroller.scrollTop = Math.max(
        0,
        scroller.scrollTop + rect.top - view.top,
      );
    },
    [cardBox, listRef, placeListTop, rulerBox],
  );

  const placeIndex = useCallback(
    (index: number, second: boolean) => {
      placeRow(Math.floor(index / columnsRef.current), second);
    },
    [placeRow],
  );

  const consumeTarget = useCallback(() => {
    if (followTarget === null) return;
    if (Math.abs(playbackTime - followTarget) < 0.75) {
      onFollowTargetConsumed();
    }
  }, [followTarget, onFollowTargetConsumed, playbackTime]);

  // The placement on showing reads the clock without re-running on each tick:
  // a later tick is a placement event of its own.
  const playbackTimeRef = useRef(playbackTime);
  useLayoutEffect(() => {
    playbackTimeRef.current = playbackTime;
  }, [playbackTime]);

  // Showing, a file change and the frames arriving all start at the playing
  // row, or at the list's top when nothing plays. Runs before paint, so the
  // first frame the reader sees is the placed one. A controller appearing is
  // read through a ref: re-running this for it would set following again.
  useLayoutEffect(() => {
    if (!shown) return;
    const geometry = measure();
    if (!geometry) return;
    setFollowing(true);
    if (rows.length === 0) {
      placeListTop();
    } else if (hasPlaybackRef.current) {
      const row = Math.floor(
        frameIndexAt(rows, playbackTimeRef.current) / geometry.columns,
      );
      placeRow(row, true);
    } else {
      placeListTop();
    }
    syncVisible();
  }, [shown, fileId, rows, measure, placeListTop, placeRow, syncVisible, setFollowing]);

  // A column or row-step change keeps the old top row's first frame in the
  // new top row, unless following with a controller: then the playing row
  // wins and the follow effect below places it.
  useLayoutEffect(() => {
    if (!shown) return;
    const geometry = measure();
    if (!geometry) return;
    const columnsChanged =
      previousColumns.current !== 0 &&
      previousColumns.current !== geometry.columns;
    const stepChanged =
      previousStep.current !== 0 &&
      previousStep.current !== geometry.step;
    previousColumns.current = geometry.columns;
    previousStep.current = geometry.step;
    if (
      (columnsChanged || stepChanged) &&
      !(followingRef.current && hasPlayback)
    ) {
      const firstFrame = topFrameRef.current;
      placeRow(Math.floor(firstFrame / geometry.columns), false);
    }
    syncVisible();
  }, [shown, viewportVersion, measure, placeRow, syncVisible, hasPlayback]);

  // A row step or the list's height can also arrive after the first paint.
  useLayoutEffect(() => {
    if (!shown) return;
    measure();
    syncVisible();
  }, [shown, rows, measure, syncVisible]);

  useLayoutEffect(() => {
    if (!shown || !following || !hasPlayback) return;
    if (rows.length === 0) return;
    const list = listRef.current;
    if (!list) return;
    const cols = columnsRef.current;
    const time = followTarget ?? playbackTime;
    const row = Math.floor(frameIndexAt(rows, time) / cols);
    const count = rowCountOf(list, cols);
    if (count === 0) return;
    const tile = tileElements(list)[clampRow(row, count) * cols];
    const view = viewportOf(list, rulerBox.current, cardBox.current);
    if (tile && isWhollyVisible(tile, view)) {
      consumeTarget();
      return;
    }
    placeRow(row, true);
    syncVisible();
    consumeTarget();
  }, [
    shown,
    following,
    hasPlayback,
    playbackTime,
    followTarget,
    rows,
    columns,
    viewportVersion,
    consumeTarget,
    listRef,
    placeRow,
    rulerBox,
    cardBox,
    syncVisible,
  ]);

  // A seek from a tile or the card asks for one placement at the sought time;
  // a player that never reports it must not pin the list forever.
  useEffect(() => {
    if (followTarget === null) return;
    const timer = setTimeout(onFollowTargetConsumed, 1000);
    return () => clearTimeout(timer);
  }, [followTarget, onFollowTargetConsumed]);

  // The reader's own input ends following; a scroll the section makes does
  // not. The scroller is the shared one in flow form, so a swipe that starts
  // on the pinned ruler counts there and nowhere else.
  useEffect(() => {
    if (!shown) return;
    const list = listRef.current;
    if (!list) return;
    const scroller = scrollerOf(list);
    const takeOver = () => {
      if (!shownRef.current || !visibleRef.current.canScroll) return;
      flushSync(() => {
        setFollowing(false);
        onReaderTakeOver();
      });
    };
    const onPointerDown = (event: PointerEvent) => {
      if (!shownRef.current || !visibleRef.current.canScroll) return;
      if (event.button === 1) {
        takeOver();
        return;
      }
      if (event.pointerType === "touch" || event.pointerType === "pen") return;
      if (event.target === scroller) takeOver();
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (!shownRef.current || !visibleRef.current.canScroll) return;
      if (!["PageUp", "PageDown", "Home", "End"].includes(event.key)) return;
      const target = event.target as HTMLElement | null;
      if (target?.closest('[role="slider"]')) return;
      takeOver();
    };
    const onFocusIn = (event: FocusEvent) => {
      if (!shownRef.current || !visibleRef.current.canScroll) return;
      const target = event.target as HTMLElement | null;
      const tile = target?.closest<HTMLElement>("button[data-scene-tile]");
      if (!tile) return;
      const view = viewportOf(list, rulerBox.current, cardBox.current);
      if (!isWhollyVisible(tile, view)) takeOver();
    };
    const onScroll = () => {
      syncVisible();
      setScrollVersion((version) => version + 1);
    };
    // Not passive: a reader's wheel has to end following before the scroll is
    // handed on, so the chip is there in the same beat as the movement.
    scroller.addEventListener("wheel", takeOver);
    scroller.addEventListener("touchmove", takeOver, { passive: true });
    scroller.addEventListener("pointerdown", onPointerDown);
    scroller.addEventListener("keydown", onKeyDown);
    scroller.addEventListener("focusin", onFocusIn);
    scroller.addEventListener("scroll", onScroll, { passive: true });
    return () => {
      scroller.removeEventListener("wheel", takeOver);
      scroller.removeEventListener("touchmove", takeOver);
      scroller.removeEventListener("pointerdown", onPointerDown);
      scroller.removeEventListener("keydown", onKeyDown);
      scroller.removeEventListener("focusin", onFocusIn);
      scroller.removeEventListener("scroll", onScroll);
    };
  }, [shown, flow, syncVisible, setFollowing, onReaderTakeOver, listRef, rulerBox, cardBox]);

  useEffect(() => {
    if (!shown) return;
    if (typeof ResizeObserver === "undefined") return;
    const list = listRef.current;
    if (!list) return;
    const observer = new ResizeObserver(() => {
      setViewportVersion((version) => version + 1);
    });
    observer.observe(list);
    const grid = list.firstElementChild;
    if (grid) observer.observe(grid);
    const tile = tileElements(list)[0];
    if (tile) observer.observe(tile);
    const scroller = scrollerOf(list);
    if (scroller !== list) observer.observe(scroller);
    if (rulerBox.current) observer.observe(rulerBox.current);
    if (cardBox.current) observer.observe(cardBox.current);
    return () => observer.disconnect();
  }, [shown, flow, rows, columns, listRef, rulerBox, cardBox]);

  return {
    following,
    setFollowing,
    columns,
    step,
    flow,
    oneLine,
    cardBottomInset,
    visible,
    scrollVersion,
    placeIndex,
    placeRow,
  };
}
