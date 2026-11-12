import { useLayoutEffect, useState, type RefObject } from "react";

import { sceneColumns, sceneRows, scaledTileHeight } from "../scenesModel";

const DEFAULT_ROWS = 3;

export interface SceneRowsRefs {
  root: RefObject<HTMLElement | null>;
  header: RefObject<HTMLElement | null>;
  ruler: RefObject<HTMLElement | null>;
  grid: RefObject<HTMLElement | null>;
  tile: RefObject<HTMLElement | null>;
  lastPart: RefObject<HTMLElement | null>;
}

export interface SceneRowsResult {
  /** Grid rows the section's scroller can hold. */
  rows: number;
  /** Columns at the width the scroller has without a classic scrollbar. */
  fullWidthColumns: number;
  /** Columns at the width left beside a classic scrollbar. */
  scrollbarColumns: number;
  /** A classic scrollbar is taking width from the scroller. */
  hasScrollbar: boolean;
}

const INITIAL: SceneRowsResult = {
  rows: DEFAULT_ROWS,
  fullWidthColumns: 0,
  scrollbarColumns: 0,
  hasScrollbar: false,
};

/**
 * Grid rows the section's scroller can hold, with the column count of the
 * width the rows are measured at. Measured before paint so a shown body never
 * paints a row count the parts in the same frame would replace, and measured
 * again whenever the scroller or a part around the grid changes size.
 */
export function useSceneRows(
  refs: SceneRowsRefs,
  active: boolean,
): SceneRowsResult {
  const [result, setResult] = useState(INITIAL);

  useLayoutEffect(() => {
    if (!active) return;

    const measure = () => {
      const next = measureRows(refs);
      if (!next) return;
      setResult((current) =>
        current.rows === next.rows &&
        current.fullWidthColumns === next.fullWidthColumns &&
        current.scrollbarColumns === next.scrollbarColumns &&
        current.hasScrollbar === next.hasScrollbar
          ? current
          : next,
      );
    };

    measure();

    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(measure);
    const scroller = refs.root.current ? findScroller(refs.root.current) : null;
    const observed = [
      scroller,
      refs.root.current,
      refs.header.current,
      refs.ruler.current,
      refs.grid.current,
      refs.tile.current,
      refs.lastPart.current,
    ];
    for (const element of observed) {
      if (element) observer.observe(element);
    }
    return () => observer.disconnect();
  });

  return result;
}

/**
 * The scroller whose height the rows follow: the nearest ancestor that scrolls
 * vertically, never the document's root scroller.
 */
function findScroller(from: HTMLElement): HTMLElement | null {
  let node: HTMLElement | null = from.parentElement;
  while (node && node !== document.documentElement && node !== document.body) {
    const overflowY = getComputedStyle(node).overflowY;
    if (overflowY === "auto" || overflowY === "scroll") return node;
    node = node.parentElement;
  }
  return null;
}

/**
 * The rows and full-width columns the section is laid out at, or null when it
 * is not: hidden, jsdom, or no tile and scroller to measure.
 */
function measureRows(refs: SceneRowsRefs): SceneRowsResult | null {
  const root = refs.root.current;
  const grid = refs.grid.current;
  const tile = refs.tile.current;
  if (!root || !grid || !tile) return null;

  const rootRect = root.getBoundingClientRect();
  if (!(rootRect.width > 0) || !(rootRect.height > 0)) return null;

  const scroller = findScroller(root);
  if (!scroller || !(scroller.clientHeight > 0)) return null;

  const tileRect = tile.getBoundingClientRect();
  if (!(tileRect.height > 0)) return null;

  const gridRect = grid.getBoundingClientRect();
  if (!(gridRect.width > 0)) return null;

  const gridStyle = getComputedStyle(grid);
  const rowGap = px(gridStyle.rowGap);
  const columnGap = px(gridStyle.columnGap);

  const scrollerStyle = getComputedStyle(scroller);
  const scrollbar = Math.max(
    0,
    scroller.offsetWidth -
      scroller.clientWidth -
      px(scrollerStyle.borderLeftWidth) -
      px(scrollerStyle.borderRightWidth),
  );
  // A row added above a classic scrollbar must still fit once the scrollbar is
  // gone, so the step is read at the width the scroller has without it.
  const fullWidth = gridRect.width + scrollbar;
  const fullColumns = sceneColumns(fullWidth);
  const fullTileWidth =
    fullColumns > 0
      ? (fullWidth - (fullColumns - 1) * columnGap) / fullColumns
      : tileRect.width;
  const scale =
    scrollbar > 0 && tileRect.width > 0 && fullTileWidth > 0
      ? fullTileWidth / tileRect.width
      : 1;

  const picture = tile.firstElementChild;
  const pictureHeight = picture
    ? picture.getBoundingClientRect().height
    : tileRect.height;
  const fullTileHeight = scaledTileHeight(
    tileRect.height,
    pictureHeight,
    scale,
  );

  const tops = new Set<number>();
  for (const child of Array.from(grid.children)) {
    const rect = child.getBoundingClientRect();
    if (rect.height > 0) tops.add(Math.round(rect.top));
  }
  const drawnRows = Math.max(1, tops.size);
  const step = fullTileHeight + rowGap;

  const last = refs.lastPart.current ?? grid;
  const paddingEdge = scroller.getBoundingClientRect().top + scroller.clientTop;
  let below = 0;
  for (
    let node = last.parentElement;
    node && node !== scroller;
    node = node.parentElement
  ) {
    const style = getComputedStyle(node);
    below +=
      px(style.paddingBottom) +
      px(style.borderBottomWidth) +
      px(style.marginBottom);
  }
  const used =
    last.getBoundingClientRect().bottom -
    paddingEdge +
    scroller.scrollTop +
    below +
    px(scrollerStyle.paddingBottom);
  const slack = scroller.clientHeight - used;
  const fullWidthSlack = slack - drawnRows * (fullTileHeight - tileRect.height);

  if (!(step > 0) || !Number.isFinite(fullWidthSlack)) return null;

  return {
    rows: sceneRows(drawnRows, fullWidthSlack, step),
    fullWidthColumns: fullColumns,
    scrollbarColumns: sceneColumns(gridRect.width),
    hasScrollbar: scrollbar > 0,
  };
}

function px(value: string): number {
  return Number.parseFloat(value) || 0;
}
