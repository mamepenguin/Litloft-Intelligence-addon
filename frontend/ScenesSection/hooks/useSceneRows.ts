import { useLayoutEffect, useState, type RefObject } from "react";

import { sceneColumns, sceneRows } from "../scenesModel";

const DEFAULT_ROWS = 3;

export interface SceneRowsRefs {
  root: RefObject<HTMLElement | null>;
  header: RefObject<HTMLElement | null>;
  ruler: RefObject<HTMLElement | null>;
  grid: RefObject<HTMLElement | null>;
  tile: RefObject<HTMLElement | null>;
  lastPart: RefObject<HTMLElement | null>;
}

/**
 * Grid rows the section's scroller can hold. Measured before paint so a shown
 * body never paints a row count the parts in the same frame would replace, and
 * measured again whenever the scroller or a part around the grid changes size.
 */
export function useSceneRows(refs: SceneRowsRefs, active: boolean): number {
  const [rows, setRows] = useState(DEFAULT_ROWS);

  useLayoutEffect(() => {
    if (!active) return;

    const measure = () => {
      const next = measureRows(refs);
      if (!next) return;
      setRows((current) => next(current));
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

  return rows;
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
 * A function of the rows already held, or null when the section is not laid
 * out: hidden, jsdom, or no tile and scroller to measure.
 */
function measureRows(
  refs: SceneRowsRefs,
): ((current: number) => number) | null {
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

  const tops = new Set<number>();
  for (const child of Array.from(grid.children)) {
    const rect = child.getBoundingClientRect();
    if (rect.height > 0) tops.add(Math.round(rect.top));
  }
  const drawnRows = Math.max(1, tops.size);
  const step = tileRect.height * scale + rowGap;

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
  const fullWidthSlack = slack - drawnRows * (tileRect.height * scale - tileRect.height);

  return (current) =>
    step > 0 && Number.isFinite(fullWidthSlack)
      ? sceneRows(drawnRows, fullWidthSlack, step)
      : current;
}

function px(value: string): number {
  return Number.parseFloat(value) || 0;
}
