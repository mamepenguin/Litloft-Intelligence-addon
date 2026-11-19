const TILE_SELECTOR = "button[data-scene-tile]";
const SHARED_SCROLLER = "[data-inspector-scroller]";

/** The full card's minimum height, in rem. */
export const FULL_CARD_MIN_REM = 7.6;

/** The root font size, 16 px when it is not a positive number. */
export function rootRemPx(): number {
  const size = Number.parseFloat(
    getComputedStyle(document.documentElement).fontSize,
  );
  return size > 0 ? size : 16;
}

export function isFlowForm(list: HTMLElement): boolean {
  return list.closest(SHARED_SCROLLER) !== null;
}

/** The box the reader scrolls: the host's marked ancestor, else the list. */
export function scrollerOf(list: HTMLElement): HTMLElement {
  return list.closest<HTMLElement>(SHARED_SCROLLER) ?? list;
}

export function tileElements(list: HTMLElement): HTMLElement[] {
  return Array.from(list.querySelectorAll<HTMLElement>(TILE_SELECTOR));
}

/** The part of the scroller's client box where a tile can be seen. */
export function viewportOf(
  list: HTMLElement,
  rulerBox: HTMLElement | null,
  cardBox: HTMLElement | null,
): { top: number; bottom: number } {
  const scroller = scrollerOf(list);
  const scrollerRect = scroller.getBoundingClientRect();
  if (scroller === list) {
    const top = scrollerRect.top + scroller.clientTop;
    return { top, bottom: top + scroller.clientHeight };
  }
  const sticky = stickyTopOf(list);
  const rulerHeight = rulerBox?.getBoundingClientRect().height ?? 0;
  const top = scrollerRect.top + scroller.clientTop + sticky + rulerHeight;
  const card = cardBox?.getBoundingClientRect();
  const bottom =
    card && card.height > 0
      ? card.top
      : scrollerRect.top + scroller.clientTop + scroller.clientHeight;
  return { top, bottom };
}

/** The host publishes where its pinned strip ends, on the section itself. */
export function stickyTopOf(section: HTMLElement): number {
  const value = Number.parseFloat(
    getComputedStyle(section).getPropertyValue("--inspector-sticky-top"),
  );
  return Number.isFinite(value) ? value : 0;
}

/** A drawn tile's height plus the grid's row gap, measured from a tile. */
export function rowStepOf(list: HTMLElement, columns: number): number {
  const tiles = tileElements(list);
  if (tiles.length === 0) return 0;
  if (tiles.length > columns) {
    const first = tiles[0].getBoundingClientRect();
    const next = tiles[columns].getBoundingClientRect();
    if (next.top > first.top) return next.top - first.top;
  }
  const height = tiles[0].getBoundingClientRect().height;
  const grid = tiles[0].parentElement;
  const gap = grid
    ? Number.parseFloat(getComputedStyle(grid).rowGap) || 0
    : 0;
  return height + gap;
}

/** Rows with at least half of a tile inside the visible area, in order. */
export function visibleRowsOf(
  list: HTMLElement,
  columns: number,
  view: { top: number; bottom: number },
): number[] {
  if (!(view.bottom > view.top) || columns <= 0) return [];
  const tiles = tileElements(list);
  if (tiles.length === 0) return [];
  const step = rowStepOf(list, columns);
  if (!(step > 0)) return [];
  const origin = tiles[0].getBoundingClientRect();
  const height = origin.height;
  const count = Math.ceil(tiles.length / columns);
  const first = Math.max(0, Math.floor((view.top - origin.top) / step));
  const last = Math.min(count - 1, Math.floor((view.bottom - origin.top) / step));
  const rows: number[] = [];
  for (let row = Math.max(0, first - 1); row <= Math.min(count - 1, last + 1); row += 1) {
    const top = origin.top + row * step;
    const inside = Math.min(top + height, view.bottom) - Math.max(top, view.top);
    if (inside >= height / 2) rows.push(row);
  }
  return rows;
}

/** The row at the top of the visible area when no row is half inside it. */
export function rowAt(
  list: HTMLElement,
  columns: number,
  y: number,
): number {
  const tiles = tileElements(list);
  if (tiles.length === 0 || columns <= 0) return 0;
  const step = rowStepOf(list, columns);
  if (!(step > 0)) return 0;
  const origin = tiles[0].getBoundingClientRect();
  const count = Math.ceil(tiles.length / columns);
  return Math.max(
    0,
    Math.min(count - 1, Math.floor((y - origin.top) / step)),
  );
}

export function rowCountOf(list: HTMLElement, columns: number): number {
  if (columns <= 0) return 0;
  return Math.ceil(tileElements(list).length / columns);
}

/**
 * The list's offset from the scroller's content origin. `offsetTop` is the
 * layout position, so a scrollport the section is already sitting in does not
 * change it; rects would report the scrolled position instead.
 */
export function contentOffsetTop(
  list: HTMLElement,
  scroller: HTMLElement,
): number {
  const absolute = (element: HTMLElement): number => {
    let top = 0;
    let node: HTMLElement | null = element;
    while (node) {
      top += node.offsetTop;
      node = node.offsetParent as HTMLElement | null;
    }
    return top;
  };
  return absolute(list) - absolute(scroller);
}

export function canScroll(scroller: HTMLElement): boolean {
  return scroller.scrollHeight > scroller.clientHeight + 0.5;
}

export function isWhollyVisible(
  tile: HTMLElement,
  view: { top: number; bottom: number },
): boolean {
  if (!(view.bottom > view.top)) return false;
  const rect = tile.getBoundingClientRect();
  return rect.top >= view.top - 0.5 && rect.bottom <= view.bottom + 0.5;
}

/**
 * From the top of the card box to the bottom of the last part after it (the
 * stale note, the action row), gaps included: the height the list gives up to
 * whatever is drawn below it.
 */
export function partsBelowList(cardBox: HTMLElement): number {
  const top = cardBox.getBoundingClientRect().top;
  let bottom = cardBox.getBoundingClientRect().bottom;
  for (
    let node = cardBox.nextElementSibling;
    node;
    node = node.nextElementSibling
  ) {
    bottom = Math.max(bottom, node.getBoundingClientRect().bottom);
  }
  return Math.max(0, bottom - top);
}
