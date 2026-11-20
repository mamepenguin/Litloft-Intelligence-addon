import { act, fireEvent, screen } from "@testing-library/react";
import { vi } from "vitest";

/**
 * jsdom lays nothing out. This is one self-consistent layout for the section:
 * three columns of tiles one `ROW` tall with no row gap, inside a box `VIEW`
 * tall and `WIDTH` wide. Every way of measuring it (rects, client, offset and
 * scroll sizes, `scrollTop`, `ResizeObserver`, `IntersectionObserver`) reads
 * the same numbers, so the result does not depend on which one the section
 * uses.
 */
export let ROW = 88;
export const VIEW = 264;
export const WIDTH = 352;
export const COLS = 3;
export const RULER_HEIGHT = 44;

const TILE_SELECTOR = 'button[aria-label^="Play from "]';
const SHARED = "[data-inspector-scroller]";

const scrollTops = new WeakMap<Element, number>();
/** Elements whose `scrollTop` the section read or wrote. */
const touched = new Set<Element>();
/** `scrollTop` writes the section made, in order. */
export const writes: { el: Element; value: number }[] = [];
let testWriting = false;

function tiles(): HTMLElement[] {
  return Array.from(document.querySelectorAll<HTMLElement>(TILE_SELECTOR));
}

function holdsTiles(el: Element): boolean {
  return el.querySelector(TILE_SELECTOR) !== null;
}

function contentHeight(): number {
  return Math.ceil(tiles().length / COLS) * ROW;
}

/** The shared scroller keeps a range of its own: other tabs fill it too. */
const SHARED_RANGE = 3000;

function clientHeightOf(el: Element): number {
  return holdsTiles(el) || el.matches(SHARED) ? VIEW : 0;
}

function scrollHeightOf(el: Element): number {
  if (el.matches(SHARED)) return Math.max(SHARED_RANGE, contentHeight());
  return holdsTiles(el) ? Math.max(VIEW, contentHeight()) : 0;
}

function scrollSum(el: Element): number {
  let sum = 0;
  for (let node = el.parentElement; node; node = node.parentElement) {
    sum += scrollTops.get(node) ?? 0;
  }
  return sum;
}

function box(left: number, top: number, width: number, height: number): DOMRect {
  return {
    left,
    top,
    width,
    height,
    right: left + width,
    bottom: top + height,
    x: left,
    y: top,
    toJSON: () => ({}),
  } as DOMRect;
}

function rectOf(el: Element): DOMRect {
  if (el.matches(TILE_SELECTOR)) {
    const index = tiles().indexOf(el as HTMLElement);
    const row = Math.floor(index / COLS);
    const width = WIDTH / COLS;
    return box((index % COLS) * width, row * ROW - scrollSum(el), width, ROW);
  }
  if (el.getAttribute("role") === "slider") return box(0, 0, WIDTH, RULER_HEIGHT);
  if (holdsTiles(el)) return box(0, 0, WIDTH, VIEW);
  return box(0, 0, 0, 0);
}

function setScrollTop(el: Element, value: number): void {
  const max = Math.max(0, scrollHeightOf(el) - clientHeightOf(el));
  const next = Math.max(0, Math.min(value, max));
  const before = scrollTops.get(el) ?? 0;
  scrollTops.set(el, next);
  if (!testWriting) writes.push({ el, value: next });
  if (next !== before) {
    setTimeout(() => el.dispatchEvent(new Event("scroll")), 0);
  }
}

function defineGetter(name: string, get: (el: HTMLElement) => number): void {
  Object.defineProperty(HTMLElement.prototype, name, {
    configurable: true,
    get(this: HTMLElement) {
      return get(this);
    },
  });
}

type Observed = { target: Element; intersecting: boolean | null };

class ResizeObserverStub {
  static all: ResizeObserverStub[] = [];
  private targets = new Set<Element>();

  constructor(private readonly callback: ResizeObserverCallback) {
    ResizeObserverStub.all.push(this);
  }

  observe(target: Element): void {
    this.targets.add(target);
    setTimeout(() => this.fire(), 0);
  }

  unobserve(target: Element): void {
    this.targets.delete(target);
  }

  disconnect(): void {
    this.targets.clear();
  }

  fire(): void {
    const entries = Array.from(this.targets)
      .filter((target) => target.isConnected)
      .map((target) => {
        const rect = rectOf(target);
        const size = [{ inlineSize: rect.width, blockSize: rect.height }];
        return {
          target,
          contentRect: rect,
          borderBoxSize: size,
          contentBoxSize: size,
          devicePixelContentBoxSize: size,
        } as unknown as ResizeObserverEntry;
      });
    if (entries.length > 0) this.callback(entries, this as unknown as ResizeObserver);
  }
}

function marginPx(margin: string | undefined, index: number, base: number): number {
  const parts = (margin ?? "0px").trim().split(/\s+/);
  const pick =
    parts.length === 1
      ? parts[0]
      : parts.length === 2
        ? parts[index % 2]
        : parts.length === 3
          ? parts[index === 3 ? 1 : index]
          : parts[index];
  const value = Number.parseFloat(pick ?? "0") || 0;
  return pick?.endsWith("%") ? (value / 100) * base : value;
}

class IntersectionObserverStub {
  static all: IntersectionObserverStub[] = [];
  private observed: Observed[] = [];
  readonly root: Element | null;
  readonly rootMargin: string;
  readonly thresholds: number[] = [0];

  constructor(
    private readonly callback: IntersectionObserverCallback,
    options: IntersectionObserverInit = {},
  ) {
    this.root = (options.root as Element | null | undefined) ?? null;
    this.rootMargin = options.rootMargin ?? "0px";
    IntersectionObserverStub.all.push(this);
  }

  observe(target: Element): void {
    this.observed.push({ target, intersecting: null });
    setTimeout(() => this.check(), 0);
  }

  unobserve(target: Element): void {
    this.observed = this.observed.filter((entry) => entry.target !== target);
  }

  disconnect(): void {
    this.observed = [];
  }

  takeRecords(): IntersectionObserverEntry[] {
    return [];
  }

  check(): void {
    const root = this.root ? rectOf(this.root) : box(0, 0, WIDTH, VIEW);
    const top = root.top - marginPx(this.rootMargin, 0, root.height);
    const bottom = root.bottom + marginPx(this.rootMargin, 2, root.height);
    const changed: IntersectionObserverEntry[] = [];
    for (const entry of this.observed) {
      if (!entry.target.isConnected) continue;
      const rect = rectOf(entry.target);
      const intersecting = rect.bottom >= top && rect.top <= bottom;
      if (intersecting === entry.intersecting) continue;
      entry.intersecting = intersecting;
      changed.push({
        target: entry.target,
        isIntersecting: intersecting,
        intersectionRatio: intersecting ? 1 : 0,
        boundingClientRect: rect,
        intersectionRect: intersecting ? rect : box(0, 0, 0, 0),
        rootBounds: root,
        time: Date.now(),
      } as IntersectionObserverEntry);
    }
    if (changed.length > 0) {
      this.callback(changed, this as unknown as IntersectionObserver);
    }
  }
}

export function installLayout(): void {
  vi.spyOn(Element.prototype, "getBoundingClientRect").mockImplementation(
    function (this: Element) {
      return rectOf(this);
    },
  );
  defineGetter("clientHeight", clientHeightOf);
  defineGetter("scrollHeight", scrollHeightOf);
  defineGetter("clientWidth", (el) =>
    holdsTiles(el) || el.getAttribute("role") === "slider" ? WIDTH : 0,
  );
  defineGetter("scrollWidth", (el) =>
    holdsTiles(el) || el.getAttribute("role") === "slider" ? WIDTH : 0,
  );
  defineGetter("offsetWidth", (el) => rectOf(el).width);
  defineGetter("offsetHeight", (el) => rectOf(el).height);
  defineGetter("offsetTop", (el) =>
    el.matches(TILE_SELECTOR)
      ? Math.floor(tiles().indexOf(el) / COLS) * ROW
      : 0,
  );
  Object.defineProperty(Element.prototype, "scrollTop", {
    configurable: true,
    get(this: Element) {
      if (holdsTiles(this)) touched.add(this);
      return scrollTops.get(this) ?? 0;
    },
    set(this: Element, value: number) {
      if (holdsTiles(this)) touched.add(this);
      setScrollTop(this, value);
    },
  });
  const scrollTo = function (
    this: Element,
    a?: ScrollToOptions | number,
    b?: number,
  ) {
    touched.add(this);
    const top = typeof a === "number" ? b : a?.top;
    if (typeof top === "number") setScrollTop(this, top);
  };
  Element.prototype.scrollTo = scrollTo as Element["scrollTo"];
  Element.prototype.scroll = scrollTo as Element["scroll"];
  Element.prototype.scrollBy = function (
    this: Element,
    a?: ScrollToOptions | number,
    b?: number,
  ) {
    touched.add(this);
    const by = typeof a === "number" ? b : a?.top;
    if (typeof by === "number") setScrollTop(this, (scrollTops.get(this) ?? 0) + by);
  } as Element["scrollBy"];
  (globalThis as unknown as { ResizeObserver: unknown }).ResizeObserver =
    ResizeObserverStub;
  (globalThis as unknown as { IntersectionObserver: unknown }).IntersectionObserver =
    IntersectionObserverStub;
  document.addEventListener(
    "scroll",
    () => {
      for (const observer of IntersectionObserverStub.all) observer.check();
    },
    true,
  );
}

/** A new row step for every tile, as a picture of another ratio would give. */
export function setRowStep(px: number): void {
  ROW = px;
}

export function resetLayout(): void {
  ROW = 88;
  writes.length = 0;
  touched.clear();
  ResizeObserverStub.all.length = 0;
  IntersectionObserverStub.all.length = 0;
}

/** Reports a size change to every live observer, as a resize would. */
export async function fireResize(): Promise<void> {
  await act(async () => {
    for (const observer of ResizeObserverStub.all) observer.fire();
  });
}

/**
 * The element the reader scrolls: the shared scroller in flow form, else the
 * innermost tile-holding element whose `scrollTop` the section reads or
 * writes, else the tiles' parent.
 */
export function scroller(): HTMLElement {
  const shared = document.querySelector<HTMLElement>(SHARED);
  if (shared) return shared;
  const candidates = Array.from(touched).filter(
    (el): el is HTMLElement =>
      el instanceof HTMLElement && el.isConnected && holdsTiles(el),
  );
  const innermost = candidates.find(
    (el) => !candidates.some((other) => other !== el && el.contains(other)),
  );
  if (innermost) return innermost;
  const first = tiles()[0];
  if (!first?.parentElement) throw new Error("no tile is drawn");
  return first.parentElement;
}

export function scrollTopOf(el: Element): number {
  return scrollTops.get(el) ?? 0;
}

/** Sets a position as another tab or a test would, not as the section. */
export function presetScrollTop(el: Element, value: number): void {
  testWriting = true;
  try {
    setScrollTop(el, value);
  } finally {
    testWriting = false;
  }
}

/**
 * A reader's wheel scroll to `top`: the wheel lands on the scroller, then the
 * browser moves it and sends `scroll`.
 */
export async function wheelTo(top: number, target: Element = scroller()): Promise<void> {
  const el = scroller();
  fireEvent.wheel(target, { deltaY: top - scrollTopOf(el) });
  presetScrollTop(el, top);
  await act(async () => {
    el.dispatchEvent(new Event("scroll"));
    vi.advanceTimersByTime(0);
  });
}

function rowOf(tile: Element): number {
  return Math.floor(tiles().indexOf(tile as HTMLElement) / COLS);
}

/** Rows with at least half a tile inside the visible area, in order. */
export function visibleRows(): number[] {
  const rows = new Set<number>();
  for (const tile of tiles()) {
    const rect = tile.getBoundingClientRect();
    const inside = Math.min(rect.bottom, VIEW) - Math.max(rect.top, 0);
    if (inside >= rect.height / 2) rows.add(rowOf(tile));
  }
  return Array.from(rows).sort((a, b) => a - b);
}

export function whollyVisible(row: number): boolean {
  const tile = tiles()[row * COLS];
  if (!tile) throw new Error(`no row ${row}`);
  const rect = tile.getBoundingClientRect();
  return rect.top >= 0 && rect.bottom <= VIEW;
}

export function tileElements(): HTMLElement[] {
  return tiles();
}

export const CHIP = "Back to current position";

export function chip(): HTMLElement | null {
  return screen.queryByRole("button", { name: CHIP });
}
