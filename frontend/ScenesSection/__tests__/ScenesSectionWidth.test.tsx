import "./intlMock";
import { describe, it, expect, vi, beforeEach, afterEach, beforeAll } from "vitest";
import { render, screen, fireEvent, act } from "@testing-library/react";

import type { MediaController } from "@/lib/mediaController";
import { formatDuration } from "@/lib/format";

vi.mock("@/addons/intelligence/api", () => ({
  getClipTimestamps: vi.fn(),
  getVideoVisualIndex: vi.fn(),
  generateVideoVisualIndex: vi.fn(),
  retryVideoVisualIndex: vi.fn(),
  getFrameUrl: vi.fn(
    (fileId: string, t: number) => `/frame/${fileId}?t=${t}`,
  ),
  getSceneTileUrl: vi.fn(
    (fileId: string, t: number) => `/scene-tile/${fileId}?t=${t}`,
  ),
}));

vi.mock("@/hooks/usePolicy", () => ({
  usePolicy: vi.fn(),
}));

import ScenesSection from "@/addons/intelligence/ScenesSection";
import {
  getClipTimestamps,
  getVideoVisualIndex,
} from "@/addons/intelligence/api";
import { usePolicy } from "@/hooks/usePolicy";
import {
  advance,
  flush,
  header,
  images,
  imgTimes,
  installPointerEvents,
  makeController,
  makeRows,
  ruler,
  SceneSectionProps,
  tileLabels,
  timestamps,
  waitForVisualIndex,
} from "./helpers";

/**
 * jsdom lays nothing out, so the section's width comes from the same places
 * the production code reads it: the ruler's border box, the `ResizeObserver`
 * callback made from a width change, and the root font size.
 */
let measuredWidth = 0;
let rootFontSize = "16px";
const observers: ResizeObserverStub[] = [];

class ResizeObserverStub {
  private targets = new Set<Element>();

  constructor(private readonly callback: ResizeObserverCallback) {
    observers.push(this);
  }

  observe(target: Element): void {
    this.targets.add(target);
  }

  unobserve(target: Element): void {
    this.targets.delete(target);
  }

  disconnect(): void {
    this.targets.clear();
  }

  fire(): void {
    if (Array.from(this.targets).some((target) => target.isConnected)) {
      this.callback([], this as unknown as ResizeObserver);
    }
  }
}

function rect(width: number): DOMRect {
  return {
    left: 0,
    top: 0,
    width,
    height: 44,
    right: width,
    bottom: 44,
    x: 0,
    y: 0,
    toJSON: () => ({}),
  } as DOMRect;
}

function renderSection(
  props: Partial<SceneSectionProps> = {},
  font = "16px",
  width = 0,
) {
  rootFontSize = font;
  measuredWidth = width;
  return render(
    <ScenesSection
      fileId="f1"
      drive="d1"
      fileType="video"
      mimeType="video/mp4"
      {...props}
    />,
  );
}

async function openNative(): Promise<void> {
  fireEvent.click(header());
  await screen.findByRole("slider");
}

/** Reports a new drawn width the way the section's own observer would. */
async function measure(width: number): Promise<void> {
  measuredWidth = width;
  await act(async () => {
    for (const observer of observers) observer.fire();
  });
}

function labelsFor(
  rows: ReturnType<typeof makeRows>,
  start: number,
  count: number,
): string[] {
  return rows.slice(start, start + count).map((row) => formatDuration(row.start));
}

/** The ruler's window band, which the section draws with left and width. */
function windowBand(): HTMLElement {
  const band = Array.from(ruler().querySelectorAll<HTMLElement>("div")).find(
    (el) => el.style.left !== "" && el.style.width !== "",
  );
  if (!band) throw new Error("no window band is drawn");
  return band;
}

function percentOf(value: string): number {
  return Number.parseFloat(value);
}

/** jsdom does not lay the grid out, so its declared tracks are what it has. */
function declaredColumns(grid: HTMLElement | null): string {
  if (!grid) throw new Error("no grid is drawn");
  return grid.style.gridTemplateColumns;
}

beforeAll(() => {
  installPointerEvents();
  (
    globalThis as unknown as { ResizeObserver: typeof ResizeObserverStub }
  ).ResizeObserver = ResizeObserverStub;
  vi.spyOn(Element.prototype, "getBoundingClientRect").mockImplementation(
    function (this: Element) {
      return rect(this.getAttribute("role") === "slider" ? measuredWidth : 0);
    },
  );
  const real = window.getComputedStyle.bind(window);
  vi.spyOn(window, "getComputedStyle").mockImplementation(((
    element: Element,
  ) => {
    if (element === document.documentElement) {
      return { fontSize: rootFontSize } as CSSStyleDeclaration;
    }
    return real(element);
  }) as typeof window.getComputedStyle);
});

beforeEach(() => {
  vi.clearAllMocks();
  observers.length = 0;
  measuredWidth = 0;
  rootFontSize = "16px";
  vi.mocked(getClipTimestamps).mockResolvedValue(timestamps(makeRows(15)));
  vi.mocked(getVideoVisualIndex).mockResolvedValue(null);
  vi.mocked(usePolicy).mockReturnValue({ enabled: true, isLoading: false });
});

afterEach(() => {
  vi.useRealTimers();
});

describe("SPEC-ADDON-093 I1 — the column count comes from the measured width", () => {
  const rows = makeRows(40);

  it.each([
    { name: "a width not measured yet", width: 0, columns: 3 },
    { name: "the 384px inspector at 352px", width: 352, columns: 3 },
    { name: "the 390px Bottom Sheet at 358px", width: 358, columns: 3 },
    { name: "the 720px Bottom Sheet at 688px", width: 688, columns: 5 },
    { name: "the 900px collection canvas", width: 900, columns: 7 },
    { name: "a 1400px host", width: 1400, columns: 11 },
    { name: "a width too narrow for three columns", width: 200, columns: 3 },
  ])(
    "SPEC-ADDON-093 I1 — $name draws three rows of $columns",
    async ({ width, columns }) => {
      vi.mocked(getClipTimestamps).mockResolvedValue(timestamps(rows));
      const { controller } = makeController({ duration: 200 });
      const { container } = renderSection(
        { mediaController: controller },
        "16px",
        width,
      );
      await openNative();
      await waitForVisualIndex();

      expect(tileLabels()).toEqual(labelsFor(rows, 0, 3 * columns));
      expect(declaredColumns(container.querySelector(".grid"))).toBe(
        `repeat(${columns}, minmax(0, 1fr))`,
      );
    },
  );

  it("SPEC-ADDON-093 I1 — a 20px root font makes 688px four columns", async () => {
    const { controller } = makeController({ duration: 200 });
    const { container } = renderSection(
      { mediaController: controller },
      "20px",
      688,
    );
    await openNative();
    await waitForVisualIndex();

    expect(tileLabels()).toEqual(labelsFor(rows, 0, 12));
    expect(declaredColumns(container.querySelector(".grid"))).toBe(
      "repeat(4, minmax(0, 1fr))",
    );
  });

  it.each(["not-a-size", "", "0", "-4px"])(
    "SPEC-ADDON-093 I1 — a root font size of %j counts as 16px",
    async (font) => {
      const { controller } = makeController({ duration: 200 });
      const { container } = renderSection(
        { mediaController: controller },
        font,
        688,
      );
      await openNative();
      await waitForVisualIndex();

      expect(tileLabels()).toEqual(labelsFor(rows, 0, 15));
      expect(declaredColumns(container.querySelector(".grid"))).toBe(
        "repeat(5, minmax(0, 1fr))",
      );
    },
  );
});

describe("SPEC-ADDON-089 I1 — the window starts from the column count", () => {
  it.each([
    { name: "i = 0", n: 40, time: 0, start: 0, count: 15 },
    { name: "a middle i", n: 40, time: 50, start: 5, count: 15 },
    { name: "i = n − 1", n: 40, time: 195, start: 25, count: 15 },
    { name: "n < 3c", n: 12, time: 0, start: 0, count: 12 },
    { name: "n not a multiple of c", n: 17, time: 80, start: 2, count: 15 },
  ])(
    "SPEC-ADDON-089 I1 — five columns at $name",
    async ({ n, time, start, count }) => {
      const rows = makeRows(n);
      vi.mocked(getClipTimestamps).mockResolvedValue(timestamps(rows));
      const { controller } = makeController({ currentTime: time, duration: 200 });
      renderSection({ mediaController: controller }, "16px", 688);
      await openNative();
      await waitForVisualIndex();

      expect(tileLabels()).toEqual(labelsFor(rows, start, count));
      if (n >= 15) expect(tileLabels()).toHaveLength(15);
    },
  );
});

describe("SPEC-ADDON-093 I3 and SPEC-ADDON-089 I8 — the pictures requested are the drawn tiles' frames", () => {
  it("SPEC-ADDON-089 I8 — requests the fifteen drawn frames after 150ms and nothing before", async () => {
    vi.useFakeTimers();
    vi.mocked(usePolicy).mockReturnValue({ enabled: false, isLoading: false });
    const rows = makeRows(40);
    vi.mocked(getClipTimestamps).mockResolvedValue(timestamps(rows));
    const { controller } = makeController({ duration: 200, paused: true });
    renderSection({ mediaController: controller }, "16px", 688);

    fireEvent.click(header());
    await flush();

    expect(tileLabels()).toEqual(labelsFor(rows, 0, 15));
    expect(images()).toHaveLength(0);

    await advance(149);
    expect(images()).toHaveLength(0);

    await advance(1);

    const drawn = rows.slice(0, 15).map((row) => row.start);
    expect(imgTimes()).toEqual(drawn);
    expect(images()).toHaveLength(15);
    for (const img of images()) {
      expect(img.getAttribute("src")).toContain("/frame/f1?t=");
    }
  });
});

describe("SPEC-ADDON-093 I4 — the ruler's window band spans the drawn frames", () => {
  it.each([
    { name: "the first window", time: 0, left: 0, width: 37.5 },
    { name: "a middle window", time: 50, left: 12.5, width: 37.5 },
    { name: "the last window, which ends at L", time: 195, left: 62.5, width: 37.5 },
  ])(
    "SPEC-ADDON-093 I4 — $name",
    async ({ time, left, width }) => {
      vi.mocked(getClipTimestamps).mockResolvedValue(timestamps(makeRows(40)));
      const { controller } = makeController({ currentTime: time, duration: 200 });
      renderSection({ mediaController: controller }, "16px", 688);
      await openNative();
      await waitForVisualIndex();

      const band = windowBand();
      expect(percentOf(band.style.left)).toBeCloseTo(left, 3);
      expect(percentOf(band.style.width)).toBeCloseTo(width, 3);
    },
  );
});

describe("SPEC-ADDON-093 I5 — a width change redraws around the same point", () => {
  it("goes from nine tiles at 352px to fifteen at 688px with the playing frame drawn", async () => {
    const rows = makeRows(40);
    vi.mocked(getClipTimestamps).mockResolvedValue(timestamps(rows));
    const { controller } = makeController({
      currentTime: 50,
      duration: 200,
      paused: true,
    });
    renderSection({ mediaController: controller }, "16px", 352);
    await openNative();
    await waitForVisualIndex();

    expect(tileLabels()).toEqual(labelsFor(rows, 6, 9));

    await measure(688);

    const window = labelsFor(rows, 5, 15);
    expect(tileLabels()).toEqual(window);
    expect(window).toContain(formatDuration(50));
  });
});

describe("SPEC-ADDON-093 I6 — the loading skeleton draws 3c blocks in c columns", () => {
  it("draws fifteen placeholders at five columns", async () => {
    let resolve!: (value: ReturnType<typeof timestamps>) => void;
    const pending = new Promise<ReturnType<typeof timestamps>>((done) => {
      resolve = done;
    });
    vi.mocked(getClipTimestamps).mockReturnValue(pending);

    const { container } = renderSection({}, "16px", 688);
    fireEvent.click(header());
    await screen.findByRole("slider");

    const grid = container.querySelector<HTMLElement>(".grid");
    expect(grid).not.toBeNull();
    expect(grid!.children).toHaveLength(15);
    expect(declaredColumns(grid)).toBe("repeat(5, minmax(0, 1fr))");
    expect(
      screen.queryAllByRole("button", { name: /^Play from / }),
    ).toHaveLength(0);

    resolve(timestamps(makeRows(40)));
    await flush();
  });
});
