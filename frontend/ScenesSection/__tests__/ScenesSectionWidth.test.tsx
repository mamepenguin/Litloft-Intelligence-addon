import "./intlMock";
import { describe, it, expect, vi, beforeEach, afterEach, beforeAll } from "vitest";
import { render, screen, fireEvent, act } from "@testing-library/react";

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
  flush,
  header,
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
 * jsdom lays nothing out. The section now reads its column count from the
 * list's content width, so this gives the list the width a host would and
 * leaves its height unmeasured.
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

function measured(el: Element): boolean {
  return (
    el.getAttribute("role") === "slider" ||
    el.matches("[data-scene-list]") ||
    el.matches("button[data-scene-tile]") ||
    el.querySelector("[data-scene-list], button[data-scene-tile]") !== null
  );
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

/** jsdom does not lay the grid out, so its declared tracks are what it has. */
function declaredColumns(grid: HTMLElement | null): string {
  if (!grid) throw new Error("no grid is drawn");
  return grid.style.gridTemplateColumns;
}

const labels = (rows: ReturnType<typeof makeRows>) =>
  rows.map((row) => formatDuration(row.start));

beforeAll(() => {
  installPointerEvents();
  (
    globalThis as unknown as { ResizeObserver: typeof ResizeObserverStub }
  ).ResizeObserver = ResizeObserverStub;
  vi.spyOn(Element.prototype, "getBoundingClientRect").mockImplementation(
    function (this: Element) {
      return rect(measured(this) ? measuredWidth : 0);
    },
  );
  Object.defineProperty(HTMLElement.prototype, "clientWidth", {
    configurable: true,
    get(this: HTMLElement) {
      return measured(this) ? measuredWidth : 0;
    },
  });
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

describe("SPEC-ADDON-093 I1 — the column count comes from the measured list width", () => {
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
    "SPEC-ADDON-093 I1 — $name draws every frame in $columns columns",
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

      expect(tileLabels()).toEqual(labels(rows));
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

    expect(tileLabels()).toEqual(labels(makeRows(15)));
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

      expect(declaredColumns(container.querySelector(".grid"))).toBe(
        "repeat(5, minmax(0, 1fr))",
      );
    },
  );
});

describe("SPEC-ADDON-093 I5 — a width change redraws in the new column count", () => {
  it("goes from 352px to 688px, drawing every frame in the new columns", async () => {
    const rows = makeRows(40);
    vi.mocked(getClipTimestamps).mockResolvedValue(timestamps(rows));
    const { controller } = makeController({
      currentTime: 50,
      duration: 200,
      paused: true,
    });
    const { container } = renderSection(
      { mediaController: controller },
      "16px",
      352,
    );
    await openNative();
    await waitForVisualIndex();

    expect(declaredColumns(container.querySelector(".grid"))).toBe(
      "repeat(3, minmax(0, 1fr))",
    );

    await measure(688);

    expect(declaredColumns(container.querySelector(".grid"))).toBe(
      "repeat(5, minmax(0, 1fr))",
    );
    expect(tileLabels()).toEqual(labels(rows));
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
