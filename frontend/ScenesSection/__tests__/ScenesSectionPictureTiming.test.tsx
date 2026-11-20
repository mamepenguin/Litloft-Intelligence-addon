import { describe, it, expect, vi, beforeEach, afterEach, beforeAll } from "vitest";
import { act, fireEvent, render } from "@testing-library/react";

vi.mock("next-intl", async () => {
  const en = (await import("@/addons/intelligence/messages/en.json")).default as unknown as Record<
    string,
    Record<string, string>
  >;
  return {
    useTranslations: (namespace?: string) => (key: string, values?: Record<string, unknown>) => {
      const table = (namespace ? en[namespace] : undefined) ?? en.file;
      const template = table?.[key] ?? key;
      return Object.entries(values ?? {}).reduce(
        (text, [name, value]) => text.replace(`{${name}}`, String(value)),
        template,
      );
    },
    useLocale: () => "en",
  };
});

vi.mock("@/addons/intelligence/api", () => ({
  getClipTimestamps: vi.fn(),
  getVideoVisualIndex: vi.fn(),
  generateVideoVisualIndex: vi.fn(),
  retryVideoVisualIndex: vi.fn(),
  getFrameUrl: vi.fn((fileId: string, t: number) => `/frame/${fileId}?t=${t}`),
  getSceneTileUrl: vi.fn((fileId: string, t: number) => `/scene-tile/${fileId}?t=${t}`),
}));

vi.mock("@/hooks/usePolicy", () => ({
  usePolicy: vi.fn(),
}));

vi.mock("@/hooks/useWebSocket", () => ({
  useWebSocket: vi.fn(() => null),
}));

import ScenesSection from "@/addons/intelligence/ScenesSection";
import { getClipTimestamps, getVideoVisualIndex } from "@/addons/intelligence/api";
import { usePolicy } from "@/hooks/usePolicy";
import {
  advance,
  firePointer,
  flush,
  images,
  imgTimes,
  indexState,
  installPointerEvents,
  LOFT_MIME,
  makeController,
  makeRows,
  pointerX,
  ruler,
  SceneSectionProps,
  tile,
  timestamps,
} from "./helpers";
import {
  ROW,
  WIDTH,
  chip,
  fireResize,
  installLayout,
  resetLayout,
  scroller,
  scrollTopOf,
  setRowStep,
  visibleRows,
  wheelTo,
} from "./scrollListLayout";

/** 30 frames 5 s apart: ten rows of three, three of them visible at once. */
const N = 30;
const ROWS = makeRows(N);
const L = 150;
const D = 200;
const LAST_ROW = N / 3 - 1;

type Props = Partial<SceneSectionProps>;

function section(props: Props = {}) {
  return (
    <ScenesSection
      fileId="f1"
      drive="d1"
      fileType="video"
      mimeType="video/mp4"
      labelledByHost
      active
      onAvailability={() => {}}
      {...props}
    />
  );
}

/** Runs what is due now; never moves the clock. */
async function now(): Promise<void> {
  await flush(8);
  await advance(0);
  await flush(8);
}

async function open(props: Props = {}) {
  const view = render(section(props));
  await now();
  await now();
  return view;
}

function rowTimes(row: number): number[] {
  return [0, 1, 2].map((col) => (row * 3 + col) * 5);
}

function rowsTimes(rows: number[]): number[] {
  return rows.flatMap(rowTimes);
}

/** The near rows by the stub's geometry: the visible rows and one on either side. */
function nearRows(): number[] {
  const visible = visibleRows();
  const first = Math.max(0, visible[0] - 1);
  const last = Math.min(LAST_ROW, visible[visible.length - 1] + 1);
  return Array.from({ length: last - first + 1 }, (_, i) => first + i);
}

function asked(times: number[]): number[] {
  const drawn = imgTimes();
  return times.filter((time) => drawn.includes(time));
}

function srcs(): string[] {
  return images().map((img) => img.getAttribute("src") ?? "");
}

/** Every `src` the page was given, in order, with the fake clock's time. */
const srcLog: { src: string; at: number }[] = [];
const removedLog: string[] = [];
let observer: MutationObserver | null = null;

function logImg(node: Node): void {
  if (!(node instanceof Element)) return;
  const imgs = node instanceof HTMLImageElement ? [node] : Array.from(node.querySelectorAll("img"));
  for (const img of imgs) {
    const src = img.getAttribute("src");
    if (src) srcLog.push({ src, at: Date.now() });
  }
}

function startLog(): void {
  srcLog.length = 0;
  removedLog.length = 0;
  observer = new MutationObserver((records) => {
    for (const record of records) {
      if (record.type === "attributes" && record.target instanceof HTMLImageElement) {
        const src = record.target.getAttribute("src");
        if (src && src !== record.oldValue) srcLog.push({ src, at: Date.now() });
      }
      record.addedNodes.forEach(logImg);
      record.removedNodes.forEach((node) => {
        if (!(node instanceof Element)) return;
        const imgs =
          node instanceof HTMLImageElement ? [node] : Array.from(node.querySelectorAll("img"));
        for (const img of imgs) removedLog.push(img.getAttribute("src") ?? "");
      });
    }
  });
  observer.observe(document, {
    subtree: true,
    childList: true,
    attributes: true,
    attributeFilter: ["src"],
    attributeOldValue: true,
  });
}

/** Ends opening with a reader's wheel one row down, then brings row 5 near with another. */
async function dwellOnRowFive(): Promise<void> {
  await wheelTo(ROW);
  expect(visibleRows()).toEqual([1, 2, 3]);
  await advance(10);
  await wheelTo(2 * ROW);
  expect(visibleRows()).toEqual([2, 3, 4]);
}

beforeAll(() => {
  installPointerEvents();
  installLayout();
});

beforeEach(() => {
  vi.clearAllMocks();
  vi.useFakeTimers();
  resetLayout();
  vi.mocked(getClipTimestamps).mockImplementation(((fileId: string) =>
    Promise.resolve(timestamps(ROWS, fileId))) as never);
  vi.mocked(getVideoVisualIndex).mockResolvedValue(indexState());
  vi.mocked(usePolicy).mockReturnValue({ enabled: true, isLoading: false });
  startLog();
});

afterEach(() => {
  observer?.disconnect();
  vi.useRealTimers();
});

describe("SPEC-ADDON-155 I1 — opening asks for the near tiles at once", () => {
  it("SPEC-ADDON-155 I1 — with the playing row placed further down, rows 4–8 have a src with no time advanced, and rows 0–3, near only before the start placement, never get one", async () => {
    await open({ mediaController: makeController({ duration: L, currentTime: 100 }).controller });
    expect(visibleRows()).toEqual([5, 6, 7]);

    expect(asked(rowsTimes([4, 5, 6, 7, 8]))).toEqual(rowsTimes([4, 5, 6, 7, 8]));
    expect(asked(rowsTimes([0, 1, 2, 3]))).toEqual([]);

    await advance(1000);
    await flush(4);
    expect(asked(rowsTimes([0, 1, 2, 3]))).toEqual([]);
  });

  it("SPEC-ADDON-155 I1 — a .loft shown before its frames arrive asks for the near tiles when they arrive, with no time advanced", async () => {
    let resolve!: (value: ReturnType<typeof timestamps>) => void;
    vi.mocked(getClipTimestamps).mockReturnValueOnce(
      new Promise((res) => {
        resolve = res;
      }) as never,
    );
    render(
      section({
        mimeType: LOFT_MIME,
        mediaController: makeController({ duration: L }).controller,
      }),
    );
    await now();
    expect(images()).toEqual([]);

    resolve(timestamps(ROWS));
    await now();
    await now();

    expect(visibleRows()).toEqual([0, 1, 2]);
    const near = rowsTimes([0, 1, 2, 3]);
    expect(near.filter((time) => !srcs().includes(`/scene-tile/f1?t=${time}`))).toEqual([]);
  });
});

describe("SPEC-ADDON-155 I2 — after opening, a tile is asked for once it has dwelt D", () => {
  it("SPEC-ADDON-155 I2 — row 5, near through scroll events every 50 ms, has no src at D − 1 ms and has one at D, while the list keeps moving", async () => {
    await open({ mediaController: makeController({ duration: L }).controller });
    await dwellOnRowFive();

    for (const top of [2 * ROW + 20, 2 * ROW, 2 * ROW + 20]) {
      await advance(50);
      await wheelTo(top);
      expect(visibleRows()).toEqual([2, 3, 4]);
    }
    await advance(D - 1 - 150);
    await flush(4);
    expect(asked(rowTimes(5)), "at D − 1 ms").toEqual([]);

    await advance(1);
    await flush(4);
    expect(asked(rowTimes(5)), "at D").toEqual(rowTimes(5));
    expect(asked(rowTimes(4)), "row 4, near since the first wheel").toEqual(rowTimes(4));

    await wheelTo(2 * ROW);
    expect(asked(rowTimes(6)), "row 6 was never near").toEqual([]);
  });
});

describe("SPEC-ADDON-155 I2 — placements the section makes after opening dwell too", () => {
  it("SPEC-ADDON-155 I2 — the chip's placement: rows it brings near that were never asked for have no src at D − 1 ms and have one at D", async () => {
    await open({ mediaController: makeController({ duration: L, currentTime: 100 }).controller });
    expect(visibleRows()).toEqual([5, 6, 7]);

    await wheelTo(10);
    act(() => setRowStep(44));
    await fireResize();
    await now();
    expect(visibleRows()).toEqual([0, 1, 2, 3, 4, 5]);
    const before = rowsTimes(nearRows());
    const askedAtOpening = rowsTimes([4, 5, 6, 7, 8]);
    fireEvent.click(chip()!);
    await now();
    expect(visibleRows()).toEqual([4, 5, 6, 7, 8, 9]);
    const fresh = rowsTimes(nearRows()).filter(
      (time) => !before.includes(time) && !askedAtOpening.includes(time),
    );
    expect(fresh).toEqual(rowTimes(9));

    expect(asked(fresh), "with no time advanced").toEqual([]);
    await advance(D - 1);
    await flush(4);
    expect(asked(fresh), "at D − 1 ms").toEqual([]);
    await advance(1);
    await flush(4);
    expect(asked(fresh), "at D").toEqual(fresh);
  });

  it("SPEC-ADDON-155 I2 — following during playback after opening: the rows its placement brings near are asked for D after it, not before", async () => {
    const { controller, state } = makeController({ duration: L, currentTime: 0, paused: true });
    await open({ mediaController: controller });
    await wheelTo(ROW);
    fireEvent.click(chip()!);
    await now();
    expect(visibleRows()).toEqual([0, 1, 2]);
    const before = rowsTimes(nearRows());

    state.paused = false;
    state.currentTime = 45;
    let waited = 0;
    while (visibleRows()[0] === 0 && waited < 2000) {
      await advance(10);
      waited += 10;
    }
    expect(visibleRows()).toEqual([2, 3, 4]);
    const fresh = rowsTimes(nearRows()).filter((time) => !before.includes(time));
    expect(fresh).toEqual(rowsTimes([4, 5]));

    await advance(D - 11);
    await flush(4);
    expect(asked(fresh), "at least 10 ms before D").toEqual([]);
    await advance(11);
    await flush(4);
    expect(asked(fresh), "at D or later").toEqual(fresh);
  });
});

describe("SPEC-ADDON-155 §2 — a ruler drag the browser cancels", () => {
  it("SPEC-ADDON-155 §2 — rows near only during the drag are never asked for; rows dwelt on during it keep their pictures after the scroll position is put back", async () => {
    await open({ mediaController: makeController({ duration: L }).controller });
    expect(asked(rowsTimes([0, 1, 2, 3])), "opening").toEqual(rowsTimes([0, 1, 2, 3]));
    const el = ruler();
    const x = (time: number) => pointerX(time, L, WIDTH);

    firePointer(el, "pointerDown", { pointerType: "touch", clientX: x(12), buttons: 1 });
    firePointer(el, "pointerMove", { pointerType: "touch", clientX: x(72), buttons: 1 });
    await now();
    expect(nearRows()).toEqual([2, 3, 4, 5, 6]);
    fireEvent.touchMove(el, { touches: [{ clientX: x(132), clientY: 10 }] });
    firePointer(el, "pointerMove", { pointerType: "touch", clientX: x(132), buttons: 1 });
    await now();
    expect(nearRows()).toEqual([6, 7, 8, 9]);
    await advance(D + 50);
    await flush(4);
    expect(asked(rowsTimes([6, 7, 8, 9])), "dwelt on during the drag").toEqual(
      rowsTimes([6, 7, 8, 9]),
    );

    firePointer(el, "pointerCancel", { pointerType: "touch", clientX: x(132) });
    await now();
    expect(visibleRows()).toEqual([0, 1, 2]);
    await advance(1000);
    await flush(4);

    expect(asked(rowsTimes([4, 5])), "near only during the drag").toEqual([]);
    expect(asked(rowsTimes([6, 7, 8, 9])), "kept after the cancel").toEqual(
      rowsTimes([6, 7, 8, 9]),
    );
  });
});

describe("SPEC-ADDON-155 I3 — a dwell broken before D asks for nothing", () => {
  it("SPEC-ADDON-155 I3 — row 5 near for 100 ms, away for 50 ms, then near again is asked for D after it came back, never sooner; row 6 passing for 100 ms is never asked for", async () => {
    await open({ mediaController: makeController({ duration: L }).controller });
    await dwellOnRowFive();

    await advance(100);
    await wheelTo(ROW);
    expect(nearRows()).toEqual([0, 1, 2, 3, 4]);
    await advance(50);
    await wheelTo(2 * ROW);
    expect(nearRows()).toContain(5);

    await advance(50);
    await flush(4);
    expect(asked(rowTimes(5)), "at D after the first dwell began").toEqual([]);
    for (const top of [2 * ROW + 20, 2 * ROW]) {
      await wheelTo(top);
      await advance(50);
    }
    await advance(D - 1 - 150);
    await flush(4);
    expect(asked(rowTimes(5)), "at D − 1 ms of the second dwell").toEqual([]);

    await advance(1);
    await flush(4);
    expect(asked(rowTimes(5)), "at D of the second dwell").toEqual(rowTimes(5));

    await wheelTo(3 * ROW);
    expect(nearRows()).toContain(6);
    await advance(100);
    await wheelTo(2 * ROW + 20);
    expect(nearRows()).not.toContain(6);
    await advance(1000);
    await flush(4);
    expect(asked(rowTimes(6))).toEqual([]);
  });
});

describe("SPEC-ADDON-155 I4 — a row-step change keeps running dwells", () => {
  it("SPEC-ADDON-155 I4 — row 5 keeps its dwell across the change, and row 6, near only after it, is not asked for until it has dwelt D", async () => {
    await open({ mediaController: makeController({ duration: L }).controller });
    await dwellOnRowFive();

    await advance(100);
    act(() => setRowStep(66));
    await fireResize();
    await now();
    expect(scrollTopOf(scroller()), "the top row before the change stays the top row").toBe(2 * 66);
    expect(visibleRows()).toEqual([2, 3, 4, 5]);
    expect(asked(rowTimes(5)), "row 5 at the change").toEqual([]);
    expect(asked(rowTimes(6)), "row 6 at the change").toEqual([]);

    await advance(D - 1 - 100);
    await flush(4);
    expect(asked(rowTimes(5)), "row 5 at D − 1 ms of its dwell").toEqual([]);
    await advance(1);
    await flush(4);
    expect(asked(rowTimes(5)), "row 5 at D of its dwell").toEqual(rowTimes(5));
    expect(asked(rowTimes(6)), "row 6, 100 ms after it came near").toEqual([]);

    await advance(99);
    await flush(4);
    expect(asked(rowTimes(6)), "row 6 at D − 1 ms").toEqual([]);
    await advance(1);
    await flush(4);
    expect(asked(rowTimes(6)), "row 6 at D").toEqual(rowTimes(6));
  });
});

describe("SPEC-ADDON-155 I5 — once asked, a picture is neither asked again nor removed", () => {
  it("SPEC-ADDON-155 I5 — rows dwelt on, left and dwelt on again are each given their src once, and no drawn picture is removed", async () => {
    await open({ mediaController: makeController({ duration: L }).controller });
    expect(asked(rowsTimes([0, 1, 2, 3]))).toEqual(rowsTimes([0, 1, 2, 3]));
    const first = images().find((img) => img.getAttribute("src") === "/frame/f1?t=0");

    await wheelTo(ROW);
    await advance(10);
    await wheelTo(6 * ROW);
    expect(visibleRows()).toEqual([6, 7, 8]);
    for (const top of [6 * ROW + 20, 6 * ROW, 6 * ROW + 20, 6 * ROW, 6 * ROW + 20]) {
      await advance(50);
      await wheelTo(top);
    }
    await flush(4);
    expect(asked(rowsTimes([5, 6, 7, 8, 9]))).toEqual(rowsTimes([5, 6, 7, 8, 9]));

    await wheelTo(10);
    expect(nearRows()).toEqual([0, 1, 2, 3]);
    await advance(2 * D);
    await flush(4);

    const counts = new Map<string, number>();
    for (const { src } of srcLog) counts.set(src, (counts.get(src) ?? 0) + 1);
    expect(Array.from(counts.values()).filter((count) => count !== 1)).toEqual([]);
    expect(removedLog).toEqual([]);
    expect(images().find((img) => img.getAttribute("src") === "/frame/f1?t=0")).toBe(first);
    expect(asked(rowsTimes([5, 6, 7, 8, 9]))).toEqual(rowsTimes([5, 6, 7, 8, 9]));
  });
});

describe("SPEC-ADDON-155 I6 — leaving active and a file change discard running dwells", () => {
  function rowFiveSrcs(from: number): string[] {
    return srcLog
      .slice(from)
      .map(({ src }) => src)
      .filter((src) => rowTimes(5).some((time) => src.endsWith(`?t=${time}`)));
  }

  it("SPEC-ADDON-155 I6 — leaving the active state while row 5 dwells: no src is set while not active, row 5 is never asked for, and showing again asks for the new opening's near tiles at once", async () => {
    const props: Props = { mediaController: makeController({ duration: L }).controller };
    const view = await open(props);
    await dwellOnRowFive();

    await advance(100);
    view.rerender(section({ ...props, active: false }));
    await now();
    const mark = srcLog.length;
    await advance(500);
    await flush(4);
    expect(srcLog.slice(mark), "a src while not active").toEqual([]);

    view.rerender(section({ ...props, active: true }));
    await now();
    await now();

    expect(visibleRows()).toEqual([0, 1, 2]);
    expect(asked(rowsTimes([0, 1, 2, 3])), "the new opening's near tiles").toEqual(
      rowsTimes([0, 1, 2, 3]),
    );
    await advance(2 * D);
    await flush(4);
    expect(rowFiveSrcs(mark)).toEqual([]);
  });

  it("SPEC-ADDON-155 I6 — a file change while row 5 dwells: row 5 is never asked for, and the new file's near tiles are asked for at once", async () => {
    const view = await open({ mediaController: makeController({ duration: L }).controller });
    await dwellOnRowFive();

    await advance(100);
    const mark = srcLog.length;
    view.rerender(
      section({ fileId: "f2", mediaController: makeController({ duration: L }).controller }),
    );
    await now();
    await now();

    expect(visibleRows()).toEqual([0, 1, 2]);
    expect(
      rowsTimes([0, 1, 2, 3]).filter((time) => !srcs().includes(`/frame/f2?t=${time}`)),
      "the new file's near tiles without a src",
    ).toEqual([]);
    await advance(2 * D);
    await flush(4);
    expect(rowFiveSrcs(mark)).toEqual([]);
  });
});

describe("SPEC-ADDON-155 I7 — the tile's img has no loading attribute", () => {
  it.each([
    { name: "a native video", mimeType: "video/mp4" },
    { name: "a .loft", mimeType: LOFT_MIME },
  ])("SPEC-ADDON-155 I7 — $name", async ({ mimeType }) => {
    await open({ mimeType, mediaController: makeController({ duration: L }).controller });
    await advance(2 * D);
    await flush(4);

    expect(images().length).toBeGreaterThan(0);
    expect(images().filter((img) => img.hasAttribute("loading")).map((img) => img.getAttribute("loading"))).toEqual([]);
  });
});

describe("SPEC-ADDON-155 I8 — a failed picture stays empty until the next start event", () => {
  it.each([
    { name: "a native frame", mimeType: "video/mp4", route: "frame" },
    { name: "a .loft tile", mimeType: LOFT_MIME, route: "scene-tile" },
  ])(
    "SPEC-ADDON-155 I8 — $name that fails stays empty however it dwells again, leaves the other pictures alone, and is asked for at once after the next start event",
    async ({ mimeType, route }) => {
      const { controller } = makeController({ duration: L });
      const props: Props = { mimeType, mediaController: controller };
      const view = await open(props);
      const failing = images().find((img) => img.getAttribute("src") === `/${route}/f1?t=5`);
      expect(failing, "the picture at 0:05 at opening").toBeDefined();
      const others = images()
        .filter((img) => img !== failing)
        .map((img) => [img, img.getAttribute("src")] as const);

      fireEvent.error(failing!);
      expect(tile(5).querySelector("img")).toBeNull();
      expect(
        others.filter(([img, src]) => !img.isConnected || img.getAttribute("src") !== src),
      ).toEqual([]);

      await wheelTo(6 * ROW);
      await advance(2 * D);
      await wheelTo(10);
      expect(nearRows()).toContain(1);
      await advance(2 * D);
      await flush(4);
      expect(tile(5).querySelector("img")).toBeNull();

      view.rerender(section({ ...props, active: false }));
      await now();
      view.rerender(section({ ...props, active: true }));
      await now();
      await now();
      expect(visibleRows()).toEqual([0, 1, 2]);
      expect(srcs()).toContain(`/${route}/f1?t=5`);
    },
  );
});

describe("SPEC-ADDON-155 I9 — what ends opening", () => {
  /**
   * After the input, a row-step change makes new tiles near. Opening over, they wait
   * D; still opening, they are asked for at once.
   */
  async function newlyNearAfterRowStep(before: number[]): Promise<number[]> {
    act(() => setRowStep(66));
    await fireResize();
    await now();
    return rowsTimes(nearRows()).filter((time) => !before.includes(time));
  }

  it.each([
    {
      name: "a reader's wheel",
      input: async () => {
        await wheelTo(ROW);
      },
      afterInput: true,
    },
    {
      name: "a pointer press on the ruler",
      input: async () => {
        firePointer(ruler(), "pointerDown", { clientX: pointerX(122, L, WIDTH), buttons: 1 });
        firePointer(ruler(), "pointerUp", { clientX: pointerX(122, L, WIDTH) });
        await now();
      },
      afterInput: false,
    },
    ...["ArrowLeft", "ArrowRight", "Home", "End", "Enter"].map((key) => ({
      name: `the ruler key ${key}`,
      input: async () => {
        act(() => ruler().focus());
        fireEvent.keyDown(ruler(), { key });
        await now();
      },
      afterInput: false,
    })),
  ])(
    "SPEC-ADDON-155 I9 — $name ends opening: a tile that first becomes near after it has no src at D − 1 ms and has one at D",
    async ({ input, afterInput }) => {
      await open({ mediaController: makeController({ duration: L }).controller });
      expect(visibleRows()).toEqual([0, 1, 2]);
      let before = rowsTimes(nearRows());

      await input();
      if (afterInput) before = rowsTimes(nearRows());
      const fresh = await newlyNearAfterRowStep(before);
      expect(fresh.length, "the input and the row-step change made no tile near").toBeGreaterThan(0);

      expect(asked(fresh), "with no time advanced").toEqual([]);
      await advance(D - 1);
      await flush(4);
      expect(asked(fresh), "at D − 1 ms").toEqual([]);
      await advance(1);
      await flush(4);
      expect(asked(fresh), "at D").toEqual(fresh);
    },
  );

  it("SPEC-ADDON-155 I9 — a re-placement for a row-step change during opening keeps opening: the tiles it makes near have a src with no time advanced", async () => {
    await open({ mediaController: makeController({ duration: L, currentTime: 100 }).controller });
    expect(visibleRows()).toEqual([5, 6, 7]);

    act(() => setRowStep(66));
    await fireResize();
    await now();

    expect(scrollTopOf(scroller()), "the playing row placed again").toBe(5 * 66);
    expect(nearRows()).toEqual([4, 5, 6, 7, 8, 9]);
    expect(asked(rowTimes(9))).toEqual(rowTimes(9));
  });

  it("SPEC-ADDON-155 I9 — a controller appearing during opening places the playing row and its near tiles have a src with no time advanced", async () => {
    const view = await open({ mediaController: null });
    expect(visibleRows()).toEqual([0, 1, 2]);

    view.rerender(
      section({ mediaController: makeController({ duration: L, currentTime: 100 }).controller }),
    );
    await now();
    await now();

    expect(visibleRows()).toEqual([5, 6, 7]);
    expect(asked(rowsTimes([4, 5, 6, 7, 8]))).toEqual(rowsTimes([4, 5, 6, 7, 8]));
  });

  it("SPEC-ADDON-155 I9 — following during playback in opening: the rows its placement makes near have a src as soon as it lands", async () => {
    const { controller, state } = makeController({ duration: L, currentTime: 0, paused: true });
    await open({ mediaController: controller });
    expect(visibleRows()).toEqual([0, 1, 2]);

    state.paused = false;
    state.currentTime = 45;
    for (let waited = 0; visibleRows()[0] === 0 && waited < 2000; waited += 10) {
      await advance(10);
    }
    await flush(4);

    expect(visibleRows()).toEqual([2, 3, 4]);
    expect(asked(rowTimes(5))).toEqual(rowTimes(5));
  });

  it("SPEC-ADDON-155 I9 — hovering the ruler does not end opening: a later re-placement's near tiles have a src with no time advanced", async () => {
    await open({ mediaController: makeController({ duration: L, currentTime: 100 }).controller });
    firePointer(ruler(), "pointerMove", { pointerType: "mouse", clientX: pointerX(30, L, WIDTH) });
    await now();
    firePointer(ruler(), "pointerMove", { pointerType: "mouse", clientX: pointerX(140, L, WIDTH) });
    await now();
    firePointer(ruler(), "pointerLeave", { pointerType: "mouse" });
    await now();

    act(() => setRowStep(66));
    await fireResize();
    await now();

    expect(scrollTopOf(scroller())).toBe(5 * 66);
    expect(asked(rowTimes(9))).toEqual(rowTimes(9));
  });
});
