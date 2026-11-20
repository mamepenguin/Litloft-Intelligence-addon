import { describe, it, expect, vi, beforeEach, afterEach, beforeAll } from "vitest";
import { act, fireEvent, render, screen } from "@testing-library/react";

import { formatDuration } from "@/lib/format";

// Strings come from the real locale files, so the chip's text is whatever key
// the addon's messages give it, read under the namespace the section asks for.
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
import enMessages from "@/addons/intelligence/messages/en.json";
import jaMessages from "@/addons/intelligence/messages/ja.json";
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
  makeController,
  makeRows,
  makeRun,
  makeScene,
  pointerX,
  ruler,
  sceneCardNamed,
  SceneSectionProps,
  tile,
  tileLabels,
  timestamps,
} from "./helpers";
import {
  CHIP,
  ROW,
  VIEW,
  WIDTH,
  chip,
  fireResize,
  installLayout,
  presetScrollTop,
  resetLayout,
  scroller,
  scrollTopOf,
  tileElements,
  visibleRows,
  wheelTo,
  whollyVisible,
  writes,
} from "./scrollListLayout";

/** 30 frames 5 s apart: ten rows of three, three of them visible at once. */
const N = 30;
const ROWS = makeRows(N);
const L = 150;

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

async function settle(ms = 0): Promise<void> {
  await flush(8);
  await advance(ms);
  await flush(8);
}

async function open(props: Props = {}) {
  const view = render(section(props));
  await settle(0);
  await settle(0);
  return view;
}

/** Lets the shared media clock read the controller again. */
async function tick(mutate?: () => void): Promise<void> {
  await advance(500, mutate);
  await flush(4);
}

function withScenes(scenes = SCENES) {
  return indexState({ active_run: makeRun({ succeeded_count: scenes.length }), scenes });
}

const SCENES = [
  makeScene({ ordering: 0, start_time: 0, section_start: 0, scene_label: "Alpha" }),
  makeScene({ ordering: 1, start_time: 70, section_start: 70, scene_label: "Bravo" }),
  makeScene({ ordering: 2, start_time: 120, section_start: 120, scene_label: "Charlie" }),
];

const labels = (times: number[]) => times.map((time) => formatDuration(time));

beforeAll(() => {
  installPointerEvents();
  installLayout();
});

beforeEach(() => {
  vi.clearAllMocks();
  vi.useFakeTimers();
  resetLayout();
  vi.mocked(getClipTimestamps).mockResolvedValue(timestamps(ROWS));
  vi.mocked(getVideoVisualIndex).mockResolvedValue(withScenes());
  vi.mocked(usePolicy).mockReturnValue({ enabled: true, isLoading: false });
});

afterEach(() => {
  vi.useRealTimers();
});

describe("SPEC-ADDON-152 I1 — every frame is drawn", () => {
  it("SPEC-ADDON-152 I1 — one, three, thirty and 500 frames: one tile per frame, in time order", async () => {
    const drawn: Record<number, string[]> = {};
    const expected: Record<number, string[]> = {};
    for (const n of [1, 3, 30, 500]) {
      const rows = makeRows(n);
      vi.mocked(getClipTimestamps).mockResolvedValue(timestamps(rows));
      const view = await open({ mediaController: makeController({ duration: n * 5 }).controller });
      drawn[n] = tileLabels();
      expected[n] = labels(rows.map((row) => row.start));
      view.unmount();
    }

    expect(drawn).toEqual(expected);
  });
});

describe("SPEC-ADDON-152 §9 — the chip's text", () => {
  /** Every string in a message file with its dotted key. */
  function entries(messages: unknown, prefix = ""): [string, string][] {
    if (typeof messages === "string") return [[prefix, messages]];
    if (messages === null || typeof messages !== "object") return [];
    return Object.entries(messages as Record<string, unknown>).flatMap(([key, child]) =>
      entries(child, prefix ? `${prefix}.${key}` : key),
    );
  }

  it("SPEC-ADDON-152 §9 — one key of its own reads Back to current position in en and 現在位置に戻る in ja; the transcript's key is not it", () => {
    const en = new Map(entries(enMessages));
    const ja = new Map(entries(jaMessages));
    const keys = Array.from(en.entries())
      .filter(([key, value]) => value === "Back to current position" && !key.endsWith("transcriptResumeFollowing"))
      .map(([key]) => key);

    expect(keys.map((key) => [key, ja.get(key)])).toEqual(
      keys.length === 1 ? [[keys[0], "現在位置に戻る"]] : [["<one new key>", "現在位置に戻る"]],
    );
  });
});

describe("SPEC-ADDON-152 I3 I4 — following and the chip", () => {
  async function following(currentTime = 0) {
    const clock = makeController({ duration: L, currentTime, paused: true });
    const view = await open({ mediaController: clock.controller });
    return { ...clock, view };
  }

  it("SPEC-ADDON-152 I4 — no chip while following; a wheel on the list draws it between the ruler and the first tile", async () => {
    await following();
    expect(chip()).toBeNull();

    await wheelTo(5 * ROW);

    const drawn = chip();
    expect(drawn, `no "${CHIP}" chip after a reader's wheel`).not.toBeNull();
    const follows = (a: Node, b: Node) =>
      (a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING) !== 0;
    expect(follows(ruler(), drawn!)).toBe(true);
    expect(follows(drawn!, tileElements()[0])).toBe(true);
  });

  it.each([
    {
      name: "a touchmove on a tile",
      input: () => fireEvent.touchMove(tileElements()[4], { touches: [{ clientX: 10, clientY: 10 }] }),
    },
    {
      name: "a mouse press on the scroller itself",
      input: () =>
        fireEvent.pointerDown(scroller(), { pointerType: "mouse", button: 0, buttons: 1 }),
    },
    {
      name: "PageDown with focus on a tile",
      input: () => {
        act(() => tileElements()[0].focus());
        fireEvent.keyDown(tileElements()[0], { key: "PageDown" });
      },
    },
    {
      name: "End with focus on a tile",
      input: () => {
        act(() => tileElements()[0].focus());
        fireEvent.keyDown(tileElements()[0], { key: "End" });
      },
    },
    {
      name: "PageUp on the scroller itself",
      input: () => {
        scroller().tabIndex = -1;
        act(() => scroller().focus());
        fireEvent.keyDown(scroller(), { key: "PageUp" });
      },
    },
    {
      name: "a middle-button press on a tile",
      input: () =>
        fireEvent.pointerDown(tileElements()[1], { pointerType: "mouse", button: 1, buttons: 4 }),
    },
    {
      name: "focus moving to a tile that is not wholly visible",
      input: () => act(() => tileElements()[5 * 3].focus()),
    },
  ])("SPEC-ADDON-152 I3 — $name clears following and draws the chip", async ({ input }) => {
    await following();
    expect(chip()).toBeNull();
    expect(tileElements()).toHaveLength(N);

    input();
    await settle(0);

    expect(chip(), `no "${CHIP}" chip`).not.toBeNull();
  });

  it("SPEC-ADDON-152 I3 — the section's own scroll, a pause, a seek from elsewhere, a resize, Space and the arrows on a tile leave following set", async () => {
    const { state } = await following(0);
    expect(chip()).toBeNull();

    await tick(() => {
      state.paused = false;
      state.currentTime = 1;
    });
    await tick(() => {
      state.paused = true;
    });
    await tick(() => {
      state.currentTime = 100;
    });
    await settle(0);
    await fireResize();
    await settle(0);
    const inView = tileElements().find((el) => {
      const rect = el.getBoundingClientRect();
      return rect.top >= 0 && rect.bottom <= VIEW;
    });
    expect(inView, "a wholly visible tile").toBeDefined();
    act(() => inView!.focus());
    for (const key of [" ", "ArrowRight", "ArrowLeft", "ArrowUp", "ArrowDown"]) {
      fireEvent.keyDown(document.activeElement!, { key });
    }
    await settle(0);
    expect(chip()).toBeNull();

    await wheelTo(0);
    expect(chip(), "a reader's wheel after them draws the chip").not.toBeNull();
  });

  it.each([
    { name: "a tile", activate: () => fireEvent.click(tile(10)) },
    {
      name: "the scene card",
      activate: () => fireEvent.click(sceneCardNamed(/Alpha|Bravo|Charlie/)!),
    },
    { name: "the chip", activate: () => fireEvent.click(chip()!) },
    {
      name: "Enter on the ruler with a ruler point",
      activate: () => {
        act(() => ruler().focus());
        fireEvent.keyDown(ruler(), { key: "ArrowRight" });
        fireEvent.keyDown(ruler(), { key: "Enter" });
      },
    },
  ])("SPEC-ADDON-152 I3 — $name sets following again", async ({ activate }) => {
    await following(0);
    await wheelTo(6 * ROW);
    expect(chip()).not.toBeNull();

    activate();
    await settle(0);

    expect(chip()).toBeNull();
  });

  it("SPEC-ADDON-152 I3 — the chip places the playing row: row 0 goes to the top of the list", async () => {
    await following(0);
    await wheelTo(6 * ROW);
    expect(visibleRows()).toEqual([6, 7, 8]);

    fireEvent.click(chip()!);
    await settle(0);

    expect(scrollTopOf(scroller())).toBe(0);
    expect(visibleRows()).toEqual([0, 1, 2]);
  });

  it("SPEC-ADDON-152 §2 — without a mediaController a wheel draws no chip and a tile plays through videoRef", async () => {
    const video = document.createElement("video");
    const play = vi.fn().mockResolvedValue(undefined);
    video.play = play as unknown as HTMLMediaElement["play"];
    await open({ mediaController: null, videoRef: { current: video } });
    expect(tileElements()).toHaveLength(N);

    await wheelTo(4 * ROW);
    expect(chip()).toBeNull();

    fireEvent.click(tile(100));
    expect(video.currentTime).toBe(100);
    expect(play).toHaveBeenCalled();
  });

  it("SPEC-ADDON-152 §2 — when every frame fits, a wheel draws no chip and a ruler press neither seeks nor clears following", async () => {
    vi.mocked(getClipTimestamps).mockResolvedValue(timestamps(makeRows(6)));
    const { controller, seek, play } = makeController({ duration: 30, currentTime: 0 });
    await open({ mediaController: controller });
    expect(tileElements()).toHaveLength(6);

    await wheelTo(0);
    expect(chip()).toBeNull();

    const el = ruler();
    firePointer(el, "pointerDown", { clientX: pointerX(22, 30, WIDTH), buttons: 1 });
    firePointer(el, "pointerUp", { clientX: pointerX(22, 30, WIDTH) });
    fireEvent.click(el, { clientX: pointerX(22, 30, WIDTH) });
    await settle(0);

    expect(seek).not.toHaveBeenCalled();
    expect(play).not.toHaveBeenCalled();
    expect(ruler()).toHaveAttribute("aria-valuenow", "20");
    expect(chip()).toBeNull();
  });
});

describe("SPEC-ADDON-152 I5 — the scene card's point", () => {
  it("SPEC-ADDON-152 I5 — the playback time while following, the top row's first frame after a reader's scroll, the ruler point once set", async () => {
    const { controller } = makeController({ duration: L, currentTime: 10 });
    await open({ mediaController: controller });
    expect(sceneCardNamed(/Alpha/)).not.toBeNull();
    expect(ruler()).toHaveAttribute("aria-valuenow", "10");

    await wheelTo(5 * ROW);
    expect(visibleRows()[0]).toBe(5);
    await settle(0);

    expect(sceneCardNamed(/Bravo/), "the card at the top row's first frame, 1:15").not.toBeNull();
    expect(ruler()).toHaveAttribute("aria-valuenow", "75");

    act(() => ruler().focus());
    fireEvent.keyDown(ruler(), { key: "End" });
    await settle(0);
    expect(sceneCardNamed(/Charlie/)).not.toBeNull();
    expect(ruler()).toHaveAttribute("aria-valuenow", "145");
  });

  it("SPEC-ADDON-152 I5 — without a controller the card is at the top row's first frame from the start", async () => {
    await open({ mediaController: null });
    expect(sceneCardNamed(/Alpha/)).not.toBeNull();

    await wheelTo(8 * ROW);
    await settle(0);

    expect(visibleRows()[0]).toBe(7);
    expect(sceneCardNamed(/Bravo/), "the top row starts at 1:45").not.toBeNull();
  });

  it("SPEC-ADDON-152 I5 — in flow form the card is one line: its label and time, never visible_text", async () => {
    vi.mocked(getVideoVisualIndex).mockResolvedValue(
      withScenes([
        makeScene({
          start_time: 0,
          section_start: 0,
          scene_label: "Alpha",
          visible_text: "WELCOME TO THE HARBOUR",
        }),
      ]),
    );
    const { controller } = makeController({ duration: L, currentTime: 10 });
    render(<div data-inspector-scroller="">{section({ mediaController: controller })}</div>);
    await settle(0);
    await settle(0);

    expect(tileElements()).toHaveLength(N);
    const card = sceneCardNamed(/Alpha/);
    expect(card).not.toBeNull();
    expect(card!.textContent).toContain("0:00");
    expect(screen.queryAllByText(/WELCOME TO THE HARBOUR/)).toEqual([]);
  });
});

describe("SPEC-ADDON-155 — pictures (SPEC-ADDON-152 I6 as revised)", () => {
  function imgAt(time: number): HTMLImageElement | undefined {
    return images().find((img) => new RegExp(`[?&]t=${time}(?:&|$)`).test(img.src));
  }

  it("SPEC-ADDON-155 I1 — on opening, the visible rows and the row within one row step below are asked for with no time advanced", async () => {
    await open({ mediaController: makeController({ duration: L }).controller });
    expect(visibleRows()).toEqual([0, 1, 2]);

    const times = imgTimes();
    for (const time of [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]) {
      expect(times, `the picture at ${time}s`).toContain(time);
    }
    for (const time of [75, 100, 145]) {
      expect(times, `the picture at ${time}s, rows away`).not.toContain(time);
    }
  });

  it("SPEC-ADDON-155 I2 I5 — while scrolling, a row first near at the last wheel is asked for 200 ms after it, not at 199 ms, and drawn pictures stay", async () => {
    await open({ mediaController: makeController({ duration: L }).controller });
    const first = imgAt(0);
    expect(first).toBeDefined();

    for (const top of [2 * ROW, 4 * ROW, 6 * ROW]) {
      await wheelTo(top);
      await advance(100);
      await flush(4);
      expect(imgTimes()).not.toContain(120);
    }
    expect(visibleRows()).toEqual([6, 7, 8]);
    await advance(99);
    await flush(4);
    expect(imgTimes()).not.toContain(120);

    await advance(1);
    await flush(4);
    for (const time of [75, 90, 105, 120, 135, 145]) {
      expect(imgTimes(), `the picture at ${time}s`).toContain(time);
    }
    expect(imgAt(0), "a drawn picture stays drawn").toBe(first);

    await wheelTo(0);
    await advance(200);
    await flush(4);
    expect(imgAt(0), "a picture is requested once").toBe(first);
    expect(images().filter((img) => /[?&]t=0(?:&|$)/.test(img.src))).toHaveLength(1);
  });

  it("SPEC-ADDON-155 I8 — a native frame that fails is not asked for again until the next showing, which asks for it at once", async () => {
    const { controller } = makeController({ duration: L });
    const view = await open({ mediaController: controller });
    fireEvent.error(imgAt(5)!);
    expect(tile(5).querySelector("img")).toBeNull();
    expect(tile(5)).toHaveAttribute("aria-label", "Play from 0:05");

    await wheelTo(6 * ROW);
    await advance(200);
    await flush(4);
    expect(imgTimes()).toContain(120);
    await wheelTo(0);
    await advance(200);
    await flush(4);
    expect(tile(5).querySelector("img")).toBeNull();

    view.rerender(section({ mediaController: controller, active: false }));
    await settle(0);
    view.rerender(section({ mediaController: controller, active: true }));
    await settle(0);
    await settle(0);
    expect(imgTimes()).toContain(5);
  });
});

describe("SPEC-ADDON-152 I7 — a hidden section leaves a shared scroller alone", () => {
  it("SPEC-ADDON-152 I7 — while another tab is shown nothing writes the shared scroller; showing Scenes places the list", async () => {
    const { controller, state } = makeController({ duration: L, currentTime: 0 });
    const shared = (active: boolean) => (
      <div data-inspector-scroller="" data-testid="shared">
        {section({ mediaController: controller, active })}
      </div>
    );
    const view = render(shared(true));
    await settle(0);
    await settle(0);
    expect(tileElements()).toHaveLength(N);

    view.rerender(shared(false));
    await settle(0);
    const el = screen.getByTestId("shared");
    presetScrollTop(el, 500);
    writes.length = 0;

    await tick(() => {
      state.paused = false;
      state.currentTime = 140;
    });
    await fireResize();
    await settle(200);
    expect(writes.filter((write) => write.el === el)).toEqual([]);
    expect(scrollTopOf(el)).toBe(500);

    await tick(() => {
      state.paused = true;
      state.currentTime = 0;
    });
    view.rerender(shared(true));
    await settle(0);

    expect(writes.filter((write) => write.el === el).length).toBeGreaterThan(0);
    expect(scrollTopOf(el)).toBeLessThan(ROW);
  });

  it.each([
    { name: "hiding and showing the panel", change: { active: false }, back: { active: true } },
    { name: "a file change", change: { fileId: "f2" }, back: null },
  ])("SPEC-ADDON-152 I7 — $name sets following and removes the ruler point", async ({ change, back }) => {
    vi.mocked(getClipTimestamps).mockImplementation(((fileId: string) =>
      Promise.resolve(timestamps(ROWS, fileId))) as never);
    const { controller } = makeController({ duration: L, currentTime: 10 });
    const view = await open({ mediaController: controller });
    const el = ruler();
    firePointer(el, "pointerDown", { clientX: pointerX(122, L, WIDTH), buttons: 1 });
    firePointer(el, "pointerUp", { clientX: pointerX(122, L, WIDTH) });
    await settle(0);
    expect(ruler()).toHaveAttribute("aria-valuenow", "120");
    expect(chip()).not.toBeNull();

    view.rerender(section({ mediaController: controller, ...change }));
    await settle(0);
    await settle(0);
    if (back) {
      view.rerender(section({ mediaController: controller, ...back }));
      await settle(0);
      await settle(0);
    }

    expect(chip()).toBeNull();
    expect(ruler()).toHaveAttribute("aria-valuenow", "10");
    expect(whollyVisible(0)).toBe(true);
  });
});

describe("SPEC-ADDON-152 §1.3 — showing starts at the playing row", () => {
  it("SPEC-ADDON-152 §1.3 — with a controller the playing row is placed second in the visible area", async () => {
    const { controller } = makeController({ duration: L, currentTime: 100 });
    await open({ mediaController: controller });

    expect(tileElements()).toHaveLength(N);
    expect(visibleRows()).toEqual([5, 6, 7]);
    expect(scrollTopOf(scroller())).toBe(5 * ROW);
    expect(chip()).toBeNull();
  });

  it("SPEC-ADDON-152 §2 — a playback time before the first frame places row 0 at the top", async () => {
    vi.mocked(getClipTimestamps).mockResolvedValue(timestamps(makeRows(N, 5, 3)));
    const { controller } = makeController({ duration: L, currentTime: 1 });
    await open({ mediaController: controller });

    expect(tileElements()).toHaveLength(N);
    expect(scrollTopOf(scroller())).toBe(0);
  });

  it("SPEC-ADDON-152 I2 — while following, a playing row already wholly visible does not scroll the list; one that leaves view is placed", async () => {
    const { controller, state } = makeController({ duration: L, currentTime: 0, paused: false });
    await open({ mediaController: controller });
    expect(visibleRows()).toEqual([0, 1, 2]);
    const before = scrollTopOf(scroller());

    await tick(() => {
      state.currentTime = 40;
    });
    expect(scrollTopOf(scroller())).toBe(before);
    expect(visibleRows()).toEqual([0, 1, 2]);

    await tick(() => {
      state.currentTime = 45;
    });
    expect(visibleRows()).toEqual([2, 3, 4]);
    expect(whollyVisible(3)).toBe(true);
  });
});
