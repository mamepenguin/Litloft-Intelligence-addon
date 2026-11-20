import { describe, it, expect, vi, beforeEach, afterEach, beforeAll } from "vitest";
import { act, fireEvent, render, screen } from "@testing-library/react";

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
  indexState,
  installPointerEvents,
  LOFT_MIME,
  makeController,
  makeRows,
  makeRun,
  makeScene,
  pointerX,
  ruler,
  sceneCardNamed,
  SceneSectionProps,
  timestamps,
} from "./helpers";
import {
  ROW,
  WIDTH,
  chip,
  installLayout,
  resetLayout,
  scroller,
  scrollTopOf,
  tileElements,
  visibleRows,
  wheelTo,
} from "./scrollListLayout";

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

const SCENES = [
  makeScene({ ordering: 0, start_time: 0, section_start: 0, scene_label: "Alpha" }),
  makeScene({ ordering: 1, start_time: 70, section_start: 70, scene_label: "Bravo" }),
  makeScene({ ordering: 2, start_time: 120, section_start: 120, scene_label: "Charlie" }),
];

function withScenes(scenes = SCENES) {
  return indexState({ active_run: makeRun({ succeeded_count: scenes.length }), scenes });
}

const x = (time: number) => pointerX(time, L, WIDTH);

function click(el: Element, time: number, pointerType = "mouse"): void {
  firePointer(el, "pointerDown", { pointerType, clientX: x(time), buttons: 1 });
  firePointer(el, "pointerUp", { pointerType, clientX: x(time) });
  fireEvent.click(el, { clientX: x(time) });
}

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

describe("SPEC-ADDON-153 I9 — no ruler input seeks except Enter", () => {
  it("SPEC-ADDON-153 I9 — mouse movement without a button moves nothing: no point, no card, no scroll, no seek", async () => {
    const { controller, seek, play } = makeController({ duration: L, currentTime: 10 });
    await open({ mediaController: controller });
    expect(tileElements()).toHaveLength(N);
    const top = scrollTopOf(scroller());

    const el = ruler();
    firePointer(el, "pointerMove", { clientX: x(100) });
    firePointer(el, "pointerMove", { clientX: x(130) });
    await settle(0);

    expect(ruler()).toHaveAttribute("aria-valuenow", "10");
    expect(sceneCardNamed(/Alpha/)).not.toBeNull();
    expect(scrollTopOf(scroller())).toBe(top);
    expect(chip()).toBeNull();

    firePointer(el, "pointerLeave", { clientX: x(130) });
    await settle(0);
    expect(ruler()).toHaveAttribute("aria-valuenow", "10");
    expect(seek).not.toHaveBeenCalled();
    expect(play).not.toHaveBeenCalled();
  });

  it("SPEC-ADDON-153 I9 — a mouse click sets the ruler point at the frame under it, places its row, clears following and never seeks", async () => {
    const { controller, seek, play } = makeController({ duration: L, currentTime: 10 });
    await open({ mediaController: controller });

    click(ruler(), 122);
    await settle(0);

    expect(seek).not.toHaveBeenCalled();
    expect(play).not.toHaveBeenCalled();
    expect(ruler()).toHaveAttribute("aria-valuenow", "120");
    expect(sceneCardNamed(/Charlie/)).not.toBeNull();
    expect(visibleRows()).toContain(8);
    expect(chip()).not.toBeNull();
  });

  it("SPEC-ADDON-153 I9 — a mouse drag moves the ruler point with the button held and the release ends it", async () => {
    const { controller, seek } = makeController({ duration: L, currentTime: 10 });
    await open({ mediaController: controller });
    const el = ruler();

    firePointer(el, "pointerDown", { clientX: x(32), buttons: 1 });
    expect(ruler()).toHaveAttribute("aria-valuenow", "30");
    firePointer(el, "pointerMove", { clientX: x(72), buttons: 1 });
    await settle(0);
    expect(ruler()).toHaveAttribute("aria-valuenow", "70");
    firePointer(el, "pointerMove", { clientX: x(141), buttons: 1 });
    await settle(0);
    expect(ruler()).toHaveAttribute("aria-valuenow", "140");
    expect(visibleRows()).toContain(9);
    firePointer(el, "pointerUp", { clientX: x(141) });
    firePointer(el, "pointerMove", { clientX: x(20) });
    await settle(0);

    expect(ruler()).toHaveAttribute("aria-valuenow", "140");
    expect(seek).not.toHaveBeenCalled();
  });

  it.each([
    { name: "a touch tap", pointerType: "touch" },
    { name: "a pen tap", pointerType: "pen" },
  ])("SPEC-ADDON-153 I9 — $name sets the ruler point, places its row and clears following", async ({ pointerType }) => {
    const { controller, seek, play } = makeController({ duration: L, currentTime: 10 });
    await open({ mediaController: controller });

    click(ruler(), 101, pointerType);
    await settle(0);

    expect(seek).not.toHaveBeenCalled();
    expect(play).not.toHaveBeenCalled();
    expect(ruler()).toHaveAttribute("aria-valuenow", "100");
    expect(visibleRows()).toContain(6);
    expect(chip()).not.toBeNull();
  });

  it("SPEC-ADDON-153 I9 — in flow form a touch the browser takes for vertical scrolling sets no ruler point and is a reader's scroll", async () => {
    const { controller } = makeController({ duration: L, currentTime: 10 });
    render(<div data-inspector-scroller="">{section({ mediaController: controller })}</div>);
    await settle(0);
    await settle(0);
    expect(tileElements()).toHaveLength(N);
    const el = ruler();

    firePointer(el, "pointerDown", { pointerType: "touch", clientX: x(102), clientY: 10 });
    fireEvent.touchMove(el, { touches: [{ clientX: x(102) + 3, clientY: 60 }] });
    firePointer(el, "pointerCancel", { pointerType: "touch", clientX: x(102) + 3, clientY: 60 });
    await settle(0);

    expect(ruler()).not.toHaveAttribute("aria-valuenow", "100");
    expect(chip(), "the vertical swipe stopped following").not.toBeNull();
  });

  it("SPEC-ADDON-153 I9 — a pointercancel after an 8 px drag puts back the scroll, following and ruler point it started from", async () => {
    const { controller } = makeController({ duration: L, currentTime: 10 });
    await open({ mediaController: controller });
    const startTop = scrollTopOf(scroller());
    const el = ruler();

    firePointer(el, "pointerDown", { pointerType: "touch", clientX: x(12), buttons: 1 });
    firePointer(el, "pointerMove", { pointerType: "touch", clientX: x(72), buttons: 1 });
    fireEvent.touchMove(el, { touches: [{ clientX: x(132), clientY: 10 }] });
    firePointer(el, "pointerMove", { pointerType: "touch", clientX: x(132), buttons: 1 });
    await settle(0);
    expect(ruler()).toHaveAttribute("aria-valuenow", "130");
    expect(scrollTopOf(scroller()), "the drag placed its row").not.toBe(startTop);

    firePointer(el, "pointerCancel", { pointerType: "touch", clientX: x(132) });
    await settle(0);

    expect(ruler()).toHaveAttribute("aria-valuenow", "10");
    expect(scrollTopOf(scroller())).toBe(startTop);
    expect(chip()).toBeNull();
  });

  it("SPEC-ADDON-153 I9 — the first arrow starts from the playing frame while following, the top row's first frame after a reader's scroll, and the ruler point once set", async () => {
    const { controller } = makeController({ duration: L, currentTime: 22 });
    await open({ mediaController: controller });
    act(() => ruler().focus());

    fireEvent.keyDown(ruler(), { key: "ArrowRight" });
    expect(ruler()).toHaveAttribute("aria-valuenow", "25");

    await wheelTo(5 * ROW);
    await settle(0);
    expect(ruler()).toHaveAttribute("aria-valuenow", "75");
    act(() => ruler().focus());
    fireEvent.keyDown(ruler(), { key: "ArrowRight" });
    expect(ruler()).toHaveAttribute("aria-valuenow", "80");
    fireEvent.keyDown(ruler(), { key: "ArrowLeft" });
    fireEvent.keyDown(ruler(), { key: "ArrowLeft" });
    expect(ruler()).toHaveAttribute("aria-valuenow", "70");

    click(ruler(), 52);
    await settle(0);
    act(() => ruler().focus());
    fireEvent.keyDown(ruler(), { key: "ArrowLeft" });
    expect(ruler()).toHaveAttribute("aria-valuenow", "45");
  });

  it("SPEC-ADDON-153 I9 — Home and End reach the first and last frame and place their rows", async () => {
    const { controller, seek } = makeController({ duration: L, currentTime: 10 });
    await open({ mediaController: controller });
    act(() => ruler().focus());

    fireEvent.keyDown(ruler(), { key: "End" });
    await settle(0);
    expect(ruler()).toHaveAttribute("aria-valuenow", "145");
    expect(visibleRows()).toContain(9);

    fireEvent.keyDown(ruler(), { key: "Home" });
    await settle(0);
    expect(ruler()).toHaveAttribute("aria-valuenow", "0");
    expect(visibleRows()).toContain(0);
    expect(seek).not.toHaveBeenCalled();
  });

  it("SPEC-ADDON-153 I9 — Enter plays from the ruler point and does nothing without one", async () => {
    const { controller, seek, play } = makeController({ duration: L, currentTime: 10 });
    await open({ mediaController: controller });
    act(() => ruler().focus());

    fireEvent.keyDown(ruler(), { key: "Enter" });
    expect(seek).not.toHaveBeenCalled();

    click(ruler(), 122);
    await settle(0);
    expect(seek).not.toHaveBeenCalled();
    act(() => ruler().focus());
    fireEvent.keyDown(ruler(), { key: "Enter" });

    expect(seek).toHaveBeenCalledTimes(1);
    expect(seek.mock.calls[0][0]).toBe(120);
    expect(play).toHaveBeenCalled();
  });
});

describe("SPEC-ADDON-153 I8 — the band", () => {
  /** The ruler's band, drawn with an inline left and width. */
  function band(): { left: number; right: number } {
    const el = Array.from(ruler().querySelectorAll<HTMLElement>("*")).find(
      (node) => node.style.left !== "" && node.style.width !== "",
    );
    if (!el) throw new Error("no band is drawn");
    const left = Number.parseFloat(el.style.left);
    return { left, right: left + Number.parseFloat(el.style.width) };
  }

  it("SPEC-ADDON-153 I8 — the band covers the visible rows and moves when the list scrolls", async () => {
    const { controller } = makeController({ duration: L, currentTime: 0 });
    await open({ mediaController: controller });
    expect(visibleRows()).toEqual([0, 1, 2]);
    expect(band().left).toBeCloseTo(0, 1);
    expect(band().right).toBeCloseTo((45 / L) * 100, 1);

    await wheelTo(6 * ROW);
    await settle(0);

    expect(visibleRows()).toEqual([6, 7, 8]);
    expect(band().left).toBeCloseTo((90 / L) * 100, 1);
    expect(band().right).toBeCloseTo((135 / L) * 100, 1);
  });

  it("SPEC-ADDON-153 I8 — at the end of the list the band runs to L", async () => {
    const { controller } = makeController({ duration: L, currentTime: 0 });
    await open({ mediaController: controller });

    await wheelTo(7 * ROW);
    await settle(0);

    expect(visibleRows()).toEqual([7, 8, 9]);
    expect(band().left).toBeCloseTo((105 / L) * 100, 1);
    expect(band().right).toBeCloseTo(100, 1);
  });
});

describe("SPEC-ADDON-129 I4 as revised by SPEC-ADDON-153 — no frames", () => {
  const D = 83.4;

  it("SPEC-ADDON-129 I4 — a click, a tap and every key on the ruler do nothing; the card's point is the playback time", async () => {
    vi.mocked(getClipTimestamps).mockResolvedValue(timestamps([]));
    const { controller, seek, play } = makeController({ duration: D, currentTime: 40 });
    await open({ mimeType: LOFT_MIME, mediaController: controller });
    const el = ruler();
    const at = (fraction: number) => fraction * WIDTH;

    firePointer(el, "pointerDown", { clientX: at(0.6), buttons: 1 });
    firePointer(el, "pointerUp", { clientX: at(0.6) });
    fireEvent.click(el, { clientX: at(0.6) });
    firePointer(el, "pointerDown", { pointerType: "touch", clientX: at(0.2) });
    firePointer(el, "pointerUp", { pointerType: "touch", clientX: at(0.2) });
    fireEvent.click(el, { clientX: at(0.2) });
    act(() => el.focus());
    for (const key of ["ArrowRight", "ArrowLeft", "Home", "End", "Enter"]) {
      fireEvent.keyDown(el, { key });
    }
    await settle(0);

    expect(seek).not.toHaveBeenCalled();
    expect(play).not.toHaveBeenCalled();
    expect(ruler()).toHaveAttribute("aria-valuenow", "40");
    expect(screen.queryAllByRole("button", { name: /^Play from / })).toEqual([]);
  });
});
