import { describe, it, expect, vi, beforeEach, afterEach, beforeAll } from "vitest";
import { render } from "@testing-library/react";

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
  makeController,
  makeRows,
  makeRun,
  makeScene,
  pointerX,
  ruler,
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
  visibleRows,
  wheelTo,
  whollyVisible,
} from "./scrollListLayout";

/** 30 frames 5 s apart: ten rows of three, three of them visible at once. */
const N = 30;
const ROWS = makeRows(N);
const L = 150;

function section(props: Partial<SceneSectionProps> = {}) {
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

async function open(props: Partial<SceneSectionProps> = {}) {
  const view = render(section(props));
  await settle(0);
  await settle(0);
  return view;
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
  vi.mocked(getVideoVisualIndex).mockResolvedValue(
    indexState({
      active_run: makeRun({ succeeded_count: 1 }),
      scenes: [
        makeScene({
          ordering: 0,
          start_time: 0,
          section_start: 0,
          scene_label: "Alpha",
        }),
      ],
    }),
  );
  vi.mocked(usePolicy).mockReturnValue({ enabled: true, isLoading: false });
});

afterEach(() => {
  vi.useRealTimers();
});

async function rulerPointAt(time: number): Promise<void> {
  const el = ruler();
  firePointer(el, "pointerDown", {
    clientX: pointerX(time, L, WIDTH),
    buttons: 1,
  });
  firePointer(el, "pointerUp", { clientX: pointerX(time, L, WIDTH) });
  await settle(0);
}

describe("SPEC-ADDON-152 I12 — a controller appearing is only a placement event", () => {
  it("SPEC-ADDON-152 I12 — after a reader's scroll it moves nothing, keeps the point and draws the chip", async () => {
    const { controller } = makeController({ duration: L, currentTime: 100 });
    const view = await open({ mediaController: null });

    await wheelTo(6 * ROW);
    await settle(0);
    expect(chip()).toBeNull();

    await rulerPointAt(120);
    expect(ruler()).toHaveAttribute("aria-valuenow", "120");
    const placed = scrollTopOf(scroller());
    const rows = visibleRows();

    view.rerender(section({ mediaController: controller }));
    await settle(0);
    await settle(0);

    expect(scrollTopOf(scroller()), "the list does not move").toBe(placed);
    expect(visibleRows()).toEqual(rows);
    expect(ruler(), "the ruler point stays").toHaveAttribute(
      "aria-valuenow",
      "120",
    );
    expect(chip(), "the chip appears with the controller").not.toBeNull();
  });

  it("SPEC-ADDON-152 I12 — while following, a controller appearing places the playing row", async () => {
    const { controller } = makeController({ duration: L, currentTime: 100 });
    const view = await open({ mediaController: null });
    expect(visibleRows()).toEqual([0, 1, 2]);

    view.rerender(section({ mediaController: controller }));
    await settle(0);
    await settle(0);

    expect(whollyVisible(6)).toBe(true);
    expect(visibleRows()[0]).toBe(5);
    expect(chip()).toBeNull();
  });
});
