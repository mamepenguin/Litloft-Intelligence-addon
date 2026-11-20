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
  flush,
  images,
  imgTimes,
  indexState,
  installPointerEvents,
  LOFT_MIME,
  makeController,
  makeRows,
  SceneSectionProps,
  timestamps,
} from "./helpers";
import {
  ROW,
  installLayout,
  resetLayout,
  scroller,
  scrollTopOf,
  visibleRows,
  wheelTo,
} from "./scrollListLayout";

/** 30 frames 5 s apart: ten rows of three, three of them visible at once. */
const N = 30;
const ROWS = makeRows(N);
const L = 150;
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
});

afterEach(() => {
  vi.useRealTimers();
});

describe("SPEC-ADDON-155 I1 — a re-show asks only for the near tiles after its start placement", () => {
  it.each([
    { name: "a native video", mimeType: "video/mp4" },
    { name: "a .loft", mimeType: LOFT_MIME },
  ])(
    "SPEC-ADDON-155 I1 — $name: after a wheel ends opening and the host hides and shows the panel, the playing row's near tiles have a src with no time advanced and no tile outside them does",
    async ({ mimeType }) => {
      const props: Props = {
        mimeType,
        mediaController: makeController({ duration: L, currentTime: 100 }).controller,
      };
      const view = render(section(props));
      await now();
      await now();
      expect(visibleRows(), "the playing row placed").toEqual([5, 6, 7]);

      await wheelTo(ROW);
      expect(visibleRows()).toEqual([1, 2, 3]);
      expect(asked(rowsTimes([0, 1, 2, 3])), "near only after opening ended").toEqual([]);

      view.rerender(section({ ...props, active: false }));
      await now();
      view.rerender(section({ ...props, active: true }));
      await now();
      await now();

      expect(visibleRows(), "the playing row placed again").toEqual([5, 6, 7]);
      expect(scrollTopOf(scroller())).toBe(5 * ROW);
      const near = rowsTimes(nearRows());
      expect(asked(near), "the near tiles with no time advanced").toEqual(near);
      expect(asked(rowsTimes([0, 1, 2, 3])), "tiles outside the near set").toEqual([]);
      const outside = srcs().filter(
        (src) => !near.some((time) => src.endsWith(`?t=${time}`)),
      );
      expect(outside).toEqual([]);
    },
  );

  it.each([
    { name: "a native video", mimeType: "video/mp4" },
    { name: "a .loft", mimeType: LOFT_MIME },
  ])(
    "SPEC-ADDON-155 I1 — $name with no controller: after the host hides and shows the panel, the list is at its top and the near tiles have a src with no time advanced",
    async ({ mimeType }) => {
      const props: Props = { mimeType, mediaController: null };
      const view = render(section(props));
      await now();
      await now();
      expect(visibleRows()).toEqual([0, 1, 2]);
      expect(asked(rowsTimes([0, 1, 2, 3])), "the opening's near tiles").toEqual(
        rowsTimes([0, 1, 2, 3]),
      );

      await wheelTo(ROW);
      expect(visibleRows()).toEqual([1, 2, 3]);

      view.rerender(section({ ...props, active: false }));
      await now();
      view.rerender(section({ ...props, active: true }));
      await now();
      await now();

      expect(visibleRows(), "the list's top placed again").toEqual([0, 1, 2]);
      expect(scrollTopOf(scroller())).toBe(0);
      const near = rowsTimes(nearRows());
      expect(asked(near), "the near tiles with no time advanced").toEqual(near);
    },
  );
});
