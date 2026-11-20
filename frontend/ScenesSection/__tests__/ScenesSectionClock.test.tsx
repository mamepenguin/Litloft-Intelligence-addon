import "./intlMock";
import { describe, it, expect, vi, beforeEach, afterEach, beforeAll } from "vitest";
import { render } from "@testing-library/react";

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
import { getClipTimestamps } from "@/addons/intelligence/api";
import { usePolicy } from "@/hooks/usePolicy";
import {
  accentTiles,
  advance,
  flush,
  installPointerEvents,
  makeController,
  makeRows,
  timestamps,
} from "./helpers";
import {
  ROW,
  installLayout,
  resetLayout,
  scroller,
  scrollTopOf,
  tileElements,
  visibleRows,
} from "./scrollListLayout";

const N = 30;
const ROWS = makeRows(N);
const L = 150;

beforeAll(() => {
  installPointerEvents();
  installLayout();
});

beforeEach(() => {
  vi.clearAllMocks();
  vi.useFakeTimers();
  resetLayout();
  vi.mocked(getClipTimestamps).mockResolvedValue(timestamps(ROWS));
  vi.mocked(usePolicy).mockReturnValue({ enabled: true, isLoading: false });
});

afterEach(() => {
  vi.useRealTimers();
});

async function settle(ms = 0): Promise<void> {
  await flush(8);
  await advance(ms);
  await flush(8);
}

describe("SPEC-ADDON-089 I2 — the clock holds its last good reading", () => {
  it("SPEC-ADDON-089 I2 — a getter that throws once while playing leaves the outline on its tile and the list where it was", async () => {
    const { controller, state } = makeController({
      duration: L,
      currentTime: 40,
      paused: false,
    });
    let throwNext = false;
    vi.spyOn(controller, "getCurrentTime").mockImplementation(() => {
      if (throwNext) {
        throwNext = false;
        throw new Error("the player is swapping media");
      }
      return state.currentTime;
    });

    render(
      <ScenesSection
        fileId="f1"
        drive="d1"
        fileType="video"
        mimeType="video/mp4"
        labelledByHost
        active
        mediaController={controller}
        onAvailability={() => {}}
      />,
    );
    await settle(0);
    await settle(0);

    expect(tileElements()).toHaveLength(N);
    expect(accentTiles()).toEqual(["0:40"]);
    expect(visibleRows()).toEqual([1, 2, 3]);
    const placed = scrollTopOf(scroller());
    expect(placed).toBe(ROW);

    state.currentTime = 100;
    throwNext = true;
    await advance(250);
    await flush(4);

    expect(
      accentTiles(),
      "the playing outline stays on the last good frame",
    ).toEqual(["0:40"]);
    expect(scrollTopOf(scroller())).toBe(placed);
    expect(visibleRows()).toEqual([1, 2, 3]);
  });
});
