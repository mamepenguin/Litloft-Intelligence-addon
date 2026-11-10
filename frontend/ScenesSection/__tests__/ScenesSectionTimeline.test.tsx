import "./intlMock";
import { describe, it, expect, vi, beforeEach, afterEach, beforeAll } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";

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
  accentTiles,
  allTiles,
  firePointer,
  flush,
  header,
  indexState,
  installPointerEvents,
  makeController,
  makeRows,
  makeRun,
  makeScene,
  pointerX,
  ruler,
  SceneSectionProps,
  stubRect,
  tileLabels,
  timestamps,
  advance,
} from "./helpers";

function renderSection(props: Partial<SceneSectionProps> = {}) {
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

function controllerAt(
  currentTime: number,
  duration: number,
  paused = true,
): { controller: MediaController; state: { currentTime: number } } {
  return makeController({ currentTime, duration, paused });
}

async function openNative(): Promise<void> {
  fireEvent.click(header());
  await screen.findByRole("slider");
}

beforeAll(installPointerEvents);

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(getClipTimestamps).mockResolvedValue(timestamps(makeRows(15)));
  vi.mocked(getVideoVisualIndex).mockResolvedValue(null);
  vi.mocked(usePolicy).mockReturnValue({ enabled: true, isLoading: false });
});

afterEach(() => {
  vi.useRealTimers();
});

describe("SPEC-ADDON-089 I1 — ruler length L", () => {
  it.each([
    {
      name: "the player duration when it reaches the last frame",
      rows: makeRows(15),
      duration: 70,
      expected: 70,
    },
    {
      name: "the player duration when it exceeds the last frame",
      rows: makeRows(14),
      duration: 70,
      expected: 70,
    },
    {
      name: "the last frame plus the median gap when the duration is shorter",
      rows: makeRows(15),
      duration: 30,
      expected: 75,
    },
    {
      name: "the last frame plus the median gap when the duration is unknown",
      rows: makeRows(15),
      duration: 0,
      expected: 75,
    },
    {
      name: "the last frame plus five in an uneven set",
      rows: [
        ...makeRows(3),
        { start: 30, content_preview: "Frame at 30s" },
      ],
      duration: 0,
      expected: 35,
    },
    {
      name: "the frame start plus one second for a single frame",
      rows: [{ start: 10, content_preview: "Frame at 10s" }],
      duration: 0,
      expected: 11,
    },
  ])("SPEC-ADDON-089 I1 — L is $name", async ({ rows, duration, expected }) => {
    const { controller } = controllerAt(0, duration);
    vi.mocked(getClipTimestamps).mockResolvedValue(timestamps(rows));

    renderSection({ mediaController: controller });
    await openNative();

    expect(ruler()).toHaveAttribute("aria-valuemax", String(expected));
  });
});

describe("SPEC-ADDON-089 I1 — the nine-frame window", () => {
  it.each([
    { n: 15, time: 0, windowStart: 0, label: "i = 0" },
    { n: 15, time: 20, windowStart: 0, label: "i = 4" },
    { n: 15, time: 35, windowStart: 3, label: "i = 7" },
    { n: 15, time: 70, windowStart: 6, label: "i = n-1" },
    { n: 5, time: 0, windowStart: 0, label: "n < 9" },
  ])(
    "SPEC-ADDON-089 I1 — the window follows the clock at $label",
    async ({ n, time, windowStart }) => {
      const rows = makeRows(n);
      const { controller } = controllerAt(time, 70);
      vi.mocked(getClipTimestamps).mockResolvedValue(timestamps(rows));

      renderSection({ mediaController: controller });
      await openNative();

      const expected = rows
        .slice(windowStart, windowStart + 9)
        .map((row) => formatDuration(row.start));
      expect(tileLabels()).toEqual(expected);
    },
  );

  it("SPEC-ADDON-089 I1 — one path or canvas carries the ticks, so their count is not observable in jsdom", async () => {
    // A single SVG path or canvas has no per-tick element to count. The
    // window and L are the parts of I1 a jsdom tree can hold.
    const { controller } = controllerAt(0, 70);
    renderSection({ mediaController: controller });
    await openNative();

    expect(ruler().querySelectorAll("canvas").length).toBeLessThanOrEqual(1);
    expect(ruler()).toHaveAttribute("aria-valuemax", "70");
    expect(tileLabels()).toHaveLength(9);
  });
});

describe("SPEC-ADDON-089 I2 — a mouse browse point", () => {
  it("moves the grid and the scene card without seeking", async () => {
    const { controller, seek, play } = makeController({
      duration: 70,
      paused: true,
    });
    vi.mocked(getVideoVisualIndex).mockResolvedValue(
      indexState({
        active_run: makeRun({ succeeded_count: 1 }),
        scenes: [makeScene({ start_time: 30, scene_label: "Thirty" })],
      }),
    );
    vi.mocked(usePolicy).mockReturnValue({ enabled: true, isLoading: false });

    renderSection({ mediaController: controller });
    await openNative();
    const el = ruler();
    stubRect(el);

    firePointer(el, "pointerMove", { pointerType: "mouse", clientX: pointerX(60, 70) });

    expect(tileLabels()).toEqual(
      makeRows(15).slice(6, 15).map((row) => formatDuration(row.start)),
    );
    expect(screen.getByRole("button", { name: /Thirty/ })).toBeInTheDocument();
    expect(seek).not.toHaveBeenCalled();
    expect(play).not.toHaveBeenCalled();
  });

  it("keeps the point over the grid and drops it when the pointer leaves the section", async () => {
    const { controller } = controllerAt(0, 70);
    const { container } = renderSection({ mediaController: controller });
    await openNative();
    const el = ruler();
    stubRect(el);

    firePointer(el, "pointerMove", { pointerType: "mouse", clientX: pointerX(60, 70) });
    expect(ruler()).toHaveAttribute("aria-valuenow", "60");

    firePointer(el, "pointerLeave", { pointerType: "mouse" });
    firePointer(allTiles()[6], "pointerMove", { pointerType: "mouse" });
    expect(ruler()).toHaveAttribute("aria-valuenow", "60");

    const root = container.firstElementChild!;
    fireEvent.pointerOut(root, { pointerType: "mouse", relatedTarget: document.body });
    firePointer(root, "pointerLeave", { pointerType: "mouse" });

    expect(ruler()).toHaveAttribute("aria-valuenow", "0");
    expect(tileLabels()).toEqual(
      makeRows(15).slice(0, 9).map((row) => formatDuration(row.start)),
    );
  });
});

describe("SPEC-ADDON-089 I2 — the playing tile follows the clock", () => {
  it("moves the accent outline while a browse point is set", async () => {
    vi.useFakeTimers();
    const { controller, state } = makeController({
      duration: 70,
      paused: false,
    });

    renderSection({ mediaController: controller });
    await openNative();
    const el = ruler();
    stubRect(el);

    firePointer(el, "pointerMove", { pointerType: "mouse", clientX: pointerX(10, 70) });
    expect(accentTiles()).toEqual(["0:00"]);

    for (let time = 1.5; time <= 6; time += 1.5) {
      await advance(250, () => {
        state.currentTime = time;
      });
    }

    expect(accentTiles()).toEqual(["0:05"]);
    expect(ruler()).toHaveAttribute("aria-valuenow", "10");
  });
});

describe("SPEC-ADDON-089 I2 — the no-controller fallback", () => {
  it("seeks and plays through videoRef when mediaController is null", async () => {
    const video = document.createElement("video");
    const play = vi.fn().mockResolvedValue(undefined);
    video.play = play as unknown as HTMLMediaElement["play"];
    const videoRef = { current: video };

    vi.mocked(getClipTimestamps).mockResolvedValue(timestamps(makeRows(15)));
    renderSection({ mediaController: null, videoRef });

    await flush();
    fireEvent.click(header());
    await screen.findByRole("slider");

    expect(accentTiles()).toEqual([]);
    expect(tileLabels()).toEqual(
      makeRows(15).slice(0, 9).map((row) => formatDuration(row.start)),
    );

    fireEvent.click(screen.getByRole("button", { name: "Play from 0:30" }));

    expect(video.currentTime).toBe(30);
    expect(play).toHaveBeenCalled();
  });
});
