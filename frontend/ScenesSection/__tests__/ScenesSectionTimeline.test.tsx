import "./intlMock";
import { describe, it, expect, vi, beforeEach, afterEach, beforeAll } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";

import type { MediaController } from "@/lib/mediaController";
import { formatDuration } from "@/lib/format";
import { estimateLabelWidth, timeLabels } from "../scenesModel";

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
  waitForVisualIndex,
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
    await waitForVisualIndex();

    expect(ruler()).toHaveAttribute("aria-valuemax", String(expected));
  });
});

describe("SPEC-ADDON-152 I1 — every frame is drawn", () => {
  it.each([
    { n: 15, label: "fifteen frames" },
    { n: 5, label: "fewer frames than a row" },
    { n: 1, label: "one frame" },
  ])("SPEC-ADDON-152 I1 — $label are all drawn, in time order", async ({ n }) => {
    const rows = makeRows(n);
    vi.mocked(getClipTimestamps).mockResolvedValue(timestamps(rows));
    const { controller } = controllerAt(0, 70);

    renderSection({ mediaController: controller });
    await openNative();
    await waitForVisualIndex();

    expect(tileLabels()).toEqual(
      rows.map((row) => formatDuration(row.start)),
    );
  });

  it("SPEC-ADDON-089 I1 — the ruler draws one tick per frame", async () => {
    const rows = makeRows(15);
    vi.mocked(getClipTimestamps).mockResolvedValue(timestamps(rows));
    renderSection({ mediaController: makeController({ duration: 70 }).controller });
    await openNative();
    await waitForVisualIndex();

    const path = ruler().querySelector("path");
    expect(path).not.toBeNull();
    expect((path!.getAttribute("d") ?? "").match(/M/g)).toHaveLength(rows.length);
  });
});

describe("I11 — time labels leave at least 4 px between drawn boxes", () => {
  it.each([
    {
      name: "a 9:46:37 player duration at 351 px",
      duration: 35197,
      width: 351,
      interior: ["2:00:00", "4:00:00", "6:00:00"],
    },
    {
      name: "a 9:46:37 player duration at 384 px",
      duration: 35197,
      width: 384,
      interior: ["2:00:00", "4:00:00", "6:00:00"],
    },
    {
      name: "an 18:44 file at 351 px",
      duration: 1124,
      width: 351,
      interior: ["5:00", "10:00", "15:00"],
    },
    {
      name: "the long board's 4:09:30 file at 351 px",
      duration: 14970,
      width: 351,
      interior: ["1:00:00", "2:00:00", "3:00:00"],
    },
    {
      name: "a 4:09:59 player duration at 351 px",
      duration: 14999,
      width: 351,
      interior: ["1:00:00", "2:00:00", "3:00:00"],
    },
  ])("I11 — $name", ({ duration, width, interior }) => {
    const labels = timeLabels(duration, width, estimateLabelWidth);
    expect(labels.map((label) => formatDuration(label.time))).toEqual(interior);

    const boxes = [
      { left: 0, right: estimateLabelWidth("0:00") },
      ...labels.map((label) => ({
        left: label.left,
        right: label.left + label.width,
      })),
      {
        left: width - estimateLabelWidth(formatDuration(duration)),
        right: width,
      },
    ];
    for (let i = 1; i < boxes.length; i += 1) {
      expect(boxes[i].left - boxes[i - 1].right).toBeGreaterThanOrEqual(4);
    }
    for (const box of boxes) {
      expect(box.left).toBeGreaterThanOrEqual(0);
      expect(box.right).toBeLessThanOrEqual(width);
    }
  });
});

describe("SPEC-ADDON-153 I9 — hovering the ruler moves nothing", () => {
  it("SPEC-ADDON-153 I9 — a mouse move without a button draws the line only", async () => {
    const { controller, seek, play } = makeController({
      duration: 70,
      paused: true,
      currentTime: 10,
    });
    vi.mocked(getVideoVisualIndex).mockResolvedValue(
      indexState({
        active_run: makeRun({ succeeded_count: 1 }),
        scenes: [makeScene({ start_time: 30, section_start: 30, scene_label: "Thirty" })],
      }),
    );

    renderSection({ mediaController: controller });
    await openNative();
    await waitForVisualIndex();
    const el = ruler();
    stubRect(el);

    firePointer(el, "pointerMove", { pointerType: "mouse", clientX: pointerX(60, 70) });

    expect(ruler()).toHaveAttribute("aria-valuenow", "10");
    expect(screen.getByRole("button", { name: /Thirty/ })).toBeInTheDocument();
    expect(allTiles()).toHaveLength(15);
    expect(seek).not.toHaveBeenCalled();
    expect(play).not.toHaveBeenCalled();
  });
});

describe("SPEC-ADDON-089 I2 — the playing tile follows the clock", () => {
  it("moves the accent outline with the shared clock", async () => {
    vi.useFakeTimers();
    const { controller, state } = makeController({
      duration: 70,
      paused: false,
    });

    renderSection({ mediaController: controller });
    await openNative();
    await waitForVisualIndex();

    expect(accentTiles()).toEqual(["0:00"]);

    for (let time = 1.5; time <= 6; time += 1.5) {
      await advance(250, () => {
        state.currentTime = time;
      });
    }

    expect(accentTiles()).toEqual(["0:05"]);
    expect(ruler()).toHaveAttribute("aria-valuenow", "6");
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
      makeRows(15).map((row) => formatDuration(row.start)),
    );

    fireEvent.click(screen.getByRole("button", { name: "Play from 0:30" }));

    expect(video.currentTime).toBe(30);
    expect(play).toHaveBeenCalled();
  });
});
