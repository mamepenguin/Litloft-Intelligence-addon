import "./intlMock";
import { describe, it, expect, vi, beforeEach, beforeAll } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";

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
import {
  getClipTimestamps,
  getVideoVisualIndex,
  type VideoVisualSceneItem,
} from "@/addons/intelligence/api";
import { usePolicy } from "@/hooks/usePolicy";
import { formatDuration } from "@/lib/format";
import {
  allTiles,
  header,
  indexState,
  installPointerEvents,
  makeController,
  makeRows,
  makeRun,
  makeScene,
  SceneSectionProps,
  sceneCardNamed,
  tile,
  timestamps,
  waitForVisualIndex,
} from "./helpers";

function withSection(
  scene: VideoVisualSceneItem,
  sectionStart: number | null,
  sectionEnd: number | null,
): VideoVisualSceneItem {
  return { ...scene, section_start: sectionStart, section_end: sectionEnd } as VideoVisualSceneItem;
}

function renderSection(props: Partial<SceneSectionProps> = {}) {
  return render(
    <ScenesSection fileId="f1" drive="d1" fileType="video" mimeType="video/mp4" {...props} />,
  );
}

async function open(): Promise<void> {
  fireEvent.click(header());
  await screen.findByRole("slider");
}

function tileDot(el: HTMLElement): boolean {
  return el.querySelector('[class*="bg-accent-teal"]') !== null;
}

const sectionScenes = [
  withSection(makeScene({ ordering: 0, start_time: 6, scene_label: "Alpha" }), 0, 20),
  withSection(makeScene({ ordering: 1, start_time: 26, scene_label: "Beta" }), 20, 70),
];

function serve(scenes: VideoVisualSceneItem[]) {
  vi.mocked(getVideoVisualIndex).mockResolvedValue(
    indexState({ active_run: makeRun({ succeeded_count: scenes.length }), scenes }),
  );
}

beforeAll(installPointerEvents);

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(getClipTimestamps).mockResolvedValue(timestamps(makeRows(15)));
  vi.mocked(usePolicy).mockReturnValue({ enabled: true, isLoading: false });
});

describe("SPEC-ADDON-090 I9 as revised by SPEC-ADDON-113 — the seek time", () => {
  it("SPEC-ADDON-113 — the card shows the scene whose section has started", async () => {
    const { controller } = makeController({ duration: 70, currentTime: 22 });
    serve(sectionScenes);
    renderSection({ mediaController: controller });
    await open();
    await waitForVisualIndex();

    expect(sceneCardNamed(/Beta/)).not.toBeNull();
    expect(sceneCardNamed(/Alpha/)).toBeNull();
  });

  it("SPEC-ADDON-113 — the card's time label is the section start", async () => {
    const { controller } = makeController({ duration: 70, currentTime: 30 });
    serve(sectionScenes);
    renderSection({ mediaController: controller });
    await open();
    await waitForVisualIndex();

    const card = sceneCardNamed(/Beta/)!;
    expect(card.textContent).toContain(formatDuration(20));
    expect(card.textContent).not.toContain(formatDuration(26));
  });

  it("SPEC-ADDON-113 — activating the card plays from the section start", async () => {
    const { controller, seek, play } = makeController({ duration: 70, currentTime: 30 });
    serve(sectionScenes);
    renderSection({ mediaController: controller });
    await open();
    await waitForVisualIndex();

    fireEvent.click(sceneCardNamed(/Beta/)!);

    expect(seek).toHaveBeenCalledWith(20);
    expect(play).toHaveBeenCalled();
  });

  it.each([
    { name: "inside a tile's range", sectionStart: 12, dotted: 10 },
    { name: "on a tile's start", sectionStart: 15, dotted: 15 },
    { name: "just before the next tile", sectionStart: 14.99, dotted: 10 },
  ])(
    "SPEC-ADDON-113 — the tile whose range holds the section start carries the dot ($name)",
    async ({ sectionStart, dotted }) => {
      serve([withSection(makeScene({ start_time: sectionStart + 3 }), sectionStart, 40)]);
      renderSection();
      await open();
      await waitForVisualIndex();

      const dots = allTiles().filter(tileDot);
      expect(dots).toEqual([tile(dotted)]);
    },
  );

  it("SPEC-ADDON-113 — the first tile also takes a section start before it", async () => {
    vi.mocked(getClipTimestamps).mockResolvedValue(timestamps(makeRows(14, 5, 5)));
    serve([withSection(makeScene({ start_time: 4 }), 0, 30)]);
    renderSection();
    await open();
    await waitForVisualIndex();

    expect(allTiles().filter(tileDot)).toEqual([tile(5)]);
  });
});
