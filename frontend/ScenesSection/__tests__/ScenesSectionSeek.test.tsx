import "./intlMock";
import { describe, it, expect, vi, beforeEach, afterEach, beforeAll } from "vitest";
import { act, render, screen, fireEvent } from "@testing-library/react";

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
  advance,
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
  tile,
  timestamps,
  waitForVisualIndex,
} from "./helpers";

function section(props: Partial<SceneSectionProps> = {}) {
  return (
    <ScenesSection
      fileId="f1"
      drive="d1"
      fileType="video"
      mimeType="video/mp4"
      {...props}
    />
  );
}

function renderSection(props: Partial<SceneSectionProps> = {}) {
  return render(section(props));
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

describe("SPEC-ADDON-089 I2 — the seeks", () => {
  it("a tile click seeks and plays", async () => {
    const { controller, seek, play } = makeController({
      duration: 70,
      paused: true,
    });
    renderSection({ mediaController: controller });
    await openNative();
    await waitForVisualIndex();

    fireEvent.click(tile(30));

    expect(seek).toHaveBeenCalledWith(30);
    expect(play).toHaveBeenCalled();
    expect(seek.mock.invocationCallOrder[0]).toBeLessThan(
      play.mock.invocationCallOrder[0],
    );
  });

  it("a scene card click seeks to its scene and plays", async () => {
    const { controller, seek, play } = makeController({
      duration: 70,
      paused: true,
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

    fireEvent.click(screen.getByRole("button", { name: /Thirty/ }));

    expect(seek).toHaveBeenCalledWith(30);
    expect(play).toHaveBeenCalled();
    expect(seek.mock.invocationCallOrder[0]).toBeLessThan(
      play.mock.invocationCallOrder[0],
    );
  });
});

describe("SPEC-ADDON-153 I9 — ruler input sets the point and never seeks", () => {
  it.each([
    {
      name: "a mouse click",
      gesture: (el: Element) => {
        firePointer(el, "pointerDown", { pointerType: "mouse", clientX: pointerX(60, 70), buttons: 1 });
        firePointer(el, "pointerUp", { pointerType: "mouse", clientX: pointerX(60, 70) });
        fireEvent.click(el, { clientX: pointerX(60, 70) });
      },
    },
    {
      name: "a mouse drag",
      gesture: (el: Element) => {
        firePointer(el, "pointerDown", { pointerType: "mouse", clientX: pointerX(10, 70), buttons: 1 });
        firePointer(el, "pointerMove", { pointerType: "mouse", clientX: pointerX(60, 70), buttons: 1 });
        firePointer(el, "pointerUp", { pointerType: "mouse", clientX: pointerX(60, 70) });
      },
    },
    {
      name: "a touch tap",
      gesture: (el: Element) => {
        firePointer(el, "pointerDown", { pointerType: "touch", clientX: pointerX(60, 70) });
        firePointer(el, "pointerUp", { pointerType: "touch", clientX: pointerX(60, 70) });
        fireEvent.click(el, { clientX: pointerX(60, 70) });
      },
    },
    {
      name: "a pen tap",
      gesture: (el: Element) => {
        firePointer(el, "pointerDown", { pointerType: "pen", clientX: pointerX(60, 70) });
        firePointer(el, "pointerUp", { pointerType: "pen", clientX: pointerX(60, 70) });
        fireEvent.click(el, { clientX: pointerX(60, 70) });
      },
    },
  ])("SPEC-ADDON-153 I9 — $name never seeks", async ({ gesture }) => {
    const { controller, seek, play } = makeController({
      duration: 70,
      paused: true,
    });
    renderSection({ mediaController: controller });
    await openNative();
    await waitForVisualIndex();
    const el = ruler();
    stubRect(el);

    gesture(el);

    expect(seek).not.toHaveBeenCalled();
    expect(play).not.toHaveBeenCalled();
    expect(ruler()).toHaveAttribute("aria-valuenow", "60");
  });

  it("SPEC-ADDON-089 I3 — pointercancel restores the point from before the touch", async () => {
    const { controller } = makeController({ duration: 70, paused: true });
    renderSection({ mediaController: controller });
    await openNative();
    await waitForVisualIndex();
    const el = ruler();
    stubRect(el);

    firePointer(el, "pointerDown", { pointerType: "touch", clientX: pointerX(10, 70) });
    firePointer(el, "pointerUp", { pointerType: "touch", clientX: pointerX(10, 70) });
    expect(ruler()).toHaveAttribute("aria-valuenow", "10");

    firePointer(el, "pointerDown", { pointerType: "touch", clientX: pointerX(60, 70), buttons: 1 });
    firePointer(el, "pointerMove", { pointerType: "touch", clientX: pointerX(60, 70), buttons: 1 });
    firePointer(el, "pointerCancel", { pointerType: "touch", clientX: pointerX(60, 70) });

    expect(ruler()).toHaveAttribute("aria-valuenow", "10");
  });

  it("SPEC-ADDON-089 I3 — a touch point survives the finger lifting", async () => {
    const { controller } = makeController({ duration: 70, paused: true });
    renderSection({ mediaController: controller });
    await openNative();
    await waitForVisualIndex();
    const el = ruler();
    stubRect(el);

    firePointer(el, "pointerDown", { pointerType: "touch", clientX: pointerX(60, 70) });
    firePointer(el, "pointerUp", { pointerType: "touch", clientX: pointerX(60, 70) });
    await flush();

    expect(ruler()).toHaveAttribute("aria-valuenow", "60");
  });

  it("SPEC-ADDON-152 I3 — a tile removes the ruler point", async () => {
    const { controller, seek, play } = makeController({
      duration: 70,
      paused: true,
    });
    renderSection({ mediaController: controller });
    await openNative();
    await waitForVisualIndex();
    const el = ruler();
    stubRect(el);

    firePointer(el, "pointerDown", { pointerType: "mouse", clientX: pointerX(60, 70), buttons: 1 });
    firePointer(el, "pointerUp", { pointerType: "mouse", clientX: pointerX(60, 70) });
    expect(ruler()).toHaveAttribute("aria-valuenow", "60");

    fireEvent.click(tile(60));

    expect(ruler()).toHaveAttribute("aria-valuenow", "0");
    expect(seek).toHaveBeenCalledWith(60);
    expect(play).toHaveBeenCalled();
  });
});

describe("SPEC-ADDON-152 I3 — the ruler point's lifetime", () => {
  it.each([
    { name: "a collapse and a re-show", event: "collapse" },
    { name: "a file change", event: "filechange" },
  ])("SPEC-ADDON-152 I3 — $name removes the point", async ({ event }) => {
    const { controller } = makeController({ duration: 70, paused: true });
    const view = renderSection({ mediaController: controller });
    await openNative();
    await waitForVisualIndex();
    const el = ruler();
    stubRect(el);

    firePointer(el, "pointerDown", { pointerType: "mouse", clientX: pointerX(60, 70), buttons: 1 });
    firePointer(el, "pointerUp", { pointerType: "mouse", clientX: pointerX(60, 70) });
    expect(ruler()).toHaveAttribute("aria-valuenow", "60");

    if (event === "collapse") {
      fireEvent.click(header());
      fireEvent.click(header());
    } else {
      view.rerender(section({ fileId: "f2", mediaController: controller }));
      fireEvent.click(header());
    }
    await screen.findByRole("slider");

    expect(ruler()).toHaveAttribute("aria-valuenow", "0");
  });

  it("SPEC-ADDON-152 I3 — a seek from elsewhere leaves the point alone", async () => {
    vi.useFakeTimers();
    const { controller, state } = makeController({
      duration: 70,
      currentTime: 10,
      paused: false,
    });
    renderSection({ mediaController: controller });
    await openNative();
    await waitForVisualIndex();
    const el = ruler();
    stubRect(el);

    firePointer(el, "pointerDown", { pointerType: "mouse", clientX: pointerX(60, 70), buttons: 1 });
    firePointer(el, "pointerUp", { pointerType: "mouse", clientX: pointerX(60, 70) });
    expect(ruler()).toHaveAttribute("aria-valuenow", "60");

    await advance(250, () => {
      state.currentTime = 40;
      state.paused = true;
    });

    expect(ruler()).toHaveAttribute("aria-valuenow", "60");
    expect(allTiles()).toHaveLength(15);
  });
});
