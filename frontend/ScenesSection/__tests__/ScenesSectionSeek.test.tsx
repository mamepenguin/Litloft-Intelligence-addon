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
  tileLabels,
  timestamps,
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

describe("SPEC-ADDON-089 I2 — the four seeks", () => {
  it("a mouse click on the ruler seeks and plays", async () => {
    const { controller, seek, play } = makeController({
      duration: 70,
      paused: true,
    });
    renderSection({ mediaController: controller });
    await openNative();
    const el = ruler();
    stubRect(el);

    firePointer(el, "pointerDown", { pointerType: "mouse", clientX: pointerX(60, 70) });
    firePointer(el, "pointerUp", { pointerType: "mouse", clientX: pointerX(60, 70) });
    fireEvent.click(el, { clientX: pointerX(60, 70) });

    expect(seek).toHaveBeenCalledWith(60);
    expect(play).toHaveBeenCalled();
    expect(seek.mock.invocationCallOrder[0]).toBeLessThan(
      play.mock.invocationCallOrder[0],
    );
  });

  it("a tile click seeks and plays", async () => {
    const { controller, seek, play } = makeController({
      duration: 70,
      paused: true,
    });
    renderSection({ mediaController: controller });
    await openNative();

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
        scenes: [makeScene({ start_time: 30, scene_label: "Thirty" })],
      }),
    );
    renderSection({ mediaController: controller });
    await openNative();

    fireEvent.click(screen.getByRole("button", { name: /Thirty/ }));

    expect(seek).toHaveBeenCalledWith(30);
    expect(play).toHaveBeenCalled();
    expect(seek.mock.invocationCallOrder[0]).toBeLessThan(
      play.mock.invocationCallOrder[0],
    );
  });
});

describe("SPEC-ADDON-089 I3 — touch and pen browsing", () => {
  it("hands vertical scrolling to the browser", async () => {
    const { controller } = makeController({ duration: 70, paused: true });
    renderSection({ mediaController: controller });
    await openNative();
    const el = ruler();

    expect(`${el.className} ${el.getAttribute("style") ?? ""}`).toMatch(
      /touch-pan-y|touch-action:\s*pan-y/,
    );
  });

  it.each([
    {
      name: "a tap",
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
    {
      name: "a horizontal drag",
      gesture: (el: Element) => {
        firePointer(el, "pointerDown", {
          pointerType: "touch",
          clientX: pointerX(10, 70),
          buttons: 1,
        });
        firePointer(el, "pointerMove", {
          pointerType: "touch",
          clientX: pointerX(60, 70),
          buttons: 1,
        });
        firePointer(el, "pointerUp", { pointerType: "touch", clientX: pointerX(60, 70) });
      },
    },
  ])(
    "SPEC-ADDON-089 I3 — $name sets the browse point and never seeks",
    async ({ gesture }) => {
      const { controller, seek, play } = makeController({
        duration: 70,
        paused: true,
      });
      renderSection({ mediaController: controller });
      await openNative();
      const el = ruler();
      stubRect(el);

      gesture(el);

      expect(seek).not.toHaveBeenCalled();
      expect(play).not.toHaveBeenCalled();
      expect(ruler()).toHaveAttribute("aria-valuenow", "60");
    },
  );

  it("SPEC-ADDON-089 I3 — pointercancel restores the browse point from before the touch", async () => {
    const { controller } = makeController({ duration: 70, paused: true });
    renderSection({ mediaController: controller });
    await openNative();
    const el = ruler();
    stubRect(el);

    firePointer(el, "pointerDown", { pointerType: "touch", clientX: pointerX(10, 70) });
    firePointer(el, "pointerUp", { pointerType: "touch", clientX: pointerX(10, 70) });
    expect(ruler()).toHaveAttribute("aria-valuenow", "10");

    firePointer(el, "pointerDown", {
      pointerType: "touch",
      clientX: pointerX(60, 70),
      buttons: 1,
    });
    firePointer(el, "pointerMove", {
      pointerType: "touch",
      clientX: pointerX(60, 70),
      buttons: 1,
    });
    firePointer(el, "pointerCancel", { pointerType: "touch", clientX: pointerX(60, 70) });

    expect(ruler()).toHaveAttribute("aria-valuenow", "10");
  });

  it.each([
    {
      expectation: "a touch browse point survives focus leaving the section",
      set: (el: HTMLElement) => {
        firePointer(el, "pointerDown", {
          pointerType: "touch",
          clientX: pointerX(60, 70),
        });
        firePointer(el, "pointerUp", {
          pointerType: "touch",
          clientX: pointerX(60, 70),
        });
        fireEvent.click(el, { clientX: pointerX(60, 70) });
        act(() => el.focus());
      },
      expected: 60,
      windowStart: 6,
    },
    {
      expectation: "a keyboard browse point is cleared when focus leaves the section",
      set: (el: HTMLElement) => {
        act(() => el.focus());
        fireEvent.keyDown(el, { key: "ArrowRight" });
      },
      expected: 10,
      windowStart: 0,
    },
  ])("SPEC-ADDON-089 I3 — $expectation", async ({ set, expected, windowStart }) => {
    const { controller } = makeController({
      duration: 70,
      currentTime: 10,
      paused: true,
    });
    renderSection({ mediaController: controller });
    await openNative();
    const el = ruler();
    stubRect(el);

    set(el);
    fireEvent.focusOut(el, { relatedTarget: document.body });

    expect(ruler()).toHaveAttribute("aria-valuenow", String(expected));
    expect(tileLabels()).toEqual(
      makeRows(15)
        .slice(windowStart, windowStart + 9)
        .map((row) => formatDuration(row.start)),
    );
  });

  it("SPEC-ADDON-089 I3 — the browse point survives the finger lifting", async () => {
    const { controller } = makeController({ duration: 70, paused: true });
    renderSection({ mediaController: controller });
    await openNative();
    const el = ruler();
    stubRect(el);

    firePointer(el, "pointerDown", { pointerType: "touch", clientX: pointerX(60, 70) });
    firePointer(el, "pointerUp", { pointerType: "touch", clientX: pointerX(60, 70) });
    await flush();

    expect(ruler()).toHaveAttribute("aria-valuenow", "60");
  });

  it("SPEC-ADDON-089 I3 — a tile tap clears the browse point", async () => {
    const { controller, seek, play } = makeController({
      duration: 70,
      paused: true,
    });
    renderSection({ mediaController: controller });
    await openNative();
    const el = ruler();
    stubRect(el);

    firePointer(el, "pointerDown", { pointerType: "touch", clientX: pointerX(60, 70) });
    firePointer(el, "pointerUp", { pointerType: "touch", clientX: pointerX(60, 70) });
    expect(ruler()).toHaveAttribute("aria-valuenow", "60");

    fireEvent.click(tile(60));

    expect(ruler()).toHaveAttribute("aria-valuenow", "0");
    expect(seek).toHaveBeenCalledWith(60);
    expect(play).toHaveBeenCalled();
  });

  it("SPEC-ADDON-089 I3 — collapse clears the browse point", async () => {
    const { controller } = makeController({ duration: 70, paused: true });
    renderSection({ mediaController: controller });
    await openNative();
    const el = ruler();
    stubRect(el);

    firePointer(el, "pointerMove", { pointerType: "mouse", clientX: pointerX(60, 70) });
    expect(ruler()).toHaveAttribute("aria-valuenow", "60");

    fireEvent.click(header());
    expect(header()).toHaveAttribute("aria-expanded", "false");
    fireEvent.click(header());
    await screen.findByRole("slider");

    expect(ruler()).toHaveAttribute("aria-valuenow", "0");
  });

  it("SPEC-ADDON-089 I3 — a file change clears the browse point", async () => {
    const { controller } = makeController({ duration: 70, paused: true });
    const view = renderSection({ mediaController: controller });
    await openNative();
    const el = ruler();
    stubRect(el);

    firePointer(el, "pointerMove", { pointerType: "mouse", clientX: pointerX(60, 70) });
    expect(ruler()).toHaveAttribute("aria-valuenow", "60");

    view.rerender(section({ fileId: "f2", mediaController: controller }));
    fireEvent.click(header());
    await screen.findByRole("slider");

    expect(ruler()).toHaveAttribute("aria-valuenow", "0");
  });
});

describe("SPEC-ADDON-089 I3 — a clock jump clears the browse point", () => {
  it.each([
    {
      name: "a 2 s step while playing keeps it",
      paused: false,
      start: 0,
      next: 2,
      ms: 250,
      kept: true,
    },
    {
      name: "a 2.5 s step while playing clears it",
      paused: false,
      start: 0,
      next: 2.5,
      ms: 250,
      kept: false,
    },
    {
      name: "a backward jump while playing clears it",
      paused: false,
      start: 60,
      next: 0,
      ms: 250,
      kept: false,
    },
    {
      name: "a 0.5 s change while paused keeps it",
      paused: true,
      start: 0,
      next: 0.5,
      ms: 1000,
      kept: true,
    },
    {
      name: "a 0.6 s change while paused clears it",
      paused: true,
      start: 0,
      next: 0.6,
      ms: 1000,
      kept: false,
    },
  ])(
    "SPEC-ADDON-089 I3 — $name",
    async ({ paused, start, next, ms, kept }) => {
      vi.useFakeTimers();
      const { controller, state } = makeController({
        duration: 70,
        paused,
        currentTime: start,
      });
      renderSection({ mediaController: controller });
      await openNative();
      const el = ruler();
      stubRect(el);

      firePointer(el, "pointerMove", { pointerType: "mouse", clientX: pointerX(60, 70) });
      expect(ruler()).toHaveAttribute("aria-valuenow", "60");

      await advance(ms, () => {
        state.currentTime = next;
      });

      if (kept) {
        expect(ruler()).toHaveAttribute("aria-valuenow", "60");
      } else {
        expect(Number(ruler().getAttribute("aria-valuenow"))).toBeLessThan(10);
      }
    },
  );
});
