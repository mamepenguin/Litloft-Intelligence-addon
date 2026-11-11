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
    {
      name: "a 10.6 s resume after a pause keeps it",
      paused: true,
      nextPaused: false,
      start: 10,
      next: 10.6,
      ms: 1000,
      kept: true,
    },
    {
      name: "a 40 s resume after a pause clears it",
      paused: true,
      nextPaused: false,
      start: 10,
      next: 40,
      ms: 1000,
      kept: false,
    },
    {
      name: "a resume backwards past 0.5 s clears it",
      paused: true,
      nextPaused: false,
      start: 10,
      next: 9,
      ms: 1000,
      kept: false,
    },
    {
      name: "a 3 s resume after a pause keeps it",
      paused: true,
      nextPaused: false,
      start: 10,
      next: 13,
      ms: 1000,
      kept: true,
    },
    {
      name: "a 0.5 s step back on resume keeps it",
      paused: true,
      nextPaused: false,
      start: 10,
      next: 9.5,
      ms: 1000,
      kept: true,
    },
    {
      name: "a 3.5 s resume at rate 2 keeps it",
      paused: true,
      nextPaused: false,
      start: 10,
      next: 13.5,
      ms: 1000,
      rate: 2,
      kept: true,
    },
  ])(
    "SPEC-ADDON-089 I3 — $name",
    async ({ paused, nextPaused = paused, start, next, ms, rate = 1, kept }) => {
      vi.useFakeTimers();
      const { controller, state } = makeController({
        duration: 70,
        paused,
        currentTime: start,
        playbackRate: rate,
      });
      renderSection({ mediaController: controller });
      await openNative();
      const el = ruler();
      stubRect(el);

      firePointer(el, "pointerDown", { pointerType: "touch", clientX: pointerX(60, 70) });
      firePointer(el, "pointerUp", { pointerType: "touch", clientX: pointerX(60, 70) });
      expect(ruler()).toHaveAttribute("aria-valuenow", "60");

      await advance(ms, () => {
        state.currentTime = next;
        state.paused = nextPaused;
      });

      expect(ruler()).toHaveAttribute(
        "aria-valuenow",
        String(kept ? 60 : Math.round(next)),
      );
    },
  );
});

type BrowseSource = "mouse" | "touch" | "keyboard";

type ClearEvent =
  | "pointerleave"
  | "focusleave"
  | "jump"
  | "seek"
  | "collapse"
  | "filechange";

const PLAYBACK_SECONDS = 10;
const JUMPED_SECONDS = 40;
const MOUSE_AND_FINGER_POINT = 60;
const KEYBOARD_POINT = 15;

function setBrowse(el: HTMLElement, source: BrowseSource): number {
  if (source === "mouse") {
    firePointer(el, "pointerMove", {
      pointerType: "mouse",
      clientX: pointerX(MOUSE_AND_FINGER_POINT, 70),
    });
    fireEvent.pointerOut(el, {
      pointerType: "mouse",
      relatedTarget: allTiles()[0],
    });
    return MOUSE_AND_FINGER_POINT;
  }
  if (source === "touch") {
    firePointer(el, "pointerDown", {
      pointerType: "touch",
      clientX: pointerX(MOUSE_AND_FINGER_POINT, 70),
    });
    firePointer(el, "pointerUp", {
      pointerType: "touch",
      clientX: pointerX(MOUSE_AND_FINGER_POINT, 70),
    });
    return MOUSE_AND_FINGER_POINT;
  }
  act(() => el.focus());
  fireEvent.keyDown(el, { key: "ArrowRight" });
  return KEYBOARD_POINT;
}

const LIFETIME: {
  name: string;
  source: BrowseSource;
  event: ClearEvent;
  kept: boolean;
}[] = [
  {
    name: "a mouse browse point is cleared when the pointer leaves the section",
    source: "mouse",
    event: "pointerleave",
    kept: false,
  },
  {
    name: "a mouse browse point survives focus leaving the section",
    source: "mouse",
    event: "focusleave",
    kept: true,
  },
  {
    name: "a mouse browse point survives a playback jump while the pointer is inside the section",
    source: "mouse",
    event: "jump",
    kept: true,
  },
  {
    name: "a mouse browse point is cleared by a seek on the ruler",
    source: "mouse",
    event: "seek",
    kept: false,
  },
  {
    name: "a mouse browse point is cleared by a collapse",
    source: "mouse",
    event: "collapse",
    kept: false,
  },
  {
    name: "a mouse browse point is cleared by a file change",
    source: "mouse",
    event: "filechange",
    kept: false,
  },
  {
    name: "a touch browse point survives the pointer leaving the section",
    source: "touch",
    event: "pointerleave",
    kept: true,
  },
  {
    name: "a touch browse point survives focus leaving the section",
    source: "touch",
    event: "focusleave",
    kept: true,
  },
  {
    name: "a touch browse point is cleared by a playback jump",
    source: "touch",
    event: "jump",
    kept: false,
  },
  {
    name: "a touch browse point is cleared by a tap on a tile",
    source: "touch",
    event: "seek",
    kept: false,
  },
  {
    name: "a touch browse point is cleared by a collapse",
    source: "touch",
    event: "collapse",
    kept: false,
  },
  {
    name: "a touch browse point is cleared by a file change",
    source: "touch",
    event: "filechange",
    kept: false,
  },
  {
    name: "a keyboard browse point survives the pointer leaving the section",
    source: "keyboard",
    event: "pointerleave",
    kept: true,
  },
  {
    name: "a keyboard browse point is cleared when focus leaves the section",
    source: "keyboard",
    event: "focusleave",
    kept: false,
  },
  {
    name: "a keyboard browse point is cleared by a playback jump",
    source: "keyboard",
    event: "jump",
    kept: false,
  },
  {
    name: "a keyboard browse point is cleared by Enter on the ruler",
    source: "keyboard",
    event: "seek",
    kept: false,
  },
  {
    name: "a keyboard browse point is cleared by a collapse",
    source: "keyboard",
    event: "collapse",
    kept: false,
  },
  {
    name: "a keyboard browse point is cleared by a file change",
    source: "keyboard",
    event: "filechange",
    kept: false,
  },
];

describe("SPEC-ADDON-089 I2, I3 — what a browse point outlives, by the input that set it", () => {
  it.each(LIFETIME)("SPEC-ADDON-089 I2, I3 — $name", async ({ source, event, kept }) => {
    const jumping = event === "jump";
    if (jumping) vi.useFakeTimers();
    const { controller, state } = makeController({
      duration: 70,
      currentTime: PLAYBACK_SECONDS,
      paused: !jumping,
    });
    const view = renderSection({ mediaController: controller });
    await openNative();
    const el = ruler();
    stubRect(el);
    const root = view.container.firstElementChild as HTMLElement;

    const browse = setBrowse(el, source);
    expect(ruler()).toHaveAttribute("aria-valuenow", String(browse));

    switch (event) {
      case "pointerleave":
        firePointer(root, "pointerLeave", {
          pointerType: source === "mouse" ? "mouse" : "touch",
        });
        break;
      case "focusleave":
        fireEvent.focusOut(el, { relatedTarget: document.body });
        break;
      case "jump":
        await advance(250, () => {
          state.currentTime = JUMPED_SECONDS;
        });
        break;
      case "seek":
        if (source === "mouse") fireEvent.click(el, { clientX: pointerX(browse, 70) });
        else if (source === "touch") fireEvent.click(tile(browse));
        else fireEvent.keyDown(el, { key: "Enter" });
        break;
      case "collapse":
        fireEvent.click(header());
        fireEvent.click(header());
        await screen.findByRole("slider");
        break;
      case "filechange":
        view.rerender(section({ fileId: "f2", mediaController: controller }));
        fireEvent.click(header());
        await screen.findByRole("slider");
        break;
    }

    const cleared = jumping ? JUMPED_SECONDS : PLAYBACK_SECONDS;
    expect(ruler()).toHaveAttribute(
      "aria-valuenow",
      String(kept ? browse : cleared),
    );
  });
});
