/**
 * RED-phase tests for TranscriptSection transcript-refine UI.
 *
 * Spec: docs/superpowers/specs/2026-04-15-intelligence-transcript-refine.md
 *
 * Covers:
 *   - "AI で修正" button appears only when features.transcript_refine !== false
 *   - "AI 修正済み" badge renders for chunks with refinedAt
 *   - textOriginal tooltip is present on refined chunks (title attr)
 *   - Revert button only appears when at least one chunk is refined
 *
 * The enhancements are not yet implemented — these tests are expected
 * to fail (RED phase). They intentionally do NOT try to patch internal
 * module state; instead they rely on props + addon status context that
 * the future implementation must accept.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  act,
  cleanup,
  createEvent,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import React from "react";

/** No VTT at either endpoint, which is what most of this file assumes. */
const FETCH_MISS = {
  ok: false,
  status: 404,
  text: async () => "",
  json: async () => null,
} as Response;

/**
 * The response the module mock resolves to, and the one a test that
 * overrides it puts back.
 *
 * `vi.hoisted` because `vi.mock` factories are hoisted above every
 * `const` in the file. Written out twice it was worse than duplication:
 * the copy inside the factory became unreachable the moment a
 * `beforeEach` started resetting the mock, so editing it changed
 * nothing and no test said so.
 */
const TRANSCRIPT_RESPONSE = vi.hoisted(() => ({
  available: true,
  file_id: "abc",
  drive: "family",
  language: "ja",
  chunks: [
    {
      index: 0,
      text: "これは修正された文章です。",
      start: 0,
      end: 5,
      refinedAt: "2026-04-15T00:00:00Z",
      textOriginal: "これはげんぶんの文章です。",
    },
    { index: 1, text: "未修正の文章。", start: 5, end: 10 },
  ],
}));

// Mock global fetch (used by component for VTT endpoints)
const fetchMock = vi.fn().mockResolvedValue(FETCH_MISS);
vi.stubGlobal("fetch", fetchMock);

// Mock the addon API module to return controlled transcript data.
// The module exports `getFileTranscript` which the component calls on
// mount. The test data includes refined + unrefined chunks.
vi.mock("@/addons/intelligence/api", () => ({
  getFileTranscript: vi.fn().mockResolvedValue(TRANSCRIPT_RESPONSE),
  refineFileTranscript: vi.fn().mockResolvedValue({
    job_id: "job-1",
    chunk_count: 2,
  }),
  revertFileTranscript: vi.fn().mockResolvedValue({ success: true }),
}));

// Mock the addon slots provider so `useAddonSlots()` exposes
// `features.transcript_refine`. The exact hook surface is to be
// finalised during implementation — we target the natural shape.
const mockAddonStatus = {
  features: { transcript_refine: "manual" as string | false },
};
vi.mock("@/components/AddonSlotsProvider", () => ({
  useAddonStatus: () => mockAddonStatus,
  useAddonSlots: () => ({ slots: {} }),
}));

// Use the real TranscriptSection from the intelligence addon directory.
// This is the same path used by the build-time symlink copy.
import TranscriptSection from "@/addons/intelligence/TranscriptSection";
import {
  clearSourceCaptures,
  getSourceCaptures,
} from "@/lib/sourceCapture";
import {
  clearTranscriptScroll,
  recallTranscriptScroll,
  rememberTranscriptScroll,
} from "@/addons/intelligence/transcriptScroll";

async function transcriptApiMock() {
  const apiMock = await import("@/addons/intelligence/api");
  return apiMock.getFileTranscript as unknown as ReturnType<typeof vi.fn>;
}

/**
 * Both mocks are one object each, shared by every test in this file.
 *
 * A test that changes what one of them answers changes it for whatever
 * runs next — and under the shuffled-order job that is not the test
 * written below it. This happened twice while the file was being
 * written: one test handing everything an untranscribed video timed out
 * eight tests in two other describes, and one leaving word-level cues in
 * `fetch` broke a test that asserts an empty panel. Both stayed
 * invisible in source order only because they happened to be near the
 * end. So neither is cleaned up by the test that dirtied it — both are
 * put back before every test, wherever it runs.
 *
 * `mockReset` and not merely `mockResolvedValue`: an unconsumed
 * `mockResolvedValueOnce` would otherwise still be at the head of the
 * queue when the next test asked.
 */
beforeEach(async () => {
  const getFileTranscript = await transcriptApiMock();
  getFileTranscript.mockReset();
  getFileTranscript.mockResolvedValue(TRANSCRIPT_RESPONSE);
  fetchMock.mockReset();
  fetchMock.mockResolvedValue(FETCH_MISS);
});


function renderSection() {
  return render(<TranscriptSection fileId="abc" drive="family" />);
}

/**
 * Wait for the highlight to reach a cue.
 *
 * Anything about following — the chip, the auto-scroll, suspending
 * either — needs the transcript rendered *and* a cue active, and the two
 * arrive on different ticks: the body from the transcript fetch, the
 * highlight from the first clock sync. Acting in the gap is acting on a
 * component that is not yet in the state under test, where doing nothing
 * is the correct behaviour and reads as a bug.
 */
async function waitForActiveCue(container: HTMLElement): Promise<void> {
  await waitFor(() =>
    expect(container.querySelector('[aria-current="true"]')).not.toBeNull(),
  );
}

describe("TranscriptSection — transcript refine UI", () => {
  beforeEach(() => {
    mockAddonStatus.features.transcript_refine = "manual";
    fetchMock.mockClear();
    // Kept across mounts on purpose, so it is kept across tests too.
    clearTranscriptScroll();
    clearSourceCaptures("family");
  });

  afterEach(() => {
    cleanup();
  });

  it("shows the transcript cleanup button when feature is enabled", async () => {
    renderSection();
    // Button label text per spec UI section.
    const btn = await screen.findByRole("button", { name: /Clean up with AI/ });
    expect(btn).toBeInTheDocument();
  });

  it("hides the refine button when feature flag is 'false'", async () => {
    mockAddonStatus.features.transcript_refine = false;
    renderSection();
    // Give the async mount a tick — chunks still render, button shouldn't.
    await screen.findByText("未修正の文章。");
    expect(
      screen.queryByRole("button", { name: /Clean up with AI/ })
    ).not.toBeInTheDocument();
  });

  it("renders the AI cleanup badge for refined chunks", async () => {
    renderSection();
    const badges = await screen.findAllByText(/AI cleaned/);
    // One badge per refined chunk (1 out of 2 in our fixture).
    expect(badges).toHaveLength(1);
  });

  it("adds a transcript cue with its time range to the capture basket", async () => {
    render(
      <TranscriptSection
        fileId="abc"
        drive="family"
        filename="meeting.mp4"
        fileType="video"
      />,
    );

    const buttons = await screen.findAllByRole("button", {
      name: /capture basket/,
    });
    fireEvent.click(buttons[0]);

    expect(getSourceCaptures("family")).toEqual([
      expect.objectContaining({
        sourceFileId: "abc",
        filename: "meeting.mp4",
        kind: "transcript",
        quote: "これは修正された文章です。",
        locator: expect.objectContaining({ seconds: 0, endSeconds: 5 }),
      }),
    ]);
  });

  // RED phase: not yet implemented
  it.todo("shows textOriginal in a tooltip (title attr) on refined chunks");

  // RED phase: not yet implemented
  it.todo("shows the revert button only when at least one chunk is refined");

  it("hides the revert button when no chunks are refined", async () => {
    const apiMock = await import("@/addons/intelligence/api");
    (apiMock.getFileTranscript as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValueOnce({
        available: true,
        file_id: "abc",
        drive: "family",
        language: "ja",
        chunks: [
          { index: 0, text: "一切修正なし。", start: 0, end: 5 },
        ],
      });

    renderSection();
    await screen.findByText("一切修正なし。");
    expect(
      screen.queryByRole("button", { name: /Undo AI refine/ })
    ).not.toBeInTheDocument();
  });
});

// Spec 2026-08-11-transcript-following-playback.md §5. The highlight
// used to bind `timeupdate` on an HTMLVideoElement, so it never worked
// for a YouTube IFrame player. It now reads the shared playback clock,
// which every backend feeds.
describe("TranscriptSection — following playback", () => {
  function stubController(state: { currentTime: number; paused: boolean }) {
    return {
      seek: vi.fn((s: number) => {
        state.currentTime = s;
      }),
      play: vi.fn(),
      pause: vi.fn(),
      togglePlay: vi.fn(),
      toggleMute: vi.fn(),
      toggleFullscreen: vi.fn(),
      getCurrentTime: () => state.currentTime,
      getDuration: () => 10,
      isPaused: () => state.paused,
      isMuted: () => false,
      getVolume: () => 1,
      setVolume: vi.fn(),
      getPlaybackRate: () => 1,
      setPlaybackRate: vi.fn(),
      getBufferedFraction: () => 0,
    };
  }

  beforeEach(() => {
    mockAddonStatus.features.transcript_refine = "manual";
    fetchMock.mockClear();
    // Kept across mounts on purpose, so it is kept across tests too.
    clearTranscriptScroll();
  });

  afterEach(() => {
    cleanup();
    vi.useRealTimers();
  });

  it("highlights the cue playback is inside, with no media element involved", async () => {
    const state = { currentTime: 7, paused: false };
    const mc = stubController(state);
    render(
      <TranscriptSection fileId="abc" drive="family" mediaController={mc} />,
    );

    // Fixture: chunk 0 spans 0-5, chunk 1 spans 5-10.
    const active = await screen.findByRole("button", { current: true });
    expect(active).toHaveTextContent("未修正の文章。");
  });

  it("moves the highlight as playback advances", async () => {
    const state = { currentTime: 1, paused: false };
    const mc = stubController(state);
    render(
      <TranscriptSection fileId="abc" drive="family" mediaController={mc} />,
    );

    let active = await screen.findByRole("button", { current: true });
    expect(active).toHaveTextContent("これは修正された文章です。");

    state.currentTime = 8;
    await waitFor(async () => {
      active = await screen.findByRole("button", { current: true });
      expect(active).toHaveTextContent("未修正の文章。");
    });
  });

  it("seeks through the controller when a row is clicked", async () => {
    const state = { currentTime: 0, paused: true };
    const mc = stubController(state);
    render(
      <TranscriptSection fileId="abc" drive="family" mediaController={mc} />,
    );

    const rows = await screen.findAllByRole("button");
    const secondCue = rows.find((r) => r.textContent?.includes("未修正の文章。"));
    fireEvent.click(secondCue!);

    expect(mc.seek).toHaveBeenCalledWith(5);
    expect(mc.play).toHaveBeenCalled();
  });

  it("highlights nothing without a controller", async () => {
    render(<TranscriptSection fileId="abc" drive="family" />);
    await screen.findAllByText("これは修正された文章です。");
    expect(screen.queryByRole("button", { current: true })).toBeNull();
  });
});

describe("TranscriptSection — rail vs stacked form", () => {
  beforeEach(() => {
    mockAddonStatus.features.transcript_refine = "manual";
    fetchMock.mockClear();
    // Kept across mounts on purpose, so it is kept across tests too.
    clearTranscriptScroll();
  });

  afterEach(() => {
    cleanup();
  });

  it("keeps a bounded box in the stacked form", async () => {
    const { container } = render(
      <TranscriptSection fileId="abc" drive="family" />,
    );
    await screen.findByText("未修正の文章。");

    // Filling the height here would mean filling the page.
    const list = container.querySelector(".overflow-y-auto");
    expect(list?.className).toContain("max-h-80");
  });

  it("fills the available height in the rail", async () => {
    const { container } = render(
      <TranscriptSection fileId="abc" drive="family" fillHeight />,
    );
    await screen.findByText("未修正の文章。");

    const list = container.querySelector(".overflow-y-auto");
    expect(list?.className).not.toContain("max-h-80");
    expect(list?.className).toContain("flex-1");
  });

  it("stages a capture from a row in either form", async () => {
    clearSourceCaptures("family");
    render(
      <TranscriptSection
        fileId="abc"
        drive="family"
        filename="meeting.mp4"
        fileType="video"
        fillHeight
      />,
    );

    const buttons = await screen.findAllByRole("button", {
      name: /capture basket/,
    });
    fireEvent.click(buttons[0]);

    expect(getSourceCaptures("family")).toHaveLength(1);
  });
});

// Spec 2026-08-11-transcript-following-playback.md §6. Auto-scroll that
// always wins is worse than none: reading ahead in a tall rail would
// mean being pulled back every few seconds.
describe("TranscriptSection — following without fighting the reader", () => {
  function stubController(state: { currentTime: number }) {
    return {
      seek: vi.fn(),
      play: vi.fn(),
      pause: vi.fn(),
      togglePlay: vi.fn(),
      toggleMute: vi.fn(),
      toggleFullscreen: vi.fn(),
      getCurrentTime: () => state.currentTime,
      getDuration: () => 10,
      isPaused: () => false,
      isMuted: () => false,
      getVolume: () => 1,
      setVolume: vi.fn(),
      getPlaybackRate: () => 1,
      setPlaybackRate: vi.fn(),
      getBufferedFraction: () => 0,
    };
  }

  const CHIP = "Back to current position";

  beforeEach(() => {
    mockAddonStatus.features.transcript_refine = "manual";
    fetchMock.mockClear();
    // Kept across mounts on purpose, so it is kept across tests too.
    clearTranscriptScroll();
  });

  afterEach(() => {
    cleanup();
  });

  async function renderFollowing() {
    const state = { currentTime: 1 };
    const mc = stubController(state);
    const utils = render(
      <TranscriptSection fileId="abc" drive="family" mediaController={mc} fillHeight />,
    );
    await screen.findByText("未修正の文章。");
    await waitForActiveCue(utils.container);
    const list = utils.container.querySelector(".overflow-y-auto")!;
    return { ...utils, mc, state, list };
  }

  it("offers no chip while it is still following", async () => {
    await renderFollowing();
    expect(screen.queryByRole("button", { name: CHIP })).toBeNull();
  });

  it("stops following when the reader scrolls the list", async () => {
    const { list } = await renderFollowing();

    fireEvent.wheel(list);

    expect(
      await screen.findByRole("button", { name: CHIP }),
    ).toBeInTheDocument();
  });

  it("stops following on a touch drag", async () => {
    const { list } = await renderFollowing();

    fireEvent.touchMove(list);

    expect(
      await screen.findByRole("button", { name: CHIP }),
    ).toBeInTheDocument();
  });

  it("treats a scrollbar drag as taking over, but not a click on a row", async () => {
    const { list } = await renderFollowing();

    // Landing on a row is someone clicking a cue, not grabbing the bar.
    const row = await screen.findByText("未修正の文章。");
    fireEvent.pointerDown(row);
    expect(screen.queryByRole("button", { name: CHIP })).toBeNull();

    fireEvent.pointerDown(list);
    expect(
      await screen.findByRole("button", { name: CHIP }),
    ).toBeInTheDocument();
  });

  it("resumes when the chip is pressed", async () => {
    const { list } = await renderFollowing();
    fireEvent.wheel(list);
    const chip = await screen.findByRole("button", { name: CHIP });

    fireEvent.click(chip);

    await waitFor(() =>
      expect(screen.queryByRole("button", { name: CHIP })).toBeNull(),
    );
  });

  it("resumes when the reader jumps to a cue", async () => {
    const { list } = await renderFollowing();
    fireEvent.wheel(list);
    await screen.findByRole("button", { name: CHIP });

    const rows = await screen.findAllByRole("button");
    const cue = rows.find((r) => r.textContent?.includes("未修正の文章。"));
    fireEvent.click(cue!);

    await waitFor(() =>
      expect(screen.queryByRole("button", { name: CHIP })).toBeNull(),
    );
  });

  it("keeps the chip out of the way when no cue is playing", async () => {
    // Nothing to go back to, so the offer would be meaningless.
    const utils = render(<TranscriptSection fileId="abc" drive="family" fillHeight />);
    await screen.findByText("未修正の文章。");
    const list = utils.container.querySelector(".overflow-y-auto")!;

    fireEvent.wheel(list);

    expect(screen.queryByRole("button", { name: CHIP })).toBeNull();
  });
});

describe("TranscriptSection — suspension actually stops the scrolling", () => {
  beforeEach(() => {
    mockAddonStatus.features.transcript_refine = "manual";
    fetchMock.mockClear();
    // Kept across mounts on purpose, so it is kept across tests too.
    clearTranscriptScroll();
  });

  afterEach(() => {
    // Deliberately not vi.restoreAllMocks(): that also unwinds the
    // module-level getFileTranscript mock and leaves the next test
    // with a function that returns undefined. The rect spies live on
    // elements this cleanup throws away.
    cleanup();
  });

  /**
   * jsdom gives every element a zero rect, so the auto-scroll's
   * "is the active cue out of view" check is never true and it never
   * runs. Giving the list a rect below the cue's makes it run, which is
   * the only way to observe that suspension stops it — without this the
   * tests above would pass against an implementation that shows the
   * chip and keeps yanking the reader back.
   */
  function makeAutoScrollReachable(list: Element) {
    vi.spyOn(list, "getBoundingClientRect").mockReturnValue({
      top: 50,
      bottom: 60,
      height: 10,
    } as DOMRect);
  }

  async function setup() {
    const state = { currentTime: 1 };
    const mc = {
      seek: vi.fn(),
      play: vi.fn(),
      pause: vi.fn(),
      togglePlay: vi.fn(),
      toggleMute: vi.fn(),
      toggleFullscreen: vi.fn(),
      getCurrentTime: () => state.currentTime,
      getDuration: () => 10,
      isPaused: () => false,
      isMuted: () => false,
      getVolume: () => 1,
      setVolume: vi.fn(),
      getPlaybackRate: () => 1,
      setPlaybackRate: vi.fn(),
      getBufferedFraction: () => 0,
    };
    const utils = render(
      <TranscriptSection fileId="abc" drive="family" mediaController={mc} fillHeight />,
    );
    await screen.findByText("未修正の文章。");
    await waitForActiveCue(utils.container);
    const list = utils.container.querySelector(".overflow-y-auto")! as HTMLElement;
    const scrollTo = vi.fn();
    list.scrollTo = scrollTo;
    makeAutoScrollReachable(list);
    return { list, scrollTo, state };
  }

  it("scrolls the list while following", async () => {
    const { scrollTo, state } = await setup();

    state.currentTime = 8;
    await waitFor(() => expect(scrollTo).toHaveBeenCalled());
  });

  it("leaves the list alone once the reader has taken over", async () => {
    const { list, scrollTo, state } = await setup();

    fireEvent.wheel(list);
    scrollTo.mockClear();

    state.currentTime = 8;
    // Wait for the highlight to actually move first. Asserting straight
    // after the wheel event would pass against any implementation at
    // all — nothing has happened yet at that point.
    await waitFor(async () => {
      const active = await screen.findByRole("button", { current: true });
      expect(active).toHaveTextContent("未修正の文章。");
    });

    // The highlight moved on; the list did not.
    expect(scrollTo).not.toHaveBeenCalled();
  });
});

describe("TranscriptSection — a list that appears after its cues", () => {
  beforeEach(() => {
    mockAddonStatus.features.transcript_refine = "manual";
    clearTranscriptScroll();
  });

  afterEach(() => {
    cleanup();
  });

  it("still stops following when the reader scrolls it", async () => {
    // Word cues of the same count arrive while the transcript is still
    // loading, so the cue count is already final when the list mounts.
    let release: (value: typeof TRANSCRIPT_RESPONSE) => void = () => undefined;
    const getFileTranscript = await transcriptApiMock();
    getFileTranscript.mockReturnValue(
      new Promise((resolve) => {
        release = resolve;
      }),
    );
    fetchMock.mockResolvedValue({
      ok: true,
      status: 200,
      text: async () =>
        "WEBVTT\n\n00:00:00.000 --> 00:00:05.000\nfirst\n\n00:00:05.000 --> 00:00:10.000\nsecond\n",
      json: async () => null,
    } as Response);
    const state = { currentTime: 1 };
    const mc = {
      seek: vi.fn(),
      play: vi.fn(),
      pause: vi.fn(),
      togglePlay: vi.fn(),
      toggleMute: vi.fn(),
      toggleFullscreen: vi.fn(),
      getCurrentTime: () => state.currentTime,
      getDuration: () => 10,
      isPaused: () => false,
      isMuted: () => false,
      getVolume: () => 1,
      setVolume: vi.fn(),
      getPlaybackRate: () => 1,
      setPlaybackRate: vi.fn(),
      getBufferedFraction: () => 0,
    };
    const utils = render(
      <TranscriptSection fileId="abc" drive="family" mediaController={mc} fillHeight />,
    );
    await act(() => new Promise((resolve) => setTimeout(resolve, 50)));
    await act(async () => release(TRANSCRIPT_RESPONSE));
    // The text chunks, which replace the word cues nobody chose.
    await screen.findByText("未修正の文章。");
    await waitForActiveCue(utils.container);

    fireEvent.wheel(utils.container.querySelector(".overflow-y-auto")!);
    expect(
      await screen.findByRole("button", { name: "Back to current position" }),
    ).toBeInTheDocument();
  });
});

describe("TranscriptSection — in a host whose scroller encloses the list", () => {
  // The column form of the inspector: the list grows to its full length
  // and the box around the whole inspector is what scrolls, with the tab
  // strip pinned over its top 40px.
  const HOST = { top: 100, bottom: 400, clientHeight: 300, scrollHeight: 3000 };
  const STRIP_PX = 40;
  const CUE = { top: 800, height: 30 };
  const CHIP = "Back to current position";

  beforeEach(() => {
    mockAddonStatus.features.transcript_refine = "manual";
    fetchMock.mockClear();
    clearTranscriptScroll();
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
    // Shadowed on HTMLElement; jsdom's own getters live on Element.
    delete (HTMLElement.prototype as { clientHeight?: number }).clientHeight;
    delete (HTMLElement.prototype as { scrollHeight?: number }).scrollHeight;
  });

  async function setup({
    hidden = false,
    published = true,
    hostOverflows = true,
  } = {}) {
    const state = { currentTime: 1 };
    const mc = {
      seek: vi.fn(),
      play: vi.fn(),
      pause: vi.fn(),
      togglePlay: vi.fn(),
      toggleMute: vi.fn(),
      toggleFullscreen: vi.fn(),
      getCurrentTime: () => state.currentTime,
      getDuration: () => 10,
      isPaused: () => false,
      isMuted: () => false,
      getVolume: () => 1,
      setVolume: vi.fn(),
      getPlaybackRate: () => 1,
      setPlaybackRate: vi.fn(),
      getBufferedFraction: () => 0,
    };

    // Geometry in place before the first render, as a browser has it.
    const byId = (id: string) =>
      document.querySelector<HTMLElement>(`[data-testid='${id}']`);
    const isList = (el: Element) =>
      el.classList.contains("overflow-y-auto") && !!byId("fits")?.contains(el);
    const sizes = (el: HTMLElement): { client: number; scroll: number } => {
      if (el === byId("host")) {
        return {
          client: HOST.clientHeight,
          scroll: hostOverflows ? HOST.scrollHeight : HOST.clientHeight,
        };
      }
      if (el === byId("tall")) return { client: 300, scroll: 5000 };
      if (el === byId("fits") || isList(el)) return { client: 4000, scroll: 4000 };
      const cue = el.getAttribute("aria-current") === "true" ? CUE.height : 0;
      return { client: cue, scroll: cue };
    };
    Object.defineProperty(HTMLElement.prototype, "clientHeight", {
      configurable: true,
      get(this: HTMLElement) {
        return sizes(this).client;
      },
    });
    Object.defineProperty(HTMLElement.prototype, "scrollHeight", {
      configurable: true,
      get(this: HTMLElement) {
        return sizes(this).scroll;
      },
    });
    vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockImplementation(
      function (this: HTMLElement) {
        if (this.closest("[hidden]")) {
          return { top: 0, bottom: 0, height: 0 } as DOMRect;
        }
        if (this === byId("host")) {
          return { top: HOST.top, bottom: HOST.bottom, height: 300 } as DOMRect;
        }
        if (this.getAttribute("aria-current") === "true") {
          return {
            top: CUE.top,
            bottom: CUE.top + CUE.height,
            height: CUE.height,
          } as DOMRect;
        }
        return { top: HOST.top, bottom: HOST.top + 4000, height: 4000 } as DOMRect;
      },
    );
    // jsdom neither inherits custom properties nor compiles the list's
    // `overflow-y-auto` class; a browser does both.
    const computed = window.getComputedStyle.bind(window);
    vi.spyOn(window, "getComputedStyle").mockImplementation((el, pseudo) => {
      const style = computed(el, pseudo);
      const host = byId("host");
      if (!host || el === host || !host.contains(el)) return style;
      return new Proxy(style, {
        get(target, key) {
          if (key === "getPropertyValue") {
            return (name: string) =>
              name === "--inspector-sticky-top" && published
                ? `${STRIP_PX}px`
                : target.getPropertyValue(name);
          }
          if (key === "overflowY" && isList(el)) return "auto";
          const value = Reflect.get(target, key);
          return typeof value === "function" ? value.bind(target) : value;
        },
      });
    });

    // Between the list and the host: a box that scrolls but has nothing
    // to scroll, and a box that overflows but does not scroll. Neither is
    // the scroller.
    const utils = render(
      <div
        data-testid="host"
        data-inspector-scroller={published ? "" : undefined}
        style={{ overflowY: "auto" }}
      >
        {/* Where the host puts `hidden`: on a panel inside its scroller. */}
        <div data-testid="tall" hidden={hidden} style={{ overflowY: "visible" }}>
          <div data-testid="fits" style={{ overflowY: "auto" }}>
            <TranscriptSection fileId="abc" drive="family" mediaController={mc} fillHeight />
          </div>
        </div>
      </div>,
    );
    const host = screen.getByTestId("host");
    const hostScrollTo = vi.fn();
    host.scrollTo = hostScrollTo;
    screen.getByTestId("tall").scrollTo = vi.fn();
    screen.getByTestId("fits").scrollTo = vi.fn();

    await screen.findByText("未修正の文章。");
    await waitForActiveCue(utils.container);

    const list = utils.container.querySelector(
      "[data-testid='fits'] .overflow-y-auto",
    ) as HTMLElement;
    const listScrollTo = vi.fn();
    list.scrollTo = listScrollTo;
    return { host, hostScrollTo, list, listScrollTo, state };
  }

  it("scrolls that scroller back to the cue, centred below the strip", async () => {
    const { list, hostScrollTo, listScrollTo } = await setup();
    fireEvent.wheel(list);
    fireEvent.click(await screen.findByRole("button", { name: CHIP }));

    // 700px down the host, less the strip, less half of what is left.
    await waitFor(() =>
      expect(hostScrollTo).toHaveBeenCalledWith({ top: 545, behavior: "smooth" }),
    );
    expect(listScrollTo).not.toHaveBeenCalled();
    expect(screen.getByTestId("fits").scrollTo).not.toHaveBeenCalled();
    expect(screen.getByTestId("tall").scrollTo).not.toHaveBeenCalled();
  });

  it("stops following when the reader scrolls that scroller outside the list", async () => {
    const { host } = await setup();
    fireEvent.touchMove(host);
    expect(await screen.findByRole("button", { name: CHIP })).toBeInTheDocument();
  });

  it("finds that scroller even when it had nothing to scroll as the cues arrived", async () => {
    // The sheet opens on another tab, which may not fill it.
    const { host } = await setup({ hostOverflows: false });
    fireEvent.touchMove(host);
    expect(await screen.findByRole("button", { name: CHIP })).toBeInTheDocument();
  });

  it("keeps following when that scroller is scrolled while the transcript is not shown", async () => {
    const { host } = await setup({ hidden: true });
    fireEvent.touchMove(host);
    fireEvent.wheel(host);
    const press = createEvent.pointerDown(host);
    Object.defineProperty(press, "pointerType", { value: "mouse" });
    fireEvent(host, press);
    screen.getByTestId("tall").hidden = false;
    await act(() => new Promise((resolve) => setTimeout(resolve, 50)));
    expect(screen.queryByRole("button", { name: CHIP })).toBeNull();
  });

  describe("shown again after another tab moved the scroller", () => {
    const sizeCallbacks: Array<(entries: { contentRect: { height: number } }[]) => void> = [];
    const original = (globalThis as { ResizeObserver?: unknown }).ResizeObserver;

    beforeEach(() => {
      sizeCallbacks.length = 0;
      (globalThis as { ResizeObserver?: unknown }).ResizeObserver = class {
        constructor(cb: (typeof sizeCallbacks)[number]) {
          sizeCallbacks.push(cb);
        }
        observe() {}
        disconnect() {}
      };
    });

    afterEach(() => {
      (globalThis as { ResizeObserver?: unknown }).ResizeObserver = original;
    });

    const resize = (height: number) =>
      act(() => {
        for (const cb of sizeCallbacks) cb([{ contentRect: { height } }]);
      });

    it("brings the playing cue back into view while following", async () => {
      const { hostScrollTo } = await setup({ hidden: true });
      await resize(0);
      hostScrollTo.mockClear();

      screen.getByTestId("tall").hidden = false;
      await resize(4000);

      expect(hostScrollTo).toHaveBeenCalledWith({ top: 545, behavior: "smooth" });
    });

    it("brings it back after being shown, hidden and shown again", async () => {
      const { hostScrollTo } = await setup();
      await resize(4000);
      screen.getByTestId("tall").hidden = true;
      await resize(0);
      hostScrollTo.mockClear();

      screen.getByTestId("tall").hidden = false;
      await resize(4000);

      expect(hostScrollTo).toHaveBeenCalledWith({ top: 545, behavior: "smooth" });
    });

    it("leaves the scroller where the reader put it once following is off", async () => {
      const { hostScrollTo, list } = await setup({ hidden: true });
      await resize(0);
      screen.getByTestId("tall").hidden = false;
      await resize(4000);
      fireEvent.wheel(list);
      await screen.findByRole("button", { name: CHIP });
      hostScrollTo.mockClear();

      screen.getByTestId("tall").hidden = true;
      await resize(0);
      screen.getByTestId("tall").hidden = false;
      await resize(4000);

      expect(hostScrollTo).not.toHaveBeenCalled();
    });

    it("does not re-aim on a change of size that was not a reveal", async () => {
      const { hostScrollTo } = await setup();
      await resize(4000);
      hostScrollTo.mockClear();
      await resize(4200);
      expect(hostScrollTo).not.toHaveBeenCalled();
    });
  });

  it("takes a mouse press on that scroller as a scrollbar drag, and a touch as nothing", async () => {
    const { host } = await setup();
    // jsdom has no PointerEvent, so the init cannot carry the type.
    const press = (pointerType: string) => {
      const event = createEvent.pointerDown(host);
      Object.defineProperty(event, "pointerType", { value: pointerType });
      fireEvent(host, event);
    };
    press("touch");
    press("pen");
    await act(() => new Promise((resolve) => setTimeout(resolve, 50)));
    expect(screen.queryByRole("button", { name: CHIP })).toBeNull();

    press("mouse");
    expect(await screen.findByRole("button", { name: CHIP })).toBeInTheDocument();
  });

  it("does not reach past its own list when the host has not said it scrolls it", async () => {
    // A page scroller around a transcript short enough to fit: scrolling
    // it would move the video off the screen.
    const { list, hostScrollTo, state } = await setup({ published: false });
    state.currentTime = 8;
    await waitFor(async () => {
      const active = await screen.findByRole("button", { current: true });
      expect(active).toHaveTextContent("未修正の文章。");
    });
    fireEvent.wheel(list);
    fireEvent.click(await screen.findByRole("button", { name: CHIP }));

    expect(hostScrollTo).not.toHaveBeenCalled();
    expect(screen.getByTestId("tall").scrollTo).not.toHaveBeenCalled();
    expect(screen.getByTestId("fits").scrollTo).not.toHaveBeenCalled();
  });

  it("follows playback by scrolling that scroller", async () => {
    const { hostScrollTo, state } = await setup();
    state.currentTime = 8;
    await waitFor(() => expect(hostScrollTo).toHaveBeenCalled());
  });

  it("leaves it alone while the transcript is not shown", async () => {
    const { hostScrollTo, state } = await setup({ hidden: true });
    state.currentTime = 8;
    await waitFor(async () => {
      const active = await screen.findByRole("button", { current: true, hidden: true });
      expect(active).toHaveTextContent("未修正の文章。");
    });
    expect(hostScrollTo).not.toHaveBeenCalled();
  });
});

// M-3. A transcript is hundreds of rows long, and until now every one
// of them drew the same quote button at all times: a grey rule down the
// right edge of the text it annotates, and — to a screen reader — the
// same four words several hundred times over, with nothing to say which
// line each one would quote.
//
// The reveal itself is CSS (`opacity-0` lifted by `group-hover/cue`,
// `group-focus-within/cue`, `pointer-coarse`), and jsdom loads no
// stylesheet, so no assertion here can see the button appear. What these
// check is the contract the CSS hangs off: the row is the group, the
// button names the signals, and it stays in the tab order while hidden.
// The appearance itself is on the manual 1512 / 400 / 375 pass.
describe("TranscriptSection — per-row capture buttons", () => {
  beforeEach(() => {
    mockAddonStatus.features.transcript_refine = "manual";
    fetchMock.mockClear();
    // Kept across mounts on purpose, so it is kept across tests too.
    clearTranscriptScroll();
    clearSourceCaptures("family");
  });

  afterEach(() => {
    cleanup();
  });

  async function captureButtons(): Promise<HTMLElement[]> {
    return screen.findAllByRole("button", { name: /capture basket/ });
  }

  it("names each button for the line it would quote", async () => {
    renderSection();
    const buttons = await captureButtons();

    // Exactly the two cues in the fixture. A lower bound would pass on a
    // render that produced one button and on one that produced fifty.
    expect(buttons).toHaveLength(2);
    const names = buttons.map((b) => b.getAttribute("aria-label"));
    expect(names).toEqual([
      "Add the 0:00 line to the capture basket",
      "Add the 0:05 line to the capture basket",
    ]);
    // The point of the timestamp is that the names differ.
    expect(new Set(names).size).toBe(names.length);
    // And the name is the only one. A `title` beside an `aria-label`
    // becomes the accessible *description*, which NVDA and JAWS read
    // after the name — the same sentence, twice.
    expect(buttons.map((b) => b.getAttribute("title"))).toEqual([null, null]);
  });

  it("hangs the reveal on the row, not on the button alone", async () => {
    renderSection();
    const buttons = await captureButtons();

    for (const button of buttons) {
      // `classList.contains` matches whole tokens. `className.toContain`
      // would not: it also says yes to `pointer-coarse:opacity-0` when
      // asked about `opacity-0`, so the assertions that matter most
      // here would survive being broken.
      expect(button.parentElement?.classList.contains("group/cue")).toBe(true);
      // Hover anywhere on the row, or focus the row's seek button, and
      // the quote button comes with it.
      expect(
        button.classList.contains("group-hover/cue:opacity-100"),
      ).toBe(true);
      expect(
        button.classList.contains("group-focus-within/cue:opacity-100"),
      ).toBe(true);
    }
  });

  it("stays in the tab order while it is invisible", async () => {
    renderSection();
    const buttons = await captureButtons();

    for (const button of buttons) {
      expect(button.classList.contains("opacity-0")).toBe(true);
      // Measured rather than inferred from class names. `hidden` and
      // `invisible` would drop the button out of the tab order and
      // `group-focus-within` would then have nothing to fire on — but so
      // would `inert`, `disabled`, an inline style, or a hidden
      // ancestor, and a denylist of class names sees none of those.
      // jsdom implements focus, so ask it.
      button.focus();
      expect(document.activeElement).toBe(button);
    }
  });

  it("grows its hit area, not its box, where there is no hover to give", async () => {
    renderSection();
    const buttons = await captureButtons();

    for (const button of buttons) {
      const classes = button.classList;
      expect(classes.contains("pointer-coarse:opacity-100")).toBe(true);
      // The target grows from a pseudo-element overhanging the box by
      // 6px a side rather than from a bigger box, so the icon stays the
      // same size at every pointer type. The row is what grows (see the
      // test below), which is also what keeps this pseudo-element from
      // overlapping its neighbour's.
      expect(classes.contains("relative")).toBe(true);
      expect(classes.contains("pointer-coarse:before:absolute")).toBe(true);
      expect(classes.contains("pointer-coarse:before:-inset-1.5")).toBe(true);
      // The box itself stays 32px at every pointer type, which already
      // clears the 24px floor for repeated icon-only controls (hako
      // Prwd_iaXmCjWfY24KjFz2).
      expect(classes.contains("h-8")).toBe(true);
      expect(classes.contains("w-8")).toBe(true);
      expect(classes.contains("pointer-coarse:h-11")).toBe(false);
      expect(classes.contains("pointer-coarse:w-11")).toBe(false);
    }
  });

  it("takes the 44px floor on the row, and only where it is a rule", async () => {
    renderSection();
    const buttons = await captureButtons();

    for (const button of buttons) {
      const row = button.parentElement!;
      // The floor lives in the mobile sizing rules, so it is about
      // touch. `pointer-coarse` is that condition; a plain `min-h-11`
      // would add 37% of height to a transcript of several hundred
      // lines on a desktop the rule was not written about, where 32px
      // already clears the 24px minimum for repeated icon-only controls
      // (hako Prwd_iaXmCjWfY24KjFz2).
      expect(row.classList.contains("pointer-coarse:min-h-11")).toBe(true);
      expect(row.classList.contains("min-h-11")).toBe(false);

      // The seek button takes it too. It is the row's *primary* action,
      // and `items-start` means it does not inherit the row's height —
      // a list whose secondary control clears the floor while its main
      // one does not has bought nothing.
      const seek = row.querySelector("button[aria-current], button:first-child");
      expect(seek).not.toBeNull();
      expect(seek!.classList.contains("pointer-coarse:min-h-11")).toBe(true);
    }
  });
});

describe("TranscriptSection — telling the host whether there is anything", () => {
  beforeEach(() => {
    mockAddonStatus.features.transcript_refine = "manual";
    fetchMock.mockClear();
    clearTranscriptScroll();
  });

  afterEach(() => {
    cleanup();
  });

  it("says no before it knows, because silence means yes to the host", async () => {
    // The host draws a tab per entry and assumes one has something
    // unless told otherwise — that is what keeps entries written before
    // this signal working. So the first thing to say is "no": waiting
    // until there is something to report would leave the empty tab
    // exactly where it was.
    const onAvailability = vi.fn();
    render(
      <TranscriptSection fileId="abc" drive="family" onAvailability={onAvailability} />,
    );

    expect(onAvailability).toHaveBeenCalledWith(false);
    expect(onAvailability.mock.calls[0]).toEqual([false]);

    // The assertion above is about the first thing said, which is why it
    // runs before anything settles. The fetch is still in flight though,
    // and letting it land after the test ends put its `setState` outside
    // `act` — the same shape as a test that releases a promise on its last
    // line and returns.
    await act(async () => {});
  });

  it("says yes once the cues arrive", async () => {
    const onAvailability = vi.fn();
    render(
      <TranscriptSection fileId="abc" drive="family" onAvailability={onAvailability} />,
    );

    await screen.findByText("未修正の文章。");
    await waitFor(() => expect(onAvailability).toHaveBeenLastCalledWith(true));
  });

  it("stays at no for a file nobody has transcribed", async () => {
    // The whole point. This file renders nothing, and before the signal
    // existed it still grew a tab that opened on an empty panel.
    const getFileTranscript = await transcriptApiMock();
    getFileTranscript.mockResolvedValue({
      available: false,
      file_id: "abc",
      drive: "family",
      language: "",
      chunks: [],
    });
    const onAvailability = vi.fn();
    const { container } = render(
      <TranscriptSection fileId="abc" drive="family" onAvailability={onAvailability} />,
    );

    await waitFor(() => expect(getFileTranscript).toHaveBeenCalled());
    // Waiting on the fetch is not enough — the answer is derived a
    // render later. Wait for the thing being asserted.
    await waitFor(() => expect(container).toBeEmptyDOMElement());
    // Both halves: it said so, and it never said otherwise. `every` alone
    // is vacuously true on a callback that was never called at all.
    expect(onAvailability).toHaveBeenCalledWith(false);
    expect(onAvailability.mock.calls.every(([v]) => v === false)).toBe(true);
  });

  it("does not re-answer when the host hands it a new closure", async () => {
    // A host writing `onAvailability={(v) => setX(v)}` inline gives this
    // a new function on every render, and this component re-renders on
    // every clock tick. Re-firing on the prop would be a state write per
    // tick, four times a second for the length of the video.
    const onAvailability = vi.fn();
    const { rerender } = render(
      <TranscriptSection fileId="abc" drive="family" onAvailability={onAvailability} />,
    );
    await screen.findByText("未修正の文章。");
    await waitFor(() => expect(onAvailability).toHaveBeenLastCalledWith(true));
    const before = onAvailability.mock.calls.length;

    rerender(
      <TranscriptSection
        fileId="abc"
        drive="family"
        onAvailability={(v: boolean) => onAvailability(v)}
      />,
    );

    expect(onAvailability.mock.calls.length).toBe(before);
  });
});

describe("TranscriptSection — whose name is on the panel", () => {
  beforeEach(() => {
    mockAddonStatus.features.transcript_refine = "manual";
    fetchMock.mockClear();
    clearTranscriptScroll();
  });

  afterEach(() => {
    cleanup();
  });

  it("writes its own title where nothing else does", async () => {
    // The box below the player has no heading, so this is the only
    // thing saying what is in it.
    render(<TranscriptSection fileId="abc" drive="family" />);

    expect(await screen.findByText("Transcript")).toBeInTheDocument();
  });

  it("drops it when the host has already written it", async () => {
    // In the tab strip the button the reader just pressed says it.
    render(<TranscriptSection fileId="abc" drive="family" labelledByHost />);

    await screen.findByText("未修正の文章。");
    expect(screen.queryByText("Transcript")).toBeNull();
  });

  it("keeps the facts about the transcript either way", async () => {
    // Only the name goes. The language and the count are facts about
    // this transcript, not a second name for it.
    render(<TranscriptSection fileId="abc" drive="family" labelledByHost />);

    await screen.findByText("未修正の文章。");
    expect(screen.getByText("ja")).toBeInTheDocument();
    expect(screen.getByText("(2)")).toBeInTheDocument();
  });

  it("keeps the controls too, which is the half that would go quietly", async () => {
    // The source toggle and the refine button are the reason the row
    // survives at all, and both are conditional already — a rule that
    // also hid them under `labelledByHost` would take them out of the
    // inspector tab, which is now the placement most readers see, with
    // nothing to say so. The toggle needs two available sources to
    // render, so the word-level fetch has to answer.
    fetchMock.mockResolvedValue({
      ok: true,
      status: 200,
      text: async () =>
        "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nfirst\n\n00:00:02.000 --> 00:00:04.000\nsecond\n",
      json: async () => null,
    } as Response);

    render(<TranscriptSection fileId="abc" drive="family" labelledByHost />);

    expect(await screen.findByRole("button", { name: "Words" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Text chunks" })).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Clean up with AI" }),
    ).toBeInTheDocument();
  });
});

const vttResponse = (text: string) =>
  ({ ok: true, status: 200, text: async () => text, json: async () => null }) as Response;
const vttOf = (cues: Array<[number, number, string]>) =>
  "WEBVTT\n\n" +
  cues
    .map(([from, to, text]) => {
      const t = (n: number) => `00:00:${n.toFixed(3).padStart(6, "0")}`;
      return `${t(from)} --> ${t(to)}\n${text}\n`;
    })
    .join("\n");

describe("TranscriptSection — where the reader had got to", () => {
  beforeEach(async () => {
    mockAddonStatus.features.transcript_refine = "manual";
    fetchMock.mockClear();
    clearTranscriptScroll();
  });

  afterEach(() => {
    cleanup();
  });

  function scrollStubController(state: { currentTime: number }) {
    return {
      seek: vi.fn(),
      play: vi.fn(),
      pause: vi.fn(),
      togglePlay: vi.fn(),
      toggleMute: vi.fn(),
      toggleFullscreen: vi.fn(),
      getCurrentTime: () => state.currentTime,
      getDuration: () => 10,
      isPaused: () => false,
      isMuted: () => false,
      getVolume: () => 1,
      setVolume: vi.fn(),
      getPlaybackRate: () => 1,
      setPlaybackRate: vi.fn(),
      getBufferedFraction: () => 0,
    };
  }

  const CHIP = "Back to current position";

  /**
   * Give the word-level source something to hold.
   *
   * Two available sources is what puts the source toggle on screen, and
   * the toggle is the only thing in this panel that changes the cue
   * count after it has settled.
   */
  function withWordCues() {
    fetchMock.mockResolvedValue({
      ok: true,
      status: 200,
      // Three, where the chunk source has two. The restore effect is
      // keyed on whether there is a list, not on how long it is — so
      // this asymmetry is what makes a count-keyed regression visible.
      // Two sources of equal length would not re-run even the broken
      // form, and the switch would prove nothing either way.
      text: async () =>
        "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nfirst\n\n00:00:02.000 --> 00:00:04.000\nsecond\n\n00:00:04.000 --> 00:00:06.000\nthird\n",
      json: async () => null,
    } as Response);
  }

  /**
   * Mount, scroll somewhere, unmount, mount again.
   *
   * That is the phone's bottom sheet collapsing and being raised: core
   * mounts the drawer only while it is expanded, because vaul renders a
   * modal Radix dialog and one left mounted at rest puts `aria-hidden`
   * on the whole application. Everything else this panel holds comes
   * back from the refetch; the offset does not.
   */
  /**
   * Wait for the cues *and* for the effects that follow them.
   *
   * `findByText` resolves as soon as the text is in the DOM, which is
   * the end of the commit — and passive effects run after that, on their
   * own schedule. The effect that attaches the save listener is one of
   * them, so a test that scrolls the moment the text appears sometimes
   * scrolls a list nothing is listening to. It failed about one run in
   * thirty, and only in CI's shuffled job often enough to see.
   *
   * This is the third detector rule applied to a test's own setup: wait
   * for what you are about to depend on, not for the thing that starts
   * it.
   */
  /**
   * jsdom lays nothing out. Rows 100px high from the list's top, moved by
   * its `scrollTop`; a hidden panel has no box.
   */
  let ROW_PX = 100;
  beforeEach(() => {
    ROW_PX = 100;
    vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockImplementation(
      function (this: HTMLElement) {
        const zero = { top: 0, bottom: 0, height: 0 } as DOMRect;
        if (this.closest("[hidden]")) return zero;
        if (this.dataset.cueStart === undefined) {
          return { top: 0, bottom: 300, height: 300 } as DOMRect;
        }
        const list = this.parentElement!;
        const index = Array.from(list.querySelectorAll("[data-cue-start]")).indexOf(this);
        const top = index * ROW_PX - list.scrollTop;
        return { top, bottom: top + ROW_PX, height: ROW_PX } as DOMRect;
      },
    );
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  async function readyList(container: HTMLElement): Promise<HTMLElement> {
    await screen.findByText("未修正の文章。");
    await act(async () => {});
    return container.querySelector(".overflow-y-auto")! as HTMLElement;
  }

  async function mountAndScroll(fileId: string, top: number) {
    const utils = render(<TranscriptSection fileId={fileId} drive="family" />);
    const list = await readyList(utils.container);
    list.scrollTop = top;
    fireEvent.scroll(list);
    return { utils, list };
  }

  async function remount(fileId: string) {
    const utils = render(<TranscriptSection fileId={fileId} drive="family" />);
    return readyList(utils.container);
  }

  it("puts the reader back where they were", async () => {
    const { utils } = await mountAndScroll("abc", 150);
    utils.unmount();

    expect((await remount("abc")).scrollTop).toBe(150);
  });

  it("puts the reader back on the same row when rows are another height", async () => {
    // Another width, or another source: the offset differs, the row does not.
    const { utils } = await mountAndScroll("abc", 150);
    utils.unmount();
    ROW_PX = 60;

    // 50px into the second row, which now starts 60px down.
    expect((await remount("abc")).scrollTop).toBe(110);
  });

  it("keeps the reader on the same row when a source arriving later replaces the one shown", async () => {
    // No text chunks: subtitles answer first and are shown, then the word
    // cues arrive and take over, with rows that start at other times.
    const { utils } = await mountAndScroll("abc", 150);
    utils.unmount();

    const getFileTranscript = await transcriptApiMock();
    getFileTranscript.mockResolvedValue({ available: false });
    let releaseWords: (r: Response) => void = () => undefined;
    const vtt = (text: string) =>
      ({ ok: true, status: 200, text: async () => text, json: async () => null }) as Response;
    fetchMock.mockImplementation((url: string) =>
      String(url).includes("subtitles.vtt")
        ? new Promise<Response>((resolve) => {
            releaseWords = resolve;
          })
        : Promise.resolve(
            vtt(
              "WEBVTT\n\n00:00:00.000 --> 00:00:02.500\nsub one\n\n00:00:02.500 --> 00:00:05.000\nsub two\n\n00:00:05.000 --> 00:00:07.500\nsub three\n\n00:00:07.500 --> 00:00:10.000\nsub four\n",
            ),
          ),
    );
    const again = render(
      <TranscriptSection
        fileId="abc"
        drive="family"
        subtitles={[{ index: 0, language: "en", format: "vtt", label: "English" }]}
      />,
    );
    await screen.findByText("sub three");
    await act(async () => {});
    const list = again.container.querySelector(".overflow-y-auto") as HTMLElement;
    // 50px into "sub three", the row that starts at 5s.
    expect(list.scrollTop).toBe(250);
    // The reader moves on to 30px into "sub four" (7.5s). A browser
    // delivers the scroll event a frame later, which may be after the
    // words have arrived.
    list.scrollTop = 330;

    await act(async () =>
      releaseWords(
        vtt(
          "WEBVTT\n\n00:00:00.000 --> 00:00:05.000\nword one\n\n00:00:05.000 --> 00:00:10.000\nword two\n",
        ),
      ),
    );
    await screen.findByText("word two");
    await act(async () => {});

    // 30px into "word two", which covers 7.5s.
    expect(list.scrollTop).toBe(130);
  });

  it("keeps the reader's row when the rows are replaced while following a cue", async () => {
    // Following, with the playhead in a pause the incoming word cues do not
    // cover, so there is no playing cue to go to on the new rows.
    const getFileTranscript = await transcriptApiMock();
    getFileTranscript.mockResolvedValue({ available: false });
    let releaseWords: (r: Response) => void = () => undefined;
    fetchMock.mockImplementation((url: string) =>
      String(url).includes("subtitles.vtt")
        ? new Promise<Response>((resolve) => {
            releaseWords = resolve;
          })
        : Promise.resolve(
            vttResponse(
              vttOf([0, 2, 4, 6, 8, 10].map((t, i) => [t, t + 2, `s${i}`])),
            ),
          ),
    );
    const state = { currentTime: 8.5 };
    const utils = render(
      <TranscriptSection
        fileId="abc"
        drive="family"
        mediaController={scrollStubController(state)}
        subtitles={[{ index: 0, language: "en", format: "vtt", label: "English" }]}
      />,
    );
    await screen.findByText("s4");
    await waitForActiveCue(utils.container);
    const list = utils.container.querySelector(".overflow-y-auto") as HTMLElement;
    // 50px into s3, which starts at 6s.
    list.scrollTop = 350;
    fireEvent.scroll(list);

    await act(async () =>
      releaseWords(
        vttResponse(
          vttOf([
            ...[0, 1, 2, 3, 4, 5, 6, 7].map((t): [number, number, string] => [t, t + 1, `w${t}`]),
            [9, 10, "w9"],
          ]),
        ),
      ),
    );
    await screen.findByText("w9");
    await act(async () => {});

    // 50px into w6, the word cue that starts at 6s.
    expect(list.scrollTop).toBe(650);
  });

  it("keeps the row when it is now shorter than how far into it the reader was", async () => {
    const { utils } = await mountAndScroll("abc", 180);
    utils.unmount();
    ROW_PX = 60;

    // As far into the second row as it goes, not onto a row after it.
    expect((await remount("abc")).scrollTop).toBe(60 + 59);
  });

  it("puts the reader on the first row when their place came before every row", async () => {
    rememberTranscriptScroll("abc", { place: { at: -3, into: 20 }, following: true });
    expect((await remount("abc")).scrollTop).toBe(20);
  });

  it("puts the reader on the first of two rows that start together", async () => {
    fetchMock.mockImplementation(() =>
      Promise.resolve(
        vttResponse(
          vttOf([
            [0, 5, "sign"],
            [5, 8, "speaker one"],
            [5, 8, "speaker two"],
            [8, 10, "after"],
          ]),
        ),
      ),
    );
    const getFileTranscript = await transcriptApiMock();
    getFileTranscript.mockResolvedValue({ available: false });
    const subtitles = [{ index: 0, language: "en", format: "vtt", label: "English" }];
    const first = render(<TranscriptSection fileId="abc" drive="family" subtitles={subtitles} />);
    await screen.findByText("speaker two");
    await act(async () => {});
    const list = first.container.querySelector(".overflow-y-auto") as HTMLElement;
    // 30px into "speaker one".
    list.scrollTop = 130;
    fireEvent.scroll(list);
    first.unmount();

    const again = render(<TranscriptSection fileId="abc" drive="family" subtitles={subtitles} />);
    await screen.findByText("speaker two");
    await act(async () => {});
    expect(
      (again.container.querySelector(".overflow-y-auto") as HTMLElement).scrollTop,
    ).toBe(130);
  });

  it("waits for the source the reader chose before putting them back on it", async () => {
    rememberTranscriptScroll("abc", {
      place: { at: 7, into: 30 },
      following: false,
      source: "external",
    });
    let releaseSubtitles: (r: Response) => void = () => undefined;
    fetchMock.mockImplementation((url: string) =>
      String(url).includes("subtitles.vtt")
        ? Promise.resolve(FETCH_MISS as Response)
        : new Promise<Response>((resolve) => {
            releaseSubtitles = resolve;
          }),
    );
    const subtitles = [{ index: 0, language: "en", format: "vtt", label: "English" }];
    const utils = render(<TranscriptSection fileId="abc" drive="family" subtitles={subtitles} />);
    await screen.findByText("未修正の文章。");
    await act(async () => {});

    await act(async () =>
      releaseSubtitles(
        vttResponse(vttOf([0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map((t) => [t, t + 1, `sub${t}`]))),
      ),
    );
    await screen.findByText("sub7");
    await act(async () => {});

    // 30px into the subtitle that starts at 7s.
    expect(
      (utils.container.querySelector(".overflow-y-auto") as HTMLElement).scrollTop,
    ).toBe(730);
  });

  describe("while the source the reader chose has not answered", () => {
    const subtitles = [{ index: 0, language: "en", format: "vtt", label: "English" }];
    let answerSubtitles: (r: Response) => void = () => undefined;

    beforeEach(() => {
      rememberTranscriptScroll("abc", {
        place: { at: 7, into: 30 },
        following: false,
        source: "external",
      });
      fetchMock.mockImplementation((url: string) =>
        String(url).includes("subtitles.vtt")
          ? Promise.resolve(FETCH_MISS as Response)
          : new Promise<Response>((resolve) => {
              answerSubtitles = resolve;
            }),
      );
    });

    async function mounted() {
      const utils = render(
        <TranscriptSection fileId="abc" drive="family" subtitles={subtitles} />,
      );
      await screen.findByText("未修正の文章。");
      await act(async () => {});
      return utils.container.querySelector(".overflow-y-auto") as HTMLElement;
    }

    it("lets the reader's own scrolling stand when it arrives", async () => {
      const list = await mounted();
      fireEvent.wheel(list);
      list.scrollTop = 150;
      fireEvent.scroll(list);

      await act(async () =>
        answerSubtitles(
          vttResponse(vttOf([0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map((t) => [t, t + 1, `sub${t}`]))),
        ),
      );
      await screen.findByText("sub7");
      await act(async () => {});

      // 50px into the text chunk at 5s is where the reader was; laid onto
      // the subtitles, that is 50px into the subtitle at 5s.
      expect(list.scrollTop).toBe(550);
    });

    it("puts the reader back on what is shown once it answers with nothing", async () => {
      const list = await mounted();
      expect(list.scrollTop).toBe(0);

      await act(async () => answerSubtitles(FETCH_MISS as Response));
      await act(async () => {});

      // 30px into the text chunk at 5s, the latest start before 7s.
      expect(list.scrollTop).toBe(130);
    });
  });

  it("puts the reader back when the file has no subtitles to wait for", async () => {
    rememberTranscriptScroll("abc", {
      place: { at: 7, into: 30 },
      following: false,
      source: "external",
    });
    const utils = render(<TranscriptSection fileId="abc" drive="family" />);
    await screen.findByText("未修正の文章。");
    await act(async () => {});
    expect(
      (utils.container.querySelector(".overflow-y-auto") as HTMLElement).scrollTop,
    ).toBe(130);
  });

  it("puts the reader on the last of the rows starting together when fewer are left", async () => {
    fetchMock.mockImplementation(() =>
      Promise.resolve(
        vttResponse(
          vttOf([
            [0, 5, "sign"],
            [5, 8, "speaker one"],
            [5, 8, "speaker two"],
            [8, 10, "after"],
          ]),
        ),
      ),
    );
    const getFileTranscript = await transcriptApiMock();
    getFileTranscript.mockResolvedValue({ available: false });
    rememberTranscriptScroll("abc", { place: { at: 5, into: 30, nth: 4 }, following: false });
    const utils = render(
      <TranscriptSection
        fileId="abc"
        drive="family"
        subtitles={[{ index: 0, language: "en", format: "vtt", label: "English" }]}
      />,
    );
    await screen.findByText("speaker two");
    await act(async () => {});
    expect(
      (utils.container.querySelector(".overflow-y-auto") as HTMLElement).scrollTop,
    ).toBe(230);
  });

  it("shows what there is when the source the reader chose is not there for this file", async () => {
    rememberTranscriptScroll("abc", {
      place: { at: 5, into: 20 },
      following: false,
      source: "words",
    });
    const utils = render(<TranscriptSection fileId="abc" drive="family" />);
    expect(await screen.findByText("未修正の文章。")).toBeInTheDocument();
    await act(async () => {});
    // And the reader is put back on it rather than kept waiting.
    expect(
      (utils.container.querySelector(".overflow-y-auto") as HTMLElement).scrollTop,
    ).toBe(120);
  });

  it("puts a reader on the second of two rows that start together back on the second", async () => {
    fetchMock.mockImplementation(() =>
      Promise.resolve(
        vttResponse(
          vttOf([
            [0, 5, "sign"],
            [5, 8, "speaker one"],
            [5, 8, "speaker two"],
            [8, 10, "after"],
          ]),
        ),
      ),
    );
    const getFileTranscript = await transcriptApiMock();
    getFileTranscript.mockResolvedValue({ available: false });
    const subtitles = [{ index: 0, language: "en", format: "vtt", label: "English" }];
    const first = render(<TranscriptSection fileId="abc" drive="family" subtitles={subtitles} />);
    await screen.findByText("speaker two");
    await act(async () => {});
    const list = first.container.querySelector(".overflow-y-auto") as HTMLElement;
    // 30px into "speaker two".
    list.scrollTop = 230;
    fireEvent.scroll(list);
    first.unmount();

    const again = render(<TranscriptSection fileId="abc" drive="family" subtitles={subtitles} />);
    await screen.findByText("speaker two");
    await act(async () => {});
    expect(
      (again.container.querySelector(".overflow-y-auto") as HTMLElement).scrollTop,
    ).toBe(230);
  });

  it("comes back on the source the reader chose, and does not take it to another file", async () => {
    fetchMock.mockImplementation((url: string) =>
      Promise.resolve(
        String(url).includes("subtitles.vtt")
          ? FETCH_MISS
          : vttResponse(vttOf([[0, 5, "subtitle one"], [5, 10, "subtitle two"]])),
      ),
    );
    const subtitles = [{ index: 0, language: "en", format: "vtt", label: "English" }];
    const first = render(<TranscriptSection fileId="abc" drive="family" subtitles={subtitles} />);
    await screen.findByText("未修正の文章。");
    fireEvent.click(await screen.findByRole("button", { name: "External" }));
    await screen.findByText("subtitle two");
    first.unmount();

    const again = render(<TranscriptSection fileId="abc" drive="family" subtitles={subtitles} />);
    expect(await screen.findByText("subtitle two")).toBeInTheDocument();

    again.rerender(<TranscriptSection fileId="def" drive="family" subtitles={subtitles} />);
    expect(await screen.findByText("未修正の文章。")).toBeInTheDocument();
    expect(screen.queryByText("subtitle two")).toBeNull();
  });

  it("shows the text chunks, not subtitles that merely answered first", async () => {
    let release: (value: typeof TRANSCRIPT_RESPONSE) => void = () => undefined;
    const getFileTranscript = await transcriptApiMock();
    getFileTranscript.mockReturnValue(
      new Promise((resolve) => {
        release = resolve;
      }),
    );
    fetchMock.mockResolvedValue({
      ok: true,
      status: 200,
      text: async () =>
        "WEBVTT\n\n00:00:00.000 --> 00:00:05.000\nsubtitle one\n\n00:00:05.000 --> 00:00:10.000\nsubtitle two\n",
      json: async () => null,
    } as Response);
    render(
      <TranscriptSection
        fileId="abc"
        drive="family"
        subtitles={[{ index: 0, language: "en", format: "vtt", label: "English" }]}
      />,
    );
    await act(() => new Promise((resolve) => setTimeout(resolve, 50)));
    await act(async () => release(TRANSCRIPT_RESPONSE));

    expect(await screen.findByText("未修正の文章。")).toBeInTheDocument();
    expect(screen.queryByText("subtitle two")).toBeNull();

    // Once the reader picks one, it stays theirs.
    fireEvent.click(screen.getByRole("button", { name: "External" }));
    expect(await screen.findByText("subtitle two")).toBeInTheDocument();
  });

  it("keeps each file's place to itself", async () => {
    // Keyed by file, so opening a second one and coming back does not
    // land the reader at someone else's offset.
    const { utils } = await mountAndScroll("abc", 150);
    utils.unmount();

    const other = await remount("def");
    expect(other.scrollTop).toBe(0);
  });

  it("restores having taken over, not just the offset", async () => {
    // Restoring the offset alone on a playing file hands the reader back
    // their place and then, a second later, drags them to the cue that
    // is playing — which is the state they left precisely by scrolling
    // away from it. The chip is the component saying it has stopped
    // following, and it is the only thing that shows the difference:
    // the offset is identical either way at the moment of the restore.
    const state = { currentTime: 1 };
    const withPlayer = () =>
      render(
        <TranscriptSection
          fileId="abc"
          drive="family"
          mediaController={scrollStubController(state)}
        />,
      );
    const utils = withPlayer();
    const list = await readyList(utils.container);
    list.scrollTop = 130;
    fireEvent.wheel(list);
    fireEvent.scroll(list);
    expect(await screen.findByRole("button", { name: CHIP })).toBeInTheDocument();
    utils.unmount();

    const back = withPlayer();
    const list2 = await readyList(back.container);
    expect(list2.scrollTop).toBe(130);
    // Still suspended, so the auto-scroll will not take the offset back
    // off them the moment playback moves on.
    expect(await screen.findByRole("button", { name: CHIP })).toBeInTheDocument();
  });

  it("hands the position back to the reader once restored", async () => {
    // Restoring happens when there is a list to restore into, not every
    // time the list changes length. Switching source changes the length,
    // and re-restoring there would pin the reader to an offset measured
    // against the list they just left.
    //
    // The move here is deliberately not a `scroll` event: switching
    // source replaces the list's contents, and the browser adjusting the
    // offset for that is not the reader scrolling. It is also what makes
    // the failure visible — a saved 10 would be restored as 10, and the
    // two would be indistinguishable.
    withWordCues();
    const { utils } = await mountAndScroll("abc", 150);
    utils.unmount();
    const list = await remount("abc");
    expect(list.scrollTop).toBe(150);
    list.scrollTop = 10;

    fireEvent.click(await screen.findByRole("button", { name: "Words" }));

    expect(list.scrollTop).toBe(10);
  });

  it("writes the position down as the reader moves, not only on the way out", async () => {
    // In a browser the unmount save is the weaker of the two: `useEffect`
    // cleanups are passive, so they run after React has detached the
    // subtree, and `scrollTop` on a detached element reads 0. jsdom keeps
    // the value, so no test can show that directly — what a test can show
    // is that the position is already written down while the panel is
    // still on screen, which is the property that makes the unmount
    // reading unnecessary.
    const utils = render(<TranscriptSection fileId="abc" drive="family" />);
    const list = await readyList(utils.container);

    list.scrollTop = 120;
    fireEvent.scroll(list);

    expect(recallTranscriptScroll("abc")).toEqual({
      place: { at: 5, into: 20 },
      following: true,
    });
  });

  it("catches a change of mind that moved nothing", async () => {
    // `following` can change with no scroll of the reader's — clicking a
    // cue resumes it — and there is no event for that. Without the save
    // in the cleanup the store would still say they had taken over, and
    // the next mount would restore them into a transcript that has
    // stopped following the playhead they just jumped to.
    const state = { currentTime: 1 };
    const utils = render(
      <TranscriptSection
        fileId="abc"
        drive="family"
        mediaController={scrollStubController(state)}
      />,
    );
    const list = await readyList(utils.container);
    list.scrollTop = 120;
    fireEvent.wheel(list);
    fireEvent.scroll(list);
    expect(recallTranscriptScroll("abc")).toEqual({
      place: { at: 5, into: 20 },
      following: false,
    });

    // Jumping to a cue is a statement about where they want to be, so it
    // resumes following — and moves no scrollbar in jsdom.
    fireEvent.click(screen.getByText("未修正の文章。"));
    utils.unmount();

    expect(recallTranscriptScroll("abc")?.following).toBe(true);
  });

  it("forgets the least recently written, not the oldest ever", async () => {
    // Twenty-one files each written once cannot tell the two apart —
    // insertion order and write order are the same list. Re-writing an
    // early one is what separates them, and it is the case that matters:
    // the file a reader keeps coming back to is the one that must not be
    // evicted for a file they opened once an hour ago.
    for (let i = 0; i < 20; i += 1) {
      const { utils } = await mountAndScroll(`f${i}`, 100 + i);
      utils.unmount();
    }
    // Touch the oldest again, then push one more in.
    const { utils: revisit } = await mountAndScroll("f0", 199);
    revisit.unmount();
    const { utils: last } = await mountAndScroll("f20", 120);
    last.unmount();

    const at = (into: number) => ({ place: { at: 5, into }, following: true });
    expect(recallTranscriptScroll("f0")).toEqual(at(99));
    expect(recallTranscriptScroll("f1")).toBeUndefined();
    expect(recallTranscriptScroll("f20")).toEqual(at(20));
  });

  it("forgets the oldest file rather than growing without limit", async () => {
    // Module state nothing ever clears. A tab left open for a week
    // browsing a large drive would otherwise keep an entry per file.
    for (let i = 0; i < 21; i += 1) {
      const { utils } = await mountAndScroll(`f${i}`, 100 + i);
      utils.unmount();
    }

    const at = (into: number) => ({ place: { at: 5, into }, following: true });
    expect(recallTranscriptScroll("f0")).toBeUndefined();
    expect(recallTranscriptScroll("f1")).toEqual(at(1));
    expect(recallTranscriptScroll("f20")).toEqual(at(20));
  });

  it("keeps the place when the list can no longer be read on the way out", async () => {
    // A browser detaches the list before passive cleanups run, and a
    // detached element reads a `scrollTop` of 0.
    const { utils, list } = await mountAndScroll("abc", 150);
    let top = list.scrollTop;
    Object.defineProperty(list, "scrollTop", {
      configurable: true,
      get: () => (list.isConnected ? top : 0),
      set: (value: number) => {
        top = value;
      },
    });
    utils.unmount();

    expect(recallTranscriptScroll("abc")?.place).toEqual({ at: 5, into: 50 });
  });

  it("puts the reader back even when the list appears after its cues", async () => {
    const { utils } = await mountAndScroll("abc", 150);
    utils.unmount();

    // Word cues of the same count arrive while the transcript loads, so
    // the list mounts after there were already cues.
    let release: (value: typeof TRANSCRIPT_RESPONSE) => void = () => undefined;
    const getFileTranscript = await transcriptApiMock();
    getFileTranscript.mockReturnValue(
      new Promise((resolve) => {
        release = resolve;
      }),
    );
    fetchMock.mockResolvedValue({
      ok: true,
      status: 200,
      text: async () =>
        // Rows that start elsewhere than the text chunks, so a place laid on
        // them before the chunks replace them lands on another offset.
        "WEBVTT\n\n00:00:00.000 --> 00:00:02.500\nfirst\n\n00:00:02.500 --> 00:00:05.000\nsecond\n\n00:00:05.000 --> 00:00:07.500\nthird\n\n00:00:07.500 --> 00:00:10.000\nfourth\n",
      json: async () => null,
    } as Response);
    const again = render(<TranscriptSection fileId="abc" drive="family" />);
    await act(() => new Promise((resolve) => setTimeout(resolve, 50)));
    await act(async () => release(TRANSCRIPT_RESPONSE));
    // The text chunks, which replace the word cues nobody chose.
    await screen.findByText("未修正の文章。");
    await act(async () => {});

    expect(
      (again.container.querySelector(".overflow-y-auto") as HTMLElement).scrollTop,
    ).toBe(150);
  });

  it("does not carry having taken over into a file with nothing remembered", async () => {
    const state = { currentTime: 1 };
    const mc = scrollStubController(state);
    const utils = render(
      <TranscriptSection fileId="abc" drive="family" mediaController={mc} />,
    );
    const list = await readyList(utils.container);
    await waitForActiveCue(utils.container);
    fireEvent.wheel(list);
    await screen.findByRole("button", { name: "Back to current position" });

    utils.rerender(
      <TranscriptSection fileId="def" drive="family" mediaController={mc} />,
    );
    await readyList(utils.container);
    await waitForActiveCue(utils.container);

    expect(
      screen.queryByRole("button", { name: "Back to current position" }),
    ).toBeNull();
  });

  describe("in a host whose scroller encloses the list", () => {
    // The list starts 600px down the host's content.
    const LIST_AT = 600;
    const sizeCallbacks: Array<(entries: { contentRect: { height: number } }[]) => void> = [];
    const original = (globalThis as { ResizeObserver?: unknown }).ResizeObserver;

    beforeEach(() => {
      sizeCallbacks.length = 0;
      (globalThis as { ResizeObserver?: unknown }).ResizeObserver = class {
        constructor(cb: (typeof sizeCallbacks)[number]) {
          sizeCallbacks.push(cb);
        }
        observe() {}
        disconnect() {}
      };
      vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockImplementation(
        function (this: HTMLElement) {
          const host = this.closest<HTMLElement>("[data-inspector-scroller]");
          if (this.closest("[hidden]") || !host) {
            return { top: 0, bottom: 0, height: 0 } as DOMRect;
          }
          if (this === host) return { top: 100, bottom: 400, height: 300 } as DOMRect;
          const listTop = 100 + LIST_AT - host.scrollTop;
          if (this.dataset.cueStart === undefined) {
            return { top: listTop, bottom: listTop + 4000, height: 4000 } as DOMRect;
          }
          const rows = Array.from(host.querySelectorAll("[data-cue-start]"));
          const top = listTop + rows.indexOf(this) * 100;
          return { top, bottom: top + 100, height: 100 } as DOMRect;
        },
      );
    });

    afterEach(() => {
      (globalThis as { ResizeObserver?: unknown }).ResizeObserver = original;
      vi.restoreAllMocks();
    });

    const resize = (height: number) =>
      act(() => {
        for (const cb of sizeCallbacks) cb([{ contentRect: { height } }]);
      });

    async function mountInHost(
      hidden: boolean,
      mediaController?: ReturnType<typeof scrollStubController>,
    ) {
      const utils = render(
        <div data-testid="host" data-inspector-scroller="">
          <div data-testid="panel" hidden={hidden}>
            <TranscriptSection
              fileId="abc"
              drive="family"
              mediaController={mediaController}
            />
          </div>
        </div>,
      );
      await screen.findByText("未修正の文章。");
      await act(async () => {});
      return { utils, host: screen.getByTestId("host") };
    }

    it("remembers the place in the list, not the scroller's own offset", async () => {
      const { host } = await mountInHost(false);
      host.scrollTop = LIST_AT + 150;
      fireEvent.scroll(host);

      expect(recallTranscriptScroll("abc")?.place).toEqual({ at: 5, into: 50 });
    });

    it("waits until the transcript is shown to put the reader back", async () => {
      const first = await mountInHost(false);
      first.host.scrollTop = LIST_AT + 150;
      fireEvent.scroll(first.host);
      first.utils.unmount();

      const { host } = await mountInHost(true);
      await resize(0);
      expect(host.scrollTop).toBe(0);

      screen.getByTestId("panel").hidden = false;
      await resize(4000);
      expect(host.scrollTop).toBe(LIST_AT + 150);
    });

    it("goes to the playing cue, not the last place, when that place was following", async () => {
      rememberTranscriptScroll("abc", { place: { at: 0, into: 0 }, following: true });
      const { host } = await mountInHost(true, scrollStubController({ currentTime: 7 }));
      const hostScrollTo = vi.fn();
      host.scrollTo = hostScrollTo;
      await resize(0);
      await waitFor(() =>
        expect(host.querySelector('[aria-current="true"]')).not.toBeNull(),
      );

      screen.getByTestId("panel").hidden = false;
      await resize(4000);

      expect(hostScrollTo).toHaveBeenCalled();
    });

    it("keeps the place through a mount that was never shown", async () => {
      const first = await mountInHost(false);
      first.host.scrollTop = LIST_AT + 150;
      fireEvent.scroll(first.host);
      first.utils.unmount();
      (await mountInHost(true)).utils.unmount();

      const { host } = await mountInHost(true);
      await resize(0);
      screen.getByTestId("panel").hidden = false;
      await resize(4000);
      expect(host.scrollTop).toBe(LIST_AT + 150);
    });

    it("puts the place back after another tab moved the scroller, without a remount", async () => {
      const { host } = await mountInHost(false);
      await resize(4000);
      host.scrollTop = LIST_AT + 150;
      fireEvent.scroll(host);

      screen.getByTestId("panel").hidden = true;
      await resize(0);
      host.scrollTop = 50;
      fireEvent.scroll(host);
      screen.getByTestId("panel").hidden = false;
      await resize(4000);

      expect(host.scrollTop).toBe(LIST_AT + 150);
    });

    it("does not write down where another tab left the scroller", async () => {
      const { host } = await mountInHost(false);
      host.scrollTop = LIST_AT + 150;
      fireEvent.scroll(host);

      screen.getByTestId("panel").hidden = true;
      host.scrollTop = 50;
      fireEvent.scroll(host);

      expect(recallTranscriptScroll("abc")?.place).toEqual({ at: 5, into: 50 });
    });
  });
});

describe("TranscriptSection — a response that arrives too late", () => {
  beforeEach(() => {
    mockAddonStatus.features.transcript_refine = "manual";
    fetchMock.mockClear();
    clearTranscriptScroll();
  });

  afterEach(() => {
    cleanup();
  });

  /** A promise this test resolves by hand, so "too late" is a moment it picks. */
  function deferred<T>() {
    let settle!: (value: T) => void;
    let reject!: (reason: unknown) => void;
    const promise = new Promise<T>((resolve, rejectFn) => {
      settle = resolve;
      reject = rejectFn;
    });
    // Nothing is listening yet when a test rejects, and an unhandled
    // rejection is reported as an error while every test still passes.
    promise.catch(() => undefined);
    return { promise, settle, reject };
  }

  it("ignores the previous file's transcript", async () => {
    // The host reuses one mount across files — the below-player
    // placement keys its nodes on the entry id, not on the file — so a
    // request started for one file can land while another is on screen.
    // Two things go wrong at once if it is not abandoned: one file's
    // cues render under another file's player, and `hasAnything` is
    // derived from the same state, so the host is told the new file has
    // a transcript because the old one did — which is a Transcript tab
    // on an untranscribed video, the defect this all exists to remove.
    const slow = deferred<unknown>();
    const getFileTranscript = await transcriptApiMock();
    getFileTranscript.mockReset();
    getFileTranscript
      .mockReturnValueOnce(slow.promise)
      .mockResolvedValue({
        available: false,
        file_id: "def",
        drive: "family",
        language: "",
        chunks: [],
      });

    const onAvailability = vi.fn();
    const { rerender, container } = render(
      <TranscriptSection fileId="abc" drive="family" onAvailability={onAvailability} />,
    );
    rerender(
      <TranscriptSection fileId="def" drive="family" onAvailability={onAvailability} />,
    );
    await waitFor(() =>
      expect(getFileTranscript).toHaveBeenCalledTimes(2),
    );

    slow.settle(TRANSCRIPT_RESPONSE);
    await slow.promise;

    await waitFor(() => expect(container).toBeEmptyDOMElement());
    expect(screen.queryByText("未修正の文章。")).toBeNull();
    expect(onAvailability).toHaveBeenLastCalledWith(false);
  });

  it("ignores the previous file's word-level cues", async () => {
    // Same shape, the other fetch. This one has no `loading` flag of its
    // own, so nothing else would notice.
    //
    // Asserted on the source toggle rather than on the cues' text, and
    // on a file that does render. An absence is the hardest thing to
    // wait for — the first poll of a `waitFor` succeeds before a stale
    // response has finished travelling through two `.then` links, so it
    // passes whether the guard is there or not. The toggle is a positive
    // signal: it appears only when two sources are available, so a stale
    // word list that landed shows up as a control that should not exist.
    const slow = deferred<Response>();
    fetchMock.mockReturnValueOnce(slow.promise).mockResolvedValue(FETCH_MISS);

    const { rerender } = render(<TranscriptSection fileId="abc" drive="family" />);
    rerender(<TranscriptSection fileId="def" drive="family" />);
    await screen.findByText("未修正の文章。");
    expect(screen.queryByRole("button", { name: "Words" })).toBeNull();

    await act(async () => {
      slow.settle({
        ok: true,
        status: 200,
        text: async () =>
          "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nstale word cue\n",
        json: async () => null,
      } as Response);
      // Drain the `.then` chain the component built on it, inside `act`,
      // so any state it would set has been committed by the time the
      // assertions below run.
      await slow.promise;
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(screen.queryByRole("button", { name: "Words" })).toBeNull();
    expect(screen.queryByText("stale word cue")).toBeNull();
  });
  it("ignores the previous file's external subtitles", async () => {
    // The third fetch, and the one with the least around it: no loading
    // flag, and its cues are only reachable through the source toggle.
    //
    // Both files carry subtitles, which is what makes the guard
    // observable at all: `externalAvailable` is also gated on the
    // *current* `subtitles` prop, so a stale list landing on a file with
    // none is invisible whether it was abandoned or not. With both
    // carrying them, only the second file's own (empty) answer should
    // count, and a stale list shows up as an "External" source that has
    // nothing behind it.
    const SUBS_A = [
      { index: 0, language: "en", format: "vtt", label: "English" },
    ];
    const SUBS_B = [
      { index: 0, language: "fr", format: "vtt", label: "French" },
    ];
    const slow = deferred<Response>();
    fetchMock.mockImplementation((url: string) =>
      !String(url).includes("subtitles.vtt") && String(url).includes("/abc/")
        ? slow.promise
        : Promise.resolve(FETCH_MISS),
    );

    const { rerender } = render(
      <TranscriptSection fileId="abc" drive="family" subtitles={SUBS_A} />,
    );
    rerender(
      <TranscriptSection fileId="def" drive="family" subtitles={SUBS_B} />,
    );
    await screen.findByText("未修正の文章。");
    expect(screen.queryByRole("button", { name: "External" })).toBeNull();

    await act(async () => {
      slow.settle({
        ok: true,
        status: 200,
        text: async () =>
          "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nstale external cue\n",
        json: async () => null,
      } as Response);
      await slow.promise;
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(screen.queryByRole("button", { name: "External" })).toBeNull();
    expect(screen.queryByText("stale external cue")).toBeNull();
  });
  it("ignores the previous file's failure as well as its answer", async () => {
    // The `catch` half. It writes an empty list, which looks harmless
    // until it lands on a file that does have one — then the previous
    // file failing takes the current file's cues away. Asserted on the
    // toggle still being there, because an assertion that something is
    // *absent* passes before a late rejection has finished travelling.
    const slow = deferred<Response>();
    const wordVtt =
      "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nfirst\n\n00:00:02.000 --> 00:00:04.000\nsecond\n";
    fetchMock.mockImplementation((url: string) =>
      String(url).includes("/abc/")
        ? slow.promise
        : Promise.resolve({
            ok: true,
            status: 200,
            text: async () => wordVtt,
            json: async () => null,
          } as Response),
    );

    const { rerender } = render(<TranscriptSection fileId="abc" drive="family" />);
    rerender(<TranscriptSection fileId="def" drive="family" />);
    expect(await screen.findByRole("button", { name: "Words" })).toBeInTheDocument();

    await act(async () => {
      slow.reject(new Error("the previous file's request failed"));
      await slow.promise.catch(() => undefined);
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(screen.getByRole("button", { name: "Words" })).toBeInTheDocument();
  });
});

