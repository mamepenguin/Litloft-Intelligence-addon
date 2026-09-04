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
import { render, screen, cleanup, fireEvent, waitFor } from "@testing-library/react";
import React from "react";

// Mock global fetch (used by component for VTT endpoints)
const fetchMock = vi.fn().mockResolvedValue({
  ok: false,
  status: 404,
  text: async () => "",
  json: async () => null,
} as Response);
vi.stubGlobal("fetch", fetchMock);

// Mock the addon API module to return controlled transcript data.
// The module exports `getFileTranscript` which the component calls on
// mount. The test data includes refined + unrefined chunks.
vi.mock("@/addons/intelligence/api", () => ({
  getFileTranscript: vi.fn().mockResolvedValue({
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
        // New fields (spec): refinedAt + textOriginal
        refinedAt: "2026-04-15T00:00:00Z",
        textOriginal: "これはげんぶんの文章です。",
      },
      {
        index: 1,
        text: "未修正の文章。",
        start: 5,
        end: 10,
      },
    ],
  }),
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
} from "@/addons/intelligence/transcriptScroll";

/**
 * The two-cue response the module mock is set up with, restated so a
 * test that overrides it can put it back.
 */
const TRANSCRIPT_RESPONSE = {
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
};

async function transcriptApiMock() {
  const apiMock = await import("@/addons/intelligence/api");
  return apiMock.getFileTranscript as unknown as ReturnType<typeof vi.fn>;
}

/**
 * One mock object is shared by every test in this file.
 *
 * A test that changes what it resolves to changes it for whatever runs
 * next — and under the shuffled-order job that is not the test written
 * below it. One test handing this file an untranscribed video was
 * enough to time out eight tests in two other describes, and it stayed
 * invisible in source order only because it happened to be near the end.
 *
 * `mockReset` and not merely `mockResolvedValue`: an unconsumed
 * `mockResolvedValueOnce` would otherwise still be at the head of the
 * queue when the next test asked.
 */
beforeEach(async () => {
  const getFileTranscript = await transcriptApiMock();
  getFileTranscript.mockReset();
  getFileTranscript.mockResolvedValue(TRANSCRIPT_RESPONSE);
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
    // Only the name goes. The language, the count and the controls are
    // facts about this transcript, not a second name for it.
    const { container } = render(
      <TranscriptSection fileId="abc" drive="family" labelledByHost />,
    );

    await screen.findByText("未修正の文章。");
    expect(screen.getByText("ja")).toBeInTheDocument();
    expect(screen.getByText("(2)")).toBeInTheDocument();
    expect(container.querySelector(".mb-2")).not.toBeEmptyDOMElement();
  });
});

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
      // keyed on the cue count, so two sources of equal length would
      // not re-run it and the switch would prove nothing.
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
  async function mountAndScroll(fileId: string, top: number) {
    const utils = render(<TranscriptSection fileId={fileId} drive="family" />);
    await screen.findByText("未修正の文章。");
    const list = utils.container.querySelector(".overflow-y-auto")! as HTMLElement;
    list.scrollTop = top;
    fireEvent.scroll(list);
    return { utils, list };
  }

  async function remount(fileId: string) {
    const utils = render(<TranscriptSection fileId={fileId} drive="family" />);
    await screen.findByText("未修正の文章。");
    return utils.container.querySelector(".overflow-y-auto")! as HTMLElement;
  }

  it("puts the reader back where they were", async () => {
    const { utils } = await mountAndScroll("abc", 420);
    utils.unmount();

    expect((await remount("abc")).scrollTop).toBe(420);
  });

  it("keeps each file's place to itself", async () => {
    // Keyed by file, so opening a second one and coming back does not
    // land the reader at someone else's offset.
    const { utils } = await mountAndScroll("abc", 420);
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
    await screen.findByText("未修正の文章。");
    const list = utils.container.querySelector(".overflow-y-auto")! as HTMLElement;
    list.scrollTop = 300;
    fireEvent.wheel(list);
    fireEvent.scroll(list);
    expect(await screen.findByRole("button", { name: CHIP })).toBeInTheDocument();
    utils.unmount();

    const back = withPlayer();
    await screen.findByText("未修正の文章。");
    const list2 = back.container.querySelector(".overflow-y-auto")! as HTMLElement;
    expect(list2.scrollTop).toBe(300);
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
    const { utils } = await mountAndScroll("abc", 420);
    utils.unmount();
    const list = await remount("abc");
    expect(list.scrollTop).toBe(420);
    list.scrollTop = 10;

    fireEvent.click(await screen.findByRole("button", { name: "Words" }));

    expect(list.scrollTop).toBe(10);
  });

  it("forgets the oldest file rather than growing without limit", async () => {
    // Module state nothing ever clears. A tab left open for a week
    // browsing a large drive would otherwise keep an entry per file.
    for (let i = 0; i < 21; i += 1) {
      const { utils } = await mountAndScroll(`f${i}`, 100 + i);
      utils.unmount();
    }

    expect(recallTranscriptScroll("f0")).toBeUndefined();
    expect(recallTranscriptScroll("f1")).toEqual({ top: 101, following: true });
    expect(recallTranscriptScroll("f20")).toEqual({ top: 120, following: true });
  });
});
