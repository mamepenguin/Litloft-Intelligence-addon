import { describe, it, expect, vi, beforeEach, afterEach, beforeAll } from "vitest";
import { useEffect } from "react";
import { render, screen, fireEvent, waitFor, within } from "@testing-library/react";

import { formatDuration } from "@/lib/format";

// The chip labels and the group name are new keys whose names the spec leaves
// to the implementation, so every string is read from the real locale files.
const intl = vi.hoisted(() => ({ locale: "en" as "en" | "ja" }));

vi.mock("next-intl", async () => {
  const en = (await import("@/addons/intelligence/messages/en.json")).default as {
    file: Record<string, string>;
  };
  const ja = (await import("@/addons/intelligence/messages/ja.json")).default as {
    file: Record<string, string>;
  };
  return {
    useTranslations: () => (key: string, values?: Record<string, unknown>) => {
      const table = intl.locale === "ja" ? ja.file : en.file;
      const template = table[key] ?? key;
      return Object.entries(values ?? {}).reduce(
        (text, [name, value]) => text.replace(`{${name}}`, String(value)),
        template,
      );
    },
    useLocale: () => intl.locale,
  };
});

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

vi.mock("@/hooks/useWebSocket", () => ({
  useWebSocket: vi.fn(() => null),
}));

import ScenesSection from "@/addons/intelligence/ScenesSection";
import {
  getClipTimestamps,
  getFrameUrl,
  getSceneTileUrl,
  getVideoVisualIndex,
  type ClipTimestampItem,
} from "@/addons/intelligence/api";
import { usePolicy } from "@/hooks/usePolicy";
import {
  ShortcutsProvider,
  useShortcutsContext,
} from "@/components/ShortcutsProvider";
import {
  advance,
  allTiles,
  firePointer,
  flush,
  header,
  images,
  imgTimes,
  indexState,
  installPointerEvents,
  LOFT_MIME,
  makeController,
  makeRows,
  makeRun,
  makeScene,
  nameVisible,
  pointerX,
  ruler,
  SceneSectionProps,
  sceneCardNamed,
  stubRect,
  tile,
  tileLabels,
  waitForVisualIndex,
} from "./helpers";

/** Candidate frames every 5 s, 0..70. */
const CANDIDATES = makeRows(15);

/** Change frames at seconds no candidate has, so a tile's label says which list is drawn. */
const CHANGE_TIMES = [1, 4, 9, 13, 18, 22, 27, 31, 38, 44, 51, 59, 66];
const CHANGES: ClipTimestampItem[] = CHANGE_TIMES.map((start) => ({
  start,
  content_preview: `Frame at ${start}s`,
}));

const labels = (times: number[]) => times.map((time) => formatDuration(time));

function answer(
  candidates: ClipTimestampItem[],
  changes?: ClipTimestampItem[],
  fileId = "f1",
) {
  return {
    file_id: fileId,
    drive: "d1",
    timestamps: candidates,
    ...(changes === undefined ? {} : { changes }),
  };
}

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

function switchGroup(): HTMLElement | null {
  return screen.queryByRole("group", { name: "Frames shown" });
}

function allChip(): HTMLElement {
  return within(switchGroup()!).getByRole("button", { name: /^Default \d+$/ });
}

function cutsChip(): HTMLElement {
  return within(switchGroup()!).getByRole("button", { name: /^Cuts \d+$/ });
}

async function expectSwitch(): Promise<HTMLElement> {
  return waitFor(() => {
    const group = switchGroup();
    expect(group).not.toBeNull();
    return group!;
  });
}

function tileDot(el: HTMLElement): boolean {
  return el.querySelector('[class*="bg-accent-teal"]') !== null;
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
}

beforeAll(installPointerEvents);

beforeEach(() => {
  vi.clearAllMocks();
  intl.locale = "en";
  vi.mocked(getClipTimestamps).mockResolvedValue(answer(CANDIDATES, CHANGES));
  vi.mocked(getVideoVisualIndex).mockResolvedValue(indexState());
  vi.mocked(usePolicy).mockReturnValue({ enabled: true, isLoading: false });
});

afterEach(() => {
  vi.useRealTimers();
});

describe("SPEC-ADDON-124 I1/I5 — the switch and its counts", () => {
  it("SPEC-ADDON-124 I1/I5 — draws 「Default n」 and 「Cuts m」 in a 'Frames shown' group, Default pressed", async () => {
    renderSection({ mediaController: makeController({ duration: 70 }).controller });
    await openNative();

    await expectSwitch();
    expect(allChip()).toHaveAccessibleName("Default 15");
    expect(cutsChip()).toHaveAccessibleName("Cuts 13");
    expect(allChip()).toHaveAttribute("aria-pressed", "true");
    expect(cutsChip()).toHaveAttribute("aria-pressed", "false");
    expect(tileLabels()).toEqual(labels([0, 5, 10, 15, 20, 25, 30, 35, 40]));
  });

  it("SPEC-ADDON-124 I1 — no switch for zero or absent changes, a switch from one change", async () => {
    vi.mocked(getClipTimestamps)
      .mockResolvedValueOnce(answer(CANDIDATES, []))
      .mockResolvedValueOnce(answer(CANDIDATES, undefined, "f2"))
      .mockResolvedValueOnce(
        answer(CANDIDATES, [{ start: 33, content_preview: "Frame at 33s" }], "f3"),
      );
    const view = renderSection();

    await openNative();
    await waitForVisualIndex();
    expect(switchGroup()).toBeNull();
    expect(tileLabels()).toEqual(labels([0, 5, 10, 15, 20, 25, 30, 35, 40]));

    view.rerender(section({ fileId: "f2" }));
    await openNative();
    await waitFor(() => expect(getClipTimestamps).toHaveBeenCalledTimes(2));
    await flush();
    expect(switchGroup()).toBeNull();
    expect(tileLabels()).toEqual(labels([0, 5, 10, 15, 20, 25, 30, 35, 40]));

    view.rerender(section({ fileId: "f3" }));
    await openNative();
    await expectSwitch();
    expect(cutsChip()).toHaveAccessibleName("Cuts 1");
    fireEvent.click(cutsChip());
    expect(tileLabels()).toEqual(["0:33"]);
  });

  it("SPEC-ADDON-124 I5 — the header's bare number is the drawn list's length; the scene count stays", async () => {
    vi.mocked(getVideoVisualIndex).mockResolvedValue(
      indexState({
        active_run: makeRun({ succeeded_count: 2 }),
        scenes: [makeScene({ start_time: 20, scene_label: "Alpha" })],
      }),
    );
    renderSection({ mediaController: makeController({ duration: 70 }).controller });
    await openNative();
    await waitForVisualIndex();
    await expectSwitch();

    expect(header().textContent).toMatch(/\b15\b/);
    expect(nameVisible(/^2 scenes$/)).toBe(true);

    fireEvent.click(cutsChip());

    expect(header().textContent).toMatch(/\b13\b/);
    expect(header().textContent).not.toMatch(/\b15\b/);
    expect(nameVisible(/^2 scenes$/)).toBe(true);
  });
});

describe("SPEC-ADDON-124 I2 — the Cuts grid is a window of the change list", () => {
  it.each([
    { time: 0, windowStart: 0, label: "the point before the first change" },
    { time: 35, windowStart: 3, label: "i = 7" },
    { time: 70, windowStart: 4, label: "the point past the last change" },
  ])(
    "SPEC-ADDON-124 I2 — the window at $label",
    async ({ time, windowStart }) => {
      const { controller } = makeController({ currentTime: time, duration: 70 });
      renderSection({ mediaController: controller });
      await openNative();
      await expectSwitch();

      fireEvent.click(cutsChip());

      expect(cutsChip()).toHaveAttribute("aria-pressed", "true");
      expect(allChip()).toHaveAttribute("aria-pressed", "false");
      expect(tileLabels()).toEqual(
        labels(CHANGE_TIMES.slice(windowStart, windowStart + 9)),
      );
    },
  );

  it("SPEC-ADDON-124 I2 — fewer change frames than the window shows every one, in time order", async () => {
    vi.mocked(getClipTimestamps).mockResolvedValue(
      answer(CANDIDATES, [
        { start: 12, content_preview: "Frame at 12s" },
        { start: 41, content_preview: "Frame at 41s" },
        { start: 63, content_preview: "Frame at 63s" },
      ]),
    );
    renderSection({ mediaController: makeController({ currentTime: 50, duration: 70 }).controller });
    await openNative();
    await expectSwitch();

    fireEvent.click(cutsChip());

    expect(tileLabels()).toEqual(["0:12", "0:41", "1:03"]);
  });

  it("SPEC-ADDON-124 §1 step 5 — a mouse browse point is kept and the window re-anchors to it", async () => {
    const { controller, seek } = makeController({ duration: 70 });
    renderSection({ mediaController: controller });
    await openNative();
    await expectSwitch();
    const el = ruler();
    stubRect(el);

    firePointer(el, "pointerMove", { pointerType: "mouse", clientX: pointerX(60, 70) });
    expect(ruler()).toHaveAttribute("aria-valuenow", "60");

    fireEvent.click(cutsChip());

    expect(ruler()).toHaveAttribute("aria-valuenow", "60");
    expect(tileLabels()).toEqual(labels(CHANGE_TIMES.slice(4, 13)));
    expect(seek).not.toHaveBeenCalled();

    fireEvent.click(allChip());

    expect(tileLabels()).toEqual(labels([30, 35, 40, 45, 50, 55, 60, 65, 70]));
  });

  it("SPEC-ADDON-124 §9 — a change frame later than L keeps its true time on its tile", async () => {
    vi.mocked(getClipTimestamps).mockResolvedValue(
      answer(CANDIDATES, [
        { start: 33, content_preview: "Frame at 33s" },
        { start: 80, content_preview: "Frame at 80s" },
      ]),
    );
    renderSection({ mediaController: makeController({ duration: 0 }).controller });
    await openNative();
    await expectSwitch();

    fireEvent.click(cutsChip());

    expect(tileLabels()).toEqual(["0:33", "1:20"]);
  });
});

describe("SPEC-ADDON-124 I3 — a chip press requests only the new window's pictures", () => {
  it("SPEC-ADDON-124 I3 — sends no clip-timestamps or visual-index request", async () => {
    renderSection({ mediaController: makeController({ duration: 70 }).controller });
    await openNative();
    await waitForVisualIndex();
    await expectSwitch();
    const timestampsCalls = vi.mocked(getClipTimestamps).mock.calls.length;
    const indexCalls = vi.mocked(getVideoVisualIndex).mock.calls.length;

    fireEvent.click(cutsChip());
    await flush(8);
    fireEvent.click(allChip());
    await flush(8);
    fireEvent.click(cutsChip());
    await flush(8);

    expect(getClipTimestamps).toHaveBeenCalledTimes(timestampsCalls);
    expect(getVideoVisualIndex).toHaveBeenCalledTimes(indexCalls);
  });

  it("SPEC-ADDON-124 I3 — the change tiles ask for their pictures after the settle delay, through the frame route", async () => {
    vi.useFakeTimers();
    vi.mocked(usePolicy).mockReturnValue({ enabled: false, isLoading: false });
    renderSection({ mediaController: makeController({ duration: 70 }).controller });
    fireEvent.click(header());
    await flush();
    expect(switchGroup()).not.toBeNull();
    await advance(150);
    expect(imgTimes()).toEqual([0, 5, 10, 15, 20, 25, 30, 35, 40]);

    fireEvent.click(cutsChip());
    await advance(149);
    for (const time of CHANGE_TIMES.slice(0, 9)) {
      expect(imgTimes()).not.toContain(time);
    }

    await advance(1);
    expect(imgTimes()).toEqual(CHANGE_TIMES.slice(0, 9));
    for (const time of CHANGE_TIMES.slice(0, 9)) {
      expect(getFrameUrl).toHaveBeenCalledWith("f1", time);
    }
    for (const time of CHANGE_TIMES.slice(9)) {
      expect(getFrameUrl).not.toHaveBeenCalledWith("f1", time);
    }
  });

  it("SPEC-ADDON-124 §4 — toggling back inside the settle delay requests no change picture", async () => {
    vi.useFakeTimers();
    vi.mocked(usePolicy).mockReturnValue({ enabled: false, isLoading: false });
    renderSection({ mediaController: makeController({ duration: 70 }).controller });
    fireEvent.click(header());
    await flush();
    await advance(150);

    fireEvent.click(cutsChip());
    await advance(100);
    fireEvent.click(allChip());
    await advance(300);

    for (const time of CHANGE_TIMES) {
      expect(getFrameUrl).not.toHaveBeenCalledWith("f1", time);
    }
    expect(imgTimes()).toEqual([0, 5, 10, 15, 20, 25, 30, 35, 40]);
  });
});

describe("SPEC-ADDON-124 I4 — everything but the grid is the same in both modes", () => {
  it("SPEC-ADDON-124 I4 — aria-valuemax, the scene card and the Visual Index count do not move", async () => {
    vi.mocked(getVideoVisualIndex).mockResolvedValue(
      indexState({
        active_run: makeRun({ succeeded_count: 2 }),
        scenes: [
          makeScene({ ordering: 0, start_time: 20, section_start: 20, scene_label: "Alpha" }),
          makeScene({ ordering: 1, start_time: 50, section_start: 50, scene_label: "Beta" }),
        ],
      }),
    );
    renderSection({ mediaController: makeController({ currentTime: 0, duration: 70 }).controller });
    await openNative();
    await waitForVisualIndex();
    await expectSwitch();
    const max = ruler().getAttribute("aria-valuemax");
    expect(max).toBe("70");
    expect(sceneCardNamed(/Alpha/)).not.toBeNull();

    fireEvent.click(cutsChip());

    expect(ruler()).toHaveAttribute("aria-valuemax", max!);
    expect(sceneCardNamed(/Alpha/)).not.toBeNull();
    expect(nameVisible(/^2 scenes$/)).toBe(true);
  });

  it("SPEC-ADDON-124 I4 — L is the candidates' last start plus the median gap in both modes when the duration is unknown", async () => {
    renderSection({ mediaController: makeController({ duration: 0 }).controller });
    await openNative();
    await expectSwitch();
    expect(ruler()).toHaveAttribute("aria-valuemax", "75");

    fireEvent.click(cutsChip());

    expect(ruler()).toHaveAttribute("aria-valuemax", "75");
  });

  it("SPEC-ADDON-090 I9 as revised by SPEC-ADDON-124 — the dot marks the tile holding section_start in the shown list", async () => {
    vi.mocked(getVideoVisualIndex).mockResolvedValue(
      indexState({
        active_run: makeRun({ succeeded_count: 1 }),
        scenes: [makeScene({ start_time: 20, section_start: 20, scene_label: "Alpha" })],
      }),
    );
    renderSection({ mediaController: makeController({ duration: 70 }).controller });
    await openNative();
    await waitForVisualIndex();
    await expectSwitch();
    expect(allTiles().filter(tileDot).map((el) => el.getAttribute("aria-label"))).toEqual([
      "Play from 0:20",
    ]);

    fireEvent.click(cutsChip());

    expect(allTiles().filter(tileDot).map((el) => el.getAttribute("aria-label"))).toEqual([
      "Play from 0:18",
    ]);
  });

  it("SPEC-ADDON-124 I4 — a change tile seeks to its own time", async () => {
    const { controller, seek, play } = makeController({ duration: 70 });
    renderSection({ mediaController: controller });
    await openNative();
    await expectSwitch();

    fireEvent.click(cutsChip());
    fireEvent.click(tile(13));

    expect(seek).toHaveBeenCalledWith(13);
    expect(play).toHaveBeenCalled();
  });
});

function PageShortcuts() {
  const { push, pop } = useShortcutsContext();
  useEffect(() => {
    push({ id: "page-test", label: "page test", shortcuts: [] });
    return () => pop("page-test");
  }, [push, pop]);
  return null;
}

describe("SPEC-ADDON-124 I4 — the ruler keys run over the shown list", () => {
  async function cutsWithFocus(
    props: Partial<SceneSectionProps>,
  ): Promise<HTMLElement> {
    render(
      <ShortcutsProvider>
        <PageShortcuts />
        {section(props)}
      </ShortcutsProvider>,
    );
    await flush();
    fireEvent.click(header());
    await screen.findByRole("slider");
    await expectSwitch();
    fireEvent.click(cutsChip());
    const el = ruler();
    el.focus();
    return el;
  }

  it("SPEC-ADDON-124 I4 — the arrows step one change frame from the playing one", async () => {
    const { controller } = makeController({ duration: 70, currentTime: 10 });
    const el = await cutsWithFocus({ mediaController: controller });

    fireEvent.keyDown(el, { key: "ArrowRight" });
    expect(ruler()).toHaveAttribute("aria-valuenow", "13");

    fireEvent.keyDown(el, { key: "ArrowLeft" });
    expect(ruler()).toHaveAttribute("aria-valuenow", "9");
  });

  it("SPEC-ADDON-124 I4 — Home and End reach the change list's ends", async () => {
    const { controller } = makeController({ duration: 70, currentTime: 10 });
    const el = await cutsWithFocus({ mediaController: controller });

    fireEvent.keyDown(el, { key: "Home" });
    expect(ruler()).toHaveAttribute("aria-valuenow", "1");

    fireEvent.keyDown(el, { key: "End" });
    expect(ruler()).toHaveAttribute("aria-valuenow", "66");
  });

  it("SPEC-ADDON-124 I4 — Enter seeks the browsed change frame", async () => {
    const { controller, seek, play } = makeController({ duration: 70, currentTime: 10 });
    const el = await cutsWithFocus({ mediaController: controller });

    fireEvent.keyDown(el, { key: "ArrowRight" });
    fireEvent.keyDown(el, { key: "Enter" });

    expect(seek).toHaveBeenCalledWith(13);
    expect(play).toHaveBeenCalled();
  });
});

type View = ReturnType<typeof renderSection>;

const REOPEN: {
  name: string;
  props: Partial<SceneSectionProps>;
  open: () => Promise<void>;
  reopen: (view: View, props: Partial<SceneSectionProps>) => Promise<void>;
}[] = [
  {
    name: "collapse and expand",
    props: {},
    open: openNative,
    reopen: async () => {
      fireEvent.click(header());
      await openNative();
    },
  },
  {
    name: "a host hide and show",
    props: { labelledByHost: true, active: true, onAvailability: () => {} },
    open: async () => {
      await screen.findByRole("slider");
    },
    reopen: async (view, props) => {
      view.rerender(section({ ...props, active: false }));
      view.rerender(section({ ...props, active: true }));
    },
  },
];

describe("SPEC-ADDON-124 I7 — the mode across responses", () => {
  it.each(REOPEN)(
    "SPEC-ADDON-124 I7 — keeps Cuts across $name while the answer has changes",
    async ({ props, open, reopen }) => {
      const base = { ...props, mediaController: makeController({ duration: 70 }).controller };
      const view = renderSection(base);
      await open();
      await expectSwitch();
      fireEvent.click(cutsChip());
      const calls = vi.mocked(getClipTimestamps).mock.calls.length;

      await reopen(view, base);
      await waitFor(() =>
        expect(vi.mocked(getClipTimestamps).mock.calls.length).toBe(calls + 1),
      );
      await flush(8);
      await expectSwitch();

      expect(cutsChip()).toHaveAttribute("aria-pressed", "true");
      expect(tileLabels()).toEqual(labels(CHANGE_TIMES.slice(0, 9)));
    },
  );

  it.each(REOPEN)(
    "SPEC-ADDON-124 I7 — an answer without changes after $name drops the switch and resets the selection to Default",
    async ({ props, open, reopen }) => {
      const base = { ...props, mediaController: makeController({ duration: 70 }).controller };
      const view = renderSection(base);
      await open();
      await expectSwitch();
      fireEvent.click(cutsChip());

      vi.mocked(getClipTimestamps).mockResolvedValue(answer(CANDIDATES, []));
      let calls = vi.mocked(getClipTimestamps).mock.calls.length;
      await reopen(view, base);
      await waitFor(() =>
        expect(vi.mocked(getClipTimestamps).mock.calls.length).toBe(calls + 1),
      );
      await flush(8);
      expect(switchGroup()).toBeNull();
      expect(tileLabels()).toEqual(labels([0, 5, 10, 15, 20, 25, 30, 35, 40]));

      vi.mocked(getClipTimestamps).mockResolvedValue(answer(CANDIDATES, CHANGES));
      calls = vi.mocked(getClipTimestamps).mock.calls.length;
      await reopen(view, base);
      await waitFor(() =>
        expect(vi.mocked(getClipTimestamps).mock.calls.length).toBe(calls + 1),
      );
      await expectSwitch();

      expect(allChip()).toHaveAttribute("aria-pressed", "true");
      expect(cutsChip()).toHaveAttribute("aria-pressed", "false");
      expect(tileLabels()).toEqual(labels([0, 5, 10, 15, 20, 25, 30, 35, 40]));
    },
  );

  it("SPEC-ADDON-124 I7 — a file change returns to Default", async () => {
    const view = renderSection({ mediaController: makeController({ duration: 70 }).controller });
    await openNative();
    await expectSwitch();
    fireEvent.click(cutsChip());

    vi.mocked(getClipTimestamps).mockResolvedValue(answer(CANDIDATES, CHANGES, "f2"));
    view.rerender(
      section({ fileId: "f2", mediaController: makeController({ duration: 70 }).controller }),
    );
    await openNative();
    await expectSwitch();

    expect(allChip()).toHaveAttribute("aria-pressed", "true");
    expect(tileLabels()).toEqual(labels([0, 5, 10, 15, 20, 25, 30, 35, 40]));
  });

  it("SPEC-ADDON-124 §3 — no switch while the first request is in flight, then the switch", async () => {
    const pending = deferred<ReturnType<typeof answer>>();
    vi.mocked(getClipTimestamps).mockReturnValueOnce(pending.promise as never);
    renderSection({ mediaController: makeController({ duration: 70 }).controller });
    fireEvent.click(header());
    await flush();

    expect(switchGroup()).toBeNull();

    pending.resolve(answer(CANDIDATES, CHANGES));
    await expectSwitch();
  });
});

describe("SPEC-ADDON-124 §2 — failure cases", () => {
  it("SPEC-ADDON-124 §2 — a .loft change tile whose picture fails keeps its time and seek and shows the empty block", async () => {
    const { controller, seek } = makeController({ duration: 70 });
    renderSection({ mimeType: LOFT_MIME, mediaController: controller });
    fireEvent.click(await screen.findByRole("button", { name: /^Scenes/ }));
    await expectSwitch();
    fireEvent.click(cutsChip());
    await waitFor(() => expect(imgTimes()).toContain(13));

    const failed = images().find((img) => /[?&]t=13(?:&|$)/.test(img.src))!;
    fireEvent.error(failed);

    expect(tile(13).querySelector("img")).toBeNull();
    expect(tile(13)).toHaveAttribute("aria-label", "Play from 0:13");
    expect(imgTimes()).toEqual(CHANGE_TIMES.slice(0, 9).filter((t) => t !== 13));
    fireEvent.click(tile(13));
    expect(seek).toHaveBeenCalledWith(13);
  });

  it("SPEC-ADDON-124 §2 — change frames without candidates never make a native video available; with candidates they draw the switch", async () => {
    const report = vi.fn();
    vi.mocked(getClipTimestamps)
      .mockResolvedValueOnce(answer([], CHANGES))
      .mockResolvedValue(answer(CANDIDATES, CHANGES, "f2"));
    const props = { labelledByHost: true, onAvailability: report };
    const view = renderSection({ ...props, active: false });
    await waitFor(() => expect(getClipTimestamps).toHaveBeenCalledTimes(1));
    await flush(8);

    expect(report.mock.calls.map(([available]) => available)).not.toContain(true);
    expect(switchGroup()).toBeNull();

    view.rerender(section({ ...props, fileId: "f2", active: true }));
    await expectSwitch();
  });

  it("SPEC-ADDON-124 §2 — a .loft with change frames, no candidates and no Visual Index stays hidden; with candidates it draws the switch", async () => {
    vi.mocked(getVideoVisualIndex).mockResolvedValue(indexState({ eligible: false }));
    vi.mocked(getClipTimestamps)
      .mockResolvedValueOnce(answer([], CHANGES))
      .mockResolvedValue(answer(CANDIDATES, CHANGES, "f2"));
    const view = renderSection({ mimeType: LOFT_MIME });
    await waitFor(() => expect(getVideoVisualIndex).toHaveBeenCalledTimes(1));
    await flush(8);

    expect(view.container.firstChild).toBeNull();

    view.rerender(section({ mimeType: LOFT_MIME, fileId: "f2" }));
    fireEvent.click(await screen.findByRole("button", { name: /^Scenes/ }));
    await expectSwitch();
  });

  it("SPEC-ADDON-124 §2 — a .loft with change frames draws the switch and takes the cut pictures from the scene-tile route", async () => {
    renderSection({ mimeType: LOFT_MIME, mediaController: makeController({ duration: 70 }).controller });
    await screen.findByRole("button", { name: /^Scenes/ });
    fireEvent.click(header());
    await expectSwitch();

    fireEvent.click(cutsChip());

    await waitFor(() => expect(imgTimes()).toEqual(CHANGE_TIMES.slice(0, 9)));
    for (const img of images()) {
      expect(img.getAttribute("src")).toContain("/scene-tile/f1?t=");
    }
    expect(getSceneTileUrl).toHaveBeenCalledWith("f1", 13);
    expect(getFrameUrl).not.toHaveBeenCalled();
  });

  it("SPEC-ADDON-124 §3 — available through the Visual Index with no candidates: Default has no tiles, Cuts shows the cuts", async () => {
    vi.mocked(getClipTimestamps).mockResolvedValue(answer([], CHANGES));
    vi.mocked(getVideoVisualIndex).mockResolvedValue(
      indexState({
        active_run: makeRun({ succeeded_count: 1 }),
        scenes: [makeScene({ start_time: 20, scene_label: "Alpha" })],
      }),
    );
    renderSection({ mimeType: LOFT_MIME });
    fireEvent.click(await screen.findByRole("button", { name: /^Scenes/ }));
    await expectSwitch();

    expect(allChip()).toHaveAccessibleName("Default 0");
    expect(cutsChip()).toHaveAccessibleName("Cuts 13");
    expect(screen.queryAllByRole("button", { name: /^Play from / })).toEqual([]);

    fireEvent.click(cutsChip());

    expect(tileLabels()).toEqual(labels(CHANGE_TIMES.slice(0, 9)));
  });
});

describe("SPEC-ADDON-124 §9 — the switch row", () => {
  it("SPEC-ADDON-104 I3 as revised by SPEC-ADDON-124 — the switch sits after the ruler and before the first tile", async () => {
    renderSection({ mediaController: makeController({ duration: 70 }).controller });
    await openNative();
    const group = await expectSwitch();

    const follows = (a: Node, b: Node) =>
      (a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING) !== 0;
    expect(follows(ruler(), group)).toBe(true);
    expect(follows(group, allTiles()[0])).toBe(true);
  });

  it("SPEC-ADDON-124 §9 — the pressed chip carries the accent border and the other the muted text", async () => {
    renderSection({ mediaController: makeController({ duration: 70 }).controller });
    await openNative();
    await expectSwitch();

    const selected = ["border-accent", "font-semibold", "text-text-primary"];
    const unselected = ["border-transparent", "text-text-muted"];
    expect(allChip()).toHaveClass(...selected);
    expect(cutsChip()).toHaveClass(...unselected);

    fireEvent.click(cutsChip());

    expect(cutsChip()).toHaveClass(...selected);
    expect(allChip()).toHaveClass(...unselected);
  });
});

describe("SPEC-ADDON-124 I8 — the new labels in both locales", () => {
  it.each([
    { locale: "en" as const, group: "Frames shown", all: "Default 15", cuts: "Cuts 13" },
    { locale: "ja" as const, group: "表示するフレーム", all: "標準 15", cuts: "カット 13" },
  ])(
    "SPEC-ADDON-124 I8 — $locale renders the group name and both chips from its locale file",
    async ({ locale, group, all, cuts }) => {
      intl.locale = locale;
      renderSection({
        active: true,
        labelledByHost: true,
        onAvailability: vi.fn(),
        mediaController: makeController({ duration: 70 }).controller,
      });

      const box = await screen.findByRole("group", { name: group });
      expect(within(box).getByRole("button", { name: all })).toBeInTheDocument();
      expect(within(box).getByRole("button", { name: cuts })).toBeInTheDocument();
    },
  );
});
