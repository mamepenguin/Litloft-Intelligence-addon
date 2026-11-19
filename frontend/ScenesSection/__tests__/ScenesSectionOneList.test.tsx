import { describe, it, expect, vi, beforeEach, afterEach, beforeAll } from "vitest";
import { useEffect } from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";

import { formatDuration } from "@/lib/format";

// Strings come from the real locale files, so a key that is missing renders as
// its path instead of passing a query that names the English text.
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
import enMessages from "@/addons/intelligence/messages/en.json";
import jaMessages from "@/addons/intelligence/messages/ja.json";
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

/** Change frames at seconds no candidate has, so a tile drawn from them shows it. */
const CHANGE_TIMES = [1, 4, 9, 13, 18, 22, 27, 31, 38, 44, 51, 59, 66];
const CHANGES: ClipTimestampItem[] = CHANGE_TIMES.map((start) => ({
  start,
  content_preview: `Frame at ${start}s`,
}));

const REMOVED_KEYS = ["scenesFramesShown", "scenesFramesAll", "scenesFramesCuts"];

/** A non-integer duration away from .5, so rounding and the fractions are both visible. */
const D = 83.4;

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

async function expanded(): Promise<void> {
  await waitFor(() => expect(getClipTimestamps).toHaveBeenCalled());
  await flush(8);
  if (!screen.queryByRole("slider")) fireEvent.click(header());
  await screen.findByRole("slider");
}

async function openLoft(): Promise<void> {
  fireEvent.click(await screen.findByRole("button", { name: /^(Scenes|シーン)/ }));
  await screen.findByRole("slider");
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
}

function tileDot(el: HTMLElement): boolean {
  return el.querySelector('[class*="bg-accent-teal"]') !== null;
}

function dottedTiles(): string[] {
  return allTiles()
    .filter(tileDot)
    .map((el) => el.getAttribute("aria-label") ?? "");
}

/** The inline `left` percentages of the ruler's point markers (scene marks and the playhead). */
function markerPercents(): number[] {
  return Array.from(ruler().querySelectorAll<HTMLElement>("span"))
    .map((el) => /^(-?\d+(?:\.\d+)?)%$/.exec(el.style.left.trim())?.[1])
    .filter((value): value is string => value !== undefined)
    .map(Number);
}

function expectMarkerAt(percent: number): void {
  const percents = markerPercents();
  expect(
    percents.some((value) => Math.abs(value - percent) < 0.01),
    `a marker at ${percent}% among ${JSON.stringify(percents)}`,
  ).toBe(true);
}

function textShown(text: string): boolean {
  return screen.queryAllByText(text).length > 0;
}

const SWITCH_NAMES = /^(Frames shown|表示するフレーム)$/;
const SWITCH_CHIPS = /^(Default|Cuts|標準|カット)(\s+\d+)?$/;
const KEY_PATH = /\bscenes[A-Z]\w*\b|\bfile\.scenes/;

/** Every name or text on screen that is the switch, one of its chips, or a key path. */
function switchTraces(): string[] {
  const found: string[] = [];
  for (const el of Array.from(document.body.querySelectorAll("*"))) {
    const name = (el.getAttribute("aria-label") ?? "").trim();
    const own = Array.from(el.childNodes)
      .filter((node) => node.nodeType === Node.TEXT_NODE)
      .map((node) => node.textContent ?? "")
      .join("")
      .trim();
    for (const value of [name, own]) {
      if (!value) continue;
      if (SWITCH_NAMES.test(value) || SWITCH_CHIPS.test(value) || KEY_PATH.test(value)) {
        found.push(value);
      }
    }
    const labelledBy = el.getAttribute("aria-labelledby");
    if (el.getAttribute("role") === "group" && labelledBy) found.push(`group:${labelledBy}`);
  }
  for (const group of screen.queryAllByRole("group")) {
    found.push(`group:${group.getAttribute("aria-label") ?? ""}`);
  }
  for (const button of screen.queryAllByRole("button")) {
    if (button.hasAttribute("aria-pressed")) {
      found.push(`pressed:${button.getAttribute("aria-label") ?? button.textContent ?? ""}`);
    }
  }
  return found;
}

function keysIn(value: unknown, prefix = ""): string[] {
  if (value === null || typeof value !== "object") return [];
  return Object.entries(value as Record<string, unknown>).flatMap(([key, child]) => [
    prefix ? `${prefix}.${key}` : key,
    ...keysIn(child, prefix ? `${prefix}.${key}` : key),
  ]);
}

const withScenes = (scenes = [makeScene({ start_time: 20, section_start: 20, scene_label: "Alpha" })]) =>
  indexState({ active_run: makeRun({ succeeded_count: scenes.length }), scenes });

beforeAll(installPointerEvents);

beforeEach(() => {
  vi.clearAllMocks();
  intl.locale = "en";
  vi.mocked(getClipTimestamps).mockReset();
  vi.mocked(getClipTimestamps).mockResolvedValue(answer(CANDIDATES, CHANGES));
  vi.mocked(getVideoVisualIndex).mockResolvedValue(indexState());
  vi.mocked(usePolicy).mockReturnValue({ enabled: true, isLoading: false });
});

afterEach(() => {
  vi.useRealTimers();
});

describe("SPEC-ADDON-129 I1 — no frame-list switch", () => {
  it.each([{ locale: "en" as const }, { locale: "ja" as const }])(
    "SPEC-ADDON-129 I1, I5 — $locale: no switch and no key path while loading, with change frames, without them, and for a .loft with only change frames",
    async ({ locale }) => {
      intl.locale = locale;
      const pending = deferred<ReturnType<typeof answer>>();
      vi.mocked(getClipTimestamps).mockImplementation(((fileId: string) => {
        if (fileId === "f2") return Promise.resolve(answer(CANDIDATES, undefined, "f2"));
        if (fileId === "f3") return Promise.resolve(answer([], CHANGES, "f3"));
        return pending.promise;
      }) as never);
      vi.mocked(getVideoVisualIndex).mockResolvedValue(withScenes());
      const controller = makeController({ duration: 70 }).controller;
      const view = render(
        section({ active: true, labelledByHost: true, onAvailability: vi.fn(), mediaController: controller }),
      );
      await flush();
      expect(switchTraces()).toEqual([]);

      pending.resolve(answer(CANDIDATES, CHANGES));
      await screen.findByRole("slider");
      await waitForVisualIndex();
      await flush(8);
      expect(screen.getAllByRole("button", { name: /^Play from |から再生$/ })).toHaveLength(15);
      expect(switchTraces()).toEqual([]);

      view.rerender(
        section({ fileId: "f2", active: true, labelledByHost: true, onAvailability: vi.fn(), mediaController: controller }),
      );
      await waitFor(() => expect(getClipTimestamps).toHaveBeenCalledWith("f2", "d1"));
      await screen.findByRole("slider");
      await flush(8);
      expect(switchTraces()).toEqual([]);

      view.rerender(
        section({
          fileId: "f3",
          mimeType: LOFT_MIME,
          active: true,
          labelledByHost: true,
          onAvailability: vi.fn(),
          mediaController: controller,
        }),
      );
      await waitFor(() => expect(getClipTimestamps).toHaveBeenCalledWith("f3", "d1"));
      await screen.findByRole("slider");
      await flush(8);
      expect(switchTraces()).toEqual([]);
      expect(screen.queryAllByRole("button", { name: /^Play from |から再生$/ })).toEqual([]);
    },
  );
});

describe("SPEC-ADDON-129 I2 — the section reads only timestamps", () => {
  async function observe(
    props: Partial<SceneSectionProps>,
    body: ReturnType<typeof answer>,
    loft: boolean,
  ) {
    vi.mocked(getClipTimestamps).mockResolvedValue(body);
    const view = renderSection(props);
    if (loft) await openLoft();
    else await openNative();
    await waitForVisualIndex();
    await flush(8);
    const seen = {
      tiles: screen.queryAllByRole("button", { name: /^Play from / }).map(
        (el) => el.getAttribute("aria-label") ?? "",
      ),
      header: header().textContent ?? "",
      headerName: header().getAttribute("aria-label") ?? header().textContent ?? "",
      max: ruler().getAttribute("aria-valuemax"),
      markup: ruler().innerHTML,
      imgs: imgTimes(),
    };
    view.unmount();
    return seen;
  }

  it.each([
    {
      name: "a native video with candidates and a duration",
      props: { mimeType: "video/mp4" },
      candidates: CANDIDATES,
      duration: 70,
      loft: false,
    },
    {
      name: "a native video with candidates and no duration",
      props: { mimeType: "video/mp4" },
      candidates: CANDIDATES,
      duration: 0,
      loft: false,
    },
    {
      name: "a .loft with no candidates and no duration",
      props: { mimeType: LOFT_MIME },
      candidates: [],
      duration: 0,
      loft: true,
    },
    {
      name: "a .loft with no candidates and a duration",
      props: { mimeType: LOFT_MIME },
      candidates: [],
      duration: 30.2,
      loft: true,
    },
  ])(
    "SPEC-ADDON-129 I2 — $name draws the same with and without a changes array",
    async ({ props, candidates, duration, loft }) => {
      vi.mocked(getVideoVisualIndex).mockResolvedValue(withScenes());
      const without = await observe(
        { ...props, mediaController: makeController({ duration }).controller },
        answer(candidates),
        loft,
      );
      const withChanges = await observe(
        { ...props, mediaController: makeController({ duration }).controller },
        answer(candidates, CHANGES),
        loft,
      );

      expect(withChanges).toEqual(without);
      const starts = new Set(candidates.map((row) => `Play from ${formatDuration(row.start)}`));
      for (const label of withChanges.tiles) expect(starts.has(label)).toBe(true);
    },
  );

  it("SPEC-ADDON-129 I2 — the header's bare number is the candidate count and its name keeps the space", async () => {
    vi.mocked(getVideoVisualIndex).mockResolvedValue(
      withScenes([
        makeScene({ ordering: 0, start_time: 20, scene_label: "Alpha" }),
        makeScene({ ordering: 1, start_time: 40, scene_label: "Beta" }),
      ]),
    );
    renderSection({ mediaController: makeController({ duration: 70 }).controller });
    await openNative();
    await waitForVisualIndex();

    expect(header()).toHaveAccessibleName(/^Scenes 15(\s|$)/);
    expect(header().textContent).toMatch(/\b15\b/);
    expect(header().textContent).not.toMatch(/\b13\b/);
    expect(switchTraces()).toEqual([]);
  });
});

describe("SPEC-ADDON-129 I4 — no candidates: the ruler takes the player's duration", () => {
  const scenes = [
    makeScene({ ordering: 0, start_time: 20, section_start: 20, scene_label: "Alpha" }),
    makeScene({ ordering: 1, start_time: 120, section_start: 120, scene_label: "Beyond" }),
  ];

  async function openEmptyLoft(state: { currentTime?: number; duration: number }) {
    vi.mocked(getClipTimestamps).mockResolvedValue(answer([]));
    vi.mocked(getVideoVisualIndex).mockResolvedValue(withScenes(scenes));
    const clock = makeController({ currentTime: 40, ...state });
    const view = renderSection({ mimeType: LOFT_MIME, mediaController: clock.controller });
    await openLoft();
    await waitForVisualIndex();
    await flush(8);
    return { ...clock, view };
  }

  it("SPEC-ADDON-129 I4 — aria-valuemax is round(D), the 0:00 and end labels show and no tile is drawn", async () => {
    await openEmptyLoft({ duration: D });

    expect(ruler()).toHaveAttribute("aria-valuemax", String(Math.round(D)));
    expect(textShown("0:00")).toBe(true);
    expect(textShown(formatDuration(D))).toBe(true);
    expect(screen.queryAllByRole("button", { name: /^Play from / })).toEqual([]);
  });

  it("SPEC-ADDON-129 I4 — a scene mark sits at min(seek, D) / D and the playhead at position / D", async () => {
    await openEmptyLoft({ duration: D, currentTime: 40 });

    expectMarkerAt((20 / D) * 100);
    expectMarkerAt(100);
    expectMarkerAt((40 / D) * 100);
  });

  it("SPEC-ADDON-129 I4 — a position past D draws the playhead at the ruler's end", async () => {
    vi.mocked(getVideoVisualIndex).mockResolvedValue(withScenes(scenes.slice(0, 1)));
    vi.mocked(getClipTimestamps).mockResolvedValue(answer([]));
    renderSection({
      mimeType: LOFT_MIME,
      mediaController: makeController({ duration: D, currentTime: 95 }).controller,
    });
    await openLoft();
    await waitForVisualIndex();
    await flush(8);

    expectMarkerAt((20 / D) * 100);
    expectMarkerAt(100);
  });

  it("SPEC-ADDON-129 I4 as revised by SPEC-ADDON-153 — a click at a fraction of the ruler does nothing", async () => {
    const { seek, play } = await openEmptyLoft({ duration: D, currentTime: 40 });
    const el = ruler();
    stubRect(el);
    const x = 0.6 * 700;

    firePointer(el, "pointerDown", { pointerType: "mouse", clientX: x });
    firePointer(el, "pointerUp", { pointerType: "mouse", clientX: x });
    fireEvent.click(el, { clientX: x });

    expect(seek).not.toHaveBeenCalled();
    expect(play).not.toHaveBeenCalled();
    expect(ruler()).toHaveAttribute("aria-valuenow", "40");
  });

  it("SPEC-ADDON-129 I4 as revised by SPEC-ADDON-153 — hover, the arrows, Home/End and Enter all do nothing", async () => {
    vi.mocked(getClipTimestamps).mockResolvedValue(answer([]));
    vi.mocked(getVideoVisualIndex).mockResolvedValue(withScenes(scenes));
    const { controller, seek } = makeController({ duration: D, currentTime: 40 });
    render(
      <ShortcutsProvider>
        <PageShortcuts />
        {section({ mimeType: LOFT_MIME, mediaController: controller })}
      </ShortcutsProvider>,
    );
    await openLoft();
    await waitForVisualIndex();
    const el = ruler();
    stubRect(el);

    firePointer(el, "pointerMove", { pointerType: "mouse", clientX: pointerX(50, D) });
    expect(ruler()).toHaveAttribute("aria-valuenow", "40");

    el.focus();
    for (const key of ["ArrowRight", "ArrowLeft", "Home", "End", "Enter"]) {
      fireEvent.keyDown(el, { key });
      expect(ruler()).toHaveAttribute("aria-valuenow", "40");
    }
    expect(seek).not.toHaveBeenCalled();
  });

  it("SPEC-ADDON-129 I4 — while the first answer is in flight the ruler, if drawn, has aria-valuemax 0 and no time labels", async () => {
    const pending = deferred<ReturnType<typeof answer>>();
    vi.mocked(getClipTimestamps).mockReturnValueOnce(pending.promise as never);
    vi.mocked(getVideoVisualIndex).mockResolvedValue(withScenes(scenes));
    renderSection({
      mimeType: LOFT_MIME,
      active: true,
      labelledByHost: true,
      onAvailability: vi.fn(),
      mediaController: makeController({ duration: D }).controller,
    });
    await flush(8);

    const slider = screen.queryByRole("slider");
    if (slider) {
      expect(slider).toHaveAttribute("aria-valuemax", "0");
    }
    expect(textShown("0:00")).toBe(false);
    expect(textShown(formatDuration(D))).toBe(false);

    pending.resolve(answer([]));
    await waitFor(() =>
      expect(screen.getByRole("slider")).toHaveAttribute(
        "aria-valuemax",
        String(Math.round(D)),
      ),
    );
  });
});

describe("SPEC-ADDON-129 §3 — ruler length transitions with no candidates", () => {
  it("SPEC-ADDON-129 §3 — L follows the player's duration in place, with no new request", async () => {
    vi.useFakeTimers();
    vi.mocked(getClipTimestamps).mockResolvedValue(answer([]));
    vi.mocked(getVideoVisualIndex).mockResolvedValue(withScenes());
    const { controller, state } = makeController({ duration: 0, currentTime: 10 });
    renderSection({ mimeType: LOFT_MIME, mediaController: controller });
    await flush(8);
    fireEvent.click(header());
    await flush(8);
    await advance(500);
    await flush(8);
    const calls = vi.mocked(getClipTimestamps).mock.calls.length;
    expect(ruler()).toHaveAttribute("aria-valuemax", "0");

    await advance(1000, () => {
      state.duration = D;
    });
    expect(ruler()).toHaveAttribute("aria-valuemax", "83");

    await advance(1000, () => {
      state.duration = 100.6;
    });
    expect(ruler()).toHaveAttribute("aria-valuemax", "101");

    for (const duration of [Number.NaN, Number.POSITIVE_INFINITY, -5, 0]) {
      await advance(1000, () => {
        state.duration = duration;
      });
      expect(ruler()).toHaveAttribute("aria-valuemax", "0");
      expect(textShown("0:00")).toBe(false);
    }

    await advance(1000, () => {
      state.duration = 60.3;
    });
    expect(ruler()).toHaveAttribute("aria-valuemax", "60");
    expect(getClipTimestamps).toHaveBeenCalledTimes(calls);
  });

  it("SPEC-ADDON-129 §3 — an answer with candidates takes the candidate rule, and a later one with none goes back to D", async () => {
    vi.mocked(getVideoVisualIndex).mockResolvedValue(withScenes());
    vi.mocked(getClipTimestamps)
      .mockResolvedValueOnce(answer([], CHANGES, "f1"))
      .mockResolvedValueOnce(answer(makeRows(3), CHANGES, "f2"))
      .mockResolvedValueOnce(answer([], CHANGES, "f3"));
    const controller = makeController({ duration: D }).controller;
    const view = renderSection({ mimeType: LOFT_MIME, mediaController: controller });
    await openLoft();
    await waitFor(() => expect(ruler()).toHaveAttribute("aria-valuemax", "83"));

    controller.getDuration = () => 5;
    view.rerender(section({ fileId: "f2", mimeType: LOFT_MIME, mediaController: controller }));
    await waitFor(() => expect(getClipTimestamps).toHaveBeenCalledTimes(2));
    await expanded();
    await waitFor(() => expect(ruler()).toHaveAttribute("aria-valuemax", "15"));

    view.rerender(section({ fileId: "f3", mimeType: LOFT_MIME, mediaController: controller }));
    await waitFor(() => expect(getClipTimestamps).toHaveBeenCalledTimes(3));
    await expanded();
    await waitFor(() => expect(ruler()).toHaveAttribute("aria-valuemax", "5"));
  });
});

describe("SPEC-ADDON-129 I5 — the switch's message keys are gone", () => {
  it.each([
    { file: "en.json", messages: enMessages as unknown },
    { file: "ja.json", messages: jaMessages as unknown },
  ])("SPEC-ADDON-129 I5 — $file has none of the three keys", ({ messages }) => {
    const present = keysIn(messages).filter((key) =>
      REMOVED_KEYS.some((removed) => key === removed || key.endsWith(`.${removed}`)),
    );
    expect(present).toEqual([]);
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

describe("SPEC-ADDON-129 kept cases — behaviour the switch's removal keeps", () => {
  it("SPEC-ADDON-129 kept — a .loft with change frames, no candidates and no Visual Index is not drawn", async () => {
    vi.mocked(getVideoVisualIndex).mockResolvedValue(indexState({ eligible: false }));
    vi.mocked(getClipTimestamps).mockResolvedValue(answer([], CHANGES));
    const view = renderSection({ mimeType: LOFT_MIME });
    await waitFor(() => expect(getVideoVisualIndex).toHaveBeenCalledTimes(1));
    await flush(8);

    expect(view.container.firstChild).toBeNull();
  });

  it("SPEC-ADDON-129 kept — a native video with change frames and no candidates is never available", async () => {
    const report = vi.fn();
    vi.mocked(getClipTimestamps).mockResolvedValue(answer([], CHANGES));
    vi.mocked(getVideoVisualIndex).mockResolvedValue(withScenes());
    const view = renderSection({
      labelledByHost: true,
      active: true,
      onAvailability: report,
      mediaController: makeController({ duration: D }).controller,
    });
    await waitFor(() => expect(getClipTimestamps).toHaveBeenCalled());
    await flush(8);

    expect(report.mock.calls.map(([available]) => available)).not.toContain(true);
    expect(view.container.querySelector('[role="slider"]')).toBeNull();
  });

  it("SPEC-ADDON-129 kept — a .loft tile whose picture fails keeps its time and seek and shows the empty block", async () => {
    const { controller, seek } = makeController({ duration: 70 });
    renderSection({ mimeType: LOFT_MIME, mediaController: controller });
    await openLoft();
    await waitFor(() => expect(imgTimes()).toContain(10));

    const failed = images().find((img) => /[?&]t=10(?:&|$)/.test(img.src))!;
    fireEvent.error(failed);

    expect(tile(10).querySelector("img")).toBeNull();
    expect(tile(10)).toHaveAttribute("aria-label", "Play from 0:10");
    expect(imgTimes()).toEqual([0, 5, 15, 20, 25, 30, 35, 40]);
    fireEvent.click(tile(10));
    expect(seek).toHaveBeenCalledWith(10);
  });

  it("SPEC-ADDON-129 kept — a .loft draws its candidate tiles from the scene-tile route", async () => {
    renderSection({ mimeType: LOFT_MIME, mediaController: makeController({ duration: 70 }).controller });
    await openLoft();

    await waitFor(() => expect(imgTimes()).toEqual([0, 5, 10, 15, 20, 25, 30, 35, 40]));
    for (const img of images()) {
      expect(img.getAttribute("src")).toContain("/scene-tile/f1?t=");
    }
    for (const time of CHANGE_TIMES) {
      expect(getSceneTileUrl).not.toHaveBeenCalledWith("f1", time);
    }
    expect(getFrameUrl).not.toHaveBeenCalled();
  });

  it("SPEC-ADDON-129 kept — L is the candidates' last start plus the median gap when the duration is unknown", async () => {
    renderSection({ mediaController: makeController({ duration: 0 }).controller });
    await openNative();
    await waitForVisualIndex();

    expect(ruler()).toHaveAttribute("aria-valuemax", "75");
    expect(tileLabels()).toEqual(labels(CANDIDATES.map((row) => row.start)));
  });

  it.each([
    { name: "inside a tile's range", rows: CANDIDATES, seek: 22, dotted: "Play from 0:20" },
    { name: "at the next frame's start", rows: CANDIDATES, seek: 25, dotted: "Play from 0:25" },
    { name: "just before the next frame's start", rows: CANDIDATES, seek: 24.9, dotted: "Play from 0:20" },
    { name: "before the first frame", rows: makeRows(15, 5, 3), seek: 1, dotted: "Play from 0:03" },
  ])(
    "SPEC-ADDON-090 I9 as revised by SPEC-ADDON-129 — the dot marks the candidate tile whose range holds section_start, $name",
    async ({ rows, seek, dotted }) => {
      vi.mocked(getClipTimestamps).mockResolvedValue(answer(rows, CHANGES));
      vi.mocked(getVideoVisualIndex).mockResolvedValue(
        withScenes([makeScene({ start_time: seek, section_start: seek, scene_label: "Alpha" })]),
      );
      renderSection({ mediaController: makeController({ duration: 70 }).controller });
      await openNative();
      await waitForVisualIndex();

      expect(dottedTiles()).toEqual([dotted]);
    },
  );

  it("SPEC-ADDON-104 I3 as revised by SPEC-ADDON-129 — header, ruler, grid, scene card, action row, with nothing between the ruler and the grid", async () => {
    vi.mocked(getVideoVisualIndex).mockResolvedValue(withScenes());
    renderSection({ mediaController: makeController({ duration: 70 }).controller });
    await openNative();
    await waitForVisualIndex();
    await waitFor(() => expect(sceneCardNamed(/Alpha/)).not.toBeNull());

    const follows = (a: Node, b: Node) =>
      (a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING) !== 0;
    const tiles = allTiles();
    const card = sceneCardNamed(/Alpha/)!;
    const regen = screen.getByRole("button", { name: "Regenerate scenes" });
    expect(follows(header(), ruler())).toBe(true);
    expect(follows(ruler(), tiles[0])).toBe(true);
    expect(follows(tiles[tiles.length - 1], card)).toBe(true);
    expect(follows(card, regen)).toBe(true);

    const between = screen
      .queryAllByRole("button")
      .filter((el) => follows(ruler(), el) && follows(el, tiles[0]));
    expect(between).toEqual([]);
    expect(switchTraces()).toEqual([]);
  });
});
