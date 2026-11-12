import "./intlMock";
import { describe, it, expect, vi, beforeEach, beforeAll } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";

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

const wsEvent = vi.hoisted(() => ({
  value: null as null | { event: string; data: Record<string, unknown> },
}));

vi.mock("@/hooks/useWebSocket", () => ({
  useWebSocket: vi.fn((filter?: string) => {
    const event = wsEvent.value;
    if (!event) return null;
    if (filter && event.event !== filter) return null;
    return event;
  }),
}));

import ScenesSection from "@/addons/intelligence/ScenesSection";
import {
  generateVideoVisualIndex,
  getClipTimestamps,
  getVideoVisualIndex,
  retryVideoVisualIndex,
} from "@/addons/intelligence/api";
import { usePolicy } from "@/hooks/usePolicy";
import {
  action,
  actionNames,
  allTiles,
  classTokens,
  flush,
  header,
  indexState,
  installPointerEvents,
  isFilledAction,
  makeController,
  makeRows,
  makeRun,
  makeScene,
  nameVisible,
  ruler,
  SceneSectionProps,
  sceneCardNamed,
  tile,
  timestamps,
  waitForVisualIndex,
} from "./helpers";

const STALE = /changed since this index was built/;
const WAITING = /Waiting on scene indexing to finish/;
const ACTION_ERROR = /Could not start visual index generation/;

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

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
}

async function open(): Promise<void> {
  fireEvent.click(header());
  await screen.findByRole("slider");
}

function marksPresent(): boolean {
  return /(?:border-(?:accent-teal|accent|danger)|bg-(?:accent-teal|accent|danger)\/|bg-accent-teal\b)/.test(
    classTokens(ruler()),
  );
}

function tileDot(el: HTMLElement): boolean {
  return el.querySelector('[class*="bg-accent-teal"]') !== null;
}

const labelledScenes = [
  makeScene({
    ordering: 0,
    start_time: 10,
    scene_label: "Alpha",
    visible_text: "Alpha text",
  }),
  makeScene({ ordering: 1, start_time: 30, scene_label: "Beta" }),
];

beforeAll(installPointerEvents);

beforeEach(() => {
  vi.clearAllMocks();
  wsEvent.value = null;
  vi.mocked(getClipTimestamps).mockResolvedValue(timestamps(makeRows(15)));
  vi.mocked(getVideoVisualIndex).mockResolvedValue(indexState());
  vi.mocked(generateVideoVisualIndex).mockResolvedValue({
    status: "accepted",
    file_id: "f1",
  });
  vi.mocked(retryVideoVisualIndex).mockResolvedValue({
    status: "accepted",
    file_id: "f1",
  });
  vi.mocked(usePolicy).mockReturnValue({ enabled: true, isLoading: false });
});

describe("SPEC-ADDON-090 I10 — the Visual Index state table", () => {
  it.each([
    {
      row: "1 — a first run is queued or running",
      state: indexState({
        staged_run: makeRun({
          status: "running",
          selected_count: 12,
          completed_count: 7,
          succeeded_count: 7,
        }),
      }),
      count: null,
      progress: /Processing\s+7\s*\/\s*12/,
      card: false,
      actions: [],
      marks: false,
      note: null,
    },
    {
      row: "2 — a staged run over an active run",
      state: indexState({
        active_run: makeRun({ succeeded_count: 2 }),
        staged_run: makeRun({
          status: "queued",
          selected_count: 10,
          completed_count: 3,
          succeeded_count: 3,
        }),
        scenes: labelledScenes,
      }),
      count: "2 scenes",
      progress: /Updating\s+3\s*\/\s*10/,
      card: true,
      actions: [],
      marks: true,
      note: null,
    },
    {
      row: "3 — failures with no active run",
      state: indexState({
        staged_run: makeRun({
          status: "partial",
          selected_count: 6,
          completed_count: 6,
          succeeded_count: 5,
          failed_count: 2,
        }),
      }),
      count: null,
      progress: null,
      card: false,
      actions: ["Retry the 2 failed", "Create scenes"],
      marks: false,
      note: "2 failed",
    },
    {
      row: "3 — failures over an active run",
      state: indexState({
        active_run: makeRun({
          status: "partial",
          succeeded_count: 4,
          failed_count: 1,
        }),
        scenes: labelledScenes,
      }),
      count: "4 scenes",
      progress: null,
      card: true,
      actions: ["Retry the 1 failed", "Regenerate scenes"],
      marks: true,
      note: "1 failed",
    },
    {
      row: "4 — an active run",
      state: indexState({
        active_run: makeRun({ succeeded_count: 2 }),
        scenes: labelledScenes,
      }),
      count: "2 scenes",
      progress: null,
      card: true,
      actions: ["Regenerate scenes"],
      marks: true,
      note: null,
    },
    {
      row: "5 — nothing staged and no active run",
      state: indexState(),
      count: null,
      progress: null,
      card: false,
      actions: ["Create scenes"],
      marks: false,
      note: null,
    },
  ])(
    "SPEC-ADDON-090 I10 — row $row draws exactly its parts",
    async ({ state, count, progress, card, actions, marks, note }) => {
      vi.mocked(getVideoVisualIndex).mockResolvedValue(state);
      renderSection();
      await open();
      await waitForVisualIndex();

      if (count === null) {
        expect(nameVisible(/^\d+ scenes$/)).toBe(false);
      } else {
        expect(nameVisible(new RegExp(`^${count}$`))).toBe(true);
      }
      if (progress) expect(nameVisible(progress)).toBe(true);
      if (note) expect(nameVisible(new RegExp(`^${note}$`))).toBe(true);
      expect(marksPresent()).toBe(marks);
      if (card) {
        expect(sceneCardNamed(/Alpha/)).not.toBeNull();
      } else {
        expect(sceneCardNamed(/Alpha/)).toBeNull();
      }
      expect(actionNames().sort()).toEqual([...actions].sort());
    },
  );

  it("SPEC-ADDON-090 I10 — a queued update wins over a retryable active run", async () => {
    vi.mocked(getVideoVisualIndex).mockResolvedValue(
      indexState({
        active_run: makeRun({ status: "partial", failed_count: 2 }),
        staged_run: makeRun({
          status: "running",
          selected_count: 5,
          completed_count: 2,
        }),
        scenes: labelledScenes,
      }),
    );
    renderSection();
    await open();
    await waitForVisualIndex();

    expect(nameVisible(/Updating\s+2\s*\/\s*5/)).toBe(true);
    expect(actionNames()).toEqual([]);
  });

  it.each([
    {
      name: "the staged partial run's failure count",
      state: indexState({
        active_run: makeRun({ status: "partial", failed_count: 1 }),
        staged_run: makeRun({ status: "partial", failed_count: 3 }),
        scenes: labelledScenes,
      }),
      retry: "Retry the 3 failed",
    },
    {
      name: "the active partial run when nothing is staged",
      state: indexState({
        active_run: makeRun({ status: "partial", failed_count: 2 }),
        scenes: labelledScenes,
      }),
      retry: "Retry the 2 failed",
    },
    {
      name: "no retry when a failed staged run has none",
      state: indexState({
        active_run: makeRun({ status: "succeeded", failed_count: 0 }),
        staged_run: makeRun({ status: "failed", failed_count: 0 }),
        scenes: labelledScenes,
      }),
      retry: null,
    },
    {
      name: "no retry when a failed staged run has none even over a partial active run",
      state: indexState({
        active_run: makeRun({ status: "partial", failed_count: 2 }),
        staged_run: makeRun({ status: "failed", failed_count: 0 }),
        scenes: labelledScenes,
      }),
      retry: null,
    },
  ])(
    "SPEC-ADDON-090 I10 — N comes from $name",
    async ({ state, retry }) => {
      vi.mocked(getVideoVisualIndex).mockResolvedValue(state);
      renderSection();
      await open();
      await waitForVisualIndex();

      if (retry === null) {
        expect(action("Retry the 0 failed")).toBeNull();
        expect(actionNames()).toEqual(["Regenerate scenes"]);
      } else {
        expect(action(retry)).not.toBeNull();
      }
    },
  );

  it.each([
    { label: "Processing", state: indexState({
      staged_run: makeRun({ status: "running", selected_count: 0, completed_count: 0 }),
    }) },
    { label: "Updating", state: indexState({
      active_run: makeRun(),
      staged_run: makeRun({ status: "queued", selected_count: 0, completed_count: 0 }),
    }) },
  ])(
    "SPEC-ADDON-090 I10 — $label shows an ellipsis while selected_count is zero",
    async ({ label, state }) => {
      vi.mocked(getVideoVisualIndex).mockResolvedValue(state);
      renderSection();
      await open();
      await waitForVisualIndex();

      expect(nameVisible(new RegExp(`${label}\\s+…`))).toBe(true);
    },
  );
});

describe("SPEC-ADDON-090 I10 — action emphasis", () => {
  it.each([
    { label: "Create scenes", state: indexState() },
    {
      label: "Retry the 2 failed",
      state: indexState({
        staged_run: makeRun({ status: "partial", failed_count: 2 }),
      }),
    },
  ])("fills $label", async ({ label, state }) => {
    vi.mocked(getVideoVisualIndex).mockResolvedValue(state);
    renderSection();
    await open();
    await waitForVisualIndex();

    expect(isFilledAction(action(label)!)).toBe(true);
  });
});

describe("SPEC-ADDON-090 I10 — the stale line", () => {
  it("shows the line and a filled Regenerate in row 4", async () => {
    vi.mocked(getVideoVisualIndex).mockResolvedValue(
      indexState({
        active_run: makeRun({ succeeded_count: 2 }),
        scenes: labelledScenes,
        stale: true,
      }),
    );
    renderSection();
    await open();
    await waitForVisualIndex();

    expect(nameVisible(STALE)).toBe(true);
    expect(isFilledAction(action("Regenerate scenes")!)).toBe(true);
  });

  it("keeps Regenerate ghost when row 4 is not stale", async () => {
    vi.mocked(getVideoVisualIndex).mockResolvedValue(
      indexState({
        active_run: makeRun({ succeeded_count: 2 }),
        scenes: labelledScenes,
      }),
    );
    renderSection();
    await open();
    await waitForVisualIndex();

    expect(nameVisible(STALE)).toBe(false);
    expect(isFilledAction(action("Regenerate scenes")!)).toBe(false);
  });

  it("shows the line in row 3", async () => {
    vi.mocked(getVideoVisualIndex).mockResolvedValue(
      indexState({
        active_run: makeRun({ status: "partial", failed_count: 1 }),
        scenes: labelledScenes,
        stale: true,
      }),
    );
    renderSection();
    await open();
    await waitForVisualIndex();

    expect(nameVisible(STALE)).toBe(true);
  });

  it.each([
    {
      name: "row 1",
      state: indexState({
        staged_run: makeRun({ status: "running" }),
        stale: true,
      }),
    },
    {
      name: "row 2",
      state: indexState({
        active_run: makeRun(),
        staged_run: makeRun({ status: "queued" }),
        scenes: labelledScenes,
        stale: true,
      }),
    },
  ])("never shows the line in $name", async ({ state }) => {
    vi.mocked(getVideoVisualIndex).mockResolvedValue(state);
    renderSection();
    await open();
    await waitForVisualIndex();

    expect(nameVisible(STALE)).toBe(false);
  });
});

describe("SPEC-ADDON-090 I9 — scene marks and tile dots", () => {
  it.each([
    { status: "succeeded" as const },
    { status: "pending" as const },
    { status: "running" as const },
    { status: "failed" as const },
  ])(
    "SPEC-ADDON-090 I9 — a $status mark is styled by its status",
    async ({ status }) => {
      vi.mocked(getVideoVisualIndex).mockResolvedValue(
        indexState({
          active_run: makeRun({ succeeded_count: status === "succeeded" ? 1 : 0 }),
          scenes: [makeScene({ start_time: 10, status })],
        }),
      );
      renderSection();
      await open();
      await waitForVisualIndex();

      const tokens = classTokens(ruler());
      if (status === "succeeded") {
        expect(tokens).toMatch(/\bbg-accent-teal\b/);
        expect(tokens).not.toMatch(/border-accent-teal/);
        return;
      }
      if (status === "failed") {
        expect(tokens).toMatch(
          /(?:^|\s)border-(?:accent|danger)(?:\/\d+)?(?=\s|$)/,
        );
        expect(tokens).toMatch(
          /(?:^|\s)bg-(?:accent|danger)\/\d+/,
        );
        return;
      }
      expect(tokens).toMatch(/border-accent-teal/);
      expect(tokens).toMatch(/bg-accent-teal\//);
    },
  );

  it.each([
    { name: "exactly 0.05 s away", start: 0.05, dot: true },
    { name: "0.051 s away", start: 0.051, dot: false },
  ])(
    "SPEC-ADDON-090 I9 — a tile carries the dot when the scene is $name",
    async ({ start, dot }) => {
      vi.mocked(getVideoVisualIndex).mockResolvedValue(
        indexState({
          active_run: makeRun({ succeeded_count: 1 }),
          scenes: [makeScene({ start_time: start })],
        }),
      );
      renderSection();
      await open();
      await waitForVisualIndex();

      expect(tileDot(tile(0))).toBe(dot);
    },
  );

  it("still marks a scene that matches no frame, and no tile carries its dot", async () => {
    vi.mocked(getVideoVisualIndex).mockResolvedValue(
      indexState({
        active_run: makeRun({ succeeded_count: 1 }),
        scenes: [makeScene({ start_time: 12.2 })],
      }),
    );
    renderSection();
    await open();
    await waitForVisualIndex();

    expect(marksPresent()).toBe(true);
    expect(allTiles().some(tileDot)).toBe(false);
  });

  it("skips an unlabelled scene in the card but still marks it and its tile", async () => {
    vi.mocked(getVideoVisualIndex).mockResolvedValue(
      indexState({
        active_run: makeRun({ succeeded_count: 1 }),
        scenes: [makeScene({ start_time: 10, scene_label: null })],
      }),
    );
    renderSection();
    await open();
    await waitForVisualIndex();

    expect(
      screen.queryByRole("button", {
        name: /^(?!Scenes|Play from |Create scenes|Regenerate scenes|Retry the ).+/,
      }),
    ).toBeNull();
    expect(marksPresent()).toBe(true);
    expect(tileDot(tile(10))).toBe(true);
  });
});

describe("SPEC-ADDON-090 I9 — the scene card", () => {
  const cardScenes = [
    makeScene({
      ordering: 0,
      start_time: 10,
      scene_label: "Alpha",
      visible_text: "Alpha text",
      transcript_excerpt: "SECRET-A",
    }),
    makeScene({ ordering: 1, start_time: 30, scene_label: "Beta" }),
    makeScene({ ordering: 2, start_time: 40, scene_label: null }),
    makeScene({
      ordering: 3,
      start_time: 50,
      status: "failed",
      scene_label: "Failed scene",
    }),
    makeScene({ ordering: 4, start_time: 60, scene_label: "Gamma" }),
  ];

  it.each([
    { time: 0, label: "Alpha", why: "before every scene" },
    { time: 25, label: "Alpha", why: "at or before the point" },
    { time: 45, label: "Beta", why: "skipping the unlabelled scene" },
    { time: 55, label: "Beta", why: "skipping the failed scene" },
    { time: 65, label: "Gamma", why: "the last one at or before the point" },
  ])(
    "SPEC-ADDON-090 I9 — shows $label when the point is $why",
    async ({ time, label }) => {
      const { controller } = makeController({ duration: 70, currentTime: time });
      vi.mocked(getVideoVisualIndex).mockResolvedValue(
        indexState({
          active_run: makeRun({ succeeded_count: 3 }),
          scenes: cardScenes,
        }),
      );
      renderSection({ mediaController: controller });
      await open();
      await waitForVisualIndex();

      expect(sceneCardNamed(new RegExp(label))).not.toBeNull();
    },
  );

  it("shows visible_text and never transcript_excerpt", async () => {
    vi.mocked(getVideoVisualIndex).mockResolvedValue(
      indexState({
        active_run: makeRun({ succeeded_count: 3 }),
        scenes: cardScenes,
      }),
    );
    renderSection();
    await open();
    await waitForVisualIndex();

    expect(screen.getByText("Alpha text")).toBeInTheDocument();
    expect(screen.queryByText("SECRET-A")).toBeNull();
  });

  it("seeks to the scene's start when activated", async () => {
    const { controller, seek, play } = makeController({
      duration: 70,
      currentTime: 45,
    });
    vi.mocked(getVideoVisualIndex).mockResolvedValue(
      indexState({
        active_run: makeRun({ succeeded_count: 3 }),
        scenes: cardScenes,
      }),
    );
    renderSection({ mediaController: controller });
    await open();
    await waitForVisualIndex();

    fireEvent.click(sceneCardNamed(/Beta/)!);

    expect(seek).toHaveBeenCalledWith(30);
    expect(play).toHaveBeenCalled();
  });
});

describe("SPEC-ADDON-090 I9 — no Visual Index", () => {
  it.each([
    {
      name: "the policy is off",
      configure: () => {
        vi.mocked(usePolicy).mockReturnValue({ enabled: false, isLoading: false });
        vi.mocked(getVideoVisualIndex).mockResolvedValue(
          indexState({
            active_run: makeRun({ succeeded_count: 1 }),
            scenes: [makeScene({ scene_label: "Alpha" })],
          }),
        );
      },
      requested: false,
    },
    {
      name: "the request failed or answered 404",
      configure: () => {
        vi.mocked(getVideoVisualIndex).mockResolvedValue(null);
      },
      requested: true,
    },
    {
      name: "the answer is eligible: false",
      configure: () => {
        vi.mocked(getVideoVisualIndex).mockResolvedValue(
          indexState({
            eligible: false,
            active_run: makeRun({ succeeded_count: 1 }),
            scenes: [makeScene({ scene_label: "Alpha" })],
          }),
        );
      },
      requested: true,
    },
    {
      name: "the answer is available: false",
      configure: () => {
        vi.mocked(getVideoVisualIndex).mockResolvedValue(
          indexState({
            available: false,
            active_run: makeRun({ succeeded_count: 1 }),
            scenes: [makeScene({ scene_label: "Alpha" })],
          }),
        );
      },
      requested: true,
    },
  ])(
    "SPEC-ADDON-090 I9 — draws no Visual Index part when $name",
    async ({ configure, requested }) => {
      configure();
      renderSection();
      await open();

      expect(nameVisible(/^\d+ scenes$/)).toBe(false);
      expect(nameVisible(/(Processing|Updating)/)).toBe(false);
      expect(marksPresent()).toBe(false);
      expect(allTiles().some(tileDot)).toBe(false);
      expect(sceneCardNamed(/Alpha/)).toBeNull();
      expect(actionNames()).toEqual([]);
      expect(screen.getByRole("slider")).toBeInTheDocument();
      expect(allTiles()).toHaveLength(9);

      if (requested) {
        expect(getVideoVisualIndex).toHaveBeenCalledTimes(1);
      } else {
        expect(getVideoVisualIndex).not.toHaveBeenCalled();
      }
    },
  );

  it("keeps the timeline while policy and visual-index are still loading", async () => {
    const pending = deferred<ReturnType<typeof indexState>>();
    vi.mocked(usePolicy).mockReturnValue({ enabled: true, isLoading: true });
    vi.mocked(getVideoVisualIndex).mockReturnValueOnce(pending.promise);

    const view = renderSection();
    await open();

    expect(getVideoVisualIndex).not.toHaveBeenCalled();
    expect(nameVisible(/^\d+ scenes$/)).toBe(false);
    expect(marksPresent()).toBe(false);
    expect(actionNames()).toEqual([]);

    vi.mocked(usePolicy).mockReturnValue({ enabled: true, isLoading: false });
    view.rerender(section());
    await waitFor(() => expect(getVideoVisualIndex).toHaveBeenCalledTimes(1));

    pending.resolve(
      indexState({
        active_run: makeRun({ succeeded_count: 1 }),
        scenes: [makeScene({ start_time: 10, scene_label: "Alpha" })],
      }),
    );
    await waitFor(() => expect(nameVisible(/^1 scenes$/)).toBe(true));
    expect(marksPresent()).toBe(true);
  });
});

describe("SPEC-ADDON-090 I10 — actions", () => {
  it.each([
    {
      name: "Create scenes calls generate",
      state: indexState(),
      click: "Create scenes",
      expected: () => expect(generateVideoVisualIndex).toHaveBeenCalledWith("f1", "d1"),
    },
    {
      name: "Regenerate scenes calls generate",
      state: indexState({
        active_run: makeRun({ succeeded_count: 2 }),
        scenes: labelledScenes,
      }),
      click: "Regenerate scenes",
      expected: () => expect(generateVideoVisualIndex).toHaveBeenCalledWith("f1", "d1"),
    },
    {
      name: "Retry calls retry",
      state: indexState({
        active_run: makeRun({ status: "partial", failed_count: 1 }),
        scenes: labelledScenes,
      }),
      click: "Retry the 1 failed",
      expected: () => expect(retryVideoVisualIndex).toHaveBeenCalledWith("f1", "d1"),
    },
  ])(
    "SPEC-ADDON-090 I10 — $name",
    async ({ state, click, expected }) => {
      vi.mocked(getVideoVisualIndex).mockResolvedValue(state);
      renderSection();
      await open();
      await waitForVisualIndex();

      fireEvent.click(action(click)!);
      await waitFor(expected);
    },
  );

  it("disables every action and reads Starting… while a request is out", async () => {
    const pending = deferred<{ status: string; file_id: string }>();
    vi.mocked(getVideoVisualIndex).mockResolvedValue(
      indexState({
        active_run: makeRun({ status: "partial", failed_count: 1 }),
        scenes: labelledScenes,
      }),
    );
    vi.mocked(retryVideoVisualIndex).mockReturnValueOnce(pending.promise);
    renderSection();
    await open();
    await waitForVisualIndex();

    fireEvent.click(action("Retry the 1 failed")!);
    await flush();

    expect(screen.getByText("Starting…")).toBeInTheDocument();
    for (const name of ["Starting…", "Create scenes", "Regenerate scenes"]) {
      const button = action(name);
      if (button) expect(button).toBeDisabled();
    }

    pending.resolve({ status: "accepted", file_id: "f1" });
    await waitFor(() => expect(screen.queryByText("Starting…")).toBeNull());
  });
});

describe("SPEC-ADDON-090 I10 — action errors", () => {
  function waitingClip() {
    return Object.assign(new Error("not_queued"), {
      info: { kind: "not_queued", reason: "waiting_clip" },
    });
  }

  it("shows the prerequisite message for waiting_clip", async () => {
    vi.mocked(generateVideoVisualIndex).mockRejectedValueOnce(waitingClip());
    renderSection();
    await open();
    await waitForVisualIndex();

    fireEvent.click(action("Create scenes")!);

    await waitFor(() => expect(nameVisible(WAITING)).toBe(true));
  });

  it("shows the generic message for any other failure", async () => {
    vi.mocked(generateVideoVisualIndex).mockRejectedValueOnce(new Error("down"));
    renderSection();
    await open();
    await waitForVisualIndex();

    fireEvent.click(action("Create scenes")!);

    await waitFor(() => expect(nameVisible(ACTION_ERROR)).toBe(true));
  });

  it("replaces the note in a retry row", async () => {
    vi.mocked(getVideoVisualIndex).mockResolvedValue(
      indexState({
        active_run: makeRun({ status: "partial", failed_count: 1 }),
        scenes: labelledScenes,
      }),
    );
    vi.mocked(retryVideoVisualIndex).mockRejectedValueOnce(waitingClip());
    renderSection();
    await open();
    await waitForVisualIndex();

    fireEvent.click(action("Retry the 1 failed")!);

    await waitFor(() => expect(nameVisible(WAITING)).toBe(true));
    expect(nameVisible(/^1 failed$/)).toBe(false);
  });

  it.each([
    { name: "the next action", clear: "action" },
    { name: "a reload", clear: "reload" },
    { name: "collapse", clear: "collapse" },
    { name: "a file change", clear: "file" },
  ])(
    "SPEC-ADDON-090 I10 — the message clears on $name",
    async ({ clear }) => {
      vi.mocked(generateVideoVisualIndex).mockRejectedValueOnce(waitingClip());
      const view = renderSection();
      await open();
      await waitForVisualIndex();

      fireEvent.click(action("Create scenes")!);
      await waitFor(() => expect(nameVisible(WAITING)).toBe(true));

      if (clear === "action") {
        fireEvent.click(action("Create scenes")!);
      } else if (clear === "reload") {
        wsEvent.value = {
          event: "intelligence.video_visual.succeeded",
          data: { file_id: "f1" },
        };
        view.rerender(section());
      } else if (clear === "collapse") {
        fireEvent.click(header());
        fireEvent.click(header());
        await screen.findByRole("slider");
      } else {
        view.rerender(section({ fileId: "f2" }));
        await open();
        await waitForVisualIndex();
      }

      await waitFor(() => expect(nameVisible(WAITING)).toBe(false));
    },
  );
});

describe("SPEC-ADDON-090 I10 — a failed reload", () => {
  it("keeps the last good state on screen", async () => {
    vi.mocked(getVideoVisualIndex)
      .mockResolvedValueOnce(
        indexState({
          active_run: makeRun({ succeeded_count: 2 }),
          scenes: labelledScenes,
        }),
      )
      .mockResolvedValueOnce(null);
    renderSection();
    await open();
    await waitForVisualIndex();

    expect(nameVisible(/^2 scenes$/)).toBe(true);
    expect(sceneCardNamed(/Alpha/)).not.toBeNull();

    fireEvent.click(action("Regenerate scenes")!);
    await waitFor(() => expect(getVideoVisualIndex).toHaveBeenCalledTimes(2));
    await flush();

    expect(nameVisible(/^2 scenes$/)).toBe(true);
    expect(sceneCardNamed(/Alpha/)).not.toBeNull();
  });
});
