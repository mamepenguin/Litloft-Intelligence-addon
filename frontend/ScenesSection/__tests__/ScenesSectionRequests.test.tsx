import "./intlMock";
import { describe, it, expect, vi, beforeEach, afterEach, beforeAll } from "vitest";
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
  getClipTimestamps,
  getFrameUrl,
  getSceneTileUrl,
  getVideoVisualIndex,
  type VideoVisualIndexResponse,
} from "@/addons/intelligence/api";
import { usePolicy } from "@/hooks/usePolicy";
import {
  actionNames,
  advance,
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

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

async function openNative(): Promise<void> {
  fireEvent.click(header());
  await screen.findByRole("slider");
}

function loft(props: Partial<SceneSectionProps> = {}) {
  return { mimeType: LOFT_MIME, ...props };
}

beforeAll(installPointerEvents);

beforeEach(() => {
  vi.clearAllMocks();
  wsEvent.value = null;
  vi.mocked(getClipTimestamps).mockResolvedValue(timestamps(makeRows(15)));
  vi.mocked(getVideoVisualIndex).mockResolvedValue(indexState());
  vi.mocked(usePolicy).mockReturnValue({ enabled: true, isLoading: false });
});

afterEach(() => {
  vi.useRealTimers();
});

describe("SPEC-ADDON-089 I6 — native request timing", () => {
  it("requests nothing before the first expand", async () => {
    renderSection();
    await flush();

    expect(getClipTimestamps).not.toHaveBeenCalled();
    expect(getVideoVisualIndex).not.toHaveBeenCalled();
  });

  it("requests clip-timestamps and visual-index on every expand", async () => {
    renderSection();

    await openNative();
    expect(getClipTimestamps).toHaveBeenCalledTimes(1);
    expect(getVideoVisualIndex).toHaveBeenCalledTimes(1);

    fireEvent.click(header());
    await openNative();

    expect(getClipTimestamps).toHaveBeenCalledTimes(2);
    expect(getVideoVisualIndex).toHaveBeenCalledTimes(2);
  });

  it("never requests visual-index when the policy is off", async () => {
    vi.mocked(usePolicy).mockReturnValue({ enabled: false, isLoading: false });
    renderSection();
    await openNative();

    expect(getClipTimestamps).toHaveBeenCalledTimes(1);
    expect(getVideoVisualIndex).not.toHaveBeenCalled();
  });

  it("requests visual-index once the loading policy resolves on", async () => {
    vi.mocked(usePolicy).mockReturnValue({ enabled: true, isLoading: true });
    const view = renderSection();

    await openNative();
    expect(getClipTimestamps).toHaveBeenCalledTimes(1);
    expect(getVideoVisualIndex).not.toHaveBeenCalled();

    vi.mocked(usePolicy).mockReturnValue({ enabled: true, isLoading: false });
    view.rerender(section());

    await waitFor(() => expect(getVideoVisualIndex).toHaveBeenCalledTimes(1));
  });

  it("discards a timestamps answer for an earlier file", async () => {
    const first = deferred<ReturnType<typeof timestamps>>();
    const second = deferred<ReturnType<typeof timestamps>>();
    vi.mocked(getClipTimestamps)
      .mockReturnValueOnce(first.promise)
      .mockReturnValueOnce(second.promise);

    const view = renderSection();
    await openNative();

    view.rerender(section({ fileId: "f2" }));
    expect(header()).toHaveAttribute("aria-expanded", "false");
    fireEvent.click(header());
    second.resolve(
      timestamps(
        [
          { start: 40, content_preview: "Frame at 40s" },
          { start: 45, content_preview: "Frame at 45s" },
        ],
        "f2",
      ),
    );
    await screen.findByRole("slider");
    await waitFor(() => expect(tileLabels()).toEqual(["0:40", "0:45"]));

    first.resolve(
      timestamps(
        [
          { start: 10, content_preview: "Frame at 10s" },
          { start: 15, content_preview: "Frame at 15s" },
        ],
        "f1",
      ),
    );
    await flush();

    expect(tileLabels()).toEqual(["0:40", "0:45"]);
  });

  it("discards a visual-index answer for an earlier file", async () => {
    const first = deferred<VideoVisualIndexResponse>();
    const second = deferred<VideoVisualIndexResponse>();
    vi.mocked(getVideoVisualIndex)
      .mockReturnValueOnce(first.promise)
      .mockReturnValueOnce(second.promise);

    const view = renderSection();
    fireEvent.click(header());
    await screen.findByRole("slider");

    view.rerender(section({ fileId: "f2" }));
    fireEvent.click(header());
    await screen.findByRole("slider");
    expect(getVideoVisualIndex).toHaveBeenCalledTimes(2);

    second.resolve(indexState());
    await flush();
    first.resolve(
      indexState({
        active_run: makeRun({ succeeded_count: 1 }),
        scenes: [makeScene({ start_time: 10, scene_label: "Earlier" })],
      }),
    );
    await flush();

    expect(sceneCardNamed(/Earlier/)).toBeNull();
    expect(nameVisible(/^1 scenes$/)).toBe(false);
  });

  it("discards a visual-index answer that arrives after collapse", async () => {
    const pending = deferred<VideoVisualIndexResponse>();
    vi.mocked(getVideoVisualIndex).mockReturnValueOnce(pending.promise);
    renderSection();
    fireEvent.click(header());
    await screen.findByRole("slider");

    fireEvent.click(header());
    pending.resolve(
      indexState({
        active_run: makeRun({ succeeded_count: 1 }),
        scenes: [makeScene({ start_time: 10, scene_label: "Later" })],
      }),
    );
    await flush();
    expect(screen.queryByRole("slider")).toBeNull();

    fireEvent.click(header());
    await screen.findByRole("slider");
    expect(sceneCardNamed(/Later/)).toBeNull();
    expect(nameVisible(/^1 scenes$/)).toBe(false);
  });

  it("discards a timestamps answer that arrives after collapse", async () => {
    const pending = deferred<ReturnType<typeof timestamps>>();
    vi.mocked(getClipTimestamps).mockReturnValueOnce(pending.promise);

    renderSection();
    fireEvent.click(header());
    fireEvent.click(header());
    expect(header()).toHaveAttribute("aria-expanded", "false");

    pending.resolve(timestamps(makeRows(3)));
    await flush();

    expect(screen.queryByRole("slider")).toBeNull();
    expect(header()).toHaveAttribute("aria-expanded", "false");
  });

  it.each([
    "intelligence.video_visual.progress",
    "intelligence.video_visual.succeeded",
    "intelligence.video_visual.partial",
    "intelligence.video_visual.failed",
  ])(
    "SPEC-ADDON-089 I6 — %s reloads only while expanded and for this file",
    async (eventName) => {
      const { controller } = makeController({ duration: 70 });
      const view = renderSection({ mediaController: controller });
      await openNative();
      expect(getVideoVisualIndex).toHaveBeenCalledTimes(1);

      wsEvent.value = { event: eventName, data: { file_id: "other" } };
      view.rerender(section({ mediaController: controller }));
      await flush();
      expect(getVideoVisualIndex).toHaveBeenCalledTimes(1);

      wsEvent.value = { event: eventName, data: { file_id: "f1" } };
      view.rerender(section({ mediaController: controller }));
      await waitFor(() => expect(getVideoVisualIndex).toHaveBeenCalledTimes(2));

      fireEvent.click(header());
      wsEvent.value = { event: eventName, data: { file_id: "f1" } };
      view.rerender(section({ mediaController: controller }));
      await flush();
      expect(getVideoVisualIndex).toHaveBeenCalledTimes(2);
    },
  );
});

describe("SPEC-ADDON-089 I7 — the .loft", () => {
  it("requests clip-timestamps on mount and on every file change", async () => {
    vi.mocked(getClipTimestamps).mockResolvedValue(timestamps(makeRows(3)));

    const view = renderSection(loft());
    await screen.findByRole("button", { name: /^Scenes/ });
    expect(getClipTimestamps).toHaveBeenCalledTimes(1);
    expect(getClipTimestamps).toHaveBeenCalledWith("f1", "d1");
    expect(getVideoVisualIndex).not.toHaveBeenCalled();

    view.rerender(section(loft({ fileId: "f2" })));
    await waitFor(() => expect(getClipTimestamps).toHaveBeenCalledTimes(2));
    expect(getClipTimestamps).toHaveBeenLastCalledWith("f2", "d1");
  });

  it("shows nothing until the first rows arrive", async () => {
    const pending = deferred<ReturnType<typeof timestamps>>();
    vi.mocked(getClipTimestamps).mockReturnValueOnce(pending.promise);

    const { container } = renderSection(loft());
    await waitFor(() => expect(getClipTimestamps).toHaveBeenCalledTimes(1));
    expect(container.firstChild).toBeNull();

    pending.resolve(timestamps(makeRows(2)));
    await screen.findByRole("button", { name: /^Scenes/ });
  });

  it.each([
    { name: "the request fails", value: null },
    {
      name: "the answer has no rows",
      value: timestamps([]),
    },
  ])(
    "SPEC-ADDON-089 I7 — stays hidden when $name",
    async ({ value }) => {
      vi.mocked(getClipTimestamps).mockResolvedValue(value);
      const { container } = renderSection(loft());

      await waitFor(() => expect(getClipTimestamps).toHaveBeenCalledTimes(1));
      await flush();

      expect(container.firstChild).toBeNull();
      expect(screen.queryByRole("button", { name: /^Scenes/ })).toBeNull();
    },
  );

  it("keeps its rows across a collapse", async () => {
    vi.mocked(getClipTimestamps).mockResolvedValue(timestamps(makeRows(3)));
    renderSection(loft());
    await screen.findByRole("button", { name: /^Scenes/ });
    fireEvent.click(header());
    expect(tileLabels()).toEqual(["0:00", "0:05", "0:10"]);

    fireEvent.click(header());
    fireEvent.click(header());

    expect(getClipTimestamps).toHaveBeenCalledTimes(1);
    expect(tileLabels()).toEqual(["0:00", "0:05", "0:10"]);
  });

  it("expands without a request or a skeleton", async () => {
    vi.mocked(getClipTimestamps).mockResolvedValue(timestamps(makeRows(3)));
    renderSection(loft());
    await screen.findByRole("button", { name: /^Scenes/ });

    fireEvent.click(header());

    expect(getClipTimestamps).toHaveBeenCalledTimes(1);
    expect(screen.getByRole("slider")).toBeInTheDocument();
    expect(getVideoVisualIndex).not.toHaveBeenCalled();
  });

  it("takes every picture from the scene-tile route", async () => {
    vi.mocked(getClipTimestamps).mockResolvedValue(timestamps(makeRows(15)));
    renderSection(loft());
    await screen.findByRole("button", { name: /^Scenes/ });
    fireEvent.click(header());

    await waitFor(() => expect(images()).toHaveLength(9));

    for (const img of images()) {
      expect(img.getAttribute("src")).toContain("/scene-tile/f1?t=");
    }
    expect(getFrameUrl).not.toHaveBeenCalled();
    expect(getSceneTileUrl).toHaveBeenCalled();
  });

  it("keeps a failed tile's time label and click-to-seek", async () => {
    const { controller, seek, play } = makeController({
      duration: 70,
      paused: true,
    });
    vi.mocked(getClipTimestamps).mockResolvedValue(
      timestamps([
        { start: 65, content_preview: "Frame at 65s" },
        { start: 70, content_preview: "Frame at 70s" },
      ]),
    );
    renderSection(loft({ mediaController: controller }));
    await screen.findByRole("button", { name: /^Scenes/ });
    fireEvent.click(header());
    await waitFor(() => expect(images()).toHaveLength(2));

    fireEvent.error(images()[0]);

    expect(screen.queryByAltText("Frame at 65s")).toBeNull();
    expect(tile(65)).toHaveAttribute("aria-label", "Play from 1:05");

    fireEvent.click(tile(65));

    expect(seek).toHaveBeenCalledWith(65);
    expect(play).toHaveBeenCalled();
  });

  it("draws no Visual Index part and never requests it", async () => {
    vi.mocked(getClipTimestamps).mockResolvedValue(timestamps(makeRows(15)));
    vi.mocked(getVideoVisualIndex).mockResolvedValue(
      indexState({
        active_run: makeRun({ succeeded_count: 1 }),
        scenes: [makeScene({ start_time: 10, scene_label: "Label" })],
      }),
    );
    renderSection(loft());
    await screen.findByRole("button", { name: /^Scenes/ });
    fireEvent.click(header());

    expect(getVideoVisualIndex).not.toHaveBeenCalled();
    expect(screen.queryByRole("slider")).toBeInTheDocument();
    expect(nameVisible(/^\d+ scenes$/)).toBe(false);
    expect(actionNames()).toEqual([]);
    expect(sceneCardNamed(/Label/)).toBeNull();
  });

  it("ignores an earlier file's late rows", async () => {
    const first = deferred<ReturnType<typeof timestamps>>();
    const second = deferred<ReturnType<typeof timestamps>>();
    vi.mocked(getClipTimestamps)
      .mockReturnValueOnce(first.promise)
      .mockReturnValueOnce(second.promise);

    const view = renderSection(loft());
    await waitFor(() => expect(getClipTimestamps).toHaveBeenCalledTimes(1));

    view.rerender(section(loft({ fileId: "f2" })));
    await waitFor(() => expect(getClipTimestamps).toHaveBeenCalledTimes(2));

    second.resolve(
      timestamps(
        [
          { start: 40, content_preview: "Frame at 40s" },
          { start: 45, content_preview: "Frame at 45s" },
        ],
        "f2",
      ),
    );
    await screen.findByRole("button", { name: /^Scenes/ });
    fireEvent.click(header());
    expect(tileLabels()).toEqual(["0:40", "0:45"]);

    first.resolve(timestamps(makeRows(2), "f1"));
    await flush();

    expect(tileLabels()).toEqual(["0:40", "0:45"]);
  });
});

describe("SPEC-ADDON-089 I8 — picture requests", () => {
  it("requests only the current window's nine pictures", async () => {
    vi.mocked(usePolicy).mockReturnValue({ enabled: false, isLoading: false });
    vi.mocked(getClipTimestamps).mockResolvedValue(timestamps(makeRows(40)));
    const { controller } = makeController({ duration: 200, paused: true });
    renderSection({ mediaController: controller });
    await openNative();

    await waitFor(() => expect(images()).toHaveLength(9));

    expect(imgTimes()).toEqual([0, 5, 10, 15, 20, 25, 30, 35, 40]);
    expect(imgTimes()).not.toContain(45);
  });

  it("requests only the final window after a sweep, and only after 150 ms", async () => {
    vi.useFakeTimers();
    vi.mocked(usePolicy).mockReturnValue({ enabled: false, isLoading: false });
    vi.mocked(getClipTimestamps).mockResolvedValue(timestamps(makeRows(40)));
    const { controller } = makeController({ duration: 200, paused: true });
    renderSection({ mediaController: controller });
    fireEvent.click(header());
    await flush();
    const el = screen.getByRole("slider");
    stubRect(el);

    await advance(150);
    const initial = imgTimes();
    expect(initial).toEqual([0, 5, 10, 15, 20, 25, 30, 35, 40]);

    firePointer(el, "pointerMove", { pointerType: "mouse", clientX: pointerX(60, 200) });
    await advance(149);
    for (const time of [45, 50, 55, 60, 65, 70]) {
      expect(imgTimes()).not.toContain(time);
    }

    firePointer(el, "pointerMove", { pointerType: "mouse", clientX: pointerX(150, 200) });
    await advance(149);
    for (const time of [135, 140, 145, 150, 155, 160, 165, 170, 175]) {
      expect(imgTimes()).not.toContain(time);
    }

    await advance(1);
    for (const time of [135, 140, 145, 150, 155, 160, 165, 170, 175]) {
      expect(imgTimes()).toContain(time);
    }
  });
});
