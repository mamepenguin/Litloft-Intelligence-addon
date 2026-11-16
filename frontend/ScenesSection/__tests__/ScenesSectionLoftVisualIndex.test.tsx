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
  getFrameUrl,
  getVideoVisualIndex,
  retryVideoVisualIndex,
} from "@/addons/intelligence/api";
import { usePolicy } from "@/hooks/usePolicy";
import {
  action,
  actionNames,
  flush,
  header,
  images,
  indexState,
  installPointerEvents,
  LOFT_MIME,
  makeRows,
  makeRun,
  makeScene,
  nameVisible,
  SceneSectionProps,
  sceneCardNamed,
  timestamps,
} from "./helpers";

function section(props: Partial<SceneSectionProps> = {}) {
  return (
    <ScenesSection
      fileId="f1"
      drive="d1"
      fileType="video"
      mimeType={LOFT_MIME}
      {...props}
    />
  );
}

function renderSection(props: Partial<SceneSectionProps> = {}) {
  return render(section(props));
}

const withScene = indexState({
  active_run: makeRun({ succeeded_count: 1 }),
  scenes: [
    makeScene({ start_time: 10, scene_label: "Salt goes in", visible_text: "SALT 5g" }),
  ],
});

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

describe("SPEC-ADDON-118 item 4 — the .loft's Visual Index requests", () => {
  it("requests visual-index on mount and on a file change only while the policy is on", async () => {
    vi.mocked(usePolicy).mockReturnValue({ enabled: false, isLoading: false });
    const view = renderSection();
    await screen.findByRole("button", { name: /^Scenes/ });
    fireEvent.click(header());
    await flush();

    expect(getClipTimestamps).toHaveBeenCalledTimes(1);
    expect(getVideoVisualIndex).not.toHaveBeenCalled();
    expect(actionNames()).toEqual([]);

    vi.mocked(usePolicy).mockReturnValue({ enabled: true, isLoading: false });
    view.rerender(section({ fileId: "f2" }));

    await waitFor(() => expect(getVideoVisualIndex).toHaveBeenCalledTimes(1));
    expect(vi.mocked(getVideoVisualIndex).mock.calls[0][0]).toBe("f2");
  });

  it("requests visual-index once the loading policy resolves on", async () => {
    vi.mocked(usePolicy).mockReturnValue({ enabled: true, isLoading: true });
    const view = renderSection();
    await screen.findByRole("button", { name: /^Scenes/ });
    await flush();
    expect(getVideoVisualIndex).not.toHaveBeenCalled();

    vi.mocked(usePolicy).mockReturnValue({ enabled: true, isLoading: false });
    view.rerender(section());

    await waitFor(() => expect(getVideoVisualIndex).toHaveBeenCalledTimes(1));
  });

  it("reloads visual-index on a WebSocket event for the current file only", async () => {
    const view = renderSection();
    await screen.findByRole("button", { name: /^Scenes/ });
    fireEvent.click(header());
    await waitFor(() => expect(getVideoVisualIndex).toHaveBeenCalledTimes(1));

    wsEvent.value = {
      event: "intelligence.video_visual.succeeded",
      data: { file_id: "other" },
    };
    view.rerender(section());
    await flush();
    expect(getVideoVisualIndex).toHaveBeenCalledTimes(1);

    wsEvent.value = {
      event: "intelligence.video_visual.succeeded",
      data: { file_id: "f1" },
    };
    view.rerender(section());
    await waitFor(() => expect(getVideoVisualIndex).toHaveBeenCalledTimes(2));
  });
});

describe("SPEC-ADDON-118 item 4 — the .loft draws the Visual Index part", () => {
  it("offers Create scenes and generates for the .loft", async () => {
    renderSection();
    await screen.findByRole("button", { name: /^Scenes/ });
    fireEvent.click(header());

    await waitFor(() => expect(actionNames()).toEqual(["Create scenes"]));
    fireEvent.click(action("Create scenes")!);

    await waitFor(() =>
      expect(generateVideoVisualIndex).toHaveBeenCalledWith("f1", "d1"),
    );
  });

  it("draws the scene card and pictures it from the scene-tile route", async () => {
    vi.mocked(getVideoVisualIndex).mockResolvedValue(withScene);
    renderSection();
    await screen.findByRole("button", { name: /^Scenes/ });
    fireEvent.click(header());

    await waitFor(() => expect(sceneCardNamed(/Salt goes in/)).not.toBeNull());
    expect(nameVisible(/^1 scenes$/)).toBe(true);
    for (const img of images()) {
      expect(img.getAttribute("src")).toContain("/scene-tile/f1?t=");
    }
    expect(getFrameUrl).not.toHaveBeenCalled();
  });

  it("shows the section with its Visual Index when it has no storyboard rows", async () => {
    vi.mocked(getClipTimestamps).mockResolvedValue(timestamps([]));
    vi.mocked(getVideoVisualIndex).mockResolvedValue(withScene);
    renderSection();

    const button = await screen.findByRole("button", { name: /^Scenes/ });
    fireEvent.click(button);

    await waitFor(() => expect(sceneCardNamed(/Salt goes in/)).not.toBeNull());
    for (const img of images()) {
      expect(img.getAttribute("src")).toContain("/scene-tile/f1?t=");
    }
  });

  it("stays hidden with no storyboard rows and no Visual Index", async () => {
    vi.mocked(getClipTimestamps).mockResolvedValue(timestamps([]));
    vi.mocked(getVideoVisualIndex).mockResolvedValue(
      indexState({ eligible: false }),
    );
    const { container } = renderSection();

    await waitFor(() => expect(getVideoVisualIndex).toHaveBeenCalledTimes(1));
    await flush();

    expect(container.firstChild).toBeNull();
  });
});
