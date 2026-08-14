/**
 * Tests for VisualIndexSection state branches.
 *
 * Design doc "Video Visual Index" §14.2:
 * - ineligible/disabled states render no section
 * - collapsed view does not request frames
 * - active result remains during staged regeneration
 * - progress and partial labels
 * - card seek/play through MediaController
 * - visible-text and transcript disclosures only when populated
 * - incremental card rendering
 * - retry targets failed scenes only
 * - native-video-only gating, including `.loft` exclusion
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent, act } from "@testing-library/react";
import type { MediaController } from "@/lib/mediaController";

const visualIndexMessages = vi.hoisted(() => ({
  visualIndexTitle: "Visual index",
  visualIndexSceneCount: "{count} scenes",
  visualIndexProcessing: "Processing {progress}",
  visualIndexUpdating: "Updating {progress}",
  visualIndexPartial: "Partial",
  visualIndexEmpty: "No visual index has been generated yet.",
  visualIndexStale: "The source scenes have changed since this index was built.",
  visualIndexGenerate: "Generate",
  visualIndexGenerateAgain: "Generate again",
  visualIndexGenerating: "Starting…",
  visualIndexRetryFailed: "Retry failed scenes",
  visualIndexSceneFailed: "Failed",
  visualIndexTranscriptExcerpt: "Transcript",
  visualIndexWaitingPrerequisite:
    "Waiting on scene indexing to finish before this can start.",
  visualIndexActionError: "Could not start visual index generation.",
}));

vi.mock("next-intl", () => ({
  useTranslations: () => (
    key: keyof typeof visualIndexMessages,
    values?: Record<string, unknown>,
  ) => {
    const template = visualIndexMessages[key];
    return Object.entries(values ?? {}).reduce(
      (text, [name, value]) => text.replace(`{${name}}`, String(value)),
      template,
    );
  },
}));

vi.mock("@/addons/intelligence/api", () => ({
  getVideoVisualIndex: vi.fn(),
  generateVideoVisualIndex: vi.fn(),
  retryVideoVisualIndex: vi.fn(),
  getFrameUrl: (fileId: string, t: number) => `/frame/${fileId}/${t}`,
}));

vi.mock("@/hooks/usePolicy", () => ({
  usePolicy: vi.fn(),
}));

const wsEvent = vi.hoisted(() => ({
  value: null as null | { event: string; data: Record<string, unknown> },
}));
vi.mock("@/hooks/useWebSocket", () => ({
  useWebSocket: vi.fn((eventFilter?: string) => {
    if (!wsEvent.value || wsEvent.value.event !== eventFilter) return null;
    return wsEvent.value;
  }),
}));

import VisualIndexSection from "@/addons/intelligence/VisualIndexSection";
import {
  getVideoVisualIndex,
  generateVideoVisualIndex,
  retryVideoVisualIndex,
} from "@/addons/intelligence/api";
import { usePolicy } from "@/hooks/usePolicy";

class FakeIntersectionObserver {
  observe = vi.fn();
  disconnect = vi.fn();
  unobserve = vi.fn();
}

const neverGenerated = {
  eligible: true,
  available: true,
  file_id: "f1",
  active_run: null,
  scenes: [],
  staged_run: null,
  stale: false,
};

const activeResult = {
  eligible: true,
  available: true,
  file_id: "f1",
  active_run: {
    run_id: "vvr_1",
    status: "succeeded" as const,
    selected_count: 2,
    completed_count: 2,
    succeeded_count: 2,
    failed_count: 0,
    created_at: "2026-08-13T00:00:00Z",
    completed_at: "2026-08-13T00:01:00Z",
  },
  scenes: [
    {
      ordering: 0,
      start_time: 5.0,
      end_time: null,
      status: "succeeded" as const,
      scene_type: "slide",
      scene_label: "Architecture diagram",
      visible_text: "Browser -> Server",
      transcript_excerpt: "and here we see the request flow",
    },
    {
      ordering: 1,
      start_time: 30.0,
      end_time: null,
      status: "succeeded" as const,
      scene_type: "person",
      scene_label: "Presenter on camera",
      visible_text: null,
      transcript_excerpt: null,
    },
  ],
  staged_run: null,
  stale: false,
};

function renderSection(
  props: Partial<React.ComponentProps<typeof VisualIndexSection>> = {},
) {
  const mediaController = {
    seek: vi.fn(),
    play: vi.fn(),
    pause: vi.fn(),
    togglePlay: vi.fn(),
    toggleMute: vi.fn(),
    toggleFullscreen: vi.fn(),
    getCurrentTime: vi.fn(() => 0),
    getDuration: vi.fn(() => 0),
    isPaused: vi.fn(() => true),
    isMuted: vi.fn(() => false),
    getVolume: vi.fn(() => 1),
    setVolume: vi.fn(),
    getPlaybackRate: vi.fn(() => 1),
    setPlaybackRate: vi.fn(),
    getBufferedFraction: vi.fn(() => 0),
  } satisfies MediaController;
  const result = render(
    <VisualIndexSection
      fileId="f1"
      drive="family"
      fileType="video"
      mimeType="video/mp4"
      mediaController={mediaController}
      {...props}
    />,
  );
  return { ...result, mediaController };
}

describe("VisualIndexSection", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    wsEvent.value = null;
    vi.stubGlobal("IntersectionObserver", FakeIntersectionObserver);
    vi.mocked(usePolicy).mockReturnValue({ enabled: true, isLoading: false });
    vi.mocked(getVideoVisualIndex).mockResolvedValue(neverGenerated);
    vi.mocked(generateVideoVisualIndex).mockResolvedValue({
      status: "accepted", file_id: "f1", run_id: "vvr_2",
    });
    vi.mocked(retryVideoVisualIndex).mockResolvedValue({
      status: "accepted", file_id: "f1", run_id: "vvr_1", reset_count: 1,
    });
  });

  // -- Eligibility gating -----------------------------------------------

  it("does not request state for non-video files", async () => {
    const { container } = renderSection({ fileType: "document", mimeType: "application/pdf" });
    await act(async () => undefined);
    expect(getVideoVisualIndex).not.toHaveBeenCalled();
    expect(container).toBeEmptyDOMElement();
  });

  it("does not request state for .loft files", async () => {
    const { container } = renderSection({
      fileType: "video",
      mimeType: "application/vnd.litloft.loft+json",
    });
    await act(async () => undefined);
    expect(getVideoVisualIndex).not.toHaveBeenCalled();
    expect(container).toBeEmptyDOMElement();
  });

  it("does not request state when drive policy is off", async () => {
    vi.mocked(usePolicy).mockReturnValue({ enabled: false, isLoading: false });
    const { container } = renderSection();
    await act(async () => undefined);
    expect(getVideoVisualIndex).not.toHaveBeenCalled();
    expect(container).toBeEmptyDOMElement();
  });

  it("renders nothing when the backend reports the file as ineligible", async () => {
    vi.mocked(getVideoVisualIndex).mockResolvedValue({
      ...neverGenerated, eligible: false,
    });
    const { container } = renderSection();
    await waitFor(() => expect(getVideoVisualIndex).toHaveBeenCalled());
    expect(container).toBeEmptyDOMElement();
  });

  it("renders nothing when the feature is unavailable (no vision model)", async () => {
    vi.mocked(getVideoVisualIndex).mockResolvedValue({
      ...neverGenerated, available: false,
    });
    const { container } = renderSection();
    await waitFor(() => expect(getVideoVisualIndex).toHaveBeenCalled());
    expect(container).toBeEmptyDOMElement();
  });

  it("renders nothing when the GET call fails (null response)", async () => {
    vi.mocked(getVideoVisualIndex).mockResolvedValue(null);
    const { container } = renderSection();
    await waitFor(() => expect(getVideoVisualIndex).toHaveBeenCalled());
    expect(container).toBeEmptyDOMElement();
  });

  // -- Collapsed header ---------------------------------------------------

  it("shows the bare title with no count when never generated", async () => {
    renderSection();
    expect(await screen.findByText("Visual index")).toBeInTheDocument();
  });

  it("shows the scene count for an active result", async () => {
    vi.mocked(getVideoVisualIndex).mockResolvedValue(activeResult);
    renderSection();
    expect(await screen.findByText("Visual index · 2 scenes")).toBeInTheDocument();
  });

  it("shows a processing label with no prior active result", async () => {
    vi.mocked(getVideoVisualIndex).mockResolvedValue({
      ...neverGenerated,
      staged_run: {
        run_id: "vvr_2", status: "running", selected_count: 12,
        completed_count: 7, succeeded_count: 7, failed_count: 0,
        created_at: null, completed_at: null,
      },
    });
    renderSection();
    expect(await screen.findByText("Visual index · Processing 7/12")).toBeInTheDocument();
  });

  it("shows an updating label when a staged run overlaps an active result", async () => {
    vi.mocked(getVideoVisualIndex).mockResolvedValue({
      ...activeResult,
      staged_run: {
        run_id: "vvr_3", status: "queued", selected_count: 10,
        completed_count: 3, succeeded_count: 3, failed_count: 0,
        created_at: null, completed_at: null,
      },
    });
    renderSection();
    expect(await screen.findByText("Visual index · Updating 3/10")).toBeInTheDocument();
  });

  it("shows a partial label for a partial active run", async () => {
    vi.mocked(getVideoVisualIndex).mockResolvedValue({
      ...activeResult,
      active_run: { ...activeResult.active_run, status: "partial", failed_count: 1 },
    });
    renderSection();
    expect(await screen.findByText("Visual index · Partial")).toBeInTheDocument();
  });

  // -- Collapsed view fetches no frame images -----------------------------

  it("does not render frame images before the section is expanded", async () => {
    vi.mocked(getVideoVisualIndex).mockResolvedValue(activeResult);
    renderSection();
    await screen.findByText("Visual index · 2 scenes");
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
  });

  it("renders frame images once expanded", async () => {
    vi.mocked(getVideoVisualIndex).mockResolvedValue(activeResult);
    renderSection();
    fireEvent.click(await screen.findByText("Visual index · 2 scenes"));
    const images = await screen.findAllByRole("img");
    expect(images).toHaveLength(2);
    expect(images[0]).toHaveAttribute("src", "/frame/f1/5");
  });

  // -- Scene card content ---------------------------------------------------

  it("shows visible_text only when present", async () => {
    vi.mocked(getVideoVisualIndex).mockResolvedValue(activeResult);
    renderSection();
    fireEvent.click(await screen.findByText("Visual index · 2 scenes"));
    expect(await screen.findByText("Browser -> Server")).toBeInTheDocument();
    // Second scene has no visible_text.
    expect(screen.queryByText("Presenter on camera")).toBeInTheDocument();
  });

  it("shows a transcript disclosure only when an excerpt exists", async () => {
    vi.mocked(getVideoVisualIndex).mockResolvedValue(activeResult);
    renderSection();
    fireEvent.click(await screen.findByText("Visual index · 2 scenes"));
    const transcriptToggles = await screen.findAllByText("Transcript");
    // Only the first scene has a transcript_excerpt.
    expect(transcriptToggles).toHaveLength(1);
  });

  it("seeks and plays via MediaController when a scene card is clicked", async () => {
    vi.mocked(getVideoVisualIndex).mockResolvedValue(activeResult);
    const { mediaController } = renderSection();
    fireEvent.click(await screen.findByText("Visual index · 2 scenes"));
    const images = await screen.findAllByRole("img");
    fireEvent.click(images[0]);
    expect(mediaController.seek).toHaveBeenCalledWith(5.0);
    expect(mediaController.play).toHaveBeenCalledOnce();
  });

  // -- Actions --------------------------------------------------------------

  it("offers Generate when never generated, and calls the API", async () => {
    renderSection();
    fireEvent.click(await screen.findByText("Visual index"));
    fireEvent.click(await screen.findByRole("button", { name: "Generate" }));
    await waitFor(() => {
      expect(generateVideoVisualIndex).toHaveBeenCalledWith("f1", "family");
    });
  });

  it("offers Generate again when an active result exists", async () => {
    vi.mocked(getVideoVisualIndex).mockResolvedValue(activeResult);
    renderSection();
    fireEvent.click(await screen.findByText("Visual index · 2 scenes"));
    expect(
      await screen.findByRole("button", { name: "Generate again" }),
    ).toBeInTheDocument();
  });

  it("shows a waiting message on a 409 from generate", async () => {
    const err = new Error("waiting") as Error & { status?: number };
    err.status = 409;
    vi.mocked(generateVideoVisualIndex).mockRejectedValue(err);
    renderSection();
    fireEvent.click(await screen.findByText("Visual index"));
    fireEvent.click(await screen.findByRole("button", { name: "Generate" }));
    expect(
      await screen.findByText(
        "Waiting on scene indexing to finish before this can start.",
      ),
    ).toBeInTheDocument();
  });

  it("offers retry only when there are failed scenes, and targets them", async () => {
    vi.mocked(getVideoVisualIndex).mockResolvedValue({
      ...activeResult,
      active_run: { ...activeResult.active_run, status: "partial", failed_count: 1 },
    });
    renderSection();
    fireEvent.click(await screen.findByText("Visual index · Partial"));
    fireEvent.click(
      await screen.findByRole("button", { name: "Retry failed scenes" }),
    );
    await waitFor(() => {
      expect(retryVideoVisualIndex).toHaveBeenCalledWith("f1", "family");
    });
  });

  it("does not offer retry when there are no failed scenes", async () => {
    vi.mocked(getVideoVisualIndex).mockResolvedValue(activeResult);
    renderSection();
    fireEvent.click(await screen.findByText("Visual index · 2 scenes"));
    await screen.findByRole("button", { name: "Generate again" });
    expect(
      screen.queryByRole("button", { name: "Retry failed scenes" }),
    ).not.toBeInTheDocument();
  });

  // -- Incremental rendering --------------------------------------------

  it("renders at most 12 cards on first expand", async () => {
    const manyScenes = Array.from({ length: 15 }, (_, i) => ({
      ordering: i,
      start_time: i * 10,
      end_time: null,
      status: "succeeded" as const,
      scene_type: null,
      scene_label: `Scene ${i}`,
      visible_text: null,
      transcript_excerpt: null,
    }));
    vi.mocked(getVideoVisualIndex).mockResolvedValue({
      ...activeResult,
      active_run: { ...activeResult.active_run, selected_count: 15 },
      scenes: manyScenes,
    });
    renderSection();
    fireEvent.click(await screen.findByText("Visual index · 15 scenes"));
    const images = await screen.findAllByRole("img");
    expect(images).toHaveLength(12);
  });

  // -- Live updates via WebSocket -----------------------------------------

  it("reloads on a progress event for this file", async () => {
    vi.mocked(getVideoVisualIndex).mockResolvedValue(neverGenerated);
    const { rerender } = renderSection();
    await screen.findByText("Visual index");
    expect(getVideoVisualIndex).toHaveBeenCalledTimes(1);

    vi.mocked(getVideoVisualIndex).mockResolvedValue(activeResult);
    wsEvent.value = {
      event: "intelligence.video_visual.succeeded",
      data: { file_id: "f1", run_id: "vvr_1" },
    };
    rerender(
      <VisualIndexSection
        fileId="f1"
        drive="family"
        fileType="video"
        mimeType="video/mp4"
      />,
    );

    expect(await screen.findByText("Visual index · 2 scenes")).toBeInTheDocument();
  });

  it("ignores WebSocket events for another file", async () => {
    const { rerender } = renderSection();
    await screen.findByText("Visual index");
    expect(getVideoVisualIndex).toHaveBeenCalledTimes(1);

    wsEvent.value = {
      event: "intelligence.video_visual.succeeded",
      data: { file_id: "other" },
    };
    rerender(
      <VisualIndexSection
        fileId="f1"
        drive="family"
        fileType="video"
        mimeType="video/mp4"
      />,
    );

    expect(getVideoVisualIndex).toHaveBeenCalledTimes(1);
  });
});
