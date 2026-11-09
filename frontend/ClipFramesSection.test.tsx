/**
 * Tests for ClipFramesSection — collapsed-by-default + filmstrip UX.
 *
 * Spec / hako: KMScjkyqGb2vy9zluV4Bp
 *
 * Covers:
 *   1. A native video: NO API call on mount, the header is visible, the
 *      frames come from the frame route, and a zero-row answer hides it.
 *   2. A mouse environment gets wrapped rows in a box about two rows
 *      tall; touch keeps one horizontally scrollable row.
 *   3. When the API reports more timestamps than `PAGE_SIZE`, the
 *      sentinel is rendered for the IntersectionObserver-driven
 *      infinite scroll.
 *   4. A `.loft` fetches its scene times on mount and shows the
 *      storyboard tiles (SPEC-ADDON-080).
 */

import { describe, it, expect, vi, beforeAll, beforeEach, afterEach } from "vitest";
import {
  render,
  screen,
  fireEvent,
  waitFor,
  act,
  cleanup,
} from "@testing-library/react";
import { NextIntlClientProvider } from "next-intl";

import type { MediaController } from "@/lib/mediaController";

vi.mock("@/addons/intelligence/api", () => ({
  getClipTimestamps: vi.fn(),
  getFrameUrl: vi.fn(
    (fileId: string, t: number) =>
      `/api/addons/intelligence/files/${fileId}/frame?t=${t}`,
  ),
  getSceneTileUrl: vi.fn(
    (fileId: string, t: number) =>
      `/api/addons/intelligence/files/${fileId}/scene-tile?t=${t}`,
  ),
}));

import ClipFramesSection from "@/addons/intelligence/ClipFramesSection";
import {
  getClipTimestamps,
  getFrameUrl,
  getSceneTileUrl,
} from "@/addons/intelligence/api";

const mocked = getClipTimestamps as unknown as ReturnType<typeof vi.fn>;
const frameUrl = getFrameUrl as unknown as ReturnType<typeof vi.fn>;
const sceneTileUrl = getSceneTileUrl as unknown as ReturnType<typeof vi.fn>;

const LOFT_MIME = "application/vnd.litloft.loft+json";

function makeTimestamps(n: number, step = 5) {
  return Array.from({ length: n }, (_, i) => ({
    start: i * step,
    content_preview: `Frame at ${i * step}s`,
  }));
}

function controller(): MediaController {
  return {
    seek: vi.fn(),
    play: vi.fn(),
    pause: vi.fn(),
    togglePlay: vi.fn(),
    toggleMute: vi.fn(),
    toggleFullscreen: vi.fn(),
    getCurrentTime: () => 0,
    getDuration: () => 0,
    isPaused: () => true,
    isMuted: () => true,
    getVolume: () => 1,
    setVolume: () => {},
    getPlaybackRate: () => 1,
    setPlaybackRate: () => {},
    getBufferedFraction: () => 0,
  };
}

function deferred<T>() {
  let resolve!: (v: T) => void;
  let reject!: (e: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function view(props: {
  fileId?: string;
  fileType?: string;
  mimeType?: string;
  mediaController?: MediaController;
} = {}) {
  return (
    <NextIntlClientProvider
      locale="en"
      messages={{ searchIndex: { clipTitle: "CLIP Frames" } }}
    >
      <ClipFramesSection
        fileId={props.fileId ?? "f1"}
        drive="drive1"
        fileType={props.fileType ?? "video"}
        mimeType={props.mimeType ?? "video/mp4"}
        mediaController={props.mediaController}
      />
    </NextIntlClientProvider>
  );
}

function renderSection(
  props: {
    fileId?: string;
    fileType?: string;
    mimeType?: string;
    mediaController?: MediaController;
  } = {},
) {
  return render(view(props));
}

function loft(props: { fileId?: string; mediaController?: MediaController } = {}) {
  return {
    ...props,
    fileType: "video",
    mimeType: LOFT_MIME,
  };
}

// jsdom doesn't ship IntersectionObserver — install a minimal stub
// good enough for the component to mount without crashing. File scope,
// because every describe below mounts the component and the run order is
// shuffled.
beforeAll(() => {
  if (!("IntersectionObserver" in globalThis)) {
    class IO {
      observe() {}
      disconnect() {}
      unobserve() {}
      takeRecords() { return []; }
    }
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (globalThis as any).IntersectionObserver = IO as any;
  }
});

describe("ClipFramesSection", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    cleanup();
  });

  it("does NOT fetch timestamps on mount (closed by default)", async () => {
    mocked.mockResolvedValue({ file_id: "f1", drive: "drive1", timestamps: makeTimestamps(3) });
    renderSection();

    // Header is visible
    expect(screen.getByRole("button", { name: /CLIP/i })).toBeInTheDocument();
    // ... but no fetch happened, even after a microtask flush
    await Promise.resolve();
    expect(mocked).not.toHaveBeenCalled();
    // Frame thumbnails must not be in the DOM
    expect(screen.queryByRole("img")).toBeNull();
  });

  it("fetches once when expanded; re-collapse + re-expand reuses cache", async () => {
    mocked.mockResolvedValue({ file_id: "f1", drive: "drive1", timestamps: makeTimestamps(3) });
    renderSection();

    const toggle = screen.getByRole("button", { name: /CLIP/i });

    // First expand → one fetch + frames render.
    fireEvent.click(toggle);
    await waitFor(() => expect(mocked).toHaveBeenCalledTimes(1));
    await waitFor(() =>
      expect(screen.getAllByRole("img").length).toBeGreaterThan(0),
    );

    // Collapse.
    fireEvent.click(toggle);
    expect(screen.queryByRole("img")).toBeNull();

    // Re-expand → no second fetch; frames come back from cached state.
    fireEvent.click(toggle);
    await waitFor(() =>
      expect(screen.getAllByRole("img").length).toBeGreaterThan(0),
    );
    expect(mocked).toHaveBeenCalledTimes(1);
  });

  it("wraps its frames on a mouse and scrolls sideways on touch", async () => {
    mocked.mockResolvedValue({ file_id: "f1", drive: "drive1", timestamps: makeTimestamps(5) });
    renderSection();

    fireEvent.click(screen.getByRole("button", { name: /CLIP/i }));
    await waitFor(() =>
      expect(screen.getAllByRole("img").length).toBe(5),
    );

    // Find the strip via the first frame's parent button → grandparent
    // is the scroll container.
    const firstThumb = screen.getAllByRole("img")[0];
    const frameButton = firstThumb.closest("button");
    expect(frameButton).not.toBeNull();
    const strip = frameButton!.parentElement!;
    expect(strip.className).toMatch(/\bflex-wrap\b/);
    expect(strip.className).toMatch(/\bmouse:overflow-y-auto\b/);
    expect(strip.className).toMatch(/\bmouse:max-h-/);
    expect(strip.className).toMatch(/\bpointer-coarse:flex-nowrap\b/);
    expect(strip.className).toMatch(/\bpointer-coarse:overflow-x-auto\b/);
    expect(strip.className).not.toMatch(/(?:^|\s)overflow-x-auto(?:\s|$)/);
    expect(strip.className).not.toMatch(/grid/);
  });

  it("hides the section entirely when there are no CLIP frames", async () => {
    mocked.mockResolvedValue({ file_id: "f1", drive: "drive1", timestamps: [] });
    const { container } = renderSection();

    // Pre-expand the section header is still visible (we don't yet know
    // the file has no frames — the whole point of lazy fetch is to
    // avoid hitting the API on mount).
    fireEvent.click(screen.getByRole("button", { name: /CLIP/i }));

    await waitFor(() => expect(mocked).toHaveBeenCalledTimes(1));
    // After the fetch resolves with no timestamps the component must
    // unmount itself — file detail pages must not show an empty header.
    await waitFor(() => expect(container.firstChild).toBeNull());
  });

  it("renders nothing (no header, no fetch) for non-video files", async () => {
    mocked.mockResolvedValue({ file_id: "f1", drive: "drive1", timestamps: makeTimestamps(3) });
    const { container } = renderSection({ fileType: "image", mimeType: "image/jpeg" });

    expect(container.firstChild).toBeNull();
    await Promise.resolve();
    expect(mocked).not.toHaveBeenCalled();
  });

  it("renders the infinite-scroll sentinel when more frames than PAGE_SIZE", async () => {
    mocked.mockResolvedValue({ file_id: "f1", drive: "drive1", timestamps: makeTimestamps(35) });
    renderSection();

    fireEvent.click(screen.getByRole("button", { name: /CLIP/i }));
    await waitFor(() =>
      expect(screen.getAllByRole("img").length).toBe(20),
    );

    // First page is exactly PAGE_SIZE (20) — sentinel must be present
    // because there are 35 total. The sentinel is the only direct child
    // of the strip with aria-hidden true.
    const firstThumb = screen.getAllByRole("img")[0];
    const strip = firstThumb.closest("button")!.parentElement!;
    const sentinel = Array.from(strip.children).find(
      (el) => el.getAttribute("aria-hidden") === "true",
    );
    expect(sentinel).toBeDefined();
  });
});

describe("the .loft panel (SPEC-ADDON-080)", () => {
  beforeEach(() => {
    // ``mockReset`` rather than ``mockClear``: an unconsumed
    // ``mockReturnValueOnce`` from one of these cases would otherwise
    // answer the next case's first request with a promise nobody settles.
    mocked.mockReset();
    frameUrl.mockClear();
    sceneTileUrl.mockClear();
  });

  afterEach(() => {
    cleanup();
  });

  it("requests the scene times on mount and renders nothing while they are pending (SPEC-ADDON-080)", async () => {
    const pending = deferred<{ file_id: string; drive: string; timestamps: { start: number; content_preview: string }[] }>();
    mocked.mockReturnValue(pending.promise);

    const { container } = renderSection(loft());

    await waitFor(() => expect(mocked).toHaveBeenCalledTimes(1));
    expect(mocked).toHaveBeenCalledWith("f1", "drive1");
    expect(container.firstChild).toBeNull();
  });

  it("stays hidden when the scene-times request fails, with no retry of its own (SPEC-ADDON-080)", async () => {
    mocked.mockResolvedValue(null);

    const { container } = renderSection(loft());

    await waitFor(() => expect(mocked).toHaveBeenCalledTimes(1));
    await act(async () => {
      await Promise.resolve();
    });
    expect(container.firstChild).toBeNull();
    expect(mocked).toHaveBeenCalledTimes(1);
  });

  it("stays hidden when the file has no scene rows (SPEC-ADDON-080)", async () => {
    mocked.mockResolvedValue({ file_id: "f1", drive: "drive1", timestamps: [] });

    const { container } = renderSection(loft());

    await waitFor(() => expect(mocked).toHaveBeenCalledTimes(1));
    await act(async () => {
      await Promise.resolve();
    });
    expect(container.firstChild).toBeNull();
  });

  it("shows the collapsed header with the count and takes twenty pictures from the scene-tile route (SPEC-ADDON-080)", async () => {
    const rows = makeTimestamps(25);
    mocked.mockResolvedValue({ file_id: "f1", drive: "drive1", timestamps: rows });

    renderSection(loft());

    const header = await screen.findByRole("button", { name: /CLIP Frames \(25\)/ });
    expect(header).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByRole("img")).toBeNull();

    fireEvent.click(header);

    await waitFor(() => expect(screen.getAllByRole("img").length).toBe(20));
    const images = screen.getAllByRole("img");
    images.forEach((image, index) => {
      expect(image.getAttribute("src")).toBe(
        `/api/addons/intelligence/files/f1/scene-tile?t=${rows[index].start}`,
      );
      expect(image.getAttribute("src")).not.toContain("/frame?");
    });
    expect(sceneTileUrl.mock.calls.slice(0, 20)).toEqual(
      rows.slice(0, 20).map((row) => ["f1", row.start]),
    );
    expect(frameUrl).not.toHaveBeenCalled();
    expect(header).toHaveAttribute("aria-expanded", "true");
  });

  it("refetches when fileId changes and ignores the earlier file's late answer (SPEC-ADDON-080)", async () => {
    const first = deferred<{ file_id: string; drive: string; timestamps: { start: number; content_preview: string }[] }>();
    const second = deferred<{ file_id: string; drive: string; timestamps: { start: number; content_preview: string }[] }>();
    mocked.mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);

    const rendered = render(view(loft({ fileId: "f1" })));
    await waitFor(() => expect(mocked).toHaveBeenCalledTimes(1));

    rendered.rerender(view(loft({ fileId: "f2" })));
    await waitFor(() => expect(mocked).toHaveBeenCalledTimes(2));
    expect(mocked.mock.calls[1]).toEqual(["f2", "drive1"]);

    second.resolve({
      file_id: "f2",
      drive: "drive1",
      timestamps: [
        { start: 40, content_preview: "Frame at 40s" },
        { start: 45, content_preview: "Frame at 45s" },
      ],
    });
    const header = await screen.findByRole("button", { name: /CLIP Frames \(2\)/ });

    first.resolve({
      file_id: "f1",
      drive: "drive1",
      timestamps: [
        { start: 10, content_preview: "Frame at 10s" },
        { start: 15, content_preview: "Frame at 15s" },
      ],
    });
    await act(async () => {
      await Promise.resolve();
    });

    fireEvent.click(header);
    await waitFor(() => expect(screen.getAllByRole("img").length).toBe(2));
    const sources = screen.getAllByRole("img").map((img) => img.getAttribute("src"));
    expect(sources).toEqual([
      "/api/addons/intelligence/files/f2/scene-tile?t=40",
      "/api/addons/intelligence/files/f2/scene-tile?t=45",
    ]);
    expect(screen.queryByAltText("Frame at 10s")).toBeNull();
  });

  it("shows a blank block with no alt text for a failed tile and keeps its time and seek (SPEC-ADDON-080)", async () => {
    const seek = vi.fn();
    const play = vi.fn();
    const mediaController = { ...controller(), seek, play };
    mocked.mockResolvedValue({
      file_id: "f1",
      drive: "drive1",
      timestamps: [
        { start: 65, content_preview: "Frame at 65s" },
        { start: 70, content_preview: "Frame at 70s" },
      ],
    });
    renderSection(loft({ mediaController }));

    fireEvent.click(await screen.findByRole("button", { name: /CLIP Frames \(2\)/ }));
    await waitFor(() => expect(screen.getAllByRole("img").length).toBe(2));
    const requested = sceneTileUrl.mock.calls.length;

    fireEvent.error(screen.getAllByRole("img")[0]);

    expect(screen.queryByAltText("Frame at 65s")).toBeNull();
    const label = screen.getByText("1:05");
    expect(label).toBeInTheDocument();
    expect(sceneTileUrl.mock.calls.length).toBe(requested);

    fireEvent.click(label.closest("button")!);

    expect(seek).toHaveBeenCalledWith(65);
    expect(play).toHaveBeenCalled();
  });

  it("keeps the native panel lazy and on the frame route (SPEC-ADDON-080)", async () => {
    mocked.mockResolvedValue({
      file_id: "f1",
      drive: "drive1",
      timestamps: makeTimestamps(2),
    });
    renderSection();

    expect(screen.getByRole("button", { name: /CLIP Frames/ })).toBeInTheDocument();
    await Promise.resolve();
    expect(mocked).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: /CLIP Frames/ }));
    await waitFor(() => expect(screen.getAllByRole("img").length).toBe(2));
    const sources = screen.getAllByRole("img").map((img) => img.getAttribute("src"));
    expect(sources).toEqual([
      "/api/addons/intelligence/files/f1/frame?t=0",
      "/api/addons/intelligence/files/f1/frame?t=5",
    ]);
    expect(sceneTileUrl).not.toHaveBeenCalled();
  });
});
