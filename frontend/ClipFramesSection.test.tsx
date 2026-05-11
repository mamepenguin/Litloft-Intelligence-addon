/**
 * Tests for ClipFramesSection — collapsed-by-default + filmstrip UX.
 *
 * Spec / hako: KMScjkyqGb2vy9zluV4Bp
 *
 * Covers:
 *   1. Closed-by-default: NO API call fires on mount. The section header
 *      is visible but `getClipTimestamps` must not run.
 *   2. Expand toggle fetches once. Re-collapse → re-expand does NOT
 *      re-fetch.
 *   3. Frames are laid out in a horizontally scrollable strip (no
 *      multi-row grid).
 *   4. When the API reports more timestamps than `PAGE_SIZE`, the
 *      sentinel is rendered for the IntersectionObserver-driven
 *      infinite scroll.
 *   5. When the API reports zero timestamps, the section unmounts so the
 *      file detail page doesn't show an empty section.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  render,
  screen,
  fireEvent,
  waitFor,
  cleanup,
} from "@testing-library/react";
import { NextIntlClientProvider } from "next-intl";

vi.mock("@/addons/intelligence/api", () => ({
  getClipTimestamps: vi.fn(),
  getFrameUrl: (fileId: string, t: number) =>
    `/api/addons/intelligence/files/${fileId}/frame?t=${t}`,
}));

import ClipFramesSection from "@/addons/intelligence/ClipFramesSection";
import { getClipTimestamps } from "@/addons/intelligence/api";

const mocked = getClipTimestamps as unknown as ReturnType<typeof vi.fn>;

function makeTimestamps(n: number) {
  return Array.from({ length: n }, (_, i) => ({
    start: i * 5,
    content_preview: `Frame at ${i * 5}s`,
  }));
}

function renderSection(props: {
  fileId?: string;
  fileType?: string;
  mimeType?: string;
} = {}) {
  return render(
    <NextIntlClientProvider
      locale="en"
      messages={{ searchIndex: { clipTitle: "CLIP Frames" } }}
    >
      <ClipFramesSection
        fileId={props.fileId ?? "f1"}
        drive="drive1"
        fileType={props.fileType ?? "video"}
        mimeType={props.mimeType ?? "video/mp4"}
      />
    </NextIntlClientProvider>,
  );
}

describe("ClipFramesSection", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // jsdom doesn't ship IntersectionObserver — install a minimal stub
    // good enough for the component to mount without crashing.
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

  afterEach(() => {
    cleanup();
  });

  it("does NOT fetch timestamps on mount (closed by default)", async () => {
    mocked.mockResolvedValue({ available: true, timestamps: makeTimestamps(3) });
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
    mocked.mockResolvedValue({ available: true, timestamps: makeTimestamps(3) });
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

  it("renders all frames inside a single horizontally-scrollable strip", async () => {
    mocked.mockResolvedValue({ available: true, timestamps: makeTimestamps(5) });
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
    expect(strip.className).toMatch(/overflow-x-auto/);
    expect(strip.className).toMatch(/flex/);
    // Filmstrip is one row, never a multi-row grid.
    expect(strip.className).not.toMatch(/grid/);
  });

  it("hides the section entirely when there are no CLIP frames", async () => {
    mocked.mockResolvedValue({ available: false });
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
    mocked.mockResolvedValue({ available: true, timestamps: makeTimestamps(3) });
    const { container } = renderSection({ fileType: "image", mimeType: "image/jpeg" });

    expect(container.firstChild).toBeNull();
    await Promise.resolve();
    expect(mocked).not.toHaveBeenCalled();
  });

  it("renders nothing (no header, no fetch) for .loft remote embeds", async () => {
    mocked.mockResolvedValue({ available: true, timestamps: makeTimestamps(3) });
    const { container } = renderSection({
      fileType: "video",
      mimeType: "application/vnd.litloft.loft+json",
    });

    expect(container.firstChild).toBeNull();
    await Promise.resolve();
    expect(mocked).not.toHaveBeenCalled();
  });

  it("renders the infinite-scroll sentinel when more frames than PAGE_SIZE", async () => {
    mocked.mockResolvedValue({ available: true, timestamps: makeTimestamps(35) });
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
