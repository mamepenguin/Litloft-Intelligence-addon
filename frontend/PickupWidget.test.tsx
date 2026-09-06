/**
 * The drive-home Pickup carousel.
 *
 * Two things here are load-bearing rather than cosmetic.
 *
 * It must ask for the day's *window*, not the head of the feed. Lanes
 * emit at positions spaced by the reciprocal of their weight, so the
 * first dozen rows belong to the heaviest lanes and a quiet interest
 * has not appeared at all — measured, 6 lanes of 24 at depth 12 against
 * all 24 by depth 40. Twelve sampled from the top forty track the
 * proportions the weighting intends.
 *
 * And the link through must not appear before there is a feed worth
 * visiting.
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";

vi.mock("@/addons/intelligence/api", () => ({
  fetchPickup: vi.fn(),
}));

vi.mock("@/lib/api", () => ({
  batchGetFiles: vi.fn(),
}));

vi.mock("next-intl", () => ({
  useTranslations: () => (key: string) => key,
}));

vi.mock("@/components/CarouselSection", () => ({
  CarouselSection: ({
    files,
    loading,
    seeAllHref,
    totalCount,
  }: {
    files: { id: string }[];
    loading: boolean;
    seeAllHref?: string;
    totalCount?: number;
  }) => (
    <div data-testid="carousel" data-loading={String(loading)}>
      <span data-testid="see-all">{seeAllHref ?? ""}</span>
      <span data-testid="total">{totalCount ?? ""}</span>
      {files.map((f) => (
        <span key={f.id} data-testid="card">
          {f.id}
        </span>
      ))}
    </div>
  ),
}));

import PickupWidget from "./PickupWidget";
import { fetchPickup } from "./api";
import { batchGetFiles } from "@/lib/api";

const mockFetch = vi.mocked(fetchPickup);
const mockBatch = vi.mocked(batchGetFiles);

function file(id: string) {
  return { id, filename: `${id}.mp4` } as never;
}

beforeEach(() => {
  vi.clearAllMocks();
  mockBatch.mockImplementation(async (ids: string[]) => ids.map(file) as never);
});

describe("PickupWidget", () => {
  it("asks for the day's window, not the head of the feed", async () => {
    mockFetch.mockResolvedValue({ file_ids: ["a", "b"], total: 300 });

    render(<PickupWidget drive="videos" />);

    await waitFor(() => expect(mockFetch).toHaveBeenCalled());
    expect(mockFetch).toHaveBeenCalledWith(
      "videos",
      expect.objectContaining({ daily: true, limit: 12 }),
    );
  });

  it("renders the files the window returned", async () => {
    mockFetch.mockResolvedValue({ file_ids: ["a", "b"], total: 300 });

    render(<PickupWidget drive="videos" />);

    await waitFor(() =>
      expect(screen.getAllByTestId("card").map((n) => n.textContent)).toEqual([
        "a",
        "b",
      ]),
    );
  });

  it("links to the feed whenever there is one", async () => {
    mockFetch.mockResolvedValue({ file_ids: ["a"], total: 40 });

    render(<PickupWidget drive="videos" />);

    await waitFor(() =>
      expect(screen.getByTestId("see-all").textContent).toBe(
        "/drive/videos/addons/intelligence/pickup",
      ),
    );
  });

  it("links to it below forty too, where the row cannot show it all", async () => {
    // The link used to appear only at forty, back when the row drew all
    // twelve of the day's window and forty was the smallest feed the
    // page was worth opening for. The row draws four on a phone now, so
    // that gate left the other twenty-one of a twenty-five-file feed
    // with no way to reach them.
    mockFetch.mockResolvedValue({ file_ids: ["a"], total: 25 });

    render(<PickupWidget drive="videos" />);

    await waitFor(() =>
      expect(screen.getByTestId("see-all").textContent).toBe(
        "/drive/videos/addons/intelligence/pickup",
      ),
    );
  });

  it("puts the size of the feed on the link, not the size of the window", async () => {
    // The row is handed the day's twelve; the link leads to all 300, and
    // the number beside it has to be the number the reader arrives at.
    mockFetch.mockResolvedValue({ file_ids: ["a", "b"], total: 300 });

    render(<PickupWidget drive="videos" />);

    await waitFor(() =>
      expect(screen.getByTestId("total").textContent).toBe("300"),
    );
  });

  it("percent-encodes a non-ASCII drive in the link", async () => {
    mockFetch.mockResolvedValue({ file_ids: ["a"], total: 100 });

    render(<PickupWidget drive="動画" />);

    await waitFor(() =>
      expect(screen.getByTestId("see-all").textContent).toBe(
        "/drive/%E5%8B%95%E7%94%BB/addons/intelligence/pickup",
      ),
    );
  });

  it("renders nothing when the feed is empty", async () => {
    mockFetch.mockResolvedValue({ file_ids: [], total: 0 });

    const { container } = render(<PickupWidget drive="videos" />);

    await waitFor(() => expect(container.firstChild).toBeNull());
    expect(mockBatch).not.toHaveBeenCalled();
  });

  it("renders nothing without a drive", async () => {
    const { container } = render(<PickupWidget />);

    await waitFor(() => expect(container.firstChild).toBeNull());
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it("survives a failing request without throwing", async () => {
    mockFetch.mockRejectedValue(new Error("boom"));

    const { container } = render(<PickupWidget drive="videos" />);

    await waitFor(() => expect(container.firstChild).toBeNull());
  });
});
