/**
 * Tests for SimilarFilesSection's disclosure shape and what starts it.
 *
 * The section used to draw a heading and a "Find similar files" button
 * on every file detail page whether or not anyone ever pressed it. The
 * heading became a disclosure first; the button is gone now too —
 * opening the disclosure is the request. That keeps detection off the
 * mount path (a file nobody opens this on never computes) while costing
 * the person who does want it a press rather than two.
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { act, render, screen, fireEvent, waitFor } from "@testing-library/react";

vi.mock("@/addons/intelligence/api", () => ({
  getSimilarFiles: vi.fn(),
}));

import SimilarFilesSection from "@/addons/intelligence/SimilarFilesSection";
import { getSimilarFiles } from "@/addons/intelligence/api";

const results = [
  {
    file_id: "other1",
    filename: "neighbour.mp4",
    shared_keywords: [{ word: "ocean", score: 0.9 }],
  },
];

function renderSection(fileId = "f1") {
  return render(<SimilarFilesSection fileId={fileId} drive="media" />);
}

const header = () => screen.getByRole("button", { name: /Similar files/ });

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(getSimilarFiles).mockResolvedValue({ results } as never);
});

describe("SimilarFilesSection", () => {
  it("starts collapsed and asks the backend for nothing", () => {
    renderSection();

    expect(header()).toHaveAttribute("aria-expanded", "false");
    // The mount path is the one this section must stay off: it is the
    // path every file detail page takes, opened or not.
    expect(getSimilarFiles).not.toHaveBeenCalled();
  });

  it("has no detect button, collapsed or expanded", async () => {
    renderSection();
    expect(screen.queryByRole("button", { name: /Find similar files/ })).toBeNull();

    fireEvent.click(header());
    expect(await screen.findByText("neighbour.mp4")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Find similar files/ })).toBeNull();
  });

  it("opening the disclosure is what fetches", async () => {
    renderSection();
    fireEvent.click(header());

    expect(header()).toHaveAttribute("aria-expanded", "true");
    expect(await screen.findByText("neighbour.mp4")).toBeInTheDocument();
    expect(getSimilarFiles).toHaveBeenCalledWith("f1", "media");
  });

  it("reserves the result height with ghosts while the answer is in flight", async () => {
    let settle: (value: unknown) => void = () => {};
    vi.mocked(getSimilarFiles).mockReturnValue(
      new Promise((resolve) => {
        settle = resolve;
      }) as never,
    );

    renderSection();
    fireEvent.click(header());

    const ghosts = await screen.findByTestId("similar-files-ghosts");
    // Six, the same `limit` the request carries, so a full result set
    // swaps in at the height the ghosts already took.
    expect(ghosts.children).toHaveLength(6);

    await act(async () => {
      settle({ results });
    });

    expect(await screen.findByText("neighbour.mp4")).toBeInTheDocument();
    expect(screen.queryByTestId("similar-files-ghosts")).toBeNull();
  });

  it("names the count in the header once results are in", async () => {
    renderSection();
    fireEvent.click(header());

    await screen.findByText("neighbour.mp4");
    expect(header().textContent).toContain("(1)");
  });

  it("collapses again without losing the results or asking twice", async () => {
    renderSection();
    fireEvent.click(header());
    await screen.findByText("neighbour.mp4");

    fireEvent.click(header());
    expect(screen.queryByText("neighbour.mp4")).toBeNull();
    fireEvent.click(header());
    expect(screen.getByText("neighbour.mp4")).toBeInTheDocument();
    // Detection ran once, not once per open.
    expect(getSimilarFiles).toHaveBeenCalledTimes(1);
  });

  it("keeps both backoff waits before it reports the backend unavailable", async () => {
    vi.mocked(getSimilarFiles).mockRejectedValue(new Error("boom"));
    vi.useFakeTimers();
    try {
      renderSection();
      fireEvent.click(header());

      // First failure: still trying, and still holding the height.
      await act(async () => {
        await vi.advanceTimersByTimeAsync(0);
      });
      expect(screen.getByTestId("similar-files-ghosts")).toBeInTheDocument();

      // 6 s carries it past the first backoff into the second attempt.
      await act(async () => {
        await vi.advanceTimersByTimeAsync(6_000);
      });
      expect(screen.getByTestId("similar-files-ghosts")).toBeInTheDocument();

      // 12 s more exhausts the second, and only then does it give up.
      await act(async () => {
        await vi.advanceTimersByTimeAsync(12_000);
      });
      expect(
        screen.getByText(/Similar file detection unavailable/i),
      ).toBeInTheDocument();
      expect(getSimilarFiles).toHaveBeenCalledTimes(3);
    } finally {
      vi.useRealTimers();
    }
  });

  it("offers a retry that succeeds after an unavailable backend", async () => {
    vi.mocked(getSimilarFiles).mockRejectedValue(new Error("boom"));
    vi.useFakeTimers();
    try {
      renderSection();
      fireEvent.click(header());
      await act(async () => {
        await vi.advanceTimersByTimeAsync(20_000);
      });
      expect(
        screen.getByText(/Similar file detection unavailable/i),
      ).toBeInTheDocument();
    } finally {
      vi.useRealTimers();
    }

    vi.mocked(getSimilarFiles).mockResolvedValue({ results } as never);
    fireEvent.click(screen.getByRole("button", { name: /Search again/ }));

    expect(await screen.findByText("neighbour.mp4")).toBeInTheDocument();
  });

  it("goes back to asking nothing when the file changes under it", async () => {
    const { rerender } = renderSection("f1");
    fireEvent.click(header());
    await screen.findByText("neighbour.mp4");
    expect(getSimilarFiles).toHaveBeenCalledTimes(1);

    rerender(<SimilarFilesSection fileId="f2" drive="media" />);

    await waitFor(() =>
      expect(header()).toHaveAttribute("aria-expanded", "false"),
    );
    expect(screen.queryByText("neighbour.mp4")).toBeNull();
    // The new file gets the same deal the old one did: nothing until asked.
    expect(getSimilarFiles).toHaveBeenCalledTimes(1);
  });
});
