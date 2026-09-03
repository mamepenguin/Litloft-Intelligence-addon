/**
 * Tests for SimilarFilesSection's disclosure shape.
 *
 * The section used to draw a heading and a "Find similar files" button
 * on every file detail page whether or not anyone ever pressed it —
 * detection is heavy on the backend, so it has always been opt-in, and
 * an opt-in with a permanent heading is a row that only says a feature
 * exists. It is now the same collapsed disclosure as the other two
 * derived views, with the button waiting inside.
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

function renderSection() {
  return render(<SimilarFilesSection fileId="f1" drive="media" />);
}

const header = () => screen.getByRole("button", { name: /Similar files/ });

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(getSimilarFiles).mockResolvedValue({ results } as never);
});

describe("SimilarFilesSection", () => {
  it("starts collapsed, with no detect button on the page", () => {
    renderSection();

    expect(header()).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByRole("button", { name: /Find similar files/ })).toBeNull();
    // Nothing is fetched until someone asks.
    expect(getSimilarFiles).not.toHaveBeenCalled();
  });

  it("keeps the detect button reachable inside the disclosure", async () => {
    renderSection();
    fireEvent.click(header());

    expect(header()).toHaveAttribute("aria-expanded", "true");
    fireEvent.click(screen.getByRole("button", { name: /Find similar files/ }));

    await waitFor(() => expect(getSimilarFiles).toHaveBeenCalledWith("f1", "media"));
    expect(await screen.findByText("neighbour.mp4")).toBeInTheDocument();
  });

  it("names the count in the header once results are in", async () => {
    renderSection();
    fireEvent.click(header());
    fireEvent.click(screen.getByRole("button", { name: /Find similar files/ }));

    await screen.findByText("neighbour.mp4");
    expect(header().textContent).toContain("(1)");
  });

  it("collapses again without losing the results", async () => {
    renderSection();
    fireEvent.click(header());
    fireEvent.click(screen.getByRole("button", { name: /Find similar files/ }));
    await screen.findByText("neighbour.mp4");

    fireEvent.click(header());
    expect(screen.queryByText("neighbour.mp4")).toBeNull();
    fireEvent.click(header());
    expect(screen.getByText("neighbour.mp4")).toBeInTheDocument();
    // Detection ran once, not once per open.
    expect(getSimilarFiles).toHaveBeenCalledTimes(1);
  });

  it("reports an unavailable backend inside the disclosure", async () => {
    vi.mocked(getSimilarFiles).mockRejectedValue(new Error("boom"));
    vi.useFakeTimers();
    try {
      renderSection();
      fireEvent.click(header());
      fireEvent.click(screen.getByRole("button", { name: /Find similar files/ }));
      // Two backoff waits before it gives up.
      await act(async () => {
        await vi.advanceTimersByTimeAsync(20_000);
      });
      expect(
        screen.getByText(/Similar file detection unavailable/i),
      ).toBeInTheDocument();
    } finally {
      vi.useRealTimers();
    }
  });
});
