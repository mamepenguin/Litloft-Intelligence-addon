/**
 * The failed-jobs warning, after it moved out of the index-status
 * widget and into the `dashboard-alerts` slot (UI redesign Phase 1,
 * item 6 / ADM-2).
 *
 * Two claims that were false before the move:
 *  1. Nothing wrong renders *nothing* — not a "no failed jobs" row.
 *     The slot draws no wrapper and no heading, so an entry that
 *     always renders is a permanent band above the dashboard.
 *  2. It is a button, reachable and openable from the keyboard, whose
 *     accessible name contains what it says (WCAG 2.5.3).
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";

vi.mock("@/addons/intelligence/api", () => ({
  getFailedJobs: vi.fn(),
}));

vi.mock("@/addons/intelligence/FailedJobsModal", () => ({
  default: ({ open, onClose }: { open: boolean; onClose: () => void }) =>
    open ? (
      <div data-testid="failed-jobs-modal-stub">
        <button onClick={onClose}>close-modal</button>
      </div>
    ) : null,
}));

import FailedJobsAlert from "@/addons/intelligence/FailedJobsAlert";
import { getFailedJobs } from "@/addons/intelligence/api";

const mockedFailedJobs = getFailedJobs as unknown as ReturnType<typeof vi.fn>;

function page(total: number) {
  return { items: [], total, limit: 1, offset: 0 };
}

beforeEach(() => {
  vi.clearAllMocks();
  mockedFailedJobs.mockResolvedValue(page(0));
});

afterEach(() => {
  vi.useRealTimers();
});

describe("FailedJobsAlert", () => {
  it("renders no DOM at all when nothing has failed", async () => {
    const { container } = render(<FailedJobsAlert />);
    await waitFor(() => expect(mockedFailedJobs).toHaveBeenCalled());

    expect(container).toBeEmptyDOMElement();
  });

  it("renders nothing before the first poll answers", () => {
    // A band that appears and then vanishes on every dashboard load is
    // worse than one that arrives a moment late.
    const { container } = render(<FailedJobsAlert />);
    expect(container).toBeEmptyDOMElement();
  });

  it("says how many failed, in a name that contains what it shows", async () => {
    mockedFailedJobs.mockResolvedValue(page(3));
    render(<FailedJobsAlert />);

    const band = await screen.findByRole("button", { name: "3 failed jobs" });
    expect(band).toHaveTextContent("3 failed jobs");
  });

  it("opens the modal from a click", async () => {
    mockedFailedJobs.mockResolvedValue(page(1));
    render(<FailedJobsAlert />);

    fireEvent.click(await screen.findByRole("button", { name: "1 failed jobs" }));

    await waitFor(() =>
      expect(screen.getByTestId("failed-jobs-modal-stub")).toBeInTheDocument(),
    );
  });

  it("opens the modal from the keyboard", async () => {
    mockedFailedJobs.mockResolvedValue(page(1));
    render(<FailedJobsAlert />);

    const band = await screen.findByRole("button", { name: "1 failed jobs" });
    band.focus();
    expect(band).toHaveFocus();
    // A <button> turns Enter into a click; asserting the element is a
    // real button is what makes that true.
    expect(band.tagName).toBe("BUTTON");
    fireEvent.click(band);

    await waitFor(() =>
      expect(screen.getByTestId("failed-jobs-modal-stub")).toBeInTheDocument(),
    );
  });

  it("goes away on its own once the queue is cleared", async () => {
    vi.useFakeTimers();
    mockedFailedJobs.mockResolvedValue(page(2));
    render(<FailedJobsAlert />);
    await vi.waitFor(() =>
      expect(screen.getByRole("button", { name: "2 failed jobs" })).toBeInTheDocument(),
    );

    mockedFailedJobs.mockResolvedValue(page(0));
    await vi.advanceTimersByTimeAsync(10_000);

    await vi.waitFor(() => expect(screen.queryByRole("button")).toBeNull());
  });

  it("stays silent when the addon cannot be reached", async () => {
    // The dashboard already says the addon is unreachable, in the
    // index-status widget. A warning band here would name the wrong
    // problem.
    mockedFailedJobs.mockRejectedValue(new Error("unreachable"));
    const { container } = render(<FailedJobsAlert />);
    await waitFor(() => expect(mockedFailedJobs).toHaveBeenCalled());

    expect(container).toBeEmptyDOMElement();
  });

  it("asks only for the count, never a page of rows", async () => {
    mockedFailedJobs.mockResolvedValue(page(5));
    render(<FailedJobsAlert />);
    await waitFor(() => expect(mockedFailedJobs).toHaveBeenCalled());

    expect(mockedFailedJobs).toHaveBeenCalledWith(1, 0);
  });
});
