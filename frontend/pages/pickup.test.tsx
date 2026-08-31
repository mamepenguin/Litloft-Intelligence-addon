/**
 * The Pickup feed page.
 *
 * Two request-lifecycle hazards, both of which are invisible until they
 * are not: a page that resolves after the viewer has switched drives,
 * and a hydration failure behind an IntersectionObserver that is
 * re-armed the moment loading ends.
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";

vi.mock("@/addons/intelligence/api", () => ({ fetchPickup: vi.fn() }));
vi.mock("@/lib/api", () => ({ batchGetFiles: vi.fn() }));
vi.mock("next-intl", () => ({ useTranslations: () => (k: string) => k }));

const currentDrive = { value: "videos" };
vi.mock("@/components/CurrentDriveProvider", () => ({
  useCurrentDrive: () => currentDrive.value,
}));

vi.mock("@/components/FileGrid", () => ({
  FileGrid: ({ files }: { files: { id: string }[] }) => (
    <div data-testid="grid">{files.map((f) => f.id).join(",")}</div>
  ),
}));

class NoopObserver {
  observe() {}
  disconnect() {}
}
globalThis.IntersectionObserver = NoopObserver as never;

import PickupPage from "./pickup";
import { fetchPickup } from "../api";
import { batchGetFiles } from "@/lib/api";

const mockFetch = vi.mocked(fetchPickup);
const mockBatch = vi.mocked(batchGetFiles);

const file = (id: string) => ({ id, filename: `${id}.mp4` }) as never;

beforeEach(() => {
  vi.clearAllMocks();
  currentDrive.value = "videos";
  mockBatch.mockImplementation(async (ids: string[]) => ids.map(file) as never);
});

describe("PickupPage", () => {
  it("renders the first page", async () => {
    mockFetch.mockResolvedValue({ file_ids: ["a", "b"], total: 2 });

    render(<PickupPage />);

    await waitFor(() =>
      expect(screen.getByTestId("grid").textContent).toBe("a,b"),
    );
  });

  it("pages in stable order without a window", async () => {
    mockFetch.mockResolvedValue({ file_ids: ["a"], total: 1 });

    render(<PickupPage />);

    await waitFor(() => expect(mockFetch).toHaveBeenCalled());
    expect(mockFetch).toHaveBeenCalledWith(
      "videos",
      expect.not.objectContaining({ daily: true }),
    );
  });

  it("stops paging when hydration fails instead of spinning", async () => {
    mockFetch.mockResolvedValue({ file_ids: ["a"], total: 100 });
    mockBatch.mockRejectedValue(new Error("hydration down"));

    render(<PickupPage />);

    await waitFor(() => expect(screen.getByText("pickup.retry")).toBeTruthy());
    const callsAfterFailure = mockFetch.mock.calls.length;
    // The sentinel is gone, so nothing re-triggers the same offset.
    await new Promise((r) => setTimeout(r, 20));
    expect(mockFetch.mock.calls.length).toBe(callsAfterFailure);
  });

  it("can be retried after a failure", async () => {
    mockFetch.mockResolvedValue({ file_ids: ["a"], total: 100 });
    mockBatch.mockRejectedValueOnce(new Error("blip"));

    render(<PickupPage />);
    await waitFor(() => expect(screen.getByText("pickup.retry")).toBeTruthy());

    mockBatch.mockImplementation(async (ids: string[]) => ids.map(file) as never);
    // Retrying must issue the request itself; the sentinel is already
    // in view and will not cross back in to trigger one.
    fireEvent.click(screen.getByText("pickup.retry"));

    await waitFor(() =>
      expect(screen.getByTestId("grid").textContent).toBe("a"),
    );
  });

  it("shows the empty state when the feed holds nothing", async () => {
    mockFetch.mockResolvedValue({ file_ids: [], total: 0 });

    render(<PickupPage />);

    await waitFor(() => expect(screen.getByText("pickup.empty")).toBeTruthy());
  });

  it("does not append a previous drive's files after a switch", async () => {
    let release: (v: { file_ids: string[]; total: number }) => void = () => {};
    mockFetch.mockImplementationOnce(
      () => new Promise((resolve) => {
        release = resolve;
      }),
    );

    const { rerender } = render(<PickupPage />);
    await waitFor(() => expect(mockFetch).toHaveBeenCalledTimes(1));

    // Switch drives while the first request is still in flight.
    currentDrive.value = "main";
    mockFetch.mockResolvedValue({ file_ids: ["m1"], total: 1 });
    rerender(<PickupPage />);

    release({ file_ids: ["v1", "v2"], total: 2 });

    await waitFor(() =>
      expect(screen.getByTestId("grid").textContent).toBe("m1"),
    );
    expect(screen.getByTestId("grid").textContent).not.toContain("v1");
  });
});
