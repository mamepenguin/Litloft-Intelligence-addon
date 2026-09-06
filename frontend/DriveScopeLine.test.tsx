/**
 * The line that says what a question will be asked of.
 *
 * The fail-silent half is the half worth testing: a wrong count here reads
 * as "the index holds this many of your files" and is believed, so the
 * line has to be absent rather than approximate when the call fails.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";

vi.mock("next-intl", () => ({
  useTranslations: () => (key: string, values?: Record<string, unknown>) =>
    values ? `${key}:${values.drive}:${values.count}` : key,
}));

const getDrives = vi.fn();
vi.mock("@/lib/api", () => ({ getDrives: (...args: unknown[]) => getDrives(...args) }));

const { DriveScopeLine } = await import("./DriveScopeLine");

const drives = [
  { name: "family", file_count: 619 },
  { name: "work", file_count: 3 },
];

describe("DriveScopeLine", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });
  afterEach(cleanup);

  it("names the drive it is scoped to and its count", async () => {
    getDrives.mockResolvedValue(drives);
    render(<DriveScopeLine drive="family" />);

    // The count of the drive asked for, not of the first one that came
    // back: both are in the response and only one is the subject.
    expect(await screen.findByTestId("drive-scope")).toHaveTextContent(
      "scope:family:619",
    );
  });

  it("says nothing when the call fails", async () => {
    getDrives.mockRejectedValue(new Error("network down"));
    render(<DriveScopeLine drive="family" />);

    await waitFor(() => expect(getDrives).toHaveBeenCalled());
    expect(screen.queryByTestId("drive-scope")).toBeNull();
  });

  it("says nothing about a drive the caller cannot see", async () => {
    // Access control lives in the response: core returns only the drives
    // this viewer may open, so an absent one is not a zero.
    getDrives.mockResolvedValue([{ name: "work", file_count: 3 }]);
    render(<DriveScopeLine drive="family" />);

    await waitFor(() => expect(getDrives).toHaveBeenCalled());
    expect(screen.queryByTestId("drive-scope")).toBeNull();
  });

  it("says nothing before a drive is known", () => {
    render(<DriveScopeLine drive={null} />);
    expect(getDrives).not.toHaveBeenCalled();
    expect(screen.queryByTestId("drive-scope")).toBeNull();
  });
});
