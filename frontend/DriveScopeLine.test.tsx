/**
 * The line that says what a question will be asked of.
 *
 * The fail-silent half is the half worth testing: a wrong count here reads
 * as "the index holds this many of your files" and is believed, so the
 * line has to be absent rather than approximate when the call fails.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { act, cleanup, render, screen, waitFor } from "@testing-library/react";

vi.mock("next-intl", () => ({
  useTranslations: () => (key: string, values?: Record<string, unknown>) =>
    values ? `${key}:${values.drive}:${values.count}` : key,
}));

const getDrives = vi.fn();
// Spread the real module rather than replacing it: a stub that *is* the
// whole module turns tomorrow's second core call into `undefined is not a
// function`, which this component would render as silence — the one
// outcome it is not allowed to reach by accident.
vi.mock("@/lib/api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/api")>()),
  getDrives: (...args: unknown[]) => getDrives(...args),
}));

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

  /**
   * The failure path, held by its cause rather than by its symptom.
   *
   * "No line" is also what an unfinished fetch looks like, so asserting
   * absence alone passes with no error handling at all — deleting the
   * `try/catch` left every test here green. This asserts the rejection was
   * *handled*: an unhandled one fails the process, and the line still has
   * to be absent afterwards.
   */
  it("swallows the failure rather than letting it escape", async () => {
    const unhandled = vi.fn();
    process.on("unhandledRejection", unhandled);
    getDrives.mockRejectedValue(new Error("network down"));
    render(<DriveScopeLine drive="family" />);

    await waitFor(() => expect(getDrives).toHaveBeenCalled());
    await new Promise((r) => setTimeout(r, 0));
    process.off("unhandledRejection", unhandled);

    expect(unhandled).not.toHaveBeenCalled();
    expect(screen.queryByTestId("drive-scope")).toBeNull();
  });

  /**
   * The guard, held by what it prevents rather than by a warning.
   *
   * React stopped warning about a `setState` after unmount, so a test that
   * unmounts and watches the console asserts nothing — the first version of
   * this one passed with the guard deleted. What the guard actually stops is
   * observable while mounted: a slow answer for the drive the reader left
   * arriving after the fast answer for the drive they are on.
   */
  it("does not let a slow answer for the last drive overwrite this one", async () => {
    let resolveSlow: (v: unknown) => void = () => {};
    getDrives
      .mockReturnValueOnce(new Promise((r) => { resolveSlow = r; }))
      .mockResolvedValueOnce(drives);

    const { rerender } = render(<DriveScopeLine drive="family" />);
    rerender(<DriveScopeLine drive="work" />);
    expect(await screen.findByTestId("drive-scope")).toHaveTextContent(
      "scope:work:3",
    );

    await act(async () => {
      resolveSlow(drives);
    });

    // Still the drive on screen. Without the guard the first effect's
    // response lands here and relabels it 619.
    expect(screen.getByTestId("drive-scope")).toHaveTextContent("scope:work:3");
  });
});
