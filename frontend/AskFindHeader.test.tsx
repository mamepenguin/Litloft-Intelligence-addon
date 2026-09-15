/**
 * The header Ask and Find share, and its scope line.
 *
 * The failure half is the half worth testing: a wrong count reads as "the
 * index holds this many of your files" and is believed, so a count that did
 * not arrive leaves the drive's name alone, never a number.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { act, cleanup, render, screen, waitFor } from "@testing-library/react";

vi.mock("next-intl", () => ({
  useTranslations:
    (ns: string) => (key: string, values?: Record<string, unknown>) =>
      key === "driveScope"
        ? `${values!.drive} · ${values!.detail}`
        : key === "items"
          ? `${values!.count} items`
          : `${ns}.${key}`,
}));
vi.mock("./ModeTabs", () => ({
  default: ({ current, query, drive }: { current: string; query: string; drive: string }) => (
    <nav data-testid="mode-tabs" data-current={current} data-query={query} data-drive={drive} />
  ),
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

const { AskFindHeader } = await import("./AskFindHeader");

const drives = [
  { name: "family", file_count: 619 },
  { name: "work", file_count: 3 },
];

describe("AskFindHeader", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });
  afterEach(cleanup);

  it("names the drive it is scoped to and its count", async () => {
    getDrives.mockResolvedValue(drives);
    render(<AskFindHeader current="ask" query="" drive="family" />);

    // The count of the drive asked for, not of the first one that came
    // back: both are in the response and only one is the subject.
    expect(await screen.findByTestId("drive-scope")).toHaveTextContent(
      "family · 619 items",
    );
  });

  it("names the drive alone when the call fails", async () => {
    getDrives.mockRejectedValue(new Error("network down"));
    render(<AskFindHeader current="ask" query="" drive="family" />);

    await waitFor(() => expect(getDrives).toHaveBeenCalled());
    expect(screen.getByTestId("drive-scope").textContent).toBe("family");
  });

  it("gives no count for a drive the caller cannot see", async () => {
    // Access control lives in the response: core returns only the drives
    // this viewer may open, so an absent one is not a zero.
    getDrives.mockResolvedValue([{ name: "work", file_count: 3 }]);
    render(<AskFindHeader current="ask" query="" drive="family" />);

    await waitFor(() => expect(getDrives).toHaveBeenCalled());
    expect(screen.getByTestId("drive-scope").textContent).toBe("family");
  });

  it("offers neither a scope nor tabs before a drive is known", () => {
    render(<AskFindHeader current="ask" query="" drive={null} />);
    expect(getDrives).not.toHaveBeenCalled();
    expect(screen.queryByTestId("drive-scope")).toBeNull();
    expect(screen.queryByTestId("mode-tabs")).toBeNull();
  });

  // Ask and Find render this same header, so switching tabs keeps the title,
  // the icon and the scope line where they were.
  it.each(["ask", "find"] as const)("titles the %s mode with the sidebar row's label and icon", async (mode) => {
    getDrives.mockResolvedValue(drives);
    const { container } = render(<AskFindHeader current={mode} query="cats" drive="family" />);
    await screen.findByText("family · 619 items");
    expect(screen.getByRole("heading", { level: 1 }).textContent).toBe("intelligence.nav.label");
    expect(container.querySelector("svg.lucide-message-circle-question-mark")).not.toBeNull();
    const tabs = screen.getByTestId("mode-tabs");
    expect(tabs.dataset).toMatchObject({ current: mode, query: "cats", drive: "family" });
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
    render(<AskFindHeader current="ask" query="" drive="family" />);

    await waitFor(() => expect(getDrives).toHaveBeenCalled());
    await new Promise((r) => setTimeout(r, 0));
    process.off("unhandledRejection", unhandled);

    expect(unhandled).not.toHaveBeenCalled();
    expect(screen.getByTestId("drive-scope").textContent).toBe("family");
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

    const { rerender } = render(<AskFindHeader current="ask" query="" drive="family" />);
    rerender(<AskFindHeader current="ask" query="" drive="work" />);
    expect(await screen.findByTestId("drive-scope")).toHaveTextContent(
      "work · 3 items",
    );

    await act(async () => {
      resolveSlow(drives);
    });

    // Still the drive on screen. Without the guard the first effect's
    // response lands here and relabels it 619.
    expect(screen.getByTestId("drive-scope")).toHaveTextContent("work · 3 items");
  });
});
