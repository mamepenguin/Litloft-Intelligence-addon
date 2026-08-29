import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";

const getRelatedPassages = vi.fn();
vi.mock("./api", () => ({
  getRelatedPassages: (...args: unknown[]) => getRelatedPassages(...args),
}));

vi.mock("next-intl", () => ({
  // Params are echoed so a test can assert an accessible name actually
  // varies per card.
  useTranslations: () => (key: string, params?: Record<string, string>) =>
    params ? `${key}:${Object.values(params).join(",")}` : key,
}));

vi.mock("next/link", () => ({
  default: ({ children, href }: { children: React.ReactNode; href: string }) => (
    <a href={href}>{children}</a>
  ),
}));

import RelatedPassagesSection, { passageHref } from "./RelatedPassagesSection";

const MINE =
  "A 400-character chunk is far too short for expository prose; an " +
  "argument that spans two pages gets shredded into fragments.";
const THEIRS =
  "At 400 characters a paragraph gets split across two chunks, and " +
  "neither half carries the claim it was part of.";

function pair(overrides: Record<string, unknown> = {}) {
  return {
    source: { text: MINE, page: null, timestamp: null },
    match: { text: THEIRS, page: 3, timestamp: null },
    file_id: "n1",
    drive: "main",
    filename: "rag-design-notes.md",
    score: 0.91,
    ...overrides,
  };
}

describe("RelatedPassagesSection", () => {
  beforeEach(() => {
    getRelatedPassages.mockReset();
    getRelatedPassages.mockResolvedValue({ results: [] });
  });

  it("runs on open, without being asked", async () => {
    getRelatedPassages.mockResolvedValue({ results: [pair()] });

    render(<RelatedPassagesSection fileId="f1" drive="main" />);

    expect(await screen.findByTestId("source-passage")).toBeTruthy();
    expect(getRelatedPassages).toHaveBeenCalledWith("f1", "main");
  });

  it("renders nothing at all when there is no connection", async () => {
    const { container } = render(
      <RelatedPassagesSection fileId="f1" drive="main" />,
    );

    await waitFor(() => expect(getRelatedPassages).toHaveBeenCalled());
    // The section being on the page is the signal that it found
    // something. Half of files produce no pair, and a viewer who
    // scrolls to a "nothing found" line has worked for nothing.
    await waitFor(() => expect(container.firstChild).toBeNull());
  });

  it("shows both passages verbatim, and links to the other one's page", async () => {
    getRelatedPassages.mockResolvedValue({ results: [pair()] });

    render(<RelatedPassagesSection fileId="f1" drive="main" />);

    const mine = await screen.findByTestId("source-passage");
    const theirs = screen.getByTestId("match-passage");
    // Byte-for-byte: nothing was summarised on either side.
    expect(mine.textContent).toBe(MINE);
    expect(theirs.textContent).toBe(THEIRS);

    const link = screen.getByText(/rag-design-notes\.md/);
    expect(link.closest("a")?.getAttribute("href")).toBe("/files/n1?page=3");
  });

  it("reports a failed lookup instead of vanishing", async () => {
    getRelatedPassages.mockRejectedValue(new Error("proxy timeout"));

    render(<RelatedPassagesSection fileId="f1" drive="main" />);

    // Silence would have hidden a 15-second timeout once already.
    expect(await screen.findByText("relatedPassagesUnavailable")).toBeTruthy();
  });

  it("re-runs when the viewer navigates to another file", async () => {
    getRelatedPassages.mockResolvedValue({ results: [pair()] });

    const { rerender } = render(
      <RelatedPassagesSection fileId="f1" drive="main" />,
    );
    await screen.findByTestId("source-passage");

    getRelatedPassages.mockResolvedValue({
      results: [pair({ file_id: "n9", filename: "other-note.md" })],
    });
    rerender(<RelatedPassagesSection fileId="f2" drive="main" />);

    // The previous file's passages must not linger under the new file.
    await waitFor(() => expect(getRelatedPassages).toHaveBeenCalledWith("f2", "main"));
    expect(await screen.findByText(/other-note\.md/)).toBeTruthy();
  });

  it("waits for the drive before asking, and asks once it lands", async () => {
    getRelatedPassages.mockResolvedValue({ results: [pair()] });

    // The /files/{id} route renders this while its own getFile is still
    // in flight, so the drive starts empty. Firing then would hit the
    // proxy without X-Lit-Drive and be rejected — and the failure would
    // stick, because the drive arriving is not a new file.
    const { rerender } = render(
      <RelatedPassagesSection fileId="f1" drive="" />,
    );
    expect(getRelatedPassages).not.toHaveBeenCalled();

    rerender(<RelatedPassagesSection fileId="f1" drive="main" />);

    expect(await screen.findByTestId("source-passage")).toBeTruthy();
    expect(getRelatedPassages).toHaveBeenCalledWith("f1", "main");
  });

  it("clamps both passages until the chevron is pressed", async () => {
    getRelatedPassages.mockResolvedValue({ results: [pair()] });

    render(<RelatedPassagesSection fileId="f1" drive="main" />);

    const mine = await screen.findByTestId("source-passage");
    // Five pairs of full transcript chunks is four thousand characters.
    expect(mine.className).toContain("line-clamp-2");
    expect(screen.getByTestId("match-passage").className).toContain(
      "line-clamp-2",
    );

    fireEvent.click(
      screen.getByLabelText("relatedPassagesExpand:rag-design-notes.md"),
    );

    await waitFor(() =>
      expect(screen.getByTestId("source-passage").className).not.toContain(
        "line-clamp-2",
      ),
    );
    expect(screen.getByTestId("match-passage").className).not.toContain(
      "line-clamp-2",
    );
  });

  it("does not expand when the passage itself is clicked", async () => {
    getRelatedPassages.mockResolvedValue({ results: [pair()] });

    render(<RelatedPassagesSection fileId="f1" drive="main" />);

    fireEvent.click(await screen.findByTestId("source-passage"));

    // Selecting a passage is how it reaches the quotation basket, so the
    // text must not double as a toggle.
    expect(screen.getByTestId("source-passage").className).toContain(
      "line-clamp-2",
    );
  });

  it("keeps the locator out of the truncated filename", async () => {
    getRelatedPassages.mockResolvedValue({
      results: [
        pair({
          filename: "a-very-long-filename-that-will-be-truncated-somewhere.md",
          match: { text: THEIRS, page: null, timestamp: 195 },
        }),
      ],
    });

    render(<RelatedPassagesSection fileId="f1" drive="main" />);

    // The filename may be cut; where the link goes may not.
    const locator = await screen.findByText(/3:15/);
    expect(locator.className).toContain("shrink-0");
    expect(locator.className).not.toContain("truncate");
    expect(locator.closest("a")?.getAttribute("href")).toBe("/files/n1?t=195");
  });

  it("names each toggle after the file it opens", async () => {
    getRelatedPassages.mockResolvedValue({
      results: [pair(), pair({ file_id: "n2", filename: "second-note.md" })],
    });

    render(<RelatedPassagesSection fileId="f1" drive="main" />);

    // Several of these buttons sit in one list; "expand" on its own
    // tells a screen-reader user nothing about which card it opens.
    expect(
      await screen.findByLabelText("relatedPassagesExpand:rag-design-notes.md"),
    ).toBeTruthy();
    expect(
      screen.getByLabelText("relatedPassagesExpand:second-note.md"),
    ).toBeTruthy();
  });

  describe("passageHref", () => {
    it("lands on the passage, not merely on the file", () => {
      expect(passageHref("v1", { text: "", page: null, timestamp: 724.6 })).toBe(
        "/files/v1?t=724",
      );
      expect(passageHref("d1", { text: "", page: 12, timestamp: null })).toBe(
        "/files/d1?page=12",
      );
      expect(passageHref("t1", { text: "", page: null, timestamp: null })).toBe(
        "/files/t1",
      );
    });
  });
});
