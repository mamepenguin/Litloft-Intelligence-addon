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
  default: ({
    children,
    href,
  }: {
    children: React.ReactNode;
    href: string;
  }) => <a href={href}>{children}</a>,
}));

const replace = vi.fn();
vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace }),
  usePathname: () => "/files/f1",
  useSearchParams: () => new URLSearchParams("t=5"),
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

function controller() {
  return { seek: vi.fn(), play: vi.fn() } as never;
}

describe("RelatedPassagesSection", () => {
  beforeEach(() => {
    getRelatedPassages.mockReset();
    getRelatedPassages.mockResolvedValue({ results: [] });
    replace.mockReset();
  });

  it("runs on open, without being asked", async () => {
    getRelatedPassages.mockResolvedValue({ results: [pair()] });

    render(<RelatedPassagesSection fileId="f1" drive="main" />);

    expect(await screen.findByTestId("match-passage")).toBeTruthy();
    expect(getRelatedPassages).toHaveBeenCalledWith("f1", "main");
  });

  it("renders nothing at all when there is no connection", async () => {
    const { container } = render(
      <RelatedPassagesSection fileId="f1" drive="main" />,
    );

    await waitFor(() => expect(getRelatedPassages).toHaveBeenCalled());
    // The section being on the page is the signal that it found
    // something. Only about a third of files produce a pair, and a
    // viewer who scrolls to a "nothing found" line has worked for
    // nothing.
    await waitFor(() => expect(container.firstChild).toBeNull());
  });

  it("shows the other file's passage, and links to it", async () => {
    getRelatedPassages.mockResolvedValue({ results: [pair()] });

    render(<RelatedPassagesSection fileId="f1" drive="main" />);

    // Byte-for-byte: nothing was summarised.
    expect((await screen.findByTestId("match-passage")).textContent).toBe(
      THEIRS,
    );

    const link = screen.getByText(/rag-design-notes\.md/);
    expect(link.closest("a")?.getAttribute("href")).toBe("/files/n1?page=3");
  });

  it("keeps this file's own passage behind the toggle", async () => {
    getRelatedPassages.mockResolvedValue({ results: [pair()] });

    render(<RelatedPassagesSection fileId="f1" drive="main" />);
    await screen.findByTestId("match-passage");

    // The reader is already reading this file. Spending the widest part
    // of the card on text they have is what made the row unreadable.
    expect(screen.queryByTestId("source-passage")).toBeNull();

    fireEvent.click(
      screen.getByLabelText("relatedPassagesExpand:rag-design-notes.md"),
    );

    const mine = await screen.findByTestId("source-passage");
    expect(mine.textContent).toBe(MINE);
    expect(screen.getByTestId("match-passage").textContent).toBe(THEIRS);
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
    await screen.findByTestId("match-passage");

    getRelatedPassages.mockResolvedValue({
      results: [pair({ file_id: "n9", filename: "other-note.md" })],
    });
    rerender(<RelatedPassagesSection fileId="f2" drive="main" />);

    // The previous file's passages must not linger under the new file.
    await waitFor(() =>
      expect(getRelatedPassages).toHaveBeenCalledWith("f2", "main"),
    );
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

    expect(await screen.findByTestId("match-passage")).toBeTruthy();
    expect(getRelatedPassages).toHaveBeenCalledWith("f1", "main");
  });

  it("clamps the passage until the toggle is pressed", async () => {
    getRelatedPassages.mockResolvedValue({ results: [pair()] });

    render(<RelatedPassagesSection fileId="f1" drive="main" />);

    const clamped = (await screen.findByTestId("match-passage")).parentElement;
    expect(clamped?.className).toContain("line-clamp-2");

    fireEvent.click(
      screen.getByLabelText("relatedPassagesExpand:rag-design-notes.md"),
    );

    await waitFor(() =>
      expect(screen.getByTestId("match-passage").className).not.toContain(
        "line-clamp-2",
      ),
    );
  });

  it("does not expand when the passage itself is clicked", async () => {
    getRelatedPassages.mockResolvedValue({ results: [pair()] });

    render(<RelatedPassagesSection fileId="f1" drive="main" />);

    fireEvent.click(await screen.findByTestId("match-passage"));

    // Selecting a passage is how it reaches the quotation basket, so the
    // text must not double as a toggle.
    expect(screen.queryByTestId("source-passage")).toBeNull();
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

  describe("the anchor into this file", () => {
    it("seeks the player instead of leaving the page", async () => {
      const media = controller();
      getRelatedPassages.mockResolvedValue({
        results: [
          pair({ source: { text: MINE, page: null, timestamp: 663.4 } }),
        ],
      });

      render(
        <RelatedPassagesSection
          fileId="f1"
          drive="main"
          mediaController={media}
        />,
      );

      fireEvent.click(await screen.findByTestId("source-anchor"));

      // The connection is the point; a round trip to another route is not.
      expect(
        (media as unknown as { seek: ReturnType<typeof vi.fn> }).seek,
      ).toHaveBeenCalledWith(663.4);
    });

    it("moves a document to the page without dropping other params", async () => {
      getRelatedPassages.mockResolvedValue({
        results: [pair({ source: { text: MINE, page: 12, timestamp: null } })],
      });

      render(<RelatedPassagesSection fileId="f1" drive="main" />);

      fireEvent.click(await screen.findByTestId("source-anchor"));

      expect(replace).toHaveBeenCalledWith("/files/f1?t=5&page=12", {
        scroll: false,
      });
    });

    it("is plain text when nothing can act on it", async () => {
      getRelatedPassages.mockResolvedValue({
        results: [
          pair({ source: { text: MINE, page: null, timestamp: 663.4 } }),
        ],
      });

      // Media with no controller: the player has published none, or the
      // file failed to load and never will.
      render(<RelatedPassagesSection fileId="f1" drive="main" />);

      const anchor = await screen.findByTestId("source-anchor");
      // DESIGN.md §2.5 — an affordance that will never act must read as
      // the prose around it, not as a dimmed button.
      expect(anchor.tagName).toBe("P");
      expect(anchor.className).not.toContain("opacity");
    });

    it("is absent entirely when the passage has no locator", async () => {
      getRelatedPassages.mockResolvedValue({ results: [pair()] });

      render(<RelatedPassagesSection fileId="f1" drive="main" />);
      await screen.findByTestId("match-passage");

      // A plain text file has neither a page nor a timestamp. No
      // placeholder, no reserved space — the row simply starts lower.
      expect(screen.queryByTestId("source-anchor")).toBeNull();
    });
  });

  describe("the windowed excerpt", () => {
    const SEVERED =
      "回対象になっています。対角線を軸として回転させると元と同じ立方体になります。";

    it("opens on a whole sentence, not the severed word before it", async () => {
      getRelatedPassages.mockResolvedValue({
        results: [pair({ match: { text: SEVERED, page: 3, timestamp: null } })],
      });

      render(<RelatedPassagesSection fileId="f1" drive="main" />);

      const shown = await screen.findByTestId("match-passage");
      expect(shown.textContent?.startsWith("対角線を軸として")).toBe(true);
    });

    it("keeps the leading ellipsis out of the quotable text", async () => {
      getRelatedPassages.mockResolvedValue({
        results: [pair({ match: { text: SEVERED, page: 3, timestamp: null } })],
      });

      render(<RelatedPassagesSection fileId="f1" drive="main" />);

      const shown = await screen.findByTestId("match-passage");
      // The passage feeds Knowledge's quotation basket. A character the
      // author never wrote must not be inside the selectable range.
      expect(shown.textContent).not.toContain("…");
      const ellipsis = shown.parentElement?.firstElementChild;
      expect(ellipsis?.textContent).toBe("…");
      expect(ellipsis?.className).toContain("select-none");
    });

    it("shows the full passage once expanded", async () => {
      getRelatedPassages.mockResolvedValue({
        results: [pair({ match: { text: SEVERED, page: 3, timestamp: null } })],
      });

      render(<RelatedPassagesSection fileId="f1" drive="main" />);
      fireEvent.click(
        await screen.findByLabelText(
          "relatedPassagesExpand:rag-design-notes.md",
        ),
      );

      // Windowing is a display choice, not a claim about the text: the
      // whole chunk is one press away and byte-identical.
      expect((await screen.findByTestId("match-passage")).textContent).toBe(
        SEVERED,
      );
    });
  });

  describe("passageHref", () => {
    it("lands on the passage, not merely on the file", () => {
      expect(
        passageHref("v1", { text: "", page: null, timestamp: 724.6 }),
      ).toBe("/files/v1?t=724");
      expect(passageHref("d1", { text: "", page: 12, timestamp: null })).toBe(
        "/files/d1?page=12",
      );
      expect(passageHref("t1", { text: "", page: null, timestamp: null })).toBe(
        "/files/t1",
      );
    });
  });
});
