import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";

const getRelatedPassages = vi.fn();
vi.mock("./api", () => ({
  getRelatedPassages: (...args: unknown[]) => getRelatedPassages(...args),
}));

vi.mock("next-intl", () => ({
  useTranslations: () => (key: string) => key,
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

  it("does not search until asked, on a file nobody is being asked about", async () => {
    render(
      <RelatedPassagesSection
        fileId="f1" drive="main" trustTier="verified" trustReviewedAt={null}
      />,
    );

    expect(await screen.findByText("relatedPassages")).toBeTruthy();
    expect(getRelatedPassages).not.toHaveBeenCalled();

    fireEvent.click(screen.getByText("relatedPassagesFind"));
    await waitFor(() => expect(getRelatedPassages).toHaveBeenCalledWith("f1", "main"));
  });

  it("shows both passages verbatim, and links to the other one's page", async () => {
    getRelatedPassages.mockResolvedValue({ results: [pair()] });

    render(
      <RelatedPassagesSection fileId="f1" drive="main" trustTier="verified" />,
    );
    fireEvent.click(await screen.findByText("relatedPassagesFind"));

    const mine = await screen.findByTestId("source-passage");
    const theirs = await screen.findByTestId("match-passage");
    // Byte-for-byte: nothing was summarised on either side.
    expect(mine.textContent).toBe(MINE);
    expect(theirs.textContent).toBe(THEIRS);

    const link = screen.getByText(/rag-design-notes\.md/);
    expect(link.closest("a")?.getAttribute("href")).toBe("/files/n1?page=3");
  });

  it("expands itself while the viewer is being asked to rule", async () => {
    getRelatedPassages.mockResolvedValue({ results: [pair()] });

    render(
      <RelatedPassagesSection
        fileId="f1" drive="main" trustTier="unverified" trustReviewedAt={null}
      />,
    );

    // No button press: the promotion panel is on screen and the evidence
    // has to be there with it.
    expect(await screen.findByTestId("source-passage")).toBeTruthy();
    expect(getRelatedPassages).toHaveBeenCalledWith("f1", "main");
  });

  it("hides itself entirely when an unruled file has no connections", async () => {
    const { container } = render(
      <RelatedPassagesSection
        fileId="f1" drive="main" trustTier="unverified" trustReviewedAt={null}
      />,
    );

    await waitFor(() => expect(getRelatedPassages).toHaveBeenCalled());
    // An empty shell above the promotion panel would be noise on every
    // clip that happens to match nothing.
    await waitFor(() => expect(container.firstChild).toBeNull());
  });

  it("says so when the viewer asked and nothing matched", async () => {
    render(
      <RelatedPassagesSection fileId="f1" drive="main" trustTier="verified" />,
    );
    fireEvent.click(await screen.findByText("relatedPassagesFind"));

    // Silence after a deliberate press reads as a broken button.
    expect(await screen.findByText("relatedPassagesEmpty")).toBeTruthy();
  });

  it("reports a failed lookup instead of pretending nothing matched", async () => {
    getRelatedPassages.mockRejectedValue(new Error("proxy timeout"));

    render(
      <RelatedPassagesSection fileId="f1" drive="main" trustTier="verified" />,
    );
    fireEvent.click(await screen.findByText("relatedPassagesFind"));

    expect(await screen.findByText("relatedPassagesUnavailable")).toBeTruthy();
  });

  it("resets when the viewer navigates to another file", async () => {
    getRelatedPassages.mockResolvedValue({ results: [pair()] });

    const { rerender } = render(
      <RelatedPassagesSection fileId="f1" drive="main" trustTier="verified" />,
    );
    fireEvent.click(await screen.findByText("relatedPassagesFind"));
    await screen.findByTestId("source-passage");

    rerender(
      <RelatedPassagesSection fileId="f2" drive="main" trustTier="verified" />,
    );

    // The previous file's passages must not linger under the new file.
    await waitFor(() =>
      expect(screen.queryByTestId("source-passage")).toBeNull(),
    );
    expect(screen.getByText("relatedPassagesFind")).toBeTruthy();
  });

  it("waits for the drive before asking, and asks once it lands", async () => {
    getRelatedPassages.mockResolvedValue({ results: [pair()] });

    // The /files/{id} route renders this while its own getFile is still
    // in flight, so the drive starts empty. Firing then would hit the
    // proxy without X-Lit-Drive and be rejected — and the failure would
    // stick, because the drive arriving is not a new file.
    const { rerender } = render(
      <RelatedPassagesSection
        fileId="f1" drive="" trustTier="unverified" trustReviewedAt={null}
      />,
    );
    expect(getRelatedPassages).not.toHaveBeenCalled();

    rerender(
      <RelatedPassagesSection
        fileId="f1" drive="main" trustTier="unverified" trustReviewedAt={null}
      />,
    );

    expect(await screen.findByTestId("source-passage")).toBeTruthy();
    expect(getRelatedPassages).toHaveBeenCalledWith("f1", "main");
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
