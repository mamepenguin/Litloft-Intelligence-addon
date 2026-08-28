import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";

const semanticSearch = vi.fn();
vi.mock("./api", () => ({
  semanticSearch: (...args: unknown[]) => semanticSearch(...args),
}));

vi.mock("next-intl", () => ({
  useTranslations: () => (key: string) => key,
}));

vi.mock("next/link", () => ({
  default: ({ children, href }: { children: React.ReactNode; href: string }) => (
    <a href={href}>{children}</a>
  ),
}));

import UnverifiedSourceSection, { splitParagraphs } from "./UnverifiedSourceSection";

const PARAGRAPH =
  "Chunk size of 400 characters is far too short for expository prose; " +
  "an argument that spans two pages gets shredded into fragments that no " +
  "longer carry the claim they were part of.";

function makeHit(
  fileId: string,
  filename: string,
  trustTier: "verified" | "unverified" | null = "verified",
) {
  return {
    file_id: fileId,
    drive: "main",
    filename,
    file_type: "document",
    score: 0.9,
    match_types: ["content"],
    segments: [],
    file: trustTier === null ? null : { trust_tier: trustTier },
  };
}

function mockClipBody(body: string) {
  vi.stubGlobal("fetch", vi.fn(async (url: string, init?: RequestInit) => {
    if (String(url).endsWith("/stream")) {
      return { ok: true, text: async () => body };
    }
    return { ok: true, json: async () => ({ id: "f1", trust_tier: "verified" }) };
  }));
}

function trustCall() {
  return (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls.find(
    (c) => String(c[0]).endsWith("/trust-tier"),
  );
}

describe("UnverifiedSourceSection", () => {
  beforeEach(() => {
    semanticSearch.mockReset();
    semanticSearch.mockResolvedValue({ available: true, results: [], total: 0 });
    mockClipBody(`${PARAGRAPH}\n\n${PARAGRAPH}`);
  });

  it("asks only about unverified files nobody has ruled on", () => {
    const { container, rerender } = render(
      <UnverifiedSourceSection
        fileId="f1" drive="main" trustTier="verified" trustReviewedAt={null}
      />,
    );
    expect(container.firstChild).toBeNull();

    // Already dismissed once: re-asking on every open would be nagging.
    rerender(
      <UnverifiedSourceSection
        fileId="f1" drive="main" trustTier="unverified"
        trustReviewedAt="2026-08-29T00:00:00Z"
      />,
    );
    expect(container.firstChild).toBeNull();
  });

  it("renders for an unreviewed unverified file", async () => {
    render(
      <UnverifiedSourceSection
        fileId="f1" drive="main" trustTier="unverified" trustReviewedAt={null}
      />,
    );
    expect(await screen.findByText("title")).toBeTruthy();
  });

  it("shows the clip's own paragraph verbatim beside the note it echoes", async () => {
    semanticSearch.mockResolvedValue({
      available: true,
      results: [makeHit("n1", "rag-design-notes.md")],
      total: 1,
    });

    render(
      <UnverifiedSourceSection
        fileId="f1" drive="main" trustTier="unverified" trustReviewedAt={null}
      />,
    );

    const excerpt = (await screen.findAllByTestId("source-excerpt"))[0];
    // Byte-for-byte the clip's own text. Nothing was summarised, and the
    // string shown is the same one that was used as the search query.
    expect(PARAGRAPH.startsWith(excerpt.textContent ?? "")).toBe(true);
    expect(semanticSearch).toHaveBeenCalledWith(PARAGRAPH, "main", { limit: 5 });

    const link = await screen.findAllByText("rag-design-notes.md");
    expect(link[0].closest("a")?.getAttribute("href")).toBe("/files/n1");
  });

  it("ignores hits that are not the viewer's own vouched-for notes", async () => {
    semanticSearch.mockResolvedValue({
      available: true,
      results: [
        makeHit("f1", "the-clip.md"),                    // the clip itself
        makeHit("v1", "video.mp4"),                      // not a note
        makeHit("c2", "another-clip.md", "unverified"),  // external content
        makeHit("u1", "unhydrated.md", null),            // trust unknown
      ],
      total: 4,
    });

    render(
      <UnverifiedSourceSection
        fileId="f1" drive="main" trustTier="unverified" trustReviewedAt={null}
      />,
    );

    await screen.findByText("title");
    await waitFor(() => expect(semanticSearch).toHaveBeenCalled());
    expect(screen.queryByTestId("source-excerpt")).toBeNull();
  });

  it("promotes on trust", async () => {
    const onFileChange = vi.fn();
    render(
      <UnverifiedSourceSection
        fileId="f1" drive="main" trustTier="unverified" trustReviewedAt={null}
        onFileChange={onFileChange}
      />,
    );

    fireEvent.click(await screen.findByText("trust"));

    await waitFor(() => expect(onFileChange).toHaveBeenCalled());
    const call = trustCall();
    expect(call?.[0]).toBe("/api/files/f1/trust-tier");
    expect(JSON.parse(call![1].body)).toEqual({ tier: "verified" });
  });

  it("dismiss records the judgement without granting trust", async () => {
    render(
      <UnverifiedSourceSection
        fileId="f1" drive="main" trustTier="unverified" trustReviewedAt={null}
      />,
    );

    fireEvent.click(await screen.findByText("dismiss"));

    await waitFor(() => expect(trustCall()).toBeTruthy());
    // Same tier it already had: the write exists to stamp the review, which
    // is what stops the panel coming back.
    expect(JSON.parse(trustCall()![1].body)).toEqual({ tier: "unverified" });
  });

  it("makes no LLM call", async () => {
    render(
      <UnverifiedSourceSection
        fileId="f1" drive="main" trustTier="unverified" trustReviewedAt={null}
      />,
    );
    await screen.findByText("title");
    await waitFor(() => expect(semanticSearch).toHaveBeenCalled());

    // Only the clip body and the embedding search. No generation endpoint.
    const urls = (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls.map(
      (c) => String(c[0]),
    );
    expect(urls.every((u) => u.endsWith("/stream"))).toBe(true);
  });

  describe("splitParagraphs", () => {
    it("drops frontmatter, headings, and short cruft", () => {
      const body = [
        "---",
        "source_url: https://example.com",
        "---",
        "# Heading",
        "By a byline",
        PARAGRAPH,
      ].join("\n\n");

      expect(splitParagraphs(body)).toEqual([PARAGRAPH]);
    });

    it("collapses internal whitespace so the query matches the display", () => {
      const [only] = splitParagraphs(PARAGRAPH.replace(/ /g, "\n  "));
      expect(only).toBe(PARAGRAPH);
    });

    it("cuts long paragraphs before use, so shown and searched agree", () => {
      const long = "x".repeat(500);
      const [only] = splitParagraphs(long);
      expect(only).toHaveLength(240);
    });
  });
});
