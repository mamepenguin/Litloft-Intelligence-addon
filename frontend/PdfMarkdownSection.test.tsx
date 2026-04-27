/**
 * Tests for PdfMarkdownSection state branches.
 *
 * Covers:
 *   1. Non-PDF files render nothing (the section gates on mime_type).
 *   2. Successful API response forwards the body to MarkdownPreview.
 *   3. 404 / null payload renders the unavailable placeholder.
 *   4. Thrown errors from the API render the error placeholder.
 *
 * Spec: docs/superpowers/specs/2026-04-27-intelligence-pdf-markdown-indexing.md §フロントエンド
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { NextIntlClientProvider } from "next-intl";

vi.mock("@/addons/intelligence/api", () => ({
  getPdfMarkdown: vi.fn(),
}));

vi.mock("@/lib/api", () => ({
  getFile: vi.fn(),
}));

// MarkdownPreview is exercised in its own suite — replace with a probe
// component so we can assert the props this section forwards without
// pulling in markdown-it / DOMPurify / mermaid for every assertion.
vi.mock("@/components/MarkdownPreview", () => ({
  MarkdownPreview: ({
    source,
    chrome,
    mermaid,
  }: {
    source: string;
    chrome?: boolean;
    mermaid?: boolean;
  }) => (
    <div
      data-testid="markdown-preview"
      data-chrome={String(chrome)}
      data-mermaid={String(mermaid)}
    >
      {source}
    </div>
  ),
}));

import PdfMarkdownSection from "@/addons/intelligence/PdfMarkdownSection";
import { getPdfMarkdown } from "@/addons/intelligence/api";
import { getFile } from "@/lib/api";

const pdfFile = {
  id: "f1",
  drive: "family",
  filename: "report.pdf",
  file_type: "document",
  mime_type: "application/pdf",
};

function renderSection(props: { fileId?: string; drive?: string } = {}) {
  return render(
    <NextIntlClientProvider locale="ja" messages={{}}>
      <PdfMarkdownSection
        fileId={props.fileId ?? "f1"}
        drive={props.drive ?? "family"}
      />
    </NextIntlClientProvider>,
  );
}

describe("PdfMarkdownSection", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    (getFile as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(pdfFile);
  });

  it("renders nothing for non-PDF files", async () => {
    (getFile as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({
      ...pdfFile,
      file_type: "video",
      mime_type: "video/mp4",
    });
    const { container } = renderSection();
    await waitFor(() => expect(getFile).toHaveBeenCalled());
    // Wait one extra microtask to ensure the post-getFile branch ran.
    await Promise.resolve();
    expect(container).toBeEmptyDOMElement();
    // The API must not be called when the file is not a PDF.
    expect(getPdfMarkdown).not.toHaveBeenCalled();
  });

  it("forwards Markdown to MarkdownPreview when the API returns a payload", async () => {
    (getPdfMarkdown as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({
      file_id: "f1",
      markdown: "# Hello\n\nA report body.",
      page_count: 12,
      extractor: "pymupdf4llm",
      generated_at: "2026-04-27T10:00:00Z",
    });

    renderSection();

    const preview = await screen.findByTestId("markdown-preview");
    expect(preview).toHaveTextContent("# Hello");
    expect(preview).toHaveTextContent("A report body.");
    // chrome=false because the section already provides its own card chrome
    // (DESIGN.md §3.4: parent owns measure when wrapper exists).
    expect(preview.dataset.chrome).toBe("false");
    // mermaid=false because PyMuPDF4LLM output is a Markdown projection
    // of an external PDF — it should never be rendered as a diagram.
    expect(preview.dataset.mermaid).toBe("false");
    expect(getPdfMarkdown).toHaveBeenCalledWith("f1", "family");
  });

  it("renders the unavailable placeholder when the API returns null (404)", async () => {
    (getPdfMarkdown as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(null);

    renderSection();

    expect(
      await screen.findByText(/Markdown 表示に対応していません/),
    ).toBeInTheDocument();
    expect(screen.queryByTestId("markdown-preview")).not.toBeInTheDocument();
  });

  it("renders the error placeholder when the API throws", async () => {
    (getPdfMarkdown as unknown as ReturnType<typeof vi.fn>).mockRejectedValue(
      new Error("network down"),
    );

    renderSection();

    expect(
      await screen.findByText(/Markdown の取得に失敗しました/),
    ).toBeInTheDocument();
    expect(screen.queryByTestId("markdown-preview")).not.toBeInTheDocument();
  });
});
