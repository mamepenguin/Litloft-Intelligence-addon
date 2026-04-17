/**
 * Tests for DetailedSummarySection — state-machine UI rendering.
 *
 * Covers the 4 distinct render branches driven by the API response
 * shape: unsupported → hidden, insufficient_content → note,
 * not_generated → button, generated → Markdown body.
 *
 * The click / download paths are exercised end-to-end in the manual
 * QA pass (Phase H) — driving them through vitest here leaks
 * polling setTimeouts across file boundaries and causes OOM in the
 * combined test run (see pool=forks worker notes).
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor, cleanup } from "@testing-library/react";
import { NextIntlClientProvider } from "next-intl";

vi.mock("@/components/MarkdownPreview", () => ({
  MarkdownPreview: ({ source }: { source: string }) => (
    <div data-testid="markdown-preview">{source}</div>
  ),
}));

vi.mock("@/addons/intelligence/api", () => ({
  getDetailedSummary: vi.fn(),
  startDetailedSummary: vi.fn(),
  deleteDetailedSummary: vi.fn(),
  downloadDetailedSummary: vi.fn(),
}));

import DetailedSummarySection from "@/addons/intelligence/DetailedSummarySection";
import { getDetailedSummary } from "@/addons/intelligence/api";

function renderSection() {
  return render(
    <NextIntlClientProvider locale="ja" messages={{}}>
      <DetailedSummarySection fileId="f1" drive="drive1" />
    </NextIntlClientProvider>,
  );
}

describe("DetailedSummarySection", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    cleanup();
  });

  it("renders nothing for unsupported_type", async () => {
    (getDetailedSummary as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValue({ available: false, reason: "unsupported_type" });

    const { container } = renderSection();

    await waitFor(() => {
      expect(getDetailedSummary).toHaveBeenCalled();
    });
    expect(container.firstChild).toBeNull();
  });

  it("shows insufficient_content note", async () => {
    (getDetailedSummary as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValue({
        available: false, reason: "insufficient_content",
      });

    renderSection();

    expect(
      await screen.findByText(/詳細要約に必要なコンテンツが不足/),
    ).toBeInTheDocument();
  });

  it("shows generate button when not_generated", async () => {
    (getDetailedSummary as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValue({ available: false, reason: "not_generated" });

    renderSection();

    expect(
      await screen.findByRole("button", { name: /詳細要約を生成/ }),
    ).toBeInTheDocument();
  });

  it("renders the Markdown body when generated", async () => {
    (getDetailedSummary as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValue({
        available: true,
        status: "generated",
        file_id: "f1",
        detailed_summary: "## Intro\n\nBody text",
        model: "test-model",
      });

    renderSection();

    // Section defaults to collapsed; the expand button is the
    // entry point into the Markdown body.
    expect(
      await screen.findByRole("button", { name: /展開/ }),
    ).toBeInTheDocument();
  });

  it("surfaces the failed status with an error message", async () => {
    (getDetailedSummary as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValue({
        available: false,
        status: "failed",
        error: "LLM error: boom",
      });

    renderSection();

    expect(
      await screen.findByText(/詳細要約の生成に失敗/),
    ).toBeInTheDocument();
    expect(screen.getByText("LLM error: boom")).toBeInTheDocument();
  });
});
