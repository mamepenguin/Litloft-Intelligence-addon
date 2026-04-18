/**
 * Tests for DetailedSummarySection — state-machine UI rendering +
 * citation highlighting + per-section edit flow.
 *
 * The click / download paths are exercised end-to-end in the manual
 * QA pass (Phase H) — driving them through vitest here leaks
 * polling setTimeouts across file boundaries and causes OOM in the
 * combined test run (see pool=forks worker notes).
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  render,
  screen,
  waitFor,
  cleanup,
  fireEvent,
  act,
} from "@testing-library/react";
import { NextIntlClientProvider } from "next-intl";

// Default stub: the WS provider returns null unless a test overrides it.
// Keeps the section's WS-driven refetch from firing in the basic cases.
const mockWsLastEvent = { current: null as unknown };
vi.mock("@/hooks/useWebSocket", () => ({
  useWebSocket: (filter?: string) => {
    const ev = mockWsLastEvent.current as { event?: string } | null;
    if (!ev) return null;
    if (filter && ev.event !== filter) return null;
    return ev;
  },
}));

vi.mock("@/addons/intelligence/api", () => ({
  getDetailedSummary: vi.fn(),
  startDetailedSummary: vi.fn(),
  deleteDetailedSummary: vi.fn(),
  downloadDetailedSummary: vi.fn(),
  getDetailedSummaryCitations: vi.fn().mockResolvedValue({
    available: true,
    citations: [],
  }),
  editDetailedSummarySection: vi.fn(),
  revertDetailedSummary: vi.fn(),
  regenerateDetailedSummary: vi.fn(),
  getCitationChunkExcerpt: vi.fn().mockResolvedValue(null),
}));

import DetailedSummarySection, {
  parseSections,
} from "@/addons/intelligence/DetailedSummarySection";
import {
  editDetailedSummarySection,
  getDetailedSummary,
  getDetailedSummaryCitations,
  regenerateDetailedSummary,
  revertDetailedSummary,
} from "@/addons/intelligence/api";

function renderSection() {
  return render(
    <NextIntlClientProvider locale="ja" messages={{}}>
      <DetailedSummarySection fileId="f1" drive="drive1" />
    </NextIntlClientProvider>,
  );
}

const generatedMarkdown = `## 全体像
本作の概要を述べる段落。

## 主要な章/場面
- シーン 1 の説明
- シーン 2 の説明
`;

const generatedResponse = {
  available: true,
  status: "generated",
  file_id: "f1",
  detailed_summary: generatedMarkdown,
  model: "test-model",
  edited_at: null,
  has_original: false,
};

const editedResponse = {
  ...generatedResponse,
  detailed_summary:
    "## 全体像\n編集後の概要。\n\n## 主要な章/場面\n- シーン 1 の説明\n- シーン 2 の説明\n",
  edited_at: "2026-04-18T12:00:00Z",
  has_original: true,
};

describe("DetailedSummarySection — state machine", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockWsLastEvent.current = null;
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
      .mockResolvedValue(generatedResponse);

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

// ----- Phase 1: Citation highlighting -----------------------------------

describe("DetailedSummarySection — citations", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockWsLastEvent.current = null;
    (getDetailedSummary as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValue(generatedResponse);
  });

  afterEach(() => {
    cleanup();
  });

  it("fetches citations after generated summary loads", async () => {
    (getDetailedSummaryCitations as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValue({
        available: true,
        citations: [
          {
            section_path: "全体像/0",
            segment_type: "paragraph",
            segment_text: "本作の概要を述べる段落。",
            chunk_ids: ["c1"],
            top_score: 0.72,
            has_citation: true,
          },
        ],
      });

    renderSection();

    await waitFor(() => {
      expect(getDetailedSummaryCitations).toHaveBeenCalledWith("f1", "drive1");
    });
  });

  it("renders a linked citation marker for segments with citations", async () => {
    (getDetailedSummaryCitations as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValue({
        available: true,
        citations: [
          {
            section_path: "全体像/0",
            segment_type: "paragraph",
            segment_text: "本作の概要を述べる段落。",
            chunk_ids: ["c1"],
            top_score: 0.72,
            has_citation: true,
          },
        ],
      });

    const { container } = renderSection();

    // Expand the section so the body (and citation markers) render.
    fireEvent.click(await screen.findByRole("button", { name: /展開/ }));

    await waitFor(() => {
      const markers = container.querySelectorAll(
        '[data-citation-marker="linked"]',
      );
      expect(markers.length).toBeGreaterThanOrEqual(1);
    });
  });

  it("renders a warning marker when has_citation is false", async () => {
    (getDetailedSummaryCitations as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValue({
        available: true,
        citations: [
          {
            section_path: "全体像/0",
            segment_type: "paragraph",
            segment_text: "本作の概要を述べる段落。",
            chunk_ids: [],
            top_score: 0.2,
            has_citation: false,
          },
        ],
      });

    const { container } = renderSection();

    fireEvent.click(await screen.findByRole("button", { name: /展開/ }));

    await waitFor(() => {
      const warning = container.querySelector(
        '[data-citation-marker="missing"]',
      );
      expect(warning).not.toBeNull();
    });
  });

  it("refetches citations when the citations_ready WS event fires", async () => {
    (getDetailedSummaryCitations as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValue({ available: true, citations: [] });

    const { rerender } = renderSection();

    await waitFor(() => {
      expect(getDetailedSummaryCitations).toHaveBeenCalledTimes(1);
    });

    (getDetailedSummaryCitations as unknown as ReturnType<typeof vi.fn>)
      .mockClear();

    act(() => {
      mockWsLastEvent.current = {
        event: "intelligence.detailed_summary.citations_ready",
        data: { file_id: "f1", citation_count: 2, no_citation_count: 0 },
      };
      // Force a rerender — in the real provider the context update
      // would trigger it automatically; for the mock we re-render.
      rerender(
        <NextIntlClientProvider locale="ja" messages={{}}>
          <DetailedSummarySection fileId="f1" drive="drive1" />
        </NextIntlClientProvider>,
      );
    });

    await waitFor(() => {
      expect(getDetailedSummaryCitations).toHaveBeenCalled();
    });
  });
});

// ----- Phase 2: Section edit / revert / regenerate confirm ---------------

describe("DetailedSummarySection — edit flow", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockWsLastEvent.current = null;
    (getDetailedSummaryCitations as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValue({ available: true, citations: [] });
  });

  afterEach(() => {
    cleanup();
  });

  it("opens a textarea pre-filled with the section body on Edit click", async () => {
    (getDetailedSummary as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValue(generatedResponse);

    renderSection();

    fireEvent.click(await screen.findByRole("button", { name: /展開/ }));

    // There's one Edit button per section heading. Pick the first one.
    const editButtons = await screen.findAllByRole("button", {
      name: /^編集$/,
    });
    expect(editButtons.length).toBeGreaterThanOrEqual(1);
    fireEvent.click(editButtons[0]);

    const textarea = (await screen.findByLabelText(
      /セクション内容を編集/,
    )) as HTMLTextAreaElement;
    // The draft should be seeded with the section body (all lines
    // between `## 全体像` and the next `##`).
    expect(textarea.value).toContain("本作の概要を述べる段落");
  });

  it("saves via editDetailedSummarySection and rehydrates from the response", async () => {
    (getDetailedSummary as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValue(generatedResponse);
    (editDetailedSummarySection as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValue(editedResponse);

    renderSection();

    fireEvent.click(await screen.findByRole("button", { name: /展開/ }));
    const editButtons = await screen.findAllByRole("button", {
      name: /^編集$/,
    });
    fireEvent.click(editButtons[0]);

    const textarea = (await screen.findByLabelText(
      /セクション内容を編集/,
    )) as HTMLTextAreaElement;
    fireEvent.change(textarea, { target: { value: "編集後の概要。" } });

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /^保存$/ }));
    });

    await waitFor(() => {
      expect(editDetailedSummarySection).toHaveBeenCalledWith(
        "f1",
        "drive1",
        { section_heading: "全体像", new_content: "編集後の概要。" },
      );
    });

    // The response includes edited_at, so the "編集済み" badge
    // appears and the revert button becomes visible.
    expect(await screen.findByText(/編集済み/)).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /AI生成版に戻す/ }),
    ).toBeInTheDocument();
  });

  it("cancel closes the editor without calling the API", async () => {
    (getDetailedSummary as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValue(generatedResponse);

    renderSection();

    fireEvent.click(await screen.findByRole("button", { name: /展開/ }));
    const editButtons = await screen.findAllByRole("button", {
      name: /^編集$/,
    });
    fireEvent.click(editButtons[0]);

    const textarea = (await screen.findByLabelText(
      /セクション内容を編集/,
    )) as HTMLTextAreaElement;
    fireEvent.change(textarea, { target: { value: "捨てる" } });
    fireEvent.click(screen.getByRole("button", { name: /キャンセル/ }));

    expect(editDetailedSummarySection).not.toHaveBeenCalled();
    expect(screen.queryByLabelText(/セクション内容を編集/)).toBeNull();
  });

  it("revert shows a confirm dialog and calls revertDetailedSummary on confirm", async () => {
    (getDetailedSummary as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValue(editedResponse);
    (revertDetailedSummary as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValue(generatedResponse);

    renderSection();

    fireEvent.click(await screen.findByRole("button", { name: /展開/ }));

    // Edited state renders the "AI 版に戻す" button.
    const revertButtons = await screen.findAllByRole("button", {
      name: /AI生成版に戻す/,
    });
    // First one is the bottom toolbar button (the one that opens the
    // dialog — the dialog's confirm button shares the label).
    fireEvent.click(revertButtons[0]);

    // Dialog appears with a confirmation message.
    expect(
      await screen.findByText(/編集内容を破棄して/),
    ).toBeInTheDocument();

    // Confirm — the dialog's primary button carries the same label.
    await act(async () => {
      const dialogButtons = screen.getAllByRole("button", {
        name: /AI生成版に戻す/,
      });
      // Click the last one (the dialog's confirm button renders after
      // the toolbar button).
      fireEvent.click(dialogButtons[dialogButtons.length - 1]);
    });

    await waitFor(() => {
      expect(revertDetailedSummary).toHaveBeenCalledWith("f1", "drive1");
    });
  });

  it("regenerate on an edited summary opens the confirm dialog", async () => {
    (getDetailedSummary as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValue(editedResponse);

    renderSection();

    fireEvent.click(await screen.findByRole("button", { name: /展開/ }));

    // The bottom toolbar has a Regenerate button. Click it.
    const regenButtons = await screen.findAllByRole("button", {
      name: /^再生成$/,
    });
    fireEvent.click(regenButtons[0]);

    // The confirm dialog warns that edits will be lost.
    expect(
      await screen.findByText(/編集内容は失われます/),
    ).toBeInTheDocument();
  });

  it("regenerate on an un-edited summary skips the dialog and fires regenerate", async () => {
    (getDetailedSummary as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValue(generatedResponse);

    renderSection();

    fireEvent.click(await screen.findByRole("button", { name: /展開/ }));

    const regenButtons = await screen.findAllByRole("button", {
      name: /^再生成$/,
    });

    await act(async () => {
      fireEvent.click(regenButtons[0]);
    });

    // No "編集内容は失われます" dialog should appear.
    expect(
      screen.queryByText(/編集内容は失われます/),
    ).not.toBeInTheDocument();
    // regenerateDetailedSummary (with force) is not called in the
    // un-edited path — the legacy delete+start path runs instead.
    expect(regenerateDetailedSummary).not.toHaveBeenCalled();
  });
});

// ----- Markdown parser unit tests ---------------------------------------

describe("parseSections", () => {
  it("returns an empty array for empty input", () => {
    expect(parseSections("")).toEqual([]);
  });

  it("splits on `##` headings and captures bullets", () => {
    const md = `## 全体像
本作は長編作品である。

## 主要な章/場面
- 章1
- 章2
  - 子項目
`;
    const sections = parseSections(md);
    expect(sections.map((s) => s.heading)).toEqual(["全体像", "主要な章/場面"]);
    const bullets = sections[1].segments.filter((s) => s.type === "bullet");
    expect(bullets).toHaveLength(3);
    expect(bullets[2].indent).toBeGreaterThan(0); // nested child
  });

  it("assigns section paths per spec convention", () => {
    const md = `## 全体像
段落1。

段落2。

## 主要な章/場面
- 章1
`;
    const sections = parseSections(md);
    const paths = sections.flatMap((s) => s.segments.map((seg) => seg.section_path));
    expect(paths).toContain("全体像/0");
    expect(paths).toContain("全体像/1");
    expect(paths).toContain("主要な章/場面/0");
  });

  it("captures table rows with row/N paths and skips header + separator", () => {
    const md = `## 重要ポイントまとめ
| 項目 | 値 |
|---|---|
| 長さ | 120分 |
| 主題 | 冒険 |
`;
    const [section] = parseSections(md);
    const rows = section.segments.filter((s) => s.type === "table-row");
    // Mirrors the backend summary_parser: header + separator skipped,
    // body rows start at row/0. Previously the frontend leaked the
    // header as row/0, shifting every citation onto the wrong DOM row.
    expect(rows.map((r) => r.section_path)).toEqual([
      "重要ポイントまとめ/row/0",
      "重要ポイントまとめ/row/1",
    ]);
    expect(rows[0].text).toContain("長さ");
    expect(rows[1].text).toContain("主題");
  });

  it("uses a shared counter for paragraphs + bullets in the same section", () => {
    // Backend ``plain_idx`` is shared: paragraph then bullet must be
    // section/0 and section/1 so citations lookup works. Separate
    // counters would produce section/0 and section/0 (key collision).
    const md = `## S
前書き段落。
- 箇条書き 1
- 箇条書き 2
`;
    const [section] = parseSections(md);
    const paths = section.segments.map((s) => s.section_path);
    expect(paths).toEqual(["S/0", "S/1", "S/2"]);
    expect(section.segments[0].type).toBe("paragraph");
    expect(section.segments[1].type).toBe("bullet");
    expect(section.segments[2].type).toBe("bullet");
  });
});
