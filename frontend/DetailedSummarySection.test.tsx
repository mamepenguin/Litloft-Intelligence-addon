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

// Self-hide hook: default to "no active summary" so the existing
// tests render the detailed summary. Tests that verify self-hide
// override this via `mockActiveSummary.current`.
const mockActiveSummary = {
  current: { data: null, loading: false } as {
    data: { has_active_summary: boolean; summary_note?: unknown } | null;
    loading: boolean;
  },
};
vi.mock("@/hooks/useActiveSummary", () => ({
  useActiveSummary: () => mockActiveSummary.current,
}));

// Knowledge addon absence by default — keeps the Save button hidden
// unless a test opts in.
const mockAddons = {
  current: {} as Record<string, unknown>,
};
vi.mock("@/components/AddonSlotsProvider", () => ({
  useAddonSlots: () => ({
    addons: mockAddons.current,
    slots: {},
    loading: false,
    getSlotEntries: () => [],
    hasSlot: () => false,
    addonStatuses: {},
  }),
}));

vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<Record<string, unknown>>("@/lib/api");
  return {
    ...actual,
    getFile: vi.fn().mockResolvedValue({ id: "f1", filename: "test.mkv" }),
  };
});

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
      // top_score=0.72 below the strong threshold (0.90), so this
      // renders as "linked-weak". Either linked tier is acceptable here.
      const markers = container.querySelectorAll(
        '[data-citation-marker^="linked-"]',
      );
      expect(markers.length).toBeGreaterThanOrEqual(1);
    });
  });

  it("does not render any marker when has_citation is false", async () => {
    // Citation-missing is a retrieval outcome (synthesis, low cosine,
    // ambiguous tie), not a hallucination warning. The UI stays quiet
    // rather than drawing attention to an internal limitation.
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

    // Wait for the summary to render, then assert no citation marker
    // is present for this segment.
    await screen.findByText(/本作の概要を述べる段落/);
    expect(container.querySelector("[data-citation-marker]")).toBeNull();
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

  it("opens a textarea pre-filled with the full H2 fragment (heading + body) on Edit click", async () => {
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
    // The draft should be seeded with the entire H2 fragment so the
    // user can rename the heading or restructure the section in one
    // pass. This is the behaviour pinned in hako ``CURC61BSCLdE6uMd31k_4``.
    expect(textarea.value).toContain("## 全体像");
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
    fireEvent.change(textarea, {
      target: { value: "## 全体像\n編集後の概要。" },
    });

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /^保存$/ }));
    });

    await waitFor(() => {
      expect(editDetailedSummarySection).toHaveBeenCalledWith(
        "f1",
        "drive1",
        {
          section_heading: "全体像",
          // H2-level edit → subsection_heading: null so the backend
          // splices the whole H2 range. H3 edits carry the H3 title.
          subsection_heading: null,
          new_content: "## 全体像\n編集後の概要。",
        },
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

  it("dormant state: hides body + edit UI, surfaces only the regenerate entry", async () => {
    (getDetailedSummary as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValue(editedResponse);
    mockActiveSummary.current = {
      data: {
        has_active_summary: true,
        summary_note: { file_id: "n1" },
      },
      loading: false,
    };

    try {
      renderSection();

      // Dormant marker is in the DOM.
      await screen.findByTestId("detailed-summary-dormant");

      // Body (heading '全体像') must not render.
      expect(screen.queryByText(/全体像/)).not.toBeInTheDocument();
      // Edit / Save-to-knowledge / Revert buttons must all be absent.
      expect(screen.queryByText(/AI生成版に戻す/)).not.toBeInTheDocument();
      expect(screen.queryByText(/knowledge に保存/)).not.toBeInTheDocument();

      // Regenerate is the only interactive action.
      const regenButton = screen.getByRole("button", { name: /再生成/ });
      fireEvent.click(regenButton);

      // The confirm dialog uses the with-note variant.
      expect(
        await screen.findByText(
          /AI 版を再生成します。現在 knowledge に保存されているノートは残りますが/,
        ),
      ).toBeInTheDocument();
    } finally {
      mockActiveSummary.current = { data: null, loading: false };
    }
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

// ----- Inline Markdown rendering ----------------------------------------

describe("DetailedSummarySection — inline markdown rendering", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockWsLastEvent.current = null;
    (getDetailedSummaryCitations as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValue({ available: true, citations: [] });
  });

  afterEach(() => {
    cleanup();
  });

  it("renders **bold** / *italic* / `code` / links inside segment text", async () => {
    const md = `## 見どころ
この作品は **重要な瞬間** と *印象的なシーン* があります。
- \`setup()\` 関数を [参照](https://example.com) してください。
`;
    (getDetailedSummary as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValue({
        available: true,
        status: "generated",
        file_id: "f1",
        detailed_summary: md,
        model: "test-model",
        edited_at: null,
        has_original: false,
      });

    const { container } = renderSection();
    fireEvent.click(await screen.findByRole("button", { name: /展開/ }));

    await waitFor(() => {
      expect(container.querySelector("strong")?.textContent).toBe(
        "重要な瞬間",
      );
    });
    expect(container.querySelector("em")?.textContent).toBe("印象的なシーン");
    expect(container.querySelector("code")?.textContent).toBe("setup()");
    const link = container.querySelector<HTMLAnchorElement>("a[href]");
    expect(link?.getAttribute("href")).toBe("https://example.com");
    expect(link?.getAttribute("target")).toBe("_blank");
    expect(link?.getAttribute("rel")).toContain("noopener");
  });

  it("renders ### subheadings as H3 edit targets with their own edit buttons", async () => {
    const md = `## 章構成
### 第一幕

ここから物語が始まります。

### 第二幕

山場の展開。
`;
    (getDetailedSummary as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValue({
        available: true,
        status: "generated",
        file_id: "f1",
        detailed_summary: md,
        model: "test-model",
        edited_at: null,
        has_original: false,
      });

    const { container } = renderSection();
    fireEvent.click(await screen.findByRole("button", { name: /展開/ }));

    await screen.findByText(/ここから物語が始まります/);

    // ``###`` is captured at parse time as a subsection edit target
    // and rendered as an <h3> label by SectionView (the outer ``##``
    // uses <h2> so the GitHub-flavored .markdown-body typography
    // cascade applies). The raw marker never reaches the DOM.
    const h3s = container.querySelectorAll("h3");
    const headingTexts = Array.from(h3s).map((el) => el.textContent);
    expect(headingTexts).toEqual(expect.arrayContaining(["第一幕", "第二幕"]));

    // The raw ``### `` marker must not leak into the rendered DOM.
    expect(container.textContent ?? "").not.toContain("### 第一幕");
    expect(container.textContent ?? "").not.toContain("### 第二幕");

    // One edit button per H3 plus one for the outer H2.
    const editButtons = container.querySelectorAll('button[aria-label="編集"]');
    expect(editButtons.length).toBeGreaterThanOrEqual(3);
  });

  it("seeds the H3 edit draft with the full subsection fragment", async () => {
    const md = `## 章構成
### 第一幕

ここから物語が始まります。

### 第二幕

山場の展開。
`;
    (getDetailedSummary as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValue({
        available: true,
        status: "generated",
        file_id: "f1",
        detailed_summary: md,
        model: "test-model",
        edited_at: null,
        has_original: false,
      });

    const { container } = renderSection();
    fireEvent.click(await screen.findByRole("button", { name: /展開/ }));

    await screen.findByText(/ここから物語が始まります/);

    // Identify the H3 subsection block and click its edit button.
    const sub = container.querySelector(
      '[data-subsection-heading="第一幕"]',
    ) as HTMLElement | null;
    expect(sub).not.toBeNull();
    const editButton = sub!.querySelector(
      'button[aria-label="編集"]',
    ) as HTMLButtonElement | null;
    expect(editButton).not.toBeNull();
    fireEvent.click(editButton!);

    const textarea = (await screen.findByLabelText(
      /セクション内容を編集/,
    )) as HTMLTextAreaElement;
    // Draft carries the H3 heading line so the user can rename it,
    // and the subsection body up to the next ``###``.
    expect(textarea.value).toContain("### 第一幕");
    expect(textarea.value).toContain("ここから物語が始まります");
    // The sibling subsection must NOT be in this H3 draft.
    expect(textarea.value).not.toContain("### 第二幕");
    expect(textarea.value).not.toContain("山場の展開");
  });

  it("sends subsection_heading when saving an H3 edit", async () => {
    const md = `## 章構成
### 第一幕

ここから物語が始まります。
`;
    (getDetailedSummary as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValue({
        available: true,
        status: "generated",
        file_id: "f1",
        detailed_summary: md,
        model: "test-model",
        edited_at: null,
        has_original: false,
      });
    (editDetailedSummarySection as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValue({
        available: true,
        status: "generated",
        file_id: "f1",
        detailed_summary: md,
        edited_at: "2026-04-18T12:00:00Z",
        has_original: true,
      });

    const { container } = renderSection();
    fireEvent.click(await screen.findByRole("button", { name: /展開/ }));

    await screen.findByText(/ここから物語が始まります/);

    const sub = container.querySelector(
      '[data-subsection-heading="第一幕"]',
    ) as HTMLElement;
    const editButton = sub.querySelector(
      'button[aria-label="編集"]',
    ) as HTMLButtonElement;
    fireEvent.click(editButton);

    const textarea = (await screen.findByLabelText(
      /セクション内容を編集/,
    )) as HTMLTextAreaElement;
    fireEvent.change(textarea, {
      target: { value: "### 第一幕\n書き直し本文" },
    });

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /^保存$/ }));
    });

    await waitFor(() => {
      expect(editDetailedSummarySection).toHaveBeenCalledWith(
        "f1",
        "drive1",
        {
          section_heading: "章構成",
          subsection_heading: "第一幕",
          new_content: "### 第一幕\n書き直し本文",
        },
      );
    });
  });

  it("neutralises javascript: URLs in links", async () => {
    const md = `## 危険リンク
- [クリック](javascript:alert('xss')) という文があります
`;
    (getDetailedSummary as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValue({
        available: true,
        status: "generated",
        file_id: "f1",
        detailed_summary: md,
        model: "test-model",
        edited_at: null,
        has_original: false,
      });

    const { container } = renderSection();
    fireEvent.click(await screen.findByRole("button", { name: /展開/ }));

    // Wait for the expanded body to render — look for the surrounding
    // text so we know markdown-it has processed the segment.
    await screen.findByText(/という文があります/);

    // markdown-it's default validateLink rejects javascript: URIs at
    // parse time, so either no anchor renders at all OR (if another
    // layer in the pipeline were to render one) the href must not
    // carry the javascript: scheme. Both outcomes are safe.
    const link = container.querySelector<HTMLAnchorElement>("a[href]");
    expect(link?.getAttribute("href") ?? "").not.toMatch(/^javascript:/i);
  });

  it("renders fenced code blocks as <pre><code> via MarkdownPreview", async () => {
    // Regression guard: the legacy renderer fed bullets + fences into
    // the line parser's paragraph buffer, flattening code into a
    // single-line pseudo-paragraph. The fix captures the fence as its
    // own segment and pipes it through SegmentMarkdown so
    // markdown-it emits ``<pre>`` + ``<code>``.
    const md = `## コード例
\`\`\`python
def foo():
    return 42
\`\`\`
`;
    (getDetailedSummary as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValue({
        available: true,
        status: "generated",
        file_id: "f1",
        detailed_summary: md,
        model: "test-model",
        edited_at: null,
        has_original: false,
      });

    const { container } = renderSection();
    fireEvent.click(await screen.findByRole("button", { name: /展開/ }));
    // Syntax-highlighted markup splits ``return 42`` across hljs span
    // boundaries, so we poll for the ``<pre>`` appearing instead of
    // the raw text — findByText can't span sibling elements.
    await waitFor(() => {
      expect(container.querySelector("pre")).not.toBeNull();
    });

    const pre = container.querySelector("pre");
    const code = pre!.querySelector("code");
    expect(code).not.toBeNull();
    // ``textContent`` concatenates across hljs spans, so indentation
    // and the full ``return 42`` literal both survive.
    expect(code!.textContent ?? "").toContain("    return 42");
    // Backtick markers must not leak into the rendered DOM as plain
    // text — that was the old fallback's signature.
    const visible = container.textContent ?? "";
    expect(visible).not.toMatch(/`{3}python/);
  });

  it("renders bullets inside <ul><li> so .markdown-body list typography applies", async () => {
    const md = `## ポイント
- 第一項
- 第二項
- 第三項
`;
    (getDetailedSummary as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValue({
        available: true,
        status: "generated",
        file_id: "f1",
        detailed_summary: md,
        model: "test-model",
        edited_at: null,
        has_original: false,
      });

    const { container } = renderSection();
    fireEvent.click(await screen.findByRole("button", { name: /展開/ }));
    await screen.findByText(/第一項/);

    const ul = container.querySelector("ul.markdown-body");
    expect(ul).not.toBeNull();
    const lis = Array.from(ul!.querySelectorAll(":scope > li"));
    expect(lis).toHaveLength(3);
    expect(lis[0].getAttribute("data-citation-section-path")).toBe(
      "ポイント/0",
    );
    expect(lis[0].textContent).toContain("第一項");
    expect(lis[2].textContent).toContain("第三項");
  });

  it("renders markdown tables as <table> with <thead> + <tbody>", async () => {
    // Previously table rows surfaced as flat ``| a | b |`` flex rows —
    // the segments were captured but the renderer only stacked them as
    // plain text. This test locks in the new ``.markdown-body``-backed
    // table render.
    const md = `## 重要ポイントまとめ
| 項目 | 値 |
|---|---|
| 長さ | 120分 |
| 主題 | 冒険 |
`;
    (getDetailedSummary as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValue({
        available: true,
        status: "generated",
        file_id: "f1",
        detailed_summary: md,
        model: "test-model",
        edited_at: null,
        has_original: false,
      });

    const { container } = renderSection();
    fireEvent.click(await screen.findByRole("button", { name: /展開/ }));
    await screen.findByText(/長さ/);

    const table = container.querySelector("table");
    expect(table).not.toBeNull();
    const theadCells = Array.from(table!.querySelectorAll("thead th")).map(
      (el) => el.textContent?.trim() ?? "",
    );
    expect(theadCells).toEqual(["項目", "値"]);
    const bodyRows = Array.from(table!.querySelectorAll("tbody tr"));
    expect(bodyRows).toHaveLength(2);
    const firstRowCells = Array.from(bodyRows[0].querySelectorAll("td")).map(
      (el) => el.textContent?.trim() ?? "",
    );
    expect(firstRowCells).toEqual(["長さ", "120分"]);
    // The row carries the citation anchor so the downstream popover
    // layer can find the DOM target.
    expect(bodyRows[0].getAttribute("data-citation-section-path")).toBe(
      "重要ポイントまとめ/row/0",
    );
    // Sanity: raw ``|`` separators must not leak into the rendered DOM
    // (that was the old fallback path's signature).
    expect(container.textContent ?? "").not.toMatch(/\|\s*項目\s*\|/);
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

  it("captures fenced code blocks as a single segment with newlines preserved", () => {
    // The legacy line-by-line parser swallowed fences into the
    // paragraph buffer and joined with " ", flattening the code block
    // into a single-line pseudo-paragraph that MarkdownPreview could
    // no longer recognise. This test locks in fence-aware parsing.
    const md = `## コード例
以下のように書きます。

\`\`\`python
def foo():
    return 42
\`\`\`

補足の説明。
`;
    const [section] = parseSections(md);
    const types = section.segments.map((s) => s.type);
    expect(types).toEqual(["paragraph", "code-block", "paragraph"]);
    const code = section.segments[1];
    // The fenced content is preserved verbatim including indentation,
    // so MarkdownPreview can recognise the fence and emit ``<pre>``.
    expect(code.text).toContain("\`\`\`python");
    expect(code.text).toContain("    return 42");
    expect(code.text.trim().endsWith("\`\`\`")).toBe(true);
    // Code blocks consume a plain_idx slot (aligning with the
    // backend's merged-paragraph behaviour for blank-line-separated
    // fences) so surrounding paragraph indices shift up by one.
    expect(section.segments.map((s) => s.section_path)).toEqual([
      "コード例/0",
      "コード例/1",
      "コード例/2",
    ]);
  });

  it("attaches tableCells to every body row and tableHeader only to the first", () => {
    // Needed so the renderer can emit a proper ``<table>`` with a
    // ``<thead>`` instead of the legacy flat ``| a | b |`` text.
    const md = `## 重要ポイントまとめ
| 項目 | 値 |
|---|---|
| 長さ | 120分 |
| 主題 | 冒険 |
`;
    const [section] = parseSections(md);
    const rows = section.segments.filter((s) => s.type === "table-row");
    expect(rows[0].tableCells).toEqual(["長さ", "120分"]);
    expect(rows[1].tableCells).toEqual(["主題", "冒険"]);
    expect(rows[0].tableHeader).toEqual(["項目", "値"]);
    // Only the first body row carries the header — duplicating it
    // would make the renderer emit multiple ``<thead>`` elements.
    expect(rows[1].tableHeader).toBeUndefined();
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

  it("captures ### subsections and attributes segments to them", () => {
    const md = `## 章構成
冒頭説明。

### 第一幕
一幕目の内容。

### 第二幕
二幕目の内容。
- ポイント1
- ポイント2
`;
    const [section] = parseSections(md);
    expect(section.subsections.map((s) => s.heading)).toEqual([
      "第一幕",
      "第二幕",
    ]);
    // Preamble segment has no subHeading.
    const preamble = section.segments.find((s) => s.subHeading === null);
    expect(preamble?.text).toContain("冒頭説明");
    // Segments under first/second act are attributed correctly.
    const firstAct = section.segments.filter((s) => s.subHeading === "第一幕");
    const secondAct = section.segments.filter((s) => s.subHeading === "第二幕");
    expect(firstAct.map((s) => s.text)).toContain("一幕目の内容。");
    expect(secondAct.some((s) => s.text === "ポイント1")).toBe(true);
    expect(secondAct.some((s) => s.text === "ポイント2")).toBe(true);
  });

  it("keeps plain_idx H2-scoped even when ### subsections intervene", () => {
    // Existing citations are anchored by ``<H2>/<plain_idx>``. The
    // frontend must treat ``###`` as plain content from the counter's
    // perspective so those paths remain valid after adding H3 edit
    // affordances. See hako ``6DcHGrYOBmehO7RJFXUN0`` — "plain_idx の
    // 番号規則を保つことで既存 citation の section_path を壊さない".
    const md = `## S
段落1。

### サブ
段落2。
- 箇条書き
`;
    const [section] = parseSections(md);
    const paths = section.segments.map((s) => s.section_path);
    expect(paths).toEqual(["S/0", "S/1", "S/2"]);
  });

  it("exposes fullFragment for H2 and H3 edit draft seeds", () => {
    const md = `## 章構成
冒頭。

### 第一幕
一幕目。
`;
    const [section] = parseSections(md);
    // H2 fullFragment covers the heading + everything up to (but not
    // including) the next ``##``.
    expect(section.fullFragment).toContain("## 章構成");
    expect(section.fullFragment).toContain("冒頭。");
    expect(section.fullFragment).toContain("### 第一幕");
    expect(section.fullFragment).toContain("一幕目。");
    // H3 fullFragment is the standalone subsection.
    expect(section.subsections[0].fullFragment).toContain("### 第一幕");
    expect(section.subsections[0].fullFragment).toContain("一幕目。");
    expect(section.subsections[0].fullFragment).not.toContain("冒頭。");
  });
});
