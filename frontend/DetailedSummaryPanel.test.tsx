import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { act, render, screen, waitFor, cleanup } from "@testing-library/react";
import { NextIntlClientProvider } from "next-intl";

vi.mock("@/hooks/useWebSocket", () => ({
  useWebSocket: () => null,
}));

const mockActiveSummary = {
  current: { data: null, loading: false } as {
    data: { has_active_summary: boolean; summary_note?: unknown } | null;
    loading: boolean;
  },
};
vi.mock("@/hooks/useActiveSummary", () => ({
  useActiveSummary: () => mockActiveSummary.current,
}));

vi.mock("@/components/AddonSlotsProvider", () => ({
  useAddonSlots: () => ({
    addons: {},
    slots: {},
    loading: false,
    getSlotEntries: () => [],
    hasSlot: () => false,
  }),
}));

vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<Record<string, unknown>>("@/lib/api");
  return {
    ...actual,
    getFile: vi.fn().mockResolvedValue({ id: "f1", filename: "test.mkv" }),
  };
});

vi.mock("@/addons/intelligence/api", async () => ({
  fetchLLMChoices: vi.fn(async () => {
    throw new Error("not stubbed");
  }),
  isProfileUnavailable: (
    await vi.importActual<typeof import("@/addons/intelligence/api")>(
      "@/addons/intelligence/api",
    )
  ).isProfileUnavailable,
  LLMRequestError: (
    await vi.importActual<typeof import("@/addons/intelligence/api")>(
      "@/addons/intelligence/api",
    )
  ).LLMRequestError,
  getDetailedSummary: vi.fn(),
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

import DetailedSummarySection from "@/addons/intelligence/DetailedSummarySection";
import { resetFileAiActions } from "@/lib/fileAiActions";
import { getDetailedSummary } from "@/addons/intelligence/api";

const Section = DetailedSummarySection as unknown as React.ComponentType<
  Record<string, unknown>
>;

function panel(props: Record<string, unknown> = {}) {
  return (
    <NextIntlClientProvider locale="ja" messages={{}}>
      <Section fileId="f1" drive="drive1" fillHeight {...props} />
    </NextIntlClientProvider>
  );
}

const generated = {
  available: true,
  status: "generated",
  file_id: "f1",
  detailed_summary: "## 全体像\n本作の概要を述べる段落。\n",
  model: "test-model",
  edited_at: null,
  has_original: false,
};

const answers = (report: ReturnType<typeof vi.fn>) =>
  report.mock.calls.map(([available]) => available);

beforeEach(() => {
  vi.clearAllMocks();
  resetFileAiActions();
  mockActiveSummary.current = { data: null, loading: false };
});

afterEach(() => {
  cleanup();
});

describe("SPEC-ADDON-099 Detailed summary as a player-side panel", () => {
  it("SPEC-ADDON-099 I10 answers false on mount, before its request returns", () => {
    vi.mocked(getDetailedSummary).mockReturnValue(new Promise(() => {}) as never);
    const report = vi.fn();

    render(panel({ active: false, onAvailability: report }));

    expect(answers(report)[0]).toBe(false);
    expect(answers(report)).not.toContain(true);
  });

  it.each([
    ["an unsupported type", { available: false, reason: "unsupported_type" }, false],
    ["never generated", { available: false, reason: "not_generated" }, false],
    ["not enough content", { available: false, reason: "insufficient_content" }, false],
    ["a failed run", { available: false, status: "failed", error: "LLM error: boom" }, true],
    ["a body", generated, true],
  ])("SPEC-ADDON-099 I10 answers %s with %s", async (_name, response, expected) => {
    vi.mocked(getDetailedSummary).mockResolvedValue(response as never);
    const report = vi.fn();

    render(panel({ active: false, onAvailability: report }));
    await waitFor(() => expect(getDetailedSummary).toHaveBeenCalled());

    if (expected) {
      await waitFor(() => expect(answers(report).at(-1)).toBe(true));
    } else {
      await act(async () => {
        for (let i = 0; i < 8; i += 1) await Promise.resolve();
      });
      expect(answers(report)).toContain(false);
      expect(answers(report)).not.toContain(true);
    }
  });

  it("SPEC-ADDON-099 answers true for the dormant line", async () => {
    vi.mocked(getDetailedSummary).mockResolvedValue(generated as never);
    mockActiveSummary.current = {
      data: { has_active_summary: true, summary_note: { file_id: "n1" } },
      loading: false,
    };
    const report = vi.fn();

    render(panel({ active: true, onAvailability: report }));
    await screen.findByTestId("detailed-summary-dormant");

    expect(answers(report)[0]).toBe(false);
    expect(answers(report).at(-1)).toBe(true);
  });

  it("SPEC-ADDON-099 I10 answers false again on a file change", async () => {
    vi.mocked(getDetailedSummary)
      .mockResolvedValueOnce(generated as never)
      .mockReturnValueOnce(new Promise(() => {}) as never);
    const report = vi.fn();
    const { rerender } = render(panel({ active: false, onAvailability: report }));
    await waitFor(() => expect(answers(report).at(-1)).toBe(true));

    rerender(panel({ fileId: "f2", active: false, onAvailability: report }));

    expect(answers(report).at(-1)).toBe(false);
  });

  it("SPEC-ADDON-099 I10 drops an earlier file's answer and its availability", async () => {
    let resolveFirst!: (value: unknown) => void;
    let resolveSecond!: (value: unknown) => void;
    vi.mocked(getDetailedSummary)
      .mockReturnValueOnce(
        new Promise((resolve) => {
          resolveFirst = resolve;
        }) as never,
      )
      .mockReturnValueOnce(
        new Promise((resolve) => {
          resolveSecond = resolve;
        }) as never,
      );
    const report = vi.fn();
    const { rerender } = render(panel({ active: false, onAvailability: report }));
    rerender(panel({ fileId: "f2", active: false, onAvailability: report }));

    resolveFirst(generated);
    await act(async () => {
      for (let i = 0; i < 8; i += 1) await Promise.resolve();
    });

    expect(screen.queryByText(/本作の概要を述べる段落/)).toBeNull();
    expect(answers(report)).not.toContain(true);

    resolveSecond({ available: false, reason: "not_generated", file_id: "f2" });
    await act(async () => {
      for (let i = 0; i < 8; i += 1) await Promise.resolve();
    });

    expect(answers(report).at(-1)).toBe(false);
  });

  it("SPEC-ADDON-099 I10 draws no collapsible header when the host labels it, and shows the body", async () => {
    vi.mocked(getDetailedSummary).mockResolvedValue(generated as never);

    render(panel({ active: true, labelledByHost: true, onAvailability: vi.fn() }));

    expect(await screen.findByText(/本作の概要を述べる段落/)).toBeVisible();
    expect(screen.queryByRole("button", { name: /Expand|Collapse/ })).toBeNull();
    expect(
      screen.queryAllByRole("button").filter((button) =>
        button.hasAttribute("aria-expanded"),
      ),
    ).toEqual([]);
  });
});
