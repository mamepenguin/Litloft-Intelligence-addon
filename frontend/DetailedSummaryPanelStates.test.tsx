import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
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
    citations: [
      {
        section_path: "全体像/0",
        segment_type: "paragraph",
        segment_text: "本作の概要を述べる段落。",
        chunk_ids: ["c1"],
        top_score: 0.9,
        has_citation: true,
      },
    ],
  }),
  editDetailedSummarySection: vi.fn(),
  revertDetailedSummary: vi.fn(),
  regenerateDetailedSummary: vi.fn(),
  getCitationChunkExcerpt: vi.fn().mockResolvedValue(null),
}));

import DetailedSummarySection from "@/addons/intelligence/DetailedSummarySection";
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

beforeEach(() => {
  vi.clearAllMocks();
  window.localStorage.clear();
  mockActiveSummary.current = { data: null, loading: false };
});

describe("SPEC-ADDON-099 I10 detailed summary's other states", () => {
  it("answers true for the generating row", async () => {
    vi.mocked(getDetailedSummary).mockResolvedValue({
      available: false,
      status: "generating",
    } as never);
    const report = vi.fn();

    render(panel({ onAvailability: report }));

    await waitFor(() =>
      expect(report.mock.calls.map(([available]) => available).at(-1)).toBe(true),
    );
  });

  it.each([
    ["absent", undefined, true],
    ["true", true, true],
    ["false", false, false],
  ])(
    "binds its window keys only while active is not false (%s)",
    async (_name, active, bound) => {
      vi.mocked(getDetailedSummary).mockResolvedValue(generated as never);

      render(
        panel({
          labelledByHost: true,
          ...(active === undefined ? {} : { active }),
        }),
      );

      await screen.findByText(/本作の概要を述べる段落/);
      const toggle = await screen.findByTestId("verify-toggle");
      expect(toggle).toHaveAttribute("aria-pressed", "false");

      fireEvent.keyDown(document.body, { key: "v" });

      expect(toggle).toHaveAttribute("aria-pressed", bound ? "true" : "false");
    },
  );
});
