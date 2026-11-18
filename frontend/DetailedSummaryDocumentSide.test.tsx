import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { NextIntlClientProvider } from "next-intl";

const routerPush = vi.hoisted(() => vi.fn());
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: routerPush, replace: vi.fn(), back: vi.fn(), prefetch: vi.fn() }),
  usePathname: () => "/drive/drive1",
  useSearchParams: () => new URLSearchParams(),
}));

vi.mock("@/hooks/useWebSocket", () => ({
  useWebSocket: () => null,
}));

type ActiveSummaryState = {
  data: { has_active_summary: boolean; summary_note?: unknown } | null;
  loading: boolean;
};

const mockActiveSummary = vi.hoisted(() => ({
  current: { data: null, loading: false } as ActiveSummaryState,
}));
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
    getFile: vi.fn().mockResolvedValue({ id: "f1", filename: "paper.pdf" }),
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
        top_score: 0.95,
        has_citation: true,
      },
    ],
  }),
  editDetailedSummarySection: vi.fn(),
  revertDetailedSummary: vi.fn(),
  regenerateDetailedSummary: vi.fn(),
  getCitationChunkExcerpt: vi.fn(),
}));

import DetailedSummarySection from "@/addons/intelligence/DetailedSummarySection";
import FileAIActionsButton from "@/addons/intelligence/FileAIActionsButton";
import { resetFileAiActions } from "@/lib/fileAiActions";
import {
  getCitationChunkExcerpt,
  getDetailedSummary,
  regenerateDetailedSummary,
} from "@/addons/intelligence/api";

const Section = DetailedSummarySection as unknown as React.ComponentType<
  Record<string, unknown>
>;

const generated = {
  available: true,
  status: "generated",
  file_id: "f1",
  detailed_summary: "## 全体像\n本作の概要を述べる段落。\n",
  model: "test-model",
  edited_at: null,
  has_original: false,
};

type Target = { fileId: string; page?: number; section?: number };

function tab(props: Record<string, unknown>) {
  return (
    <NextIntlClientProvider locale="ja" messages={{}}>
      <Section
        fileId="f1"
        drive="drive1"
        fillHeight
        labelledByHost
        active
        onAvailability={vi.fn()}
        select={vi.fn()}
        {...props}
      />
    </NextIntlClientProvider>
  );
}

/** Opens the citation's excerpt card and returns the card, the jump control. */
async function openExcerptCard(excerpt: Record<string, unknown>) {
  vi.mocked(getCitationChunkExcerpt).mockResolvedValue({
    chunk_id: "c1",
    file_id: "f1",
    prefix: "",
    target: "章の本文。",
    suffix: "",
    start_time: null,
    end_time: null,
    page: null,
    section: null,
    section_title: null,
    ...excerpt,
  } as never);
  await screen.findByText(/本作の概要を述べる段落/);
  const toggle = await screen.findByTestId("verify-toggle");
  if (toggle.getAttribute("aria-pressed") !== "true") fireEvent.click(toggle);
  await act(async () => {
    fireEvent.click(
      await screen.findByRole("button", { name: /Clear source citation|根拠が明確/ }),
    );
  });
  return (await screen.findByText("章の本文。")).closest("button")!;
}

beforeEach(() => {
  vi.clearAllMocks();
  mockActiveSummary.current = { data: null, loading: false };
  resetFileAiActions();
  window.localStorage.clear();
});

afterEach(() => {
  cleanup();
});

describe("SPEC-ADDON-135 a citation's jump asks the host first", () => {
  it.each([
    [{ page: 4 }, { fileId: "f1", page: 4 }],
    [{ section: 3, section_title: "Chapter Three" }, { fileId: "f1", section: 3 }],
  ])("jumps in place when openInViewer answers true (%o)", async (excerpt, target) => {
    vi.mocked(getDetailedSummary).mockResolvedValue(generated as never);
    const openInViewer = vi.fn((_target: Target) => true);
    render(tab({ openInViewer }));

    fireEvent.click(await openExcerptCard(excerpt));

    expect(openInViewer.mock.calls).toEqual([[target]]);
    expect(routerPush).not.toHaveBeenCalled();
  });

  it.each([
    [{ page: 4 }, "/files/f1?page=4"],
    [{ section: 3, section_title: "Chapter Three" }, "/files/f1?section=3"],
  ])("navigates as before when openInViewer answers false (%o)", async (excerpt, href) => {
    vi.mocked(getDetailedSummary).mockResolvedValue(generated as never);
    const openInViewer = vi.fn((_target: Target) => false);
    render(tab({ openInViewer }));

    fireEvent.click(await openExcerptCard(excerpt));

    expect(openInViewer).toHaveBeenCalledTimes(1);
    expect(routerPush.mock.calls).toEqual([[href]]);
    expect(openInViewer.mock.invocationCallOrder[0]).toBeLessThan(
      routerPush.mock.invocationCallOrder[0],
    );
  });
});

describe("SPEC-ADDON-135 starting a detailed summary selects its tab", () => {
  it("reports available, then calls select(), when the reader starts it from the AI menu", async () => {
    vi.mocked(getDetailedSummary).mockResolvedValue({
      available: false,
      reason: "not_generated",
    } as never);
    vi.mocked(regenerateDetailedSummary).mockResolvedValue({
      available: false,
      status: "generating",
    } as never);
    const onAvailability = vi.fn();
    const select = vi.fn();

    render(
      <NextIntlClientProvider locale="ja" messages={{}}>
        <Section
          fileId="f1"
          drive="drive1"
          fillHeight
          labelledByHost
          active={false}
          onAvailability={onAvailability}
          select={select}
        />
        <FileAIActionsButton fileId="f1" />
      </NextIntlClientProvider>,
    );
    fireEvent.click(await screen.findByRole("button", { name: "AI" }));
    fireEvent.click(screen.getByRole("menuitem", { name: /Create detailed summary/ }));

    await waitFor(() => expect(select).toHaveBeenCalledTimes(1));
    const selectedAt = select.mock.invocationCallOrder[0];
    const reportsBefore = onAvailability.mock.calls.filter(
      (_call, index) => onAvailability.mock.invocationCallOrder[index] < selectedAt,
    );
    expect(reportsBefore.at(-1)).toEqual([true]);
  });
});

describe("SPEC-ADDON-135 the AI menu starts generation beside a summary note", () => {
  it("generates and selects when a summary note is active and nothing was generated yet", async () => {
    mockActiveSummary.current = {
      data: { has_active_summary: true, summary_note: { file_id: "n1" } },
      loading: false,
    };
    vi.mocked(getDetailedSummary).mockResolvedValue({
      available: false,
      reason: "not_generated",
    } as never);
    vi.mocked(regenerateDetailedSummary).mockReturnValue(
      new Promise(() => {}) as never,
    );
    const select = vi.fn();

    render(
      <NextIntlClientProvider locale="ja" messages={{}}>
        <Section
          fileId="f1"
          drive="drive1"
          fillHeight
          labelledByHost
          active={false}
          onAvailability={vi.fn()}
          select={select}
        />
        <FileAIActionsButton fileId="f1" />
      </NextIntlClientProvider>,
    );
    fireEvent.click(await screen.findByRole("button", { name: "AI" }));
    fireEvent.click(screen.getByRole("menuitem", { name: /Create detailed summary/ }));

    await waitFor(() => expect(regenerateDetailedSummary).toHaveBeenCalledTimes(1));
    expect(select).toHaveBeenCalledTimes(1);
    expect(screen.queryByText(/The summary will be created again/)).toBeNull();
  });
});

describe("SPEC-ADDON-135 regenerating inside the tab leaves the selection alone", () => {
  it.each<[string, ActiveSummaryState]>([
    ["no summary note", { data: null, loading: false }],
    [
      "a summary note",
      {
        data: { has_active_summary: true, summary_note: { file_id: "n1" } },
        loading: false,
      },
    ],
  ])("retries a failed run without selecting its tab (%s)", async (_name, note) => {
    mockActiveSummary.current = note;
    vi.mocked(getDetailedSummary).mockResolvedValue({
      available: false,
      status: "failed",
      error: "boom",
    } as never);
    vi.mocked(regenerateDetailedSummary).mockReturnValue(
      new Promise(() => {}) as never,
    );
    const select = vi.fn();

    render(tab({ select, active: false }));
    fireEvent.click(await screen.findByRole("button", { name: /Retry/ }));

    await waitFor(() => expect(regenerateDetailedSummary).toHaveBeenCalledTimes(1));
    expect(select).not.toHaveBeenCalled();
  });

  it("regenerates from the body without selecting its tab", async () => {
    vi.mocked(getDetailedSummary).mockResolvedValue(generated as never);
    vi.mocked(regenerateDetailedSummary).mockReturnValue(
      new Promise(() => {}) as never,
    );
    const select = vi.fn();

    render(tab({ select }));
    fireEvent.click(await screen.findByRole("button", { name: /^Create again$/ }));

    await waitFor(() => expect(regenerateDetailedSummary).toHaveBeenCalledTimes(1));
    expect(select).not.toHaveBeenCalled();
  });

  it("confirms a regenerate over a summary note without selecting its tab", async () => {
    mockActiveSummary.current = {
      data: { has_active_summary: true, summary_note: { file_id: "n1" } },
      loading: false,
    };
    vi.mocked(getDetailedSummary).mockResolvedValue(generated as never);
    vi.mocked(regenerateDetailedSummary).mockReturnValue(
      new Promise(() => {}) as never,
    );
    const select = vi.fn();

    render(tab({ select }));
    fireEvent.click(await screen.findByRole("button", { name: /^Create again$/ }));
    const message = await screen.findByText(/The summary will be created again/);
    const dialog = message.closest("[aria-modal='true']")!;
    fireEvent.click(
      within(dialog as HTMLElement).getByRole("button", { name: /^Create again$/ }),
    );

    await waitFor(() => expect(regenerateDetailedSummary).toHaveBeenCalledTimes(1));
    expect(select).not.toHaveBeenCalled();
  });
});

describe("SPEC-ADDON-135 no empty tab beside a summary note", () => {
  it("never reports available while a summary note is active and no detailed summary was generated", async () => {
    mockActiveSummary.current = {
      data: { has_active_summary: true, summary_note: { file_id: "n1" } },
      loading: false,
    };
    vi.mocked(getDetailedSummary).mockResolvedValue({
      available: false,
      reason: "not_generated",
    } as never);
    const onAvailability = vi.fn();
    render(tab({ onAvailability, active: false }));

    await waitFor(() => expect(getDetailedSummary).toHaveBeenCalled());
    await act(async () => {
      for (let i = 0; i < 8; i += 1) await Promise.resolve();
    });
    expect(onAvailability.mock.calls.map(([available]) => available)).not.toContain(true);
  });
});
