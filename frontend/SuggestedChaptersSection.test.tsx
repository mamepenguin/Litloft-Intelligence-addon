import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const translations = vi.hoisted(() => ({
  aiFileActions: "AI",
  suggestedChapters: "AI chapter candidates",
  generateChapters: "Create AI chapter candidates",
  generatingChapters: "Creating chapters...",
  approveChapters: "Approve all",
  approvingChapters: "Approving...",
  dismissChapters: "Dismiss",
  dismissingChapters: "Dismissing...",
  regenerateChapters: "Create again",
  chapterCandidatesLoadError: "Could not load chapter candidates. Try again.",
  chapterCandidatesActionError: "The chapter action failed. Try again.",
  chapterCandidatesGenerationFailed:
    "Chapter generation failed. Try creating them again.",
  chapterCandidatesTokenBudget:
    "The model ran out of output budget before it finished, so no chapters "
    + "came back. Raise llm.max_tokens, or set llm.reasoning to disabled if "
    + "the provider is spending the budget on thinking.",
}));
const translate = vi.hoisted(
  () => (key: keyof typeof translations) => translations[key],
);

vi.mock("next-intl", () => ({
  useTranslations: () => translate,
}));

vi.mock("@/addons/intelligence/api", () => ({
  getSuggestedChapters: vi.fn(),
  generateSuggestedChapters: vi.fn(),
  approveSuggestedChapters: vi.fn(),
  dismissSuggestedChapters: vi.fn(),
}));

vi.mock("@/hooks/usePolicy", () => ({
  usePolicy: vi.fn(),
}));

const wsEvent = vi.hoisted(() => ({ value: null as null | {
  event: string;
  data: Record<string, unknown>;
} }));
vi.mock("@/hooks/useWebSocket", () => ({
  useWebSocket: vi.fn((eventFilter?: string) => {
    if (!wsEvent.value || wsEvent.value.event !== eventFilter) return null;
    return wsEvent.value;
  }),
}));

import SuggestedChaptersSection from "@/addons/intelligence/SuggestedChaptersSection";
import FileAIActionsButton from "@/addons/intelligence/FileAIActionsButton";
import { resetFileAiActions } from "@/addons/intelligence/fileAiActions";
import {
  approveSuggestedChapters,
  dismissSuggestedChapters,
  generateSuggestedChapters,
  getSuggestedChapters,
} from "@/addons/intelligence/api";
import { usePolicy } from "@/hooks/usePolicy";
import { FILE_CHAPTERS_UPDATED_EVENT } from "@/lib/addonEvents";

const pendingResponse = {
  enabled: true,
  available: true,
  file_id: "f1",
  model: "test-model",
  status: "pending" as const,
  created_at: "2026-08-12T00:00:00Z",
  chapters: [
    { start_time: 0, end_time: 65, title: "Opening" },
    { start_time: 65, end_time: null, title: "Main topic" },
  ],
};

function renderSection() {
  return render(
    <SuggestedChaptersSection fileId="f1" drive="media" fileType="video" />,
  );
}

/**
 * The section plus the action row's "AI" menu. With no candidates
 * waiting, the section renders nothing and the offer to create some is
 * only observable through the menu.
 */
function renderWithActionMenu() {
  return render(
    <>
      <SuggestedChaptersSection fileId="f1" drive="media" fileType="video" />
      <FileAIActionsButton fileId="f1" />
    </>,
  );
}

async function openActionMenu() {
  fireEvent.click(await screen.findByRole("button", { name: "AI" }));
}

describe("SuggestedChaptersSection", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    wsEvent.value = null;
    resetFileAiActions();
    vi.mocked(usePolicy).mockReturnValue({ enabled: true, isLoading: false });
    vi.mocked(getSuggestedChapters).mockResolvedValue(pendingResponse);
    vi.mocked(generateSuggestedChapters).mockResolvedValue(undefined);
    vi.mocked(approveSuggestedChapters).mockResolvedValue(undefined);
    vi.mocked(dismissSuggestedChapters).mockResolvedValue(undefined);
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("renders a pending chapter preview with timestamps, titles, and model", async () => {
    renderSection();

    expect(await screen.findByText("Opening")).toBeInTheDocument();
    expect(screen.getByText("Main topic")).toBeInTheDocument();
    expect(screen.getByText("0:00")).toBeInTheDocument();
    expect(screen.getByText("1:05")).toBeInTheDocument();
    expect(screen.getByText("test-model")).toBeInTheDocument();
  });

  it("approves the whole set and announces the core refresh event", async () => {
    const listener = vi.fn();
    window.addEventListener(FILE_CHAPTERS_UPDATED_EVENT, listener);

    renderSection();
    fireEvent.click(await screen.findByRole("button", { name: "Approve all" }));

    await waitFor(() => {
      expect(approveSuggestedChapters).toHaveBeenCalledWith("f1", "media");
      expect(listener).toHaveBeenCalledOnce();
    });
    const event = listener.mock.calls[0][0] as CustomEvent<{ fileId: string }>;
    expect(event.detail).toEqual({ fileId: "f1" });
    // Approved chapters live in the core chapter rail; this section has
    // nothing left to say and stops taking a row.
    expect(screen.queryByText("Chapter candidates approved")).not.toBeInTheDocument();
    expect(screen.queryByText("Opening")).not.toBeInTheDocument();

    window.removeEventListener(FILE_CHAPTERS_UPDATED_EVENT, listener);
  });

  it("dismisses the candidate set and hands 'create again' to the AI menu", async () => {
    renderWithActionMenu();
    fireEvent.click(await screen.findByRole("button", { name: "Dismiss" }));

    await waitFor(() => {
      expect(dismissSuggestedChapters).toHaveBeenCalledWith("f1", "media");
    });
    expect(screen.queryByText("Chapter candidates dismissed")).not.toBeInTheDocument();

    await openActionMenu();
    expect(
      screen.getByRole("menuitem", { name: "Create again" }),
    ).toBeInTheDocument();
  });

  it("offers generation through the AI menu when no candidates exist", async () => {
    vi.mocked(getSuggestedChapters).mockResolvedValue({
      enabled: true,
      available: false,
      chapters: [],
    });
    renderWithActionMenu();

    // The menu trigger is the only button: the section draws nothing.
    await screen.findByRole("button", { name: "AI" });
    expect(screen.getAllByRole("button")).toHaveLength(1);

    await openActionMenu();
    fireEvent.click(
      screen.getByRole("menuitem", { name: "Create AI chapter candidates" }),
    );

    await waitFor(() => {
      expect(generateSuggestedChapters).toHaveBeenCalledWith("f1", "media");
    });
  });

  it("withdraws the AI menu entry while candidates are waiting", async () => {
    renderWithActionMenu();

    await screen.findByText("Opening");
    expect(screen.queryByRole("button", { name: "AI" })).toBeNull();
  });

  it("offers nothing to the AI menu when the drive has the feature off", async () => {
    vi.mocked(getSuggestedChapters).mockResolvedValue({
      enabled: false,
      available: false,
      chapters: [],
    });
    renderWithActionMenu();

    await waitFor(() => expect(getSuggestedChapters).toHaveBeenCalled());
    expect(screen.queryByRole("button", { name: "AI" })).toBeNull();
  });

  it("does not render when chapter suggestions are disabled for the drive", async () => {
    vi.mocked(getSuggestedChapters).mockResolvedValue({
      enabled: false,
      available: false,
      chapters: [],
    });
    const { container } = renderSection();

    await waitFor(() => expect(getSuggestedChapters).toHaveBeenCalled());
    expect(container).toBeEmptyDOMElement();
  });

  it("does not request chapter suggestions for non-media files", async () => {
    const { container } = render(
      <SuggestedChaptersSection fileId="f1" drive="media" fileType="document" />,
    );

    await act(async () => undefined);
    expect(getSuggestedChapters).not.toHaveBeenCalled();
    expect(container).toBeEmptyDOMElement();
  });

  it("does not request chapter suggestions when drive policy is off", async () => {
    vi.mocked(usePolicy).mockReturnValue({ enabled: false, isLoading: false });
    const { container } = renderSection();

    await act(async () => undefined);
    expect(usePolicy).toHaveBeenCalledWith(
      "media",
      "intelligence",
      "chapter_suggestions",
    );
    expect(getSuggestedChapters).not.toHaveBeenCalled();
    expect(container).toBeEmptyDOMElement();
  });

  it("waits for drive policy before requesting chapter suggestions", async () => {
    vi.mocked(usePolicy).mockReturnValue({ enabled: true, isLoading: true });
    const { container } = renderSection();

    await act(async () => undefined);
    expect(getSuggestedChapters).not.toHaveBeenCalled();
    expect(container).toBeEmptyDOMElement();
  });

  it("reloads from the ready WebSocket event without a polling deadline", async () => {
    vi.mocked(getSuggestedChapters)
      .mockResolvedValueOnce(pendingResponse)
      .mockResolvedValueOnce({
        ...pendingResponse,
        created_at: "2026-08-12T00:01:00Z",
        chapters: [
          { start_time: 12, end_time: null, title: "Replacement" },
        ],
      });

    const { rerender } = renderSection();
    expect(await screen.findByText("Opening")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Create again" }));

    await waitFor(() => {
      expect(generateSuggestedChapters).toHaveBeenCalledWith("f1", "media");
    });
    expect(getSuggestedChapters).toHaveBeenCalledTimes(1);

    wsEvent.value = {
      event: "intelligence.chapter_suggestions.ready",
      data: { file_id: "f1" },
    };
    rerender(
      <SuggestedChaptersSection fileId="f1" drive="media" fileType="video" />,
    );

    expect(await screen.findByText("Replacement")).toBeInTheDocument();
    expect(getSuggestedChapters).toHaveBeenCalledTimes(2);
  });

  it("ignores ready events for another file", async () => {
    const { rerender } = renderSection();
    expect(await screen.findByText("Opening")).toBeInTheDocument();

    wsEvent.value = {
      event: "intelligence.chapter_suggestions.ready",
      data: { file_id: "other" },
    };
    rerender(
      <SuggestedChaptersSection fileId="f1" drive="media" fileType="video" />,
    );

    expect(getSuggestedChapters).toHaveBeenCalledTimes(1);
  });

  it("stops generating and explains a failed WebSocket completion", async () => {
    const { rerender } = renderSection();
    expect(await screen.findByText("Opening")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Create again" }));
    expect(
      await screen.findByRole("button", { name: "Creating chapters..." }),
    ).toBeDisabled();

    wsEvent.value = {
      event: "intelligence.chapter_suggestions.failed",
      data: { file_id: "f1", reason: "invalid_model_output" },
    };
    rerender(
      <SuggestedChaptersSection fileId="f1" drive="media" fileType="video" />,
    );

    expect(
      await screen.findByText("Chapter generation failed. Try creating them again."),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Create again" })).toBeEnabled();
    expect(getSuggestedChapters).toHaveBeenCalledTimes(1);
  });

  it("names the token budget when that is what failed", async () => {
    const { rerender } = renderSection();
    expect(await screen.findByText("Opening")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Create again" }));
    expect(
      await screen.findByRole("button", { name: "Creating chapters..." }),
    ).toBeDisabled();

    wsEvent.value = {
      event: "intelligence.chapter_suggestions.failed",
      data: { file_id: "f1", reason: "model_token_budget" },
    };
    rerender(
      <SuggestedChaptersSection fileId="f1" drive="media" fileType="video" />,
    );

    expect(
      await screen.findByText(
        "The model ran out of output budget before it finished, so no "
        + "chapters came back. Raise llm.max_tokens, or set llm.reasoning to "
        + "disabled if the provider is spending the budget on thinking.",
      ),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Create again" })).toBeEnabled();
  });

  it("shows an actionable error when an operation fails", async () => {
    vi.mocked(approveSuggestedChapters).mockRejectedValue(new Error("503"));
    renderSection();
    fireEvent.click(await screen.findByRole("button", { name: "Approve all" }));

    expect(
      await screen.findByText("The chapter action failed. Try again."),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Approve all" })).toBeEnabled();
  });

  it("shows an actionable error when the initial request fails", async () => {
    vi.mocked(getSuggestedChapters).mockRejectedValue(new Error("offline"));
    renderSection();

    expect(
      await screen.findByText("Could not load chapter candidates. Try again."),
    ).toBeInTheDocument();
  });
});
