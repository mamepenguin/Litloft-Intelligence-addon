import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const translations = vi.hoisted(() => ({
  suggestedChapters: "AI chapter candidates",
  generateChapters: "Create AI chapter candidates",
  generatingChapters: "Creating chapters...",
  approveChapters: "Approve all",
  approvingChapters: "Approving...",
  dismissChapters: "Dismiss",
  dismissingChapters: "Dismissing...",
  regenerateChapters: "Create again",
  chapterCandidatesAccepted: "Chapter candidates approved",
  chapterCandidatesDismissed: "Chapter candidates dismissed",
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

describe("SuggestedChaptersSection", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    wsEvent.value = null;
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
    expect(screen.getByText("Chapter candidates approved")).toBeInTheDocument();
    expect(screen.queryByText("Opening")).not.toBeInTheDocument();

    window.removeEventListener(FILE_CHAPTERS_UPDATED_EVENT, listener);
  });

  it("dismisses the candidate set into the compact regenerate state", async () => {
    renderSection();
    fireEvent.click(await screen.findByRole("button", { name: "Dismiss" }));

    await waitFor(() => {
      expect(dismissSuggestedChapters).toHaveBeenCalledWith("f1", "media");
    });
    expect(screen.getByText("Chapter candidates dismissed")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Create again" })).toBeInTheDocument();
  });

  it("shows compact generate UI when no candidates exist", async () => {
    vi.mocked(getSuggestedChapters).mockResolvedValue({
      enabled: true,
      available: false,
      chapters: [],
    });
    renderSection();

    const button = await screen.findByRole("button", {
      name: "Create AI chapter candidates",
    });
    fireEvent.click(button);

    await waitFor(() => {
      expect(generateSuggestedChapters).toHaveBeenCalledWith("f1", "media");
    });
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
