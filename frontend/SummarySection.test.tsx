/**
 * Tests for the SummarySection edit / revert UX.
 *
 * Covers:
 *   1. The "Edit" button switches the section into an editable state with
 *      textareas seeded from the current summary.
 *   2. Saving POSTs via `editSummary` and rehydrates from its response
 *      without a follow-up GET.
 *   3. Cancel restores the read-only view without touching the API.
 *   4. "Edited" badge and "Revert" button surface only when the API
 *      reports `edited_at` / `has_original`.
 *   5. Revert calls `revertSummary` and drops the badge when the
 *      response clears `edited_at`.
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor, act } from "@testing-library/react";
import { NextIntlClientProvider } from "next-intl";

vi.mock("@/addons/intelligence/api", () => ({
  getSummary: vi.fn(),
  editSummary: vi.fn(),
  revertSummary: vi.fn(),
  regenerateSummary: vi.fn(),
}));

import SummarySection from "@/addons/intelligence/SummarySection";
import FileAIActionsButton from "@/addons/intelligence/FileAIActionsButton";
import { resetFileAiActions } from "@/addons/intelligence/fileAiActions";
import {
  editSummary,
  getSummary,
  regenerateSummary,
  revertSummary,
} from "@/addons/intelligence/api";

const aiResponse = {
  available: true,
  file_id: "f1",
  short_summary: "AI short",
  long_summary: "AI long text",
  model: "gemma",
  status: "generated",
  has_original: false,
  edited_at: null,
};

const editedResponse = {
  ...aiResponse,
  short_summary: "user short",
  long_summary: "user long",
  edited_at: "2026-04-16T12:00:00+00:00",
  has_original: true,
};

function renderSection() {
  return render(
    <NextIntlClientProvider locale="en" messages={{}}>
      <SummarySection fileId="f1" drive="drive1" />
    </NextIntlClientProvider>,
  );
}

function renderWithActionMenu() {
  return render(
    <NextIntlClientProvider locale="en" messages={{}}>
      <SummarySection fileId="f1" drive="drive1" />
      <FileAIActionsButton fileId="f1" />
    </NextIntlClientProvider>,
  );
}

describe("SummarySection — the offer moves to the AI menu", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetFileAiActions();
  });

  it("renders nothing and offers generation when none has been made", async () => {
    vi.mocked(getSummary).mockResolvedValue({
      available: false,
      reason: "not_generated",
    } as never);
    renderWithActionMenu();

    const trigger = await screen.findByRole("button", { name: "AI" });
    // The section contributes no control of its own — the menu trigger
    // is the only button on the page. Asserting the count rather than
    // the absence of one label is what makes this catch a section that
    // quietly starts drawing its heading and button again.
    expect(screen.getAllByRole("button")).toHaveLength(1);
    fireEvent.click(trigger);
    fireEvent.click(screen.getByRole("menuitem", { name: /Create AI summary/ }));
    await waitFor(() =>
      expect(regenerateSummary).toHaveBeenCalledWith("f1", "drive1"),
    );
  });

  it("says the run is under way after the menu closes on it", async () => {
    // The menu closes when an item is pressed, so the section is the
    // only place left that can say anything is happening.
    vi.mocked(getSummary).mockResolvedValue({
      available: false,
      reason: "not_generated",
    } as never);
    vi.mocked(regenerateSummary).mockReturnValue(new Promise(() => {}) as never);
    renderWithActionMenu();

    fireEvent.click(await screen.findByRole("button", { name: "AI" }));
    await act(async () => {
      fireEvent.click(screen.getByRole("menuitem", { name: /Create AI summary/ }));
    });

    expect(await screen.findByText(/Creating summary/i)).toBeInTheDocument();
  });

  it("offers nothing for a file type that cannot be summarised", async () => {
    vi.mocked(getSummary).mockResolvedValue({
      available: false,
      reason: "unsupported_type",
    } as never);
    renderWithActionMenu();

    await waitFor(() => expect(getSummary).toHaveBeenCalled());
    expect(screen.queryByRole("button", { name: "AI" })).toBeNull();
  });

  it("offers nothing when there is too little text to summarise", async () => {
    // The note stays — it answers a question the menu could not — but
    // the menu must not offer a run that the backend would skip.
    vi.mocked(getSummary).mockResolvedValue({
      available: false,
      reason: "insufficient_content",
    } as never);
    renderWithActionMenu();

    expect(
      await screen.findByText(/Not enough content to summarize/i),
    ).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "AI" })).toBeNull();
  });

  it("withdraws the offer once a summary exists", async () => {
    vi.mocked(getSummary).mockResolvedValue(aiResponse as never);
    renderWithActionMenu();

    await screen.findByText("AI short");
    expect(screen.queryByRole("button", { name: "AI" })).toBeNull();
  });
});

describe("SummarySection — edit flow", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetFileAiActions();
  });

  it("enters edit mode with the current summary pre-filled", async () => {
    (getSummary as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(
      aiResponse,
    );

    renderSection();

    await waitFor(() =>
      expect(screen.getByText("AI short")).toBeInTheDocument(),
    );
    fireEvent.click(screen.getByRole("button", { name: /^Edit$/ }));

    const shortBox = (await screen.findByLabelText(/Short summary/)) as
      HTMLTextAreaElement;
    const longBox = screen.getByLabelText(/Detailed summary/) as
      HTMLTextAreaElement;

    expect(shortBox.value).toBe("AI short");
    expect(longBox.value).toBe("AI long text");
  });

  it("posts edits and hydrates from the response", async () => {
    (getSummary as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(
      aiResponse,
    );
    (editSummary as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(
      editedResponse,
    );

    renderSection();

    await waitFor(() =>
      expect(screen.getByText("AI short")).toBeInTheDocument(),
    );
    fireEvent.click(screen.getByRole("button", { name: /^Edit$/ }));

    const shortBox = (await screen.findByLabelText(/Short summary/)) as
      HTMLTextAreaElement;
    const longBox = screen.getByLabelText(/Detailed summary/) as
      HTMLTextAreaElement;

    fireEvent.change(shortBox, { target: { value: "user short" } });
    fireEvent.change(longBox, { target: { value: "user long" } });

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /Save/ }));
    });

    await waitFor(() => {
      expect(editSummary).toHaveBeenCalledWith("f1", "drive1", {
        short_summary: "user short",
        long_summary: "user long",
      });
    });

    // Rehydrates from the edit response — the new short text + the
    // "Edited" badge appear without a second getSummary call.
    expect(await screen.findByText("user short")).toBeInTheDocument();
    expect(screen.getByText(/Edited/)).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /Revert to AI version/ }),
    ).toBeInTheDocument();
    expect(getSummary).toHaveBeenCalledTimes(1); // only the initial load
  });

  it("cancel exits edit mode without calling the API", async () => {
    (getSummary as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(
      aiResponse,
    );

    renderSection();

    await waitFor(() =>
      expect(screen.getByText("AI short")).toBeInTheDocument(),
    );
    fireEvent.click(screen.getByRole("button", { name: /^Edit$/ }));

    const shortBox = (await screen.findByLabelText(/Short summary/)) as
      HTMLTextAreaElement;
    fireEvent.change(shortBox, { target: { value: "throwaway" } });
    fireEvent.click(screen.getByRole("button", { name: /Cancel/ }));

    expect(editSummary).not.toHaveBeenCalled();
    // Back to the read-only view with the original AI text.
    expect(screen.getByText("AI short")).toBeInTheDocument();
    expect(screen.queryByLabelText(/Short summary/)).toBeNull();
  });

  it("hides the Revert button for untouched AI summaries", async () => {
    (getSummary as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(
      aiResponse,
    );

    renderSection();

    await waitFor(() =>
      expect(screen.getByText("AI short")).toBeInTheDocument(),
    );
    expect(screen.queryByText(/Edited/)).toBeNull();
    expect(
      screen.queryByRole("button", { name: /Revert to AI version/ }),
    ).toBeNull();
  });

  it("reverts to the AI version and drops the badge", async () => {
    (getSummary as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(
      editedResponse,
    );
    (revertSummary as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(
      aiResponse,
    );

    renderSection();

    await waitFor(() =>
      expect(screen.getByText("user short")).toBeInTheDocument(),
    );
    expect(screen.getByText(/Edited/)).toBeInTheDocument();

    await act(async () => {
      fireEvent.click(
        screen.getByRole("button", { name: /Revert to AI version/ }),
      );
    });

    await waitFor(() => {
      expect(revertSummary).toHaveBeenCalledWith("f1", "drive1");
    });
    expect(await screen.findByText("AI short")).toBeInTheDocument();
    expect(screen.queryByText(/Edited/)).toBeNull();
    expect(
      screen.queryByRole("button", { name: /Revert to AI version/ }),
    ).toBeNull();
  });

  it("disables Save when either field is empty", async () => {
    (getSummary as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(
      aiResponse,
    );

    renderSection();

    await waitFor(() =>
      expect(screen.getByText("AI short")).toBeInTheDocument(),
    );
    fireEvent.click(screen.getByRole("button", { name: /^Edit$/ }));

    const shortBox = (await screen.findByLabelText(/Short summary/)) as
      HTMLTextAreaElement;
    fireEvent.change(shortBox, { target: { value: "" } });

    const save = screen.getByRole("button", { name: /Save/ }) as
      HTMLButtonElement;
    expect(save.disabled).toBe(true);
  });
});
