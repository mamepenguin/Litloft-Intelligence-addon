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

vi.mock("@/addons/intelligence/api", async () => {
  const actual = await vi.importActual<
    typeof import("@/addons/intelligence/api")
  >("@/addons/intelligence/api");
  return {
    getSummary: vi.fn(),
    editSummary: vi.fn(),
    revertSummary: vi.fn(),
    regenerateSummary: vi.fn(),
    fetchLLMChoices: vi.fn(async () => {
      throw new Error("not stubbed");
    }),
    isProfileUnavailable: actual.isProfileUnavailable,
    LLMRequestError: actual.LLMRequestError,
  };
});

import { ShortcutsProvider } from "@/components/ShortcutsProvider";
import { useShortcuts } from "@/hooks/useShortcuts";
import SummarySection from "@/addons/intelligence/SummarySection";
import FileAIActionsButton from "@/addons/intelligence/FileAIActionsButton";
import { resetFileAiActions } from "@/lib/fileAiActions";
import {
  editSummary,
  fetchLLMChoices,
  getSummary,
  LLMRequestError,
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
      expect(regenerateSummary).toHaveBeenCalledWith("f1", "drive1", undefined),
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

describe("SummarySection — long summary shape", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetFileAiActions();
  });

  it.each([
    ["- one\n- two\n- three", ["one", "two", "three"]],
    ["- one\n\n- two\n", ["one", "two"]],
  ])("renders %j as a list without markers", async (long, items) => {
    vi.mocked(getSummary).mockResolvedValue({
      ...aiResponse,
      long_summary: long,
    } as never);
    renderSection();

    const list = await screen.findByRole("list");
    expect(
      Array.from(list.querySelectorAll("li")).map((li) => li.textContent),
    ).toEqual(items);
  });

  it.each([
    "A plain paragraph.",
    "- one\nnot a bullet",
    "intro\n- one\n- two",
  ])("renders %j as the paragraph it was", async (long) => {
    vi.mocked(getSummary).mockResolvedValue({
      ...aiResponse,
      long_summary: long,
    } as never);
    renderSection();

    await waitFor(() =>
      expect(screen.getByText("AI short")).toBeInTheDocument(),
    );
    expect(screen.queryByRole("list")).not.toBeInTheDocument();
    expect(
      screen.getByText((_, el) => el?.tagName === "P" && el.textContent === long),
    ).toBeInTheDocument();
  });
});

describe("SummarySection — regenerate with another model", () => {
  const UNAVAILABLE =
    "This model can't be used right now. The choices have been updated.";
  const local = { name: "local", model: "gemma3", offhost: false };
  const cloud = { name: "cloud", model: "gpt-x", offhost: true };

  beforeEach(() => {
    vi.clearAllMocks();
    resetFileAiActions();
    vi.mocked(getSummary).mockResolvedValue(aiResponse as never);
  });

  const trigger = () =>
    screen.queryByRole("button", { name: "Regenerate with…" });

  it.each([
    { label: "0 choices, auto set", value: { auto: "local", choices: [] }, shown: false },
    { label: "1 choice, auto set", value: { auto: "local", choices: [local] }, shown: false },
    { label: "1 choice, auto null", value: { auto: null, choices: [local] }, shown: true },
    { label: "2 choices, auto set", value: { auto: "local", choices: [local, cloud] }, shown: true },
    { label: "0 choices, auto null", value: { auto: null, choices: [] }, shown: false },
  ])("$label → offered: $shown", async ({ value, shown }) => {
    vi.mocked(fetchLLMChoices).mockResolvedValue(value as never);
    renderSection();
    await screen.findByText("AI short");
    await waitFor(() =>
      expect(fetchLLMChoices).toHaveBeenCalledWith("summaries", "drive1"),
    );
    if (shown) {
      expect(await screen.findByRole("button", { name: "Regenerate with…" })).toBeInTheDocument();
    } else {
      await act(async () => {});
      expect(trigger()).toBeNull();
    }
  });

  it("offers nothing when the choices request fails", async () => {
    vi.mocked(fetchLLMChoices).mockRejectedValue(new Error("404"));
    renderSection();
    await screen.findByText("AI short");
    await waitFor(() => expect(fetchLLMChoices).toHaveBeenCalledTimes(1));
    await act(async () => {});
    expect(trigger()).toBeNull();
  });

  it("does not ask for choices while the feature is off", async () => {
    vi.mocked(getSummary).mockResolvedValue({ available: false } as never);
    vi.mocked(fetchLLMChoices).mockResolvedValue({
      auto: "local",
      choices: [local, cloud],
    } as never);
    renderSection();
    await act(async () => {});
    expect(fetchLLMChoices).not.toHaveBeenCalled();
    expect(trigger()).toBeNull();
  });

  it("lists each choice as name — model and marks the external one", async () => {
    vi.mocked(fetchLLMChoices).mockResolvedValue({
      auto: "local",
      choices: [local, cloud],
    } as never);
    renderSection();
    fireEvent.click(await screen.findByRole("button", { name: "Regenerate with…" }));
    expect(screen.getAllByRole("menuitem")).toEqual([
      screen.getByRole("menuitem", { name: "local — gemma3" }),
      screen.getByRole("menuitem", { name: "cloud — gpt-x (External server)" }),
    ]);
  });

  it("keeps the menu's keys from the page's shortcuts", async () => {
    vi.mocked(fetchLLMChoices).mockResolvedValue({
      auto: "local",
      choices: [local, cloud, { name: "third", model: "m3", offhost: false }],
    } as never);
    const pageKeys = vi.fn();
    function PageShortcuts() {
      useShortcuts(
        "page",
        "Page",
        ["arrowdown", "arrowup", "home", "end", "escape"].map((key) => ({
          key,
          label: key,
          handler: () => pageKeys(key),
        })),
      );
      return null;
    }
    render(
      <NextIntlClientProvider locale="en" messages={{}}>
        <ShortcutsProvider>
          <PageShortcuts />
          <SummarySection fileId="f1" drive="drive1" />
        </ShortcutsProvider>
      </NextIntlClientProvider>,
    );
    const button = await screen.findByRole("button", { name: "Regenerate with…" });
    fireEvent.click(button);
    const [first, second, third] = screen.getAllByRole("menuitem");
    await waitFor(() => expect(first).toHaveFocus());

    fireEvent.keyDown(first, { key: "ArrowDown" });
    expect(second).toHaveFocus();
    fireEvent.keyDown(second, { key: "End" });
    expect(third).toHaveFocus();
    fireEvent.keyDown(third, { key: "ArrowDown" });
    expect(first).toHaveFocus();
    fireEvent.keyDown(first, { key: "ArrowUp" });
    expect(third).toHaveFocus();
    fireEvent.keyDown(third, { key: "Home" });
    expect(first).toHaveFocus();
    fireEvent.keyDown(first, { key: "Escape" });
    expect(screen.queryByRole("menu")).toBeNull();
    expect(button).toHaveFocus();
    expect(pageKeys).not.toHaveBeenCalled();

    fireEvent.keyDown(button, { key: "ArrowDown" });
    expect(pageKeys).toHaveBeenCalledWith("arrowdown");
  });

  it("sends the chosen profile, and none from the plain button", async () => {
    vi.mocked(regenerateSummary).mockReturnValue(new Promise(() => {}) as never);
    vi.mocked(fetchLLMChoices).mockResolvedValue({
      auto: "local",
      choices: [local, cloud],
    } as never);
    renderSection();
    fireEvent.click(await screen.findByRole("button", { name: "Regenerate with…" }));
    fireEvent.click(screen.getByRole("menuitem", { name: /cloud — gpt-x/ }));
    expect(regenerateSummary).toHaveBeenCalledTimes(1);
    expect(regenerateSummary).toHaveBeenCalledWith("f1", "drive1", "cloud");
  });

  it("disables the plain button and every choice while a run is in flight", async () => {
    vi.mocked(regenerateSummary).mockReturnValue(new Promise(() => {}) as never);
    vi.mocked(fetchLLMChoices).mockResolvedValue({
      auto: "local",
      choices: [local, cloud],
    } as never);
    renderSection();
    fireEvent.click(await screen.findByRole("button", { name: "Regenerate with…" }));
    fireEvent.click(screen.getByRole("button", { name: /^Create again$/ }));

    expect(regenerateSummary).toHaveBeenCalledWith("f1", "drive1", undefined);
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /Creating summary/ })).toBeDisabled(),
    );
    for (const item of screen.getAllByRole("menuitem")) {
      expect(item).toBeDisabled();
    }
    expect(screen.getByRole("button", { name: "Regenerate with…" })).toBeDisabled();
  });

  it("says the model is unavailable and asks for the choices again", async () => {
    vi.mocked(regenerateSummary).mockRejectedValue(
      new LLMRequestError("API error: 400", 400, "profile_unavailable"),
    );
    vi.mocked(fetchLLMChoices).mockResolvedValue({
      auto: "local",
      choices: [local, cloud],
    } as never);
    renderSection();
    fireEvent.click(await screen.findByRole("button", { name: "Regenerate with…" }));
    await act(async () => {
      fireEvent.click(screen.getByRole("menuitem", { name: /cloud — gpt-x/ }));
    });

    expect(
      await screen.findByText(
        "This model can't be used right now. The choices have been updated.",
      ),
    ).toBeInTheDocument();
    await waitFor(() => expect(fetchLLMChoices).toHaveBeenCalledTimes(2));
  });

  async function rejectChoice() {
    fireEvent.click(await screen.findByRole("button", { name: "Regenerate with…" }));
    await act(async () => {
      fireEvent.click(screen.getByRole("menuitem", { name: /cloud — gpt-x/ }));
    });
    await screen.findByText(UNAVAILABLE);
  }

  it("drops the unavailable-model note when the next run starts", async () => {
    vi.mocked(regenerateSummary)
      .mockRejectedValueOnce(
        new LLMRequestError("API error: 400", 400, "profile_unavailable"),
      )
      .mockReturnValue(new Promise(() => {}) as never);
    vi.mocked(fetchLLMChoices).mockResolvedValue({
      auto: "local",
      choices: [local, cloud],
    } as never);
    renderSection();
    await rejectChoice();

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /^Create again$/ }));
    });
    expect(regenerateSummary).toHaveBeenCalledTimes(2);
    expect(screen.queryByText(UNAVAILABLE)).toBeNull();
  });

  it("drops the unavailable-model note on the next file", async () => {
    vi.mocked(regenerateSummary).mockRejectedValueOnce(
      new LLMRequestError("API error: 400", 400, "profile_unavailable"),
    );
    vi.mocked(fetchLLMChoices).mockResolvedValue({
      auto: "local",
      choices: [local, cloud],
    } as never);
    const ui = (fileId: string) => (
      <NextIntlClientProvider locale="en" messages={{}}>
        <SummarySection fileId={fileId} drive="drive1" />
      </NextIntlClientProvider>
    );
    const { rerender } = render(ui("f1"));
    await rejectChoice();

    rerender(ui("f2"));
    await waitFor(() => expect(getSummary).toHaveBeenCalledWith("f2", "drive1"));
    expect(screen.queryByText(UNAVAILABLE)).toBeNull();
  });

  it("says nothing about the model for another 400", async () => {
    vi.mocked(regenerateSummary).mockRejectedValue(
      new LLMRequestError("API error: 400", 400, "insufficient_content"),
    );
    vi.mocked(fetchLLMChoices).mockResolvedValue({
      auto: "local",
      choices: [local, cloud],
    } as never);
    renderSection();
    fireEvent.click(await screen.findByRole("button", { name: "Regenerate with…" }));
    await act(async () => {
      fireEvent.click(screen.getByRole("menuitem", { name: /cloud — gpt-x/ }));
    });
    await act(async () => {});
    expect(screen.queryByText(/This model can't be used/)).toBeNull();
    expect(fetchLLMChoices).toHaveBeenCalledTimes(1);
  });
});
