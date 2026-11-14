/**
 * SPEC-ADDON-107: sending a question from the keyboard on the Ask page.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";

import { COMPOSITION_GRACE_MS } from "@/lib/ime";

vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams(""),
  usePathname: () => "/drive/family/addons/intelligence",
}));

vi.mock("@/components/CurrentDriveProvider", () => ({
  useCurrentDrive: () => "family",
}));

// The save dialog's folder picker lists folders on mount; none are needed here.
vi.mock("@/lib/api", async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getDrives: vi.fn(async () => [{ name: "family", file_count: 619 }]),
  getFolders: vi.fn().mockResolvedValue([]),
  getFolderTree: vi.fn().mockResolvedValue([]),
}));

type QueueEntry = { done: false; value: unknown } | { done: true };

function makeController() {
  const queue: QueueEntry[] = [];
  const waiters: Array<(entry: QueueEntry) => void> = [];
  const deliver = (entry: QueueEntry) => {
    const waiter = waiters.shift();
    if (waiter) waiter(entry);
    else queue.push(entry);
  };
  async function* generator() {
    while (true) {
      const entry = await new Promise<QueueEntry>((resolve) => {
        const head = queue.shift();
        if (head) resolve(head);
        else waiters.push(resolve);
      });
      if (entry.done) return;
      yield entry.value;
    }
  }
  return {
    push: (value: unknown) => deliver({ done: false, value }),
    end: () => deliver({ done: true }),
    generator,
  };
}

const streamState = { current: makeController() };

const READY_STATUS = {
  status: "ok",
  features: { indexing: true, search: true, auto_tags: "false", summaries: "false", rag: true },
  llm: { provider: "ollama", model: "qwen", enabled: true, output_language: "auto" },
};

vi.mock("@/addons/intelligence/api", async () => {
  const actual = await vi.importActual<typeof import("@/addons/intelligence/api")>(
    "@/addons/intelligence/api",
  );
  return {
    ...actual,
    askQuestionStream: vi.fn(() => streamState.current.generator()),
    fetchLLMChoices: vi.fn(async () => {
      throw new Error("not stubbed");
    }),
    getIntelligenceStatus: vi.fn(),
  };
});

import IntelligenceAskPage from "@/addons/intelligence/Page";
import {
  askQuestionStream,
  fetchLLMChoices,
  getIntelligenceStatus,
} from "@/addons/intelligence/api";

function json(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: String(status),
    json: async () => body,
    text: async () => JSON.stringify(body),
    body: null,
  } as unknown as Response;
}

const saves: string[] = [];

const fetchMock = vi.fn(async (input: RequestInfo | URL, init: RequestInit = {}) => {
  const url = String(input);
  if (url.includes("/addons/knowledge/notes") && String(init.method).toUpperCase() === "POST") {
    saves.push(url);
    return json({ note_file_id: "note00000001", note_path: "Ask/x.md" }, 201);
  }
  if (url.includes("/files")) return json({ files: [], meta: { total: 0 } });
  return json([]);
});
vi.stubGlobal("fetch", fetchMock);

// ---------------------------------------------------------------------------
// Keys
// ---------------------------------------------------------------------------

type Mods = { shiftKey?: boolean; altKey?: boolean; ctrlKey?: boolean; metaKey?: boolean };

const ENTER = { key: "Enter", code: "Enter", keyCode: 13 };

/** Returns false when the handler prevented the browser's default. */
function press(box: HTMLElement, mods: Mods = {}): boolean {
  return fireEvent.keyDown(box, { ...ENTER, ...mods });
}

function typeInto(box: HTMLTextAreaElement, text: string, selection?: [number, number]): void {
  fireEvent.change(box, { target: { value: text } });
  const [start, end] = selection ?? [text.length, text.length];
  box.setSelectionRange(start, end);
}

const SEND_KEYS: { name: string; mods: Mods }[] = [
  { name: "Enter", mods: {} },
  { name: "Ctrl+Enter", mods: { ctrlKey: true } },
  { name: "Cmd+Enter", mods: { metaKey: true } },
];

/**
 * Alt/Option+Enter is inserted by some browsers and not by others, so one
 * newline in both holds only when the handler prevents the default and inserts
 * it itself.
 */
const NEWLINE_KEYS: { name: string; mods: Mods }[] = [
  { name: "Alt+Enter", mods: { altKey: true } },
];

function pressNewline(box: HTMLTextAreaElement, key: (typeof NEWLINE_KEYS)[number]): void {
  expect(press(box, key.mods)).toBe(false);
}

const OTHER_COMBINATIONS: { name: string; mods: Mods }[] = [
  { name: "Shift+Alt+Enter", mods: { shiftKey: true, altKey: true } },
  { name: "Cmd+Shift+Enter", mods: { metaKey: true, shiftKey: true } },
  { name: "Ctrl+Shift+Enter", mods: { ctrlKey: true, shiftKey: true } },
  { name: "Cmd+Alt+Enter", mods: { metaKey: true, altKey: true } },
  { name: "Ctrl+Alt+Enter", mods: { ctrlKey: true, altKey: true } },
];

const asked = () => vi.mocked(askQuestionStream);

async function questionBox(): Promise<HTMLTextAreaElement> {
  return (await screen.findByRole("textbox", { name: /question input/i })) as HTMLTextAreaElement;
}

function submitButton(): HTMLButtonElement {
  return screen.getByTestId("ask-submit") as HTMLButtonElement;
}

async function renderReady(): Promise<HTMLTextAreaElement> {
  render(<IntelligenceAskPage />);
  const box = await questionBox();
  await waitFor(() => expect(box.disabled).toBe(false));
  return box;
}

async function settle(): Promise<void> {
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 0));
  });
}

async function finishAnswer(withCitation = false): Promise<void> {
  await act(async () => {
    if (withCitation) {
      streamState.current.push({
        kind: "citation",
        index: 0,
        citation: {
          file_id: "file-a",
          drive: "family",
          filename: "a.md",
          file_type: "document",
          quote: "x",
          relevance: 0.9,
          segment_location: null,
        },
      });
    }
    streamState.current.push({ kind: "answer_chunk", delta: "an answer" });
    streamState.current.push({ kind: "done" });
    streamState.current.end();
  });
  streamState.current = makeController();
}

beforeEach(() => {
  streamState.current = makeController();
  saves.length = 0;
  sessionStorage.clear();
  localStorage.clear();
  asked().mockClear();
  vi.mocked(fetchLLMChoices).mockReset();
  vi.mocked(fetchLLMChoices).mockRejectedValue(new Error("not stubbed"));
  vi.mocked(getIntelligenceStatus).mockReset();
  vi.mocked(getIntelligenceStatus).mockResolvedValue(READY_STATUS as never);
});

afterEach(() => {
  streamState.current.end();
});

// ---------------------------------------------------------------------------
// I1
// ---------------------------------------------------------------------------

describe("SPEC-ADDON-107 I1 - a sending key starts the send the Ask button starts", () => {
  it.each(SEND_KEYS)(
    "$name sends once with the chosen profile and drops the previous answer's saved-note link",
    async ({ mods }) => {
      vi.mocked(fetchLLMChoices).mockResolvedValue({
        auto: "local",
        choices: [
          { name: "local", model: "gemma3", offhost: false },
          { name: "cloud", model: "gpt-x", offhost: true },
        ],
      } as never);
      const box = await renderReady();
      const select = (await screen.findByTestId("ask-profile")) as HTMLSelectElement;
      fireEvent.change(select, { target: { value: "cloud" } });

      typeInto(box, "what is the plot?");
      fireEvent.click(submitButton());
      await waitFor(() => expect(asked()).toHaveBeenCalledTimes(1));
      await finishAnswer(true);
      fireEvent.click(await screen.findByTestId("ask-save-note"));
      fireEvent.click(await screen.findByRole("button", { name: "Save" }));
      await waitFor(() => expect(saves).toHaveLength(1));
      await screen.findByRole("link", { name: /Open note/ });

      typeInto(box, "who is the hero?");
      const notPrevented = press(box, mods);

      await waitFor(() => expect(asked()).toHaveBeenCalledTimes(2));
      await settle();
      expect(asked()).toHaveBeenCalledTimes(2);
      expect(asked().mock.calls[1][0]).toBe("who is the hero?");
      expect(asked().mock.calls[1][2]?.profile).toBe("cloud");
      expect(notPrevented).toBe(false);

      await finishAnswer(true);
      await waitFor(() => expect(screen.queryByRole("link", { name: /Open note/ })).toBeNull());
      expect(await screen.findByTestId("ask-save-note")).toBeInTheDocument();
    },
  );
});

// ---------------------------------------------------------------------------
// I2
// ---------------------------------------------------------------------------

describe("SPEC-ADDON-107 I2 and failure cases - Shift+Enter and Alt/Option+Enter insert one newline", () => {
  it.each(NEWLINE_KEYS)(
    "$name puts one newline at the caret, sends nothing, and the next Enter sends both lines",
    async (key) => {
      const box = await renderReady();
      typeInto(box, "first line", [5, 5]);

      pressNewline(box, key);

      await waitFor(() => expect(box.value).toBe("first\n line"));
      await waitFor(() => expect([box.selectionStart, box.selectionEnd]).toEqual([6, 6]));
      await settle();
      expect(asked()).not.toHaveBeenCalled();

      typeInto(box, "first\nsecond line");
      press(box);
      await waitFor(() => expect(asked()).toHaveBeenCalledTimes(1));
      expect(asked().mock.calls[0][0]).toBe("first\nsecond line");
    },
  );

  it.each(NEWLINE_KEYS)("$name replaces the selection with the newline", async (key) => {
    const box = await renderReady();
    typeInto(box, "abc def", [2, 5]);

    pressNewline(box, key);

    await waitFor(() => expect(box.value).toBe("ab\nef"));
    await waitFor(() => expect([box.selectionStart, box.selectionEnd]).toEqual([3, 3]));
    await settle();
    expect(asked()).not.toHaveBeenCalled();
  });

  it.each(NEWLINE_KEYS)(
    "$name inserts a newline while Ask availability is being checked",
    async (key) => {
      vi.mocked(getIntelligenceStatus).mockReturnValue(new Promise(() => {}));
      render(<IntelligenceAskPage />);
      const box = await questionBox();
      expect(box.disabled).toBe(false);
      typeInto(box, "what is the plot?");

      pressNewline(box, key);

      await waitFor(() => expect(box.value).toBe("what is the plot?\n"));
      await settle();
      expect(asked()).not.toHaveBeenCalled();
    },
  );
});

// ---------------------------------------------------------------------------
// I3
// ---------------------------------------------------------------------------

describe("SPEC-ADDON-107 I3 - an Enter the IME owns is left to the IME", () => {
  it.each([
    { name: "Enter", mods: {} },
    { name: "Shift+Enter", mods: { shiftKey: true } },
    { name: "Alt+Enter", mods: { altKey: true } },
    { name: "Cmd+Enter", mods: { metaKey: true } },
  ])("leaves $name inside the grace window alone", async ({ mods }) => {
    const box = await renderReady();
    const now = vi.spyOn(Date, "now").mockReturnValue(1_000_000);
    fireEvent.compositionStart(box);
    fireEvent.change(box, { target: { value: "計画の結論は" } });
    fireEvent.compositionEnd(box, { data: "計画の結論は" });
    now.mockReturnValue(1_000_000 + COMPOSITION_GRACE_MS - 1);

    const notPrevented = press(box, mods);
    now.mockRestore();
    await settle();

    expect(notPrevented).toBe(true);
    expect(box.value).toBe("計画の結論は");
    expect(asked()).not.toHaveBeenCalled();
  });
});

// ---------------------------------------------------------------------------
// I4
// ---------------------------------------------------------------------------

describe("SPEC-ADDON-107 I4 - a sending key while the button is disabled does nothing", () => {
  it.each(SEND_KEYS)(
    "$name on a blank question inserts nothing, and sends nothing once it has text",
    async ({ mods }) => {
      const box = await renderReady();
      typeInto(box, "   ");
      expect(submitButton()).toBeDisabled();

      const notPrevented = press(box, mods);
      await settle();

      expect(notPrevented).toBe(false);
      expect(box.value).toBe("   ");
      expect(asked()).not.toHaveBeenCalled();

      typeInto(box, "what is the plot?");
      expect(submitButton()).toBeEnabled();
      await settle();
      expect(asked()).not.toHaveBeenCalled();
    },
  );

  it.each(SEND_KEYS)(
    "$name while Ask availability is being checked inserts nothing, and sends nothing once it is available",
    async ({ mods }) => {
      let answerStatus!: (value: unknown) => void;
      vi.mocked(getIntelligenceStatus).mockReturnValue(
        new Promise((resolve) => {
          answerStatus = resolve;
        }) as never,
      );
      render(<IntelligenceAskPage />);
      const box = await questionBox();
      typeInto(box, "what is the plot?");

      const notPrevented = press(box, mods);
      await settle();

      expect(notPrevented).toBe(false);
      expect(box.value).toBe("what is the plot?");
      expect(asked()).not.toHaveBeenCalled();

      await act(async () => {
        answerStatus(READY_STATUS);
      });
      await waitFor(() => expect(submitButton()).toBeEnabled());
      await settle();
      expect(asked()).not.toHaveBeenCalled();
    },
  );

  it.each(SEND_KEYS)(
    "$name while an answer streams inserts nothing and sends nothing when the answer settles",
    async ({ mods }) => {
      const box = await renderReady();
      typeInto(box, "what is the plot?");
      fireEvent.click(submitButton());
      await waitFor(() => expect(asked()).toHaveBeenCalledTimes(1));
      typeInto(box, "who is the hero?");

      const notPrevented = press(box, mods);
      await settle();

      expect(notPrevented).toBe(false);
      expect(asked()).toHaveBeenCalledTimes(1);

      await finishAnswer();
      await settle();
      expect(asked()).toHaveBeenCalledTimes(1);
    },
  );
});

// ---------------------------------------------------------------------------
// I5
// ---------------------------------------------------------------------------

describe("SPEC-ADDON-107 I5 - other Enter combinations are left to the browser", () => {
  it.each(OTHER_COMBINATIONS)("$name is not prevented and sends nothing", async ({ mods }) => {
    const box = await renderReady();
    typeInto(box, "what is the plot?");

    const notPrevented = press(box, mods);
    await settle();

    expect(notPrevented).toBe(true);
    expect(box.value).toBe("what is the plot?");
    expect(asked()).not.toHaveBeenCalled();
  });
});
