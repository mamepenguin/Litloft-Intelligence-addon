/**
 * SPEC-ADDON-107: sending a question from the keyboard in the file page's Ask
 * tab. The tab is rendered through the addon's `slots.ts` entry, as the file
 * page does, and every route is a function.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { Suspense } from "react";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

import { confirmConversionThenEnter } from "@/__tests__/helpers/imeEnter";
import { COMPOSITION_GRACE_MS } from "@/lib/ime";

const ADDON_ROOT = [
  resolve(process.cwd(), "../addons/intelligence"),
  resolve(process.cwd(), "addons/intelligence"),
].find((candidate) => existsSync(resolve(candidate, "manifest.json")))!;

const MANIFEST = JSON.parse(
  readFileSync(resolve(ADDON_ROOT, "manifest.json"), "utf-8"),
) as { slots?: Record<string, { id: string }[]> };
const ENTRY = (MANIFEST.slots?.["inspector-tabs"] ?? [])[0];

vi.mock("next-intl", async () => {
  const { existsSync: exists, readFileSync: read } = await import("node:fs");
  const { resolve: join } = await import("node:path");
  const root = [
    join(process.cwd(), "../addons/intelligence"),
    join(process.cwd(), "addons/intelligence"),
  ].find((candidate) => exists(candidate));
  const messages = JSON.parse(read(join(root!, "frontend/messages/en.json"), "utf-8"));
  const lookup = (path: string): unknown =>
    path.split(".").reduce<unknown>(
      (tree, part) =>
        typeof tree === "object" && tree !== null
          ? (tree as Record<string, unknown>)[part]
          : undefined,
      messages,
    );
  const useTranslations = (namespace?: string) => {
    return (key: string, values?: Record<string, unknown>) => {
      const path = namespace === undefined ? key : `${namespace}.${key}`;
      const found = lookup(path);
      const raw = typeof found === "string" ? found : path;
      if (!values) return raw;
      return Object.entries(values).reduce(
        (text, [name, value]) => text.replace(`{${name}}`, String(value)),
        raw,
      );
    };
  };
  return {
    useTranslations,
    useLocale: () => "en",
    NextIntlClientProvider: ({ children }: { children: unknown }) => children,
  };
});

vi.mock("@/hooks/useWebSocket", () => ({
  useWebSocket: () => null,
}));

import { slotComponents } from "@/addons/intelligence/slots";

// ---------------------------------------------------------------------------
// The connection
// ---------------------------------------------------------------------------

interface Call {
  url: string;
  method: string;
  body: Record<string, unknown> | undefined;
}

type Reply = (call: Call) => Response | Promise<Response>;

const net: { calls: Call[]; choices: Reply; answer: Reply } = {
  calls: [],
  choices: () => json({ auto: "local", choices: [] }),
  answer: () => json({ detail: "not stubbed" }, 500),
};

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

function sse() {
  let controller!: ReadableStreamDefaultController<Uint8Array>;
  const body = new ReadableStream<Uint8Array>({
    start(next) {
      controller = next;
    },
  });
  const encoder = new TextEncoder();
  return {
    response: {
      ok: true,
      status: 200,
      statusText: "200",
      json: async () => ({}),
      text: async () => "",
      body,
    } as unknown as Response,
    push(frame: string) {
      controller.enqueue(encoder.encode(frame));
    },
    close() {
      controller.close();
    },
  };
}

function frame(event: string, data: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

const fetchMock = vi.fn(async (url: string, init: RequestInit = {}) => {
  const call: Call = {
    url: String(url),
    method: String(init.method ?? "GET").toUpperCase(),
    body: typeof init.body === "string" ? JSON.parse(init.body) : undefined,
  };
  net.calls.push(call);
  if (call.url.includes("/llm/choices/rag")) return net.choices(call);
  if (call.method === "POST") return net.answer(call);
  return json({ available: true });
});
vi.stubGlobal("fetch", fetchMock);

function questions(): Call[] {
  return net.calls.filter((call) => call.method === "POST" && call.url.includes("/ask"));
}

// ---------------------------------------------------------------------------
// The tab
// ---------------------------------------------------------------------------

const FILE = "key000000001";

function tabProps(overrides: Record<string, unknown> = {}) {
  return {
    fileId: FILE,
    drive: "family",
    filename: "lecture.mp4",
    fileType: "video",
    mimeType: "video/mp4",
    videoRef: { current: null },
    subtitleTracks: [],
    mediaController: null,
    documentCaptureController: null,
    documentController: null,
    subtitles: [],
    lowerInspector: () => {},
    trustTier: "verified",
    trustReviewedAt: null,
    onFileChange: () => {},
    fillHeight: true,
    labelledByHost: true,
    onAvailability: vi.fn(),
    ...overrides,
  };
}

function AskTab(props: ReturnType<typeof tabProps>) {
  const Component = slotComponents[ENTRY?.id as string];
  return (
    <Suspense fallback={null}>
      <Component {...props} />
    </Suspense>
  );
}

async function renderTab(overrides: Record<string, unknown> = {}) {
  const props = tabProps(overrides);
  render(<AskTab {...props} />);
  await waitFor(() => expect(props.onAvailability).toHaveBeenCalledWith(true));
  return props;
}

async function questionBox(): Promise<HTMLTextAreaElement> {
  return (await screen.findByPlaceholderText("Ask about this file")) as HTMLTextAreaElement;
}

function submitButton(): HTMLElement {
  return screen.getByRole("button", { name: "Ask" });
}

function pageClosed(): CustomEvent<{ fileId: string }> {
  return new CustomEvent("litloft:file-page-closed", { detail: { fileId: FILE } });
}

async function settle(): Promise<void> {
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 0));
  });
}

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

/**
 * jsdom performs no default action for a key. When the handler leaves the key
 * to the browser and `browserInserts` is set, this does what a browser does
 * for it in a textarea: replace the selection with a newline, within
 * `maxLength`.
 */
function pressNewlineKey(
  box: HTMLTextAreaElement,
  mods: Mods,
  { browserInserts }: { browserInserts: boolean },
): boolean {
  const notPrevented = press(box, mods);
  if (notPrevented && browserInserts) {
    const start = box.selectionStart;
    const end = box.selectionEnd;
    const next = box.value.slice(0, start) + "\n" + box.value.slice(end);
    if (box.maxLength < 0 || next.length <= box.maxLength) {
      fireEvent.change(box, { target: { value: next } });
      box.setSelectionRange(start + 1, start + 1);
    }
  }
  return notPrevented;
}

const SEND_KEYS: { name: string; mods: Mods }[] = [
  { name: "Enter", mods: {} },
  { name: "Ctrl+Enter", mods: { ctrlKey: true } },
  { name: "Cmd+Enter", mods: { metaKey: true } },
];

/**
 * Shift+Enter is inserted natively by every browser, so it may be left to the
 * browser. Alt/Option+Enter is inserted by some browsers and not by others,
 * so one newline in both holds only when the handler prevents the default and
 * inserts it itself.
 */
const NEWLINE_KEYS: { name: string; mods: Mods; mustHandle: boolean }[] = [
  { name: "Shift+Enter", mods: { shiftKey: true }, mustHandle: false },
  { name: "Alt+Enter", mods: { altKey: true }, mustHandle: true },
];

function pressNewline(box: HTMLTextAreaElement, key: (typeof NEWLINE_KEYS)[number]): void {
  const notPrevented = pressNewlineKey(box, key.mods, { browserInserts: !key.mustHandle });
  if (key.mustHandle) expect(notPrevented).toBe(false);
}

const OTHER_COMBINATIONS: { name: string; mods: Mods }[] = [
  { name: "Shift+Alt+Enter", mods: { shiftKey: true, altKey: true } },
  { name: "Cmd+Shift+Enter", mods: { metaKey: true, shiftKey: true } },
  { name: "Ctrl+Shift+Enter", mods: { ctrlKey: true, shiftKey: true } },
  { name: "Cmd+Alt+Enter", mods: { metaKey: true, altKey: true } },
  { name: "Ctrl+Alt+Enter", mods: { ctrlKey: true, altKey: true } },
];

beforeEach(() => {
  net.calls = [];
  net.choices = () => json({ auto: "local", choices: [] });
  net.answer = () => sse().response;
  localStorage.clear();
  sessionStorage.clear();
  fetchMock.mockClear();
  window.dispatchEvent(pageClosed());
});

afterEach(() => {
  window.dispatchEvent(pageClosed());
  vi.restoreAllMocks();
});

// ---------------------------------------------------------------------------
// I1
// ---------------------------------------------------------------------------

describe("SPEC-ADDON-107 I1 - a sending key starts the send the Ask button starts", () => {
  const TWO_PROFILES = {
    auto: "local",
    choices: [
      { name: "local", model: "llama3", offhost: false },
      { name: "cloud", model: "gpt-4o-mini", offhost: true },
    ],
  };

  async function sendWith(trigger: "button" | Mods) {
    net.choices = () => json(TWO_PROFILES);
    await renderTab({ mediaController: { seek: vi.fn(), getCurrentTime: () => 42 } });
    const select = (await screen.findByRole("combobox")) as HTMLSelectElement;
    fireEvent.change(select, { target: { value: "cloud" } });
    const box = await questionBox();
    typeInto(box, "who wrote this");

    let notPrevented = true;
    if (trigger === "button") fireEvent.click(submitButton());
    else notPrevented = press(box, trigger);

    await waitFor(() => expect(questions()).toHaveLength(1));
    await settle();
    const result = {
      sends: questions().length,
      body: questions()[0].body,
      boxAfter: box.value,
      notPrevented,
    };
    cleanup();
    window.dispatchEvent(pageClosed());
    net.calls = [];
    return result;
  }

  it.each(SEND_KEYS)(
    "$name sends one question with the same text, profile and player position as a click",
    async ({ mods }) => {
      const byButton = await sendWith("button");
      const byKey = await sendWith(mods);

      expect(byKey.sends).toBe(1);
      expect(byKey.body).toEqual(byButton.body);
      expect(byKey.body?.question).toBe("who wrote this");
      expect(byKey.body?.profile).toBe("cloud");
      expect(byKey.body?.position_seconds).toBe(42);
      expect(byKey.boxAfter).toBe(byButton.boxAfter);
      expect(byKey.notPrevented).toBe(false);
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
      await renderTab();
      const box = await questionBox();
      typeInto(box, "first line", [5, 5]);

      pressNewline(box, key);

      await waitFor(() => expect(box.value).toBe("first\n line"));
      await waitFor(() => expect([box.selectionStart, box.selectionEnd]).toEqual([6, 6]));
      await settle();
      expect(questions()).toHaveLength(0);

      press(box);
      await waitFor(() => expect(questions()).toHaveLength(1));
      expect(questions()[0].body?.question).toBe("first\n line");
    },
  );

  it.each(NEWLINE_KEYS)("$name replaces the selection with the newline", async (key) => {
    await renderTab();
    const box = await questionBox();
    typeInto(box, "abc def", [2, 5]);

    pressNewline(box, key);

    await waitFor(() => expect(box.value).toBe("ab\nef"));
    await waitFor(() => expect([box.selectionStart, box.selectionEnd]).toEqual([3, 3]));
    await settle();
    expect(questions()).toHaveLength(0);

    press(box);
    await waitFor(() => expect(questions()).toHaveLength(1));
    expect(questions()[0].body?.question).toBe("ab\nef");
  });

  it.each(NEWLINE_KEYS)(
    "$name inserts a newline into a question too short to send, which Enter sends once it is long enough",
    async (key) => {
      await renderTab();
      const box = await questionBox();
      typeInto(box, "ab");

      pressNewline(box, key);

      await waitFor(() => expect(box.value).toBe("ab\n"));
      await settle();
      expect(questions()).toHaveLength(0);

      typeInto(box, "ab\nc");
      press(box);
      await waitFor(() => expect(questions()).toHaveLength(1));
      expect(questions()[0].body?.question).toBe("ab\nc");
    },
  );

  it.each(NEWLINE_KEYS)(
    "$name inserts a newline while an answer streams, and Enter sends the text after it settles",
    async (key) => {
      const stream = sse();
      net.answer = () => stream.response;
      await renderTab();
      const box = await questionBox();
      typeInto(box, "first question");
      fireEvent.click(submitButton());
      await waitFor(() => expect(questions()).toHaveLength(1));
      typeInto(box, "next");

      pressNewline(box, key);

      await waitFor(() => expect(box.value).toBe("next\n"));
      await settle();
      expect(questions()).toHaveLength(1);

      await act(async () => {
        stream.push(frame("answer_chunk", { delta: "An answer." }));
        stream.push(frame("done", {}));
        stream.close();
      });
      typeInto(box, "next\nquestion");
      await waitFor(() => expect(submitButton()).toBeEnabled());
      press(box);
      await waitFor(() => expect(questions()).toHaveLength(2));
      expect(questions()[1].body?.question).toBe("next\nquestion");
    },
  );

  it.each(NEWLINE_KEYS)(
    "$name inserts nothing at the maximum length, and Enter still sends the full question",
    async (key) => {
      await renderTab();
      const box = await questionBox();
      const full = "a".repeat(box.maxLength);
      expect(full).toHaveLength(1000);
      typeInto(box, full);

      pressNewlineKey(box, key.mods, { browserInserts: !key.mustHandle });
      await settle();

      expect(box.value).toBe(full);
      expect(questions()).toHaveLength(0);

      press(box);
      await waitFor(() => expect(questions()).toHaveLength(1));
      expect(questions()[0].body?.question).toBe(full);
    },
  );
});

// ---------------------------------------------------------------------------
// I3
// ---------------------------------------------------------------------------

describe("SPEC-ADDON-107 I3 - an Enter the IME owns is left to the IME", () => {
  it("does not send on the Enter that confirms a conversion, and sends once on the next Enter", async () => {
    await renderTab();
    const box = await questionBox();

    const notPrevented = confirmConversionThenEnter(box, "計画の結論は");
    await settle();

    expect(notPrevented).toBe(true);
    expect(box.value).toBe("計画の結論は");
    expect(questions()).toHaveLength(0);

    press(box);
    await waitFor(() => expect(questions()).toHaveLength(1));
    await settle();
    expect(questions()).toHaveLength(1);
    expect(questions()[0].body?.question).toBe("計画の結論は");
  });

  it.each([
    { name: "Shift+Enter", mods: { shiftKey: true } },
    { name: "Alt+Enter", mods: { altKey: true } },
    { name: "Cmd+Enter", mods: { metaKey: true } },
  ])("leaves $name inside the grace window alone, and the next Enter sends", async ({ mods }) => {
    await renderTab();
    const box = await questionBox();
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
    expect(questions()).toHaveLength(0);

    press(box);
    await waitFor(() => expect(questions()).toHaveLength(1));
  });

  it.each([
    { name: "isComposing", init: { isComposing: true } },
    { name: "keyCode 229", init: { keyCode: 229 } },
  ])("leaves an Enter reporting $name alone, and Enter after the composition sends", async ({ init }) => {
    await renderTab();
    const box = await questionBox();
    fireEvent.compositionStart(box);
    fireEvent.change(box, { target: { value: "計画の結論は" } });

    const notPrevented = fireEvent.keyDown(box, { ...ENTER, ...init });
    await settle();

    expect(notPrevented).toBe(true);
    expect(box.value).toBe("計画の結論は");
    expect(questions()).toHaveLength(0);

    const now = vi.spyOn(Date, "now").mockReturnValue(1_000_000);
    fireEvent.compositionEnd(box, { data: "計画の結論は" });
    now.mockReturnValue(1_000_000 + COMPOSITION_GRACE_MS);
    press(box);
    now.mockRestore();
    await waitFor(() => expect(questions()).toHaveLength(1));
  });
});

// ---------------------------------------------------------------------------
// I4
// ---------------------------------------------------------------------------

describe("SPEC-ADDON-107 I4 - a sending key while the button is disabled does nothing", () => {
  it.each(SEND_KEYS)(
    "$name on a question too short to send inserts nothing, and sends nothing once it is long enough",
    async ({ mods }) => {
      await renderTab();
      const box = await questionBox();
      typeInto(box, " ab ");
      expect(submitButton()).toBeDisabled();

      const notPrevented = press(box, mods);
      await settle();

      expect(notPrevented).toBe(false);
      expect(box.value).toBe(" ab ");
      expect(questions()).toHaveLength(0);

      typeInto(box, " abc ");
      expect(submitButton()).toBeEnabled();
      await settle();
      expect(questions()).toHaveLength(0);
    },
  );

  it.each(SEND_KEYS)(
    "$name while an answer streams inserts nothing, keeps the text, and sends nothing when the answer settles",
    async ({ mods }) => {
      const stream = sse();
      net.answer = () => stream.response;
      await renderTab();
      const box = await questionBox();
      typeInto(box, "first question");
      fireEvent.click(submitButton());
      await waitFor(() => expect(questions()).toHaveLength(1));
      typeInto(box, "second question");

      const notPrevented = press(box, mods);
      await settle();

      expect(notPrevented).toBe(false);
      expect(box.value).toBe("second question");
      expect(questions()).toHaveLength(1);

      await act(async () => {
        stream.push(frame("answer_chunk", { delta: "An answer." }));
        stream.push(frame("done", {}));
        stream.close();
      });
      await waitFor(() => expect(submitButton()).toBeEnabled());
      await settle();
      expect(questions()).toHaveLength(1);
      expect(box.value).toBe("second question");
    },
  );
});

// ---------------------------------------------------------------------------
// I5
// ---------------------------------------------------------------------------

describe("SPEC-ADDON-107 I5 - other Enter combinations are left to the browser", () => {
  it.each(OTHER_COMBINATIONS)(
    "$name is not prevented and sends nothing, and Enter then sends",
    async ({ mods }) => {
      await renderTab();
      const box = await questionBox();
      typeInto(box, "who wrote this");

      const notPrevented = press(box, mods);
      await settle();

      expect(notPrevented).toBe(true);
      expect(box.value).toBe("who wrote this");
      expect(questions()).toHaveLength(0);

      press(box);
      await waitFor(() => expect(questions()).toHaveLength(1));
    },
  );
});
