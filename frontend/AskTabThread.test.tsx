/**
 * SPEC-ADDON-082 and SPEC-ADDON-083: the Ask tab keeps a thread of turns per
 * file, sends the answered ones as history, and saves the thread as a note.
 *
 * The tab is the addon's `slots.ts` component for its `inspector-tabs` entry,
 * and the save goes through the real dialog, so what is checked is the request
 * the browser would send to the knowledge addon.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { Suspense } from "react";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

const ADDON_ROOT = [
  resolve(process.cwd(), "../addons/intelligence"),
  resolve(process.cwd(), "addons/intelligence"),
].find((candidate) => existsSync(resolve(candidate, "manifest.json")))!;

const MANIFEST = JSON.parse(
  readFileSync(resolve(ADDON_ROOT, "manifest.json"), "utf-8"),
) as { slots?: Record<string, { id: string; i18n_key?: string }[]> };
const ENTRY = (MANIFEST.slots?.["inspector-tabs"] ?? [])[0];
const EN = JSON.parse(
  readFileSync(resolve(ADDON_ROOT, "frontend/messages/en.json"), "utf-8"),
);
const JA = JSON.parse(
  readFileSync(resolve(ADDON_ROOT, "frontend/messages/ja.json"), "utf-8"),
);

vi.mock("next-intl", async () => {
  const { existsSync: exists, readFileSync: read } = await import("node:fs");
  const { resolve: join } = await import("node:path");
  const root = [
    join(process.cwd(), "../addons/intelligence"),
    join(process.cwd(), "addons/intelligence"),
  ].find((candidate) => exists(candidate));
  const messages = JSON.parse(
    read(join(root!, "frontend/messages/en.json"), "utf-8"),
  );
  const lookup = (path: string): unknown =>
    path.split(".").reduce<unknown>(
      (tree, part) =>
        typeof tree === "object" && tree !== null
          ? (tree as Record<string, unknown>)[part]
          : undefined,
      messages,
    );
  const useTranslations = (namespace?: string) => {
    const t = (key: string, values?: Record<string, unknown>) => {
      const path = namespace === undefined ? key : `${namespace}.${key}`;
      const found = lookup(path);
      const raw = typeof found === "string" ? found : path;
      if (!values) return raw;
      return Object.entries(values).reduce(
        (text, [name, value]) => text.replace(`{${name}}`, String(value)),
        raw,
      );
    };
    return t;
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

// The save dialog's folder picker lists folders on mount; none are needed here.
vi.mock("@/lib/api", async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getFolders: vi.fn().mockResolvedValue([]),
  getFolderTree: vi.fn().mockResolvedValue([]),
}));

import { slotComponents } from "@/addons/intelligence/slots";
import { queryToFilename } from "@/addons/intelligence/askNoteFormat";
import { accentFills } from "@/__tests__/helpers/accentFills";

// ---------------------------------------------------------------------------
// The connection
// ---------------------------------------------------------------------------

interface Call {
  url: string;
  method: string;
  body: Record<string, unknown> | undefined;
  headers: Record<string, string>;
  signal: AbortSignal | undefined;
}

type Reply = (call: Call) => Response | Promise<Response>;

const net: { calls: Call[]; answer: Reply; save: Reply; knowledge: Reply } = {
  calls: [],
  knowledge: () => json({}),
  answer: () => json({ detail: "not stubbed" }, 500),
  save: () => json({ note_file_id: "note00000001", note_path: "Ask/x.md" }, 201),
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

type Stream = ReturnType<typeof sse>;

function frame(event: string, data: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

const fetchMock = vi.fn(async (url: string, init: RequestInit = {}) => {
  const call: Call = {
    url: String(url),
    method: String(init.method ?? "GET").toUpperCase(),
    body: typeof init.body === "string" ? JSON.parse(init.body) : undefined,
    headers: Object.fromEntries(new Headers(init.headers).entries()),
    signal: init.signal ?? undefined,
  };
  net.calls.push(call);
  if (call.url.includes("/llm/choices/rag")) return json({ auto: "local", choices: [] });
  if (call.url.includes("/addons/knowledge/notes") && call.method === "POST") {
    return net.save(call);
  }
  if (call.url.includes("/addons/knowledge")) return net.knowledge(call);
  if (call.method === "POST") return net.answer(call);
  return json({ available: true });
});
vi.stubGlobal("fetch", fetchMock);

function questions(): Call[] {
  return net.calls.filter(
    (call) => call.method === "POST" && call.url.includes("/intelligence/files/"),
  );
}

function saves(): Call[] {
  return net.calls.filter(
    (call) => call.method === "POST" && call.url.includes("/addons/knowledge/notes"),
  );
}

// ---------------------------------------------------------------------------
// The tab
// ---------------------------------------------------------------------------

const FILE = "vid000000001";
const OTHER_FILE = "vid000000002";
const NOTE_FILE = "note00000009";

function tabProps(overrides: Record<string, unknown> = {}) {
  return {
    fileId: FILE,
    drive: "family",
    filename: "lecture.mp4",
    fileType: "video",
    mimeType: "video/mp4",
    videoRef: { current: null },
    subtitleTracks: [],
    mediaController: { seek: vi.fn() },
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
  } as Record<string, unknown> & { onAvailability: ReturnType<typeof vi.fn> };
}

function AskTab(props: Record<string, unknown>) {
  const Component = slotComponents[ENTRY?.id as string];
  if (!Component) throw new Error("no inspector-tabs component");
  return (
    <Suspense fallback={null}>
      <Component {...props} />
    </Suspense>
  );
}

async function renderTab(overrides: Record<string, unknown> = {}) {
  const props = tabProps(overrides);
  const utils = render(<AskTab {...props} />);
  await waitFor(() => expect(props.onAvailability).toHaveBeenCalled());
  return { ...utils, props };
}

async function box(): Promise<HTMLTextAreaElement> {
  return (await screen.findByPlaceholderText("Ask about this file")) as HTMLTextAreaElement;
}

async function push(stream: Stream, text: string): Promise<void> {
  await act(async () => {
    stream.push(text);
  });
}

/** Send a question and leave its turn streaming. */
async function send(question: string): Promise<Stream> {
  const stream = sse();
  net.answer = () => stream.response;
  const before = questions().length;
  fireEvent.change(await box(), { target: { value: question } });
  fireEvent.click(await screen.findByRole("button", { name: "Ask" }));
  await waitFor(() => expect(questions()).toHaveLength(before + 1));
  return stream;
}

interface Answer {
  text: string;
  citations?: unknown[];
  meta?: Record<string, unknown>;
}

async function finish(stream: Stream, answer: Answer): Promise<void> {
  await push(stream, frame("answer_chunk", { delta: answer.text }));
  await push(stream, frame("citations", answer.citations ?? []));
  await push(stream, frame("meta", answer.meta ?? { partial: false }));
  await push(stream, frame("done", {}));
}

async function answered(question: string, answer: Answer): Promise<void> {
  const stream = await send(question);
  await finish(stream, answer);
}

async function failed(question: string, received = ""): Promise<void> {
  const stream = await send(question);
  if (received) await push(stream, frame("answer_chunk", { delta: received }));
  await push(stream, frame("done", { error: "Answer generation failed" }));
  await screen.findAllByText("Could not answer");
}

function textNode(text: string): Node {
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    // A controlled textarea keeps its value as a child text node.
    if (node.parentElement?.closest("textarea")) continue;
    if (node.textContent?.includes(text)) return node;
  }
  throw new Error(`no text node holds ${JSON.stringify(text)}`);
}

function inOrder(...nodes: Node[]): boolean {
  return nodes.every(
    (node, i) =>
      i === 0 ||
      (nodes[i - 1].compareDocumentPosition(node) & Node.DOCUMENT_POSITION_FOLLOWING) !== 0,
  );
}

function pageClosed(fileId: string): CustomEvent<{ fileId: string }> {
  return new CustomEvent("litloft:file-page-closed", { detail: { fileId } });
}

const scrollIntoView = vi.fn();

beforeEach(() => {
  net.calls = [];
  net.answer = () => json({ detail: "not stubbed" }, 500);
  net.save = () => json({ note_file_id: "note00000001", note_path: "Ask/x.md" }, 201);
  net.knowledge = () => json({});
  localStorage.clear();
  sessionStorage.clear();
  fetchMock.mockClear();
  scrollIntoView.mockClear();
  Element.prototype.scrollIntoView = scrollIntoView;
  for (const fileId of [FILE, OTHER_FILE, NOTE_FILE]) {
    window.dispatchEvent(pageClosed(fileId));
  }
});

afterEach(() => {
  vi.restoreAllMocks();
});

// ---------------------------------------------------------------------------
// History
// ---------------------------------------------------------------------------

describe("SPEC-ADDON-082 2 and I4 - each question sends the answered turns before it", () => {
  it("sends no history first, then the earlier turn as the question sent and the answer received", async () => {
    await renderTab();

    await answered("who wrote this", { text: "The author [1].", citations: [{ n: 1, position: { seconds: 5 } }] });
    await answered("and when", { text: "In 2020." });
    await send("and why");

    const [first, second, third] = questions();
    expect(first.body?.history ?? []).toEqual([]);
    expect(second.body?.history).toEqual([
      { question: "who wrote this", answer: "The author [1]." },
    ]);
    expect(third.body?.history).toEqual([
      { question: "who wrote this", answer: "The author [1]." },
      { question: "and when", answer: "In 2020." },
    ]);
  });

  it("never sends a failed turn", async () => {
    await renderTab();

    await answered("who wrote this", { text: "The author." });
    await failed("what broke", "Half an answer.");
    await send("and why");

    expect(questions()[2].body?.history).toEqual([
      { question: "who wrote this", answer: "The author." },
    ]);
    expect(document.body.textContent).toContain("what broke");
    expect(document.body.textContent).toContain("Half an answer.");
  });

  it("sends at most the last 10 answered turns, oldest first", async () => {
    await renderTab();

    for (let i = 1; i <= 11; i++) {
      await answered(`question ${i}`, { text: `answer ${i}` });
    }
    await send("question 12");

    expect(questions()[11].body?.history).toEqual(
      Array.from({ length: 10 }, (_, i) => ({
        question: `question ${i + 2}`,
        answer: `answer ${i + 2}`,
      })),
    );
  });

  it("cuts each answer it sends to its first 20,000 characters", async () => {
    await renderTab();
    const long = "abcdefghij".repeat(2_001);

    await answered("who wrote this", { text: long });
    await send("and why");

    expect(questions()[1].body?.history).toEqual([
      { question: "who wrote this", answer: long.slice(0, 20_000) },
    ]);
  });
});

// ---------------------------------------------------------------------------
// Drawing the thread
// ---------------------------------------------------------------------------

describe("SPEC-ADDON-082 2 and 3 - the thread is drawn oldest first above an emptied box", () => {
  it("empties the box on send and draws each turn above it", async () => {
    await renderTab();

    await answered("first question", { text: "First answer." });
    expect((await box()).value).toBe("");
    await answered("second question", { text: "Second answer." });

    expect(
      inOrder(
        textNode("first question"),
        textNode("First answer."),
        textNode("second question"),
        textNode("Second answer."),
        await box(),
      ),
    ).toBe(true);
  });

  it("scrolls the new question into view once, and not while its answer streams", async () => {
    await renderTab();

    await answered("first question", { text: "First answer." });
    const stream = await send("second question");
    await waitFor(() => expect(scrollIntoView).toHaveBeenCalledTimes(2));
    await push(stream, frame("answer_chunk", { delta: "Second " }));
    await push(stream, frame("answer_chunk", { delta: "answer." }));

    expect(scrollIntoView).toHaveBeenCalledTimes(2);
    expect(scrollIntoView).toHaveBeenLastCalledWith({ block: "nearest" });
    const scrolled = scrollIntoView.mock.contexts[1] as Element;
    expect(scrolled.textContent).toContain("second question");
    expect(scrolled.textContent).not.toContain("first question");
  });

  it("keeps the submit button the only accent fill with a thread on screen", async () => {
    const view = await renderTab();

    await answered("first question", { text: "First answer." });
    await screen.findByRole("button", { name: "New conversation" });

    expect(accentFills(view.container)).toHaveLength(1);
    expect(accentFills(view.container)[0]).toHaveAttribute("type", "submit");
  });
});

describe("SPEC-ADDON-082 4 and I6 - each turn keeps its own pills and notices", () => {
  it("moves the player to each turn's own cited position", async () => {
    const seek = vi.fn();
    await renderTab({ mediaController: { seek } });

    await answered("first question", { text: "See [1].", citations: [{ n: 1, position: { seconds: 10 } }] });
    await answered("second question", { text: "See [1].", citations: [{ n: 1, position: { seconds: 20 } }] });

    fireEvent.click(screen.getByRole("button", { name: "0:10" }));
    fireEvent.click(screen.getByRole("button", { name: "0:20" }));

    expect(seek.mock.calls).toEqual([[10], [20]]);
  });

  it("puts the partial and failure lines under the turn they belong to", async () => {
    await renderTab();

    await answered("first question", { text: "First answer.", meta: { partial: true } });
    await failed("second question");
    await answered("third question", { text: "Third answer." });

    expect(screen.getAllByText("Answered from part of the file")).toHaveLength(1);
    expect(screen.getAllByText("Could not answer")).toHaveLength(1);
    expect(
      inOrder(
        textNode("first question"),
        textNode("Answered from part of the file"),
        textNode("second question"),
        textNode("Could not answer"),
        textNode("third question"),
      ),
    ).toBe(true);
  });

  it("shows the image line only under an answered turn whose image was unavailable", async () => {
    await renderTab();
    const notice = EN.askTab.imageUnavailable as string;

    await answered("first question", { text: "First answer.", meta: { partial: false, image: "unavailable" } });
    await answered("second question", { text: "Second answer.", meta: { partial: false, image: "attached" } });

    expect(screen.getAllByText(notice)).toHaveLength(1);
    expect(inOrder(textNode("First answer."), textNode(notice), textNode("second question"))).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// New conversation
// ---------------------------------------------------------------------------

describe("SPEC-ADDON-082 5 and I5 - New conversation", () => {
  it("is offered only with a turn and none streaming", async () => {
    await renderTab();
    expect(screen.queryByRole("button", { name: "New conversation" })).toBeNull();

    const stream = await send("first question");
    await push(stream, frame("answer_chunk", { delta: "Half" }));
    expect(screen.queryByRole("button", { name: "New conversation" })).toBeNull();

    await push(stream, frame("done", {}));
    expect(await screen.findByRole("button", { name: "New conversation" })).toBeTruthy();
  });

  it("empties the thread and leaves the text box as it was", async () => {
    await renderTab();
    await answered("first question", { text: "First answer." });
    await answered("second question", { text: "Second answer." });
    fireEvent.change(await box(), { target: { value: "a draft" } });

    fireEvent.click(screen.getByRole("button", { name: "New conversation" }));

    await waitFor(() => expect(document.body.textContent).not.toContain("First answer."));
    expect(document.body.textContent).not.toContain("Second answer.");
    expect((await box()).value).toBe("a draft");
    expect(screen.queryByRole("button", { name: "New conversation" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Save as Markdown" })).toBeNull();
  });

  it("is offered after a failed turn alone, while saving is not", async () => {
    await renderTab();

    await failed("first question");

    expect(screen.getByRole("button", { name: "New conversation" })).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Save as Markdown" })).toBeNull();
  });

  it("starts a new conversation with no history", async () => {
    await renderTab();
    await answered("first question", { text: "First answer." });
    fireEvent.click(screen.getByRole("button", { name: "New conversation" }));

    await send("fresh question");

    expect(questions()[1].body?.history ?? []).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// Failures
// ---------------------------------------------------------------------------

describe("SPEC-ADDON-082 failure cases and I5 - a failed question goes back into an empty box", () => {
  it("puts the failed question back when the box is empty", async () => {
    await renderTab();
    await answered("first question", { text: "First answer." });

    await failed("second question");

    expect((await box()).value).toBe("second question");
    expect(inOrder(textNode("First answer."), textNode("second question"))).toBe(true);
  });

  it("leaves text typed while it streamed", async () => {
    await renderTab();
    const stream = await send("first question");
    expect((await box()).value).toBe("");
    fireEvent.change(await box(), { target: { value: "my next thought" } });

    await push(stream, frame("done", { error: "Answer generation failed" }));
    await screen.findByText("Could not answer");

    expect((await box()).value).toBe("my next thought");
  });
});

// ---------------------------------------------------------------------------
// How long the thread lives
// ---------------------------------------------------------------------------

describe("SPEC-ADDON-082 6 and SPEC-ADDON-056 I11 as revised - the thread lives with the page", () => {
  it("keeps every turn and the box's contents across a closed sheet, and stores nothing", async () => {
    const view = await renderTab();
    await answered("first question", { text: "First answer." });
    await answered("second question", { text: "Second answer." });
    fireEvent.change(await box(), { target: { value: "a draft" } });

    view.unmount();
    render(<AskTab {...view.props} />);

    await waitFor(() => expect(document.body.textContent).toContain("Second answer."));
    expect(document.body.textContent).toContain("First answer.");
    expect((await box()).value).toBe("a draft");
    expect(localStorage.length).toBe(0);
    expect(sessionStorage.length).toBe(0);
  });

  it("forgets the whole thread when the open file changes", async () => {
    const view = await renderTab();
    await answered("first question", { text: "First answer." });
    await answered("second question", { text: "Second answer." });
    expect(document.body.textContent).toContain("First answer.");

    view.rerender(<AskTab {...view.props} fileId={OTHER_FILE} />);
    view.rerender(<AskTab {...view.props} fileId={FILE} />);

    await waitFor(() => expect(document.body.textContent).not.toContain("Second answer."));
    expect(document.body.textContent).not.toContain("First answer.");
  });
});

// ---------------------------------------------------------------------------
// Saving
// ---------------------------------------------------------------------------

async function saveThread(): Promise<Call> {
  const before = saves().length;
  fireEvent.click(await screen.findByRole("button", { name: "Save as Markdown" }));
  fireEvent.click(await screen.findByRole("button", { name: "Save" }));
  await waitFor(() => expect(saves()).toHaveLength(before + 1));
  return saves()[before];
}

function content(call: Call): string {
  return String(call.body?.content ?? "");
}

const SAVED_AT = /^saved_at: \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$/m;

describe("SPEC-ADDON-083 1 and I5 - when saving is offered", () => {
  it("is hidden until a turn is answered and while one streams", async () => {
    await renderTab();
    expect(screen.queryByRole("button", { name: "Save as Markdown" })).toBeNull();

    await answered("first question", { text: "First answer." });
    expect(await screen.findByRole("button", { name: "Save as Markdown" })).toBeTruthy();

    const stream = await send("second question");
    await push(stream, frame("answer_chunk", { delta: "Half" }));
    expect(screen.queryByRole("button", { name: "Save as Markdown" })).toBeNull();
    expect(screen.queryByRole("button", { name: "New conversation" })).toBeNull();
  });

  it("is offered whether or not the knowledge addon answers", async () => {
    net.knowledge = () => json({ detail: "Not Found" }, 404);
    await renderTab();

    await answered("first question", { text: "First answer." });

    expect(await screen.findByRole("button", { name: "Save as Markdown" })).toBeTruthy();
  });
});

describe("SPEC-ADDON-083 2 and 3 and I7 - the saved note", () => {
  it("names the file from the first answered question and posts an Ask note for this file", async () => {
    await renderTab();
    await failed("broken question");
    await answered("who wrote this", { text: "The author." });
    await answered("and when", { text: "In 2020." });

    fireEvent.click(await screen.findByRole("button", { name: "Save as Markdown" }));
    await waitFor(() =>
      expect(
        document.querySelector<HTMLInputElement>("#file-save-dialog-filename")?.value,
      ).toBe(queryToFilename("who wrote this")),
    );
    fireEvent.click(await screen.findByRole("button", { name: "Save" }));
    await waitFor(() => expect(saves()).toHaveLength(1));

    const call = saves()[0];
    expect(call.body?.origin).toBe("ask_answer");
    expect(call.body?.source_file_ids).toEqual([FILE]);
    expect(call.body?.folder).toBe("Ask");
  });

  it("writes every answered turn in thread order under the frontmatter and the source line", async () => {
    await renderTab();
    await answered("who  wrote\nthis", {
      text: "Intro [1] and [2].",
      citations: [{ n: 1, position: { seconds: 1705 } }],
    });
    await failed("broken question", "Half an answer.");
    await answered("and then", {
      text: "Later [1].",
      citations: [{ n: 1, position: { seconds: 3723 } }],
    });

    const note = content(await saveThread());

    expect(note).toMatch(SAVED_AT);
    expect(note.replace(SAVED_AT, "saved_at: X").trimEnd()).toBe(
      [
        "---",
        "origin: ask_answer",
        `query: ${JSON.stringify("who  wrote\nthis")}`,
        `source_file_ids: ["${FILE}"]`,
        "saved_at: X",
        "---",
        "",
        "# who wrote this",
        "",
        `[lecture.mp4](loft://${FILE})`,
        "",
        "## who wrote this",
        "",
        `Intro [28:25](loft://${FILE}?t=1705) and .`,
        "",
        "## and then",
        "",
        `Later [1:02:03](loft://${FILE}?t=3723).`,
      ].join("\n"),
    );
    expect(note).not.toContain("broken question");
    expect(note).not.toContain("Half an answer.");
  });

  it.each([
    { name: "a time rounds down to a whole second", position: { seconds: 83.9 }, link: new RegExp(`\\[[^\\]]+\\]\\(loft://${FILE}\\?t=83\\)`) },
    { name: "a time under a minute", position: { seconds: 59 }, link: `[0:59](loft://${FILE}?t=59)` },
    { name: "a page", position: { page: 12 }, link: `[p.12](loft://${FILE}?page=12)` },
    {
      name: "a section title, escaped",
      position: { section: 3, title: "Part [A] \\ B" },
      link: `[Part \\[A\\] \\\\ B](loft://${FILE}?section=3)`,
    },
    { name: "an untitled section", position: { section: 4, title: null }, link: `[§4](loft://${FILE}?section=4)` },
  ])("rewrites a cited marker as a link: $name", async ({ position, link }) => {
    await renderTab();
    await answered("first question", { text: "See [1] now.", citations: [{ n: 1, position }] });

    const note = content(await saveThread());

    const line = note.split("\n").find((l) => l.startsWith("See "));
    expect(line).toBeDefined();
    if (typeof link === "string") {
      expect(line).toBe(`See ${link} now.`);
    } else {
      expect(line).toMatch(link);
      expect(line).not.toContain("[1]");
    }
  });

  it("removes a marker whose citation has no position", async () => {
    await renderTab();
    await answered("first question", { text: "See [1] now.", citations: [{ n: 1, position: null }] });

    const note = content(await saveThread());

    expect(note.split("\n")).toContain("See  now.");
  });

  it("writes code as received, markers included", async () => {
    await renderTab();
    const answer = [
      "Out [1] here `in [1]` and ``two [1] ticks``.",
      "",
      "```",
      "fence [1]",
      "```",
      "",
      "~~~",
      "tilde [1]",
      "~~~",
      "",
      "    indented [1]",
      "",
      "After [1].",
      "",
      "```",
      "open [1]",
    ].join("\n");
    await answered("first question", { text: answer, citations: [{ n: 1, position: { seconds: 5 } }] });

    const note = content(await saveThread());

    const link = `[0:05](loft://${FILE}?t=5)`;
    expect(note.trimEnd().endsWith(
      [
        `Out ${link} here \`in [1]\` and \`\`two [1] ticks\`\`.`,
        "",
        "```",
        "fence [1]",
        "```",
        "",
        "~~~",
        "tilde [1]",
        "~~~",
        "",
        "    indented [1]",
        "",
        `After ${link}.`,
        "",
        "```",
        "open [1]",
      ].join("\n"),
    )).toBe(true);
  });

  it("names a Markdown file by a wiki link", async () => {
    await renderTab({
      fileId: NOTE_FILE,
      filename: "Design Notes.md",
      fileType: "document",
      mimeType: "text/markdown",
      mediaController: null,
    });
    await answered("first question", { text: "An answer." });

    const note = content(await saveThread());

    expect(note.split("\n")).toContain("[[Design Notes]]");
    expect(note).toContain(`source_file_ids: ["${NOTE_FILE}"]`);
  });
});

describe("SPEC-ADDON-083 4 and I5 - after a save", () => {
  it("shows the note link until the thread holds more answered turns, then saves them all again", async () => {
    const view = await renderTab();
    await answered("first question", { text: "First answer." });
    await saveThread();

    expect(await screen.findByRole("link", { name: /Open note/ })).toBeTruthy();
    expect(document.body.textContent).toContain("Markdown saved");
    expect(screen.queryByRole("button", { name: "Save as Markdown" })).toBeNull();

    view.unmount();
    render(<AskTab {...view.props} />);
    expect(await screen.findByRole("link", { name: /Open note/ })).toBeTruthy();

    const streaming = await send("second question");
    await push(streaming, frame("answer_chunk", { delta: "Half" }));
    expect(screen.getByRole("link", { name: /Open note/ })).toBeTruthy();
    await push(streaming, frame("done", { error: "Answer generation failed" }));
    await screen.findByText("Could not answer");
    expect(screen.getByRole("link", { name: /Open note/ })).toBeTruthy();

    await answered("third question", { text: "Third answer." });
    expect(screen.queryByRole("link", { name: /Open note/ })).toBeNull();

    const second = content(await saveThread());
    expect(second.split("\n")).toContain("## first question");
    expect(second.split("\n")).toContain("## third question");
    expect(second).not.toContain("## second question");
    expect(saves()).toHaveLength(2);
  });

  it("New conversation drops the saved note's link", async () => {
    await renderTab();
    await answered("first question", { text: "First answer." });
    await saveThread();
    await screen.findByRole("link", { name: /Open note/ });

    fireEvent.click(screen.getByRole("button", { name: "New conversation" }));
    await answered("fresh question", { text: "Fresh answer." });

    expect(screen.queryByRole("link", { name: /Open note/ })).toBeNull();
    expect(screen.getByRole("button", { name: "Save as Markdown" })).toBeTruthy();
  });
});

describe("SPEC-ADDON-083 failure cases - a failed save", () => {
  it.each([
    { status: 502, detail: "The note was saved, but it could not be marked unverified" },
    { status: 404, detail: "Addon not found" },
  ])("shows the knowledge route's $status error in the dialog and leaves the thread", async ({ status, detail }) => {
    net.save = () => json({ detail }, status);
    await renderTab();
    await answered("first question", { text: "First answer." });

    await saveThread();

    await waitFor(() => expect(document.body.textContent).toContain(detail));
    expect(document.body.textContent).toContain("First answer.");
    expect(screen.queryByRole("link", { name: /Open note/ })).toBeNull();
  });
});

describe("SPEC-ADDON-082 and 083 - the strings a viewer reads", () => {
  const flatten = (tree: unknown): string[] => {
    if (typeof tree === "string") return [tree];
    if (typeof tree !== "object" || tree === null) return [];
    return Object.values(tree as Record<string, unknown>).flatMap(flatten);
  };

  it.each([
    { catalogue: EN, text: "New conversation" },
    { catalogue: JA, text: "新しい会話" },
  ])("holds $text", ({ catalogue, text }) => {
    expect(flatten(catalogue).includes(text)).toBe(true);
  });
});
