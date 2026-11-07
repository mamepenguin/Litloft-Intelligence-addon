/**
 * SPEC-ADDON-071, -072 and -073: the tab side of pictures, the playing
 * position and Markdown answers. The tab is rendered through the addon's
 * `slots.ts` entry, as the file page does, and every route is a function.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
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
) as MessageTree;
const JA = JSON.parse(
  readFileSync(resolve(ADDON_ROOT, "frontend/messages/ja.json"), "utf-8"),
) as MessageTree;

type MessageTree = { [key: string]: string | MessageTree };

const lookup = (tree: MessageTree, path: string): unknown =>
  path.split(".").reduce<unknown>(
    (node, part) =>
      typeof node === "object" && node !== null
        ? (node as Record<string, unknown>)[part]
        : undefined,
    tree,
  );

/** The addon's own catalogue stands in for the merged one. */
vi.mock("next-intl", async () => {
  const { existsSync: exists, readFileSync: read } = await import("node:fs");
  const { resolve: join } = await import("node:path");
  const root = [
    join(process.cwd(), "../addons/intelligence"),
    join(process.cwd(), "addons/intelligence"),
  ].find((candidate) => exists(candidate));
  const messages = JSON.parse(
    read(resolve(root!, "frontend/messages/en.json"), "utf-8"),
  ) as MessageTree;
  const lookupIn = (path: string): unknown =>
    path.split(".").reduce<unknown>(
      (node, part) =>
        typeof node === "object" && node !== null
          ? (node as Record<string, unknown>)[part]
          : undefined,
      messages,
    );
  const useTranslations = (namespace?: string) => {
    const t = (key: string, values?: Record<string, unknown>) => {
      const path = namespace === undefined ? key : `${namespace}.${key}`;
      const found = lookupIn(path);
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

const ws = vi.hoisted(() => ({ event: null as Record<string, unknown> | null }));

vi.mock("@/hooks/useWebSocket", () => ({
  useWebSocket: (filter?: string) =>
    filter && ws.event?.event !== filter ? null : ws.event,
}));

import { slotComponents } from "@/addons/intelligence/slots";
import { MarkdownPreview } from "@/components/MarkdownPreview";
import type { MediaController } from "@/lib/mediaController";
import type { DocumentController } from "@/lib/documentController";
import type { FileCitation } from "./askStream";

interface Call {
  url: string;
  method: string;
  body: Record<string, unknown> | undefined;
  headers: Record<string, string>;
  signal: AbortSignal | undefined;
}

type Reply = (call: Call) => Response | Promise<Response>;

interface Network {
  calls: Call[];
  availability: Reply;
  choices: Reply;
  answer: Reply;
}

const net: Network = {
  calls: [],
  availability: () => json({ available: true }),
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

function sseFrom(frames: string[]): Response {
  const stream = sse();
  for (const text of frames) stream.push(text);
  stream.close();
  return stream.response;
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
  if (call.url.includes("/llm/choices/rag")) return net.choices(call);
  if (call.method === "POST") return net.answer(call);
  return net.availability(call);
});
vi.stubGlobal("fetch", fetchMock);

function callsTo(method: string, path: string): Call[] {
  return net.calls.filter((call) => call.method === method && call.url.includes(path));
}

type TabProps = Record<string, unknown> & {
  fileId: string;
  drive: string;
  mediaController: Partial<MediaController> | null;
  documentController: Partial<DocumentController> | null;
  lowerInspector: () => void;
  onAvailability: (available: boolean) => void;
};

function tabProps(overrides: Record<string, unknown> = {}): TabProps {
  return {
    fileId: "ask000000001",
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
  } as TabProps;
}

function AskTab(props: TabProps) {
  const Component = slotComponents[ENTRY?.id as string];
  if (!Component) {
    throw new Error(
      MANIFEST.slots?.["inspector-tabs"]?.length
        ? `slots.ts registers no component for the inspector-tabs entry "${ENTRY.id}"`
        : "manifest.json declares no inspector-tabs entry",
    );
  }
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

async function questionBox(): Promise<HTMLTextAreaElement> {
  return (await screen.findByPlaceholderText(
    "Ask about this file",
  )) as HTMLTextAreaElement;
}

async function submitButton(): Promise<HTMLElement> {
  return screen.findByRole("button", { name: "Ask" });
}

async function ask(text: string): Promise<void> {
  const before = callsTo("POST", "/ask").length;
  fireEvent.change(await questionBox(), { target: { value: text } });
  fireEvent.click(await submitButton());
  await waitFor(() => expect(callsTo("POST", "/ask")).toHaveLength(before + 1));
}

function pageClosed(fileId: string): CustomEvent<{ fileId: string }> {
  return new CustomEvent("litloft:file-page-closed", { detail: { fileId } });
}

/** The ids this file asks about; their store entries are module state. */
const ASKED_FILE_IDS = ["ask000000001", "ask000000002"];

async function push(
  stream: { push: (frame: string) => void },
  text: string,
): Promise<void> {
  await act(async () => {
    stream.push(text);
  });
}

interface AnswerOptions {
  props?: Record<string, unknown>;
  citations?: FileCitation[];
  image?: "attached" | "unavailable" | "none";
  error?: string;
}

async function answerWith(source: string, options: AnswerOptions = {}) {
  const stream = sse();
  net.answer = () => stream.response;
  const view = await renderTab(options.props ?? {});
  await ask("who wrote this");
  await push(stream, frame("answer_chunk", { delta: source }));
  if (options.citations !== undefined) {
    await push(stream, frame("citations", options.citations));
  }
  if (options.image !== undefined) {
    await push(stream, frame("meta", { partial: false, image: options.image }));
  }
  await push(stream, frame("done", options.error ? { error: options.error } : {}));
  return { view, stream };
}

const CITED_TIME: FileCitation[] = [{ n: 1, position: { seconds: 1705 } }];
const TIME_PILL = "28:25";

const IMAGE_NOTICE =
  "The image could not be used, so this answer is from the text only.";
const NEEDS_VISION =
  "This picture has no text, and the selected model cannot read images.";
const IMAGE_UNUSABLE =
  "This picture has no text and could not be read as an image.";
const GENERIC_FAILURE = "Could not answer";

beforeEach(() => {
  net.calls = [];
  net.availability = () => json({ available: true });
  net.choices = () => json({ auto: "local", choices: [] });
  net.answer = () => json({ detail: "not stubbed" }, 500);
  ws.event = null;
  localStorage.clear();
  sessionStorage.clear();
  fetchMock.mockClear();
  for (const fileId of ASKED_FILE_IDS) {
    window.dispatchEvent(pageClosed(fileId));
  }
});

afterEach(() => {
  for (const fileId of ASKED_FILE_IDS) {
    window.dispatchEvent(pageClosed(fileId));
  }
  vi.restoreAllMocks();
});

// ---------------------------------------------------------------------------
// SPEC-ADDON-072: the position travels with the question
// ---------------------------------------------------------------------------

describe("SPEC-ADDON-072 step 2 - the position at submit time", () => {
  it.each([
    {
      name: "a playing controller gives its current time",
      mediaController: () => ({ seek: vi.fn(), getCurrentTime: () => 42 }),
      expected: 42,
    },
    {
      name: "time zero is a position",
      mediaController: () => ({ seek: vi.fn(), getCurrentTime: () => 0 }),
      expected: 0,
    },
    {
      name: "an interrupted player sends no position",
      mediaController: () => ({
        seek: vi.fn(),
        getCurrentTime: () => 42,
        isInterrupted: () => true,
      }),
      expected: undefined,
    },
    {
      name: "a non-interrupted player sends its position",
      mediaController: () => ({
        seek: vi.fn(),
        getCurrentTime: () => 42,
        isInterrupted: () => false,
      }),
      expected: 42,
    },
    {
      name: "a non-finite time sends no position",
      mediaController: () => ({ seek: vi.fn(), getCurrentTime: () => NaN }),
      expected: undefined,
    },
    {
      name: "an infinite time sends no position",
      mediaController: () => ({
        seek: vi.fn(),
        getCurrentTime: () => Infinity,
      }),
      expected: undefined,
    },
    {
      name: "a negative time sends no position",
      mediaController: () => ({ seek: vi.fn(), getCurrentTime: () => -1 }),
      expected: undefined,
    },
    {
      name: "no controller sends no position",
      mediaController: () => null,
      expected: undefined,
    },
  ])("$name", async ({ mediaController, expected }) => {
    const stream = sse();
    net.answer = () => stream.response;
    await renderTab({ mediaController: mediaController() });

    await ask("who wrote this");

    const body = callsTo("POST", "/ask")[0].body!;
    if (expected === undefined) {
      expect("position_seconds" in body).toBe(false);
    } else {
      expect(body.position_seconds).toBe(expected);
    }
  });

  it("snapshots the position at submit rather than following the player", async () => {
    const stream = sse();
    net.answer = () => stream.response;
    const getCurrentTime = vi.fn(() => 42);
    await renderTab({ mediaController: { seek: vi.fn(), getCurrentTime } });

    await ask("who wrote this");
    const readsAtSubmit = getCurrentTime.mock.calls.length;
    getCurrentTime.mockReturnValue(555);
    await push(stream, frame("answer_chunk", { delta: "An answer." }));
    await push(stream, frame("done", {}));

    expect(callsTo("POST", "/ask")[0].body?.position_seconds).toBe(42);
    expect(getCurrentTime.mock.calls.length).toBe(readsAtSubmit);
    expect(document.body.textContent).not.toContain("0:42");
  });
});

// ---------------------------------------------------------------------------
// SPEC-ADDON-071: the image notice and the two picture failure lines
// ---------------------------------------------------------------------------

describe("SPEC-ADDON-071 I3 - the image notice follows meta.image", () => {
  it.each([
    {
      name: "unavailable",
      image: "unavailable" as const,
      shown: true,
      expectation: "shows the notice",
    },
    {
      name: "attached",
      image: "attached" as const,
      shown: false,
      expectation: "shows no notice",
    },
    {
      name: "none",
      image: "none" as const,
      shown: false,
      expectation: "shows no notice",
    },
    {
      name: "absent",
      image: undefined,
      shown: false,
      expectation: "shows no notice",
    },
  ])("meta.image $name $expectation", async ({ image, shown }) => {
    await answerWith("An answer.", {
      props: { fileType: "image" },
      image,
    });

    if (shown) {
      expect(await screen.findByText(IMAGE_NOTICE)).toBeInTheDocument();
    } else {
      expect(screen.queryByText(IMAGE_NOTICE)).toBeNull();
    }
  });

  it("never shows the notice on a failed entry, even after an unavailable meta", async () => {
    await answerWith("An answer.", {
      props: { fileType: "image" },
      image: "unavailable",
      error: "image_unusable",
    });

    expect(await screen.findByText(IMAGE_UNUSABLE)).toBeInTheDocument();
    expect(screen.queryByText(IMAGE_NOTICE)).toBeNull();
  });

  it("clears the notice when a new question begins", async () => {
    await answerWith("An answer.", {
      props: { fileType: "image" },
      image: "unavailable",
    });
    expect(await screen.findByText(IMAGE_NOTICE)).toBeInTheDocument();

    net.answer = () => json({ detail: "picture_needs_vision" }, 409);
    await ask("what is on the receipt");

    expect(screen.queryByText(IMAGE_NOTICE)).toBeNull();
    expect(await screen.findByText(NEEDS_VISION)).toBeInTheDocument();
    expect(screen.queryByText(IMAGE_NOTICE)).toBeNull();
  });
});

describe("SPEC-ADDON-071 I6 - a picture with no text gets its own failure line", () => {
  it.each([
    {
      name: "a 409 picture_needs_vision",
      reply: () => json({ detail: "picture_needs_vision" }, 409),
      shown: NEEDS_VISION,
      hidden: [IMAGE_UNUSABLE, GENERIC_FAILURE],
    },
    {
      name: "a done carrying image_unusable",
      reply: () => sseFrom([frame("done", { error: "image_unusable" })]),
      shown: IMAGE_UNUSABLE,
      hidden: [NEEDS_VISION, GENERIC_FAILURE],
    },
    {
      name: "any other failure",
      reply: () => json({ detail: "boom" }, 500),
      shown: GENERIC_FAILURE,
      hidden: [NEEDS_VISION, IMAGE_UNUSABLE],
    },
  ])("shows only its line for $name", async ({ reply, shown, hidden }) => {
    net.answer = reply as Reply;
    await renderTab({ fileType: "image" });

    await ask("what is on the receipt");

    expect(await screen.findByText(shown)).toBeInTheDocument();
    for (const text of hidden) {
      expect(screen.queryByText(text)).toBeNull();
    }
  });

  it("keeps a failed entry's text, drawn as Markdown, above the failure line", async () => {
    await answerWith("**Bold** first.", {
      props: { fileType: "image" },
      error: "image_unusable",
    });

    expect(screen.getByText("Bold").closest("strong")).not.toBeNull();
    expect(await screen.findByText(IMAGE_UNUSABLE)).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// SPEC-ADDON-073: the answer is Markdown
// ---------------------------------------------------------------------------

describe("SPEC-ADDON-073 I10 - the answer is drawn as Markdown", () => {
  it.each([
    {
      name: "a heading",
      source: "# Overview",
      check: () => {
        expect(
          screen.getByRole("heading", { level: 1, name: "Overview" }),
        ).toBeInTheDocument();
      },
    },
    {
      name: "a bullet list",
      source: "- one\n- two\n- three",
      check: () => {
        expect(screen.getAllByRole("listitem")).toHaveLength(3);
      },
    },
    {
      name: "a table",
      source: "| A | B |\n| --- | --- |\n| 1 | 2 |",
      check: () => {
        expect(screen.getByRole("table")).toBeInTheDocument();
        expect(screen.getByRole("cell", { name: "1" })).toBeInTheDocument();
        expect(screen.getByRole("cell", { name: "2" })).toBeInTheDocument();
      },
    },
    {
      name: "bold text",
      source: "This is **Bold** text.",
      check: () => {
        expect(screen.getByText("Bold").closest("strong")).not.toBeNull();
      },
    },
  ])("draws $name as its element", async ({ source, check }) => {
    await answerWith(source);

    check();
  });

  it("shows raw HTML as text, not as elements", async () => {
    await answerWith("<b>raw</b> and <script>alert(1)</script>");

    expect(
      screen.getByText("<b>raw</b> and <script>alert(1)</script>"),
    ).toBeInTheDocument();
    expect(document.querySelector("b")).toBeNull();
    expect(document.querySelector("script")).toBeNull();
  });

  it("shows a Markdown image's alternative text and draws no image", async () => {
    await answerWith("Look: ![a cat](https://example.com/cat.png)");

    expect(document.body.textContent).toContain("a cat");
    expect(document.body.textContent).not.toContain("cat.png");
    expect(document.querySelector("img")).toBeNull();
    expect(net.calls.some((call) => call.url.includes("example.com"))).toBe(false);
  });

  it.each([
    { scheme: "https", label: "site", href: "https://example.com/page" },
    { scheme: "http", label: "site", href: "http://example.com/page" },
    { scheme: "mailto", label: "mail", href: "mailto:someone@example.com" },
  ])("draws a $scheme link as an anchor that opens in a new tab", async ({ label, href }) => {
    await answerWith(`[${label}](${href})`);

    const link = screen.getByRole("link", { name: label });
    expect(link).toHaveAttribute("href", href);
    expect(link).toHaveAttribute("target", "_blank");
    expect(link.getAttribute("rel")).toContain("noopener");
    expect(link.getAttribute("rel")).toContain("noreferrer");
  });

  it.each([
    { scheme: "javascript", source: "[run](javascript:x)" },
    { scheme: "data", source: "[run](data:text/html,hello)" },
    { scheme: "file", source: "[run](file:///etc/passwd)" },
  ])("shows a $scheme link as plain text", async ({ source }) => {
    await answerWith(source);

    expect(screen.queryByRole("link")).toBeNull();
    expect(document.body.textContent).toContain("run");
    expect(document.body.textContent).not.toContain("javascript:");
    expect(document.body.textContent).not.toContain("data:");
    expect(document.body.textContent).not.toContain("file:");
  });
});

describe("SPEC-ADDON-073 I10 - cited markers become pills where they sit", () => {
  it.each([
    { name: "a heading", source: "## See [1]", host: "h2" },
    { name: "a list item", source: "- See [1]", host: "li" },
    { name: "a table cell", source: "| A | B |\n| --- | --- |\n| See [1] | y |", host: "td" },
    { name: "bold text", source: "**See [1]**", host: "strong" },
    {
      name: "link text",
      source: "[See [1]](https://example.com)",
      host: "a",
    },
  ])("turns a cited marker in $name into a pill", async ({ source, host }) => {
    await answerWith(source, {
      citations: CITED_TIME,
      props: { mediaController: { seek: vi.fn() } },
    });

    const pill = screen.getByRole("button", { name: TIME_PILL });
    expect(pill.closest(host)).not.toBeNull();
    expect(document.body.textContent).not.toContain("[1]");
  });

  it.each([
    { name: "a code span", source: "Show `[1]` as written" },
    { name: "a code block", source: "```\n[1]\n```" },
  ])("leaves a marker in $name as written", async ({ source }) => {
    await answerWith(source, {
      citations: CITED_TIME,
      props: { mediaController: { seek: vi.fn() } },
    });

    expect(screen.queryByRole("button", { name: TIME_PILL })).toBeNull();
    expect(document.body.textContent).toContain("[1]");
  });

  it("drops an uncited marker and keeps the cited one", async () => {
    await answerWith("One [1], two [2].", {
      citations: CITED_TIME,
      props: { mediaController: { seek: vi.fn() } },
    });

    expect(screen.getByRole("button", { name: TIME_PILL })).toBeInTheDocument();
    expect(document.body.textContent).not.toContain("[2]");
  });

  it.each([
    {
      name: "a null-position citation",
      citations: [{ n: 1, position: null }] as FileCitation[],
    },
    { name: "an empty citations event", citations: [] as FileCitation[] },
  ])("leaves no pill for $name", async ({ citations }) => {
    await answerWith("See [1].", {
      citations,
      props: { mediaController: { seek: vi.fn() } },
    });

    expect(screen.queryByRole("button", { name: TIME_PILL })).toBeNull();
    expect(document.body.textContent).not.toContain(TIME_PILL);
    expect(document.body.textContent).not.toContain("[1]");
  });
});

describe("SPEC-ADDON-073 I10 - the answer streams as it arrives", () => {
  it("renders the text received so far on every answer_chunk", async () => {
    const stream = sse();
    net.answer = () => stream.response;
    await renderTab();
    await ask("who wrote this");

    await push(stream, frame("answer_chunk", { delta: "First." }));
    await waitFor(() =>
      expect(document.body.textContent).toContain("First."),
    );

    await push(stream, frame("answer_chunk", { delta: " Second." }));
    await waitFor(() =>
      expect(document.body.textContent).toContain("First. Second."),
    );
  });

  it("survives an unclosed fence and corrects it as it closes", async () => {
    const stream = sse();
    net.answer = () => stream.response;
    await renderTab();
    await ask("who wrote this");

    await push(stream, frame("answer_chunk", { delta: "```js\nconst x = 1;" }));
    await waitFor(() =>
      expect(document.body.textContent).toContain("const x = 1;"),
    );

    await push(stream, frame("answer_chunk", { delta: "\n```" }));
    await waitFor(() => expect(document.querySelector("pre code")).not.toBeNull());
  });

  it("survives a half-written table", async () => {
    const stream = sse();
    net.answer = () => stream.response;
    await renderTab();
    await ask("who wrote this");

    await push(stream, frame("answer_chunk", { delta: "| A | B |\n| --- |" }));
    await waitFor(() => expect(document.body.textContent).toContain("A"));

    await push(stream, frame("answer_chunk", { delta: "\n| 1 | 2 |" }));
    await waitFor(() => expect(document.body.textContent).toContain("1"));
  });
});

// ---------------------------------------------------------------------------
// SPEC-ADDON-053: the stream belongs to the file
// ---------------------------------------------------------------------------

describe("SPEC-ADDON-053 I11 - one stream per file", () => {
  it("sends no second request while an answer is streaming", async () => {
    const stream = sse();
    net.answer = () => stream.response;
    await renderTab();
    await ask("who wrote this");
    await push(stream, frame("answer_chunk", { delta: "Half an answer." }));

    fireEvent.click(screen.getByRole("button", { name: "Ask" }));

    expect(callsTo("POST", "/ask")).toHaveLength(1);
  });

  it("aborts the stream when the open file changes", async () => {
    const stream = sse();
    net.answer = () => stream.response;
    const view = await renderTab();
    await ask("who wrote this");
    await push(stream, frame("answer_chunk", { delta: "Half an answer." }));
    const inFlight = callsTo("POST", "/ask")[0];

    view.rerender(<AskTab {...view.props} fileId="ask000000002" />);

    await waitFor(() => expect(inFlight.signal?.aborted).toBe(true));
  });

  it("aborts the stream when the file page unmounts", async () => {
    const stream = sse();
    net.answer = () => stream.response;
    const view = await renderTab();
    await ask("who wrote this");
    await push(stream, frame("answer_chunk", { delta: "Half an answer." }));
    const inFlight = callsTo("POST", "/ask")[0];

    act(() => {
      window.dispatchEvent(pageClosed("ask000000001"));
    });
    view.unmount();

    await waitFor(() => expect(inFlight.signal?.aborted).toBe(true));
  });
});

// ---------------------------------------------------------------------------
// The surfaces beside the answer keep their rendering
// ---------------------------------------------------------------------------

describe("SPEC-ADDON-073 I10 - the shared Markdown component is unchanged", () => {
  it("still draws a Markdown image and an http link for the detailed summary", () => {
    render(
      <MarkdownPreview source={"![a cat](https://example.com/cat.png)\n\n[site](http://example.com)"} />,
    );

    const image = document.querySelector("img");
    expect(image).not.toBeNull();
    expect(image?.getAttribute("src")).toBe("https://example.com/cat.png");

    const link = screen.getByRole("link", { name: "site" });
    expect(link).toHaveAttribute("href", "http://example.com");
    expect(link).toHaveAttribute("target", "_blank");
    expect(link.getAttribute("rel")).toContain("noopener");
    expect(link.getAttribute("rel")).toContain("noreferrer");
  });
});

// ---------------------------------------------------------------------------
// The catalogue the tab reads
// ---------------------------------------------------------------------------

describe("SPEC-ADDON-071 step 8 - the new lines exist in both catalogues", () => {
  function findPath(
    tree: MessageTree,
    target: string,
    prefix = "",
  ): string | null {
    for (const [key, value] of Object.entries(tree)) {
      const path = prefix ? `${prefix}.${key}` : key;
      if (value === target) return path;
      if (typeof value === "object" && value !== null) {
        const found = findPath(value, target, path);
        if (found) return found;
      }
    }
    return null;
  }

  it.each([IMAGE_NOTICE, NEEDS_VISION, IMAGE_UNUSABLE])(
    "both catalogues carry %s",
    (text) => {
      const key = findPath(EN, text);
      expect(key).not.toBeNull();
      expect(lookup(EN, key!)).toBe(text);
      expect(typeof lookup(JA, key!)).toBe("string");
      expect(lookup(JA, key!)).not.toBe("");
    },
  );
});
