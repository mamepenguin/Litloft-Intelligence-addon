/**
 * SPEC-ADDON-056: the **Ask** tab on the file page.
 *
 * The tab under test is the addon's `slots.ts` component for its
 * `inspector-tabs` entry, so this suite holds what the core would draw. Each
 * route of the network is a function; nothing reaches a real model.
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

/**
 * The addon's own catalogue stands in for the merged one, so a key lands as
 * its English text here as it does on a page, without waiting for a merge.
 */
vi.mock("next-intl", async () => {
  const { existsSync: exists, readFileSync: read } = await import("node:fs");
  const { resolve: join } = await import("node:path");
  const root = [
    join(process.cwd(), "../addons/intelligence"),
    join(process.cwd(), "addons/intelligence"),
  ].find((candidate) => exists(candidate));
  const messages = JSON.parse(
    read(resolve(root!, "frontend/messages/en.json"), "utf-8"),
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

const ws = vi.hoisted(() => ({ event: null as Record<string, unknown> | null }));

vi.mock("@/hooks/useWebSocket", () => ({
  useWebSocket: (filter?: string) =>
    filter && ws.event?.event !== filter ? null : ws.event,
}));

import { slotComponents } from "@/addons/intelligence/slots";
import type { MediaController } from "@/lib/mediaController";
import type { DocumentController } from "@/lib/documentController";

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

/** A response body the test fills one frame at a time. */
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

// ---------------------------------------------------------------------------
// The tab and its props
// ---------------------------------------------------------------------------

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
    fileId: "vid000000001",
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
  fireEvent.change(await questionBox(), { target: { value: text } });
  fireEvent.click(await submitButton());
  await waitFor(() => expect(callsTo("POST", "/ask")).toHaveLength(1));
}

function pageClosed(fileId: string): CustomEvent<{ fileId: string }> {
  return new CustomEvent("litloft:file-page-closed", { detail: { fileId } });
}

async function push(stream: { push: (frame: string) => void }, text: string): Promise<void> {
  await act(async () => {
    stream.push(text);
  });
}

beforeEach(() => {
  net.calls = [];
  net.availability = () => json({ available: true });
  net.choices = () => json({ auto: "local", choices: [] });
  net.answer = () => json({ detail: "not stubbed" }, 500);
  ws.event = null;
  localStorage.clear();
  sessionStorage.clear();
  fetchMock.mockClear();
});

afterEach(() => {
  vi.restoreAllMocks();
});

// ---------------------------------------------------------------------------
// Availability
// ---------------------------------------------------------------------------

describe("SPEC-ADDON-056 I7 - the tab is listed only for a file it can answer about", () => {
  it("reports unavailable before the route answers", async () => {
    net.availability = () => new Promise<Response>(() => {});
    const onAvailability = vi.fn();

    render(<AskTab {...tabProps({ onAvailability })} />);

    await waitFor(() => expect(onAvailability).toHaveBeenCalledWith(false));
    expect(onAvailability).not.toHaveBeenCalledWith(true);
  });

  it("asks the availability route for the open file, with the drive", async () => {
    await renderTab();

    expect(callsTo("GET", "/files/vid000000001/ask")).toHaveLength(1);
    expect(callsTo("GET", "/files/vid000000001/ask")[0].url).toBe(
      "/api/addons/intelligence/files/vid000000001/ask",
    );
    expect(callsTo("GET", "/files/vid000000001/ask")[0].headers["x-lit-drive"]).toBe("family");
  });

  it.each([
    { name: "available false", reply: () => json({ available: false }) },
    { name: "an empty object", reply: () => json({}) },
    { name: "a 404", reply: () => json({ detail: "File not found" }, 404) },
    { name: "a 503", reply: () => json({ detail: "busy" }, 503) },
    { name: "a failed request", reply: () => Promise.reject(new TypeError("network down")) },
  ])("stays unavailable on $name", async ({ reply }) => {
    net.availability = reply as Reply;
    const onAvailability = vi.fn();

    render(<AskTab {...tabProps({ onAvailability })} />);

    await waitFor(() => expect(net.calls.length).toBeGreaterThan(0));
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
    expect(onAvailability).toHaveBeenCalled();
    expect(onAvailability).not.toHaveBeenCalledWith(true);
  });

  it("asks again on a fileId change without a remount", async () => {
    net.availability = (call) =>
      json({ available: !call.url.includes("vid000000002") });
    const onAvailability = vi.fn();

    const { rerender } = render(<AskTab {...tabProps({ onAvailability })} />);
    await waitFor(() => expect(onAvailability).toHaveBeenCalledWith(true));

    rerender(<AskTab {...tabProps({ fileId: "vid000000002", onAvailability })} />);

    await waitFor(() =>
      expect(callsTo("GET", "/files/vid000000002/ask")).toHaveLength(1),
    );
    await waitFor(() =>
      expect(onAvailability.mock.calls[onAvailability.mock.calls.length - 1][0]).toBe(false),
    );
  });

  it("reports the new file's true, not the old file's answer", async () => {
    net.availability = (call) =>
      json({ available: call.url.includes("vid000000002") });
    const onAvailability = vi.fn();

    const { rerender } = render(<AskTab {...tabProps({ onAvailability })} />);
    await waitFor(() => expect(onAvailability).toHaveBeenLastCalledWith(false));

    rerender(<AskTab {...tabProps({ fileId: "vid000000002", onAvailability })} />);

    await waitFor(() =>
      expect(callsTo("GET", "/files/vid000000002/ask")).toHaveLength(1),
    );
    await waitFor(() => expect(onAvailability).toHaveBeenLastCalledWith(true));
  });

  it("re-checks a media file once its transcription completes", async () => {
    net.availability = () => json({ available: false });
    const view = await renderTab();
    const before = callsTo("GET", "/files/vid000000001/ask").length;

    net.availability = () => json({ available: true });
    ws.event = {
      event: "intelligence.transcription.completed",
      data: { file_id: "vid000000001", drive: "family" },
    };
    view.rerender(<AskTab {...view.props} />);

    await waitFor(() =>
      expect(callsTo("GET", "/files/vid000000001/ask").length).toBeGreaterThan(before),
    );
    await waitFor(() => expect(view.props.onAvailability).toHaveBeenLastCalledWith(true));
  });

  it("ignores a transcription event for another file", async () => {
    net.availability = () => json({ available: false });
    const view = await renderTab();
    const before = callsTo("GET", "/files/vid000000001/ask").length;

    ws.event = {
      event: "intelligence.transcription.completed",
      data: { file_id: "vid000000009", drive: "family" },
    };
    view.rerender(<AskTab {...view.props} />);

    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
    expect(callsTo("GET", "/files/vid000000001/ask")).toHaveLength(before);
  });
});

// ---------------------------------------------------------------------------
// The question box
// ---------------------------------------------------------------------------

describe("SPEC-ADDON-056 2 - the question box", () => {
  it("takes at most 1000 characters", async () => {
    await renderTab();

    expect(await questionBox()).toHaveAttribute("maxlength", "1000");
  });

  it("is disabled while the question holds fewer than 3 non-space characters", async () => {
    await renderTab();
    const box = await questionBox();
    const button = await submitButton();

    expect(button).toBeDisabled();
    fireEvent.change(box, { target: { value: " ab " } });
    expect(button).toBeDisabled();
    fireEvent.change(box, { target: { value: "abc" } });
    expect(button).toBeEnabled();
  });

  it("shows the question it sent", async () => {
    const stream = sse();
    net.answer = () => stream.response;
    await renderTab();

    await ask("who wrote this");

    expect(document.body.textContent).toContain("who wrote this");
    expect(callsTo("POST", "/ask")[0].body?.question).toBe("who wrote this");
  });
});

// ---------------------------------------------------------------------------
// The model select
// ---------------------------------------------------------------------------

describe("SPEC-ADDON-056 2 - the model select", () => {
  it("offers the choices Ask offers, defaulting to Auto", async () => {
    net.choices = () =>
      json({
        auto: "local",
        choices: [
          { name: "local", model: "llama3", offhost: false },
          { name: "cloud", model: "gpt-4o-mini", offhost: true },
        ],
      });
    await renderTab();

    const select = (await screen.findByRole("combobox")) as HTMLSelectElement;

    expect(Array.from(select.options).map((option) => option.value)).toEqual([
      "",
      "local",
      "cloud",
    ]);
    expect(select.value).toBe("");
  });

  it("is not drawn when Ask would not offer a choice", async () => {
    net.choices = () =>
      json({ auto: "local", choices: [{ name: "local", model: "llama3", offhost: false }] });
    await renderTab();

    await waitFor(() => expect(callsTo("GET", "/llm/choices/rag")).toHaveLength(1));
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
    expect(screen.queryByRole("combobox")).toBeNull();
  });

  it("defaults to the only profile when routing picks none", async () => {
    net.choices = () =>
      json({ auto: null, choices: [{ name: "cloud", model: "gpt-4o-mini", offhost: true }] });
    const stream = sse();
    net.answer = () => stream.response;
    await renderTab();

    expect(await screen.findByRole("combobox")).toBeTruthy();
    await ask("who wrote this");

    expect(callsTo("POST", "/ask")[0].body?.profile).toBe("cloud");
  });

  it("sends no profile while Auto is chosen", async () => {
    net.choices = () =>
      json({
        auto: "local",
        choices: [
          { name: "local", model: "llama3", offhost: false },
          { name: "cloud", model: "gpt-4o-mini", offhost: true },
        ],
      });
    const stream = sse();
    net.answer = () => stream.response;
    await renderTab();

    await ask("who wrote this");

    expect(callsTo("POST", "/ask")[0].body?.profile ?? null).toBeNull();
  });

  it("shows Ask's message for an unusable profile and fetches the choices again", async () => {
    net.choices = () =>
      json({
        auto: "local",
        choices: [
          { name: "local", model: "llama3", offhost: false },
          { name: "cloud", model: "gpt-4o-mini", offhost: true },
        ],
      });
    net.answer = () => json({ detail: "profile_unavailable" }, 400);
    await renderTab();
    await waitFor(() => expect(callsTo("GET", "/llm/choices/rag")).toHaveLength(1));

    await ask("who wrote this");

    expect(await screen.findByText(EN.llmChoice.profileUnavailable)).toBeTruthy();
    await waitFor(() => expect(callsTo("GET", "/llm/choices/rag")).toHaveLength(2));
  });
});

// ---------------------------------------------------------------------------
// The answer
// ---------------------------------------------------------------------------

describe("SPEC-ADDON-056 3 - markers become pills only once citations arrive", () => {
  it("shows an [n] marker as typed while the answer streams", async () => {
    const stream = sse();
    net.answer = () => stream.response;
    await renderTab();

    await ask("who wrote this");
    await push(stream, frame("answer_chunk", { delta: "See [1] for the answer." }));

    expect(document.body.textContent).toContain("See [1] for the answer.");
    expect(screen.queryByRole("button", { name: "28:25" })).toBeNull();
  });

  it("turns a cited time into a pill in M:SS", async () => {
    const stream = sse();
    net.answer = () => stream.response;
    await renderTab({ mediaController: { seek: vi.fn() } });

    await ask("who wrote this");
    await push(stream, frame("answer_chunk", { delta: "See [1]." }));
    await push(stream, frame("citations", [{ n: 1, position: { seconds: 1705 } }]));

    expect(screen.getByRole("button", { name: "28:25" })).toBeTruthy();
    expect(document.body.textContent).not.toContain("[1]");
  });

  it("formats an hour or more as H:MM:SS", async () => {
    const stream = sse();
    net.answer = () => stream.response;
    await renderTab({ mediaController: { seek: vi.fn() } });

    await ask("who wrote this");
    await push(stream, frame("answer_chunk", { delta: "See [1]." }));
    await push(stream, frame("citations", [{ n: 1, position: { seconds: 3723 } }]));

    expect(screen.getByRole("button", { name: "1:02:03" })).toBeTruthy();
  });

  it.each([
    { label: "p.12", position: { page: 12 }, target: { page: 12 } },
    {
      label: "Chapter Three",
      position: { section: 3, title: "Chapter Three" },
      target: { section: 3 },
    },
    { label: "§4", position: { section: 4, title: null }, target: { section: 4 } },
  ])("labels a $label position and moves the viewer to it", async ({ label, position, target }) => {
    const stream = sse();
    net.answer = () => stream.response;
    const goTo = vi.fn();
    const lowerInspector = vi.fn();
    await renderTab({ documentController: { goTo }, lowerInspector });

    await ask("who wrote this");
    await push(stream, frame("answer_chunk", { delta: "See [1]." }));
    await push(stream, frame("citations", [{ n: 1, position }]));

    fireEvent.click(screen.getByRole("button", { name: label }));

    expect(goTo).toHaveBeenCalledWith(target);
    expect(lowerInspector).toHaveBeenCalled();
  });

  it("turns an uncited or null-position marker into nothing", async () => {
    const stream = sse();
    net.answer = () => stream.response;
    await renderTab({ mediaController: { seek: vi.fn() } });

    await ask("who wrote this");
    await push(stream, frame("answer_chunk", { delta: "One [1], two [2], three [3]." }));
    await push(
      stream,
      frame("citations", [
        { n: 1, position: { seconds: 5 } },
        { n: 3, position: null },
      ]),
    );

    expect(document.body.textContent).not.toContain("[2]");
    expect(document.body.textContent).not.toContain("[3]");
    expect(screen.queryByRole("button", { name: /^§/ })).toBeNull();
  });

  it("seeks the player and lowers the inspector when a time pill is clicked", async () => {
    const stream = sse();
    net.answer = () => stream.response;
    const seek = vi.fn();
    const lowerInspector = vi.fn();
    await renderTab({ mediaController: { seek }, lowerInspector });

    await ask("who wrote this");
    await push(stream, frame("answer_chunk", { delta: "See [1]." }));
    await push(stream, frame("citations", [{ n: 1, position: { seconds: 1705 } }]));

    fireEvent.click(screen.getByRole("button", { name: "28:25" }));

    expect(seek).toHaveBeenCalledWith(1705);
    expect(lowerInspector).toHaveBeenCalled();
  });

  it("leaves the position as plain text when the page cannot move there", async () => {
    const stream = sse();
    net.answer = () => stream.response;
    await renderTab({ mediaController: null, documentController: null });

    await ask("who wrote this");
    await push(stream, frame("answer_chunk", { delta: "See [1]." }));
    await push(stream, frame("citations", [{ n: 1, position: { seconds: 1705 } }]));

    expect(screen.getByText("28:25")).toBeTruthy();
    expect(screen.queryByRole("button", { name: "28:25" })).toBeNull();
  });

  it("shows the partial notice when the file only partly fit", async () => {
    const stream = sse();
    net.answer = () => stream.response;
    await renderTab();

    await ask("who wrote this");
    await push(stream, frame("answer_chunk", { delta: "An answer." }));
    await push(stream, frame("citations", []));
    await push(stream, frame("meta", { partial: true }));

    expect(await screen.findByText("Answered from part of the file")).toBeTruthy();
  });

  it("shows no partial notice when the whole file was read", async () => {
    const stream = sse();
    net.answer = () => stream.response;
    await renderTab();

    await ask("who wrote this");
    await push(stream, frame("answer_chunk", { delta: "An answer." }));
    await push(stream, frame("citations", []));
    await push(stream, frame("meta", { partial: false }));

    await waitFor(() => expect(document.body.textContent).toContain("An answer."));
    expect(screen.queryByText("Answered from part of the file")).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// Failures
// ---------------------------------------------------------------------------

describe("SPEC-ADDON-056 7 - one line per kind of failure", () => {
  it.each([
    { name: "no LLM is enabled", status: 400, detail: "LLM is not enabled" },
    { name: "the RAG feature is off", status: 400, detail: "RAG feature is disabled" },
  ])("shows the no-model line when $name", async ({ status, detail }) => {
    net.answer = () => json({ detail }, status);
    await renderTab();

    await ask("who wrote this");

    expect(await screen.findByText("No model is available for this question")).toBeTruthy();
  });

  it("shows the busy line for a full semaphore", async () => {
    net.answer = () =>
      json({ detail: "Too many concurrent Ask requests, please retry shortly" }, 503);
    await renderTab();

    await ask("who wrote this");

    expect(await screen.findByText("Busy. Try again in a moment")).toBeTruthy();
  });

  it.each([
    {
      name: "the policy is unavailable",
      reply: () => json({ detail: "LLM policy is unavailable" }, 503),
    },
    {
      name: "the stream closes without done",
      reply: () => {
        const stream = sse();
        stream.close();
        return stream.response;
      },
    },
    {
      name: "a request fails outright",
      reply: () => Promise.reject(new TypeError("network down")),
    },
  ])("shows the generic line when $name", async ({ reply }) => {
    net.answer = reply as Reply;
    await renderTab();

    await ask("who wrote this");

    expect(await screen.findByText("Could not answer")).toBeTruthy();
  });

  it("keeps the text it received before a done carrying an error", async () => {
    const stream = sse();
    net.answer = () => stream.response;
    await renderTab();

    await ask("who wrote this");
    await push(stream, frame("answer_chunk", { delta: "Half an answer." }));
    await push(stream, frame("done", { error: "Answer generation failed" }));

    expect(await screen.findByText("Could not answer")).toBeTruthy();
    expect(document.body.textContent).toContain("Half an answer.");
  });
});

// ---------------------------------------------------------------------------
// The per-file store
// ---------------------------------------------------------------------------

describe("SPEC-ADDON-056 6 and I11 - what belongs to the file", () => {
  async function answered(): Promise<void> {
    const stream = sse();
    net.answer = () => stream.response;
    await ask("who wrote this");
    await push(stream, frame("answer_chunk", { delta: "The author [1]." }));
    await push(stream, frame("citations", [{ n: 1, position: { seconds: 1705 } }]));
    await push(stream, frame("meta", { partial: false }));
    await push(stream, frame("done", {}));
    await waitFor(() => expect(document.body.textContent).toContain("The author"));
  }

  it("keeps the question and the answer across a tab switch or a closed sheet", async () => {
    const view = await renderTab();
    await answered();

    view.unmount();
    await act(async () => {
      render(<AskTab {...view.props} />);
    });

    expect(await screen.findByDisplayValue("who wrote this")).toBeTruthy();
    expect(document.body.textContent).toContain("The author");
  });

  it("drops the answer when the open file changes", async () => {
    const view = await renderTab();
    await answered();

    view.rerender(<AskTab {...view.props} fileId="vid000000002" />);

    await waitFor(() =>
      expect(document.body.textContent).not.toContain("The author"),
    );
    expect((await questionBox()).value).toBe("");
  });

  it("aborts the stream when the open file changes", async () => {
    const stream = sse();
    net.answer = () => stream.response;
    const view = await renderTab();
    await ask("who wrote this");
    await push(stream, frame("answer_chunk", { delta: "Half an answer." }));
    const inFlight = callsTo("POST", "/ask")[0];

    view.rerender(<AskTab {...view.props} fileId="vid000000002" />);

    await waitFor(() => expect(inFlight.signal?.aborted).toBe(true));
  });

  it("SPEC-ADDON-056 I11 - aborts the stream when the file page closes", async () => {
    const stream = sse();
    net.answer = () => stream.response;
    await renderTab();
    await ask("who wrote this");
    await push(stream, frame("answer_chunk", { delta: "Half an answer." }));
    const inFlight = callsTo("POST", "/ask")[0];

    act(() => {
      window.dispatchEvent(pageClosed("vid000000001"));
    });

    await waitFor(() => expect(inFlight.signal?.aborted).toBe(true));
  });

  it("SPEC-ADDON-056 I11 - keeps a streaming answer across a tab unmount and shows it whole on return", async () => {
    const stream = sse();
    net.answer = () => stream.response;
    const view = await renderTab({ fileId: "vid000000011" });
    await ask("who wrote this");
    await push(stream, frame("answer_chunk", { delta: "Half an answer." }));

    view.unmount();
    await push(stream, frame("answer_chunk", { delta: " The rest." }));
    await push(stream, frame("done", {}));

    render(<AskTab {...view.props} />);

    expect(await screen.findByDisplayValue("who wrote this")).toBeTruthy();
    await waitFor(() =>
      expect(document.body.textContent).toContain("Half an answer. The rest."),
    );
    expect(callsTo("POST", "/ask")).toHaveLength(1);
  });

  it("SPEC-ADDON-056 I11 - shows the text so far when the tab returns mid-stream and continues into it", async () => {
    const stream = sse();
    net.answer = () => stream.response;
    const view = await renderTab({ fileId: "vid000000012" });
    await ask("who wrote this");
    await push(stream, frame("answer_chunk", { delta: "Half an answer." }));

    view.unmount();
    render(<AskTab {...view.props} />);

    expect(await screen.findByDisplayValue("who wrote this")).toBeTruthy();
    expect(document.body.textContent).toContain("Half an answer.");

    await push(stream, frame("answer_chunk", { delta: " The rest." }));

    await waitFor(() =>
      expect(document.body.textContent).toContain("Half an answer. The rest."),
    );
    expect(callsTo("POST", "/ask")).toHaveLength(1);
  });

  it("SPEC-ADDON-056 I11 - the file-page-closed event aborts the file's stream and forgets it", async () => {
    const stream = sse();
    net.answer = () => stream.response;
    const view = await renderTab({ fileId: "vid000000013" });
    await ask("who wrote this");
    await push(stream, frame("answer_chunk", { delta: "Half an answer." }));
    const inFlight = callsTo("POST", "/ask")[0];

    act(() => {
      window.dispatchEvent(pageClosed("vid000000013"));
    });

    await waitFor(() => expect(inFlight.signal?.aborted).toBe(true));

    view.unmount();
    render(<AskTab {...view.props} />);

    expect((await questionBox()).value).toBe("");
    expect(document.body.textContent).not.toContain("Half an answer.");
  });

  it("SPEC-ADDON-056 I11 - the file-page-closed event for another file changes nothing", async () => {
    const stream = sse();
    net.answer = () => stream.response;
    await renderTab({ fileId: "vid000000014" });
    await ask("who wrote this");
    await push(stream, frame("answer_chunk", { delta: "Half an answer." }));
    const inFlight = callsTo("POST", "/ask")[0];

    act(() => {
      window.dispatchEvent(pageClosed("vid000000099"));
    });

    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
    expect(inFlight.signal?.aborted).toBe(false);
    expect(document.body.textContent).toContain("Half an answer.");

    await push(stream, frame("answer_chunk", { delta: " The rest." }));
    await waitFor(() =>
      expect(document.body.textContent).toContain("Half an answer. The rest."),
    );
  });

  it("SPEC-ADDON-056 I11 - a settled answer is forgotten when the file page closes", async () => {
    const stream = sse();
    net.answer = () => stream.response;
    const view = await renderTab({ fileId: "vid000000016" });
    await ask("who wrote this");
    await push(stream, frame("answer_chunk", { delta: "The author [1]." }));
    await push(stream, frame("done", {}));
    await waitFor(() => expect(document.body.textContent).toContain("The author"));

    act(() => {
      window.dispatchEvent(pageClosed("vid000000016"));
    });
    view.unmount();
    render(<AskTab {...view.props} />);

    expect((await questionBox()).value).toBe("");
    expect(document.body.textContent).not.toContain("The author");
  });

  it("sends no second request while an answer is streaming", async () => {
    const stream = sse();
    net.answer = () => stream.response;
    await renderTab();
    await ask("who wrote this");
    await push(stream, frame("answer_chunk", { delta: "Half an answer." }));

    fireEvent.click(screen.getByRole("button", { name: "Ask" }));

    expect(callsTo("POST", "/ask")).toHaveLength(1);
  });

  it("stores nothing in the browser", async () => {
    await renderTab();
    await answered();

    expect(localStorage.length).toBe(0);
    expect(sessionStorage.length).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// The catalogue the tab reads
// ---------------------------------------------------------------------------

describe("SPEC-ADDON-056 - the strings a viewer reads", () => {
  const flatten = (tree: unknown): string[] => {
    if (typeof tree === "string") return [tree];
    if (typeof tree !== "object" || tree === null) return [];
    return Object.values(tree as Record<string, unknown>).flatMap(flatten);
  };

  it.each([
    { locale: "en", catalogue: EN, text: "Ask about this file" },
    { locale: "en", catalogue: EN, text: "Answered from part of the file" },
    { locale: "en", catalogue: EN, text: "No model is available for this question" },
    { locale: "en", catalogue: EN, text: "Busy. Try again in a moment" },
    { locale: "en", catalogue: EN, text: "Could not answer" },
    { locale: "ja", catalogue: JA, text: "このファイルについて質問" },
    { locale: "ja", catalogue: JA, text: "ファイルの一部を読んで答えました" },
    { locale: "ja", catalogue: JA, text: "この質問に使えるモデルがありません" },
    {
      locale: "ja",
      catalogue: JA,
      text: "混み合っています。少し待ってからもう一度お試しください",
    },
    { locale: "ja", catalogue: JA, text: "答えを作れませんでした" },
  ])("$locale catalogue holds $text", ({ catalogue, text }) => {
    expect(flatten(catalogue).includes(text)).toBe(true);
  });

  it("names the tab in both catalogues", () => {
    const key = ENTRY?.i18n_key as string;
    const lookup = (tree: unknown, path: string): unknown =>
      path.split(".").reduce<unknown>(
        (node, part) =>
          typeof node === "object" && node !== null
            ? (node as Record<string, unknown>)[part]
            : undefined,
        tree,
      );

    expect(typeof lookup(EN, key)).toBe("string");
    expect(typeof lookup(JA, key)).toBe("string");
  });
});
