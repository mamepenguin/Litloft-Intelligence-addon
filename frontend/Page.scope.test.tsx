/**
 * SPEC-ADDON-064: the Ask page's scope — chips and the 「＋ 範囲」 menu, the
 * scope line, the request body, the scoped cache key, the empty-scope answer,
 * the scope-unavailable error, and the scope line a saved note records.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

const ADDON_ROOT = [
  resolve(process.cwd(), "../addons/intelligence"),
  resolve(process.cwd(), "addons/intelligence"),
].find((candidate) => existsSync(resolve(candidate, "manifest.json")))!;

function readMessages(path: string): Record<string, unknown> {
  return JSON.parse(readFileSync(path, "utf-8")) as Record<string, unknown>;
}

function merge(
  base: Record<string, unknown>,
  extra: Record<string, unknown>,
): Record<string, unknown> {
  const out: Record<string, unknown> = { ...base };
  for (const [key, value] of Object.entries(extra)) {
    const current = out[key];
    if (
      typeof value === "object" && value !== null && !Array.isArray(value) &&
      typeof current === "object" && current !== null && !Array.isArray(current)
    ) {
      out[key] = merge(
        current as Record<string, unknown>,
        value as Record<string, unknown>,
      );
    } else {
      out[key] = value;
    }
  }
  return out;
}

vi.mock("next-intl", async () => {
  const { existsSync: exists, readFileSync: read } = await import("node:fs");
  const { resolve: join } = await import("node:path");
  const root = [
    join(process.cwd(), "../addons/intelligence"),
    join(process.cwd(), "addons/intelligence"),
  ].find((candidate) => exists(candidate))!;
  const messages = merge(
    JSON.parse(read(join(process.cwd(), "src/messages/en.json"), "utf-8")),
    JSON.parse(read(join(root, "frontend/messages/en.json"), "utf-8")),
  );
  const lookup = (path: string): unknown =>
    path.split(".").reduce<unknown>(
      (tree, part) =>
        typeof tree === "object" && tree !== null
          ? (tree as Record<string, unknown>)[part]
          : undefined,
      messages,
    );
  return {
    useTranslations: (namespace?: string) => {
      const t = (key: string, values?: Record<string, unknown>) => {
        const found = lookup(namespace ? `${namespace}.${key}` : key);
        const raw = typeof found === "string" ? found : key;
        if (!values) return raw;
        return Object.entries(values).reduce(
          (text, [name, value]) => text.replace(`{${name}}`, String(value)),
          raw,
        );
      };
      return t;
    },
    useLocale: () => "en",
    NextIntlClientProvider: ({ children }: { children: unknown }) => children,
  };
});

const navigationSearch = vi.hoisted(() => ({ current: null as string | null }));

vi.mock("next/navigation", () => ({
  useSearchParams: () =>
    new URLSearchParams(navigationSearch.current ?? window.location.search),
  usePathname: () => "/drive/family/addons/intelligence",
}));

vi.mock("@/components/CurrentDriveProvider", () => ({
  useCurrentDrive: () => "family",
}));

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    getDrives: vi.fn(async () => [{ name: "family", file_count: 619 }]),
  };
});

vi.mock("@/addons/intelligence/api", async () => {
  const actual = await vi.importActual<
    typeof import("@/addons/intelligence/api")
  >("@/addons/intelligence/api");
  return {
    ...actual,
    askQuestionStream: vi.fn(),
    fetchLLMChoices: vi.fn(async () => {
      throw new Error("not stubbed");
    }),
    getIntelligenceStatus: vi.fn().mockResolvedValue({
      status: "ok",
      features: {
        indexing: true,
        search: true,
        auto_tags: "false",
        summaries: "false",
        rag: true,
      },
      llm: {
        provider: "ollama",
        model: "qwen",
        enabled: true,
        output_language: "auto",
      },
    }),
  };
});

vi.mock("@/addons/intelligence/AskSaveDialog", () => ({
  AskSaveDialog: ({ open, content }: { open: boolean; content: string }) =>
    open ? <pre data-testid="ask-note-content">{content}</pre> : null,
}));

import IntelligenceAskPage from "@/addons/intelligence/Page";
import {
  askQuestionStream,
  LLMRequestError,
  type AskStreamEvent,
} from "@/addons/intelligence/api";

const SCOPE_CACHE_PREFIX = "intelligence-ask-cache:v1";

const stream: { events: AskStreamEvent[]; error: unknown } = {
  events: [],
  error: null,
};

function json(body: unknown, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: String(status),
    json: async () => body,
    text: async () => JSON.stringify(body),
    body: null,
  } as unknown as Response;
}

const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
  const url = new URL(String(input), "http://localhost");
  if (url.pathname === "/api/drives/family/files") {
    return json({ files: [], meta: { total: 7 } });
  }
  if (url.pathname.endsWith("/tags")) return json([]);
  if (url.pathname.endsWith("/folders")) return json([]);
  if (url.pathname.endsWith("/collections")) return json([]);
  return json({});
});

function setUrl(search: string) {
  window.history.replaceState(
    null,
    "",
    `/drive/family/addons/intelligence${search}`,
  );
}

function optionsScope(call: unknown[]): unknown {
  for (const arg of call.slice(2)) {
    if (arg && typeof arg === "object" && "scope" in (arg as object)) {
      return (arg as { scope: unknown }).scope;
    }
  }
  return undefined;
}

function expectedScope() {
  return {
    folders: ["Work"],
    collections: [],
    tags: ["会議"],
    kinds: [],
  };
}

function storageKeys(): string[] {
  const keys: string[] = [];
  for (let i = 0; i < window.sessionStorage.length; i += 1) {
    const key = window.sessionStorage.key(i);
    if (key !== null) keys.push(key);
  }
  return keys;
}

function scopedKeyFor(question: string): string {
  const key = storageKeys().find(
    (candidate) =>
      candidate.startsWith(`${SCOPE_CACHE_PREFIX}:family:`) &&
      candidate.endsWith(`:${question}`),
  );
  expect(key).toBeDefined();
  expect(key).not.toBe(`${SCOPE_CACHE_PREFIX}:family:${question}`);
  return key!;
}

function filesRequest() {
  return fetchMock.mock.calls
    .map(([input]) => new URL(String(input), "http://localhost"))
    .find((url) => url.pathname === "/api/drives/family/files");
}

async function answer(events: AskStreamEvent[]) {
  stream.events = events;
}

beforeEach(() => {
  window.sessionStorage.clear();
  setUrl("");
  stream.events = [];
  stream.error = null;
  navigationSearch.current = null;
  fetchMock.mockClear();
  vi.stubGlobal("fetch", fetchMock);
  vi.mocked(askQuestionStream).mockReset();
  vi.mocked(askQuestionStream).mockImplementation(async function* () {
    if (stream.error) throw stream.error;
    for (const event of stream.events) yield event;
  });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("SPEC-ADDON-064 the Ask page's scope", () => {
  it("reads ?scope= on mount, draws its chips and menu, and counts the scope", async () => {
    setUrl("?scope=path%3AWork");

    render(<IntelligenceAskPage />);

    expect(
      await screen.findByRole("button", { name: "Remove Folder Work" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Scope/ })).toBeInTheDocument();
    await waitFor(() =>
      expect(screen.getByTestId("drive-scope")).toHaveTextContent(
        "family · 7 items",
      ),
    );

    const listing = filesRequest();
    expect(listing).toBeDefined();
    expect(listing!.searchParams.getAll("folder")).toContain("Work");
    expect(listing!.searchParams.get("limit")).toBe("1");

    for (const link of screen.getAllByRole("link")) {
      expect(
        new URL(link.getAttribute("href")!, "http://localhost").searchParams.get(
          "scope",
        ),
      ).toBe("path:Work");
    }
  });

  it("sends the four normalised keys as the request body's scope", async () => {
    await answer([
      { kind: "scope", scope: { folders: ["Work"], collections: [], tags: [], kinds: [] }, matched_file_count: 2 },
      { kind: "answer_chunk", delta: "answer" },
      { kind: "citations", citations: [] },
      { kind: "done", took_ms: 4 },
    ] as unknown as AskStreamEvent[]);
    setUrl(
      "?q=what%20did%20we%20decide&scope=path%3AWork%20tag%3A%E4%BC%9A%E8%AD%B0",
    );

    render(<IntelligenceAskPage />);

    await waitFor(() => expect(askQuestionStream).toHaveBeenCalled());
    const call = vi.mocked(askQuestionStream).mock.calls[0];
    expect(optionsScope(call)).toEqual(expectedScope());
  });

  it("a removed chip aborts the stream, clears the answer, keeps the question, and rewrites the URL", async () => {
    let aborted = false;
    vi.mocked(askQuestionStream).mockImplementation(async function* (
      _query: string,
      _drive: string,
      options?: { signal?: AbortSignal },
    ) {
      yield {
        kind: "sources",
        sources: [
          {
            file_id: "f1",
            drive: "family",
            filename: "source.md",
            file_type: "document",
          },
        ],
      } as unknown as AskStreamEvent;
      yield { kind: "answer_chunk", delta: "partial answer" } as AskStreamEvent;
      await new Promise<void>((_resolve, reject) => {
        const signal = options?.signal;
        const abort = () => {
          aborted = true;
          reject(new DOMException("Aborted", "AbortError"));
        };
        if (!signal) {
          reject(new Error("the page asked without an abort signal"));
          return;
        }
        if (signal.aborted) abort();
        else signal.addEventListener("abort", abort);
      });
    });
    setUrl(
      "?q=cats&scope=path%3AWork%20tag%3A%E4%BC%9A%E8%AD%B0",
    );
    render(<IntelligenceAskPage />);
    expect(await screen.findByText("partial answer")).toBeInTheDocument();
    expect(screen.getByText("source.md")).toBeInTheDocument();
    expect(askQuestionStream).toHaveBeenCalledTimes(1);

    fireEvent.click(
      screen.getByRole("button", { name: /Remove Tag 会議/ }),
    );

    await waitFor(() => expect(aborted).toBe(true));
    await waitFor(() =>
      expect(screen.queryByText("partial answer")).not.toBeInTheDocument(),
    );
    expect(screen.queryByText("source.md")).not.toBeInTheDocument();
    expect(screen.getByLabelText("Question input")).toHaveValue("cats");
    const url = new URL(window.location.href);
    expect(url.searchParams.get("scope")).toBe("path:Work");
    expect(url.searchParams.get("q")).toBeNull();
    expect(askQuestionStream).toHaveBeenCalledTimes(1);
  });

  it("a menu choice adds the scope without asking again", async () => {
    await answer([
      { kind: "answer_chunk", delta: "answer" },
      { kind: "citations", citations: [] },
      { kind: "done", took_ms: 4 },
    ] as unknown as AskStreamEvent[]);
    setUrl("?q=cats");
    render(<IntelligenceAskPage />);
    expect(await screen.findByText("answer")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /Scope/ }));
    fireEvent.click(screen.getByRole("menuitem", { name: "Type" }));
    fireEvent.click(await screen.findByRole("menuitem", { name: "Video" }));

    expect(
      await screen.findByRole("button", { name: "Remove Type Video" }),
    ).toBeInTheDocument();
    expect(screen.getByLabelText("Question input")).toHaveValue("cats");
    const url = new URL(window.location.href);
    expect(url.searchParams.get("scope")).toBe("type:video");
    expect(url.searchParams.get("q")).toBeNull();
    expect(askQuestionStream).toHaveBeenCalledTimes(1);
  });

  it("does not read a useSearchParams change", async () => {
    const view = render(<IntelligenceAskPage />);
    await screen.findByRole("button", { name: /Scope/ });

    navigationSearch.current = "?q=cats";
    view.rerender(<IntelligenceAskPage />);

    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 25));
    });
    expect(askQuestionStream).not.toHaveBeenCalled();
  });

  it("a popstate read shows the pair in the new URL", async () => {
    render(<IntelligenceAskPage />);
    await screen.findByRole("button", { name: /Scope/ });

    act(() => {
      setUrl("?q=cats&scope=tag%3A%E4%BC%9A%E8%AD%B0");
      window.dispatchEvent(new PopStateEvent("popstate"));
    });

    await waitFor(() => expect(askQuestionStream).toHaveBeenCalled());
    const call = vi.mocked(askQuestionStream).mock.calls[0];
    expect(optionsScope(call)).toMatchObject({ tags: ["会議"] });
    expect(call[0]).toBe("cats");
  });

  it("caches a scoped answer with its count, and treats a missing count as a miss", async () => {
    const events = [
      { kind: "scope", scope: { folders: ["Work"], collections: [], tags: [], kinds: [] }, matched_file_count: 34 },
      { kind: "answer_chunk", delta: "scoped answer" },
      { kind: "citations", citations: [] },
      { kind: "done", took_ms: 4 },
    ] as unknown as AskStreamEvent[];

    await answer(events);
    setUrl("?q=cats&scope=path%3AWork");
    render(<IntelligenceAskPage />);
    await waitFor(() =>
      expect(screen.getByText("scoped answer")).toBeInTheDocument(),
    );

    const key = scopedKeyFor("cats");
    const entry = JSON.parse(window.sessionStorage.getItem(key)!);
    expect(entry).toMatchObject({ matchedFileCount: 34 });
    expect(JSON.parse(key.split(":").slice(3, -1).join(":"))).toMatchObject({
      folders: ["Work"],
    });

    vi.mocked(askQuestionStream).mockClear();
    // The second mount must ask again rather than restore the cached answer;
    // give that re-run no events so a replay cannot paint the same text.
    await answer([]);
    window.sessionStorage.setItem(
      key,
      JSON.stringify({ ...entry, matchedFileCount: null }),
    );
    cleanup();
    render(<IntelligenceAskPage />);

    await waitFor(() => expect(askQuestionStream).toHaveBeenCalledTimes(1));
    expect(screen.queryByText("scoped answer")).not.toBeInTheDocument();
  });

  it("restores a scoped answer and shows its matched_file_count", async () => {
    await answer([
      { kind: "scope", scope: { folders: ["Work"], collections: [], tags: [], kinds: [] }, matched_file_count: 34 },
      { kind: "answer_chunk", delta: "restored answer" },
      { kind: "citations", citations: [] },
      { kind: "done", took_ms: 3 },
    ] as unknown as AskStreamEvent[]);
    setUrl("?q=cats&scope=path%3AWork");
    render(<IntelligenceAskPage />);
    await screen.findByText("restored answer");
    scopedKeyFor("cats");

    vi.mocked(askQuestionStream).mockClear();
    cleanup();
    render(<IntelligenceAskPage />);

    expect(await screen.findByText("restored answer")).toBeInTheDocument();
    expect(askQuestionStream).not.toHaveBeenCalled();
    expect(screen.getByTestId("drive-scope")).toHaveTextContent("34");
  });

  it("shows the empty-scope message and does not cache the outcome", async () => {
    await answer([
      { kind: "scope", scope: { folders: ["Work"], collections: [], tags: [], kinds: [] }, matched_file_count: 0 },
      { kind: "citations", citations: [] },
      { kind: "done", retrieved_count: 0, scope_empty: true, took_ms: 1 },
    ] as unknown as AskStreamEvent[]);
    setUrl("?q=cats&scope=path%3AWork");

    render(<IntelligenceAskPage />);

    expect(
      await screen.findByText("No files in this scope"),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Remove Folder Work" }),
    ).toBeInTheDocument();
    expect(
      storageKeys().filter((key) => key.endsWith(":cats")),
    ).toEqual([]);
  });

  it("shows the scope-unavailable message for that 503 only", async () => {
    stream.error = new LLMRequestError("503 Service Unavailable", 503, {
      code: "scope_unavailable",
      message: "Scope unavailable",
    });
    setUrl("?q=cats&scope=path%3AWork");

    render(<IntelligenceAskPage />);

    expect(
      await screen.findByText("Could not load the scope"),
    ).toBeInTheDocument();
  });

  it("records the scope string in a saved note and omits it when unscoped", async () => {
    await answer([
      { kind: "answer_chunk", delta: "saved answer" },
      {
        kind: "citations",
        citations: [
          {
            file_id: "f1",
            drive: "family",
            filename: "doc.md",
            file_type: "document",
            quote: "q",
            relevance: 0.9,
            segment_location: null,
          },
        ],
      },
      { kind: "done", took_ms: 4 },
    ] as unknown as AskStreamEvent[]);
    setUrl("?q=cats&scope=path%3AWork");
    render(<IntelligenceAskPage />);
    await screen.findByText("saved answer");

    fireEvent.click(await screen.findByTestId("ask-save-note"));

    const content = await screen.findByTestId("ask-note-content");
    const text = content.textContent ?? "";
    expect(text).toMatch(/^query: "cats"$/m);
    expect(text).toMatch(/^scope: "path:Work"$/m);
    expect(text.indexOf('scope: "path:Work"')).toBeGreaterThan(
      text.indexOf('query: "cats"'),
    );

    cleanup();
    setUrl("?q=dogs");
    render(<IntelligenceAskPage />);
    await screen.findByText("saved answer");
    fireEvent.click(await screen.findByTestId("ask-save-note"));
    const unscoped = await screen.findByTestId("ask-note-content");
    expect(unscoped.textContent).not.toMatch(/^scope:/m);
  });
});

describe("SPEC-ADDON-064 the scope the page actually sends", () => {
  it("sends the chip scope with Ask, Enter and retry", async () => {
    await answer([
      { kind: "answer_chunk", delta: "answer one" },
      { kind: "citations", citations: [] },
      { kind: "done", took_ms: 1 },
    ] as unknown as AskStreamEvent[]);
    setUrl("");
    render(<IntelligenceAskPage />);
    await screen.findByRole("button", { name: /Scope/ });

    fireEvent.click(screen.getByRole("button", { name: /Scope/ }));
    fireEvent.click(screen.getByRole("menuitem", { name: "Type" }));
    fireEvent.click(await screen.findByRole("menuitem", { name: "Video" }));
    await screen.findByRole("button", { name: "Remove Type Video" });

    const input = screen.getByLabelText("Question input");
    fireEvent.change(input, { target: { value: "cats" } });
    fireEvent.click(screen.getByTestId("ask-submit"));

    await waitFor(() => expect(askQuestionStream).toHaveBeenCalledTimes(1));
    expect(
      optionsScope(vi.mocked(askQuestionStream).mock.calls[0]),
    ).toMatchObject({ kinds: ["video"] });
    await screen.findByText("answer one");

    fireEvent.keyDown(input, { key: "Enter", shiftKey: false });
    await waitFor(() => expect(askQuestionStream).toHaveBeenCalledTimes(2));
    expect(
      optionsScope(vi.mocked(askQuestionStream).mock.calls[1]),
    ).toMatchObject({ kinds: ["video"] });
    await screen.findByText("answer one");

    stream.error = new LLMRequestError("503 Service Unavailable", 503, {
      code: "scope_unavailable",
      message: "Scope unavailable",
    });
    fireEvent.click(screen.getByTestId("ask-submit"));
    await waitFor(() => expect(askQuestionStream).toHaveBeenCalledTimes(3));
    expect(await screen.findByText("Could not load the scope")).toBeInTheDocument();

    stream.error = null;
    await answer([
      { kind: "answer_chunk", delta: "answer two" },
      { kind: "citations", citations: [] },
      { kind: "done", took_ms: 1 },
    ] as unknown as AskStreamEvent[]);
    fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    await waitFor(() => expect(askQuestionStream).toHaveBeenCalledTimes(4));
    expect(
      optionsScope(vi.mocked(askQuestionStream).mock.calls[3]),
    ).toMatchObject({ kinds: ["video"] });
  });

  it("sends no scope after the last chip is removed", async () => {
    await answer([
      { kind: "answer_chunk", delta: "answer" },
      { kind: "citations", citations: [] },
      { kind: "done", took_ms: 1 },
    ] as unknown as AskStreamEvent[]);
    setUrl("?scope=type%3Avideo");
    render(<IntelligenceAskPage />);

    fireEvent.click(
      await screen.findByRole("button", { name: "Remove Type Video" }),
    );
    const input = screen.getByLabelText("Question input");
    fireEvent.change(input, { target: { value: "cats" } });
    fireEvent.click(screen.getByTestId("ask-submit"));

    await waitFor(() => expect(askQuestionStream).toHaveBeenCalledTimes(1));
    expect(optionsScope(vi.mocked(askQuestionStream).mock.calls[0])).toBeUndefined();
  });

  it("aborts the open stream on a popstate read", async () => {
    let aborted = false;
    vi.mocked(askQuestionStream).mockImplementation(async function* (
      _query: string,
      _drive: string,
      options?: { signal?: AbortSignal },
    ) {
      yield {
        kind: "sources",
        sources: [
          {
            file_id: "f1",
            drive: "family",
            filename: "source.md",
            file_type: "document",
          },
        ],
      } as unknown as AskStreamEvent;
      yield { kind: "answer_chunk", delta: "partial answer" } as AskStreamEvent;
      await new Promise<void>((_resolve, reject) => {
        const signal = options?.signal;
        const abort = () => {
          aborted = true;
          reject(new DOMException("Aborted", "AbortError"));
        };
        if (!signal) {
          reject(new Error("the page asked without an abort signal"));
          return;
        }
        if (signal.aborted) abort();
        else signal.addEventListener("abort", abort);
      });
    });
    setUrl("?q=cats&scope=path%3AWork");
    render(<IntelligenceAskPage />);
    expect(await screen.findByText("partial answer")).toBeInTheDocument();

    act(() => {
      setUrl("?scope=path%3AWork");
      window.dispatchEvent(new PopStateEvent("popstate"));
    });

    await waitFor(() => expect(aborted).toBe(true));
    await waitFor(() =>
      expect(screen.queryByText("partial answer")).not.toBeInTheDocument(),
    );
    expect(
      screen.getByRole("button", { name: "Remove Folder Work" }),
    ).toBeInTheDocument();
  });
});
