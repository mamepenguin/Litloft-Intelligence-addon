/**
 * SPEC-ADDON-110 steps 3-5, 7, 8; I7, I9, I10 (Ask half); SPEC-ADDON-054 I6/I7 as
 * revised: Smart Collection chips on the Ask page.
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

const evaluation = vi.hoisted(() => ({ fn: vi.fn() }));
vi.mock("@/lib/smartCollectionEval", () => ({
  evaluateSmartCollection: (...args: unknown[]) => evaluation.fn(...args),
}));

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

type Held = { promise: Promise<Response>; release: (r: Response) => void };
function hold(): Held {
  let release: (r: Response) => void = () => {};
  const promise = new Promise<Response>((r) => {
    release = r;
  });
  return { promise, release };
}

const lists = {
  collections: [] as Array<Record<string, unknown>>,
  smartFolders: [] as Array<Record<string, unknown>>,
  collectionsStatus: 200,
  smartFoldersStatus: 200,
  collectionsHeld: null as Held | null,
  smartFoldersHeld: null as Held | null,
};

const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
  const url = new URL(String(input), "http://localhost");
  if (url.pathname === "/api/drives/family/files") {
    return json({ files: [], meta: { total: 7 } });
  }
  if (url.pathname.endsWith("/smart-folders")) {
    if (lists.smartFoldersHeld) return lists.smartFoldersHeld.promise;
    return json(lists.smartFolders, lists.smartFoldersStatus);
  }
  if (url.pathname.endsWith("/collections")) {
    if (lists.collectionsHeld) return lists.collectionsHeld.promise;
    return json(lists.collections, lists.collectionsStatus);
  }
  if (url.pathname.endsWith("/tags")) return json([]);
  if (url.pathname.endsWith("/folders")) return json([]);
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


function storageKeys(): string[] {
  const keys: string[] = [];
  for (let i = 0; i < window.sessionStorage.length; i += 1) {
    const key = window.sessionStorage.key(i);
    if (key !== null) keys.push(key);
  }
  return keys;
}



async function answer(events: AskStreamEvent[]) {
  stream.events = events;
}

beforeEach(() => {
  window.sessionStorage.clear();
  lists.collections = [];
  lists.smartFolders = [];
  lists.collectionsStatus = 200;
  lists.smartFoldersStatus = 200;
  lists.collectionsHeld = null;
  lists.smartFoldersHeld = null;
  evaluation.fn.mockReset();
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

const CACHE_PREFIX = "intelligence-ask-cache:v1";
const TOO_LARGE = "The scope is too large (up to 10,000 files). Narrow it down.";
const UNAVAILABLE = "Could not load the scope";

function collectionRow(name: string, updated = "2026-01-01T00:00:00Z") {
  return {
    id: `c-${name}`,
    name,
    description: null,
    drive: "family",
    item_count: 1,
    first_file_id: null,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: updated,
  };
}

function smartRow(name: string, query = "cats", fileType: string | null = "video") {
  return {
    id: `s-${name}`,
    drive: "family",
    name,
    query,
    file_type: fileType,
    sort_by: null,
    sort_order: null,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: null,
  };
}

const member = (id: string) => ({ id, drive: "family", file_type: "video", title: id });

function members(...ids: string[]) {
  return { complete: true, files: ids.map(member) };
}

function scopeParam(chips: string) {
  return `?scope=${encodeURIComponent(chips)}`;
}

function deferred<T>() {
  let resolve: (v: T) => void = () => {};
  const promise = new Promise<T>((r) => {
    resolve = r;
  });
  return { promise, resolve };
}

const settle = () =>
  act(async () => {
    await new Promise((r) => setTimeout(r, 30));
  });

function answered(text = "answer") {
  return [
    { kind: "answer_chunk", delta: text },
    { kind: "citations", citations: [] },
    { kind: "done", took_ms: 1 },
  ] as unknown as AskStreamEvent[];
}

function sentScope(index = 0) {
  return optionsScope(vi.mocked(askQuestionStream).mock.calls[index]) as Record<string, unknown> | undefined;
}

async function ask(question = "cats") {
  const input = await screen.findByLabelText("Question input");
  fireEvent.change(input, { target: { value: question } });
  fireEvent.click(screen.getByTestId("ask-submit"));
}

describe("SPEC-ADDON-110 step 3 / I7 an in: chip resolves to a collection, then a Smart Collection", () => {
  it("a name that matches a collection case-insensitively is sent in collections, even when a Smart Collection shares it", async () => {
    lists.collections = [collectionRow("trip")];
    lists.smartFolders = [smartRow("Trip", "trip query"), smartRow("Mix", "mix query")];
    evaluation.fn.mockResolvedValue(members("m1"));
    await answer(answered());
    setUrl(`?q=cats&scope=${encodeURIComponent('in:"Trip" in:"Mix"')}`);

    render(<IntelligenceAskPage />);

    await waitFor(() => expect(askQuestionStream).toHaveBeenCalledTimes(1));
    const scope = sentScope()!;
    expect(scope.collections).toEqual(["Trip"]);
    expect(scope.file_ids).toEqual(["m1"]);
    expect(evaluation.fn).toHaveBeenCalledTimes(1);
    expect(evaluation.fn.mock.calls[0][0]).toMatchObject({ query: "mix query" });
  });

  it("a name that matches only a Smart Collection is sent as its members' ids, never in collections", async () => {
    lists.smartFolders = [smartRow("Evening Mix", "kyoto tag:trip", "video")];
    evaluation.fn.mockResolvedValue(members("a", "b"));
    await answer(answered());
    setUrl(`?q=cats&scope=${encodeURIComponent('in:"evening mix"')}`);

    render(<IntelligenceAskPage />);

    await waitFor(() => expect(askQuestionStream).toHaveBeenCalledTimes(1));
    expect(sentScope()).toEqual({ folders: [], collections: [], tags: [], kinds: [], file_ids: ["a", "b"] });
    expect(evaluation.fn.mock.calls[0][0]).toMatchObject({
      drive: "family",
      query: "kyoto tag:trip",
      type: "video",
      sort: "relevance",
      includeSceneClip: false,
    });
  });

  it("a name that matches neither is sent as a collection name, as today", async () => {
    await answer(answered());
    setUrl(`?q=cats&scope=${encodeURIComponent('in:"Ghost"')}`);

    render(<IntelligenceAskPage />);

    await waitFor(() => expect(askQuestionStream).toHaveBeenCalledTimes(1));
    expect(sentScope()!.collections).toEqual(["Ghost"]);
    expect(sentScope()).not.toHaveProperty("file_ids");
    expect(evaluation.fn).not.toHaveBeenCalled();
    expect(fetchMock.mock.calls.some(([u]) => String(u).includes("/smart-folders"))).toBe(true);
  });
});

describe("SPEC-ADDON-110 step 4 the union of Smart Collection members is sent as file_ids", () => {
  it("unions several Smart Collections in first-seen order, beside the other chips", async () => {
    lists.smartFolders = [smartRow("One", "q1"), smartRow("Two", "q2")];
    evaluation.fn.mockImplementation(async (input: { query: string }) =>
      input.query === "q1" ? members("a", "b", "c") : members("c", "d", "a"),
    );
    await answer(answered());
    setUrl(`?q=cats&scope=${encodeURIComponent('in:"One" path:Work in:"Two"')}`);

    render(<IntelligenceAskPage />);

    await waitFor(() => expect(askQuestionStream).toHaveBeenCalledTimes(1));
    expect(sentScope()).toEqual({
      folders: ["Work"],
      collections: [],
      tags: [],
      kinds: [],
      file_ids: ["a", "b", "c", "d"],
    });
  });

  it("sends file_ids: [] for a Smart Collection with no members", async () => {
    lists.smartFolders = [smartRow("Empty")];
    evaluation.fn.mockResolvedValue(members());
    await answer(answered());
    setUrl(`?q=cats&scope=${encodeURIComponent('in:"Empty"')}`);

    render(<IntelligenceAskPage />);

    await waitFor(() => expect(askQuestionStream).toHaveBeenCalledTimes(1));
    expect(sentScope()!.file_ids).toEqual([]);
  });

  it("sends exactly 10,000 ids", async () => {
    lists.smartFolders = [smartRow("Big")];
    evaluation.fn.mockResolvedValue(members(...Array.from({ length: 10_000 }, (_, i) => `f${i}`)));
    await answer(answered());
    setUrl(`?q=cats&scope=${encodeURIComponent('in:"Big"')}`);

    render(<IntelligenceAskPage />);

    await waitFor(() => expect(askQuestionStream).toHaveBeenCalledTimes(1));
    expect((sentScope()!.file_ids as string[]).length).toBe(10_000);
  });

  it("refuses more than 10,000 ids: nothing is sent, the question stays, and the message shows", async () => {
    lists.smartFolders = [smartRow("Huge")];
    evaluation.fn.mockResolvedValue(members(...Array.from({ length: 10_001 }, (_, i) => `f${i}`)));
    setUrl(`?scope=${encodeURIComponent('in:"Huge"')}`);

    render(<IntelligenceAskPage />);
    await ask("cats");

    expect(await screen.findByText(TOO_LARGE)).toBeInTheDocument();
    expect(askQuestionStream).not.toHaveBeenCalled();
    expect(screen.getByLabelText("Question input")).toHaveValue("cats");
  });

  it.each([
    ["throws", () => evaluation.fn.mockRejectedValue(new Error("listing failed"))],
    ["is incomplete", () => evaluation.fn.mockResolvedValue({ complete: false, files: [member("a")] })],
  ])("an evaluation that %s sends nothing, keeps the question, and shows the scope-unavailable text", async (_label, arrange) => {
    lists.smartFolders = [smartRow("Mix")];
    arrange();
    setUrl(`?scope=${encodeURIComponent('in:"Mix"')}`);

    render(<IntelligenceAskPage />);
    await ask("cats");

    expect(await screen.findByText(UNAVAILABLE)).toBeInTheDocument();
    expect(askQuestionStream).not.toHaveBeenCalled();
    expect(screen.getByLabelText("Question input")).toHaveValue("cats");
  });

  it("evaluates anew before every send: the Ask button, Enter and the retry", async () => {
    lists.smartFolders = [smartRow("Mix")];
    evaluation.fn.mockResolvedValueOnce(members("a")).mockResolvedValueOnce(members("b")).mockResolvedValue(members("c"));
    await answer(answered("first"));
    setUrl(`?scope=${encodeURIComponent('in:"Mix"')}`);
    render(<IntelligenceAskPage />);
    const before = () => evaluation.fn.mock.calls.length;

    await ask("cats");
    await waitFor(() => expect(askQuestionStream).toHaveBeenCalledTimes(1));
    const afterAsk = before();
    expect(sentScope(0)!.file_ids).toBeDefined();
    await screen.findByText("first");

    fireEvent.keyDown(screen.getByLabelText("Question input"), { key: "Enter", shiftKey: false });
    await waitFor(() => expect(askQuestionStream).toHaveBeenCalledTimes(2));
    expect(before()).toBe(afterAsk + 1);

    stream.error = new LLMRequestError("503 Service Unavailable", 503, {
      code: "scope_unavailable",
      message: "Scope unavailable",
    });
    fireEvent.click(screen.getByTestId("ask-submit"));
    await waitFor(() => expect(askQuestionStream).toHaveBeenCalledTimes(3));
    stream.error = null;
    const beforeRetry = before();
    fireEvent.click(await screen.findByRole("button", { name: "Retry" }));
    await waitFor(() => expect(askQuestionStream).toHaveBeenCalledTimes(4));
    expect(before()).toBe(beforeRetry + 1);
  });
});

describe("SPEC-ADDON-110 step 4 / SPEC-ADDON-054 I6 while evaluating", () => {
  it("the Ask button and Enter start nothing more", async () => {
    lists.smartFolders = [smartRow("Mix")];
    const pending = deferred<unknown>();
    evaluation.fn.mockReturnValue(pending.promise);
    await answer(answered());
    setUrl(`?scope=${encodeURIComponent('in:"Mix"')}`);
    render(<IntelligenceAskPage />);

    await ask("cats");
    await waitFor(() => expect(evaluation.fn).toHaveBeenCalled());
    const evaluations = evaluation.fn.mock.calls.length;
    fireEvent.click(screen.getByTestId("ask-submit"));
    fireEvent.keyDown(screen.getByLabelText("Question input"), { key: "Enter", shiftKey: false });
    await settle();
    expect(evaluation.fn.mock.calls.length).toBe(evaluations);
    expect(askQuestionStream).not.toHaveBeenCalled();

    await act(async () => pending.resolve(members("a")));
    await waitFor(() => expect(askQuestionStream).toHaveBeenCalledTimes(1));
    await settle();
    expect(askQuestionStream).toHaveBeenCalledTimes(1);
  });

  it("a chip change discards the evaluation: nothing is sent", async () => {
    lists.smartFolders = [smartRow("Mix")];
    const pending = deferred<unknown>();
    evaluation.fn.mockReturnValue(pending.promise);
    setUrl(`?scope=${encodeURIComponent('in:"Mix" path:Work')}`);
    render(<IntelligenceAskPage />);

    await ask("cats");
    await waitFor(() => expect(evaluation.fn).toHaveBeenCalled());
    fireEvent.click(screen.getByRole("button", { name: "Remove Folder Work" }));
    await act(async () => pending.resolve(members("a")));
    await settle();

    expect(askQuestionStream).not.toHaveBeenCalled();
    expect(screen.getByLabelText("Question input")).toHaveValue("cats");
  });
});

describe("SPEC-ADDON-110 step 3 / I7 the two lists gate a scope with an in: chip", () => {
  it.each([
    ["collections", () => (lists.collectionsHeld = hold()), () => lists.collectionsHeld!.release(json([]))],
    ["Smart Collections", () => (lists.smartFoldersHeld = hold()), () => lists.smartFoldersHeld!.release(json([]))],
  ])("while the %s list loads, the URL read waits and sending is disabled", async (_label, arrange, release) => {
    arrange();
    await answer(answered());
    setUrl(`?q=cats&scope=${encodeURIComponent('in:"Ghost"')}`);
    render(<IntelligenceAskPage />);

    await settle();
    expect(askQuestionStream).not.toHaveBeenCalled();
    fireEvent.click(screen.getByTestId("ask-submit"));
    await settle();
    expect(askQuestionStream).not.toHaveBeenCalled();

    await act(async () => release());
    await waitFor(() => expect(askQuestionStream).toHaveBeenCalledTimes(1));
    expect(sentScope()!.collections).toEqual(["Ghost"]);
  });

  it.each([
    ["collections", () => (lists.collectionsStatus = 500)],
    ["Smart Collections", () => (lists.smartFoldersStatus = 500)],
  ])("a failed %s list sends nothing and shows the scope-unavailable text", async (_label, arrange) => {
    arrange();
    setUrl(`?q=cats&scope=${encodeURIComponent('in:"Ghost"')}`);
    render(<IntelligenceAskPage />);

    expect(await screen.findByText(UNAVAILABLE)).toBeInTheDocument();
    await ask("cats");
    await settle();
    expect(askQuestionStream).not.toHaveBeenCalled();
  });

  it("a scope with no in: chip does not wait for the lists; adding one makes the send wait", async () => {
    lists.collectionsHeld = hold();
    lists.smartFoldersHeld = hold();
    await answer(answered("first"));
    setUrl("?q=cats&scope=path%3AWork");
    render(<IntelligenceAskPage />);

    await waitFor(() => expect(askQuestionStream).toHaveBeenCalledTimes(1));
    expect(sentScope()).toMatchObject({ folders: ["Work"] });
    await screen.findByText("first");

    act(() => {
      setUrl(`?q=dogs&scope=${encodeURIComponent('path:Work in:"Ghost"')}`);
      window.dispatchEvent(new PopStateEvent("popstate"));
    });
    await settle();
    expect(askQuestionStream).toHaveBeenCalledTimes(1);

    await act(async () => {
      lists.collectionsHeld!.release(json([]));
      lists.smartFoldersHeld!.release(json([]));
    });
    await waitFor(() => expect(askQuestionStream).toHaveBeenCalledTimes(2));
    expect(sentScope(1)).toMatchObject({ folders: ["Work"], collections: ["Ghost"] });
  });
});

describe("SPEC-ADDON-110 step 5 / I9 Smart Collection answers stay out of the answer cache", () => {
  it("writes nothing under the answer cache, restores from the tab on a URL read without evaluating or asking", async () => {
    lists.smartFolders = [smartRow("Mix")];
    evaluation.fn.mockResolvedValue(members("a"));
    await answer([
      { kind: "scope", scope: { folders: [], collections: [], tags: [], kinds: [] }, matched_file_count: 1 },
      ...answered("kept answer"),
    ] as unknown as AskStreamEvent[]);
    setUrl(`?q=cats&scope=${encodeURIComponent('in:"Mix"')}`);
    render(<IntelligenceAskPage />);
    await screen.findByText("kept answer");

    expect(
      storageKeys().filter((key) => key.startsWith(`${CACHE_PREFIX}:`) && key.endsWith(":cats")),
    ).toEqual([]);

    cleanup();
    vi.mocked(askQuestionStream).mockClear();
    evaluation.fn.mockClear();
    await answer([]);
    render(<IntelligenceAskPage />);

    expect(await screen.findByText("kept answer")).toBeInTheDocument();
    expect(askQuestionStream).not.toHaveBeenCalled();
    expect(evaluation.fn).not.toHaveBeenCalled();
  });

  it("restores on popstate too, and the Ask button evaluates and asks again", async () => {
    lists.smartFolders = [smartRow("Mix")];
    evaluation.fn.mockResolvedValue(members("a"));
    await answer(answered("kept answer"));
    setUrl(`?q=cats&scope=${encodeURIComponent('in:"Mix"')}`);
    render(<IntelligenceAskPage />);
    await screen.findByText("kept answer");

    act(() => {
      setUrl("?q=dogs");
      window.dispatchEvent(new PopStateEvent("popstate"));
    });
    await waitFor(() => expect(askQuestionStream).toHaveBeenCalledTimes(2));

    vi.mocked(askQuestionStream).mockClear();
    evaluation.fn.mockClear();
    await answer(answered("fresh answer"));
    act(() => {
      setUrl(`?q=cats&scope=${encodeURIComponent('in:"Mix"')}`);
      window.dispatchEvent(new PopStateEvent("popstate"));
    });
    expect(await screen.findByText("kept answer")).toBeInTheDocument();
    await settle();
    expect(askQuestionStream).not.toHaveBeenCalled();
    expect(evaluation.fn).not.toHaveBeenCalled();

    fireEvent.click(screen.getByTestId("ask-submit"));
    await waitFor(() => expect(askQuestionStream).toHaveBeenCalledTimes(1));
    expect(evaluation.fn).toHaveBeenCalledTimes(1);
    expect(await screen.findByText("fresh answer")).toBeInTheDocument();
  });
});

describe("SPEC-ADDON-110 step 8 / SPEC-ADDON-054 I7 the scope line", () => {
  it("shows the size of the union for a scope of Smart Collection chips only, then the scope event's count after an answer", async () => {
    lists.smartFolders = [smartRow("One", "q1"), smartRow("Two", "q2")];
    evaluation.fn.mockImplementation(async (input: { query: string }) =>
      input.query === "q1" ? members("a", "b") : members("b", "c"),
    );
    await answer([
      { kind: "scope", scope: { folders: [], collections: [], tags: [], kinds: [] }, matched_file_count: 1 },
      ...answered("scoped"),
    ] as unknown as AskStreamEvent[]);
    setUrl(scopeParam('in:"One" in:"Two"'));
    render(<IntelligenceAskPage />);

    await waitFor(() => expect(screen.getByTestId("drive-scope")).toHaveTextContent(/^family · 3\b/));

    await ask("cats");
    await screen.findByText("scoped");
    await waitFor(() => expect(screen.getByTestId("drive-scope")).toHaveTextContent(/^family · 1\b/));
  });

  it("shows the drive alone when a Smart Collection chip sits with another chip", async () => {
    lists.smartFolders = [smartRow("One")];
    evaluation.fn.mockResolvedValue(members("a", "b"));
    setUrl(scopeParam('in:"One" path:Work'));
    render(<IntelligenceAskPage />);

    await screen.findByRole("button", { name: "Remove Folder Work" });
    await settle();
    await waitFor(() => expect(screen.getByTestId("drive-scope").textContent?.trim()).toBe("family"));
  });

});

describe("SPEC-ADDON-110 step 7 / I10 ＋ 範囲 → Collection lists both kinds", () => {
  const openCollections = async () => {
    fireEvent.click(await screen.findByRole("button", { name: /Scope/ }));
    fireEvent.click(screen.getByRole("menuitem", { name: "Collection" }));
  };
  const rows = () => screen.queryAllByRole("menuitem").map((row) => row.textContent?.trim() ?? "");

  it("lists collections and Smart Collections together in numeric collator order, each with its icon", async () => {
    lists.collections = [collectionRow("b10", "2026-03-01T00:00:00Z"), collectionRow("Alpha", "2026-02-01T00:00:00Z")];
    lists.smartFolders = [smartRow("B2"), smartRow("alpha 2")];
    render(<IntelligenceAskPage />);
    await openCollections();

    await waitFor(() => expect(rows()).toHaveLength(4));
    expect(rows()).toEqual(["Alpha", "alpha 2", "B2", "b10"]);
    const icon = (name: string) =>
      screen.getByRole("menuitem", { name }).querySelector("svg")?.getAttribute("class") ?? "";
    expect(icon("Alpha")).toContain("lucide-library");
    expect(icon("B2")).toContain("lucide-list-filter");
  });

  it("choosing a Smart Collection adds its in: chip", async () => {
    lists.smartFolders = [smartRow("Evening Mix")];
    render(<IntelligenceAskPage />);
    await openCollections();
    fireEvent.click(await screen.findByRole("menuitem", { name: "Evening Mix" }));

    const url = new URL(window.location.href);
    expect(url.searchParams.get("scope")).toBe('in:"Evening Mix"');
  });

  it("leaves out chosen names and names holding a quote, for both kinds", async () => {
    lists.collections = [collectionRow("Chosen"), collectionRow('Say "hi"')];
    lists.smartFolders = [smartRow("Kept"), smartRow("Wide＂quote"), smartRow("chosen")];
    setUrl(scopeParam('in:"Chosen"'));
    render(<IntelligenceAskPage />);
    await openCollections();

    await waitFor(() => expect(rows()).toContain("Kept"));
    expect(rows()).toEqual(["Kept"]);
  });

  it("groups typed text prefix-first, then substring, each in collator order", async () => {
    lists.collections = [collectionRow("Ecotrip"), collectionRow("Travel")];
    lists.smartFolders = [smartRow("trip"), smartRow("Retro")];
    render(<IntelligenceAskPage />);
    await openCollections();
    await waitFor(() => expect(rows().length).toBe(4));

    fireEvent.change(screen.getByLabelText("Filter collections"), { target: { value: "tr" } });

    await waitFor(() => expect(rows()).toEqual(["Travel", "trip", "Ecotrip", "Retro"]));
  });

  it("caps the whole list at today's option limit", async () => {
    lists.collections = Array.from({ length: 6 }, (_, i) => collectionRow(`c${i}`));
    lists.smartFolders = Array.from({ length: 6 }, (_, i) => smartRow(`s${i}`));
    render(<IntelligenceAskPage />);
    await openCollections();

    await waitFor(() => expect(rows().some((r) => r.startsWith("s"))).toBe(true));
    expect(rows()).toHaveLength(8);
  });

  it("shows the loading row while either list loads", async () => {
    lists.collections = [collectionRow("Alpha")];
    lists.smartFoldersHeld = hold();
    render(<IntelligenceAskPage />);
    await openCollections();

    expect(await screen.findByText("Loading collections…")).toBeInTheDocument();
    await act(async () => lists.smartFoldersHeld!.release(json([smartRow("Beta")])));
    await waitFor(() => expect(rows()).toEqual(["Alpha", "Beta"]));
  });

  it("when one list fails, shows the other's rows with the failure row under them", async () => {
    lists.collections = [collectionRow("Alpha")];
    lists.smartFoldersStatus = 500;
    render(<IntelligenceAskPage />);
    await openCollections();

    await waitFor(() => expect(rows()).toContain("Alpha"));
    const failure = await screen.findByText("Could not load collections");
    const alpha = screen.getByRole("menuitem", { name: "Alpha" });
    expect(alpha.compareDocumentPosition(failure) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it("counts chips of both kinds toward the ten in: chips", async () => {
    lists.collections = Array.from({ length: 5 }, (_, i) => collectionRow(`c${i}`));
    lists.smartFolders = Array.from({ length: 5 }, (_, i) => smartRow(`s${i}`));
    lists.smartFolders.push(smartRow("extra"));
    const chips = [...Array.from({ length: 5 }, (_, i) => `in:c${i}`), ...Array.from({ length: 5 }, (_, i) => `in:s${i}`)];
    setUrl(scopeParam(chips.join(" ")));
    render(<IntelligenceAskPage />);

    fireEvent.click(await screen.findByRole("button", { name: /Scope/ }));
    expect(screen.getByRole("menuitem", { name: "Collection" })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: /Scope/ }));

    fireEvent.click(screen.getByRole("button", { name: "Remove Collection s4" }));
    fireEvent.click(await screen.findByRole("button", { name: /Scope/ }));
    fireEvent.click(screen.getByRole("menuitem", { name: "Collection" }));
    await waitFor(() => expect(rows()).toEqual(["extra", "s4"]));
  });
});
