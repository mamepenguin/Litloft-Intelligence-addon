import { describe, it, expect, vi, afterEach } from "vitest";

import {
  askQuestionStream,
  fetchLLMChoices,
  isProfileUnavailable,
  regenerateDetailedSummary,
  regenerateSummary,
} from "./api";
import { choiceOffered } from "./llmChoice";

function okResponse(body: unknown = { status: "accepted", message: "" }) {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

function errorResponse(status: number, detail: unknown) {
  return new Response(JSON.stringify({ detail }), {
    status,
    statusText: "Bad Request",
    headers: { "Content-Type": "application/json" },
  });
}

function mockFetch(response: () => Response) {
  const fn = vi.fn(async () => response());
  vi.stubGlobal("fetch", fn);
  return fn;
}

function sentBody(fn: ReturnType<typeof mockFetch>): unknown {
  const init = (fn.mock.calls[0] as unknown as [string, RequestInit])[1];
  return init.body === undefined ? undefined : JSON.parse(init.body as string);
}

function sentUrl(fn: ReturnType<typeof mockFetch>): string {
  return (fn.mock.calls[0] as unknown as [string, RequestInit])[0];
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("choiceOffered", () => {
  const choice = (name: string) => ({ name, model: `${name}-model`, offhost: false });

  it.each([
    { label: "not loaded or failed", value: null, want: false },
    { label: "0 choices, auto set", value: { auto: "a", choices: [] }, want: false },
    { label: "0 choices, auto null", value: { auto: null, choices: [] }, want: false },
    { label: "1 choice, auto set", value: { auto: "a", choices: [choice("a")] }, want: false },
    { label: "1 choice, auto null", value: { auto: null, choices: [choice("a")] }, want: true },
    {
      label: "2 choices, auto set",
      value: { auto: "a", choices: [choice("a"), choice("b")] },
      want: true,
    },
    {
      label: "2 choices, auto null",
      value: { auto: null, choices: [choice("a"), choice("b")] },
      want: true,
    },
  ])("$label → $want", ({ value, want }) => {
    expect(choiceOffered(value)).toBe(want);
  });
});

describe("request bodies name a profile only when one was chosen", () => {
  it("summary regenerate sends no body for Auto", async () => {
    const fn = mockFetch(() => okResponse());
    await regenerateSummary("f1", "d1");
    expect(sentBody(fn)).toBeUndefined();
  });

  it("summary regenerate sends the chosen profile", async () => {
    const fn = mockFetch(() => okResponse());
    await regenerateSummary("f1", "d1", "cloud");
    expect(sentBody(fn)).toEqual({ profile: "cloud" });
  });

  it("detailed regenerate has no profile key for Auto", async () => {
    const fn = mockFetch(() => okResponse());
    await regenerateDetailedSummary("f1", "d1", { force: false });
    expect(sentBody(fn)).toStrictEqual({ force: false });
  });

  it("detailed regenerate sends force and the chosen profile", async () => {
    const fn = mockFetch(() => okResponse());
    await regenerateDetailedSummary("f1", "d1", { force: true, profile: "cloud" });
    expect(sentBody(fn)).toStrictEqual({ force: true, profile: "cloud" });
  });

  it("Ask has no profile key for Auto", async () => {
    const fn = mockFetch(() => new Response("", { status: 200 }));
    for await (const _ of askQuestionStream("what happened", "d1")) {
      void _;
    }
    expect(sentBody(fn)).toStrictEqual({ query: "what happened" });
  });

  it("Ask sends the chosen profile", async () => {
    const fn = mockFetch(() => new Response("", { status: 200 }));
    for await (const _ of askQuestionStream("what happened", "d1", { profile: "cloud" })) {
      void _;
    }
    expect(sentBody(fn)).toStrictEqual({ query: "what happened", profile: "cloud" });
  });
});

describe("profile_unavailable is told apart from other errors", () => {
  it.each([
    { label: "summary", call: () => regenerateSummary("f1", "d1", "cloud") },
    {
      label: "detailed summary",
      call: () => regenerateDetailedSummary("f1", "d1", { profile: "cloud" }),
    },
    {
      label: "Ask",
      call: async () => {
        for await (const _ of askQuestionStream("what happened", "d1", { profile: "cloud" })) {
          void _;
        }
      },
    },
  ])("$label: 400 profile_unavailable is recognised", async ({ call }) => {
    mockFetch(() => errorResponse(400, "profile_unavailable"));
    const err = await call().then(
      () => null,
      (e: unknown) => e,
    );
    expect(isProfileUnavailable(err)).toBe(true);
  });

  it.each([
    { label: "another 400 detail", status: 400, detail: "insufficient_content" },
    { label: "the same detail on another status", status: 503, detail: "profile_unavailable" },
  ])("$label is not", async ({ status, detail }) => {
    mockFetch(() => errorResponse(status, detail));
    const err = await regenerateSummary("f1", "d1", "cloud").then(
      () => null,
      (e: unknown) => e,
    );
    expect(err).toBeInstanceOf(Error);
    expect(isProfileUnavailable(err)).toBe(false);
  });

  it("Ask keeps the server's detail as its message", async () => {
    mockFetch(() => errorResponse(400, "RAG feature is disabled"));
    const err = await (async () => {
      for await (const _ of askQuestionStream("what happened", "d1")) {
        void _;
      }
    })().then(
      () => null,
      (e: unknown) => e,
    );
    expect((err as Error).message).toBe("RAG feature is disabled");
  });
});

describe("fetchLLMChoices", () => {
  it.each([
    ["summaries", "/api/addons/intelligence/llm/choices/summaries"],
    ["detailed_summaries", "/api/addons/intelligence/llm/choices/detailed_summaries"],
    ["rag", "/api/addons/intelligence/llm/choices/rag"],
  ] as const)("%s reads its own path with the drive header", async (feature, url) => {
    const fn = mockFetch(() => okResponse({ auto: null, choices: [] }));
    await fetchLLMChoices(feature, "ドライブ");
    expect(sentUrl(fn)).toBe(url);
    const init = (fn.mock.calls[0] as unknown as [string, RequestInit])[1];
    expect(init.headers).toEqual({ "X-Lit-Drive": encodeURIComponent("ドライブ") });
  });
});
