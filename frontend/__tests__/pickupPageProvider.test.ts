/**
 * SPEC-ADDON-094: Intelligence pages the video app's Recommended See all through
 * `slotPageProviders.pickup`, answering the whole restricted feed at once.
 *
 * The feed is served from a stubbed `fetch` so the list provider and the page provider read
 * the same rows through whichever client function each uses.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

import * as slots from "@/addons/intelligence/slots";

type Restrict = (ids: string[]) => Promise<string[]>;
type ListProvider = (args: { drive: string; limit: number; restrict: Restrict }) => Promise<string[]>;
type PageArgs = { drive: string; locale: string; limit: number; cursor: string | null; restrict: Restrict };
type PageAnswer = { items: unknown[]; cursor: unknown };
type PageProvider = (args: PageArgs) => Promise<PageAnswer>;

const DRIVE = "my videos";
const originalFetch = globalThis.fetch;

function pageProvider(): PageProvider {
  const providers = (slots as { slotPageProviders?: Record<string, PageProvider> }).slotPageProviders;
  const provider = providers?.pickup;
  expect(typeof provider).toBe("function");
  return provider!;
}

function listProvider(): ListProvider {
  const providers = (slots as { slotProviders?: Record<string, ListProvider> }).slotProviders;
  return providers!.pickup;
}

const feedOf = (n: number) => Array.from({ length: n }, (_, i) => `f${String(i).padStart(3, "0")}`);
const index = (id: string) => Number(id.slice(1));

/** Requests and restrict calls, in the order they happened. */
let events: string[] = [];

type Reply = { ok: boolean; status?: number; body?: unknown } | Error;

/**
 * Serves `rows` in rank order by the request's offset and limit; `total` defaults to their
 * count. `override` replaces the reply for the request at a given offset.
 */
function serveFeed(rows: string[], total: number = rows.length, override: Record<number, Reply> = {}) {
  const fetchMock = vi.fn(async (input: RequestInfo | URL, _init?: RequestInit) => {
    const url = new URL(String(input), "http://localhost");
    const offset = Number(url.searchParams.get("offset") ?? 0);
    const limit = Number(url.searchParams.get("limit") ?? 12);
    events.push(`fetch:${offset}`);
    const reply: Reply = override[offset] ?? { ok: true, body: { file_ids: rows.slice(offset, offset + limit), total } };
    if (reply instanceof Error) throw reply;
    return {
      ok: reply.ok,
      status: reply.status ?? (reply.ok ? 200 : 500),
      json: async () => reply.body,
    } as Response;
  });
  globalThis.fetch = fetchMock as never;
  return fetchMock;
}

function accepting(pred: (id: string) => boolean) {
  return vi.fn<Restrict>(async (ids) => {
    events.push(`restrict:${ids.length}`);
    return ids.filter(pred);
  });
}

const fetchCalls = () => vi.mocked(globalThis.fetch).mock.calls;
const urls = () => fetchCalls().map(([input]) => new URL(String(input), "http://localhost"));
const offsetsAsked = () => urls().map((u) => Number(u.searchParams.get("offset") ?? 0));

const ask = (restrict: Restrict, over: Partial<PageArgs> = {}) =>
  pageProvider()({ drive: DRIVE, locale: "en", limit: 60, cursor: null, restrict, ...over });

const idsOf = (answer: PageAnswer) => answer.items.map((i) => (i as { id: string }).id);

const dedupe = (ids: string[]) => [...new Set(ids)];

beforeEach(() => {
  events = [];
  vi.useFakeTimers({ toFake: ["Date"] });
  vi.setSystemTime(new Date(2026, 10, 12, 12, 0, 0));
});

afterEach(() => {
  vi.useRealTimers();
  globalThis.fetch = originalFetch;
});

describe("SPEC-ADDON-094 the page provider is exported", () => {
  it("SPEC-ADDON-094 slots.ts exports slotPageProviders.pickup as a function", () => {
    pageProvider();
  });
});

describe("SPEC-ADDON-094 I7 reading the feed", () => {
  it("SPEC-ADDON-094 I7 reads the drive's Pickup feed 60 at a time in rank order, with no window, at most 5 requests", async () => {
    serveFeed(feedOf(300));

    await ask(accepting(() => true));

    expect(offsetsAsked()).toEqual([0, 60, 120, 180, 240]);
    for (const u of urls()) {
      expect(u.pathname).toBe("/api/addons/intelligence/pickup");
      expect(u.searchParams.get("limit")).toBe("60");
      expect(u.searchParams.has("window")).toBe(false);
      expect(u.searchParams.has("date")).toBe(false);
    }
    for (const [, init] of fetchCalls()) {
      expect(new Headers(init?.headers).get("X-Lit-Drive")).toBe(encodeURIComponent(DRIVE));
    }
  });

  it("SPEC-ADDON-094 I7 never sends more than 5 feed requests, even when the feed claims more rows", async () => {
    serveFeed(feedOf(600));

    await ask(accepting(() => true));

    expect(offsetsAsked()).toEqual([0, 60, 120, 180, 240]);
  });

  it("SPEC-ADDON-094 I7 stops at a page shorter than 60", async () => {
    serveFeed(feedOf(70), 300);

    const answer = await ask(accepting(() => true));

    expect(offsetsAsked()).toEqual([0, 60]);
    expect(idsOf(answer)).toHaveLength(70);
  });

  it("SPEC-ADDON-094 I7 stops when the offset reaches total", async () => {
    serveFeed(feedOf(120));

    await ask(accepting(() => true));

    expect(offsetsAsked()).toEqual([0, 60]);
  });

  it("SPEC-ADDON-094 restrict is called once, after the last feed read, with every id read in rank order", async () => {
    const feed = feedOf(150);
    serveFeed(feed);
    const restrict = accepting(() => true);

    await ask(restrict);

    expect(restrict).toHaveBeenCalledTimes(1);
    expect(restrict.mock.calls[0][0]).toEqual(feed);
    expect(events).toEqual(["fetch:0", "fetch:60", "fetch:120", "restrict:150"]);
  });
});

describe("SPEC-ADDON-094 I6 the answer", () => {
  it("SPEC-ADDON-094 I7 answers with cursor null and items of the shape { id }, with no byline", async () => {
    serveFeed(feedOf(10));

    const answer = await ask(accepting(() => true));

    expect(answer.cursor).toBeNull();
    expect(answer.items).toHaveLength(10);
    for (const it of answer.items) {
      expect(Object.keys(it as object)).toEqual(["id"]);
    }
  });

  it("SPEC-ADDON-094 I6 holds every accepted id of the feed exactly once, and nothing restrict refused", async () => {
    const feed = feedOf(300);
    serveFeed(feed);
    const accepted = (id: string) => index(id) % 3 !== 1;

    const ids = idsOf(await ask(accepting(accepted)));

    expect(new Set(ids).size).toBe(ids.length);
    expect([...ids].sort()).toEqual(feed.filter(accepted));
  });

  it("SPEC-ADDON-094 I6 the first 40 are the first 40 accepted ids, and the rest follow in rank order", async () => {
    const feed = feedOf(300);
    serveFeed(feed);
    const accepted = feed.filter((id) => index(id) % 2 === 0);

    const ids = idsOf(await ask(accepting((id) => index(id) % 2 === 0)));

    expect([...ids.slice(0, 40)].sort()).toEqual(accepted.slice(0, 40));
    expect(ids.slice(40)).toEqual(accepted.slice(40));
  });

  it("SPEC-ADDON-094 I6 what Discover draws from the list provider is the start of the page, in the same order", async () => {
    serveFeed(feedOf(300));
    const accepted = (id: string) => index(id) % 2 === 1;

    const listed = await listProvider()({ drive: DRIVE, limit: 12, restrict: accepting(accepted) });
    const paged = idsOf(await ask(accepting(accepted)));

    const drawn = dedupe(listed);
    expect(drawn.length).toBeGreaterThan(0);
    expect(paged.slice(0, drawn.length)).toEqual(drawn);
  });

  it("SPEC-ADDON-094 I6 the list provider's whole sample of 40 is the first 40 of the page", async () => {
    serveFeed(feedOf(300));
    const accepted = (id: string) => index(id) % 4 !== 0;

    const listed = await listProvider()({ drive: DRIVE, limit: 40, restrict: accepting(accepted) });
    const paged = idsOf(await ask(accepting(accepted)));

    expect(listed).toHaveLength(40);
    expect(paged.slice(0, 40)).toEqual(listed);
  });

  it("SPEC-ADDON-094 I6 a row repeated by a rebuild between feed reads appears once, at its first position, and Discover's shelf is still the start of the page", async () => {
    // Page 1 accepts fewer than 40, page 2 repeats an id page 1 already gave: the sample is
    // taken before repeats are dropped, on both sides.
    const first = feedOf(60);
    const second = [first[0], first[2], ...feedOf(120).slice(60, 118)];
    const rows = [...first, ...second];
    serveFeed(rows, 120);
    const accepted = (id: string) => index(id) % 2 === 0;

    const listed = await listProvider()({ drive: DRIVE, limit: 12, restrict: accepting(accepted) });
    const ids = idsOf(await ask(accepting(accepted)));

    expect(new Set(ids).size).toBe(ids.length);
    expect([...ids].sort()).toEqual(dedupe(rows.filter(accepted)).sort());
    const drawn = dedupe(listed);
    expect(ids.slice(0, drawn.length)).toEqual(drawn);
  });

  it("SPEC-ADDON-094 I6 the same feed, accepted set, drive and date give the same answer", async () => {
    serveFeed(feedOf(300));
    const accepted = (id: string) => index(id) % 2 === 0;

    const one = idsOf(await ask(accepting(accepted)));
    const two = idsOf(await ask(accepting(accepted)));

    expect(two).toEqual(one);
  });

  it("SPEC-ADDON-094 the answer is the whole feed whatever limit is asked", async () => {
    serveFeed(feedOf(300));

    const one = await ask(accepting(() => true), { limit: 1 });
    const sixty = await ask(accepting(() => true), { limit: 60 });

    expect(idsOf(one)).toHaveLength(300);
    expect(idsOf(one)).toEqual(idsOf(sixty));
    expect(one.cursor).toBeNull();
  });

  it("SPEC-ADDON-094 an empty feed answers no items and cursor null", async () => {
    serveFeed([]);

    expect(await ask(accepting(() => true))).toEqual({ items: [], cursor: null });
  });

  it("SPEC-ADDON-094 a feed restrict refuses entirely answers no items and cursor null", async () => {
    serveFeed(feedOf(90));

    expect(await ask(accepting(() => false))).toEqual({ items: [], cursor: null });
  });

  it.each([["a string", "c1"], ["an empty string", ""]])(
    "SPEC-ADDON-094 I7 a cursor that is %s answers no items and cursor null without a request",
    async (_name, cursor) => {
      serveFeed(feedOf(300));
      const restrict = accepting(() => true);

      expect(await ask(restrict, { cursor })).toEqual({ items: [], cursor: null });
      expect(fetchCalls()).toHaveLength(0);
      expect(restrict).not.toHaveBeenCalled();
    },
  );
});

describe("SPEC-ADDON-094 I7 failures reject", () => {
  it.each([
    ["the first read fails on the network", { 0: new Error("offline") }],
    ["the first read is not OK", { 0: { ok: false, status: 503 } }],
    ["a later read is not OK after earlier ones succeeded", { 120: { ok: false, status: 500 } }],
    ["a later read fails on the network", { 60: new Error("offline") }],
    ["a body's file_ids is not an array", { 0: { ok: true, body: { file_ids: "f000", total: 300 } } }],
    ["a body has no total", { 60: { ok: true, body: { file_ids: feedOf(120).slice(60) } } }],
    ["a body's total is a string", { 0: { ok: true, body: { file_ids: feedOf(60), total: "300" } } }],
  ] as [string, Record<number, Reply>][])("SPEC-ADDON-094 I7 rejects when %s, and never restricts", async (_name, override) => {
    serveFeed(feedOf(300), 300, override);
    const restrict = accepting(() => true);

    await expect(ask(restrict)).rejects.toBeDefined();
    expect(restrict).not.toHaveBeenCalled();
  });

  it("SPEC-ADDON-094 I7 rejects when restrict rejects", async () => {
    serveFeed(feedOf(300));
    const restrict = vi.fn<Restrict>(async () => {
      throw new Error("restrict failed");
    });

    await expect(ask(restrict)).rejects.toThrow();
  });
});
