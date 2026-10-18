/**
 * SPEC-ADDON-024: `getSimilarFiles(fileId, drive, { limit?, signal? })`.
 */

import { describe, it, expect, vi, afterEach } from "vitest";

import { getSimilarFiles, SIMILAR_FILES_LIMIT } from "./api";

function mockFetchJson(body: unknown) {
  const fn = vi.fn(async (_url: unknown, _init?: RequestInit) => ({
    ok: true,
    status: 200,
    statusText: "OK",
    headers: new Headers({ "content-type": "application/json" }),
    json: async () => body,
    text: async () => JSON.stringify(body),
  }));
  vi.stubGlobal("fetch", fn);
  return fn;
}

function lastCall(fn: ReturnType<typeof mockFetchJson>) {
  expect(fn).toHaveBeenCalledTimes(1);
  const [url, init] = fn.mock.calls[0] as [unknown, RequestInit | undefined];
  return {
    url: String(url),
    headers: new Headers(init?.headers),
    signal: init?.signal,
  };
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("getSimilarFiles (SPEC-ADDON-024)", () => {
  it("SPEC-ADDON-024 item 2: passes the caller's signal to fetch, with the drive header and the default limit", async () => {
    const fetchMock = mockFetchJson({ results: [] });
    const controller = new AbortController();

    await getSimilarFiles("f1", "media", { signal: controller.signal });

    const call = lastCall(fetchMock);
    expect(call.signal).toBe(controller.signal);
    expect(call.headers.get("X-Lit-Drive")).toBe("media");
    const url = new URL(call.url, "http://host");
    expect(url.pathname).toBe("/api/addons/intelligence/similar/f1");
    expect(url.searchParams.get("limit")).toBe(String(SIMILAR_FILES_LIMIT));
  });

  it("SPEC-ADDON-024 touch points: limit overrides the default", async () => {
    const fetchMock = mockFetchJson({ results: [] });

    await getSimilarFiles("f1", "media", { limit: 3 });

    const url = new URL(lastCall(fetchMock).url, "http://host");
    expect(url.searchParams.get("limit")).toBe("3");
  });
});
