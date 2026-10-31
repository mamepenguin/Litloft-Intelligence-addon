/**
 * SPEC-ADDON-042: the client for GET /search/scene-segments.
 */

import { describe, it, expect, vi, afterEach } from "vitest";

import { fetchSceneSegments } from "./api";

const BODY = {
  query: "cats",
  segments: [
    { file_id: "f1", filename: "a.mp4", start: 10, end: 40, peak: 20, score: 0.31 },
  ],
  partial: false,
};

function stubFetch(response: Response | Error) {
  const fn = vi.fn(async (_url: string, _init?: RequestInit) => {
    if (response instanceof Error) throw response;
    return response;
  });
  vi.stubGlobal("fetch", fn);
  return fn;
}

const json = (status: number, body: unknown = BODY) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("fetchSceneSegments (SPEC-ADDON-042)", () => {
  it("resolves to the response body on 200", async () => {
    stubFetch(json(200));
    await expect(fetchSceneSegments("cats", "main")).resolves.toEqual(BODY);
  });

  it("calls the intelligence scene-segments route with the query and the drive header", async () => {
    const fetchMock = stubFetch(json(200));
    await fetchSceneSegments("cats & dogs", "ライブラリ");

    const [url, init] = fetchMock.mock.calls[0];
    const parsed = new URL(url, "http://localhost");
    expect(parsed.pathname).toBe("/api/addons/intelligence/search/scene-segments");
    expect(parsed.searchParams.get("q")).toBe("cats & dogs");
    const headers = new Headers(init?.headers);
    expect(headers.get("X-Lit-Drive")).toBe(encodeURIComponent("ライブラリ"));
  });

  it("passes the abort signal to fetch", async () => {
    const fetchMock = stubFetch(json(200));
    const controller = new AbortController();
    await fetchSceneSegments("cats", "main", controller.signal);

    const init = fetchMock.mock.calls[0][1];
    expect(init?.signal).toBe(controller.signal);
  });

  it("resolves to null on 404, the policy-off answer", async () => {
    stubFetch(json(404, { detail: "Not found" }));
    await expect(fetchSceneSegments("cats", "main")).resolves.toBeNull();
  });

  it.each([400, 403, 422, 500, 502, 503])("rejects on %i", async (status) => {
    stubFetch(json(status, { detail: "x" }));
    await expect(fetchSceneSegments("cats", "main")).rejects.toBeDefined();
  });

  it("rejects when the network fails", async () => {
    stubFetch(new TypeError("network"));
    await expect(fetchSceneSegments("cats", "main")).rejects.toBeDefined();
  });
});
