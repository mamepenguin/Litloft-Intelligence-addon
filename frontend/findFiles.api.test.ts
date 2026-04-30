/**
 * Unit tests for the new ``findFiles`` API client (Phase 4 RED).
 *
 * Spec: ``docs/superpowers/specs/2026-04-30-intelligence-find-mode.md``
 * §3.2 — POST /api/addons/intelligence/find with required X-Lit-Drive
 * header. The viewer-id is injected by the host addon_proxy from the
 * ``lit_viewer`` cookie (same as /ask); the frontend MUST NOT read the
 * cookie or send X-Lit-Viewer-Id directly — the proxy strips
 * client-supplied values to prevent forgery.
 *
 * These tests are intentionally written against a function that does
 * not exist yet (``findFiles`` is missing from ``./api``). They will
 * fail with an import error during RED. Phase 4 implementation lands
 * in ``api.ts``.
 */

import { describe, it, expect, vi, afterEach } from "vitest";

// Importing the (yet-unimplemented) export pins the public contract.
import { findFiles, type FindResponse } from "./api";

const sampleResponse: FindResponse = {
  decomposed: {
    time_range: {
      kind: "relative",
      value: "last_week",
      after: "2026-04-23T00:00:00Z",
      before: "2026-04-30T00:00:00Z",
    },
    personal_scope: "viewed",
    file_type_hint: "video",
    semantic_query: "SF",
    category_expansion: ["science fiction", "宇宙船", "ロボット", "ディストピア"],
  },
  results: [
    {
      file_id: "f-abc123",
      score: 0.82,
      hit: {
        kind: "transcript",
        location: { start_seconds: 415.2, end_seconds: 460.0 },
        text: "...宇宙船が時空を超えて...",
      },
      file: {
        name: "Interstellar.mp4",
        file_type: "video",
        thumbnail_url: "/api/files/f-abc123/thumbnail",
        viewed_at: "2026-04-27T19:30:00Z",
      },
    },
  ],
  total: 8,
  limit: 20,
};

function mockFetchJson(body: unknown, status = 200): ReturnType<typeof vi.fn> {
  const fn = vi.fn(async () => ({
    ok: status >= 200 && status < 300,
    status,
    statusText: status === 200 ? "OK" : "Error",
    headers: new Headers({ "content-type": "application/json" }),
    json: async () => body,
    text: async () => JSON.stringify(body),
  }));
  return fn as unknown as ReturnType<typeof vi.fn>;
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("findFiles — request shape", () => {
  it("issues POST to /api/addons/intelligence/find with JSON body", async () => {
    const fetchMock = mockFetchJson(sampleResponse);
    vi.stubGlobal("fetch", fetchMock);

    const out = await findFiles("SF っぽいの", "family", { limit: 20 });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("/api/addons/intelligence/find");
    expect(init.method).toBe("POST");
    const body = JSON.parse(init.body as string);
    expect(body.question).toBe("SF っぽいの");
    expect(body.limit).toBe(20);
    expect(out.total).toBe(8);
  });

  it("sends the X-Lit-Drive header (drive header convention from existing api.ts)", async () => {
    const fetchMock = mockFetchJson(sampleResponse);
    vi.stubGlobal("fetch", fetchMock);

    await findFiles("先週観た映画", "家族");

    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    const headers = new Headers(init.headers as HeadersInit);
    // Non-ASCII drive names are percent-encoded to satisfy ISO-8859-1
    // header constraints — same convention as ``driveHeaders`` in api.ts.
    expect(headers.get("x-lit-drive")).toBe(encodeURIComponent("家族"));
    expect(headers.get("content-type")).toMatch(/application\/json/i);
  });

  it("does not set viewer-id headers directly (host proxy injects X-Lit-Viewer-Id)", async () => {
    // The addon_proxy reads the lit_viewer cookie server-side and replaces
    // X-Lit-Viewer-Id; client-supplied values are stripped to prevent forgery.
    // findFiles must not duplicate that work or send a competing header alias.
    document.cookie = "lit_viewer=sha256-deadbeef; path=/";
    const fetchMock = mockFetchJson(sampleResponse);
    vi.stubGlobal("fetch", fetchMock);

    await findFiles("ask", "family");

    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    const headers = new Headers(init.headers as HeadersInit);
    expect(headers.get("x-lit-viewer-id")).toBeNull();
    expect(headers.get("x-hv-viewer-id")).toBeNull();
  });

  it("includes overrides in the request body when provided (chip × re-POST path)", async () => {
    const fetchMock = mockFetchJson(sampleResponse);
    vi.stubGlobal("fetch", fetchMock);

    await findFiles("先週観た映画でSF", "family", {
      overrides: {
        time_range: "none",
        personal_scope: "viewed",
        file_type_hint: "video",
        semantic_query: "SF",
      },
    });

    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    const body = JSON.parse(init.body as string);
    expect(body.overrides).toEqual({
      time_range: "none",
      personal_scope: "viewed",
      file_type_hint: "video",
      semantic_query: "SF",
    });
  });

  it("omits overrides from the body when not provided (initial query path)", async () => {
    const fetchMock = mockFetchJson(sampleResponse);
    vi.stubGlobal("fetch", fetchMock);

    await findFiles("先週観た映画でSF", "family");

    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    const body = JSON.parse(init.body as string);
    expect(body).not.toHaveProperty("overrides");
  });
});

describe("findFiles — response handling", () => {
  it("returns the parsed JSON body matching the FindResponse shape", async () => {
    const fetchMock = mockFetchJson(sampleResponse);
    vi.stubGlobal("fetch", fetchMock);

    const result = await findFiles("q", "family");

    expect(result.decomposed.semantic_query).toBe("SF");
    expect(result.decomposed.time_range.value).toBe("last_week");
    expect(result.results).toHaveLength(1);
    expect(result.results[0].file.name).toBe("Interstellar.mp4");
    expect(result.results[0].hit.text).toContain("宇宙船");
    expect(result.total).toBe(8);
    expect(result.limit).toBe(20);
  });

  it("rejects with an error when the server returns a non-2xx status", async () => {
    const fetchMock = mockFetchJson({ detail: "feature off" }, 400);
    vi.stubGlobal("fetch", fetchMock);

    await expect(findFiles("q", "family")).rejects.toThrow();
  });
});
