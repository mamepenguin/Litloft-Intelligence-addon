/**
 * The Pickup client.
 *
 * The date is computed here, not on the server, so the carousel's day
 * turns over at the viewer's own midnight rather than UTC's. It only
 * seeds a shuffle, so getting it wrong is not dangerous — but getting
 * it wrong by a timezone means the cards change in the middle of an
 * evening.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { fetchPickup, localDateStamp } from "./api";

const originalFetch = globalThis.fetch;

function ok(body: unknown) {
  return Promise.resolve({ ok: true, json: async () => body } as Response);
}

function url(): string {
  const call = vi.mocked(globalThis.fetch).mock.calls[0][0];
  return String(call);
}

beforeEach(() => {
  globalThis.fetch = vi.fn() as never;
});

afterEach(() => {
  globalThis.fetch = originalFetch;
});

describe("localDateStamp", () => {
  it("uses local calendar fields, not UTC", () => {
    // 2026-03-02T23:30 local. In any timezone east of UTC this is
    // already the 3rd in UTC; the stamp must still say the 2nd.
    const d = new Date(2026, 2, 2, 23, 30, 0);
    expect(localDateStamp(d)).toBe("2026-03-02");
  });

  it("zero-pads", () => {
    expect(localDateStamp(new Date(2026, 0, 5))).toBe("2026-01-05");
  });
});

describe("fetchPickup", () => {
  it("asks for the daily window with the local date", async () => {
    vi.mocked(globalThis.fetch).mockReturnValue(
      ok({ file_ids: ["a"], total: 5 }),
    );

    await fetchPickup("videos", { limit: 12, daily: true });

    expect(url()).toContain("window=daily");
    expect(url()).toContain(`date=${localDateStamp()}`);
    expect(url()).toContain("limit=12");
  });

  it("sends no window when paging", async () => {
    vi.mocked(globalThis.fetch).mockReturnValue(
      ok({ file_ids: [], total: 0 }),
    );

    await fetchPickup("videos", { limit: 40, offset: 80 });

    expect(url()).toContain("offset=80");
    expect(url()).not.toContain("window=");
    expect(url()).not.toContain("date=");
  });

  it("percent-encodes the drive header", async () => {
    vi.mocked(globalThis.fetch).mockReturnValue(
      ok({ file_ids: [], total: 0 }),
    );

    await fetchPickup("動画", {});

    const init = vi.mocked(globalThis.fetch).mock.calls[0][1] as RequestInit;
    expect(init.headers).toEqual({ "X-Lit-Drive": "%E5%8B%95%E7%94%BB" });
  });

  it("returns an empty page rather than throwing on a bad response", async () => {
    vi.mocked(globalThis.fetch).mockReturnValue(
      Promise.resolve({ ok: false } as Response),
    );

    expect(await fetchPickup("videos", {})).toEqual({ file_ids: [], total: 0 });
  });

  it("returns an empty page when the network fails", async () => {
    vi.mocked(globalThis.fetch).mockRejectedValue(new Error("offline"));

    expect(await fetchPickup("videos", {})).toEqual({ file_ids: [], total: 0 });
  });

  it("tolerates a malformed payload", async () => {
    vi.mocked(globalThis.fetch).mockReturnValue(ok({ nonsense: true }));

    expect(await fetchPickup("videos", {})).toEqual({ file_ids: [], total: 0 });
  });
});
