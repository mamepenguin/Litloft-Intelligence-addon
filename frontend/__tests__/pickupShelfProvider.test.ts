/**
 * SPEC-ADDON-087: Intelligence supplies the video app's Recommended shelf
 * from the Pickup feed, through `slotProviders.pickup`.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

const fetchPickup = vi.fn();

vi.mock("@/addons/intelligence/api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/addons/intelligence/api")>()),
  fetchPickup: (...args: unknown[]) => fetchPickup(...args),
}));

import * as slots from "@/addons/intelligence/slots";

type Restrict = (ids: string[]) => Promise<string[]>;
type Provider = (args: { drive: string; limit: number; restrict: Restrict }) => Promise<string[]>;

const ADDON_ROOT = [
  resolve(process.cwd(), "../addons/intelligence"),
  resolve(process.cwd(), "addons/intelligence"),
].find((candidate) => existsSync(resolve(candidate, "manifest.json")))!;

const DRIVE = "videos";

function pickup(): Provider {
  const providers = (slots as { slotProviders?: Record<string, Provider> }).slotProviders;
  const provider = providers?.pickup;
  expect(typeof provider).toBe("function");
  return provider!;
}

const feedOf = (n: number) => Array.from({ length: n }, (_, i) => `f${String(i).padStart(3, "0")}`);

/** Serves `rows` in rank order; `total` defaults to their count. */
function serveFeed(rows: string[], total = rows.length) {
  fetchPickup.mockImplementation(async (_drive: string, opts: { limit?: number; offset?: number } = {}) => {
    const offset = opts.offset ?? 0;
    const limit = opts.limit ?? 12;
    return { file_ids: rows.slice(offset, offset + limit), total };
  });
}

function accepting(pred: (id: string) => boolean) {
  return vi.fn<Restrict>(async (ids) => ids.filter(pred));
}

const index = (id: string) => Number(id.slice(1));

const offsetsAsked = () =>
  fetchPickup.mock.calls.map(([, opts]) => (opts as { offset?: number } | undefined)?.offset ?? 0);

beforeEach(() => {
  fetchPickup.mockReset();
  vi.useFakeTimers({ toFake: ["Date"] });
  vi.setSystemTime(new Date(2026, 10, 9, 12, 0, 0));
});

afterEach(() => {
  vi.useRealTimers();
});

describe("SPEC-ADDON-087 Pickup registers for video-discover-recommended", () => {
  it("SPEC-ADDON-087 the manifest declares entry pickup under video-discover-recommended with no href", () => {
    const manifest = JSON.parse(readFileSync(resolve(ADDON_ROOT, "manifest.json"), "utf-8")) as {
      slots?: Record<string, { id: string; href?: string }[]>;
    };
    const entries = manifest.slots?.["video-discover-recommended"] ?? [];
    const entry = entries.find((e) => e.id === "pickup");

    expect(entry).toBeDefined();
    expect(entry).not.toHaveProperty("href");
  });

  it("SPEC-ADDON-087 slots.ts exports slotProviders.pickup as a function", () => {
    pickup();
  });
});

describe("SPEC-ADDON-087 I12 reading the feed", () => {
  it("SPEC-ADDON-087 I12 reads the drive's feed in rank order, 60 per page, passing each page through restrict", async () => {
    const feed = feedOf(300);
    serveFeed(feed);
    const restrict = accepting((id) => index(id) % 10 === 0);

    await pickup()({ drive: DRIVE, limit: 12, restrict });

    expect(offsetsAsked()).toEqual([0, 60, 120, 180, 240]);
    for (const [drive, opts] of fetchPickup.mock.calls) {
      expect(drive).toBe(DRIVE);
      expect(opts).toMatchObject({ limit: 60 });
      expect((opts as { daily?: boolean }).daily).not.toBe(true);
    }
    expect(restrict.mock.calls.map(([ids]) => ids)).toEqual([0, 60, 120, 180, 240].map((o) => feed.slice(o, o + 60)));
  });

  it("SPEC-ADDON-087 I12 never reads more than 5 pages, even when nothing matches", async () => {
    serveFeed(feedOf(300));

    const out = await pickup()({ drive: DRIVE, limit: 12, restrict: accepting(() => false) });

    expect(fetchPickup).toHaveBeenCalledTimes(5);
    expect(out).toEqual([]);
  });

  it("SPEC-ADDON-087 I12 stops once 40 ids have matched within the first page", async () => {
    serveFeed(feedOf(300));

    await pickup()({ drive: DRIVE, limit: 12, restrict: accepting(() => true) });

    expect(offsetsAsked()).toEqual([0]);
  });

  it("SPEC-ADDON-087 I12 stops at the page that brings the matches to 40", async () => {
    serveFeed(feedOf(300));

    await pickup()({ drive: DRIVE, limit: 12, restrict: accepting((id) => index(id) % 2 === 0) });

    expect(offsetsAsked()).toEqual([0, 60]);
  });

  it("SPEC-ADDON-087 I12 stops at a page shorter than 60", async () => {
    serveFeed(feedOf(70), 300);

    await pickup()({ drive: DRIVE, limit: 12, restrict: accepting(() => false) });

    expect(offsetsAsked()).toEqual([0, 60]);
  });

  it("SPEC-ADDON-087 I12 stops when the offset reaches total", async () => {
    serveFeed(feedOf(120));

    await pickup()({ drive: DRIVE, limit: 12, restrict: accepting(() => false) });

    expect(offsetsAsked()).toEqual([0, 60]);
  });
});

describe("SPEC-ADDON-087 I8 I12 the returned ids", () => {
  it("SPEC-ADDON-087 I12 samples only from the first 40 matched ids in rank order", async () => {
    serveFeed(feedOf(300));
    const matched = feedOf(300).filter((id) => index(id) % 2 === 0);
    const pool = new Set(matched.slice(0, 40));

    const out = await pickup()({ drive: DRIVE, limit: 12, restrict: accepting((id) => index(id) % 2 === 0) });

    expect(out).toHaveLength(12);
    expect(out.every((id) => pool.has(id))).toBe(true);
  });

  it("SPEC-ADDON-087 I8 returns at most limit distinct ids, all from the feed and accepted by restrict", async () => {
    const feed = feedOf(300);
    serveFeed(feed);
    const accepted = (id: string) => index(id) % 3 === 0;

    for (const limit of [12, 3, 1]) {
      const out = await pickup()({ drive: DRIVE, limit, restrict: accepting(accepted) });

      expect(out).toHaveLength(limit);
      expect(new Set(out).size).toBe(out.length);
      expect(out.every((id) => feed.includes(id) && accepted(id))).toBe(true);
    }
  });

  it("SPEC-ADDON-087 I8 returns the same ids in the same order for the same drive, pool and date", async () => {
    serveFeed(feedOf(300));
    const run = () => pickup()({ drive: DRIVE, limit: 12, restrict: accepting((id) => index(id) % 2 === 0) });

    const first = await run();
    const second = await run();

    expect(second).toEqual(first);
  });

  it("SPEC-ADDON-087 returns the whole pool when fewer than limit ids match", async () => {
    serveFeed(feedOf(300));
    const accepted = ["f007", "f130", "f299"];

    const out = await pickup()({ drive: DRIVE, limit: 12, restrict: accepting((id) => accepted.includes(id)) });

    expect([...out].sort()).toEqual(accepted);
  });

  it("SPEC-ADDON-087 returns [] for an empty feed", async () => {
    serveFeed([]);
    const restrict = accepting(() => true);

    const out = await pickup()({ drive: DRIVE, limit: 12, restrict });

    expect(out).toEqual([]);
  });

  it("SPEC-ADDON-087 rejects when restrict rejects", async () => {
    serveFeed(feedOf(300));
    const restrict = vi.fn<Restrict>(async () => {
      throw new Error("restrict failed");
    });

    await expect(pickup()({ drive: DRIVE, limit: 12, restrict })).rejects.toThrow();
  });
});
