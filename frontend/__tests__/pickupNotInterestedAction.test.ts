/**
 * SPEC-ADDON-154 rule 10: the `pickup` entry's item action for the video app, through
 * `slotItemActions` (SPEC-CORE-150 rule 2), observed at the fetch boundary.
 */

import { describe, it, expect, vi, afterEach } from "vitest";

import en from "@/messages/en.json";
import * as slots from "@/addons/intelligence/slots";

type RunResult = { message: string; undo?: () => Promise<void> };
type Action = { id: string; label: string; icon?: unknown; run: (fileId: string) => Promise<RunResult> };
type Factory = (args: { drive: string; locale: string; t: (key: string) => string }) => Action[];

const FILE = "abc_def-ghij";
const URL_PATH = `/api/addons/intelligence/pickup/not-interested/${FILE}`;

function t(key: string): string {
  const value = key
    .split(".")
    .reduce<unknown>((node, part) => (node && typeof node === "object" ? (node as Record<string, unknown>)[part] : undefined), en);
  return typeof value === "string" ? value : key;
}

function stubFetch(status = 204, body = "") {
  const fn = vi.fn(async (_url: unknown, _init?: RequestInit) => new Response(status === 204 ? null : body, { status }));
  vi.stubGlobal("fetch", fn);
  return fn;
}

function calls(fn: ReturnType<typeof stubFetch>) {
  return fn.mock.calls.map(([url, init]) => ({
    path: new URL(String(url), "http://host").pathname,
    method: (init?.method ?? "GET").toUpperCase(),
    drive: new Headers(init?.headers).get("X-Lit-Drive"),
  }));
}

function actions(drive = "videos"): Action[] {
  const factory = (slots as { slotItemActions?: Record<string, Factory> }).slotItemActions?.pickup;
  expect(typeof factory).toBe("function");
  return factory!({ drive, locale: "en", t });
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("SPEC-ADDON-154 rule 10 the video app's Not interested action", () => {
  it("SPEC-ADDON-154 rule 10 is one action labelled from the addon's messages through the core's translator", () => {
    const list = actions();

    expect(list).toHaveLength(1);
    expect(typeof list[0].id).toBe("string");
    expect(list[0].label).toBe("Not interested");
    expect(typeof list[0].run).toBe("function");
  });

  it("SPEC-ADDON-154 rule 10 run sends PUT for the drive and resolves with the message and an undo that sends DELETE", async () => {
    const fetchMock = stubFetch(204);
    const [action] = actions("videos");

    const result = await action.run(FILE);

    expect(result.message).toBe("Removed from your picks");
    expect(typeof result.undo).toBe("function");
    expect(calls(fetchMock)).toEqual([{ path: URL_PATH, method: "PUT", drive: "videos" }]);

    await result.undo!();

    expect(calls(fetchMock)).toEqual([
      { path: URL_PATH, method: "PUT", drive: "videos" },
      { path: URL_PATH, method: "DELETE", drive: "videos" },
    ]);
  });

  it.each([
    ["400 without a viewer", 400, '{"detail":"viewer_required"}'],
    ["404 with the index policy off", 404, '{"detail":"Not Found"}'],
    ["500", 500, "boom"],
  ])("SPEC-ADDON-154 rule 10 run rejects on %s", async (_name, status, body) => {
    stubFetch(status, body);
    const [action] = actions();

    await expect(action.run(FILE)).rejects.toBeDefined();
  });

  it("SPEC-ADDON-154 rule 10 run rejects on a network failure", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => {
      throw new TypeError("Failed to fetch");
    }));
    const [action] = actions();

    await expect(action.run(FILE)).rejects.toBeDefined();
  });

  it("SPEC-ADDON-154 rule 10 undo rejects on a non-2xx answer", async () => {
    stubFetch(204);
    const [action] = actions();
    const { undo } = await action.run(FILE);
    stubFetch(500, "boom");

    await expect(undo!()).rejects.toBeDefined();
  });
});

describe("SPEC-ADDON-154 section 9 the addon's own strings", () => {
  const values = (tree: unknown): string[] =>
    tree && typeof tree === "object" ? Object.values(tree as Record<string, unknown>).flatMap(values) : [tree as string];

  it.each([
    ["en", ["Not interested", "Removed from your picks"]],
    ["ja", ["興味なし", "おすすめから外しました"]],
  ])("SPEC-ADDON-154 the %s messages hold the label and the toast", async (locale, expected) => {
    const messages = (await import(`../messages/${locale}.json`)).default;

    expect(values(messages)).toEqual(expect.arrayContaining(expected));
  });
});
