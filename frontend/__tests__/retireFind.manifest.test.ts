/**
 * SPEC-ADDON-158: Find is gone from the manifest and from every client.
 */

import { describe, it, expect } from "vitest";
import { existsSync, lstatSync, readdirSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";

const ADDON_ROOT = [
  resolve(process.cwd(), "../addons/intelligence"),
  resolve(process.cwd(), "addons/intelligence"),
].find((candidate) => existsSync(resolve(candidate, "manifest.json")))!;

const CORE_ROOT = resolve(ADDON_ROOT, "../..");

type Route = { path: string; methods: string[] };
type Manifest = {
  slots?: Record<string, unknown[]>;
  proxy: { routes: Route[] };
};

const MANIFEST = JSON.parse(
  readFileSync(resolve(ADDON_ROOT, "manifest.json"), "utf-8"),
) as Manifest;

// The manifest at the commit the spec was approved against, minus the two
// entries the spec removes. Bound to this change: drop it once it merges.
const BASELINE = JSON.parse(
  readFileSync(resolve(ADDON_ROOT, "frontend/__tests__/retireFind.manifestBaseline.json"), "utf-8"),
) as { slots: Record<string, unknown[]>; routes: Route[] };

describe("SPEC-ADDON-158 I5 the manifest declares no Find", () => {
  it("has no search-modes key and no POST /find route, and every other slot and route as before", () => {
    expect(Object.keys(MANIFEST.slots ?? {})).not.toContain("search-modes");
    expect(
      MANIFEST.proxy.routes.filter((route) => route.path === "/find"),
    ).toEqual([]);

    expect(MANIFEST.slots).toEqual(BASELINE.slots);
    expect(MANIFEST.proxy.routes).toEqual(BASELINE.routes);
  });
});

const FIND_HREFS = ["/addons/intelligence/find", "/api/addons/intelligence/find"];
const SKIP_DIRS = new Set(["node_modules", ".next", "dist", "build", ".build", "__tests__"]);
const TEST_FILE = /\.(test|spec)\.[cm]?[jt]sx?$|Tests?\.swift$/;

function walk(dir: string, out: string[]): void {
  for (const name of readdirSync(dir)) {
    if (SKIP_DIRS.has(name)) continue;
    const path = join(dir, name);
    const stat = lstatSync(path);
    if (stat.isSymbolicLink()) continue;
    if (stat.isDirectory()) {
      walk(path, out);
    } else if (stat.isFile() && !TEST_FILE.test(name)) {
      out.push(path);
    }
  }
}

function roots(): string[] {
  const addonFrontends = readdirSync(join(CORE_ROOT, "addons"))
    .map((name) => join(CORE_ROOT, "addons", name, "frontend"))
    .filter((path) => existsSync(path));
  return [
    join(CORE_ROOT, "frontend/src"),
    ...addonFrontends,
    join(CORE_ROOT, "mcp-server/src"),
    join(CORE_ROOT, "ios"),
  ];
}

describe("SPEC-ADDON-158 I3 no client refers to Find", () => {
  it("leaves no reference to the Find page or the Find endpoint outside tests", () => {
    const offenders: string[] = [];
    for (const root of roots()) {
      const files: string[] = [];
      walk(root, files);
      expect(files.length, `no files collected under ${root}`).toBeGreaterThan(0);
      for (const file of files) {
        const text = readFileSync(file, "utf-8");
        if (FIND_HREFS.some((href) => text.includes(href))) {
          offenders.push(file.slice(CORE_ROOT.length + 1));
        }
      }
    }
    expect(offenders).toEqual([]);
  });
});
