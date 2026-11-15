/**
 * SPEC-ADDON-110 step 2, I6 (addon half): 「ここに質問」 in a Smart Collection's
 * … menu opens Ask with that Smart Collection as the only chip.
 */

import { Suspense } from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

const ADDON_ROOT = [
  resolve(process.cwd(), "../addons/intelligence"),
  resolve(process.cwd(), "addons/intelligence"),
].find((candidate) => existsSync(resolve(candidate, "manifest.json")))!;

const MANIFEST = JSON.parse(
  readFileSync(resolve(ADDON_ROOT, "manifest.json"), "utf-8"),
) as { slots?: Record<string, { id: string; priority?: number }[]> };

vi.mock("next-intl", () => {
  const messages = JSON.parse(
    readFileSync(resolve(ADDON_ROOT, "frontend/messages/en.json"), "utf-8"),
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
  };
});

const order: string[] = [];
const push = vi.fn((url: string) => {
  order.push(`push:${url}`);
});
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push }),
}));

type SlotComponent = React.ComponentType<Record<string, unknown>>;

function entryIds(): string[] {
  return (MANIFEST.slots?.["smart-collection-actions"] ?? []).map((row) => row.id);
}

async function entry(): Promise<SlotComponent> {
  const ids = entryIds();
  expect(ids.length).toBeGreaterThan(0);
  const { slotComponents } = await import("@/addons/intelligence/slots");
  const component = slotComponents[ids[0]];
  expect(component).toBeDefined();
  return component as unknown as SlotComponent;
}

async function renderEntry(name: string) {
  const Entry = await entry();
  const closeMenu = vi.fn(() => {
    order.push("closeMenu");
  });
  const view = render(
    <Suspense fallback={null}>
      <Entry
        drive="family"
        smartCollectionId="sf1"
        smartCollectionName={name}
        closeMenu={closeMenu}
      />
    </Suspense>,
  );
  return { ...view, closeMenu };
}

beforeEach(() => {
  push.mockClear();
  order.length = 0;
});

describe("SPEC-ADDON-110 step 2 / I6 Ask here on a Smart Collection", () => {
  it("registers an entry in smart-collection-actions", () => {
    expect(entryIds()).toHaveLength(1);
  });

  it("closes the menu, then opens Ask with the only chip in:\"<name>\"", async () => {
    await renderEntry("Evening mix");

    const row = await screen.findByRole("menuitem", { name: "Ask here" });
    expect(row.querySelector("svg")?.getAttribute("class") ?? "").toMatch(/lucide-message-circle-question/);
    fireEvent.click(row);

    expect(order[0]).toBe("closeMenu");
    expect(push).toHaveBeenCalledTimes(1);
    const target = new URL(push.mock.calls[0][0], "http://localhost");
    expect(target.pathname).toBe("/drive/family/addons/intelligence");
    expect(target.searchParams.get("scope")).toBe('in:"Evening mix"');
    expect([...target.searchParams.keys()]).toEqual(["scope"]);
  });

  it.each(['a"b', "a＂b"])("draws nothing for a name holding %s", async (name) => {
    const plain = await renderEntry("Plain");
    await screen.findByRole("menuitem", { name: "Ask here" });
    plain.unmount();

    const { container } = await renderEntry(name);

    expect(container.innerHTML).toBe("");
  });
});
