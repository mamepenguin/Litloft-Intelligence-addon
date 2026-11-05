/**
 * SPEC-ADDON-064 entry points: 「ここに質問」 in the folder ``…`` menu and on
 * the collection page opens Ask with that folder or collection as the scope,
 * and is absent at the drive root or for a name the quoting writer cannot
 * express.
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

const ADDON_ROOT = [
  resolve(process.cwd(), "../addons/intelligence"),
  resolve(process.cwd(), "addons/intelligence"),
].find((candidate) => existsSync(resolve(candidate, "manifest.json")))!;

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

const push = vi.fn();
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push }),
}));

type SlotComponent = React.ComponentType<Record<string, unknown>>;

async function folderItem(): Promise<SlotComponent> {
  const mod = (await vi.importActual("./FolderAskMenuItem")) as {
    default?: SlotComponent;
    FolderAskMenuItem?: SlotComponent;
  };
  return mod.default ?? mod.FolderAskMenuItem!;
}

async function collectionButton(): Promise<SlotComponent> {
  const mod = (await vi.importActual("./CollectionAskButton")) as {
    default?: SlotComponent;
    CollectionAskButton?: SlotComponent;
  };
  return mod.default ?? mod.CollectionAskButton!;
}

function scopeOf(url: string): string | null {
  return new URL(url, "http://localhost").searchParams.get("scope");
}

beforeEach(() => {
  push.mockClear();
});

describe("SPEC-ADDON-064 the folder menu row", () => {
  it("asks from a non-root folder, closing the menu first", async () => {
    const FolderAskMenuItem = await folderItem();
    const onRequestClose = vi.fn();
    render(
      <FolderAskMenuItem
        drive="family"
        path="Work/2024"
        fileIds={[]}
        onRequestClose={onRequestClose}
      />,
    );

    const row = screen.getByRole("menuitem", { name: "Ask here" });
    fireEvent.click(row);

    expect(onRequestClose).toHaveBeenCalledTimes(1);
    expect(push).toHaveBeenCalledTimes(1);
    expect(push.mock.calls[0][0]).toMatch(
      /^\/drive\/family\/addons\/intelligence\?/,
    );
    expect(scopeOf(push.mock.calls[0][0])).toBe("path:Work/2024");
  });

  it("draws nothing at the drive root", async () => {
    const FolderAskMenuItem = await folderItem();
    const { container } = render(
      <FolderAskMenuItem drive="family" path="" fileIds={["a"]} />,
    );

    expect(container.innerHTML).toBe("");
  });

  it.each(['a"b', "a＂b"])(
    "draws nothing when the path contains %s",
    async (path) => {
      const FolderAskMenuItem = await folderItem();
      const { container } = render(
        <FolderAskMenuItem drive="family" path={path} fileIds={["a"]} />,
      );

      expect(container.innerHTML).toBe("");
    },
  );
});

describe("SPEC-ADDON-064 the collection button", () => {
  it("asks from a collection, including a name with spaces", async () => {
    const CollectionAskButton = await collectionButton();
    render(
      <CollectionAskButton
        drive="family"
        collectionId="c1"
        collectionName="読書 メモ"
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Ask here" }));

    await waitFor(() => expect(push).toHaveBeenCalledTimes(1));
    expect(push.mock.calls[0][0]).toMatch(
      /^\/drive\/family\/addons\/intelligence\?/,
    );
    expect(scopeOf(push.mock.calls[0][0])).toBe("in:\"読書 メモ\"");
  });

  it.each(['a"b', "a＂b"])(
    "draws nothing when the name contains %s",
    async (name) => {
      const CollectionAskButton = await collectionButton();
      const { container } = render(
        <CollectionAskButton
          drive="family"
          collectionId="c1"
          collectionName={name}
        />,
      );

      expect(container.innerHTML).toBe("");
    },
  );
});
