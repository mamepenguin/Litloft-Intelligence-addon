import type { ReactNode } from "react";

import jaMessages from "@/messages/ja.json";

type MessageTree = { [key: string]: string | MessageTree };

function lookup(tree: MessageTree, path: string): unknown {
  let current: unknown = tree;
  for (const part of path.split(".")) {
    if (typeof current !== "object" || current === null) return undefined;
    current = (current as Record<string, unknown>)[part];
  }
  return current;
}

const messages = jaMessages as unknown as MessageTree;

export function useTranslations(namespace?: string) {
  const t = (key: string, values?: Record<string, unknown>) => {
    const path = namespace === undefined ? key : `${namespace}.${key}`;
    const raw = lookup(messages, path);
    let text = typeof raw === "string" ? raw : path;
    if (values) {
      text = Object.entries(values).reduce(
        (str, [name, value]) => str.replace(`{${name}}`, String(value)),
        text,
      );
    }
    return text;
  };
  return t;
}

export function useLocale() {
  return "ja";
}

export function NextIntlClientProvider({ children }: { children: ReactNode }) {
  return children;
}
