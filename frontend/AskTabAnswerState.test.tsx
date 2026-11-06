/**
 * The tab's answer state while its data is still arriving: the image notice
 * waits for `done`, and a file change starts from the new file's entry.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

const ADDON_ROOT = [
  resolve(process.cwd(), "../addons/intelligence"),
  resolve(process.cwd(), "addons/intelligence"),
].find((candidate) => existsSync(resolve(candidate, "manifest.json")))!;

type MessageTree = { [key: string]: string | MessageTree };

const EN = JSON.parse(
  readFileSync(resolve(ADDON_ROOT, "frontend/messages/en.json"), "utf-8"),
) as MessageTree;

const lookup = (tree: MessageTree, path: string): unknown =>
  path.split(".").reduce<unknown>(
    (node, part) =>
      typeof node === "object" && node !== null
        ? (node as Record<string, unknown>)[part]
        : undefined,
    tree,
  );

vi.mock("next-intl", () => {
  const useTranslations = (namespace?: string) => {
    const t = (key: string) => {
      const found = lookup(
        EN,
        namespace === undefined ? key : `${namespace}.${key}`,
      );
      return typeof found === "string" ? found : key;
    };
    return t;
  };
  return {
    useTranslations,
    useLocale: () => "en",
    NextIntlClientProvider: ({ children }: { children: unknown }) => children,
  };
});

vi.mock("@/hooks/useWebSocket", () => ({
  useWebSocket: () => null,
}));

import AskTab from "./AskTab";

interface Call {
  url: string;
  method: string;
  body: Record<string, unknown> | undefined;
}

const net: {
  calls: Call[];
  answer: () => Response;
} = {
  calls: [],
  answer: () => json({ detail: "not stubbed" }, 500),
};

function json(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: String(status),
    json: async () => body,
    text: async () => JSON.stringify(body),
    body: null,
  } as unknown as Response;
}

function sse() {
  let controller!: ReadableStreamDefaultController<Uint8Array>;
  const body = new ReadableStream<Uint8Array>({
    start(next) {
      controller = next;
    },
  });
  const encoder = new TextEncoder();
  return {
    response: {
      ok: true,
      status: 200,
      statusText: "200",
      json: async () => ({}),
      text: async () => "",
      body,
    } as unknown as Response,
    push(text: string) {
      controller.enqueue(encoder.encode(text));
    },
    close() {
      controller.close();
    },
  };
}

function frame(event: string, data: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

const fetchMock = vi.fn(async (url: string, init: RequestInit = {}) => {
  net.calls.push({
    url: String(url),
    method: String(init.method ?? "GET").toUpperCase(),
    body: typeof init.body === "string" ? JSON.parse(init.body) : undefined,
  });
  if (String(url).includes("/llm/choices/rag")) {
    return json({ auto: "local", choices: [] });
  }
  if (String(init.method ?? "GET").toUpperCase() === "POST") {
    return net.answer();
  }
  return json({ available: true });
});
vi.stubGlobal("fetch", fetchMock);

function pageClosed(fileId: string): CustomEvent<{ fileId: string }> {
  return new CustomEvent("litloft:file-page-closed", { detail: { fileId } });
}

const ASKED_FILE_IDS = ["ask000000001", "ask000000002"];

async function renderTab(fileId = ASKED_FILE_IDS[0]) {
  const props = {
    fileId,
    drive: "family",
    filename: "receipt.png",
    fileType: "image",
    mimeType: "image/png",
    mediaController: null,
    documentController: null,
    lowerInspector: () => {},
    onAvailability: vi.fn(),
  };
  const view = render(<AskTab {...props} />);
  await waitFor(() => expect(props.onAvailability).toHaveBeenCalled());
  await waitFor(() =>
    expect(
      net.calls.some((call) => call.url.includes("/llm/choices/rag")),
    ).toBe(true),
  );
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 20));
  });
  return { ...view, props };
}

async function ask(text: string): Promise<void> {
  fireEvent.change(
    await screen.findByPlaceholderText("Ask about this file"),
    { target: { value: text } },
  );
  fireEvent.click(screen.getByRole("button", { name: "Ask" }));
  await waitFor(() =>
    expect(net.calls.filter((call) => call.method === "POST")).toHaveLength(1),
  );
}

async function push(
  stream: { push: (text: string) => void },
  text: string,
): Promise<void> {
  await act(async () => {
    stream.push(text);
  });
}

const IMAGE_NOTICE =
  "The image could not be used, so this answer is from the text only.";

beforeEach(() => {
  net.calls = [];
  net.answer = () => json({ detail: "not stubbed" }, 500);
  fetchMock.mockClear();
  for (const fileId of ASKED_FILE_IDS) {
    window.dispatchEvent(pageClosed(fileId));
  }
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("SPEC-ADDON-071 I3 - the notice waits for the answer to finish", () => {
  it("shows no notice while an unavailable meta is still streaming", async () => {
    const stream = sse();
    net.answer = () => stream.response;
    await renderTab();

    await ask("what is on the receipt");
    await push(stream, frame("answer_chunk", { delta: "An answer." }));
    await push(stream, frame("meta", { partial: false, image: "unavailable" }));

    expect(screen.queryByText(IMAGE_NOTICE)).toBeNull();

    await push(stream, frame("done", {}));

    expect(await screen.findByText(IMAGE_NOTICE)).toBeInTheDocument();
  });
});

describe("SPEC-ADDON-053 I11 - a new file shows only its own entry", () => {
  it("drops the streaming answer when the open file changes", async () => {
    const stream = sse();
    net.answer = () => stream.response;
    const view = await renderTab();

    await ask("who wrote this");
    await push(stream, frame("answer_chunk", { delta: "First file answer." }));
    expect(document.body.textContent).toContain("First file answer.");

    view.rerender(<AskTab {...view.props} fileId={ASKED_FILE_IDS[1]} />);

    await waitFor(() =>
      expect(document.body.textContent).not.toContain("First file answer."),
    );
    expect(
      (await screen.findByPlaceholderText("Ask about this file") as HTMLTextAreaElement).value,
    ).toBe("");
  });
});
