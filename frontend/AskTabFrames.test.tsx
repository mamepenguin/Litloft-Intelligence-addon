/**
 * SPEC-ADDON-108: the tab side of a video with no text. The tab is rendered
 * through the addon's `slots.ts` entry, as the file page does, and every
 * route is a function.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { Suspense } from "react";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

const ADDON_ROOT = [
  resolve(process.cwd(), "../addons/intelligence"),
  resolve(process.cwd(), "addons/intelligence"),
].find((candidate) => existsSync(resolve(candidate, "manifest.json")))!;

const MANIFEST = JSON.parse(
  readFileSync(resolve(ADDON_ROOT, "manifest.json"), "utf-8"),
) as { slots?: Record<string, { id: string; i18n_key?: string }[]> };
const ENTRY = (MANIFEST.slots?.["inspector-tabs"] ?? [])[0];
const EN = JSON.parse(
  readFileSync(resolve(ADDON_ROOT, "frontend/messages/en.json"), "utf-8"),
) as MessageTree;
const JA = JSON.parse(
  readFileSync(resolve(ADDON_ROOT, "frontend/messages/ja.json"), "utf-8"),
) as MessageTree;

type MessageTree = { [key: string]: string | MessageTree };

const lookup = (tree: MessageTree, path: string): unknown =>
  path.split(".").reduce<unknown>(
    (node, part) =>
      typeof node === "object" && node !== null
        ? (node as Record<string, unknown>)[part]
        : undefined,
    tree,
  );

/** The addon's own catalogue stands in for the merged one. */
vi.mock("next-intl", async () => {
  const { existsSync: exists, readFileSync: read } = await import("node:fs");
  const { resolve: join } = await import("node:path");
  const root = [
    join(process.cwd(), "../addons/intelligence"),
    join(process.cwd(), "addons/intelligence"),
  ].find((candidate) => exists(candidate));
  const messages = JSON.parse(
    read(resolve(root!, "frontend/messages/en.json"), "utf-8"),
  ) as MessageTree;
  const lookupIn = (path: string): unknown =>
    path.split(".").reduce<unknown>(
      (node, part) =>
        typeof node === "object" && node !== null
          ? (node as Record<string, unknown>)[part]
          : undefined,
      messages,
    );
  const useTranslations = (namespace?: string) => {
    const t = (key: string, values?: Record<string, unknown>) => {
      const path = namespace === undefined ? key : `${namespace}.${key}`;
      const found = lookupIn(path);
      const raw = typeof found === "string" ? found : path;
      if (!values) return raw;
      return Object.entries(values).reduce(
        (text, [name, value]) => text.replace(`{${name}}`, String(value)),
        raw,
      );
    };
    return t;
  };
  return {
    useTranslations,
    useLocale: () => "en",
    NextIntlClientProvider: ({ children }: { children: unknown }) => children,
  };
});

const ws = vi.hoisted(() => ({ event: null as Record<string, unknown> | null }));

vi.mock("@/hooks/useWebSocket", () => ({
  useWebSocket: (filter?: string) =>
    filter && ws.event?.event !== filter ? null : ws.event,
}));

import { slotComponents } from "@/addons/intelligence/slots";
import type { MediaController } from "@/lib/mediaController";
import type { DocumentController } from "@/lib/documentController";

interface Call {
  url: string;
  method: string;
  body: Record<string, unknown> | undefined;
  headers: Record<string, string>;
  signal: AbortSignal | undefined;
}

type Reply = (call: Call) => Response | Promise<Response>;

interface Network {
  calls: Call[];
  availability: Reply;
  choices: Reply;
  answer: Reply;
}

const net: Network = {
  calls: [],
  availability: () => json({ available: true }),
  choices: () => json({ auto: "local", choices: [] }),
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
    push(frame: string) {
      controller.enqueue(encoder.encode(frame));
    },
    close() {
      controller.close();
    },
  };
}

function frame(event: string, data: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

function sseFrom(frames: string[]): Response {
  const stream = sse();
  for (const text of frames) stream.push(text);
  stream.close();
  return stream.response;
}

const fetchMock = vi.fn(async (url: string, init: RequestInit = {}) => {
  const call: Call = {
    url: String(url),
    method: String(init.method ?? "GET").toUpperCase(),
    body: typeof init.body === "string" ? JSON.parse(init.body) : undefined,
    headers: Object.fromEntries(new Headers(init.headers).entries()),
    signal: init.signal ?? undefined,
  };
  net.calls.push(call);
  if (call.url.includes("/llm/choices/rag")) return net.choices(call);
  if (call.method === "POST") return net.answer(call);
  return net.availability(call);
});
vi.stubGlobal("fetch", fetchMock);

function callsTo(method: string, path: string): Call[] {
  return net.calls.filter((call) => call.method === method && call.url.includes(path));
}

type TabProps = Record<string, unknown> & {
  fileId: string;
  drive: string;
  mediaController: Partial<MediaController> | null;
  documentController: Partial<DocumentController> | null;
  lowerInspector: () => void;
  onAvailability: (available: boolean) => void;
};

function tabProps(overrides: Record<string, unknown> = {}): TabProps {
  return {
    fileId: "ask000000001",
    drive: "family",
    filename: "lecture.mp4",
    fileType: "video",
    mimeType: "video/mp4",
    videoRef: { current: null },
    subtitleTracks: [],
    mediaController: null,
    documentCaptureController: null,
    documentController: null,
    subtitles: [],
    lowerInspector: () => {},
    trustTier: "verified",
    trustReviewedAt: null,
    onFileChange: () => {},
    fillHeight: true,
    labelledByHost: true,
    onAvailability: vi.fn(),
    ...overrides,
  } as TabProps;
}

function AskTab(props: TabProps) {
  const Component = slotComponents[ENTRY?.id as string];
  if (!Component) {
    throw new Error(
      MANIFEST.slots?.["inspector-tabs"]?.length
        ? `slots.ts registers no component for the inspector-tabs entry "${ENTRY.id}"`
        : "manifest.json declares no inspector-tabs entry",
    );
  }
  return (
    <Suspense fallback={null}>
      <Component {...props} />
    </Suspense>
  );
}

async function renderTab(overrides: Record<string, unknown> = {}) {
  const props = tabProps(overrides);
  const utils = render(<AskTab {...props} />);
  await waitFor(() => expect(props.onAvailability).toHaveBeenCalled());
  return { ...utils, props };
}

async function questionBox(): Promise<HTMLTextAreaElement> {
  return (await screen.findByPlaceholderText(
    "Ask about this file",
  )) as HTMLTextAreaElement;
}

async function submitButton(): Promise<HTMLElement> {
  return screen.findByRole("button", { name: "Ask" });
}

async function ask(text: string): Promise<void> {
  const before = callsTo("POST", "/ask").length;
  fireEvent.change(await questionBox(), { target: { value: text } });
  fireEvent.click(await submitButton());
  await waitFor(() => expect(callsTo("POST", "/ask")).toHaveLength(before + 1));
}

function pageClosed(fileId: string): CustomEvent<{ fileId: string }> {
  return new CustomEvent("litloft:file-page-closed", { detail: { fileId } });
}

/** The ids this file asks about; their store entries are module state. */
const ASKED_FILE_IDS = ["ask000000001", "ask000000002"];

const VIDEO_NEEDS_VISION =
  "This video has no text, and the selected model cannot read images.";
const VIDEO_NEEDS_VISION_JA =
  "この動画には文字情報がなく、選んだモデルは画像を読めません。";
const FRAMES_UNUSABLE =
  "This video has no text and its frames could not be read.";
const FRAMES_UNUSABLE_JA =
  "この動画には文字情報がなく、フレームも読めませんでした。";
const FRAMES_REJECTED =
  "The selected model did not accept this video's frames. Try another model.";
const FRAMES_REJECTED_JA =
  "選んだモデルがこの動画のフレームを受け付けませんでした。別のモデルで試してください。";
const PICTURE_NEEDS_VISION =
  "This picture has no text, and the selected model cannot read images.";
const IMAGE_UNUSABLE =
  "This picture has no text and could not be read as an image.";
const GENERIC_FAILURE = "Could not answer";

const LINES = [VIDEO_NEEDS_VISION, FRAMES_UNUSABLE, FRAMES_REJECTED];

beforeEach(() => {
  net.calls = [];
  net.availability = () => json({ available: true });
  net.choices = () => json({ auto: "local", choices: [] });
  net.answer = () => json({ detail: "not stubbed" }, 500);
  ws.event = null;
  localStorage.clear();
  sessionStorage.clear();
  fetchMock.mockClear();
  for (const fileId of ASKED_FILE_IDS) {
    window.dispatchEvent(pageClosed(fileId));
  }
});

afterEach(() => {
  for (const fileId of ASKED_FILE_IDS) {
    window.dispatchEvent(pageClosed(fileId));
  }
  vi.restoreAllMocks();
});

describe("SPEC-ADDON-108 I9 - a video with no text gets its own failure lines", () => {
  it.each([
    {
      name: "a 409 video_needs_vision",
      reply: () => json({ detail: "video_needs_vision" }, 409),
      shown: VIDEO_NEEDS_VISION,
    },
    {
      name: "a done carrying frames_unusable",
      reply: () =>
        sseFrom([
          frame("meta", { partial: false, image: "unavailable" }),
          frame("done", { error: "frames_unusable" }),
        ]),
      shown: FRAMES_UNUSABLE,
    },
    {
      name: "a done carrying frames_rejected",
      reply: () =>
        sseFrom([
          frame("meta", { partial: false, image: "unavailable" }),
          frame("done", { error: "frames_rejected" }),
        ]),
      shown: FRAMES_REJECTED,
    },
  ])("shows only its line for $name", async ({ reply, shown }) => {
    net.answer = reply as Reply;
    await renderTab({ fileType: "video", mimeType: "video/mp4" });

    await ask("what happens in this video");

    expect(await screen.findByText(shown)).toBeInTheDocument();
    for (const text of [
      ...LINES.filter((line) => line !== shown),
      GENERIC_FAILURE,
      PICTURE_NEEDS_VISION,
      IMAGE_UNUSABLE,
    ]) {
      expect(screen.queryByText(text, { exact: false })).toBeNull();
    }
  });
});

describe("SPEC-ADDON-108 section 8 - the lines exist in both catalogues", () => {
  function findPath(
    tree: MessageTree,
    target: string,
    prefix = "",
  ): string | null {
    for (const [key, value] of Object.entries(tree)) {
      const path = prefix ? `${prefix}.${key}` : key;
      if (value === target) return path;
      if (typeof value === "object" && value !== null) {
        const found = findPath(value, target, path);
        if (found) return found;
      }
    }
    return null;
  }

  it.each([
    [VIDEO_NEEDS_VISION, VIDEO_NEEDS_VISION_JA],
    [FRAMES_UNUSABLE, FRAMES_UNUSABLE_JA],
    [FRAMES_REJECTED, FRAMES_REJECTED_JA],
  ])("the key holding %s holds the Japanese line in ja", (en, ja) => {
    const key = findPath(EN, en);
    expect(key).not.toBeNull();
    expect(lookup(JA, key!)).toBe(ja);
  });
});
