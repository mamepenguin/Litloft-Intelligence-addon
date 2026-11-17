/**
 * SPEC-ID: SPEC-ADDON-125
 * SPEC-ID: SPEC-ADDON-126
 *
 * Row actions and the correction dialog on a coarse pointer, and the
 * unchanged fine path. The pointer is answered by `matchMedia`, which
 * jsdom does not implement, so each test sets it before mounting.
 */

import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { SubtitleInfo } from "@/types";

vi.mock("next-intl", async () => await import("./testI18nJa"));

const addon = vi.hoisted(() => ({
  features: { transcript_refine: "manual" as string | false },
}));
vi.mock("@/components/AddonSlotsProvider", () => ({
  useAddonStatus: () => addon,
  useAddonSlots: () => ({ slots: {} }),
}));

const ws = vi.hoisted(() => ({
  event: null as null | { event: string; data: unknown },
}));
vi.mock("@/hooks/useWebSocket", () => ({
  useWebSocket: (filter?: string) =>
    !filter || ws.event?.event === filter ? ws.event : null,
}));

import TranscriptSection from "@/addons/intelligence/TranscriptSection";
import { clearTranscriptScroll } from "@/addons/intelligence/transcriptScroll";
import { ShortcutsProvider } from "@/components/ShortcutsProvider";
import type { MediaController } from "@/lib/mediaController";
import { clearSourceCaptures, getSourceCaptures } from "@/lib/sourceCapture";

const FILE = "abc";
const OTHER_FILE = "def";
const DRIVE = "family";

const TEXT = [
  "一行目の文章です。",
  "二行目の文章です。",
  "三行目の文章です。",
];
const OTHER_FILE_TEXT = "別ファイルの一行目です。";
const WORDS_TEXT = "単語の一行目。";
const EXTERNAL_TEXT = "外部字幕の一行目。";

const CHANGED_MESSAGE =
  "内容が更新されました。読み込み直してから修正してください。";
const BUSY_MESSAGE = "処理中のため、いまは操作できません。";
const TOO_LONG_MESSAGE = "長すぎます";
const LAST_UTTERANCE_MESSAGE = "最後の発話は空にできません";
const UNAVAILABLE_MESSAGE = "このファイルは編集できなくなりました。";
const SAVE_FAILED_MESSAGE = "保存できませんでした";

const EXTERNAL_TRACKS: SubtitleInfo[] = [
  { index: 0, language: "ja", format: "vtt", label: "日本語" },
];

type Chunk = {
  index: number;
  text: string;
  start: number;
  end: number;
  correctedAt?: string | null;
  refinedAt?: string | null;
};

type Picture = {
  chunks: Chunk[];
  has_original: boolean;
  corrected: boolean;
  busy: boolean;
  followups_pending: boolean;
};

type Recorded = {
  method: string;
  url: string;
  headers: Record<string, string>;
  body: unknown;
};

function rows(): Chunk[] {
  return TEXT.map((text, index) => ({
    index,
    text,
    start: index * 5,
    end: index * 5 + 5,
  }));
}

function picture(overrides: Partial<Picture> = {}): Picture {
  return {
    chunks: rows(),
    has_original: false,
    corrected: false,
    busy: false,
    followups_pending: false,
    ...overrides,
  };
}

function otherPicture(): Picture {
  return {
    chunks: [{ index: 0, text: OTHER_FILE_TEXT, start: 0, end: 5 }],
    has_original: false,
    corrected: false,
    busy: false,
    followups_pending: false,
  };
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function vtt(text: string): Response {
  return new Response(
    `WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n${text}\n`,
    { status: 200, headers: { "Content-Type": "text/vtt" } },
  );
}

function headerRecord(headers: HeadersInit | undefined): Record<string, string> {
  if (!headers) return {};
  if (headers instanceof Headers) {
    return Object.fromEntries(headers.entries());
  }
  if (Array.isArray(headers)) return Object.fromEntries(headers);
  return headers as Record<string, string>;
}

let files: Record<string, Picture> = {};
let putError: { status: number; body: unknown } | null = null;
let pendingPut: { promise: Promise<Response>; resolve: (res: Response) => void } | null =
  null;
let requests: Recorded[] = [];

function deferred() {
  let resolve!: (res: Response) => void;
  const promise = new Promise<Response>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
  const url =
    typeof input === "string"
      ? input
      : input instanceof URL
        ? input.toString()
        : input.url;
  const method = init?.method ?? "GET";
  const body = init?.body ? JSON.parse(String(init.body)) : null;
  requests.push({ method, url, headers: headerRecord(init?.headers), body });

  const fileId = url.match(/\/files\/([^/]+)\//)?.[1] ?? FILE;

  if (url.includes("/subtitles.vtt")) return vtt(WORDS_TEXT);
  if (/\/subtitles\/\d+$/.test(url)) return vtt(EXTERNAL_TEXT);
  if (url.endsWith("/transcript") && method === "GET") {
    return json({
      available: true,
      file_id: fileId,
      drive: DRIVE,
      language: "ja",
      ...(files[fileId] ?? picture()),
    });
  }
  if (url.includes("/transcript/utterances/") && method === "PUT") {
    if (pendingPut) return pendingPut.promise;
    if (putError) return json(putError.body, putError.status);
    const index = Number(url.split("/utterances/")[1]);
    const row = files[fileId].chunks.find((chunk) => chunk.index === index)!;
    row.text = body.text;
    row.correctedAt = "2026-11-17T00:00:00Z";
    files[fileId].has_original = true;
    return json(row);
  }
  return json({ detail: "not found" }, 404);
});

vi.stubGlobal("fetch", fetchMock);

type Pointer = "coarse" | "fine" | "unknown";
let pointer: Pointer = "coarse";
const mediaListeners = new Set<() => void>();

function mediaQuery(query: string): MediaQueryList {
  const wantsCoarse = query.includes("coarse");
  const wantsFine = query.includes("fine");
  return {
    matches: wantsCoarse ? pointer === "coarse" : wantsFine ? pointer === "fine" : false,
    media: query,
    onchange: null,
    addEventListener: (_type: string, listener: () => void) => {
      mediaListeners.add(listener);
    },
    removeEventListener: (_type: string, listener: () => void) => {
      mediaListeners.delete(listener);
    },
    addListener: () => {},
    removeListener: () => {},
    dispatchEvent: () => false,
  } as unknown as MediaQueryList;
}

function setPointer(next: Pointer): void {
  pointer = next;
  act(() => {
    for (const listener of [...mediaListeners]) listener();
  });
}

type Controller = MediaController & { seek: ReturnType<typeof vi.fn>; play: ReturnType<typeof vi.fn> };

function controller(state: { currentTime: number; paused: boolean }): Controller {
  return {
    seek: vi.fn((seconds: number) => {
      state.currentTime = seconds;
    }),
    play: vi.fn(),
    pause: vi.fn(),
    togglePlay: vi.fn(),
    toggleMute: vi.fn(),
    toggleFullscreen: vi.fn(),
    getCurrentTime: () => state.currentTime,
    getDuration: () => 60,
    isPaused: () => state.paused,
    isMuted: () => false,
    getVolume: () => 1,
    setVolume: vi.fn(),
    getPlaybackRate: () => 1,
    setPlaybackRate: vi.fn(),
    getBufferedFraction: () => 0,
  };
}

interface RenderOptions {
  fileId?: string;
  mediaController?: MediaController | null;
}

function ui({ fileId = FILE, mediaController = null }: RenderOptions = {}) {
  return (
    <ShortcutsProvider>
      <TranscriptSection
        fileId={fileId}
        drive={DRIVE}
        filename="meeting.mp4"
        mediaController={mediaController}
        subtitles={EXTERNAL_TRACKS}
      />
    </ShortcutsProvider>
  );
}

function renderSection(options: RenderOptions = {}) {
  return render(ui(options));
}

/** The row block the list's scroll code identifies by its start time. */
function rowOf(text: string): HTMLElement {
  const row = screen.getByText(text).closest("[data-cue-start]");
  if (!(row instanceof HTMLElement)) {
    throw new Error(`no transcript row holds ${text}`);
  }
  return row;
}

/** The row's own seek control, which stays its first button. */
function seekOf(text: string): HTMLElement {
  return within(rowOf(text)).getAllByRole("button")[0];
}

function tapRow(text: string): void {
  fireEvent.click(seekOf(text));
}

function editButtons(): HTMLElement[] {
  return screen.queryAllByRole("button", { name: "修正" });
}

function stripQuotes(): HTMLElement[] {
  return screen.queryAllByRole("button", { name: "引用" });
}

function putRequests(): Recorded[] {
  return requests.filter((request) => request.method === "PUT");
}

function transcriptReads(): number {
  return requests.filter(
    (request) =>
      request.method === "GET" && request.url.endsWith("/transcript"),
  ).length;
}

async function reload(
  view: ReturnType<typeof renderSection>,
  fileId = FILE,
): Promise<void> {
  const before = transcriptReads();
  ws.event = {
    event: "intelligence.transcript.updated",
    data: { file_id: fileId },
  };
  view.rerender(ui({ fileId }));
  await waitFor(() => expect(transcriptReads()).toBeGreaterThan(before));
  // The read has been issued, not answered: let its response reach the rows.
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 0));
  });
}

/** A tap on the tapped row's 修正, then the dialog it must have opened. */
function openDialog(text: string): HTMLElement {
  tapRow(text);
  fireEvent.click(within(rowOf(text)).getByRole("button", { name: "修正" }));
  return screen.getByRole("dialog");
}

beforeEach(() => {
  files = { [FILE]: picture(), [OTHER_FILE]: otherPicture() };
  putError = null;
  pendingPut = null;
  requests = [];
  fetchMock.mockClear();
  ws.event = null;
  addon.features.transcript_refine = "manual";
  clearTranscriptScroll();
  clearSourceCaptures(DRIVE);
  pointer = "coarse";
  mediaListeners.clear();
  Object.defineProperty(window, "matchMedia", {
    configurable: true,
    writable: true,
    value: vi.fn((query: string) => mediaQuery(query)),
  });
});

afterEach(() => {
  cleanup();
});

describe("SPEC-ADDON-125 I1 — no row control before a tap", () => {
  const VIEWS = [
    { name: "分割テキスト", toggle: null, text: TEXT[0] },
    { name: "単語", toggle: "単語", text: WORDS_TEXT },
    { name: "外部字幕", toggle: "外部字幕", text: EXTERNAL_TEXT },
  ];

  for (const view of VIEWS) {
    it(`draws no edit or quote control on the ${view.name} rows`, async () => {
      renderSection();
      await screen.findByText(TEXT[0]);
      if (view.toggle) {
        fireEvent.click(await screen.findByRole("button", { name: view.toggle }));
      }
      await screen.findByText(view.text);

      expect(editButtons()).toHaveLength(0);
      expect(screen.queryAllByRole("button", { name: /引用/ })).toHaveLength(0);
    }, 10000);
  }
});

describe("SPEC-ADDON-125 I2 I3 — the tapped row and its strip", () => {
  it("seeks to the row's start, plays, and draws one strip on that row", async () => {
    const state = { currentTime: 0, paused: true };
    const mc = controller(state);
    renderSection({ mediaController: mc });
    await screen.findByText(TEXT[1]);

    tapRow(TEXT[1]);

    expect(mc.seek).toHaveBeenCalledWith(5);
    expect(mc.play).toHaveBeenCalled();
    expect(editButtons()).toHaveLength(1);
    expect(stripQuotes()).toHaveLength(1);
    expect(
      within(rowOf(TEXT[1])).getByRole("button", { name: "修正" }),
    ).toBeInTheDocument();
    expect(
      within(rowOf(TEXT[1])).getByRole("button", { name: "引用" }),
    ).toBeInTheDocument();
  });

  it("moves the strip to the row tapped last", async () => {
    renderSection();
    await screen.findByText(TEXT[0]);

    tapRow(TEXT[0]);
    tapRow(TEXT[2]);

    expect(editButtons()).toHaveLength(1);
    expect(stripQuotes()).toHaveLength(1);
    expect(
      within(rowOf(TEXT[2])).getByRole("button", { name: "修正" }),
    ).toBeInTheDocument();
    expect(
      within(rowOf(TEXT[0])).queryByRole("button", { name: "修正" }),
    ).toBeNull();
    expect(
      within(rowOf(TEXT[0])).queryByRole("button", { name: "引用" }),
    ).toBeNull();
  });

  it("keeps the strip when the selected row is tapped again", async () => {
    renderSection();
    await screen.findByText(TEXT[1]);

    tapRow(TEXT[1]);
    tapRow(TEXT[1]);

    expect(editButtons()).toHaveLength(1);
    expect(
      within(rowOf(TEXT[1])).getByRole("button", { name: "修正" }),
    ).toBeInTheDocument();
  });

  it("leaves the strip alone when the playback highlight moves", async () => {
    const state = { currentTime: 0, paused: false };
    const mc = controller(state);
    renderSection({ mediaController: mc });
    await screen.findByText(TEXT[0]);
    tapRow(TEXT[0]);

    state.currentTime = 12;
    await waitFor(
      () => {
        expect(screen.getByRole("button", { current: true })).toHaveTextContent(
          TEXT[2],
        );
      },
      { timeout: 3000 },
    );

    expect(editButtons()).toHaveLength(1);
    expect(
      within(rowOf(TEXT[0])).getByRole("button", { name: "修正" }),
    ).toBeInTheDocument();
  });

  it("selects the row with no media controller and without seeking", async () => {
    renderSection();
    await screen.findByText(TEXT[1]);

    tapRow(TEXT[1]);

    expect(editButtons()).toHaveLength(1);
    expect(stripQuotes()).toHaveLength(1);
    expect(
      within(rowOf(TEXT[1])).getByRole("button", { name: "修正" }),
    ).toBeInTheDocument();
  });

  const RELOADS = [
    {
      name: "keeps the selection when the row keeps its index and start",
      change: () => {},
      kept: true,
    },
    {
      name: "drops the selection when the row's start changes",
      change: () => {
        files[FILE].chunks[1] = { ...files[FILE].chunks[1], start: 6 };
      },
      kept: false,
    },
    {
      name: "drops the selection when the row's index is gone",
      change: () => {
        files[FILE].chunks = [files[FILE].chunks[0], files[FILE].chunks[2]];
      },
      kept: false,
    },
  ];

  for (const reloadCase of RELOADS) {
    it(`a reload ${reloadCase.name}`, async () => {
      const view = renderSection();
      await screen.findByText(TEXT[1]);
      tapRow(TEXT[1]);
      expect(stripQuotes()).toHaveLength(1);

      reloadCase.change();
      await reload(view);

      if (reloadCase.kept) {
        expect(editButtons()).toHaveLength(1);
        expect(
          within(rowOf(TEXT[1])).getByRole("button", { name: "修正" }),
        ).toBeInTheDocument();
      } else {
        expect(stripQuotes()).toHaveLength(0);
      }
    });
  }

  it("clears the selection when the source view switches", async () => {
    renderSection();
    await screen.findByText(TEXT[1]);
    tapRow(TEXT[1]);
    expect(stripQuotes()).toHaveLength(1);

    fireEvent.click(await screen.findByRole("button", { name: "単語" }));
    await screen.findByText(WORDS_TEXT);

    await waitFor(() => expect(stripQuotes()).toHaveLength(0));
    expect(editButtons()).toHaveLength(0);
  });

  it("clears the selection when the file changes", async () => {
    const view = renderSection();
    await screen.findByText(TEXT[1]);
    tapRow(TEXT[1]);
    expect(stripQuotes()).toHaveLength(1);

    view.rerender(ui({ fileId: OTHER_FILE }));
    await screen.findByText(OTHER_FILE_TEXT);

    expect(stripQuotes()).toHaveLength(0);
    expect(editButtons()).toHaveLength(0);
  });
});

describe("SPEC-ADDON-125 I4 — 修正 and 引用 in the strip", () => {
  for (const view of [
    { name: "単語", toggle: "単語", text: WORDS_TEXT },
    { name: "外部字幕", toggle: "外部字幕", text: EXTERNAL_TEXT },
  ]) {
    it(`offers 引用 but no 修正 in the ${view.name} view`, async () => {
      renderSection();
      await screen.findByText(TEXT[0]);
      fireEvent.click(await screen.findByRole("button", { name: view.toggle }));
      await screen.findByText(view.text);

      tapRow(view.text);

      const row = rowOf(view.text);
      expect(
        within(row).getByRole("button", { name: "引用" }),
      ).toBeInTheDocument();
      expect(
        within(row).queryByRole("button", { name: "修正" }),
      ).toBeNull();
    }, 10000);
  }

  it("disables 修正 and leaves 引用 enabled while the file is busy", async () => {
    files[FILE].busy = true;
    renderSection();
    await screen.findByText(TEXT[1]);

    tapRow(TEXT[1]);

    const row = rowOf(TEXT[1]);
    expect(within(row).getByRole("button", { name: "修正" })).toBeDisabled();
    expect(within(row).getByRole("button", { name: "引用" })).toBeEnabled();
  });

  it("takes the same capture entry as the fine-path quote button", async () => {
    renderSection();
    await screen.findByText(TEXT[0]);
    tapRow(TEXT[0]);
    fireEvent.click(within(rowOf(TEXT[0])).getByRole("button", { name: "引用" }));
    const coarse = getSourceCaptures(DRIVE);
    expect(stripQuotes()).toHaveLength(1);
    expect(editButtons()).toHaveLength(1);

    cleanup();
    pointer = "fine";
    clearSourceCaptures(DRIVE);
    clearTranscriptScroll();
    renderSection();
    await screen.findByText(TEXT[0]);
    fireEvent.click(screen.getAllByRole("button", { name: /引用バスケット/ })[0]);
    const fine = getSourceCaptures(DRIVE);

    const shape = (capture: (typeof coarse)[number]) => ({
      sourceFileId: capture.sourceFileId,
      filename: capture.filename,
      fileType: capture.fileType,
      kind: capture.kind,
      locator: capture.locator,
      quote: capture.quote,
    });
    expect(coarse.map(shape)).toEqual(fine.map(shape));
  });
});

describe("SPEC-ADDON-126 I5 — 修正 opens the dialog", () => {
  it("opens a modal dialog titled with the row's start, never an inline field", async () => {
    renderSection();
    await screen.findByText(TEXT[1]);

    const dialog = openDialog(TEXT[1]);

    expect(dialog).toHaveAttribute("aria-modal", "true");
    const heading = within(dialog).getByRole("heading");
    expect(heading).toHaveTextContent("修正");
    expect(heading).toHaveTextContent("0:05");
    expect(
      within(dialog).getByRole("button", { name: "閉じる" }),
    ).toBeInTheDocument();
    expect(rowOf(TEXT[1]).querySelector("input")).toBeNull();
    expect(rowOf(TEXT[1]).querySelector("textarea")).toBeNull();
  });

  it("starts from the stored text after a cancelled draft", async () => {
    renderSection();
    await screen.findByText(TEXT[1]);

    const first = openDialog(TEXT[1]);
    fireEvent.change(within(first).getByRole("textbox"), {
      target: { value: "すてた下書き" },
    });
    fireEvent.click(within(first).getByRole("button", { name: "キャンセル" }));

    expect(within(openDialog(TEXT[2])).getByRole("textbox")).toHaveValue(
      TEXT[2],
    );
    fireEvent.click(
      within(screen.getByRole("dialog")).getByRole("button", {
        name: "キャンセル",
      }),
    );
    expect(within(openDialog(TEXT[1])).getByRole("textbox")).toHaveValue(
      TEXT[1],
    );
  });

  it("saves the text with the expected text and start captured at open", async () => {
    renderSection();
    await screen.findByText(TEXT[1]);

    const dialog = openDialog(TEXT[1]);
    fireEvent.change(within(dialog).getByRole("textbox"), {
      target: { value: "直した二行目です。" },
    });
    fireEvent.click(within(dialog).getByRole("button", { name: "保存" }));

    await waitFor(() => expect(putRequests()).toHaveLength(1));
    const put = putRequests()[0];
    expect(put.url).toContain(`/files/${FILE}/transcript/utterances/1`);
    expect(put.body).toMatchObject({
      text: "直した二行目です。",
      expected_text: TEXT[1],
      expected_start: 5,
    });
    expect(put.headers["X-Lit-Drive"]).toBe(DRIVE);
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
  });
});

describe("SPEC-ADDON-126 I6 — keys and line breaks in the dialog field", () => {
  it("does not save or break the line on Return outside composition", async () => {
    renderSection();
    await screen.findByText(TEXT[1]);
    const dialog = openDialog(TEXT[1]);
    const field = within(dialog).getByRole("textbox");
    fireEvent.change(field, { target: { value: "下書きの二行目" } });

    const notPrevented = fireEvent.keyDown(field, { key: "Enter" });

    expect(notPrevented).toBe(false);
    expect(field).toHaveValue("下書きの二行目");
    expect(putRequests()).toHaveLength(0);
    expect(screen.getByRole("dialog")).toBeInTheDocument();
  });

  for (const composition of [
    { name: "the composition flag", init: { isComposing: true } },
    { name: "key code 229", init: { keyCode: 229 } },
  ]) {
    it(`leaves Return to the IME on ${composition.name}`, async () => {
      renderSection();
      await screen.findByText(TEXT[1]);
      const dialog = openDialog(TEXT[1]);
      const field = within(dialog).getByRole("textbox");
      fireEvent.change(field, { target: { value: "変換中の下書き" } });

      const notPrevented = fireEvent.keyDown(field, {
        key: "Enter",
        ...composition.init,
      });

      expect(notPrevented).toBe(true);
      expect(putRequests()).toHaveLength(0);
      expect(field).toHaveValue("変換中の下書き");
    });
  }

  it("leaves Escape to the IME while it composes", async () => {
    renderSection();
    await screen.findByText(TEXT[1]);
    const dialog = openDialog(TEXT[1]);
    const field = within(dialog).getByRole("textbox");

    const notPrevented = fireEvent.keyDown(field, {
      key: "Escape",
      isComposing: true,
    });

    expect(notPrevented).toBe(true);
    expect(screen.getByRole("dialog")).toBeInTheDocument();
  });

  it("removes a line break that arrives by paste and saves no line break", async () => {
    renderSection();
    await screen.findByText(TEXT[1]);
    const dialog = openDialog(TEXT[1]);
    const field = within(dialog).getByRole("textbox");

    fireEvent.change(field, { target: { value: "前半\n後半" } });
    expect((field as HTMLTextAreaElement).value).not.toContain("\n");

    fireEvent.click(within(dialog).getByRole("button", { name: "保存" }));
    await waitFor(() => expect(putRequests()).toHaveLength(1));
    expect(putRequests()[0].body).toMatchObject({ text: "前半後半" });
  });
});

describe("SPEC-ADDON-126 I7 — a save in flight and its failures", () => {
  it("ignores a second 保存 and every way out while the save runs", async () => {
    renderSection();
    await screen.findByText(TEXT[1]);
    const dialog = openDialog(TEXT[1]);
    const field = within(dialog).getByRole("textbox");
    fireEvent.change(field, { target: { value: "送信中の二行目" } });

    pendingPut = deferred();
    fireEvent.click(within(dialog).getByRole("button", { name: "保存" }));
    await waitFor(() => expect(putRequests()).toHaveLength(1));

    expect(field).toHaveAttribute("readonly");
    fireEvent.click(within(dialog).getByRole("button", { name: "保存" }));
    fireEvent.click(within(dialog).getByRole("button", { name: "キャンセル" }));
    fireEvent.click(within(dialog).getByRole("button", { name: "閉じる" }));
    fireEvent.keyDown(field, { key: "Escape" });

    expect(screen.getByRole("dialog")).toBeInTheDocument();
    expect(putRequests()).toHaveLength(1);

    pendingPut.resolve(
      json({ index: 1, text: "送信中の二行目", start: 5, end: 10 }),
    );
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
  });

  const FAILURES = [
    {
      name: "busy",
      status: 409,
      body: { detail: { code: "busy" } },
      message: BUSY_MESSAGE,
    },
    {
      name: "too_long",
      status: 400,
      body: { detail: { code: "too_long" } },
      message: TOO_LONG_MESSAGE,
    },
    {
      name: "last_utterance",
      status: 400,
      body: { detail: { code: "last_utterance" } },
      message: LAST_UTTERANCE_MESSAGE,
    },
    {
      name: "404",
      status: 404,
      body: { detail: "not found" },
      message: UNAVAILABLE_MESSAGE,
    },
    {
      name: "an unknown error",
      status: 500,
      body: { detail: { code: "something_else" } },
      message: SAVE_FAILED_MESSAGE,
    },
  ];

  for (const failure of FAILURES) {
    it(`keeps the dialog and the draft on ${failure.name}, with the message inside it`, async () => {
      renderSection();
      await screen.findByText(TEXT[1]);
      const dialog = openDialog(TEXT[1]);
      fireEvent.change(within(dialog).getByRole("textbox"), {
        target: { value: "失敗する下書き" },
      });

      putError = { status: failure.status, body: failure.body };
      fireEvent.click(within(dialog).getByRole("button", { name: "保存" }));

      const shown = await within(dialog).findAllByText(failure.message);
      expect(shown.length).toBeGreaterThan(0);
      expect(screen.getByRole("dialog")).toBeInTheDocument();
      expect(within(dialog).getByRole("textbox")).toHaveValue("失敗する下書き");
      expect(
        within(dialog).getByRole("button", { name: "保存" }),
      ).toBeEnabled();
      const panelErrors = screen
        .queryAllByRole("status")
        .filter((element) => !dialog.contains(element));
      expect(panelErrors).toHaveLength(0);
    });
  }

  it("closes on changed, reloads, and shows the message in the panel", async () => {
    renderSection();
    await screen.findByText(TEXT[1]);
    const dialog = openDialog(TEXT[1]);
    fireEvent.change(within(dialog).getByRole("textbox"), {
      target: { value: "競合した下書き" },
    });
    const readsBefore = transcriptReads();

    putError = { status: 409, body: { detail: { code: "changed" } } };
    fireEvent.click(within(dialog).getByRole("button", { name: "保存" }));

    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    await waitFor(() => expect(transcriptReads()).toBeGreaterThan(readsBefore));
    expect(screen.getByRole("status")).toHaveTextContent(CHANGED_MESSAGE);
  });
});

describe("SPEC-ADDON-126 B1 — the dialog suspends following", () => {
  it("stops following while the dialog is open and does not resume on close", async () => {
    const state = { currentTime: 0, paused: false };
    const mc = controller(state);
    renderSection({ mediaController: mc });
    await screen.findByText(TEXT[0]);
    tapRow(TEXT[0]);
    await waitFor(() =>
      expect(screen.getByRole("button", { current: true })).toHaveTextContent(
        TEXT[0],
      ),
    );
    expect(screen.queryByRole("button", { name: "再生位置に戻る" })).toBeNull();

    const dialog = openDialog(TEXT[0]);
    expect(
      await screen.findByRole("button", { name: "再生位置に戻る" }),
    ).toBeInTheDocument();

    fireEvent.click(within(dialog).getByRole("button", { name: "キャンセル" }));
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(
      screen.getByRole("button", { name: "再生位置に戻る" }),
    ).toBeInTheDocument();
  });
});

describe("SPEC-ADDON-126 I8 — what keeps and what discards the draft", () => {
  it("keeps the dialog and the draft through a backdrop tap", async () => {
    renderSection();
    await screen.findByText(TEXT[1]);
    const dialog = openDialog(TEXT[1]);
    const field = within(dialog).getByRole("textbox");
    fireEvent.change(field, { target: { value: "のこす下書き" } });

    const backdrop = document.querySelector('[class~="bg-black/60"]');
    expect(backdrop).not.toBeNull();
    fireEvent.click(backdrop!);

    expect(screen.getByRole("dialog")).toBeInTheDocument();
    expect(within(screen.getByRole("dialog")).getByRole("textbox")).toHaveValue(
      "のこす下書き",
    );
  });

  it("keeps the dialog, the draft and the expected text through a reload", async () => {
    const view = renderSection();
    await screen.findByText(TEXT[1]);
    const dialog = openDialog(TEXT[1]);
    fireEvent.change(within(dialog).getByRole("textbox"), {
      target: { value: "のこす下書き" },
    });

    files[FILE].chunks[1] = {
      ...files[FILE].chunks[1],
      text: "外から更新された二行目。",
    };
    await reload(view);
    await screen.findByText("外から更新された二行目。");

    expect(screen.getByRole("dialog")).toBeInTheDocument();
    expect(within(screen.getByRole("dialog")).getByRole("textbox")).toHaveValue(
      "のこす下書き",
    );
    fireEvent.click(
      within(screen.getByRole("dialog")).getByRole("button", { name: "保存" }),
    );
    await waitFor(() => expect(putRequests()).toHaveLength(1));
    expect(putRequests()[0].body).toMatchObject({
      expected_text: TEXT[1],
      expected_start: 5,
      text: "のこす下書き",
    });
  });

  it("keeps the dialog through a pointer-mode change", async () => {
    renderSection();
    await screen.findByText(TEXT[1]);
    const dialog = openDialog(TEXT[1]);
    fireEvent.change(within(dialog).getByRole("textbox"), {
      target: { value: "のこす下書き" },
    });

    setPointer("fine");

    expect(screen.getByRole("dialog")).toBeInTheDocument();
    expect(within(screen.getByRole("dialog")).getByRole("textbox")).toHaveValue(
      "のこす下書き",
    );

    fireEvent.click(
      within(screen.getByRole("dialog")).getByRole("button", { name: "キャンセル" }),
    );
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(stripQuotes()).toHaveLength(0);
  });

  const DISMISSALS = [
    {
      name: "キャンセル",
      dismiss: (dialog: HTMLElement) =>
        fireEvent.click(within(dialog).getByRole("button", { name: "キャンセル" })),
    },
    {
      name: "the close button",
      dismiss: (dialog: HTMLElement) =>
        fireEvent.click(within(dialog).getByRole("button", { name: "閉じる" })),
    },
    {
      name: "Escape",
      dismiss: (dialog: HTMLElement) => {
        const field = within(dialog).getByRole("textbox");
        act(() => field.focus());
        fireEvent.keyDown(field, { key: "Escape" });
      },
    },
  ];

  for (const dismissal of DISMISSALS) {
    it(`discards the draft on ${dismissal.name}`, async () => {
      renderSection();
      await screen.findByText(TEXT[1]);
      const dialog = openDialog(TEXT[1]);
      fireEvent.change(within(dialog).getByRole("textbox"), {
        target: { value: "すてる下書き" },
      });

      dismissal.dismiss(dialog);

      expect(screen.queryByRole("dialog")).toBeNull();
      expect(putRequests()).toHaveLength(0);

      const reopened = openDialog(TEXT[1]);
      expect(within(reopened).getByRole("textbox")).toHaveValue(TEXT[1]);
    });
  }

  it("closes and discards the draft when the file changes", async () => {
    const view = renderSection();
    await screen.findByText(TEXT[1]);
    const dialog = openDialog(TEXT[1]);
    fireEvent.change(within(dialog).getByRole("textbox"), {
      target: { value: "すてる下書き" },
    });

    view.rerender(ui({ fileId: OTHER_FILE }));
    await screen.findByText(OTHER_FILE_TEXT);

    expect(screen.queryByRole("dialog")).toBeNull();
    expect(stripQuotes()).toHaveLength(0);
  });

  it("closes and discards the draft when the source switches by itself", async () => {
    const view = renderSection();
    await screen.findByText(TEXT[1]);
    const dialog = openDialog(TEXT[1]);
    fireEvent.change(within(dialog).getByRole("textbox"), {
      target: { value: "すてる下書き" },
    });

    files[FILE].chunks = [];
    await reload(view);
    await screen.findByText(WORDS_TEXT);

    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(stripQuotes()).toHaveLength(0);
  });
});

describe("SPEC-ADDON-125 I9 — the fine path is unchanged", () => {
  for (const mode of ["fine", "unknown"] as const) {
    it(`on a ${mode} pointer a tap selects nothing and 修正 opens the inline field`, async () => {
      setPointer(mode);
      const state = { currentTime: 0, paused: true };
      const mc = controller(state);
      renderSection({ mediaController: mc });
      await screen.findByText(TEXT[1]);

      tapRow(TEXT[1]);

      expect(mc.seek).toHaveBeenCalledWith(5);
      expect(mc.play).toHaveBeenCalled();
      expect(stripQuotes()).toHaveLength(0);
      expect(screen.queryByRole("dialog")).toBeNull();

      fireEvent.click(within(rowOf(TEXT[1])).getByRole("button", { name: "修正" }));

      expect(screen.getByRole("textbox")).toHaveValue(TEXT[1]);
      expect(screen.queryByRole("dialog")).toBeNull();
    });
  }

  for (const mode of ["fine", "unknown"] as const) {
    it(`on a ${mode} pointer Return saves the inline field`, async () => {
      setPointer(mode);
      renderSection();
      await screen.findByText(TEXT[1]);
      fireEvent.click(within(rowOf(TEXT[1])).getByRole("button", { name: "修正" }));
      const field = screen.getByRole("textbox");
      fireEvent.change(field, { target: { value: "直した二行目です。" } });

      fireEvent.keyDown(field, { key: "Enter" });

      await waitFor(() => expect(putRequests()).toHaveLength(1));
      expect(putRequests()[0].body).toMatchObject({
        text: "直した二行目です。",
        expected_text: TEXT[1],
        expected_start: 5,
      });
    });
  }

  for (const mode of ["fine", "unknown"] as const) {
    it(`on a ${mode} pointer Escape cancels the inline field`, async () => {
      setPointer(mode);
      renderSection();
      await screen.findByText(TEXT[1]);
      fireEvent.click(within(rowOf(TEXT[1])).getByRole("button", { name: "修正" }));
      const field = screen.getByRole("textbox");
      fireEvent.change(field, { target: { value: "すてる下書き" } });

      fireEvent.keyDown(field, { key: "Escape" });

      expect(screen.queryByRole("textbox")).toBeNull();
      expect(putRequests()).toHaveLength(0);
    });
  }
});
