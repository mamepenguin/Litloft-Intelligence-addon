/**
 * SPEC-ID: SPEC-ADDON-065
 *
 * The transcript panel's edit mode, its marks and the restore control.
 * The API module is left real and `fetch` is intercepted at the URL the
 * spec fixes, so these tests do not depend on the name of a new wrapper.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";

vi.mock("next-intl", async () => await import("./testI18nJa"));

const addon = vi.hoisted(() => ({
  features: { transcript_refine: "manual" as string | false },
}));
vi.mock("@/components/AddonSlotsProvider", () => ({
  useAddonStatus: () => addon,
  useAddonSlots: () => ({ slots: {} }),
}));

const ws = vi.hoisted(() => ({ event: null as null | { event: string; data: unknown } }));
vi.mock("@/hooks/useWebSocket", () => ({
  useWebSocket: (filter?: string) =>
    !filter || ws.event?.event === filter ? ws.event : null,
}));

import TranscriptSection from "@/addons/intelligence/TranscriptSection";
import { clearTranscriptScroll } from "@/addons/intelligence/transcriptScroll";

type Chunk = {
  index: number;
  text: string;
  start: number;
  end: number;
  correctedAt?: string | null;
  wordsPending?: boolean;
  refinedAt?: string | null;
};

type Picture = {
  chunks: Chunk[];
  has_original: boolean;
  corrected: boolean;
  busy: boolean;
  followups_pending: boolean;
};

let state: Picture;
let requests: { method: string; url: string; body: unknown }[];

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

function freshState(overrides: Partial<Picture> = {}): Picture {
  return {
    chunks: [
      { index: 0, text: "これは未修正の文章です。", start: 0, end: 5 },
      { index: 1, text: "二行目の文章です。", start: 5, end: 10 },
    ],
    has_original: false,
    corrected: false,
    busy: false,
    followups_pending: false,
    ...overrides,
  };
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
  requests.push({ method, url, body });

  if (url.includes("/subtitles/")) return vtt("外部字幕の一行目。");
  if (url.includes("/subtitles.vtt")) return vtt("単語の字幕。");
  if (url.includes("/transcript/restore") && method === "POST") {
    state = freshState();
    return json({ available: true, file_id: "abc", drive: "family", language: "ja", ...state });
  }
  if (url.includes("/transcript/utterances/") && method === "PUT") {
    const index = Number(url.split("/utterances/")[1]);
    const row = state.chunks.find((chunk) => chunk.index === index)!;
    row.text = body.text;
    row.correctedAt = "2026-11-06T00:00:00Z";
    row.wordsPending = true;
    state.has_original = true;
    state.corrected = true;
    return json(row);
  }
  if (url.endsWith("/transcript")) {
    return json({
      available: true,
      file_id: "abc",
      drive: "family",
      language: "ja",
      ...state,
    });
  }
  return json({ detail: "not found" }, 404);
});

vi.stubGlobal("fetch", fetchMock);

const EXTERNAL_TRACKS = [
  { index: 0, language: "ja", format: "vtt", label: "日本語" },
];

function renderSection() {
  return render(
    <TranscriptSection
      fileId="abc"
      drive="family"
      subtitles={EXTERNAL_TRACKS}
    />,
  );
}

beforeEach(() => {
  state = freshState();
  requests = [];
  fetchMock.mockClear();
  ws.event = null;
  addon.features.transcript_refine = "manual";
  clearTranscriptScroll();
});

afterEach(() => {
  cleanup();
});

describe("SPEC-ADDON-065 — edit controls", () => {
  it("offers 修正 on every 分割テキスト row", async () => {
    renderSection();

    await screen.findByText("これは未修正の文章です。");
    const edits = screen.getAllByRole("button", { name: "修正" });
    expect(edits).toHaveLength(2);
  });

  it("offers no edit in the 単語 or 外部字幕 views", async () => {
    renderSection();
    await screen.findByText("これは未修正の文章です。");

    fireEvent.click(await screen.findByRole("button", { name: "単語" }));
    expect(screen.queryByRole("button", { name: "修正" })).not.toBeInTheDocument();

    // The 外部字幕 option appears because the component was given one
    // external subtitle track and its VTT answered.
    fireEvent.click(await screen.findByRole("button", { name: "外部字幕" }));
    expect(screen.queryByRole("button", { name: "修正" })).not.toBeInTheDocument();
  }, 10000);

  it("turns the row into a field holding its text with 保存 and キャンセル", async () => {
    renderSection();
    await screen.findByText("これは未修正の文章です。");

    fireEvent.click(screen.getAllByRole("button", { name: "修正" })[0]);

    const field = screen.getByRole("textbox");
    expect(field).toHaveValue("これは未修正の文章です。");
    expect(screen.getByRole("button", { name: "保存" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "キャンセル" })).toBeInTheDocument();
  });

  it("sends the expected text and start and shows 修正済み", async () => {
    renderSection();
    await screen.findByText("これは未修正の文章です。");
    fireEvent.click(screen.getAllByRole("button", { name: "修正" })[0]);

    const field = screen.getByRole("textbox");
    fireEvent.change(field, { target: { value: "直した文章です。" } });
    fireEvent.keyDown(field, { key: "Enter" });

    await waitFor(() => {
      const put = requests.find(
        (request) =>
          request.method === "PUT" &&
          request.url.includes("/transcript/utterances/0"),
      );
      expect(put).toBeTruthy();
      expect(put!.body).toMatchObject({
        text: "直した文章です。",
        expected_text: "これは未修正の文章です。",
        expected_start: 0,
      });
    });
    expect(await screen.findByText("修正済み")).toBeInTheDocument();
  });

  it("keeps at most one row in edit mode", async () => {
    renderSection();
    await screen.findByText("これは未修正の文章です。");

    fireEvent.click(screen.getAllByRole("button", { name: "修正" })[0]);
    fireEvent.click(screen.getByRole("button", { name: "修正" }));

    expect(screen.getAllByRole("textbox")).toHaveLength(1);
    expect(screen.getByRole("textbox")).toHaveValue("二行目の文章です。");
  });

  it("cancels with Escape without writing", async () => {
    renderSection();
    await screen.findByText("これは未修正の文章です。");
    fireEvent.click(screen.getAllByRole("button", { name: "修正" })[0]);

    fireEvent.keyDown(screen.getByRole("textbox"), { key: "Escape" });

    expect(screen.queryByRole("textbox")).not.toBeInTheDocument();
    expect(
      requests.some((request) => request.method === "PUT"),
    ).toBe(false);
    expect(screen.getByText("これは未修正の文章です。")).toBeInTheDocument();
  });
});

describe("SPEC-ADDON-065 — marks and busy", () => {
  it("shows 修正済み next to AI整形済み when both apply", async () => {
    state = freshState({
      corrected: true,
      chunks: [
        {
          index: 0,
          text: "直した文章です。",
          start: 0,
          end: 5,
          correctedAt: "2026-11-06T00:00:00Z",
          refinedAt: "2026-11-06T00:00:00Z",
        },
      ],
    });
    renderSection();

    expect(await screen.findByText("修正済み")).toBeInTheDocument();
    expect(screen.getByText("AI整形済み")).toBeInTheDocument();
  });

  it("shows 修正 and 元の結果に戻す disabled while busy", async () => {
    state = freshState({ busy: true, has_original: true });
    renderSection();
    await screen.findByText("これは未修正の文章です。");

    expect(screen.getByRole("button", { name: "AIで整える" })).toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: "修正" })[0]).toBeDisabled();
    expect(
      screen.getByRole("button", { name: "元の結果に戻す" }),
    ).toBeDisabled();
  });

  it("offers 元の結果に戻す even when Refine is disabled", async () => {
    addon.features.transcript_refine = false;
    state = freshState({ has_original: true });
    renderSection();
    await screen.findByText("これは未修正の文章です。");

    expect(
      screen.getByRole("button", { name: "元の結果に戻す" }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "AIで整える" }),
    ).not.toBeInTheDocument();
  });
});

describe("SPEC-ADDON-065 — restore", () => {
  it("offers 元の結果に戻す only in the 分割テキスト view", async () => {
    state = freshState({ has_original: true });
    renderSection();
    await screen.findByText("これは未修正の文章です。");

    expect(
      screen.getByRole("button", { name: "元の結果に戻す" }),
    ).toBeInTheDocument();

    fireEvent.click(await screen.findByRole("button", { name: "単語" }));
    expect(
      screen.queryByRole("button", { name: "元の結果に戻す" }),
    ).not.toBeInTheDocument();
  }, 10000);

  it("confirms in Japanese and posts the restore", async () => {
    state = freshState({ has_original: true });
    renderSection();
    await screen.findByText("これは未修正の文章です。");

    fireEvent.click(screen.getByRole("button", { name: "元の結果に戻す" }));
    expect(
      screen.getByText(
        "修正をすべて取り消し、機械が出した元の結果に戻します。",
      ),
    ).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "戻す" }));

    await waitFor(() =>
      expect(
        requests.some(
          (request) =>
            request.method === "POST" &&
            request.url.includes("/transcript/restore"),
        ),
      ).toBe(true),
    );
    expect(
      await screen.findByRole("button", { name: "元の結果に戻す" }),
    ).not.toBeInTheDocument();
  });

  it("closes the confirmation on キャンセル without posting", async () => {
    state = freshState({ has_original: true });
    renderSection();
    await screen.findByText("これは未修正の文章です。");
    fireEvent.click(screen.getByRole("button", { name: "元の結果に戻す" }));

    fireEvent.click(screen.getByRole("button", { name: "キャンセル" }));

    expect(
      requests.some(
        (request) =>
          request.method === "POST" &&
          request.url.includes("/transcript/restore"),
      ),
    ).toBe(false);
  });
});

describe("SPEC-ADDON-065 — an update event refetches", () => {
  it("fetches the transcript and the subtitles again", async () => {
    const view = renderSection();
    await screen.findByText("これは未修正の文章です。");
    const transcriptBefore = requests.filter(
      (request) => request.method === "GET" && request.url.endsWith("/transcript"),
    ).length;
    const subtitlesBefore = requests.filter(
      (request) => request.method === "GET" && request.url.includes("/subtitles.vtt"),
    ).length;

    state = freshState({
      corrected: true,
      chunks: [
        {
          index: 0,
          text: "更新後の文章です。",
          start: 0,
          end: 5,
          correctedAt: "2026-11-06T00:00:00Z",
        },
      ],
    });
    ws.event = {
      event: "intelligence.transcript.updated",
      data: { file_id: "abc", drive: "family", reason: "corrected" },
    };
    view.rerender(
      <TranscriptSection
        fileId="abc"
        drive="family"
        subtitles={EXTERNAL_TRACKS}
      />,
    );

    await waitFor(() => {
      const transcriptAfter = requests.filter(
        (request) => request.method === "GET" && request.url.endsWith("/transcript"),
      ).length;
      const subtitlesAfter = requests.filter(
        (request) => request.method === "GET" && request.url.includes("/subtitles.vtt"),
      ).length;
      expect(transcriptAfter).toBeGreaterThan(transcriptBefore);
      expect(subtitlesAfter).toBeGreaterThan(subtitlesBefore);
    });
    expect(await screen.findByText("更新後の文章です。")).toBeInTheDocument();
  }, 10000);
});
