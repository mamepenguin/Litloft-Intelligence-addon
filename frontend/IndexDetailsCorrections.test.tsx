/**
 * SPEC-ID: SPEC-ADDON-068
 *
 * Index details' confirmation for a corrected file and the flag it sends.
 * The API module is real; the request body is the observable.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";

vi.mock("next-intl", async () => await import("./testI18nJa"));

vi.mock("@/components/ConfirmDialog", () => ({
  ConfirmDialog: ({
    open,
    title,
    message,
    onConfirm,
    onCancel,
  }: {
    open: boolean;
    title?: string;
    message?: string;
    onConfirm: () => void;
    onCancel: () => void;
  }) =>
    open ? (
      <div data-testid="confirm-dialog">
        <span data-testid="confirm-title">{title}</span>
        <span data-testid="confirm-message">{message}</span>
        <button onClick={onConfirm}>dialog-confirm</button>
        <button onClick={onCancel}>dialog-cancel</button>
      </div>
    ) : null,
}));

import IndexDetailsMenuItem from "@/addons/intelligence/IndexDetailsMenuItem";
import { ShortcutsProvider } from "@/components/ShortcutsProvider";

let corrected: { transcript: boolean; picture_text: boolean };
let redoEnded: { transcript: boolean; picture_text: boolean };
let requests: { method: string; url: string; body: unknown }[];

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
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

  if (url.includes("/index-details")) {
    return json({
      file_id: "abc12345",
      drive: "family",
      filename: "movie.mp4",
      image_ocr: "on_index",
      status: {
        metadata: true,
        clip: true,
        whisper: true,
        text: false,
        ocr: false,
      },
      indexed_at: "2026-11-06T00:00:00Z",
      embeddings: {},
      provider_stats: {},
      corrected,
      redo_ended_without_result: redoEnded,
    });
  }
  if (url.includes("/reindex") && method === "POST") {
    return json({ status: "accepted", file_id: "abc12345", tasks_reset: body.tasks });
  }
  return json({ detail: "not found" }, 404);
});

vi.stubGlobal("fetch", fetchMock);

function renderEntry() {
  return render(
    <ShortcutsProvider>
      <IndexDetailsMenuItem
        fileId="abc12345"
        drive="family"
        mimeType="video/mp4"
        fileType="video"
      />
    </ShortcutsProvider>,
  );
}

beforeEach(() => {
  corrected = { transcript: false, picture_text: false };
  redoEnded = { transcript: false, picture_text: false };
  requests = [];
  fetchMock.mockClear();
});

afterEach(() => {
  cleanup();
});

describe("SPEC-ADDON-068 — a corrected transcript", () => {
  it("shows the correction warning instead of the usual one", async () => {
    corrected = { transcript: true, picture_text: false };
    renderEntry();
    fireEvent.click(
      screen.getByRole("menuitem", { name: /インデックス詳細|Index Details/i }),
    );
    await waitFor(() =>
      expect(document.querySelector('[data-task-row="whisper"]')).not.toBeNull(),
    );
    const row = document.querySelector('[data-task-row="whisper"]') as HTMLElement;

    fireEvent.click(within(row).getByRole("button", { name: "再生成" }));

    expect(screen.getByTestId("confirm-message")).toHaveTextContent(
      "このファイルには手で直した箇所があります。再生成すると、修正と保存してある元の結果は消えます。",
    );
    expect(screen.getAllByTestId("confirm-dialog")).toHaveLength(1);
  });

  it("sends overwrite_corrections only after 続ける", async () => {
    corrected = { transcript: true, picture_text: false };
    renderEntry();
    fireEvent.click(
      screen.getByRole("menuitem", { name: /インデックス詳細|Index Details/i }),
    );
    await waitFor(() =>
      expect(document.querySelector('[data-task-row="whisper"]')).not.toBeNull(),
    );
    const row = document.querySelector('[data-task-row="whisper"]') as HTMLElement;

    fireEvent.click(within(row).getByRole("button", { name: "再生成" }));
    expect(requests.some((request) => request.url.includes("/reindex"))).toBe(false);

    fireEvent.click(screen.getByRole("button", { name: "dialog-confirm" }));

    await waitFor(() => {
      const post = requests.find((request) => request.url.includes("/reindex"));
      expect(post).toBeTruthy();
      expect(post!.body).toEqual({
        tasks: ["whisper"],
        overwrite_corrections: true,
      });
    });
  });

  it("does not send when the confirmation is cancelled", async () => {
    corrected = { transcript: true, picture_text: false };
    renderEntry();
    fireEvent.click(
      screen.getByRole("menuitem", { name: /インデックス詳細|Index Details/i }),
    );
    await waitFor(() =>
      expect(document.querySelector('[data-task-row="whisper"]')).not.toBeNull(),
    );
    const row = document.querySelector('[data-task-row="whisper"]') as HTMLElement;
    fireEvent.click(within(row).getByRole("button", { name: "再生成" }));

    fireEvent.click(screen.getByRole("button", { name: "dialog-cancel" }));

    expect(requests.some((request) => request.url.includes("/reindex"))).toBe(false);
  });

  it("shows the ended-without-result line on the task row", async () => {
    corrected = { transcript: true, picture_text: false };
    redoEnded = { transcript: true, picture_text: false };
    renderEntry();
    fireEvent.click(
      screen.getByRole("menuitem", { name: /インデックス詳細|Index Details/i }),
    );
    await waitFor(() =>
      expect(document.querySelector('[data-task-row="whisper"]')).not.toBeNull(),
    );
    const row = document.querySelector('[data-task-row="whisper"]') as HTMLElement;

    expect(
      within(row).getByText(
        "再生成できませんでした。修正はそのまま残っています。",
      ),
    ).toBeInTheDocument();
  });
});
