/**
 * Collection errors on the operator surfaces (SPEC-ADDON-068 C8).
 *
 * SPEC-ID: SPEC-ADDON-068
 */

import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("next-intl", async () => await import("./testI18nJa"));
vi.mock("@/components/ConfirmDialog", () => ({
  ConfirmDialog: ({
    open,
    message,
    onConfirm,
    onCancel,
  }: {
    open: boolean;
    message?: string;
    onConfirm: () => void;
    onCancel: () => void;
  }) =>
    open ? (
      <div data-testid="confirm-dialog">
        <span data-testid="confirm-message">{message}</span>
        <button onClick={onConfirm}>dialog-confirm</button>
        <button onClick={onCancel}>dialog-cancel</button>
      </div>
    ) : null,
}));

import IndexDetailsMenuItem from "@/addons/intelligence/IndexDetailsMenuItem";
import FailedJobsModal from "@/addons/intelligence/FailedJobsModal";
import { ShortcutsProvider } from "@/components/ShortcutsProvider";

let requests: { method: string; url: string; body: unknown }[];
let reindexAnswers: Response[];

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
  const url = String(input);
  const method = init?.method ?? "GET";
  const body = init?.body ? JSON.parse(String(init.body)) : null;
  requests.push({ method, url, body });
  if (url.includes("/index-details")) {
    return json({
      file_id: "abc12345",
      drive: "family",
      filename: "movie.mp4",
      image_ocr: "on_index",
      status: { metadata: true, clip: true, whisper: true, text: false, ocr: true },
      indexed_at: "2026-11-06T00:00:00Z",
      embeddings: {},
      provider_stats: {},
      corrected: { transcript: false, picture_text: false },
      redo_ended_without_result: { transcript: false, picture_text: false },
    });
  }
  if (url.includes("/admin/failed-jobs")) {
    return json({
      items: [{
        file_id: "abc12345",
        filename: "movie.mp4",
        drive: "family",
        job_kind: "transcription",
        provider: "whisper_local",
        error_class: "FatalError",
        error_message_excerpt: "provider failed",
        attempted_at: "2026-11-06T00:00:00Z",
        attempts: 2,
      }],
      total: 1, limit: 50, offset: 0,
    });
  }
  if (url.includes("/reindex")) {
    return reindexAnswers.shift() ?? json({ status: "accepted" });
  }
  return json({ detail: "not found" }, 404);
});
vi.stubGlobal("fetch", fetchMock);

beforeEach(() => {
  requests = [];
  reindexAnswers = [];
  fetchMock.mockClear();
});
afterEach(() => cleanup());

describe("SPEC-ADDON-068 — operator surfaces on a corrected file", () => {
  it("shows the corrected warning after a 409 corrected and resends with the flag", async () => {
    reindexAnswers = [
      json({ detail: { code: "corrected" } }, 409),
      json({ status: "accepted" }),
    ];
    render(
      <ShortcutsProvider>
        <IndexDetailsMenuItem
          fileId="abc12345"
          drive="family"
          mimeType="image/png"
          fileType="image"
        />
      </ShortcutsProvider>,
    );
    fireEvent.click(screen.getByRole("menuitem", { name: /インデックス詳細/ }));
    await waitFor(() =>
      expect(document.querySelector('[data-task-row="ocr"]')).not.toBeNull(),
    );
    const row = document.querySelector('[data-task-row="ocr"]') as HTMLElement;
    fireEvent.click(within(row).getByRole("button", { name: "再生成" }));
    fireEvent.click(screen.getByRole("button", { name: "dialog-confirm" }));

    expect(
      await screen.findByText(
        "このファイルには手で直した箇所があります。再生成すると、修正と保存してある元の結果は消えます。",
      ),
    ).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "dialog-confirm" }));

    await waitFor(() => {
      const posts = requests.filter((r) => r.url.includes("/reindex"));
      expect(posts.at(-1)!.body).toMatchObject({ overwrite_corrections: true });
    });
  });

  it("shows the busy text when a failed-job retry answers 409 busy", async () => {
    reindexAnswers = [json({ detail: { code: "busy" } }, 409)];
    render(<FailedJobsModal open onClose={() => {}} />);
    await screen.findByText("movie.mp4");
    fireEvent.click(screen.getByRole("button", { name: "再試行" }));

    expect(
      await screen.findByText("処理中のため、いまは操作できません。"),
    ).toBeInTheDocument();
  });
});
