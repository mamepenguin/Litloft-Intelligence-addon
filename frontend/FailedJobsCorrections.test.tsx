/**
 * SPEC-ID: SPEC-ADDON-068
 *
 * The failed jobs list does not know a file is corrected. Its 409
 * `corrected` answer opens the same confirmation, and the retry is sent
 * with the flag only after 続ける.
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

import FailedJobsModal from "@/addons/intelligence/FailedJobsModal";

let requests: { method: string; url: string; body: unknown }[];
let reindexAnswers: Response[];

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
      total: 1,
      limit: 50,
      offset: 0,
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
  reindexAnswers = [
    json({ detail: { code: "corrected" } }, 409),
    json({ status: "accepted", file_id: "abc12345", tasks_reset: ["whisper"] }),
  ];
  fetchMock.mockClear();
});

afterEach(() => {
  cleanup();
});

describe("SPEC-ADDON-068 — failed jobs retry of a corrected file", () => {
  it("confirms on 409 corrected and resends with the flag", async () => {
    render(<FailedJobsModal open onClose={() => {}} />);
    await screen.findByText("movie.mp4");

    fireEvent.click(screen.getByRole("button", { name: "再試行" }));

    expect(
      await screen.findByText(
        "このファイルには手で直した箇所があります。再生成すると、修正と保存してある元の結果は消えます。",
      ),
    ).toBeInTheDocument();
    const posts = requests.filter((request) => request.url.includes("/reindex"));
    expect(posts).toHaveLength(1);
    expect(posts[0].body).toEqual({ tasks: ["whisper"] });

    fireEvent.click(screen.getByRole("button", { name: "dialog-confirm" }));

    await waitFor(() => {
      const after = requests.filter((request) => request.url.includes("/reindex"));
      expect(after).toHaveLength(2);
      expect(after[1].body).toEqual({
        tasks: ["whisper"],
        overwrite_corrections: true,
      });
    });
  });

  it("cancelling sends nothing more", async () => {
    render(<FailedJobsModal open onClose={() => {}} />);
    await screen.findByText("movie.mp4");
    fireEvent.click(screen.getByRole("button", { name: "再試行" }));
    await screen.findByTestId("confirm-dialog");

    fireEvent.click(screen.getByRole("button", { name: "dialog-cancel" }));

    const posts = requests.filter((request) => request.url.includes("/reindex"));
    expect(posts).toHaveLength(1);
  });
});
