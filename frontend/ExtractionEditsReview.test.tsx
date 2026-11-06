/**
 * The review defects that only the component can show.
 *
 * SPEC-ID: SPEC-ADDON-070
 * SPEC-ID: SPEC-ADDON-067
 */

import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("next-intl", async () => await import("./testI18nJa"));
vi.mock("@/components/AddonSlotsProvider", () => ({
  useAddonStatus: () => ({ features: { transcript_refine: "manual" } }),
  useAddonSlots: () => ({ slots: {} }),
}));

const ws = vi.hoisted(() => ({ event: null as null | { event: string; data: unknown } }));
vi.mock("@/hooks/useWebSocket", () => ({
  useWebSocket: (filter?: string) =>
    !filter || ws.event?.event === filter ? ws.event : null,
}));

import TranscriptSection from "@/addons/intelligence/TranscriptSection";
import PictureTextSection from "@/addons/intelligence/PictureTextSection";

type Chunk = { index: number; text: string; start: number; end: number; correctedAt?: string | null; refinedAt?: string | null };
type FileState = {
  chunks: Chunk[];
  has_original: boolean;
  corrected: boolean;
  busy: boolean;
  followups_pending: boolean;
  picture: string;
};
let files: Record<string, FileState>;
let requests: { method: string; url: string; body: unknown }[];
let hold: Promise<void> | null = null;
let putStatus: { status: number; body: unknown } | null = null;
let refineStatus: { status: number; body: unknown } | null = null;

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}
function state(id: string): FileState {
  return files[id];
}

const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
  const url = String(input);
  const method = init?.method ?? "GET";
  const body = init?.body ? JSON.parse(String(init.body)) : null;
  requests.push({ method, url, body });
  const id = url.split("/files/")[1]?.split("/")[0] ?? "";
  if (url.includes("subtitles")) return new Response("", { status: 404 });
  if (url.endsWith("/transcript") && method === "GET") {
    if (hold) await hold;
    return json({
      available: true,
      file_id: id,
      drive: "d",
      language: "ja",
      ...state(id),
    });
  }
  if (url.includes("/utterances/") && method === "PUT") {
    if (putStatus) return json(putStatus.body, putStatus.status);
    const row = state(id).chunks.find((c) => url.endsWith(String(c.index)))!;
    row.text = body.text;
    row.correctedAt = "2026-11-06T00:00:00Z";
    state(id).has_original = true;
    state(id).corrected = true;
    return json(row);
  }
  if (url.includes("/transcript/restore")) {
    const s = state(id);
    s.has_original = false;
    s.corrected = false;
    return json({ available: true, file_id: id, drive: "d", language: "ja", ...s });
  }
  if (url.includes("/refine/files/")) {
    if (refineStatus) return json(refineStatus.body, refineStatus.status);
    return json({ job_id: "j", chunk_count: 1 });
  }
  if (url.endsWith("/picture-text") && method === "GET") {
    return json({
      file_id: id,
      text: state(id).picture,
      corrected: state(id).corrected,
      has_original: state(id).has_original,
      busy: state(id).busy,
    });
  }
  if (url.endsWith("/picture-text") && method === "PUT") {
    state(id).picture = body.text;
    state(id).corrected = true;
    state(id).has_original = true;
    return json({
      file_id: id,
      text: body.text,
      corrected: true,
      has_original: true,
      busy: false,
    });
  }
  if (url.includes("/picture-text/restore")) {
    const s = state(id);
    s.corrected = false;
    s.has_original = false;
    s.picture = "machine text";
    return json({
      file_id: id,
      text: s.picture,
      corrected: false,
      has_original: false,
      busy: false,
    });
  }
  return json({ detail: "not found" }, 404);
});
vi.stubGlobal("fetch", fetchMock);

function fresh(id: string, picture = "machine text"): FileState {
  return {
    chunks: [
      { index: 0, text: `text-${id}-0`, start: 0, end: 5 },
      { index: 1, text: `text-${id}-1`, start: 5, end: 10 },
    ],
    has_original: false,
    corrected: false,
    busy: false,
    followups_pending: false,
    picture,
  };
}

function transcript(id: string) {
  return render(<TranscriptSection fileId={id} drive="d" />);
}
function picture(id: string) {
  return render(<PictureTextSection fileId={id} drive="d" />);
}

beforeEach(() => {
  files = { A: fresh("A"), B: fresh("B", "machine B") };
  requests = [];
  hold = null;
  putStatus = null;
  refineStatus = null;
  ws.event = null;
  fetchMock.mockClear();
});
afterEach(() => cleanup());

describe("SPEC-ADDON-070 — edit state follows the file", () => {
  it("drops an open transcript edit when the file changes", async () => {
    const view = transcript("A");
    await screen.findByText("text-A-0");
    fireEvent.click(screen.getAllByRole("button", { name: "修正" })[0]);
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "typed for A" } });

    view.rerender(<TranscriptSection fileId="B" drive="d" />);
    await screen.findByText("text-B-1");

    expect(screen.queryByRole("textbox")).not.toBeInTheDocument();
  });

  it("drops an open restore confirmation when the file changes", async () => {
    files.A.has_original = true;
    const view = transcript("A");
    await screen.findByText("text-A-0");
    fireEvent.click(screen.getByRole("button", { name: "元の結果に戻す" }));
    expect(
      screen.getByText("修正をすべて取り消し、機械が出した元の結果に戻します。"),
    ).toBeInTheDocument();

    view.rerender(<TranscriptSection fileId="B" drive="d" />);
    await screen.findByText("text-B-1");

    expect(
      screen.queryByText("修正をすべて取り消し、機械が出した元の結果に戻します。"),
    ).not.toBeInTheDocument();
  });

  it("drops a picture draft and its confirmation when the picture changes", async () => {
    files.A.has_original = true;
    const view = picture("A");
    await screen.findByText("machine text");
    fireEvent.click(screen.getByRole("button", { name: "修正" }));
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "draft for A" } });

    view.rerender(<PictureTextSection fileId="B" drive="d" />);
    await screen.findByText("machine B");

    expect(screen.queryByRole("textbox")).not.toBeInTheDocument();
  });

  it("keeps the panel and the field during a refetch", async () => {
    const view = transcript("A");
    await screen.findByText("text-A-0");
    fireEvent.click(screen.getAllByRole("button", { name: "修正" })[1]);
    let release!: () => void;
    hold = new Promise((resolve) => {
      release = resolve;
    });
    ws.event = {
      event: "intelligence.transcript.updated",
      data: { file_id: "A", reason: "corrected" },
    };
    act(() => {
      view.rerender(<TranscriptSection fileId="A" drive="d" />);
    });

    expect(screen.getByRole("textbox")).toHaveValue("text-A-1");
    expect(screen.getByText("text-A-0")).toBeInTheDocument();
    await act(async () => {
      release();
    });
  });

  it("sends the text shown when editing began", async () => {
    const view = transcript("A");
    await screen.findByText("text-A-0");
    fireEvent.click(screen.getAllByRole("button", { name: "修正" })[0]);
    files.A.chunks[0].text = "changed on the server";
    ws.event = {
      event: "intelligence.transcript.updated",
      data: { file_id: "A", reason: "replaced" },
    };
    const before = requests.filter(
      (r) => r.method === "GET" && r.url.endsWith("/transcript"),
    ).length;
    act(() => {
      view.rerender(<TranscriptSection fileId="A" drive="d" />);
    });
    await waitFor(() => {
      const after = requests.filter(
        (r) => r.method === "GET" && r.url.endsWith("/transcript"),
      ).length;
      expect(after).toBeGreaterThan(before);
    });
    fireEvent.keyDown(screen.getByRole("textbox"), { key: "Enter" });

    await waitFor(() => {
      const put = requests.find(
        (r) => r.method === "PUT" && r.url.includes("/utterances/0"),
      );
      expect(put!.body).toMatchObject({ expected_text: "text-A-0" });
    });
  });

  it("closes the field and refetches on 409 changed", async () => {
    putStatus = { status: 409, body: { detail: { code: "changed" } } };
    const view = transcript("A");
    await screen.findByText("text-A-0");
    fireEvent.click(screen.getAllByRole("button", { name: "修正" })[0]);
    const before = requests.filter(
      (r) => r.method === "GET" && r.url.endsWith("/transcript"),
    ).length;
    fireEvent.keyDown(screen.getByRole("textbox"), { key: "Enter" });

    await waitFor(() => {
      expect(screen.queryByRole("textbox")).not.toBeInTheDocument();
      const after = requests.filter(
        (r) => r.method === "GET" && r.url.endsWith("/transcript"),
      ).length;
      expect(after).toBeGreaterThan(before);
    });
  });

  it("keeps a draft when another line is opened", async () => {
    transcript("A");
    await screen.findByText("text-A-0");
    fireEvent.click(screen.getAllByRole("button", { name: "修正" })[0]);
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "draft A0" } });
    fireEvent.click(screen.getByRole("button", { name: "修正" }));
    expect(screen.getByRole("textbox")).toHaveValue("text-A-1");
    fireEvent.click(screen.getByRole("button", { name: "修正" }));
    expect(screen.getByRole("textbox")).toHaveValue("draft A0");
  });
});

describe("SPEC-ADDON-070 — Refine and picture events", () => {
  it("opens the corrected-file confirmation on 409 corrected", async () => {
    refineStatus = { status: 409, body: { detail: { code: "corrected" } } };
    transcript("A");
    await screen.findByText("text-A-0");
    fireEvent.click(screen.getByRole("button", { name: "AIで整える" }));

    expect(
      await screen.findByText(
        "このファイルには手で直した箇所があります。AIで整えると、修正後の文章も書き換わります。",
      ),
    ).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "続ける" }));

    await waitFor(() => {
      const posts = requests.filter((r) => r.url.includes("/refine/files/"));
      expect(posts.at(-1)!.body).toMatchObject({ overwrite_corrections: true });
    });
  });

  it("shows the busy text on 409 busy", async () => {
    refineStatus = { status: 409, body: { detail: { code: "busy" } } };
    transcript("A");
    await screen.findByText("text-A-0");
    fireEvent.click(screen.getByRole("button", { name: "AIで整える" }));

    expect(
      await screen.findByText("処理中のため、いまは操作できません。"),
    ).toBeInTheDocument();
    expect(
      screen.queryByText(
        "このファイルには手で直した箇所があります。AIで整えると、修正後の文章も書き換わります。",
      ),
    ).not.toBeInTheDocument();
  });

  it("refetches the picture on its update event", async () => {
    const view = picture("A");
    await screen.findByText("machine text");
    const before = requests.filter(
      (r) => r.method === "GET" && r.url.endsWith("/picture-text"),
    ).length;
    files.A.picture = "text from the worker";
    ws.event = {
      event: "intelligence.picture_text.updated",
      data: { file_id: "A", reason: "rebuilt" },
    };
    act(() => {
      view.rerender(<PictureTextSection fileId="A" drive="d" />);
    });

    expect(await screen.findByText("text from the worker")).toBeInTheDocument();
    const after = requests.filter(
      (r) => r.method === "GET" && r.url.endsWith("/picture-text"),
    ).length;
    expect(after).toBeGreaterThan(before);
  });

  it("does not cancel a picture composition on Escape", async () => {
    picture("A");
    await screen.findByText("machine text");
    fireEvent.click(screen.getByRole("button", { name: "修正" }));
    const box = screen.getByRole("textbox");
    fireEvent.compositionEnd(box);
    fireEvent.keyDown(box, { key: "Escape" });
    expect(screen.getByRole("textbox")).toBeInTheDocument();
  });
});

describe("SPEC-ADDON-070 — design requirements", () => {
  it("wraps the header and keeps the two actions together", async () => {
    files.A.has_original = true;
    transcript("A");
    await screen.findByText("text-A-0");
    const header = screen.getByText("文字起こし").closest("div")!;
    expect(header.className).toContain("flex-wrap");
    const group = screen.getByRole("button", { name: "元の結果に戻す" }).parentElement!;
    expect(group.className).toContain("flex-nowrap");
  });

  it("uses the design badge, input radius and touch sizes", async () => {
    files.A.corrected = true;
    files.A.has_original = true;
    files.A.chunks[0].correctedAt = "2026-11-06T00:00:00Z";
    transcript("A");
    await screen.findByText("text-A-0");
    expect(screen.getByText("修正済み").className).toContain("bg-warm-light");
    const edit = screen.getAllByRole("button", { name: "修正" })[0];
    expect(edit.className).toContain("pointer-coarse:min-h-11");
    fireEvent.click(edit);
    expect(screen.getByRole("textbox").className).toContain("rounded-2xl");
  });

  it("puts the picture controls under the text", async () => {
    picture("A");
    await screen.findByText("machine text");
    const section = screen.getByText("画像の文字").closest("section")!;
    const text = screen.getByText("machine text");
    const edit = screen.getByRole("button", { name: "修正" });
    expect(
      text.compareDocumentPosition(edit) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    expect(section.contains(edit)).toBe(true);
  });
});
