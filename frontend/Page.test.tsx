/**
 * Tests for the progressive-citations + improved loading UX of the
 * intelligence Ask page.
 *
 * Covers:
 *   1. `parseSseFrame` recognises the new `event: citation` frame and
 *      yields `{ kind: "citation"; citation; index }`.
 *   2. The terminal `event: citations` (plural) is still parsed.
 *   3. The Ask page renders citations progressively — each per-citation
 *      event appends a CitationCard immediately.
 *   4. Inline `[N]` chips render as interactive buttons once citation N
 *      has arrived, and as muted text while it is still pending.
 *   5. While the stream is live and no answer has begun, a `Thinking`
 *      indicator (stable testid `ask-thinking`) is visible; it goes
 *      away once the first answer_chunk lands.
 *   6. When the terminal `citations` frame arrives, it replaces the
 *      progressively-accumulated list.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, act, waitFor } from "@testing-library/react";
import React from "react";

// --- Mock next/navigation (Ask page reads useSearchParams + usePathname). ----
vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams(""),
  usePathname: () => "/drive/family/addons/intelligence",
}));

// --- Mock CurrentDriveProvider (Page reads useCurrentDrive). ----
vi.mock("@/components/CurrentDriveProvider", () => ({
  useCurrentDrive: () => "family",
}));

// --- Shared stream controller the mocked `askQuestionStream` binds to.
//     Tests call `streamState.reset()` to get a clean queue each run;
//     the `askQuestionStream` mock delegates to `streamState.generator()`
//     which always reads from whichever queue is active. ---
type QueueEntry =
  | { done: false; value: unknown }
  | { done: true };

type StreamController = {
  push: (value: unknown) => void;
  end: () => void;
  generator: () => AsyncGenerator<unknown>;
};

function makeController(): StreamController {
  const queue: QueueEntry[] = [];
  const waiters: Array<(entry: QueueEntry) => void> = [];

  const push = (value: unknown) => {
    const w = waiters.shift();
    if (w) w({ done: false, value });
    else queue.push({ done: false, value });
  };
  const end = () => {
    const w = waiters.shift();
    if (w) w({ done: true });
    else queue.push({ done: true });
  };
  async function* generator() {
    while (true) {
      const entry = await new Promise<QueueEntry>((resolve) => {
        const head = queue.shift();
        if (head) resolve(head);
        else waiters.push(resolve);
      });
      if (entry.done) return;
      yield entry.value;
    }
  }
  return { push, end, generator };
}

const streamState: { current: StreamController } = {
  current: makeController(),
};

vi.mock("@/addons/intelligence/api", async () => {
  const actual = await vi.importActual<
    typeof import("@/addons/intelligence/api")
  >("@/addons/intelligence/api");
  return {
    ...actual,
    askQuestionStream: vi.fn(() => streamState.current.generator()),
    getIntelligenceStatus: vi.fn().mockResolvedValue({
      status: "ok",
      features: {
        indexing: true,
        search: true,
        auto_tags: "false",
        summaries: "false",
        rag: true,
      },
      llm: {
        provider: "ollama",
        model: "qwen",
        enabled: true,
        output_language: "auto",
      },
    }),
  };
});

// Import _after_ mocks are registered.
import IntelligenceAskPage from "@/addons/intelligence/Page";
import {
  parseSseFrame,
  type AskStreamEvent,
  type Citation,
} from "@/addons/intelligence/api";

const sampleCitation = (n: number): Citation => ({
  file_id: `file-${n}`,
  drive: "family",
  filename: `doc-${n}.md`,
  file_type: "document",
  quote: `Quote for citation ${n}`,
  relevance: 0.9,
  segment_location: null,
});

describe("parseSseFrame — single-citation event", () => {
  it("parses a `citation` frame into { kind: 'citation', citation, index }", () => {
    expect(typeof parseSseFrame).toBe("function");
    const frame = [
      "event: citation",
      `data: ${JSON.stringify({ citation: sampleCitation(1), index: 1 })}`,
    ].join("\n");
    const parsed = parseSseFrame(frame);
    expect(parsed).not.toBeNull();
    expect(parsed!.kind).toBe("citation");
    if (parsed!.kind === "citation") {
      expect(parsed!.index).toBe(1);
      expect(parsed!.citation.file_id).toBe("file-1");
      expect(parsed!.citation.quote).toBe("Quote for citation 1");
    }
  });

  it("still parses the terminal `citations` (plural) frame", () => {
    const frame = [
      "event: citations",
      `data: ${JSON.stringify({
        citations: [sampleCitation(1), sampleCitation(2)],
      })}`,
    ].join("\n");
    const parsed = parseSseFrame(frame) as AskStreamEvent | null;
    expect(parsed).not.toBeNull();
    expect(parsed!.kind).toBe("citations");
    if (parsed!.kind === "citations") {
      expect(parsed!.citations).toHaveLength(2);
    }
  });

  it("returns null for malformed citation frames (missing index)", () => {
    const frame = [
      "event: citation",
      `data: ${JSON.stringify({ citation: sampleCitation(1) })}`,
    ].join("\n");
    expect(parseSseFrame(frame)).toBeNull();
  });
});

describe("IntelligenceAskPage — progressive citations + thinking indicator", () => {
  beforeEach(() => {
    streamState.current = makeController();
  });

  afterEach(() => {
    // Drain the controller so any dangling generator exits and nothing
    // leaks between tests.
    streamState.current.end();
  });

  async function mountAndStart(question = "what is the plot?") {
    const utils = render(<IntelligenceAskPage />);
    const textarea = (await screen.findByRole("textbox", {
      name: /question input/i,
    })) as HTMLTextAreaElement;
    // Feed the controlled input via the native value setter so React
    // sees a real change event (mirrors what userEvent.type does, but
    // without the keystroke overhead).
    const setter = Object.getOwnPropertyDescriptor(
      HTMLTextAreaElement.prototype,
      "value",
    )!.set!;
    const form = textarea.closest("form")!;
    await act(async () => {
      setter.call(textarea, question);
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
      form.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
    });
    return utils;
  }

  it("appends each CitationCard as per-citation events arrive", async () => {
    await mountAndStart();
    await act(async () => {
      streamState.current.push({ kind: "keywords", keywords: "plot story" });
      streamState.current.push({
        kind: "sources",
        sources: [
          {
            file_id: "file-src",
            drive: "family",
            filename: "source-doc.md",
            file_type: "document",
            score: 1.0,
            match_types: [],
          },
        ],
      });
      streamState.current.push({
        kind: "answer_chunk",
        delta: "The protagonist [1]",
      });
    });
    await act(async () => {
      streamState.current.push({
        kind: "citation",
        citation: sampleCitation(1),
        index: 1,
      });
    });
    await waitFor(() => {
      expect(screen.getByText("doc-1.md")).toBeInTheDocument();
    });
    await act(async () => {
      streamState.current.push({
        kind: "answer_chunk",
        delta: " meets a companion [2].",
      });
      streamState.current.push({
        kind: "citation",
        citation: sampleCitation(2),
        index: 2,
      });
    });
    await waitFor(() => {
      expect(screen.getByText("doc-2.md")).toBeInTheDocument();
    });
    await act(async () => {
      streamState.current.push({ kind: "done", took_ms: 1234 });
      streamState.current.end();
    });
  });

  it("renders pending [N] chips as muted until their citation arrives", async () => {
    await mountAndStart();
    await act(async () => {
      streamState.current.push({ kind: "keywords", keywords: "x" });
      streamState.current.push({ kind: "sources", sources: [] });
      streamState.current.push({
        kind: "answer_chunk",
        delta: "See [1] soon.",
      });
    });
    await waitFor(() => {
      expect(screen.getByText(/See/)).toBeInTheDocument();
    });
    // Citation 1 hasn't arrived; the chip must NOT be a button.
    expect(
      screen.queryByRole("button", { name: /Jump to citation 1/ }),
    ).toBeNull();

    await act(async () => {
      streamState.current.push({
        kind: "citation",
        citation: sampleCitation(1),
        index: 1,
      });
    });
    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: /Jump to citation 1/ }),
      ).toBeInTheDocument();
    });

    await act(async () => {
      streamState.current.push({ kind: "done" });
      streamState.current.end();
    });
  });

  it("shows a 'thinking' indicator while the answer buffer is empty", async () => {
    await mountAndStart();
    await act(async () => {
      streamState.current.push({ kind: "keywords", keywords: "x" });
      streamState.current.push({ kind: "sources", sources: [] });
    });
    await waitFor(() => {
      expect(screen.getByTestId("ask-thinking")).toBeInTheDocument();
    });

    await act(async () => {
      streamState.current.push({ kind: "answer_chunk", delta: "Hello" });
    });
    await waitFor(() => {
      expect(screen.queryByTestId("ask-thinking")).toBeNull();
    });
    await act(async () => {
      streamState.current.push({ kind: "done" });
      streamState.current.end();
    });
  });

  it("replaces progressive citations with the terminal `citations` list", async () => {
    await mountAndStart();
    await act(async () => {
      streamState.current.push({ kind: "keywords", keywords: "x" });
      streamState.current.push({ kind: "sources", sources: [] });
      streamState.current.push({
        kind: "answer_chunk",
        delta: "A [1] B [2]",
      });
      streamState.current.push({
        kind: "citation",
        citation: sampleCitation(1),
        index: 1,
      });
      streamState.current.push({
        kind: "citation",
        citation: sampleCitation(2),
        index: 2,
      });
    });
    await waitFor(() => {
      expect(screen.getByText("doc-1.md")).toBeInTheDocument();
      expect(screen.getByText("doc-2.md")).toBeInTheDocument();
    });
    const replacement: Citation = {
      ...sampleCitation(2),
      filename: "doc-FINAL.md",
    };
    await act(async () => {
      streamState.current.push({
        kind: "citations",
        citations: [sampleCitation(1), replacement],
      });
      streamState.current.push({ kind: "done", took_ms: 42 });
      streamState.current.end();
    });
    await waitFor(() => {
      expect(screen.getByText("doc-FINAL.md")).toBeInTheDocument();
      expect(screen.queryByText("doc-2.md")).toBeNull();
    });
  });
});
