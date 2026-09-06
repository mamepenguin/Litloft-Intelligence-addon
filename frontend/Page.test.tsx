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
 *   4. The answer body renders as Markdown (sanitized). The prompt
 *      bans `[1][2]` markers so attribution lives in the citations
 *      list, not as inline chips.
 *   5. While the stream is live and no answer has begun, a `Thinking`
 *      indicator (stable testid `ask-thinking`) is visible; it goes
 *      away once the first answer_chunk lands.
 *   6. When the terminal `citations` frame arrives, it replaces the
 *      progressively-accumulated list.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, act, waitFor, fireEvent } from "@testing-library/react";
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
import { accentFills } from "@/__tests__/helpers/accentFills";
import {
  getIntelligenceStatus,
  parseSseFrame,
  type AskStreamEvent,
  type Citation,
} from "@/addons/intelligence/api";
import {
  clearSourceCaptures,
  getSourceCaptures,
} from "@/lib/sourceCapture";

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
    clearSourceCaptures("family");
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

  it("adds an Ask citation to the capture basket", async () => {
    await mountAndStart();
    await act(async () => {
      streamState.current.push({ kind: "answer_chunk", delta: "Answer" });
      streamState.current.push({
        kind: "citation",
        citation: {
          ...sampleCitation(1),
          file_type: "video",
          segment_location: "1:05",
        },
        index: 1,
      });
    });

    fireEvent.click(
      await screen.findByRole("button", {
        name: "Add citation to capture basket",
      }),
    );

    expect(getSourceCaptures("family")).toEqual([
      expect.objectContaining({
        sourceFileId: "file-1",
        kind: "ask_citation",
        quote: "Quote for citation 1",
        locator: expect.objectContaining({ seconds: 65, label: "1:05" }),
      }),
    ]);
    await act(async () => {
      streamState.current.push({ kind: "done", took_ms: 10 });
      streamState.current.end();
    });
  });

  it("renders the answer body as Markdown (bold, headings, lists)", async () => {
    const utils = await mountAndStart();
    await act(async () => {
      streamState.current.push({ kind: "keywords", keywords: "x" });
      streamState.current.push({ kind: "sources", sources: [] });
      streamState.current.push({
        kind: "answer_chunk",
        delta: "## Heading\n\n**bold text** and a list:\n\n- one\n- two",
      });
    });
    await waitFor(() => {
      expect(utils.container.querySelector("h2")).toHaveTextContent("Heading");
    });
    expect(utils.container.querySelector("strong")).toHaveTextContent(
      "bold text",
    );
    expect(utils.container.querySelectorAll("li")).toHaveLength(2);
    // No `[N]` chips are created — attribution lives in the citations list.
    expect(
      screen.queryByRole("button", { name: /Jump to citation/ }),
    ).toBeNull();

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

  it("renders a thumbnail for image citations (tier 3 exception prerequisite)", async () => {
    // The vision_describe spec (`Wewd0UyArEW49kE3UCUY6`) makes RAG trust
    // image citations on the basis that the user can verify them
    // visually. That trust is only valid when the UI actually shows the
    // thumbnail — this regression test pins the contract.
    await mountAndStart();
    await act(async () => {
      streamState.current.push({ kind: "keywords", keywords: "x" });
      streamState.current.push({ kind: "sources", sources: [] });
      streamState.current.push({
        kind: "answer_chunk",
        delta: "See the photo [1].",
      });
      const imageCitation: Citation = {
        file_id: "img-1",
        drive: "family",
        filename: "sunset.jpg",
        file_type: "image",
        quote: "A sunset over the ocean",
        relevance: 0.9,
        segment_location: null,
      };
      streamState.current.push({
        kind: "citation",
        citation: imageCitation,
        index: 1,
      });
    });
    const thumb = await screen.findByTestId("ask-citation-thumbnail-1");
    expect(thumb).toBeInTheDocument();
    expect(thumb.getAttribute("src")).toBe("/api/files/img-1/thumbnail");
    await act(async () => {
      streamState.current.push({ kind: "done" });
      streamState.current.end();
    });
  });

  it("does not render a thumbnail for non-image citations", async () => {
    await mountAndStart();
    await act(async () => {
      streamState.current.push({ kind: "keywords", keywords: "x" });
      streamState.current.push({ kind: "sources", sources: [] });
      streamState.current.push({
        kind: "answer_chunk",
        delta: "See page 4 [1].",
      });
      streamState.current.push({
        kind: "citation",
        citation: sampleCitation(1),
        index: 1,
      });
    });
    await waitFor(() => {
      expect(screen.getByText("doc-1.md")).toBeInTheDocument();
    });
    expect(screen.queryByTestId("ask-citation-thumbnail-1")).toBeNull();
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

  it("builds citation URLs with ?t= for time, ?page= for paginated, ?highlight= for text", async () => {
    await mountAndStart();
    await act(async () => {
      streamState.current.push({ kind: "keywords", keywords: "x" });
      streamState.current.push({ kind: "sources", sources: [] });
      streamState.current.push({
        kind: "answer_chunk",
        delta: "Mixed citations [1][2][3].",
      });
      streamState.current.push({
        kind: "citation",
        citation: {
          ...sampleCitation(1),
          file_id: "video-1",
          filename: "lecture.mp4",
          file_type: "video",
          segment_location: "12:34",
        },
        index: 1,
      });
      streamState.current.push({
        kind: "citation",
        citation: {
          ...sampleCitation(2),
          file_id: "pdf-1",
          filename: "paper.pdf",
          file_type: "document",
          segment_location: "page 7",
        },
        index: 2,
      });
      streamState.current.push({
        kind: "citation",
        citation: {
          ...sampleCitation(3),
          file_id: "md-1",
          filename: "notes.md",
          file_type: "document",
          quote: "the cited passage about dragons",
          segment_location: "chunk 5",
        },
        index: 3,
      });
    });

    const videoLink = (await screen.findByText("lecture.mp4")).closest("a");
    expect(videoLink?.getAttribute("href")).toBe("/files/video-1?t=754");

    const pdfLink = (await screen.findByText("paper.pdf")).closest("a");
    expect(pdfLink?.getAttribute("href")).toBe("/files/pdf-1?page=7");

    const mdLink = (await screen.findByText("notes.md")).closest("a");
    expect(mdLink?.getAttribute("href")).toBe(
      `/files/md-1?highlight=${encodeURIComponent("the cited passage about dragons")}`,
    );

    await act(async () => {
      streamState.current.push({ kind: "done" });
      streamState.current.end();
    });
  });

  it("prefers verbatim segment_location over citation.quote for highlight", async () => {
    // Local LLMs (Ollama / Qwen / Gemma) commonly ignore the
    // `location: '0:45' | 'page 3'` instruction and put a verbatim
    // sentence from the cited passage there instead. That sentence
    // matches the source file character-for-character, so it makes
    // a far better highlight target than `citation.quote` — which
    // the backend defaults to the file long_summary when no
    // chunk-level snippet matched.
    await mountAndStart();
    await act(async () => {
      streamState.current.push({ kind: "keywords", keywords: "x" });
      streamState.current.push({ kind: "sources", sources: [] });
      streamState.current.push({
        kind: "answer_chunk",
        delta: "Verbatim location citation [1].",
      });
      streamState.current.push({
        kind: "citation",
        citation: {
          ...sampleCitation(1),
          file_id: "md-2",
          filename: "story.md",
          file_type: "document",
          // citation.quote here is the file's auto-summary (what the
          // backend picks when chunk lookup fails).
          quote:
            "物語は退職を控えた郵便配達員が古い手紙を見つけるところから始まる。",
          // The LLM put a verbatim source sentence in `location`.
          segment_location:
            "退職の日、徹は妻の節子と一緒に、藤原健一の墓を訪れた。",
        },
        index: 1,
      });
    });

    const link = (await screen.findByText("story.md")).closest("a");
    expect(link?.getAttribute("href")).toBe(
      `/files/md-2?highlight=${encodeURIComponent("退職の日、徹は妻の節子と一緒に、藤原健一の墓を訪れた。")}`,
    );

    await act(async () => {
      streamState.current.push({ kind: "done" });
      streamState.current.end();
    });
  });
});

describe("IntelligenceAskPage — back-navigation cache (sessionStorage)", () => {
  beforeEach(() => {
    streamState.current = makeController();
    if (typeof window !== "undefined") window.sessionStorage.clear();
  });

  afterEach(() => {
    streamState.current.end();
    if (typeof window !== "undefined") window.sessionStorage.clear();
  });

  it("restores an answered snapshot from sessionStorage when ?q= matches", async () => {
    // Pre-seed the cache as if a previous run had completed.
    const cached = {
      keywords: "plot story",
      clues: null,
      personalHistory: null,
      sources: [],
      answer: "Cached answer body.",
      citations: [
        {
          ...sampleCitation(1),
          filename: "from-cache.md",
        },
      ],
      tookMs: 999,
    };
    window.sessionStorage.setItem(
      "intelligence-ask-cache:v1:family:what is the plot?",
      JSON.stringify(cached),
    );

    // Mock useSearchParams to surface the seed query.
    const navMock = await import("next/navigation");
    const original = navMock.useSearchParams;
    (navMock as unknown as { useSearchParams: () => URLSearchParams }).useSearchParams =
      () => new URLSearchParams("q=what is the plot?");

    try {
      render(<IntelligenceAskPage />);
      await waitFor(() => {
        expect(screen.getByText("Cached answer body.")).toBeInTheDocument();
        expect(screen.getByText("from-cache.md")).toBeInTheDocument();
      });
    } finally {
      (navMock as unknown as { useSearchParams: typeof original }).useSearchParams =
        original;
    }
  });

  it("writes ?q= to the URL on submit so a back-navigation lands with a non-empty seed query", async () => {
    // Reset the URL so the test does not inherit ?q= from a prior run.
    const original = window.location.href;
    window.history.replaceState(
      null,
      "",
      "/drive/family/addons/intelligence",
    );

    try {
      const utils = render(<IntelligenceAskPage />);
      const textarea = (await screen.findByRole("textbox", {
        name: /question input/i,
      })) as HTMLTextAreaElement;
      const setter = Object.getOwnPropertyDescriptor(
        HTMLTextAreaElement.prototype,
        "value",
      )!.set!;
      const form = textarea.closest("form")!;
      await act(async () => {
        setter.call(textarea, "what is the plot?");
        textarea.dispatchEvent(new Event("input", { bubbles: true }));
        form.dispatchEvent(
          new Event("submit", { bubbles: true, cancelable: true }),
        );
      });

      // The submit handler updates the URL synchronously via
      // replaceState before the SSE stream starts. By the time
      // streaming events flush (next microtask) the search param
      // must already be visible.
      await waitFor(() => {
        const params = new URLSearchParams(window.location.search);
        expect(params.get("q")).toBe("what is the plot?");
      });

      // Drain a minimal stream so the answered transition runs and
      // writes the cache. A subsequent mount with the same ?q= would
      // then hit the cache (covered by the prior test).
      await act(async () => {
        streamState.current.push({ kind: "keywords", keywords: "x" });
        streamState.current.push({ kind: "sources", sources: [] });
        streamState.current.push({ kind: "answer_chunk", delta: "OK." });
        streamState.current.push({ kind: "done" });
        streamState.current.end();
      });

      await waitFor(() => {
        const raw = window.sessionStorage.getItem(
          "intelligence-ask-cache:v1:family:what is the plot?",
        );
        expect(raw).not.toBeNull();
      });

      utils.unmount();
    } finally {
      window.history.replaceState(null, "", original);
    }
  });
});

describe("IntelligenceAskPage — status probe fallback", () => {
  // The addon's /status route is admin-gated (it surfaces process-
  // global queue counters). Non-admin viewers — anyone who has not
  // unlocked every protected drive — get null back from
  // ``getIntelligenceStatus``. Treating that as "LLM not configured"
  // hides Ask from the very viewers it is meant to serve. Instead the
  // form must render enabled and let the actual /ask call surface any
  // real backend error.
  it("renders the form enabled and hides the llmDisabled alert when /status is null", async () => {
    vi.mocked(getIntelligenceStatus).mockResolvedValueOnce(null);

    render(<IntelligenceAskPage />);

    const textarea = (await screen.findByRole("textbox", {
      name: /question input/i,
    })) as HTMLTextAreaElement;

    await waitFor(() => {
      expect(textarea.disabled).toBe(false);
    });
    expect(screen.queryByText("LLM is not configured")).toBeNull();
  });

  it("disables the form and shows the llmDisabled alert when llm.enabled is false", async () => {
    vi.mocked(getIntelligenceStatus).mockResolvedValueOnce({
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
        enabled: false,
        output_language: "auto",
      },
    });

    render(<IntelligenceAskPage />);

    const textarea = (await screen.findByRole("textbox", {
      name: /question input/i,
    })) as HTMLTextAreaElement;

    await waitFor(() => {
      expect(textarea.disabled).toBe(true);
    });
    expect(
      await screen.findByText("LLM is not configured"),
    ).toBeInTheDocument();
  });
});

/**
 * The page's own chrome, after adopting core's `PageHeader` and `PageTabs`
 * (UI redesign Phase 3, C2a).
 *
 * Three things this page used to say for itself and now takes from core: the
 * heading and its size, the mode row's vocabulary, and how many accent fills
 * the screen is allowed.
 */
describe("IntelligenceAskPage — page header, mode tabs and accent budget", () => {
  beforeEach(() => {
    streamState.current = makeController();
    clearSourceCaptures("family");
  });

  afterEach(() => {
    streamState.current.end();
  });

  async function answered() {
    const utils = render(<IntelligenceAskPage />);
    const textarea = (await screen.findByRole("textbox", {
      name: /question input/i,
    })) as HTMLTextAreaElement;
    const setter = Object.getOwnPropertyDescriptor(
      HTMLTextAreaElement.prototype,
      "value",
    )!.set!;
    await act(async () => {
      setter.call(textarea, "what is the plot?");
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
    });
    // The button, not a synthetic `submit` on the form. Dispatching on the
    // form reaches the answered state even when the control that starts a
    // question is disabled — so a test named for asking would pass on a
    // screen where asking is impossible.
    await act(async () => {
      fireEvent.click(screen.getByTestId("ask-submit"));
    });
    await act(async () => {
      streamState.current.push({ kind: "keywords", keywords: "plot" });
      streamState.current.push({
        kind: "citation",
        index: 0,
        citation: {
          file_id: "file-a",
          drive: "family",
          filename: "a.md",
          folder_path: "",
          excerpt: "x",
        },
      });
      streamState.current.push({ kind: "answer_chunk", delta: "an answer" });
      streamState.current.push({ kind: "done" });
      streamState.current.end();
    });
    return utils;
  }

  /**
   * A-2: the page told the reader it was an "AI answer" before it had
   * given one, offered its own `?q=` back as a placeholder, and put "AI" on
   * the button where a verb belongs. Find, one tab away, does all three
   * right — the point of this group is that the two now read the same.
   */
  it("names what the page is for, not what it will produce", async () => {
    render(<IntelligenceAskPage />);
    const heading = await screen.findByRole("heading", { level: 1 });
    expect(heading.textContent).toContain("Ask this drive");
    // The answer's own heading keeps the old words, one level down, and
    // only once there is an answer under it.
    expect(screen.queryByText("AI answer")).toBeNull();
  });

  it("shows an example in the input, and a verb on the button", async () => {
    render(<IntelligenceAskPage />);
    const textarea = (await screen.findByRole("textbox", {
      name: /question input/i,
    })) as HTMLTextAreaElement;

    expect(textarea.placeholder).toMatch(/^e\.g\./);
    expect(textarea.value).toBe("");
    expect(screen.getByTestId("ask-submit").textContent).toContain("Ask");
  });

  it("says what it will send before it sends it", async () => {
    render(<IntelligenceAskPage />);
    // The privacy line the rules require of every feature that ships file
    // content to an LLM API — `.claude/rules/design-decisions.md`.
    expect(await screen.findByText(/LLM API/)).toBeInTheDocument();
  });

  it("names itself once, and lets core choose the size", async () => {
    const { container } = render(<IntelligenceAskPage />);
    await screen.findByRole("textbox", { name: /question input/i });
    const h1s = container.querySelectorAll("h1");
    expect(h1s).toHaveLength(1);
    // DESIGN.md §3.2 gives H1 one size, and `PageHeader` is the only thing
    // that writes it. Asserting it here is what would catch this page going
    // back to writing its own heading with its own `text-lg`.
    expect(h1s[0].className).toContain("text-2xl");
  });

  it("marks the current mode the way a set of links does, and not twice", async () => {
    render(<IntelligenceAskPage />);
    const ask = await screen.findByRole("link", { name: /ask/i });
    expect(ask).toHaveAttribute("aria-current", "page");
    // Ask and Find are separate routes, so the row navigates. `role="tab"`
    // promises a screen reader that activating it swaps a region in this
    // view, which a `<Link>` does not do — and this row used to carry both
    // that promise and `aria-current` at once. Media Import resolved the same
    // pairing from the other end: its two views are one page, so it kept the
    // tablist and dropped `aria-current`.
    expect(ask).not.toHaveAttribute("aria-selected");
    expect(screen.queryByRole("tablist")).toBeNull();
    expect(screen.queryAllByRole("tab")).toHaveLength(0);
    // The other mode is present and is not marked current — without this the
    // assertions above would pass on a row that had lost its second tab.
    const find = screen.getByRole("link", { name: /find/i });
    expect(find).not.toHaveAttribute("aria-current");
  });

  it("spends its one accent fill on asking, with the answer on screen", async () => {
    // Measured in the answered state, not at rest. At rest only the submit
    // button exists and a budget of one would hold no matter what the save
    // action wore; these two are on screen together here, and before this
    // migration they were the two fills.
    //
    // It is not the state with the most fills. Opening the save dialog adds
    // `AskSaveDialog`'s own primary button, which renders inside `container`
    // because that dialog does not portal — so one click away, this counts
    // two.
    //
    // **There is no precedent either way.** An earlier version of this comment
    // said dialogs are outside the budget by existing practice, citing the
    // four `() => null` dialog mocks in core's `accent-budget.test.tsx`. Those
    // mocks are real, and the inference from them is not: deleting all four
    // leaves that suite green at the same count, because two of the dialogs
    // return `null` unless opened and the other two carry no `bg-accent` at
    // all. The comment above them says what they are for — scaffolding to get
    // the drive root to draw — and names `Button` and `AddButton` as the only
    // things deliberately left real. Core has never measured a dialog-open
    // screen.
    //
    // So this measures the closed state as a choice, not as a convention: a
    // dialog is a surface of its own, and `AskSaveDialog`'s primary button is
    // C2b's to place.
    const { container } = await answered();
    // The save action only exists once an answer has citations, so waiting on
    // it is waiting on the state this test is about rather than on the render
    // that precedes it. Found by `data-testid` for the reason the one on the
    // thinking indicator gives a few hundred lines up — it survives both the
    // translation catalogue and the icon library.
    await screen.findByTestId("ask-save-note");
    expect(accentFills(container)).toHaveLength(1);
    // And it is the submit button that keeps it.
    expect(accentFills(container)[0]).toHaveAttribute("type", "submit");
  });

  it("does not repaint the submit button under the cursor while it is disabled", async () => {
    // `Button` guards its hover colour behind `enabled:`, and `Button.tsx`
    // says why: a bare `hover:` repaints a *disabled* button the moment the
    // pointer rests on it, which tells the reader it is live. Both of this
    // page's fills used to be written by hand with a bare `hover:`.
    //
    // **This does not pin that the button is core's `Button`.** A hand-written
    // recipe that carries the guard passes it, measured. What is asserted is
    // the guard, and the guard is a rule about which CSS exists — the one
    // thing a class string can prove. Appearance is not claimed: jsdom loads
    // no stylesheet, so nothing here observes a colour.
    render(<IntelligenceAskPage />);
    const submit = await screen.findByTestId("ask-submit");
    // Disabled because the input is empty — measured, not assumed: dropping
    // the length condition from `canSubmit` reddens this, and dropping the
    // `ragAvailable` one does not, so the probe has already resolved.
    expect(submit).toBeDisabled();
    // Tokens, and the absence of an unguarded one. `toContain` on the whole
    // class string was the first version and it is a substring match: a
    // `primary` variant carrying a bare `hover:bg-accent-hover` alongside an
    // unrelated `enabled:hover:underline` passed it, with the very defect
    // this test is named for live on the screen.
    const tokens = submit.className.split(/\s+/);
    expect(tokens).toContain("enabled:hover:bg-accent-hover");
    expect(tokens.filter((t) => /^hover:bg-accent/.test(t))).toEqual([]);
  });
});
