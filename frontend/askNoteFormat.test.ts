import { describe, expect, it } from "vitest";
import type { Citation } from "./api";
import type { FileCitation } from "./askStream";
import {
  buildAskNoteMarkdown,
  buildAskThreadMarkdown,
  citationToLoftUrl,
  formatCitationListItem,
  parseSegmentLocation,
  queryToFilename,
} from "./askNoteFormat";

function makeCitation(overrides: Partial<Citation> = {}): Citation {
  return {
    file_id: "abc123def456",
    drive: "test-drive",
    filename: "lecture.mp4",
    file_type: "video",
    quote: "",
    relevance: 0.9,
    segment_location: null,
    ...overrides,
  };
}

describe("parseSegmentLocation", () => {
  it("parses m:ss timestamp", () => {
    const r = parseSegmentLocation("12:34");
    expect(r).toEqual({ label: "12:34", seconds: 754, page: null, section: null, verbatim: null });
  });

  it("parses page N", () => {
    const r = parseSegmentLocation("page 5");
    expect(r).toEqual({ label: "page 5", seconds: null, page: 5, section: null, verbatim: null });
  });

  it('parses "section 3"', () => {
    expect(parseSegmentLocation("section 3")).toEqual({
      label: "section 3", seconds: null, page: null, section: 3, verbatim: null,
    });
    expect(parseSegmentLocation("Section 12")?.section).toBe(12);
    expect(parseSegmentLocation("section 0")?.section).toBeNull();
  });

  it("ignores chunk N sentinel", () => {
    const r = parseSegmentLocation("chunk 7");
    expect(r?.seconds).toBeNull();
    expect(r?.verbatim).toBeNull();
  });

  it("returns null on empty / null input", () => {
    expect(parseSegmentLocation(null)).toBeNull();
    expect(parseSegmentLocation("")).toBeNull();
  });
});

describe("citationToLoftUrl", () => {
  it("appends ?t= for timestamp segment_location", () => {
    const c = makeCitation({ segment_location: "1:23" });
    expect(citationToLoftUrl(c)).toBe("loft://abc123def456?t=83");
  });

  it("appends ?page= for page segment_location", () => {
    const c = makeCitation({ segment_location: "page 4" });
    expect(citationToLoftUrl(c)).toBe("loft://abc123def456?page=4");
  });

  it("returns base loft URL when no segment_location", () => {
    expect(citationToLoftUrl(makeCitation())).toBe("loft://abc123def456");
  });
});

describe("formatCitationListItem — wiki-link for .md citations (Phase E)", () => {
  it("emits [[basename]] for markdown citations", () => {
    const c = makeCitation({
      filename: "2026-振り返り.md",
      file_type: "document",
    });
    expect(formatCitationListItem(c)).toBe("- [[2026-振り返り]]");
  });

  it("emits [[basename]] when file_type is exactly 'markdown' regardless of filename", () => {
    const c = makeCitation({
      filename: "note-without-ext",
      file_type: "markdown",
    });
    expect(formatCitationListItem(c)).toBe("- [[note-without-ext]]");
  });

  it("strips .md case-insensitively from basename", () => {
    const c = makeCitation({
      filename: "Notes.MD",
      file_type: "document",
    });
    expect(formatCitationListItem(c)).toBe("- [[Notes]]");
  });

  it("appends ' — locLabel' for .md citations when segment_location is present", () => {
    const c = makeCitation({
      filename: "transcript.md",
      file_type: "document",
      segment_location: "page 3",
    });
    expect(formatCitationListItem(c)).toBe("- [[transcript]] — page 3");
  });

  it("drops query parameters in wiki-link form (anchors not applicable to .md)", () => {
    const c = makeCitation({
      filename: "design.md",
      file_type: "document",
      segment_location: "1:30",
    });
    // Wiki-link form should NOT include ?t= — that belongs to loft:// only.
    expect(formatCitationListItem(c)).not.toContain("?t=");
    expect(formatCitationListItem(c)).not.toContain("loft://");
  });
});

describe("formatCitationListItem — loft:// for non-.md citations (existing)", () => {
  it("emits [filename](loft://id) for video citations", () => {
    const c = makeCitation({
      filename: "lecture.mp4",
      file_type: "video",
    });
    expect(formatCitationListItem(c)).toBe(
      "- [lecture.mp4](loft://abc123def456)",
    );
  });

  it("appends ?t= for timestamped video citations", () => {
    const c = makeCitation({
      filename: "lecture.mp4",
      file_type: "video",
      segment_location: "2:05",
    });
    expect(formatCitationListItem(c)).toBe(
      "- [lecture.mp4](loft://abc123def456?t=125) — 2:05",
    );
  });

  it("appends ?page= for PDF citations", () => {
    const c = makeCitation({
      filename: "paper.pdf",
      file_type: "document",
      segment_location: "page 7",
    });
    expect(formatCitationListItem(c)).toBe(
      "- [paper.pdf](loft://abc123def456?page=7) — page 7",
    );
  });
});

describe("buildAskNoteMarkdown", () => {
  it("renders frontmatter + body + 引用元 section", () => {
    const md = buildAskNoteMarkdown(
      "What is Litloft?",
      "Litloft is …",
      [makeCitation({ filename: "intro.mp4", file_type: "video" })],
    );
    expect(md).toContain("origin: ask_answer");
    expect(md).toContain("# What is Litloft?");
    expect(md).toContain("## 引用元");
    expect(md).toContain("- [intro.mp4](loft://abc123def456)");
  });

  it("mixes wiki-link and loft:// citations in the same note", () => {
    const md = buildAskNoteMarkdown(
      "Cross reference",
      "Answer body.",
      [
        makeCitation({
          file_id: "vid000000000",
          filename: "talk.mp4",
          file_type: "video",
          segment_location: "0:30",
        }),
        makeCitation({
          file_id: "doc000000000",
          filename: "summary.md",
          file_type: "document",
        }),
      ],
    );
    expect(md).toContain("- [talk.mp4](loft://vid000000000?t=30) — 0:30");
    expect(md).toContain("- [[summary]]");
  });

  it("preserves quote blockquote indentation", () => {
    const md = buildAskNoteMarkdown(
      "Q",
      "A",
      [
        makeCitation({
          filename: "note.md",
          file_type: "document",
          quote: "first line\nsecond line",
        }),
      ],
    );
    expect(md).toContain("  > first line");
    expect(md).toContain("  > second line");
  });

  it("emits 'source_file_ids' frontmatter with deduplicated ids", () => {
    const md = buildAskNoteMarkdown(
      "Q",
      "A",
      [
        makeCitation({ file_id: "x000000000xx" }),
        makeCitation({ file_id: "x000000000xx" }),
        makeCitation({ file_id: "y000000000yy" }),
      ],
    );
    expect(md).toContain('source_file_ids: ["x000000000xx", "y000000000yy"]');
  });

  it("omits ## 引用元 when no citations", () => {
    const md = buildAskNoteMarkdown("Q", "A", []);
    expect(md).not.toContain("## 引用元");
  });
});

describe("buildAskThreadMarkdown — code is what AnswerMarkdown renders as code", () => {
  const THREAD_FILE = "vid000000001";
  const CITED: FileCitation[] = [{ n: 1, position: { seconds: 5 } }];
  const link = `[0:05](loft://${THREAD_FILE}?t=5)`;

  function threadNote(answer: string): string {
    return [
      "---",
      "origin: ask_answer",
      'query: "why"',
      `source_file_ids: ["${THREAD_FILE}"]`,
      "saved_at: 2026-01-01T00:00:00.000Z",
      "---",
      "",
      "# why",
      "",
      `[lecture.mp4](loft://${THREAD_FILE})`,
      "",
      "## why",
      "",
      answer,
      "",
    ].join("\n");
  }

  it.each([
    {
      name: "a fence inside a list item",
      answer: "1. Run:\n    ```python\n    y = arr[1]\n    ```",
      expected: "1. Run:\n    ```python\n    y = arr[1]\n    ```",
    },
    {
      name: "a list continuation paragraph",
      answer: "- item\n\n    see [1]",
      expected: `- item\n\n    see ${link}`,
    },
    {
      name: "a top-level indented code block",
      answer: "text\n\n    code [1]",
      expected: "text\n\n    code [1]",
    },
    {
      name: "a fence inside a blockquote",
      answer: "> ```\n> a[1]\n> ```",
      expected: "> ```\n> a[1]\n> ```",
    },
    {
      name: "a fence two list levels deep",
      answer: "1. outer\n    1. inner\n\n        ```\n        b[1]\n        ```",
      expected: "1. outer\n    1. inner\n\n        ```\n        b[1]\n        ```",
    },
    {
      name: "a span crossing a soft line break",
      answer: "See `x\ny [1]` z",
      expected: "See `x\ny [1]` z",
    },
    {
      name: "a span whose closing run is two lines after its opening run",
      answer: "a `x\nb [1]\nc` d",
      expected: "a `x\nb [1]\nc` d",
    },
    {
      name: "a span crossing a line break inside a blockquote",
      answer: "> See `x\n> y [1]` z",
      expected: "> See `x\n> y [1]` z",
    },
    {
      name: "a span crossing a line break in a list item paragraph",
      answer: "- See `x\n  y [1]` z",
      expected: "- See `x\n  y [1]` z",
    },
    {
      name: "a backtick never closed before a paragraph break",
      answer: "open `x [1]\n\nafter [1]",
      expected: `open \`x ${link}\n\nafter ${link}`,
    },
  ])("writes $name as received exactly where it is code", ({ answer, expected }) => {
    const note = buildAskThreadMarkdown({
      fileId: THREAD_FILE,
      filename: "lecture.mp4",
      fileType: "video",
      turns: [{ question: "why", answer, citations: CITED }],
      savedAt: "2026-01-01T00:00:00.000Z",
    });

    expect(note).toBe(threadNote(expected));
  });
});

describe("queryToFilename", () => {
  it("slugifies a query", () => {
    expect(queryToFilename("Hello World!")).toBe("hello-world.md");
  });

  it("falls back to 'ask-note' for an empty query", () => {
    expect(queryToFilename("")).toBe("ask-note.md");
  });

  it("preserves CJK characters", () => {
    expect(queryToFilename("2026 年振り返り")).toBe("2026-年振り返り.md");
  });

  it("caps slug at 60 chars", () => {
    const long = "a".repeat(100);
    const r = queryToFilename(long);
    // 60 chars of "a" + ".md" suffix
    expect(r.length).toBe(63);
  });
});

describe("formatCitationListItem — EPUB sections", () => {
  it("labels a section citation with its title", () => {
    const book = makeCitation({
      filename: "novel.epub",
      file_type: "document",
      segment_location: "section 3",
    });

    expect(formatCitationListItem({ ...book, section_title: "第三章 旅立ち" })).toBe(
      "- [novel.epub](loft://abc123def456?section=3) — 第三章 旅立ち",
    );
    expect(formatCitationListItem({ ...book, section_title: null })).toBe(
      "- [novel.epub](loft://abc123def456?section=3) — section 3",
    );
  });
});
