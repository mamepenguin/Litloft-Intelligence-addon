/**
 * Tests for the "accept suggested tag" dispatch integration.
 *
 * Phase 7 of the knowledge tag unification (spec
 * ``docs/superpowers/specs/2026-04-24-knowledge-tag-unification.md``
 * §D9): approving a tag on a ``.md`` must write frontmatter via
 * ``saveFileTags`` rather than PUT /files/{id}/tags directly. Without
 * the dispatch, the scanner's next pass would overwrite the approved
 * tag with whatever the unchanged frontmatter said.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";

const savedCalls: Array<{ file: { id: string; mime_type: string }; tags: string[] }> = [];

vi.mock("@/lib/tags", () => {
  class MockConflictError extends Error {
    constructor() {
      super("ETag mismatch");
      this.name = "ConflictError";
    }
  }
  return {
    saveFileTags: vi.fn(async (file: { id: string; mime_type: string }, tags: string[]) => {
      savedCalls.push({ file, tags });
    }),
    ConflictError: MockConflictError,
  };
});

vi.mock("@/lib/api", () => ({
  fetchJSON: vi.fn(),
}));

vi.mock("./api", () => ({
  getSuggestedTags: vi.fn(),
  dismissSuggestedTags: vi.fn(),
  regenerateSuggestedTags: vi.fn(),
}));

import SuggestedTagsSection from "@/addons/intelligence/SuggestedTagsSection";
import { fetchJSON } from "@/lib/api";
import { getSuggestedTags } from "@/addons/intelligence/api";
import { saveFileTags } from "@/lib/tags";

const mdFile = {
  id: "fMd000000001",
  filename: "note.md",
  title: "note.md",
  mime_type: "text/markdown",
  drive: "media",
  file_type: "document",
  folder_path: "",
  description: "",
  thumbnail_url: "",
  has_thumbnail: false,
  file_size: 10,
  duration: null,
  likes: 0,
  is_favorite: false,
  tags: ["existing"],
  subtitles: [],
  deleted_at: null,
  missing_since: null,
  created_at: "2026-04-01T00:00:00Z",
  updated_at: "2026-04-01T00:00:00Z",
};

beforeEach(() => {
  savedCalls.length = 0;
  vi.clearAllMocks();
  vi.mocked(getSuggestedTags).mockResolvedValue({
    available: true,
    tags: ["new-one", "new-two"],
    status: "pending",
    model: "test-model",
  } as any);
  vi.mocked(fetchJSON).mockResolvedValue(mdFile);
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("SuggestedTagsSection accept dispatch", () => {
  it("accept single tag routes through saveFileTags with merged list", async () => {
    render(<SuggestedTagsSection fileId="fMd000000001" drive="media" />);
    await waitFor(() => {
      expect(screen.getByText("new-one")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByLabelText(/new-one.*追加/));

    await waitFor(() => {
      expect(saveFileTags).toHaveBeenCalled();
    });
    const last = savedCalls[savedCalls.length - 1];
    expect(last.file.id).toBe("fMd000000001");
    expect(last.file.mime_type).toBe("text/markdown");
    expect(last.tags.sort()).toEqual(["existing", "new-one"].sort());
  });

  it("accept all routes through saveFileTags with every pending tag", async () => {
    render(<SuggestedTagsSection fileId="fMd000000001" drive="media" />);
    await waitFor(() => {
      expect(screen.getByText("new-one")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText(/すべて承認/));

    await waitFor(() => {
      expect(saveFileTags).toHaveBeenCalled();
    });
    const last = savedCalls[savedCalls.length - 1];
    expect(last.tags.sort()).toEqual(
      ["existing", "new-one", "new-two"].sort(),
    );
  });

  it("retries once after ConflictError, re-merging against fresh file", async () => {
    const { ConflictError } = await import("@/lib/tags");
    const staleFile = { ...mdFile, tags: ["existing"] };
    const freshFile = { ...mdFile, tags: ["existing", "later-edit"] };
    vi.mocked(fetchJSON)
      .mockResolvedValueOnce(staleFile)
      .mockResolvedValueOnce(freshFile);
    vi.mocked(saveFileTags)
      .mockImplementationOnce(async () => {
        throw new ConflictError();
      })
      .mockImplementationOnce(async (file: any, tags: string[]) => {
        savedCalls.push({ file, tags });
      });

    render(<SuggestedTagsSection fileId="fMd000000001" drive="media" />);
    await waitFor(() => {
      expect(screen.getByText("new-one")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByLabelText(/new-one.*追加/));

    await waitFor(() => {
      expect(saveFileTags).toHaveBeenCalledTimes(2);
    });
    // Second call uses the fresh merge, preserving the out-of-band edit
    expect(savedCalls[savedCalls.length - 1].tags.sort()).toEqual(
      ["existing", "later-edit", "new-one"].sort(),
    );
  });

  it("logs other errors to console so silent failure is debuggable", async () => {
    const consoleSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    vi.mocked(saveFileTags).mockRejectedValue(new Error("500 boom"));

    render(<SuggestedTagsSection fileId="fMd000000001" drive="media" />);
    await waitFor(() => {
      expect(screen.getByText("new-one")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByLabelText(/new-one.*追加/));

    await waitFor(() => {
      expect(consoleSpy).toHaveBeenCalled();
    });
    expect(consoleSpy.mock.calls[0][0]).toContain("accept-tag failed");
  });

  it("never calls the old PUT /files/{id}/tags endpoint directly", async () => {
    render(<SuggestedTagsSection fileId="fMd000000001" drive="media" />);
    await waitFor(() => {
      expect(screen.getByText("new-one")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByLabelText(/new-one.*追加/));
    await waitFor(() => expect(saveFileTags).toHaveBeenCalled());

    const putCalls = vi
      .mocked(fetchJSON)
      .mock.calls.filter(
        ([url, init]) =>
          typeof url === "string" &&
          url.includes("/tags") &&
          typeof init === "object" &&
          (init as RequestInit | undefined)?.method === "PUT",
      );
    expect(putCalls).toHaveLength(0);
  });
});
