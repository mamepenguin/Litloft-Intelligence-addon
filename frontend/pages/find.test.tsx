/**
 * Tests for the Intelligence Find page (``/addons/intelligence/find``).
 *
 * Spec: ``docs/superpowers/specs/2026-04-30-intelligence-find-mode.md``
 *
 * Behaviour pinned by this suite:
 *  1. Renders a query input + submit button.
 *  2. On submit, calls the new ``findFiles(question, options)`` API
 *     client (mocked) with the active drive.
 *  3. Renders one chip per non-"none"/non-empty decomposed slot.
 *  4. Clicking a chip's × invokes ``findFiles`` again with
 *     ``overrides`` having that slot reset to "none" (or "" for
 *     ``semantic_query``).
 *  5. Renders a file card per ``results[i]`` entry: thumbnail,
 *     filename, file_type, viewed_at, score, and ``hit.text`` snippet.
 *  6. Shows the total count message ("8 件" style).
 *  7. Empty results renders a graceful empty-state message, not an
 *     error / 404.
 *  8. Loading state shows a spinner / skeleton while the request is
 *     pending.
 *  9. Error state shows an error message when ``findFiles`` rejects.
 *
 * The Find page's production code does not yet exist; these tests are
 * intentionally RED until Phase 4 implements ``pages/find.tsx`` plus
 * the ``FindChip`` and ``findFiles`` exports.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  render,
  screen,
  fireEvent,
  waitFor,
  cleanup,
  act,
} from "@testing-library/react";
import React from "react";

vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams(""),
  usePathname: () => "/drive/family/addons/intelligence/find",
}));

vi.mock("@/components/CurrentDriveProvider", () => ({
  useCurrentDrive: () => "family",
}));

// Mock the (yet-unimplemented) findFiles export. Using vi.hoisted so
// the same fn instance is referenced from both the mock factory and
// the test body — vi.mock factories are hoisted above imports.
const findFilesMock = vi.hoisted(() => vi.fn());

vi.mock("../api", async () => {
  const actual = await vi.importActual<typeof import("../api")>("../api");
  return {
    ...actual,
    findFiles: findFilesMock,
  };
});

import FindPage from "./find";
import { accentFills } from "@/__tests__/helpers/accentFills";
import type { FindResponse } from "../api";

const fullResponse: FindResponse = {
  decomposed: {
    time_range: {
      kind: "relative",
      value: "last_week",
      after: "2026-04-23T00:00:00Z",
      before: "2026-04-30T00:00:00Z",
    },
    personal_scope: "viewed",
    file_type_hint: "video",
    semantic_query: "SF",
    category_expansion: ["science fiction", "宇宙船"],
  },
  results: [
    {
      file_id: "f-abc123",
      score: 0.82,
      hit: {
        kind: "transcript",
        location: { start_seconds: 415.2, end_seconds: 460.0 },
        text: "...宇宙船が時空を超えて...",
      },
      file: {
        name: "Interstellar.mp4",
        file_type: "video",
        thumbnail_url: "/api/files/f-abc123/thumbnail",
        viewed_at: "2026-04-27T19:30:00Z",
      },
    },
    {
      file_id: "f-def456",
      score: 0.74,
      hit: {
        kind: "transcript",
        location: { start_seconds: 12, end_seconds: 32 },
        text: "...エイリアンが地球に降り立つ...",
      },
      file: {
        name: "Arrival.mp4",
        file_type: "video",
        thumbnail_url: "/api/files/f-def456/thumbnail",
        viewed_at: null,
      },
    },
  ],
  total: 8,
  limit: 20,
};

const emptyResponse: FindResponse = {
  decomposed: {
    time_range: { kind: "none", value: "none", after: null, before: null },
    personal_scope: "none",
    file_type_hint: "none",
    semantic_query: "",
    category_expansion: [],
  },
  results: [],
  total: 0,
  limit: 20,
};

beforeEach(() => {
  findFilesMock.mockReset();
});

afterEach(() => {
  cleanup();
});

async function submitQuery(question: string) {
  const input = await screen.findByRole("textbox");
  fireEvent.change(input, { target: { value: question } });
  const form = (input as HTMLElement).closest("form");
  if (form) {
    fireEvent.submit(form);
  } else {
    const submit = screen.getByRole("button", { name: /search|送信|find/i });
    fireEvent.click(submit);
  }
}

describe("FindPage — input + submit", () => {
  it("renders a query input and a submit affordance", async () => {
    render(<FindPage />);
    expect(await screen.findByRole("textbox")).toBeInTheDocument();
  });

  it("calls findFiles with the typed question + active drive on submit", async () => {
    findFilesMock.mockResolvedValue(fullResponse);
    render(<FindPage />);
    await submitQuery("先週観た映画でSFっぽいの");

    await waitFor(() => {
      expect(findFilesMock).toHaveBeenCalled();
    });
    const call = findFilesMock.mock.calls[0];
    expect(call[0]).toBe("先週観た映画でSFっぽいの");
    // Either positional drive arg or options-bag — accept the union the
    // implementation will land on (mirrors the existing api.ts shape:
    // drive is positional, options are in the trailing object).
    const driveArg = call[1];
    expect(driveArg).toBe("family");
  });
});

describe("FindPage — chips from decomposed", () => {
  it("renders one chip per non-'none' / non-empty decomposed slot", async () => {
    findFilesMock.mockResolvedValue(fullResponse);
    render(<FindPage />);
    await submitQuery("先週観た映画でSF");

    await waitFor(() => {
      expect(findFilesMock).toHaveBeenCalled();
    });
    // Wait for the chip row to materialize.
    await waitFor(() => {
      expect(screen.getAllByRole("button", { name: /×|remove|削除/i }).length)
        .toBeGreaterThanOrEqual(4);
    });
    // Each non-"none" slot from fullResponse must be present somewhere
    // in the chip row text.
    const chipText = screen.getByTestId("find-chips").textContent ?? "";
    expect(chipText).toMatch(/last_week|先週|4\/23|4\/30/);
    expect(chipText).toMatch(/viewed|視聴/);
    expect(chipText).toMatch(/video/i);
    expect(chipText).toMatch(/SF/);
  });

  it("does not render chips for 'none' / empty slots", async () => {
    findFilesMock.mockResolvedValue({
      ...fullResponse,
      decomposed: {
        ...fullResponse.decomposed,
        personal_scope: "none",
        file_type_hint: "none",
      },
    });
    render(<FindPage />);
    await submitQuery("SF");

    await waitFor(() => expect(findFilesMock).toHaveBeenCalled());
    await waitFor(() => {
      const chips = screen.queryByTestId("find-chips");
      // Should still render the chips region with the remaining 2 chips
      // (time_range + semantic_query) but NOT the personal/file_type ones.
      expect(chips?.textContent ?? "").not.toMatch(/viewed|視聴/);
    });
  });

  it("clicking a chip × calls findFiles again with that slot reset to 'none'", async () => {
    findFilesMock.mockResolvedValue(fullResponse);
    render(<FindPage />);
    await submitQuery("先週観た映画でSF");

    // The chips are rendered from the *response*, so waiting on the mock
    // having been called does not wait for them: the call happens during
    // the first commit and the wait is already satisfied when it runs.
    // Wait for the chips themselves.
    //
    // Find the time-range chip's × button. The chip carries a
    // ``data-slot="time_range"`` marker (or the aria-label mentions
    // the slot label) so we can target it deterministically.
    const chips = await screen.findAllByRole("button", { name: /×|remove|削除/i });
    // Identify the time_range chip — the one whose enclosing chip
    // element advertises the slot.
    const timeChip = chips.find((btn) => {
      const chipRoot = btn.closest("[data-slot]");
      return chipRoot?.getAttribute("data-slot") === "time_range";
    });
    expect(timeChip).toBeDefined();
    await act(async () => {
      timeChip!.click();
    });

    await waitFor(() => expect(findFilesMock).toHaveBeenCalledTimes(2));
    const secondCall = findFilesMock.mock.calls[1];
    // Options bag is the trailing argument. The implementation should
    // build ``overrides`` from the current decomposed snapshot with the
    // clicked slot replaced by "none".
    const opts = secondCall[secondCall.length - 1] as
      | { overrides?: Record<string, string> }
      | undefined;
    expect(opts?.overrides?.time_range).toBe("none");
    // Other slots stay at their currently-resolved values.
    expect(opts?.overrides?.personal_scope).toBe("viewed");
    expect(opts?.overrides?.file_type_hint).toBe("video");
    expect(opts?.overrides?.semantic_query).toBe("SF");
  });

  it("clicking the semantic_query chip × resets it to '' (empty), not 'none'", async () => {
    findFilesMock.mockResolvedValue(fullResponse);
    render(<FindPage />);
    await submitQuery("SF");

    // Same race as above: wait for the chips the response renders, not for
    // the request that starts it.
    const chips = await screen.findAllByRole("button", { name: /×|remove|削除/i });
    const semanticChip = chips.find((btn) => {
      const chipRoot = btn.closest("[data-slot]");
      return chipRoot?.getAttribute("data-slot") === "semantic_query";
    });
    expect(semanticChip).toBeDefined();
    await act(async () => {
      semanticChip!.click();
    });

    await waitFor(() => expect(findFilesMock).toHaveBeenCalledTimes(2));
    const opts = findFilesMock.mock.calls[1][findFilesMock.mock.calls[1].length - 1] as
      | { overrides?: Record<string, string> }
      | undefined;
    expect(opts?.overrides?.semantic_query).toBe("");
  });
});

describe("FindPage — results list", () => {
  it("renders a card for each result with filename, score, viewed_at, hit text", async () => {
    findFilesMock.mockResolvedValue(fullResponse);
    render(<FindPage />);
    await submitQuery("SF");

    await waitFor(() => {
      expect(screen.getByText("Interstellar.mp4")).toBeInTheDocument();
      expect(screen.getByText("Arrival.mp4")).toBeInTheDocument();
    });
    // Hit text snippets show through.
    expect(screen.getByText(/宇宙船が時空を超えて/)).toBeInTheDocument();
    expect(screen.getByText(/エイリアンが地球に降り立つ/)).toBeInTheDocument();
    // Score appears (rendered as 0.82 / 0.74 — be lenient about
    // formatting precision).
    expect(screen.getByText(/0\.82/)).toBeInTheDocument();
    expect(screen.getByText(/0\.74/)).toBeInTheDocument();
  });

  it("renders the thumbnail with the spec'd thumbnail_url", async () => {
    findFilesMock.mockResolvedValue(fullResponse);
    render(<FindPage />);
    await submitQuery("SF");

    await waitFor(() => {
      const imgs = document.querySelectorAll<HTMLImageElement>("img");
      const srcs = Array.from(imgs).map((img) => img.getAttribute("src"));
      expect(srcs).toContain("/api/files/f-abc123/thumbnail");
      expect(srcs).toContain("/api/files/f-def456/thumbnail");
    });
  });

  it("renders the total count message", async () => {
    findFilesMock.mockResolvedValue(fullResponse);
    render(<FindPage />);
    await submitQuery("SF");

    // Tolerate the implementation choosing "8 件見つかりました" vs
    // "8 results" vs "8 件" — pin only the count + "件 / result"
    // marker so wording can evolve without churning the test.
    await waitFor(() => {
      const body = document.body.textContent ?? "";
      expect(body).toMatch(/\b8\b/);
      expect(body).toMatch(/件|result/i);
    });
  });
});

describe("FindPage — empty / loading / error", () => {
  it("renders a graceful empty-state message when results is []", async () => {
    findFilesMock.mockResolvedValue(emptyResponse);
    render(<FindPage />);
    await submitQuery("nonsense");

    await waitFor(() => expect(findFilesMock).toHaveBeenCalled());
    // We do NOT throw, do NOT render a 404 or "error" surface — the
    // empty-state copy is informational. Pin the absence of error
    // signals + presence of the empty marker (data-testid is the
    // stable hook so wording can be translated without churn).
    await waitFor(() => {
      expect(screen.getByTestId("find-empty")).toBeInTheDocument();
    });
    expect(screen.queryByTestId("find-error")).toBeNull();
  });

  it("shows a loading indicator while findFiles is pending", async () => {
    let resolve!: (v: FindResponse) => void;
    findFilesMock.mockImplementation(
      () =>
        new Promise<FindResponse>((r) => {
          resolve = r;
        }),
    );
    render(<FindPage />);
    await submitQuery("SF");

    await waitFor(() => {
      expect(screen.getByTestId("find-loading")).toBeInTheDocument();
    });
    // Resolve so React's effects can settle before teardown.
    await act(async () => {
      resolve(emptyResponse);
    });
    await waitFor(() => {
      expect(screen.queryByTestId("find-loading")).toBeNull();
    });
  });

  it("renders an error message when findFiles rejects", async () => {
    findFilesMock.mockRejectedValue(new Error("boom"));
    render(<FindPage />);
    await submitQuery("SF");

    await waitFor(() => {
      expect(screen.getByTestId("find-error")).toBeInTheDocument();
    });
    // Empty-state must NOT be rendered alongside the error — error
    // wins (otherwise the user sees a contradictory "no results" +
    // "request failed" double surface).
    expect(screen.queryByTestId("find-empty")).toBeNull();
  });
});

/**
 * The page's own chrome, after adopting core's `PageHeader` and `PageTabs`
 * (UI redesign Phase 3, C2a). The Ask page carries the same three.
 */
describe("FindPage — page header, mode tabs and accent budget", () => {
  it("names itself once, and lets core choose the size", () => {
    const { container } = render(<FindPage />);
    const h1s = container.querySelectorAll("h1");
    expect(h1s).toHaveLength(1);
    expect(h1s[0].className).toContain("text-2xl");
  });

  it("marks the current mode the way a set of links does, and not twice", () => {
    render(<FindPage />);
    const find = screen.getByRole("link", { name: /find/i });
    expect(find).toHaveAttribute("aria-current", "page");
    expect(find).not.toHaveAttribute("aria-selected");
    expect(screen.queryByRole("tablist")).toBeNull();
    expect(screen.queryAllByRole("tab")).toHaveLength(0);
    const ask = screen.getByRole("link", { name: /ask/i });
    expect(ask).not.toHaveAttribute("aria-current");
  });

  it("spends its one accent fill on searching", () => {
    // Two before this migration: the submit button and the selected mode tab,
    // which spent the screen's fill on saying which mode you are already
    // looking at. `PageTabs` marks the selection with a border instead, so
    // this also fails if the tab row goes back to filling.
    const { container } = render(<FindPage />);
    const fills = accentFills(container);
    expect(fills).toHaveLength(1);
    expect(fills[0]).toHaveAttribute("type", "submit");
  });
});
