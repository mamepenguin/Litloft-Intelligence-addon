/**
 * SPEC-ADDON-024: the Similar files section searches on mount, starts
 * expanded, and aborts its open request on a new file/drive pair and on
 * unmount.
 */

import { StrictMode } from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { act, render, screen, fireEvent } from "@testing-library/react";

vi.mock("@/addons/intelligence/api", async () => {
  // Only the request is stubbed. `SIMILAR_FILES_LIMIT` stays the real
  // one, so the ghosts are checked against the limit the wire uses.
  const actual = await vi.importActual<
    typeof import("@/addons/intelligence/api")
  >("@/addons/intelligence/api");
  return { ...actual, getSimilarFiles: vi.fn() };
});

import SimilarFilesSection from "@/addons/intelligence/SimilarFilesSection";
import {
  getSimilarFiles,
  SIMILAR_FILES_LIMIT,
} from "@/addons/intelligence/api";

type Result = {
  file_id: string;
  filename: string;
  shared_keywords: { word: string; score: number }[];
};

const results: Result[] = [
  {
    file_id: "other1",
    filename: "neighbour.mp4",
    shared_keywords: [{ word: "ocean", score: 0.9 }],
  },
];

const staleResults: Result[] = [
  {
    file_id: "stale1",
    filename: "stale.mp4",
    shared_keywords: [{ word: "desert", score: 0.8 }],
  },
];

type Call = {
  fileId: string;
  drive: string;
  signal: AbortSignal | undefined;
  resolve: (value: unknown) => void;
  reject: (reason: unknown) => void;
};

/** Every call gets its own promise, settled by the test. */
function deferEveryCall(): Call[] {
  const calls: Call[] = [];
  vi.mocked(getSimilarFiles).mockImplementation(((
    fileId: string,
    drive: string,
    opts?: { signal?: AbortSignal },
  ) =>
    new Promise((resolve, reject) => {
      calls.push({ fileId, drive, signal: opts?.signal, resolve, reject });
    })) as never);
  return calls;
}

function section(fileId = "f1", drive = "media") {
  return <SimilarFilesSection fileId={fileId} drive={drive} />;
}

const header = () => screen.getByRole("button", { name: /Similar files/ });
const ghosts = () => screen.queryByTestId("similar-files-ghosts");
const unavailable = () =>
  screen.queryByText(/Similar file detection unavailable/i);

async function flush(ms = 0) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(getSimilarFiles).mockResolvedValue({ results } as never);
});

afterEach(() => {
  vi.useRealTimers();
});

describe("SimilarFilesSection on mount (SPEC-ADDON-024)", () => {
  it("SPEC-ADDON-024 I1/I2: mounts expanded, shows the ghosts and sends one request with its own signal", () => {
    const calls = deferEveryCall();
    render(section("f1", "media"));

    expect(header()).toHaveAttribute("aria-expanded", "true");
    expect(ghosts()).not.toBeNull();
    expect(ghosts()!.children).toHaveLength(SIMILAR_FILES_LIMIT);
    expect(header().textContent).not.toContain("(");

    expect(getSimilarFiles).toHaveBeenCalledTimes(1);
    expect(calls).toHaveLength(1);
    expect(calls[0].fileId).toBe("f1");
    expect(calls[0].drive).toBe("media");
    expect(calls[0].signal).toBeInstanceOf(AbortSignal);
    expect(calls[0].signal!.aborted).toBe(false);
  });

  it("SPEC-ADDON-024 item 2: never shows a detect button, before or after the answer", async () => {
    render(section());
    expect(screen.queryByRole("button", { name: /Find similar files/ })).toBeNull();
    expect(await screen.findByText("neighbour.mp4")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Find similar files/ })).toBeNull();
  });

  it("SPEC-ADDON-024 item 3: the answer replaces the ghosts and the heading counts it", async () => {
    const calls = deferEveryCall();
    render(section());
    expect(calls).toHaveLength(1);

    await act(async () => {
      calls[0].resolve({ results });
    });

    expect(screen.getByText("neighbour.mp4")).toBeInTheDocument();
    expect(ghosts()).toBeNull();
    expect(header().textContent).toContain("Similar files (1)");
    expect(header()).toHaveAttribute("aria-expanded", "true");
  });

  it("SPEC-ADDON-024 item 3: an empty answer shows the empty text and (0)", async () => {
    const calls = deferEveryCall();
    render(section());
    expect(calls).toHaveLength(1);

    await act(async () => {
      calls[0].resolve({ results: [] });
    });

    expect(screen.getByText("No similar files found")).toBeInTheDocument();
    expect(ghosts()).toBeNull();
    expect(header().textContent).toContain("Similar files (0)");
  });

  it("SPEC-ADDON-024 item 6: a remount on the same file shows the ghosts and asks again", async () => {
    const first = render(section("f1"));
    await screen.findByText("neighbour.mp4");
    first.unmount();

    const calls = deferEveryCall();
    render(section("f1"));

    expect(header()).toHaveAttribute("aria-expanded", "true");
    expect(ghosts()).not.toBeNull();
    expect(calls).toHaveLength(1);
    expect(calls[0].fileId).toBe("f1");
    expect(getSimilarFiles).toHaveBeenCalledTimes(2);
  });
});

describe("SimilarFilesSection with the same pair (SPEC-ADDON-024 I3/I4)", () => {
  it("SPEC-ADDON-024 I3: collapsing and expanding sends no request and shows the results again", async () => {
    render(section());
    await screen.findByText("neighbour.mp4");

    fireEvent.click(header());
    expect(header()).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByText("neighbour.mp4")).toBeNull();

    fireEvent.click(header());
    expect(header()).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByText("neighbour.mp4")).toBeInTheDocument();
    expect(getSimilarFiles).toHaveBeenCalledTimes(1);
  });

  it("SPEC-ADDON-024 I3: re-rendering with the same fileId and drive sends no request", async () => {
    const { rerender } = render(section("f1", "media"));
    await screen.findByText("neighbour.mp4");

    rerender(section("f1", "media"));
    rerender(section("f1", "media"));

    expect(screen.getByText("neighbour.mp4")).toBeInTheDocument();
    expect(getSimilarFiles).toHaveBeenCalledTimes(1);
  });

  it("SPEC-ADDON-024 I3: re-rendering while the request is pending neither aborts it nor sends another", async () => {
    const calls = deferEveryCall();
    const { rerender } = render(section("f1", "media"));

    rerender(section("f1", "media"));

    expect(calls).toHaveLength(1);
    expect(calls[0].signal!.aborted).toBe(false);
    await act(async () => {
      calls[0].resolve({ results });
    });
    expect(screen.getByText("neighbour.mp4")).toBeInTheDocument();
  });

  it("SPEC-ADDON-024 I4: collapsing while pending keeps the request, and its answer does not re-expand", async () => {
    const calls = deferEveryCall();
    render(section());

    fireEvent.click(header());
    expect(header()).toHaveAttribute("aria-expanded", "false");
    expect(calls[0].signal!.aborted).toBe(false);

    await act(async () => {
      calls[0].resolve({ results });
    });

    expect(header()).toHaveAttribute("aria-expanded", "false");
    expect(header().textContent).toContain("(1)");
    expect(screen.queryByText("neighbour.mp4")).toBeNull();
    expect(getSimilarFiles).toHaveBeenCalledTimes(1);

    fireEvent.click(header());
    expect(screen.getByText("neighbour.mp4")).toBeInTheDocument();
    expect(getSimilarFiles).toHaveBeenCalledTimes(1);
  });
});

describe("SimilarFilesSection on a new pair (SPEC-ADDON-024 I2/I5)", () => {
  it("SPEC-ADDON-024 I2: a new fileId after a collapse resets to expanded and loading and asks once for it", async () => {
    const { rerender } = render(section("f1"));
    await screen.findByText("neighbour.mp4");
    fireEvent.click(header());

    const calls = deferEveryCall();
    rerender(section("f2"));

    expect(header()).toHaveAttribute("aria-expanded", "true");
    expect(ghosts()).not.toBeNull();
    expect(header().textContent).not.toContain("(");
    expect(screen.queryByText("neighbour.mp4")).toBeNull();
    expect(calls).toHaveLength(1);
    expect(calls[0].fileId).toBe("f2");
    expect(calls[0].drive).toBe("media");
  });

  it("SPEC-ADDON-024 I2: a new drive with the same fileId asks again with that drive", async () => {
    const { rerender } = render(section("f1", "media"));
    await screen.findByText("neighbour.mp4");

    const calls = deferEveryCall();
    rerender(section("f1", "other"));

    expect(header()).toHaveAttribute("aria-expanded", "true");
    expect(ghosts()).not.toBeNull();
    expect(calls).toHaveLength(1);
    expect(calls[0].fileId).toBe("f1");
    expect(calls[0].drive).toBe("other");
  });

  it("SPEC-ADDON-024 I5: a new fileId aborts the old request and the old answer is never shown", async () => {
    const calls = deferEveryCall();
    const { rerender } = render(section("f1"));

    rerender(section("f2"));

    expect(calls).toHaveLength(2);
    expect(calls[0].signal!.aborted).toBe(true);
    expect(calls[1].fileId).toBe("f2");
    expect(calls[1].signal!.aborted).toBe(false);

    await act(async () => {
      calls[0].resolve({ results: staleResults });
    });
    expect(screen.queryByText("stale.mp4")).toBeNull();
    expect(ghosts()).not.toBeNull();
    expect(header().textContent).not.toContain("(");

    await act(async () => {
      calls[1].resolve({ results });
    });
    expect(screen.getByText("neighbour.mp4")).toBeInTheDocument();
    expect(screen.queryByText("stale.mp4")).toBeNull();
    expect(header().textContent).toContain("(1)");
  });

  it("SPEC-ADDON-024 I5: the old answer landing after the new one does not replace it", async () => {
    const calls = deferEveryCall();
    const { rerender } = render(section("f1"));
    rerender(section("f2"));
    expect(calls).toHaveLength(2);

    await act(async () => {
      calls[1].resolve({ results });
    });
    await act(async () => {
      calls[0].resolve({ results: staleResults });
    });

    expect(screen.getByText("neighbour.mp4")).toBeInTheDocument();
    expect(screen.queryByText("stale.mp4")).toBeNull();
    expect(header().textContent).toContain("(1)");
  });

  it("SPEC-ADDON-024 I5: a new fileId during a backoff clears it; no retry for the old file is sent", async () => {
    vi.useFakeTimers();
    const calls = deferEveryCall();
    const { rerender } = render(section("f1"));
    expect(calls).toHaveLength(1);

    await act(async () => {
      calls[0].reject(new Error("cold"));
    });
    await flush(0);
    expect(calls).toHaveLength(1);

    rerender(section("f2"));
    expect(calls).toHaveLength(2);

    await flush(60_000);

    expect(calls.filter((c) => c.fileId === "f1")).toHaveLength(1);
    expect(calls.filter((c) => c.fileId === "f2")).toHaveLength(1);
    expect(unavailable()).toBeNull();
    expect(ghosts()).not.toBeNull();
  });

  it("SPEC-ADDON-024 I5: the old pair's unavailable state is not carried to the new pair", async () => {
    vi.useFakeTimers();
    vi.mocked(getSimilarFiles).mockRejectedValue(new Error("boom"));
    const { rerender } = render(section("f1"));
    await flush(20_000);
    expect(unavailable()).not.toBeNull();

    const calls = deferEveryCall();
    rerender(section("f2"));

    expect(unavailable()).toBeNull();
    expect(ghosts()).not.toBeNull();
    expect(header()).toHaveAttribute("aria-expanded", "true");
    expect(calls).toHaveLength(1);
    expect(calls[0].fileId).toBe("f2");
  });
});

describe("SimilarFilesSection failures and retry (SPEC-ADDON-024 I2/I6/I10)", () => {
  it("SPEC-ADDON-024 I6/I2: retries after 6 s and 12 s, ghosts and no count during both backoffs, then unavailable", async () => {
    vi.useFakeTimers();
    vi.mocked(getSimilarFiles).mockRejectedValue(new Error("boom"));
    render(section());

    await flush(0);
    expect(getSimilarFiles).toHaveBeenCalledTimes(1);
    expect(ghosts()).not.toBeNull();
    expect(header().textContent).not.toContain("(");
    expect(unavailable()).toBeNull();

    await flush(5_999);
    expect(getSimilarFiles).toHaveBeenCalledTimes(1);
    await flush(1);
    expect(getSimilarFiles).toHaveBeenCalledTimes(2);
    expect(ghosts()).not.toBeNull();
    expect(header().textContent).not.toContain("(");
    expect(unavailable()).toBeNull();

    await flush(11_999);
    expect(getSimilarFiles).toHaveBeenCalledTimes(2);
    expect(unavailable()).toBeNull();
    await flush(1);
    expect(getSimilarFiles).toHaveBeenCalledTimes(3);
    expect(unavailable()).not.toBeNull();
    expect(ghosts()).toBeNull();
    expect(screen.getByRole("button", { name: /Search again/ })).toBeInTheDocument();

    await flush(60_000);
    expect(getSimilarFiles).toHaveBeenCalledTimes(3);
  });

  it("SPEC-ADDON-024 I6: the retry button sends a new request and shows its answer", async () => {
    vi.useFakeTimers();
    vi.mocked(getSimilarFiles).mockRejectedValue(new Error("boom"));
    render(section());
    await flush(20_000);
    expect(unavailable()).not.toBeNull();

    vi.mocked(getSimilarFiles).mockResolvedValue({ results } as never);
    fireEvent.click(screen.getByRole("button", { name: /Search again/ }));
    await flush(0);

    expect(getSimilarFiles).toHaveBeenCalledTimes(4);
    expect(screen.getByText("neighbour.mp4")).toBeInTheDocument();
    expect(unavailable()).toBeNull();
  });

  it("SPEC-ADDON-024 item 2: the retry button starts the same 6 s / 12 s sequence", async () => {
    vi.useFakeTimers();
    vi.mocked(getSimilarFiles).mockRejectedValue(new Error("boom"));
    render(section());
    await flush(20_000);
    expect(getSimilarFiles).toHaveBeenCalledTimes(3);

    fireEvent.click(screen.getByRole("button", { name: /Search again/ }));
    await flush(0);
    expect(getSimilarFiles).toHaveBeenCalledTimes(4);
    expect(ghosts()).not.toBeNull();
    expect(unavailable()).toBeNull();

    await flush(5_999);
    expect(getSimilarFiles).toHaveBeenCalledTimes(4);
    await flush(1);
    expect(getSimilarFiles).toHaveBeenCalledTimes(5);
    await flush(11_999);
    expect(getSimilarFiles).toHaveBeenCalledTimes(5);
    await flush(1);
    expect(getSimilarFiles).toHaveBeenCalledTimes(6);
    expect(unavailable()).not.toBeNull();
  });

  it("SPEC-ADDON-024 I10: unavailable survives collapse, expand and a same-pair re-render without a request", async () => {
    vi.useFakeTimers();
    vi.mocked(getSimilarFiles).mockRejectedValue(new Error("boom"));
    const { rerender } = render(section("f1", "media"));
    await flush(20_000);
    expect(unavailable()).not.toBeNull();

    fireEvent.click(header());
    fireEvent.click(header());
    rerender(section("f1", "media"));
    await flush(60_000);

    expect(unavailable()).not.toBeNull();
    expect(ghosts()).toBeNull();
    expect(getSimilarFiles).toHaveBeenCalledTimes(3);
  });
});

describe("SimilarFilesSection on unmount (SPEC-ADDON-024 I7)", () => {
  it("SPEC-ADDON-024 I7: unmounting with a request open aborts it, and its later failure sends nothing", async () => {
    vi.useFakeTimers();
    const calls = deferEveryCall();
    const { unmount } = render(section());
    expect(calls).toHaveLength(1);

    unmount();
    expect(calls[0].signal!.aborted).toBe(true);

    await act(async () => {
      calls[0].reject(new Error("boom"));
    });
    await flush(60_000);

    expect(calls).toHaveLength(1);
    expect(vi.getTimerCount()).toBe(0);
  });

  it("SPEC-ADDON-024 I7: unmounting during a backoff leaves no timer and sends no retry", async () => {
    vi.useFakeTimers();
    vi.mocked(getSimilarFiles).mockRejectedValue(new Error("boom"));
    const { unmount } = render(section());
    await flush(0);
    expect(getSimilarFiles).toHaveBeenCalledTimes(1);

    unmount();
    expect(vi.getTimerCount()).toBe(0);

    await flush(60_000);
    expect(getSimilarFiles).toHaveBeenCalledTimes(1);
  });

  it("SPEC-ADDON-024 I7: under StrictMode it asks exactly twice, aborts the first, and ends loaded with the second's answer", async () => {
    const calls = deferEveryCall();
    render(<StrictMode>{section("f1", "media")}</StrictMode>);

    expect(getSimilarFiles).toHaveBeenCalledTimes(2);
    expect(calls[0].signal!.aborted).toBe(true);
    expect(calls[1].signal!.aborted).toBe(false);
    expect(calls[1].fileId).toBe("f1");
    expect(calls[1].drive).toBe("media");

    await act(async () => {
      calls[1].resolve({ results });
    });
    await act(async () => {
      calls[0].resolve({ results: staleResults });
    });

    expect(screen.getByText("neighbour.mp4")).toBeInTheDocument();
    expect(screen.queryByText("stale.mp4")).toBeNull();
    expect(header().textContent).toContain("(1)");
    expect(getSimilarFiles).toHaveBeenCalledTimes(2);
  });

  it("SPEC-ADDON-024 I7: under StrictMode the first run's answer landing first does not end the loading", async () => {
    const calls = deferEveryCall();
    render(<StrictMode>{section("f1", "media")}</StrictMode>);
    expect(calls).toHaveLength(2);

    await act(async () => {
      calls[0].resolve({ results: staleResults });
    });
    expect(screen.queryByText("stale.mp4")).toBeNull();
    expect(ghosts()).not.toBeNull();

    await act(async () => {
      calls[1].resolve({ results });
    });
    expect(screen.getByText("neighbour.mp4")).toBeInTheDocument();
  });
});

describe("SimilarFilesSection and aborted requests (SPEC-ADDON-024 I8)", () => {
  it.each([
    ["an AbortError", () => new DOMException("aborted", "AbortError")],
    ["an ordinary error", () => new Error("boom")],
  ])(
    "SPEC-ADDON-024 I8: an aborted request rejecting with %s starts no backoff and shows nothing",
    async (_label, makeError) => {
      vi.useFakeTimers();
      const calls = deferEveryCall();
      const { rerender } = render(section("f1"));
      rerender(section("f2"));
      expect(calls).toHaveLength(2);
      expect(calls[0].signal!.aborted).toBe(true);

      await act(async () => {
        calls[0].reject(makeError());
      });
      await flush(60_000);

      expect(calls.filter((c) => c.fileId === "f1")).toHaveLength(1);
      expect(calls).toHaveLength(2);
      expect(unavailable()).toBeNull();
      expect(ghosts()).not.toBeNull();
    },
  );

  it("SPEC-ADDON-024 I8: under StrictMode a rejected first run starts no backoff", async () => {
    vi.useFakeTimers();
    const calls = deferEveryCall();
    render(<StrictMode>{section("f1", "media")}</StrictMode>);
    expect(calls).toHaveLength(2);

    await act(async () => {
      calls[0].reject(new DOMException("aborted", "AbortError"));
    });
    await flush(60_000);
    expect(getSimilarFiles).toHaveBeenCalledTimes(2);
    expect(unavailable()).toBeNull();

    await act(async () => {
      calls[1].resolve({ results });
    });
    expect(screen.getByText("neighbour.mp4")).toBeInTheDocument();
  });
});

describe("SimilarFilesSection thumbnails (SPEC-ADDON-024 I9)", () => {
  it("SPEC-ADDON-024 I9: every result thumbnail loads lazily", async () => {
    const many: Result[] = Array.from({ length: SIMILAR_FILES_LIMIT }, (_, i) => ({
      file_id: `n${i}`,
      filename: `neighbour-${i}.mp4`,
      shared_keywords: [{ word: "ocean", score: 0.9 }],
    }));
    vi.mocked(getSimilarFiles).mockResolvedValue({ results: many } as never);
    const { container } = render(section());
    await screen.findByText("neighbour-0.mp4");

    const imgs = Array.from(container.querySelectorAll("img"));
    expect(imgs.length).toBeGreaterThanOrEqual(SIMILAR_FILES_LIMIT);
    for (const img of imgs) {
      expect(img.getAttribute("loading")).toBe("lazy");
    }
  });
});

describe("SimilarFilesSection — its weight inside the Related group", () => {
  it("draws its heading at the group weight, not at section weight", async () => {
    // The values are `DESIGN.md` §The Related group; core's
    // `RelatedFilesSection` test holds the other member of the group, so
    // the two assertions are what keep the pair in step.
    render(section());

    const heading = await screen.findByRole("button", { name: /Similar files/i });
    for (const cls of ["text-xs", "font-medium", "text-text-muted"]) {
      expect(heading.classList.contains(cls)).toBe(true);
    }
    expect(heading.classList.contains("text-sm")).toBe(false);
    expect(heading.classList.contains("font-semibold")).toBe(false);
  });
});
