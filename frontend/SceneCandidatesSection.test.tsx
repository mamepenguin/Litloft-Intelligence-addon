/**
 * SPEC-ADDON-042: the scene candidates section on the search page.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor, act, cleanup } from "@testing-library/react";
import { NextIntlClientProvider } from "next-intl";

import enMessages from "@/messages/en.json";
import jaMessages from "@/messages/ja.json";

// The global setup answers every lookup from the English catalogue whatever
// the provider says, which would make the Japanese cases below pass or fail
// on the wrong catalogue.
vi.unmock("next-intl");

const push = vi.hoisted(() => vi.fn());

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push, replace: vi.fn(), back: vi.fn(), prefetch: vi.fn() }),
}));

vi.mock("@/addons/intelligence/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/addons/intelligence/api")>();
  return { ...actual, fetchSceneSegments: vi.fn() };
});

import SceneCandidatesSection from "@/addons/intelligence/SceneCandidatesSection";
import { fetchSceneSegments } from "@/addons/intelligence/api";

const fetchMock = fetchSceneSegments as unknown as ReturnType<typeof vi.fn>;

interface Segment {
  file_id: string;
  filename: string;
  start: number;
  end: number;
  peak: number;
  score: number;
}
interface Response_ {
  query: string;
  segments: Segment[];
  partial: boolean;
}

const seg = (over: Partial<Segment> = {}): Segment => ({
  file_id: "f1",
  filename: "holiday.mp4",
  start: 83,
  end: 96,
  peak: 84,
  score: 0.4321,
  ...over,
});

const response = (segments: Segment[], partial = false, query = "cats"): Response_ => ({
  query,
  segments,
  partial,
});

interface Props {
  query: string;
  drive: string;
  includeSceneClip: boolean;
  typeFilter: string | null;
  smartFolderId: string | null;
}

const BASE: Props = {
  query: "cats",
  drive: "main",
  includeSceneClip: true,
  typeFilter: null,
  smartFolderId: null,
};

function tree(props: Props, locale: "en" | "ja" = "en") {
  return (
    <NextIntlClientProvider
      locale={locale}
      messages={locale === "en" ? enMessages : jaMessages}
    >
      <SceneCandidatesSection {...props} />
    </NextIntlClientProvider>
  );
}

function mount(over: Partial<Props> = {}, locale: "en" | "ja" = "en") {
  const props = { ...BASE, ...over };
  const view = render(tree(props, locale));
  return {
    ...view,
    update: (next: Partial<Props>) => view.rerender(tree({ ...props, ...next }, locale)),
  };
}

function deferred<T>() {
  let resolve!: (v: T) => void;
  let reject!: (e: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

const HEADING_EN = "Scene candidates";
const NO_INDEX_EN = "This drive has no scene index";
const PARTIAL_EN = "Some scenes in this drive may have been missed";
const FAILED_EN = "Couldn't search scenes";
const LOADING_EN = "Searching for scenes";

beforeEach(() => {
  vi.clearAllMocks();
  fetchMock.mockReset();
  push.mockReset();
  if (!("IntersectionObserver" in globalThis)) {
    class IO {
      observe() {}
      disconnect() {}
      unobserve() {}
      takeRecords() {
        return [];
      }
    }
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (globalThis as any).IntersectionObserver = IO;
  }
});

afterEach(cleanup);

describe("when the section requests candidates (SPEC-ADDON-042)", () => {
  it.each([
    ["no type filter", { typeFilter: null }],
    ["the video type filter", { typeFilter: "video" }],
  ])("requests with the query, drive and an abort signal under %s", async (_n, over) => {
    fetchMock.mockResolvedValue(response([seg()]));
    mount(over);
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    const [query, drive, signal] = fetchMock.mock.calls[0];
    expect(query).toBe("cats");
    expect(drive).toBe("main");
    expect(signal).toBeInstanceOf(AbortSignal);
  });

  it.each([
    ["Scene search is off", { includeSceneClip: false }],
    ["the query is empty", { query: "" }],
    ["the type filter is not video", { typeFilter: "image" }],
    ["a smart folder is active", { smartFolderId: "sf1" }],
  ])("makes no request and shows nothing when %s", async (_n, over) => {
    fetchMock.mockResolvedValue(response([seg()]));
    const { container } = mount(over);
    await act(async () => {
      await Promise.resolve();
    });
    expect(fetchMock).not.toHaveBeenCalled();
    expect(screen.queryByText(HEADING_EN)).toBeNull();
    expect(screen.queryByText("holiday.mp4")).toBeNull();
    expect(container.textContent).toBe("");
  });
});

describe("the loaded state (SPEC-ADDON-042)", () => {
  it("shows the heading, the filename and the time range of each candidate in response order", async () => {
    fetchMock.mockResolvedValue(
      response([
        seg({ file_id: "f2", filename: "second-best.mp4", start: 5, end: 20, peak: 10 }),
        seg({ file_id: "f1", filename: "first.mp4", start: 83, end: 96, peak: 84 }),
      ]),
    );
    mount();
    expect(await screen.findByText("second-best.mp4")).toBeInTheDocument();
    expect(screen.getByText(HEADING_EN)).toBeInTheDocument();

    const names = screen.getAllByText(/\.mp4$/).map((el) => el.textContent);
    expect(names).toEqual(["second-best.mp4", "first.mp4"]);
    expect(screen.getByText("0:05-0:20")).toBeInTheDocument();
    expect(screen.getByText("1:23-1:36")).toBeInTheDocument();
  });

  it.each([
    ["under a minute", 5, 20, "0:05-0:20"],
    ["start floored and end rounded up", 83.7, 95.2, "1:23-1:36"],
    ["a whole-second end stays", 10, 40, "0:10-0:40"],
    ["59:50 to 1:00:10 across the hour, each value on its own", 3590, 3610, "59:50-1:00:10"],
    ["from exactly one hour", 3600, 3661, "1:00:00-1:01:01"],
    ["a fractional end just above a minute", 0, 60.01, "0:00-1:01"],
  ])("formats the range: %s", async (_n, start, end, expected) => {
    fetchMock.mockResolvedValue(response([seg({ start, end, peak: 4 })]));
    mount();
    expect(await screen.findByText(expected)).toBeInTheDocument();
  });

  it("loads the frame lazily from the frame route at the peak time", async () => {
    fetchMock.mockResolvedValue(response([seg({ file_id: "f9", start: 83, peak: 90, end: 100 })]));
    const { container } = mount();
    await screen.findByText("holiday.mp4");
    const img = container.querySelector("img");
    expect(img).not.toBeNull();
    expect(img!.getAttribute("src")).toMatch(/\/files\/f9\/frame\?t=90$/);
    expect(img!.getAttribute("loading")).toBe("lazy");
  });

  it.each(["ref.loft", "ref.LOFT"])(
    "requests no frame for a %s candidate and still requests one for a video",
    async (loftName) => {
      fetchMock.mockResolvedValue(
        response([
          seg({ file_id: "f1", filename: loftName, start: 0, end: 10 }),
          seg({ file_id: "f2", filename: "holiday.mp4", start: 83, end: 96, peak: 84 }),
        ]),
      );
      const { container } = mount();
      await screen.findByText(loftName);

      const cards = Array.from(container.querySelectorAll("a[href^='/files/']"));
      expect(cards).toHaveLength(2);
      expect(cards[0].querySelector("img")).toBeNull();
      expect(cards[0].textContent).toContain("0:00-0:10");
      expect(cards[1].querySelector("img")?.getAttribute("src")).toMatch(
        /\/files\/f2\/frame\?t=84$/,
      );
      expect(container.querySelectorAll("img")).toHaveLength(1);
    },
  );

  it("shows no score", async () => {
    fetchMock.mockResolvedValue(response([seg({ score: 0.4321 })]));
    const { container } = mount();
    await screen.findByText("holiday.mp4");
    expect(container.textContent).not.toMatch(/0[.,]4321|0[.,]43|43\s?%/);
  });

  it("keeps the candidate usable when its frame fails to load", async () => {
    fetchMock.mockResolvedValue(response([seg({ file_id: "f1", start: 83 })]));
    const { container } = mount();
    await screen.findByText("holiday.mp4");
    fireEvent.error(container.querySelector("img")!);

    expect(screen.getByText("holiday.mp4")).toBeInTheDocument();
    expect(screen.getByText("1:23-1:36")).toBeInTheDocument();
    fireEvent.click(screen.getByText("holiday.mp4"));
    expect(navigatedTo()).toEqual({ pathname: "/files/f1", t: "83" });
  });
});

/** The destination of the candidate just activated: a router push or the link's own href. */
function navigatedTo(): { pathname: string; t: string | null } | null {
  const target =
    push.mock.calls.at(-1)?.[0] ??
    (document.querySelector("a[href^='/files/']") as HTMLAnchorElement | null)?.getAttribute("href");
  if (!target) return null;
  const url = new URL(String(target), "http://localhost");
  return { pathname: url.pathname, t: url.searchParams.get("t") };
}

describe("opening a candidate (SPEC-ADDON-042)", () => {
  it.each([
    ["a whole second", 83, "83"],
    ["a fractional start is floored", 83.9, "83"],
    ["the start of the video", 0, "0"],
    ["past an hour", 3725.4, "3725"],
  ])("opens the detail page at ?t= the floored start: %s", async (_n, start, t) => {
    fetchMock.mockResolvedValue(response([seg({ file_id: "f1", start, end: start + 10, peak: 4 })]));
    mount();
    fireEvent.click(await screen.findByText("holiday.mp4"));
    expect(navigatedTo()).toEqual({ pathname: "/files/f1", t });
  });
});

describe("the empty, partial and failed states (SPEC-ADDON-042)", () => {
  it("says the drive has no scene index when there are no candidates and partial is false", async () => {
    fetchMock.mockResolvedValue(response([], false));
    mount();
    expect(await screen.findByText(NO_INDEX_EN)).toBeInTheDocument();
    expect(screen.getByText(HEADING_EN)).toBeInTheDocument();
    expect(screen.queryByText(PARTIAL_EN)).toBeNull();
  });

  it("shows only the partial line when there are no candidates and partial is true", async () => {
    fetchMock.mockResolvedValue(response([], true));
    mount();
    expect(await screen.findByText(PARTIAL_EN)).toBeInTheDocument();
    expect(screen.getByText(HEADING_EN)).toBeInTheDocument();
    expect(screen.queryByText(NO_INDEX_EN)).toBeNull();
  });

  it("shows the partial line next to the candidates when partial is true", async () => {
    fetchMock.mockResolvedValue(response([seg()], true));
    mount();
    expect(await screen.findByText("holiday.mp4")).toBeInTheDocument();
    expect(screen.getByText(PARTIAL_EN)).toBeInTheDocument();
    expect(screen.queryByText(NO_INDEX_EN)).toBeNull();
  });

  it("shows no partial line when partial is false", async () => {
    fetchMock.mockResolvedValue(response([seg()], false));
    mount();
    await screen.findByText("holiday.mp4");
    expect(screen.queryByText(PARTIAL_EN)).toBeNull();
  });

  it("shows the heading and a one-line failure, with no retry control, when the request fails", async () => {
    fetchMock.mockRejectedValue(new Error("API error: 502"));
    mount();
    expect(await screen.findByText(FAILED_EN)).toBeInTheDocument();
    expect(screen.getByText(HEADING_EN)).toBeInTheDocument();
    expect(screen.queryByRole("button")).toBeNull();
  });

  it("enters idle and shows nothing when the route answers 404", async () => {
    fetchMock.mockResolvedValue(null);
    const { container } = mount();
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    await act(async () => {
      await Promise.resolve();
    });
    expect(screen.queryByText(HEADING_EN)).toBeNull();
    expect(screen.queryByText(LOADING_EN)).toBeNull();
    expect(container.textContent).toBe("");
  });
});

describe("the loading state (SPEC-ADDON-042)", () => {
  it("shows the heading and the searching line while the request is in flight", async () => {
    const d = deferred<Response_>();
    fetchMock.mockReturnValue(d.promise);
    mount();
    expect(screen.getByText(HEADING_EN)).toBeInTheDocument();
    expect(screen.getByText(LOADING_EN)).toBeInTheDocument();

    await act(async () => {
      d.resolve(response([seg()]));
    });
    expect(await screen.findByText("holiday.mp4")).toBeInTheDocument();
    expect(screen.queryByText(LOADING_EN)).toBeNull();
  });
});

describe("transitions (SPEC-ADDON-042)", () => {
  it("does not refetch on a re-render with the same inputs", async () => {
    fetchMock.mockResolvedValue(response([seg()]));
    const view = mount();
    await screen.findByText("holiday.mp4");
    view.update({});
    view.update({});
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it.each<[string, Partial<Props>, Partial<Props>]>([
    ["query", { query: "dogs" }, {}],
    ["drive", { drive: "other" }, {}],
    ["type filter video to none", { typeFilter: null }, { typeFilter: "video" }],
  ])("cancels the current request and loads again when the %s changes", async (_n, change, initial) => {
    const first = deferred<Response_>();
    const second = deferred<Response_>();
    fetchMock.mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);
    const view = mount(initial);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const firstSignal = fetchMock.mock.calls[0][2] as AbortSignal;

    view.update(change);

    expect(firstSignal.aborted).toBe(true);
    expect(fetchMock).toHaveBeenCalledTimes(2);
    const secondSignal = fetchMock.mock.calls[1][2] as AbortSignal;
    expect(secondSignal.aborted).toBe(false);
    expect(screen.getByText(LOADING_EN)).toBeInTheDocument();
  });

  it("ignores the response to an earlier query", async () => {
    const first = deferred<Response_>();
    const second = deferred<Response_>();
    fetchMock.mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);
    const view = mount({ query: "cats" });
    view.update({ query: "dogs" });

    await act(async () => {
      first.resolve(response([seg({ filename: "stale.mp4" })], false, "cats"));
    });
    expect(screen.queryByText("stale.mp4")).toBeNull();
    expect(screen.getByText(LOADING_EN)).toBeInTheDocument();

    await act(async () => {
      second.resolve(response([seg({ filename: "fresh.mp4" })], false, "dogs"));
    });
    expect(await screen.findByText("fresh.mp4")).toBeInTheDocument();
    expect(screen.queryByText("stale.mp4")).toBeNull();
  });

  it("ignores a late response that would replace newer candidates", async () => {
    const first = deferred<Response_>();
    const second = deferred<Response_>();
    fetchMock.mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);
    const view = mount({ query: "cats" });
    view.update({ query: "dogs" });
    await act(async () => {
      second.resolve(response([seg({ filename: "fresh.mp4" })], false, "dogs"));
    });
    await screen.findByText("fresh.mp4");

    await act(async () => {
      first.resolve(response([seg({ filename: "stale.mp4" })], false, "cats"));
    });
    expect(screen.queryByText("stale.mp4")).toBeNull();
    expect(screen.getByText("fresh.mp4")).toBeInTheDocument();
  });

  it("ignores an earlier failure once a newer request is current", async () => {
    const first = deferred<Response_>();
    const second = deferred<Response_>();
    fetchMock.mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);
    const view = mount({ query: "cats" });
    view.update({ query: "dogs" });

    await act(async () => {
      first.reject(new Error("API error: 500"));
    });
    expect(screen.queryByText(FAILED_EN)).toBeNull();
    expect(screen.getByText(LOADING_EN)).toBeInTheDocument();
  });

  it.each([
    ["Scene search turned off", { includeSceneClip: false }],
    ["the type filter set to image", { typeFilter: "image" }],
    ["a smart folder chosen", { smartFolderId: "sf1" }],
    ["the query cleared", { query: "" }],
  ])("goes idle and cancels the request when %s", async (_n, change) => {
    const d = deferred<Response_>();
    fetchMock.mockReturnValueOnce(d.promise);
    const view = mount();
    const signal = fetchMock.mock.calls[0][2] as AbortSignal;

    view.update(change);

    expect(signal.aborted).toBe(true);
    expect(screen.queryByText(HEADING_EN)).toBeNull();
    expect(screen.queryByText(LOADING_EN)).toBeNull();
  });

  it("a response to a request made before going idle cannot leave idle", async () => {
    const d = deferred<Response_>();
    fetchMock.mockReturnValueOnce(d.promise);
    const view = mount();
    view.update({ includeSceneClip: false });

    await act(async () => {
      d.resolve(response([seg({ filename: "stale.mp4" })]));
    });
    expect(screen.queryByText("stale.mp4")).toBeNull();
    expect(screen.queryByText(HEADING_EN)).toBeNull();
  });

  it("a failure of a request made before going idle cannot leave idle", async () => {
    const d = deferred<Response_>();
    fetchMock.mockReturnValueOnce(d.promise);
    const view = mount();
    view.update({ smartFolderId: "sf1" });

    await act(async () => {
      d.reject(new Error("API error: 500"));
    });
    expect(screen.queryByText(FAILED_EN)).toBeNull();
    expect(screen.queryByText(HEADING_EN)).toBeNull();
  });

  it("a 404 answered after going idle and back does not hide the new request", async () => {
    const first = deferred<Response_ | null>();
    const second = deferred<Response_>();
    fetchMock.mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);
    const view = mount();
    view.update({ includeSceneClip: false });
    view.update({ includeSceneClip: true });
    expect(screen.getByText(LOADING_EN)).toBeInTheDocument();

    await act(async () => {
      first.resolve(null);
    });
    expect(screen.getByText(LOADING_EN)).toBeInTheDocument();
  });

  it("leaves failed only through a changed input, which loads again", async () => {
    fetchMock.mockRejectedValueOnce(new Error("API error: 500"));
    const view = mount({ query: "cats" });
    await screen.findByText(FAILED_EN);

    const d = deferred<Response_>();
    fetchMock.mockReturnValueOnce(d.promise);
    view.update({ query: "dogs" });
    expect(screen.queryByText(FAILED_EN)).toBeNull();
    expect(screen.getByText(LOADING_EN)).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("returns from idle to loading when the conditions hold again", async () => {
    fetchMock.mockResolvedValue(response([seg()]));
    const view = mount({ includeSceneClip: false });
    expect(fetchMock).not.toHaveBeenCalled();
    view.update({ includeSceneClip: true });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(await screen.findByText("holiday.mp4")).toBeInTheDocument();
  });
});

describe("the section's text in Japanese (SPEC-ADDON-042)", () => {
  it("shows the Japanese heading and loading line", async () => {
    fetchMock.mockReturnValue(new Promise(() => {}));
    mount({}, "ja");
    expect(screen.getByText("シーン候補")).toBeInTheDocument();
    expect(screen.getByText("シーンを探しています")).toBeInTheDocument();
  });

  it("shows the Japanese failure line", async () => {
    fetchMock.mockRejectedValue(new Error("x"));
    mount({}, "ja");
    expect(await screen.findByText("シーンを検索できませんでした")).toBeInTheDocument();
  });

  it("shows the Japanese partial line", async () => {
    fetchMock.mockResolvedValue(response([], true));
    mount({}, "ja");
    expect(
      await screen.findByText("このドライブのシーンを見逃している可能性があります"),
    ).toBeInTheDocument();
  });

  it("shows the Japanese no-index line", async () => {
    fetchMock.mockResolvedValue(response([], false));
    mount({}, "ja");
    expect(
      await screen.findByText("このドライブにはシーンの索引がありません"),
    ).toBeInTheDocument();
  });
});
