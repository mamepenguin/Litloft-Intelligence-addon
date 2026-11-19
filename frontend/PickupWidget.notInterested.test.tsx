/**
 * SPEC-ADDON-154 rule 9 and 11: Not interested on the drive home's Pickup row.
 *
 * `CarouselSection` is stood in by a list that draws each card's `extraMenuItems` rows as
 * buttons, which is the contract SPEC-CORE-150 rule 1 gives the addon; the toasts are the
 * real provider's.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";

vi.mock("@/addons/intelligence/api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/addons/intelligence/api")>()),
  fetchPickup: vi.fn(),
}));

vi.mock("@/lib/api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/api")>()),
  batchGetFiles: vi.fn(),
}));

const profile = { nickname: "alice" as string | null };
vi.mock("@/components/ProfileProvider", () => ({
  useProfile: () => ({ nickname: profile.nickname, setNickname: vi.fn(), clearNickname: vi.fn() }),
}));

type Row = { id: string; label: string; onSelect: () => unknown; disabled?: boolean };

vi.mock("@/components/CarouselSection", () => ({
  CarouselSection: ({
    files,
    totalCount,
    extraMenuItems,
  }: {
    files: { id: string }[];
    totalCount?: number;
    extraMenuItems?: (file: { id: string }) => Row[];
  }) => (
    <div data-testid="carousel">
      <span data-testid="total">{totalCount ?? ""}</span>
      {files.map((f) => (
        <div key={f.id} data-testid="card" data-id={f.id}>
          {(extraMenuItems?.(f) ?? []).map((row) => (
            <button key={row.id} disabled={row.disabled} onClick={() => row.onSelect()}>
              {`${row.label} ${f.id}`}
            </button>
          ))}
        </div>
      ))}
    </div>
  ),
}));

import { ToastProvider } from "@/components/ToastProvider";
import PickupWidget from "./PickupWidget";
import { fetchPickup } from "./api";
import { batchGetFiles } from "@/lib/api";

const mockFetchPickup = vi.mocked(fetchPickup);
const mockBatch = vi.mocked(batchGetFiles);

const ACTION = "Not interested";
const MESSAGE = "Removed from your picks";
const UNDO = "Undo";
const FAILED = "Couldn't save the change";
const ROUTE = "/api/addons/intelligence/pickup/not-interested/";

type Answer = { status: number } | Promise<{ status: number }>;
let answers: Answer[] = [];

function stubWrites() {
  const fn = vi.fn(async (_url: unknown, _init?: RequestInit) => {
    const next = await (answers.shift() ?? { status: 204 });
    return new Response(null, { status: next.status });
  });
  vi.stubGlobal("fetch", fn);
  return fn;
}

function writes(fn: ReturnType<typeof stubWrites>) {
  return fn.mock.calls
    .map(([url, init]) => ({ path: new URL(String(url), "http://host").pathname, method: init?.method ?? "GET" }))
    .filter((c) => c.path.startsWith(ROUTE))
    .map((c) => `${c.method} ${c.path.slice(ROUTE.length)}`);
}

const cards = () => screen.queryAllByTestId("card").map((c) => c.dataset.id);
const total = () => screen.queryByTestId("total")?.textContent;

function widget(drive = "videos") {
  return (
    <ToastProvider>
      <PickupWidget drive={drive} />
    </ToastProvider>
  );
}

async function show(ids: string[], feedTotal: number, drive = "videos") {
  mockFetchPickup.mockResolvedValue({ file_ids: ids, total: feedTotal });
  const view = render(widget(drive));
  await waitFor(() => expect(cards()).toEqual(ids));
  return view;
}

async function choose(id: string) {
  await act(async () => {
    fireEvent.click(screen.getByRole("button", { name: `${ACTION} ${id}` }));
  });
}

async function pressUndo(index = 0) {
  await act(async () => {
    fireEvent.click(screen.getAllByRole("button", { name: UNDO })[index]);
  });
}

let fetchMock: ReturnType<typeof stubWrites>;

beforeEach(() => {
  vi.clearAllMocks();
  profile.nickname = "alice";
  answers = [];
  fetchMock = stubWrites();
  mockBatch.mockImplementation(async (ids: string[]) => ids.map((id) => ({ id, filename: `${id}.mp4` })) as never);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("SPEC-ADDON-154 rule 9 Not interested on the home row", () => {
  it("SPEC-ADDON-154 rule 9 11 every card offers Not interested with a profile, and none without", async () => {
    const view = await show(["a", "b"], 30);
    expect(screen.getByRole("button", { name: `${ACTION} a` })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: `${ACTION} b` })).toBeInTheDocument();
    view.unmount();

    profile.nickname = null;
    await show(["a", "b"], 30);
    expect(screen.queryByRole("button", { name: new RegExp(ACTION) })).toBeNull();
  });

  it("SPEC-ADDON-154 rule 9 D3 D4 the row is disabled while PUT is pending; on 2xx the card leaves, the count drops and the toast offers Undo", async () => {
    let release: (v: { status: number }) => void = () => {};
    answers = [new Promise((r) => { release = r; })];
    await show(["a", "b", "c"], 30);

    await choose("b");
    expect(writes(fetchMock)).toEqual(["PUT b"]);
    expect(cards()).toEqual(["a", "b", "c"]);
    expect(screen.getByRole("button", { name: `${ACTION} b` })).toBeDisabled();

    await act(async () => release({ status: 204 }));
    await waitFor(() => expect(cards()).toEqual(["a", "c"]));
    expect(total()).toBe("29");
    expect(screen.getByText(MESSAGE).closest('[role="status"]')).not.toBeNull();
    expect(screen.getByRole("button", { name: UNDO })).toBeInTheDocument();
  });

  it("SPEC-ADDON-154 rule 9 Undo sends DELETE and returns the card at its former index with the count", async () => {
    await show(["a", "b", "c"], 30);
    await choose("b");
    await waitFor(() => expect(cards()).toEqual(["a", "c"]));

    await pressUndo();

    await waitFor(() => expect(cards()).toEqual(["a", "b", "c"]));
    expect(writes(fetchMock)).toEqual(["PUT b", "DELETE b"]);
    expect(total()).toBe("30");
  });

  it("SPEC-ADDON-154 rule 9 each undone card returns at the index it had when it left", async () => {
    await show(["a", "b", "c", "d"], 30);
    await choose("b");
    await waitFor(() => expect(cards()).toEqual(["a", "c", "d"]));
    await choose("c");
    await waitFor(() => expect(cards()).toEqual(["a", "d"]));

    const [first, second] = screen.getAllByRole("button", { name: UNDO });
    expect(second).toBeDefined();
    await act(async () => {
      fireEvent.click(first);
    });
    await waitFor(() => expect(cards()).toHaveLength(3));
    const firstBack = cards().find((id) => id === "b" || id === "c");
    expect(cards()).toEqual(["a", firstBack, "d"]);
    await act(async () => {
      fireEvent.click(second);
    });
    await waitFor(() => expect(cards()).toHaveLength(4));
    expect(cards()).toEqual(firstBack === "b" ? ["a", "c", "b", "d"] : ["a", "b", "c", "d"]);
  });

  it("SPEC-ADDON-154 rule 9 when the last card leaves the row disappears, and Undo draws it again", async () => {
    const { container } = await show(["a"], 1);

    await choose("a");
    await waitFor(() => expect(screen.queryByTestId("carousel")).toBeNull());

    await pressUndo();
    await waitFor(() => expect(cards()).toEqual(["a"]));
    expect(total()).toBe("1");
    expect(container).not.toBeEmptyDOMElement();
  });

  it.each([
    ["a non-2xx answer", { status: 500 }],
    ["400 without a viewer", { status: 400 }],
  ])("SPEC-ADDON-154 failure cases on %s the card stays and the error toast appears", async (_name, answer) => {
    answers = [answer];
    await show(["a", "b"], 30);

    await choose("a");

    await waitFor(() => expect(screen.getByText(FAILED).closest('[role="alert"]')).not.toBeNull());
    expect(cards()).toEqual(["a", "b"]);
    expect(total()).toBe("30");
  });

  it("SPEC-ADDON-154 failure cases a failed DELETE keeps the card removed and shows the error toast", async () => {
    answers = [{ status: 204 }, { status: 500 }];
    await show(["a", "b"], 30);
    await choose("a");
    await waitFor(() => expect(cards()).toEqual(["b"]));

    await pressUndo();

    await waitFor(() => expect(screen.getByText(FAILED).closest('[role="alert"]')).not.toBeNull());
    expect(cards()).toEqual(["b"]);
    expect(total()).toBe("29");
  });

  it("SPEC-ADDON-154 rule 9 after the list was replaced, Undo only sends DELETE", async () => {
    const view = await show(["a", "b"], 30, "videos");
    await choose("a");
    await waitFor(() => expect(cards()).toEqual(["b"]));

    mockFetchPickup.mockResolvedValue({ file_ids: ["p1", "p2"], total: 2 });
    view.rerender(widget("photos"));
    await waitFor(() => expect(cards()).toEqual(["p1", "p2"]));

    await pressUndo();

    await waitFor(() => expect(writes(fetchMock)).toEqual(["PUT a", "DELETE a"]));
    expect(cards()).toEqual(["p1", "p2"]);
    expect(total()).toBe("2");
  });

  it("SPEC-ADDON-154 rule 9 a write that settles after the list was replaced still shows its toast", async () => {
    let release: (v: { status: number }) => void = () => {};
    answers = [new Promise((r) => { release = r; })];
    const view = await show(["a", "b"], 30, "videos");
    await choose("a");

    mockFetchPickup.mockResolvedValue({ file_ids: ["p1"], total: 1 });
    view.rerender(widget("photos"));
    await waitFor(() => expect(cards()).toEqual(["p1"]));
    await act(async () => release({ status: 204 }));

    await waitFor(() => expect(screen.getByText(MESSAGE)).toBeInTheDocument());
    expect(cards()).toEqual(["p1"]);
  });

  it("SPEC-ADDON-154 rule 9 after a profile change Undo does nothing", async () => {
    const view = await show(["a", "b"], 30);
    await choose("a");
    await waitFor(() => expect(cards()).toEqual(["b"]));

    profile.nickname = "bob";
    view.rerender(widget());
    await pressUndo();
    await act(async () => {
      await new Promise((r) => setTimeout(r, 0));
    });

    expect(writes(fetchMock)).toEqual(["PUT a"]);
  });
});
