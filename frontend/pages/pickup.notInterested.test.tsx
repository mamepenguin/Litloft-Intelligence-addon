/**
 * SPEC-ADDON-154 rule 9 and I5: Not interested on the Pickup page, against a feed that
 * filters marks the way the endpoint does, so the paging arithmetic is judged by what
 * the reader ends up seeing.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";

vi.mock("@/addons/intelligence/api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/addons/intelligence/api")>()),
  fetchPickup: vi.fn(),
}));

vi.mock("@/lib/api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/api")>()),
  batchGetFiles: vi.fn(),
}));

const currentDrive = { value: "videos" };
vi.mock("@/components/CurrentDriveProvider", () => ({
  useCurrentDrive: () => currentDrive.value,
}));

const profile = { nickname: "alice" as string | null };
vi.mock("@/components/ProfileProvider", () => ({
  useProfile: () => ({ nickname: profile.nickname, setNickname: vi.fn(), clearNickname: vi.fn() }),
}));

type Row = { id: string; label: string; onSelect: () => unknown; disabled?: boolean };

vi.mock("@/components/FileGrid", () => ({
  FileGrid: ({
    files,
    extraMenuItems,
  }: {
    files: { id: string }[];
    extraMenuItems?: (file: { id: string }) => Row[];
  }) => (
    <div data-testid="grid">
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

class FakeObserver {
  static instances: FakeObserver[] = [];
  private watching = 0;
  constructor(private callback: IntersectionObserverCallback) {
    FakeObserver.instances.push(this);
  }
  observe() {
    this.watching += 1;
  }
  disconnect() {
    this.watching = 0;
  }
  unobserve() {
    this.watching = Math.max(0, this.watching - 1);
  }
  takeRecords() {
    return [];
  }
  trigger() {
    if (this.watching === 0) return;
    this.callback([{ isIntersecting: true } as IntersectionObserverEntry], this as never);
  }
}

import { ToastProvider } from "@/components/ToastProvider";
import PickupPage from "./pickup";
import { fetchPickup } from "../api";
import { batchGetFiles } from "@/lib/api";

const mockFetchPickup = vi.mocked(fetchPickup);
const mockBatch = vi.mocked(batchGetFiles);

const ACTION = "Not interested";
const MESSAGE = "Removed from your picks";
const UNDO = "Undo";
const FAILED = "Couldn't save the change";
const ROUTE = "/api/addons/intelligence/pickup/not-interested/";

/** The stored feed and the viewer's marks, served as the endpoint serves them. */
const server = {
  feed: [] as string[],
  marked: new Set<string>(),
  failNext: false,
  hold: null as Promise<void> | null,
  filtered() {
    return this.feed.filter((id) => !this.marked.has(id));
  },
};

function stubWrites() {
  const fn = vi.fn(async (url: unknown, init?: RequestInit) => {
    const path = new URL(String(url), "http://host").pathname;
    if (!path.startsWith(ROUTE)) return new Response(null, { status: 404 });
    if (server.failNext) {
      server.failNext = false;
      return new Response("boom", { status: 500 });
    }
    if (server.hold) await server.hold;
    const id = path.slice(ROUTE.length);
    if (init?.method === "PUT") server.marked.add(id);
    if (init?.method === "DELETE") server.marked.delete(id);
    return new Response(null, { status: 204 });
  });
  vi.stubGlobal("fetch", fn);
  return fn;
}

const ids = (n: number) => Array.from({ length: n }, (_, i) => `f${String(i).padStart(3, "0")}`);
const cards = () => screen.queryAllByTestId("card").map((c) => c.dataset.id!);

async function settle(rounds = 5) {
  for (let i = 0; i < rounds; i++) {
    await act(async () => {
      await new Promise((r) => setTimeout(r, 0));
    });
  }
}

async function nearEnd() {
  await act(async () => {
    for (const o of FakeObserver.instances) o.trigger();
  });
  await settle();
}

const page = () => (
  <ToastProvider>
    <PickupPage />
  </ToastProvider>
);

async function open(feed: string[]) {
  server.feed = feed;
  const view = render(page());
  await waitFor(() => expect(cards().length).toBeGreaterThan(0));
  await settle();
  return view;
}

async function choose(id: string) {
  await act(async () => {
    fireEvent.click(screen.getByRole("button", { name: `${ACTION} ${id}` }));
  });
  await settle();
}

async function pressUndo(index = 0) {
  await act(async () => {
    fireEvent.click(screen.getAllByRole("button", { name: UNDO })[index]);
  });
  await settle();
}

/** I5: what is drawn is always the head of the filtered feed, no repeats and no gaps. */
function expectHeadOfFeed() {
  const drawn = cards();
  expect(new Set(drawn).size).toBe(drawn.length);
  expect(drawn).toEqual(server.filtered().slice(0, drawn.length));
}

beforeEach(() => {
  vi.clearAllMocks();
  currentDrive.value = "videos";
  profile.nickname = "alice";
  server.feed = [];
  server.marked = new Set();
  server.failNext = false;
  server.hold = null;
  FakeObserver.instances = [];
  vi.stubGlobal("IntersectionObserver", FakeObserver);
  stubWrites();
  mockFetchPickup.mockImplementation(async (_drive: string, opts: { limit?: number; offset?: number } = {}) => {
    const list = server.filtered();
    const offset = opts.offset ?? 0;
    const limit = opts.limit ?? 40;
    return { file_ids: list.slice(offset, offset + limit), total: list.length };
  });
  mockBatch.mockImplementation(async (list: string[]) => list.map((id) => ({ id, filename: `${id}.mp4` })) as never);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("SPEC-ADDON-154 rule 9 Not interested on the Pickup page", () => {
  it("SPEC-ADDON-154 rule 9 11 every card offers Not interested with a profile, and none without", async () => {
    await open(ids(5));
    expect(screen.getByRole("button", { name: `${ACTION} f000` })).toBeInTheDocument();
    cleanup();

    profile.nickname = null;
    await open(ids(5));
    expect(screen.queryByRole("button", { name: new RegExp(ACTION) })).toBeNull();
  });

  it("SPEC-ADDON-154 rule 9 I5 after marks the next page neither repeats nor skips a file", async () => {
    await open(ids(300));
    const first = cards();
    expect(first.length).toBeLessThan(300);

    await choose(first[1]);
    await choose(first[3]);
    expect(screen.getAllByText(MESSAGE).length).toBe(2);
    expectHeadOfFeed();

    await nearEnd();
    expect(cards().length).toBeGreaterThan(first.length - 2);
    expectHeadOfFeed();
  });

  it("SPEC-ADDON-154 rule 9 I5 after a mark and its undo the next page neither repeats nor skips a file", async () => {
    await open(ids(300));
    const first = cards();

    await choose(first[2]);
    expect(cards()).toEqual(first.filter((id) => id !== first[2]));
    await pressUndo();
    expect(cards()).toEqual(first);

    await nearEnd();
    expect(cards().length).toBeGreaterThan(first.length);
    expectHeadOfFeed();
  });

  it("SPEC-ADDON-154 rule 9 when the last loaded card leaves and more remain, the next page loads", async () => {
    await open(ids(300));
    const first = cards();

    for (const id of first) await choose(id);

    await waitFor(() => expect(cards().length).toBeGreaterThan(0));
    expect(cards()[0]).toBe(server.filtered()[0]);
    expectHeadOfFeed();
  });

  it("SPEC-ADDON-154 rule 9 when the last card of the whole feed leaves, the page shows its empty text and asks for nothing", async () => {
    await open(ids(2));
    const calls = mockFetchPickup.mock.calls.length;

    await choose("f000");
    await choose("f001");

    expect(cards()).toEqual([]);
    expect(screen.getByText("Nothing to suggest yet.")).toBeInTheDocument();
    expect(mockFetchPickup.mock.calls.length).toBe(calls);

    await pressUndo(0);
    expect(cards().length).toBe(1);
  });

  it("SPEC-ADDON-154 failure cases a failed PUT keeps the card and shows the error toast", async () => {
    await open(ids(5));
    server.failNext = true;

    await choose("f001");

    expect(cards()).toEqual(ids(5));
    expect(screen.getByText(FAILED).closest('[role="alert"]')).not.toBeNull();
    expect(server.marked.size).toBe(0);
  });

  it("SPEC-ADDON-154 rule 9 D4 while PUT is pending the card stays and its row is disabled", async () => {
    let release: () => void = () => {};
    server.hold = new Promise((r) => { release = r; });
    await open(ids(5));

    await choose("f001");
    expect(cards()).toEqual(ids(5));
    expect(screen.getByRole("button", { name: `${ACTION} f001` })).toBeDisabled();

    server.hold = null;
    await act(async () => release());
    await settle();
    expect(cards()).toEqual(ids(5).filter((id) => id !== "f001"));
  });

  it("SPEC-ADDON-154 failure cases a failed DELETE keeps the card removed and shows the error toast", async () => {
    await open(ids(5));
    await choose("f001");
    server.failNext = true;

    await pressUndo();

    expect(cards()).toEqual(ids(5).filter((id) => id !== "f001"));
    expect(screen.getByText(FAILED).closest('[role="alert"]')).not.toBeNull();
    expect(server.marked.has("f001")).toBe(true);
  });

  it("SPEC-ADDON-154 rule 9 after the drive changed, Undo only sends DELETE", async () => {
    const view = await open(ids(5));
    await choose("f001");

    currentDrive.value = "photos";
    server.feed = ["p000", "p001"];
    view.rerender(page());
    await waitFor(() => expect(cards()).toEqual(["p000", "p001"]));
    await pressUndo();

    expect(server.marked.has("f001")).toBe(false);
    expect(cards()).toEqual(["p000", "p001"]);
  });

  it("SPEC-ADDON-154 rule 9 after a profile change Undo does nothing", async () => {
    const view = await open(ids(5));
    await choose("f001");

    profile.nickname = "bob";
    view.rerender(page());
    await settle();
    await pressUndo();

    expect(server.marked.has("f001")).toBe(true);
  });
});
