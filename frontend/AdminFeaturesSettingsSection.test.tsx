import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";

import AdminFeaturesSettingsSection from "./AdminFeaturesSettingsSection";

const ENDPOINT = "/api/addons/intelligence/admin/features";
const LLM = "/api/addons/intelligence/admin/llm";

type Reply = { status?: number; body: unknown };
type Routes = Record<string, Reply | Reply[]>;

const mockFetch = vi.fn();

function jsonResponse({ status = 200, body }: Reply) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

/** Routes by "METHOD url"; an array is consumed one reply per call, the last one repeating. */
function serve(routes: Routes) {
  const queues = Object.fromEntries(
    Object.entries(routes).map(([k, v]) => [k, Array.isArray(v) ? [...v] : [v]]),
  );
  mockFetch.mockImplementation(async (url: string, init?: RequestInit) => {
    const key = `${init?.method ?? "GET"} ${url}`;
    const queue = queues[key];
    if (!queue) throw new Error(`unexpected ${key}`);
    return jsonResponse(queue.length > 1 ? queue.shift()! : queue[0]);
  });
}

function callsTo(method: string, url: string) {
  return mockFetch.mock.calls.filter((c) => (c[1]?.method ?? "GET") === method && c[0] === url);
}

function bodyOf(method: string, url: string) {
  const calls = callsTo(method, url);
  return JSON.parse(String(calls[calls.length - 1][1].body));
}

beforeEach(() => {
  vi.stubGlobal("fetch", mockFetch);
});

afterEach(() => {
  mockFetch.mockReset();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

function defaultPayload(overrides: Record<string, unknown> = {}) {
  return {
    indexing: true,
    search: true,
    rag: false,
    auto_tags: "manual",
    summaries: "manual",
    detailed_summaries: "false",
    transcript_refine: "false",
    vision_describe: "manual",
    retrieval_keywords: "false",
    chapter_suggestions: "manual",
    video_visual_index: "false",
    tristate_values: ["false", "manual", "on_index"],
    overrides_present: false,
    ...overrides,
  };
}

async function renderWith(routes: Routes) {
  serve({ [`GET ${ENDPOINT}`]: { body: defaultPayload() }, ...routes });
  render(<AdminFeaturesSettingsSection />);
  await screen.findByRole("button", { name: "Save" });
}

describe("AdminFeaturesSettingsSection", () => {
  it("a mode select per feature, no profile column, and a pointer to LLM profiles", async () => {
    await renderWith({});
    expect(screen.getAllByRole("combobox", { name: /^When .* runs$/ })).toHaveLength(11);
    expect(screen.getAllByRole("columnheader").map((h) => h.textContent)).toEqual([
      "Feature",
      "When it runsRestart required",
    ]);
    expect(
      screen.getByText("Which model a feature runs on is chosen under LLM profiles › Routing."),
    ).toBeInTheDocument();
  });

  it("a save writes the modes, never /admin/llm, and says a restart is needed", async () => {
    await renderWith({
      [`PUT ${ENDPOINT}`]: { body: { status: "saved", restart_required: true } },
    });
    fireEvent.change(screen.getByRole("combobox", { name: "When Indexing runs" }), {
      target: { value: "false" },
    });
    fireEvent.change(screen.getByRole("combobox", { name: "When AI chapter candidates runs" }), {
      target: { value: "on_index" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    expect(
      await screen.findByText("Saved. Restart required for the change to take effect"),
    ).toBeInTheDocument();
    await waitFor(() => expect(callsTo("GET", ENDPOINT)).toHaveLength(2));
    expect(bodyOf("PUT", ENDPOINT)).toEqual({
      indexing: false,
      search: true,
      rag: false,
      auto_tags: "manual",
      summaries: "manual",
      detailed_summaries: "false",
      transcript_refine: "false",
      vision_describe: "manual",
      retrieval_keywords: "false",
      chapter_suggestions: "on_index",
      video_visual_index: "false",
    });
    expect(mockFetch.mock.calls.filter((c) => String(c[0]).startsWith(LLM))).toEqual([]);
  });

  it("shows the save's 400 detail", async () => {
    await renderWith({
      [`PUT ${ENDPOINT}`]: { status: 400, body: { detail: "invalid mode" } },
    });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    expect(await screen.findByText("invalid mode")).toBeInTheDocument();
  });

  it("hides the overrides banner when no override is active", async () => {
    await renderWith({});
    expect(screen.queryByTestId("features-overrides-banner")).toBeNull();
  });

  it("DELETEs on reset and refetches", async () => {
    await renderWith({
      [`GET ${ENDPOINT}`]: [
        { body: defaultPayload({ overrides_present: true }) },
        { body: defaultPayload({ overrides_present: false }) },
      ],
      [`DELETE ${ENDPOINT}`]: { body: { status: "reset", removed: true, restart_required: true } },
    });
    expect(screen.getByTestId("features-overrides-banner")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /reset/i }));
    await waitFor(() => expect(callsTo("DELETE", ENDPOINT)).toHaveLength(1));
    await waitFor(() => expect(screen.queryByTestId("features-overrides-banner")).toBeNull());
  });

  it("renders load error inline when GET fails", async () => {
    serve({ [`GET ${ENDPOINT}`]: { status: 500, body: { detail: "boom" } } });
    render(<AdminFeaturesSettingsSection />);
    expect(await screen.findByText("boom")).toBeInTheDocument();
  });

  it("a save whose reload fails still reports the save", async () => {
    await renderWith({
      [`GET ${ENDPOINT}`]: [
        { body: defaultPayload() },
        { status: 500, body: { detail: "reload boom" } },
      ],
      [`PUT ${ENDPOINT}`]: { body: { status: "saved", restart_required: true } },
    });
    fireEvent.change(screen.getByRole("combobox", { name: "When Indexing runs" }), {
      target: { value: "false" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    expect(
      await screen.findByText("Saved. Restart required for the change to take effect"),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Reloading failed, so the screen still shows the values from before saving."),
    ).toBeInTheDocument();
  });
});
