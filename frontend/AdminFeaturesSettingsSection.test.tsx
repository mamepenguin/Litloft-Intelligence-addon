import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor, within } from "@testing-library/react";

import AdminFeaturesSettingsSection from "./AdminFeaturesSettingsSection";

const ENDPOINT = "/api/addons/intelligence/admin/features";
const LLM = "/api/addons/intelligence/admin/llm";
const EXPOSURE = "/api/addons/intelligence/admin/llm/exposure";

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

const LLM_FEATURES = [
  "rag",
  "summaries",
  "detailed_summaries",
  "auto_tags",
  "transcript_refine",
  "retrieval_keywords",
  "chapter_suggestions",
  "vision_describe",
  "video_visual_index",
];

const LOCAL = { provider: "ollama", model: "qwen3:14b", offhost: false, api_key_present: false };
const CLAUDE = {
  provider: "openai_compatible",
  model: "claude",
  offhost: true,
  api_key_env: "LLM_API_KEY_CLAUDE",
  api_key_present: true,
};

function llmView(profiles: Record<string, unknown>, routing: Record<string, unknown>) {
  return {
    profiles,
    routing,
    legacy: false,
    error: null,
    output_language: "ja",
    output_language_restart_pending: false,
    features: LLM_FEATURES,
    available_providers: ["disabled", "ollama", "openai_compatible"],
    available_output_languages: ["auto", "ja", "en"],
    overrides_present: true,
  };
}

const TWO = llmView(
  { local: LOCAL, claude: CLAUDE },
  { default: "local", local_fallback: "local", features: { rag: "claude" } },
);
const ONE = llmView({ default: LOCAL }, { default: "default" });
const NO_EXPOSURE = { body: { features: {}, local_fallback: null } };

async function renderWith(routes: Routes) {
  serve({
    [`GET ${ENDPOINT}`]: { body: defaultPayload() },
    [`GET ${LLM}`]: { body: ONE },
    [`GET ${EXPOSURE}`]: NO_EXPOSURE,
    ...routes,
  });
  render(<AdminFeaturesSettingsSection />);
  await screen.findByRole("button", { name: "Save" });
}

describe("AdminFeaturesSettingsSection", () => {
  it("one profile: a mode select per feature and no profile column", async () => {
    await renderWith({});
    expect(screen.getAllByRole("combobox", { name: /^When .* runs$/ })).toHaveLength(11);
    expect(screen.queryByRole("columnheader", { name: /^Profile/ })).toBeNull();
    expect(screen.queryAllByRole("combobox", { name: /^Profile for/ })).toHaveLength(0);
  });

  it("two profiles: a profile select on every LLM feature row", async () => {
    await renderWith({ [`GET ${LLM}`]: { body: TWO } });
    expect(screen.getByRole("columnheader", { name: /^Profile/ })).toBeInTheDocument();
    expect(screen.getAllByRole("combobox", { name: /^Profile for/ })).toHaveLength(9);
    const rag = screen.getByRole("combobox", {
      name: "Profile for AI question answering (Ask)",
    }) as HTMLSelectElement;
    expect(rag.value).toBe("claude");
    expect(within(rag).getAllByRole("option").map((o) => o.textContent)).toEqual([
      "Default (local)",
      "local",
      "claude (off-host)",
    ]);
  });

  it("lists each drive's destination under an off-host row", async () => {
    await renderWith({
      [`GET ${LLM}`]: { body: TWO },
      [`GET ${EXPOSURE}`]: {
        body: {
          features: {
            rag: {
              profile: "claude",
              offhost: true,
              drives: { media: "sends", misc: "sends", private: "falls_back", old: "skips", nas: "unknown" },
            },
            summaries: { profile: "local", offhost: false },
          },
          local_fallback: "local",
        },
      },
    });
    const destinations = screen.getAllByTestId("feature-destinations");
    expect(destinations).toHaveLength(1);
    expect([...destinations[0].children].map((s) => s.textContent)).toEqual([
      "Sends media, misc",
      "Runs on local private",
      "Does not run old",
      "Unknown nas",
    ]);
  });

  it("a profile change alone saves the routing and needs no restart", async () => {
    await renderWith({
      [`GET ${LLM}`]: { body: TWO },
      [`PUT ${LLM}`]: { body: { status: "saved", restart_required: false } },
    });
    fireEvent.change(screen.getByRole("combobox", { name: "Profile for AI summary (short)" }), {
      target: { value: "claude" },
    });
    fireEvent.change(
      screen.getByRole("combobox", { name: "Profile for AI question answering (Ask)" }),
      { target: { value: "" } },
    );
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    expect(await screen.findByText("Saved. Profile choices are applied now.")).toBeInTheDocument();
    expect(callsTo("PUT", ENDPOINT)).toHaveLength(0);
    expect(bodyOf("PUT", LLM)).toEqual({
      profiles: {
        local: { provider: "ollama", model: "qwen3:14b", offhost: false },
        claude: {
          provider: "openai_compatible",
          model: "claude",
          offhost: true,
          api_key_env: "LLM_API_KEY_CLAUDE",
        },
      },
      routing: { default: "local", local_fallback: "local", features: { summaries: "claude" } },
      output_language: "ja",
    });
  });

  it("a mode change writes the modes and says a restart is needed", async () => {
    await renderWith({
      [`GET ${LLM}`]: { body: TWO },
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
      await screen.findByText('"When it runs" changed, so a restart is required after saving.'),
    ).toBeInTheDocument();
    expect(callsTo("PUT", LLM)).toHaveLength(0);
    const body = bodyOf("PUT", ENDPOINT);
    expect(body.indexing).toBe(false);
    expect(body.chapter_suggestions).toBe("on_index");
    expect(body.summaries).toBe("manual");
  });

  it("shows the routing save's 400 detail", async () => {
    await renderWith({
      [`GET ${LLM}`]: { body: TWO },
      [`PUT ${LLM}`]: { status: 400, body: { detail: "llm.routing.default is required" } },
    });
    fireEvent.change(screen.getByRole("combobox", { name: "Profile for AI summary (short)" }), {
      target: { value: "claude" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    expect(await screen.findByText("llm.routing.default is required")).toBeInTheDocument();
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
    serve({
      [`GET ${ENDPOINT}`]: { status: 500, body: { detail: "boom" } },
      [`GET ${LLM}`]: { body: ONE },
      [`GET ${EXPOSURE}`]: NO_EXPOSURE,
    });
    render(<AdminFeaturesSettingsSection />);
    expect(await screen.findByText("boom")).toBeInTheDocument();
  });

  const SAVED = { body: { status: "saved", restart_required: false } };
  const RAG = "Profile for AI question answering (Ask)";
  const SUMMARIES = "Profile for AI summary (short)";

  it("names each option by its saved default and off-host flag", async () => {
    await renderWith({
      [`GET ${LLM}`]: {
        body: llmView(
          { local: LOCAL, claude: CLAUDE, lan: { provider: "openai_compatible" } },
          { default: "claude", features: { rag: "ghost" } },
        ),
      },
    });
    const summaries = screen.getByRole("combobox", { name: SUMMARIES });
    expect(within(summaries).getAllByRole("option").map((o) => o.textContent)).toEqual([
      "Default (claude)",
      "local",
      "claude (off-host)",
      "lan (off-host)",
    ]);
    const rag = screen.getByRole("combobox", { name: RAG }) as HTMLSelectElement;
    expect(rag.selectedOptions[0].textContent).toBe("ghost (no such profile)");
  });

  it("changing a mode and a profile writes both and says a restart is needed", async () => {
    await renderWith({
      [`GET ${LLM}`]: { body: TWO },
      [`PUT ${LLM}`]: SAVED,
      [`PUT ${ENDPOINT}`]: { body: { status: "saved", restart_required: true } },
    });
    fireEvent.change(screen.getByRole("combobox", { name: SUMMARIES }), {
      target: { value: "claude" },
    });
    fireEvent.change(screen.getByRole("combobox", { name: "When Indexing runs" }), {
      target: { value: "false" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    expect(
      await screen.findByText('"When it runs" changed, so a restart is required after saving.'),
    ).toBeInTheDocument();
    expect(bodyOf("PUT", LLM).routing.features).toEqual({ rag: "claude", summaries: "claude" });
    expect(bodyOf("PUT", ENDPOINT).indexing).toBe(false);
  });

  it("a routing save followed by a failed mode save says what was saved", async () => {
    await renderWith({
      [`GET ${LLM}`]: { body: TWO },
      [`PUT ${LLM}`]: SAVED,
      [`PUT ${ENDPOINT}`]: { status: 500, body: { detail: "modes boom" } },
    });
    fireEvent.change(screen.getByRole("combobox", { name: SUMMARIES }), {
      target: { value: "claude" },
    });
    fireEvent.change(screen.getByRole("combobox", { name: "When Indexing runs" }), {
      target: { value: "false" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    expect(
      await screen.findByText(
        'Profile choices were saved. "When it runs" could not be saved: modes boom',
      ),
    ).toBeInTheDocument();
    await waitFor(() => expect(callsTo("GET", ENDPOINT)).toHaveLength(2));
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
      await screen.findByText('"When it runs" changed, so a restart is required after saving.'),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Reloading failed, so the screen still shows the values from before saving."),
    ).toBeInTheDocument();
  });

  it("destinations disappear once the row's choice differs from the saved routing", async () => {
    await renderWith({
      [`GET ${LLM}`]: { body: TWO },
      [`GET ${EXPOSURE}`]: {
        body: {
          features: { rag: { profile: "claude", offhost: true, drives: { media: "sends" } } },
          local_fallback: "local",
        },
      },
    });
    expect(screen.getAllByTestId("feature-destinations")).toHaveLength(1);
    fireEvent.change(screen.getByRole("combobox", { name: RAG }), { target: { value: "local" } });
    expect(screen.queryByTestId("feature-destinations")).toBeNull();
    fireEvent.change(screen.getByRole("combobox", { name: RAG }), { target: { value: "claude" } });
    expect(screen.getAllByTestId("feature-destinations")).toHaveLength(1);
  });
});
