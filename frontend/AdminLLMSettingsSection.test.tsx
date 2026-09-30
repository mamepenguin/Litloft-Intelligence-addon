import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor, within } from "@testing-library/react";

import AdminLLMSettingsSection from "./AdminLLMSettingsSection";

const ENDPOINT = "/api/addons/intelligence/admin/llm";
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

function callsTo(method: string, url = ENDPOINT) {
  return mockFetch.mock.calls.filter((c) => (c[1]?.method ?? "GET") === method && c[0] === url);
}

function putBody() {
  const calls = callsTo("PUT");
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

const LOCAL = {
  provider: "ollama",
  base_url: "http://host.docker.internal:11434",
  model: "qwen3:14b",
  vision_model: "gemma3:12b",
  offhost: false,
  agentic: false,
  api_key_env: "LLM_API_KEY",
  api_key_present: false,
};
const CLAUDE = {
  provider: "openai_compatible",
  base_url: "https://openrouter.ai/api/v1",
  model: "anthropic/claude-sonnet-5",
  vision_model: "",
  offhost: true,
  agentic: true,
  api_key_env: "LLM_API_KEY_CLAUDE",
  api_key_present: true,
};
const CLOUD2 = { ...CLAUDE, api_key_env: "LLM_API_KEY_CLOUD2" };

function view(overrides: Record<string, unknown> = {}) {
  return {
    profiles: { default: LOCAL },
    routing: { default: "default" },
    legacy: true,
    error: null,
    output_language: "auto",
    output_language_restart_pending: false,
    features: ["rag", "summaries"],
    available_providers: ["disabled", "ollama", "openai_compatible"],
    available_output_languages: ["auto", "ja", "en"],
    overrides_present: false,
    ...overrides,
  };
}

const TWO = view({
  legacy: false,
  profiles: { local: LOCAL, claude: CLAUDE },
  routing: { default: "local", local_fallback: "local" },
});

const NO_EXPOSURE = { body: { features: {}, local_fallback: null } };

async function renderWith(routes: Routes) {
  serve({ [`GET ${EXPOSURE}`]: NO_EXPOSURE, ...routes });
  render(<AdminLLMSettingsSection />);
  await screen.findByRole("button", { name: "Save" });
}

describe("AdminLLMSettingsSection", () => {
  it("one profile renders the single layout: no list, no routing", async () => {
    await renderWith({ [`GET ${ENDPOINT}`]: { body: view() } });
    expect(screen.getByRole("heading", { name: "Model connection" })).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Default routing" })).toBeNull();
    expect(screen.queryByRole("button", { name: /^Delete profile/ })).toBeNull();
    expect(screen.getByRole("group", { name: "Connection" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Ollama" })).toHaveAttribute("aria-pressed", "true");
  });

  it("two profiles render the list with per-profile actions and the routing selects", async () => {
    await renderWith({ [`GET ${ENDPOINT}`]: { body: TWO } });
    expect(screen.getByRole("heading", { name: "Profiles" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Edit profile local" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Edit profile claude" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Default routing" })).toBeInTheDocument();
  });

  it("the fallback select offers only local profiles and none", async () => {
    await renderWith({
      [`GET ${ENDPOINT}`]: {
        body: view({
          legacy: false,
          profiles: { local: LOCAL, claude: CLAUDE, cloud2: CLOUD2, small: { ...LOCAL } },
          routing: { default: "claude" },
        }),
      },
    });
    const fallback = screen.getByRole("combobox", { name: "Use instead on drives without cloud" });
    const options = within(fallback).getAllByRole("option").map((o) => o.textContent);
    expect(options).toEqual(["local", "small", "None (do not run)"]);
    const defaults = screen.getByRole("combobox", { name: "Default profile" });
    expect(within(defaults).getAllByRole("option").map((o) => o.textContent)).toEqual([
      "local",
      "claude",
      "cloud2",
      "small",
    ]);
  });

  it.each([
    ["WORK", "LLM_API_KEY_WORK"],
    ["A_B2", "LLM_API_KEY_A_B2"],
    ["", "LLM_API_KEY"],
  ])("key suffix %j is saved as %s", async (suffix, env) => {
    await renderWith({
      [`GET ${ENDPOINT}`]: { body: TWO },
      [`PUT ${ENDPOINT}`]: { body: { status: "saved", restart_required: false } },
    });
    fireEvent.click(screen.getByRole("button", { name: "Edit profile claude" }));
    fireEvent.change(screen.getByLabelText("API key environment variable"), {
      target: { value: suffix },
    });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() => expect(callsTo("PUT")).toHaveLength(1));
    expect(putBody().profiles.claude.api_key_env).toBe(env);
  });

  it.each([
    ["key suffix", "API key environment variable", "work", /API key variable of profile claude/],
    ["name", "Name", "Claude", /Profile name “Claude”/],
    ["duplicate name", "Name", "local", /Profile name “local” is used twice/],
  ])("an invalid %s disables save and says why", async (_what, field, value, reason) => {
    await renderWith({ [`GET ${ENDPOINT}`]: { body: TWO } });
    fireEvent.click(screen.getByRole("button", { name: "Edit profile claude" }));
    fireEvent.change(screen.getByLabelText(field), { target: { value } });
    const save = screen.getByRole("button", { name: "Save" });
    expect(save).toBeDisabled();
    expect(screen.getByText(reason)).toBeInTheDocument();
  });

  it("saving the legacy view writes it as the profile named default", async () => {
    await renderWith({
      [`GET ${ENDPOINT}`]: { body: view() },
      [`PUT ${ENDPOINT}`]: { body: { status: "saved", restart_required: false } },
    });
    fireEvent.change(screen.getByLabelText("Model"), { target: { value: "gemma4:e4b" } });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() => expect(callsTo("PUT")).toHaveLength(1));
    expect(putBody()).toEqual({
      profiles: {
        default: {
          provider: "ollama",
          base_url: "http://host.docker.internal:11434",
          model: "gemma4:e4b",
          vision_model: "gemma3:12b",
          offhost: false,
          agentic: false,
          api_key_env: "LLM_API_KEY",
        },
      },
      routing: { default: "default" },
      output_language: "auto",
    });
    expect(await screen.findByText("Saved. Applied now.")).toBeInTheDocument();
  });

  it("renaming a profile carries its routing references with it", async () => {
    await renderWith({
      [`GET ${ENDPOINT}`]: {
        body: {
          ...TWO,
          routing: { default: "local", local_fallback: "local", features: { rag: "claude" } },
        },
      },
      [`PUT ${ENDPOINT}`]: { body: { status: "saved", restart_required: false } },
    });
    fireEvent.click(screen.getByRole("button", { name: "Edit profile local" }));
    fireEvent.change(screen.getByLabelText("Name"), { target: { value: "home" } });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() => expect(callsTo("PUT")).toHaveLength(1));
    const body = putBody();
    expect(Object.keys(body.profiles)).toEqual(["home", "claude"]);
    expect(body.routing).toEqual({
      default: "home",
      local_fallback: "home",
      features: { rag: "claude" },
    });
  });

  it("deleting the default profile leaves the single layout and routes to the survivor", async () => {
    await renderWith({
      [`GET ${ENDPOINT}`]: {
        body: { ...TWO, routing: { default: "claude", features: { rag: "claude" } } },
      },
      [`PUT ${ENDPOINT}`]: { body: { status: "saved", restart_required: false } },
    });
    fireEvent.click(screen.getByRole("button", { name: "Edit profile claude" }));
    fireEvent.click(screen.getByRole("button", { name: "Delete profile claude" }));
    expect(screen.getByRole("heading", { name: "Model connection" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() => expect(callsTo("PUT")).toHaveLength(1));
    const body = putBody();
    expect(Object.keys(body.profiles)).toEqual(["local"]);
    expect(body.routing).toEqual({ default: "local" });
  });

  it("turning a fallback profile off-host clears the fallback", async () => {
    await renderWith({
      [`GET ${ENDPOINT}`]: { body: TWO },
      [`PUT ${ENDPOINT}`]: { body: { status: "saved", restart_required: false } },
    });
    fireEvent.click(screen.getByRole("button", { name: "Edit profile local" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "Send off-host" }));
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() => expect(callsTo("PUT")).toHaveLength(1));
    expect(putBody().routing).toEqual({ default: "local" });
  });

  it("shows the server's 400 detail in the save error", async () => {
    await renderWith({
      [`GET ${ENDPOINT}`]: { body: TWO },
      [`PUT ${ENDPOINT}`]: {
        status: 400,
        body: { detail: "llm.routing.local_fallback 'x' must be offhost: false" },
      },
    });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    const alert = await screen.findByTestId("llm-save-error");
    expect(alert).toHaveTextContent("Could not save");
    expect(alert).toHaveTextContent("llm.routing.local_fallback 'x' must be offhost: false");
  });

  it("reports that the output language still needs a restart", async () => {
    await renderWith({
      [`GET ${ENDPOINT}`]: { body: view() },
      [`PUT ${ENDPOINT}`]: { body: { status: "saved", restart_required: true } },
    });
    fireEvent.change(screen.getByRole("combobox", { name: "Output language" }), {
      target: { value: "ja" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    expect(
      await screen.findByText("Saved. The output language takes effect after a restart."),
    ).toBeInTheDocument();
    expect(putBody().output_language).toBe("ja");
  });

  it("shows the routing error from GET", async () => {
    await renderWith({
      [`GET ${ENDPOINT}`]: {
        body: view({ error: "llm.routing.features.rag: unknown profile 'cloud-strong'" }),
      },
    });
    const alert = screen.getByTestId("llm-routing-error");
    expect(alert).toHaveTextContent("The current settings stop every AI feature");
    expect(alert).toHaveTextContent("llm.routing.features.rag: unknown profile 'cloud-strong'");
  });

  const skipsOn = (destination: string) => ({
    body: {
      features: {
        rag: { profile: "default", offhost: true, drives: { private: destination, shared: "sends" } },
        summaries: { profile: "default", offhost: true, drives: { private: destination, shared: "sends" } },
      },
      local_fallback: null,
    },
  });

  it.each([
    ["the only profile is off-host and a drive skips", { default: CLAUDE }, "skips", true],
    ["the only profile is local", { default: LOCAL }, "skips", false],
    ["no drive skips", { default: CLAUDE }, "sends", false],
    ["there are two profiles", { a: CLAUDE, b: CLOUD2 }, "skips", false],
  ])("the off-host-only warning: %s", async (_case, profiles, destination, shown) => {
    await renderWith({
      [`GET ${ENDPOINT}`]: { body: view({ profiles, routing: { default: Object.keys(profiles)[0] } }) },
      [`GET ${EXPOSURE}`]: skipsOn(destination),
    });
    if (shown) {
      const warning = await screen.findByTestId("llm-offhost-only-warning");
      expect(warning).toHaveTextContent("AI features do not run on the drive “private”");
    } else {
      expect(screen.queryByTestId("llm-offhost-only-warning")).toBeNull();
    }
  });

  it("reverts to YAML with DELETE when overrides exist", async () => {
    await renderWith({
      [`GET ${ENDPOINT}`]: [
        { body: view({ overrides_present: true }) },
        { body: view({ overrides_present: false }) },
      ],
      [`DELETE ${ENDPOINT}`]: { body: { status: "reset", removed: true, restart_required: false } },
    });
    fireEvent.click(screen.getByRole("button", { name: "Revert to YAML settings" }));
    await waitFor(() => expect(callsTo("DELETE")).toHaveLength(1));
    await waitFor(() =>
      expect(screen.queryByRole("button", { name: "Revert to YAML settings" })).toBeNull(),
    );
  });
});
