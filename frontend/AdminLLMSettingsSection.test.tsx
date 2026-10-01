import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor, within } from "@testing-library/react";

import AdminLLMSettingsSection from "./AdminLLMSettingsSection";

const ENDPOINT = "/api/addons/intelligence/admin/llm";
const EXPOSURE = "/api/addons/intelligence/admin/llm/exposure";
const FEATURES = "/api/addons/intelligence/admin/features";

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
const MODES = { body: { indexing: true, search: true, rag: true, summaries: "manual" } };

async function renderWith(routes: Routes) {
  serve({ [`GET ${EXPOSURE}`]: NO_EXPOSURE, [`GET ${FEATURES}`]: MODES, ...routes });
  render(<AdminLLMSettingsSection />);
  await screen.findByRole("button", { name: "Save" });
}

describe("AdminLLMSettingsSection", () => {
  it("one profile renders the single layout: no list, no routing", async () => {
    await renderWith({ [`GET ${ENDPOINT}`]: { body: view() } });
    expect(screen.getByRole("heading", { name: "Model connection" })).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Routing" })).toBeNull();
    expect(screen.queryByRole("button", { name: /^Delete profile/ })).toBeNull();
    expect(screen.getByRole("group", { name: "Connection" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Ollama" })).toHaveAttribute("aria-pressed", "true");
  });

  it("two profiles render the list with per-profile actions and the routing selects", async () => {
    await renderWith({ [`GET ${ENDPOINT}`]: { body: TWO } });
    expect(screen.getByRole("heading", { name: "Profiles" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Edit profile local" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Edit profile claude" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Routing" })).toBeInTheDocument();
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
    ["shared", "LLM_API_KEY"],
    ["none", undefined],
  ])("key choice %s saves api_key_env %s", async (choice, env) => {
    await renderWith({
      [`GET ${ENDPOINT}`]: { body: TWO },
      [`PUT ${ENDPOINT}`]: { body: { status: "saved", restart_required: false } },
    });
    fireEvent.click(screen.getByRole("button", { name: "Edit profile claude" }));
    fireEvent.change(screen.getByRole("combobox", { name: "API key" }), {
      target: { value: choice },
    });
    expect(screen.queryByLabelText("API key environment variable")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() => expect(callsTo("PUT")).toHaveLength(1));
    expect(putBody().profiles.claude.api_key_env).toBe(env);
  });

  it("the key-name input says what it saves as to assistive tech", async () => {
    await renderWith({ [`GET ${ENDPOINT}`]: { body: TWO } });
    fireEvent.click(screen.getByRole("button", { name: "Edit profile claude" }));
    const input = screen.getByLabelText("API key environment variable");
    expect(input).toHaveAccessibleDescription(/^Saved as LLM_API_KEY_CLAUDE\. /);
    fireEvent.change(input, { target: { value: "WORK" } });
    expect(input).toHaveAccessibleDescription(/^Saved as LLM_API_KEY_WORK\. /);
  });

  it.each([
    ["key suffix", "API key environment variable", "work", /API key variable of profile claude/],
    ["empty key suffix", "API key environment variable", "", /API key variable of profile claude/],
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
          routing: {
            default: "local",
            local_fallback: "local",
            features: { rag: "claude", summaries: "local" },
          },
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
      features: { rag: "claude", summaries: "home" },
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
    fireEvent.click(screen.getByRole("checkbox", { name: "External server" }));
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

  const SAVED = { body: { status: "saved", restart_required: false } };

  const LAN = {
    provider: "openai_compatible",
    base_url: "http://lan:8000/v1",
    model: "lan-model",
    temperature: 0.2,
    api_key_present: false,
  };
  const SHARED = {
    provider: "openai_compatible",
    base_url: "https://api.example.com/v1",
    model: "shared-model",
    vision_model: "",
    offhost: true,
    agentic: false,
    api_key_env: "LLM_API_KEY",
    api_key_present: true,
  };
  const ROUND_TRIP = view({
    legacy: false,
    profiles: { lan: LAN, claude: CLAUDE, shared: SHARED },
    routing: { default: "claude", features: { rag: "lan" }, later_key: 1 },
    output_language: "ja",
  });

  it("an untouched save sends back what GET returned", async () => {
    await renderWith({ [`GET ${ENDPOINT}`]: { body: ROUND_TRIP }, [`PUT ${ENDPOINT}`]: SAVED });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() => expect(callsTo("PUT")).toHaveLength(1));
    expect(putBody()).toEqual({
      profiles: {
        lan: {
          provider: "openai_compatible",
          base_url: "http://lan:8000/v1",
          model: "lan-model",
          temperature: 0.2,
        },
        claude: {
          provider: "openai_compatible",
          base_url: "https://openrouter.ai/api/v1",
          model: "anthropic/claude-sonnet-5",
          vision_model: "",
          offhost: true,
          agentic: true,
          api_key_env: "LLM_API_KEY_CLAUDE",
        },
        shared: {
          provider: "openai_compatible",
          base_url: "https://api.example.com/v1",
          model: "shared-model",
          vision_model: "",
          offhost: true,
          agentic: false,
          api_key_env: "LLM_API_KEY",
        },
      },
      routing: { default: "claude", features: { rag: "lan" }, later_key: 1 },
      output_language: "ja",
    });
  });

  it("an edited profile keeps its key, knobs, off-host default and agentic flag", async () => {
    await renderWith({ [`GET ${ENDPOINT}`]: { body: ROUND_TRIP }, [`PUT ${ENDPOINT}`]: SAVED });
    for (const [name, model] of [
      ["lan", "lan-2"],
      ["claude", "claude-2"],
      ["shared", "shared-2"],
    ]) {
      fireEvent.click(screen.getByRole("button", { name: `Edit profile ${name}` }));
      fireEvent.change(screen.getByLabelText("Model"), { target: { value: model } });
    }
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() => expect(callsTo("PUT")).toHaveLength(1));
    expect(putBody().profiles).toEqual({
      lan: {
        temperature: 0.2,
        provider: "openai_compatible",
        base_url: "http://lan:8000/v1",
        model: "lan-2",
        vision_model: "",
        offhost: true,
        agentic: false,
      },
      claude: {
        provider: "openai_compatible",
        base_url: "https://openrouter.ai/api/v1",
        model: "claude-2",
        vision_model: "",
        offhost: true,
        agentic: true,
        api_key_env: "LLM_API_KEY_CLAUDE",
      },
      shared: {
        provider: "openai_compatible",
        base_url: "https://api.example.com/v1",
        model: "shared-2",
        vision_model: "",
        offhost: true,
        agentic: false,
        api_key_env: "LLM_API_KEY",
      },
    });
  });

  it.each([
    ["lan", "Do not use a key", true, false],
    ["claude", "LLM_API_KEY_… (named)", true, true],
    ["shared", "LLM_API_KEY", true, false],
  ])("profile %s loads with key choice %s, off-host=%s, agentic=%s", async (name, choice, offhost, agentic) => {
    await renderWith({ [`GET ${ENDPOINT}`]: { body: ROUND_TRIP } });
    fireEvent.click(screen.getByRole("button", { name: `Edit profile ${name}` }));
    const select = screen.getByRole("combobox", { name: "API key" }) as HTMLSelectElement;
    expect(select.selectedOptions[0].textContent).toBe(choice);
    expect(screen.getByRole("checkbox", { name: "External server" })).toHaveProperty("checked", offhost);
    const agenticBox = screen.getByRole("checkbox", { name: "Use for Agentic Ask" });
    expect(agenticBox).toHaveProperty("checked", agentic);
    fireEvent.click(agenticBox);
    expect(agenticBox).toHaveProperty("checked", !agentic);
  });

  it("a new profile starts off-host and reads no key", async () => {
    await renderWith({ [`GET ${ENDPOINT}`]: { body: TWO }, [`PUT ${ENDPOINT}`]: SAVED });
    fireEvent.click(screen.getByRole("button", { name: "Add profile" }));
    const select = screen.getByRole("combobox", { name: "API key" }) as HTMLSelectElement;
    expect(select.selectedOptions[0].textContent).toBe("Do not use a key");
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() => expect(callsTo("PUT")).toHaveLength(1));
    expect(putBody().profiles["profile-3"]).toEqual({
      provider: "openai_compatible",
      base_url: "",
      model: "",
      vision_model: "",
      offhost: true,
      agentic: false,
    });
  });

  it.each([
    ["legacy key from YAML", true, "yaml", true],
    ["legacy key from the environment", true, "env", false],
    ["saved profiles", false, "yaml", false],
  ])("the YAML key notice: %s", async (_case, legacy, source, shown) => {
    await renderWith({
      [`GET ${ENDPOINT}`]: {
        body: view({
          legacy,
          profiles: { default: { ...LOCAL, api_key_present: true, api_key_source: source } },
        }),
      },
    });
    if (shown) {
      expect(screen.getByTestId("llm-yaml-key-warning")).toHaveTextContent(
        "The API key is read from search-config.yml",
      );
    } else {
      expect(screen.queryByTestId("llm-yaml-key-warning")).toBeNull();
    }
    expect(screen.getByRole("button", { name: "Save" })).toBeEnabled();
  });

  it("a save whose reload fails says it was saved", async () => {
    await renderWith({
      [`GET ${ENDPOINT}`]: [{ body: view() }, { status: 500, body: { detail: "reload boom" } }],
      [`PUT ${ENDPOINT}`]: SAVED,
    });
    fireEvent.change(screen.getByLabelText("Model"), { target: { value: "gemma4:e4b" } });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    expect(await screen.findByText("Saved. Applied now.")).toBeInTheDocument();
    const alert = screen.getByTestId("llm-save-error");
    expect(alert).toHaveTextContent("Saved, but reloading failed");
    expect(alert).not.toHaveTextContent("Could not save");
  });

  it.each([
    ["an off-host profile", "claude", "claude (external server, cannot be used)"],
    ["a missing profile", "ghost", "ghost (no such profile)"],
  ])("a saved fallback naming %s is shown as it is", async (_case, fallback, label) => {
    await renderWith({
      [`GET ${ENDPOINT}`]: {
        body: { ...TWO, routing: { default: "local", local_fallback: fallback } },
      },
      [`PUT ${ENDPOINT}`]: SAVED,
    });
    const select = screen.getByRole("combobox", {
      name: "Use instead on drives without cloud",
    }) as HTMLSelectElement;
    expect(select.selectedOptions[0].textContent).toBe(label);
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() => expect(callsTo("PUT")).toHaveLength(1));
    expect(putBody().routing).toEqual({ default: "local", local_fallback: fallback });
    fireEvent.change(select, { target: { value: "" } });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() => expect(callsTo("PUT")).toHaveLength(2));
    expect(putBody().routing).toEqual({ default: "local" });
  });

  const KEY_TEXTS = [
    "Set",
    "Not set",
    "Not set (not needed for Ollama)",
    "Not set — tasks using this profile will fail",
  ];

  it.each([
    ["openai_compatible", true, null, "Set"],
    ["openai_compatible", false, null, "Not set — tasks using this profile will fail"],
    ["ollama", false, null, "Not set (not needed for Ollama)"],
    ["disabled", false, null, "Not set"],
    ["openai_compatible", true, "OTHER", null],
    ["openai_compatible", false, "OTHER", null],
  ])("key presence: %s, present=%s, suffix changed to %s -> %s", async (provider, present, suffix, text) => {
    await renderWith({
      [`GET ${ENDPOINT}`]: {
        body: {
          ...TWO,
          profiles: {
            local: LOCAL,
            x: { ...CLAUDE, provider, api_key_env: "LLM_API_KEY_X", api_key_present: present },
          },
        },
      },
    });
    fireEvent.click(screen.getByRole("button", { name: "Edit profile x" }));
    const editor = screen.getByTestId("llm-profile-editor-p1");
    if (suffix !== null) {
      fireEvent.change(within(editor).getByLabelText("API key environment variable"), {
        target: { value: suffix },
      });
    }
    const shown = KEY_TEXTS.filter((k) => within(editor).queryByText(k) !== null);
    expect(shown).toEqual(text === null ? [] : [text]);
  });

  it.each([
    ["openai_compatible", "LLM_API_KEY_X", false, true],
    ["openai_compatible", "LLM_API_KEY_X", true, false],
    ["ollama", "LLM_API_KEY_X", false, false],
    ["openai_compatible", undefined, false, false],
  ])("the collapsed row warns of a missing key: %s %s present=%s -> %s", async (provider, env, present, warns) => {
    await renderWith({
      [`GET ${ENDPOINT}`]: {
        body: {
          ...TWO,
          profiles: {
            local: LOCAL,
            x: { ...CLAUDE, provider, api_key_env: env, api_key_present: present },
          },
        },
      },
    });
    const row = screen.getByTestId("llm-profile-row-p1");
    if (warns) {
      expect(row).toHaveTextContent("LLM_API_KEY_X");
      expect(row).toHaveTextContent("Not set — tasks using this profile will fail");
    } else {
      expect(row).not.toHaveTextContent("LLM_API_KEY");
      expect(row).not.toHaveTextContent("Not set");
    }
  });

  it("reverting whose reload fails says it was reverted, not saved", async () => {
    await renderWith({
      [`GET ${ENDPOINT}`]: [
        { body: view({ overrides_present: true }) },
        { status: 500, body: { detail: "reload boom" } },
      ],
      [`DELETE ${ENDPOINT}`]: { body: { status: "reset", removed: true, restart_required: false } },
    });
    fireEvent.click(screen.getByRole("button", { name: "Revert to YAML settings" }));
    const alert = await screen.findByTestId("llm-save-error");
    expect(alert).toHaveTextContent("Reverted to the YAML settings, but reloading failed");
    expect(alert).not.toHaveTextContent("Saved");
  });

  it("a second click while a save is in flight sends nothing", async () => {
    await renderWith({ [`GET ${ENDPOINT}`]: { body: TWO } });
    const served = mockFetch.getMockImplementation()!;
    let finish: (r: Response) => void = () => undefined;
    mockFetch.mockImplementation((url: string, init?: RequestInit) =>
      init?.method === "PUT" ? new Promise<Response>((r) => (finish = r)) : served(url, init),
    );
    const save = screen.getByRole("button", { name: "Save" });
    fireEvent.click(save);
    fireEvent.click(save);
    fireEvent.click(screen.getByRole("button", { name: "Saving..." }));
    expect(callsTo("PUT")).toHaveLength(1);
    finish(new Response(JSON.stringify({ status: "saved", restart_required: false })));
    expect(await screen.findByText("Saved. Applied now.")).toBeInTheDocument();
    expect(callsTo("PUT")).toHaveLength(1);
  });

  const ROUTED = view({
    legacy: false,
    profiles: { local: LOCAL, claude: CLAUDE },
    routing: { default: "local", local_fallback: "local", features: { rag: "claude" } },
  });
  const RAG = "Profile for AI question answering (Ask)";
  const SUMMARIES = "Profile for AI summary (short)";

  it.each([
    ["one profile", view(), 0],
    ["two profiles", ROUTED, 2],
  ])("the feature table with %s has %i profile selects", async (_case, body, count) => {
    await renderWith({ [`GET ${ENDPOINT}`]: { body } });
    expect(screen.queryAllByRole("combobox", { name: /^Profile for/ })).toHaveLength(count);
  });

  it("each feature option names the default and the off-host profiles", async () => {
    await renderWith({ [`GET ${ENDPOINT}`]: { body: ROUTED } });
    const rag = screen.getByRole("combobox", { name: RAG }) as HTMLSelectElement;
    expect(rag.selectedOptions[0].textContent).toBe("claude (external server)");
    expect(within(rag).getAllByRole("option").map((o) => o.textContent)).toEqual([
      "Default (local)",
      "local",
      "claude (external server)",
    ]);
  });

  it("a saved feature naming no profile is shown as it is and saved back", async () => {
    await renderWith({
      [`GET ${ENDPOINT}`]: {
        body: { ...ROUTED, routing: { default: "local", features: { rag: "ghost" } } },
      },
      [`PUT ${ENDPOINT}`]: SAVED,
    });
    const rag = screen.getByRole("combobox", { name: RAG }) as HTMLSelectElement;
    expect(rag.selectedOptions[0].textContent).toBe("ghost (no such profile)");
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() => expect(callsTo("PUT")).toHaveLength(1));
    expect(putBody().routing).toEqual({ default: "local", features: { rag: "ghost" } });
  });

  it("a feature whose mode is off is dimmed and labelled, and still selectable", async () => {
    await renderWith({
      [`GET ${ENDPOINT}`]: { body: ROUTED },
      [`GET ${FEATURES}`]: { body: { rag: false, summaries: "manual" } },
    });
    expect(screen.getByTestId("feature-route-rag")).toHaveTextContent("Off (Feature toggles)");
    expect(screen.getByTestId("feature-route-summaries")).not.toHaveTextContent("Off");
    const rag = screen.getByRole("combobox", { name: RAG }) as HTMLSelectElement;
    expect(rag).toBeEnabled();
    fireEvent.change(rag, { target: { value: "p0" } });
    expect(rag.selectedOptions[0].textContent).toBe("local");
  });

  it("a feature reads as off for a tristate \"false\" too", async () => {
    await renderWith({
      [`GET ${ENDPOINT}`]: { body: ROUTED },
      [`GET ${FEATURES}`]: { body: { rag: true, summaries: "false" } },
    });
    expect(screen.getByTestId("feature-route-summaries")).toHaveTextContent("Off (Feature toggles)");
    expect(screen.getByTestId("feature-route-rag")).not.toHaveTextContent("Off");
  });

  it("feature choices save in the LLM body with the rest of the section", async () => {
    await renderWith({ [`GET ${ENDPOINT}`]: { body: ROUTED }, [`PUT ${ENDPOINT}`]: SAVED });
    fireEvent.change(screen.getByRole("combobox", { name: SUMMARIES }), {
      target: { value: "p1" },
    });
    fireEvent.change(screen.getByRole("combobox", { name: RAG }), { target: { value: "" } });
    fireEvent.change(screen.getByRole("combobox", { name: "Output language" }), {
      target: { value: "en" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() => expect(callsTo("PUT")).toHaveLength(1));
    expect(putBody()).toEqual({
      profiles: {
        local: {
          provider: "ollama",
          base_url: "http://host.docker.internal:11434",
          model: "qwen3:14b",
          vision_model: "gemma3:12b",
          offhost: false,
          agentic: false,
          api_key_env: "LLM_API_KEY",
        },
        claude: {
          provider: "openai_compatible",
          base_url: "https://openrouter.ai/api/v1",
          model: "anthropic/claude-sonnet-5",
          vision_model: "",
          offhost: true,
          agentic: true,
          api_key_env: "LLM_API_KEY_CLAUDE",
        },
      },
      routing: { default: "local", local_fallback: "local", features: { summaries: "claude" } },
      output_language: "en",
    });
    expect(callsTo("PUT", FEATURES)).toHaveLength(0);
  });

  it("deleting a profile drops the feature choices that named it", async () => {
    await renderWith({ [`GET ${ENDPOINT}`]: { body: ROUTED }, [`PUT ${ENDPOINT}`]: SAVED });
    fireEvent.click(screen.getByRole("button", { name: "Edit profile claude" }));
    fireEvent.click(screen.getByRole("button", { name: "Delete profile claude" }));
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() => expect(callsTo("PUT")).toHaveLength(1));
    expect(putBody().routing).toEqual({ default: "local", local_fallback: "local" });
  });

  const RAG_EXPOSURE = {
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
  };

  it("lists each drive's destination under an off-host row", async () => {
    await renderWith({ [`GET ${ENDPOINT}`]: { body: ROUTED }, [`GET ${EXPOSURE}`]: RAG_EXPOSURE });
    const destinations = screen.getAllByTestId("feature-destinations");
    expect(destinations).toHaveLength(1);
    expect([...destinations[0].children].map((s) => s.textContent)).toEqual([
      "Sends media, misc",
      "Runs on local private",
      "Does not run old",
      "Unknown nas",
    ]);
  });

  it("lists destinations for an off-host feature that runs on the default profile", async () => {
    await renderWith({
      [`GET ${ENDPOINT}`]: {
        body: { ...ROUTED, routing: { default: "claude", local_fallback: "local", features: {} } },
      },
      [`GET ${EXPOSURE}`]: RAG_EXPOSURE,
    });
    const destinations = screen.getAllByTestId("feature-destinations");
    expect(destinations).toHaveLength(1);
    expect(destinations[0].children[0].textContent).toBe("Sends media, misc");
  });

  it("destinations disappear once the row's choice differs from the saved routing", async () => {
    await renderWith({ [`GET ${ENDPOINT}`]: { body: ROUTED }, [`GET ${EXPOSURE}`]: RAG_EXPOSURE });
    expect(screen.getAllByTestId("feature-destinations")).toHaveLength(1);
    fireEvent.change(screen.getByRole("combobox", { name: RAG }), { target: { value: "p0" } });
    expect(screen.queryByTestId("feature-destinations")).toBeNull();
    fireEvent.change(screen.getByRole("combobox", { name: RAG }), { target: { value: "p1" } });
    expect(screen.getAllByTestId("feature-destinations")).toHaveLength(1);
  });
});
