import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor, within } from "@testing-library/react";

import AdminFeaturesSettingsSection from "./AdminFeaturesSettingsSection";
import AdminLLMSettingsSection from "./AdminLLMSettingsSection";

const FEATURES = "/api/addons/intelligence/admin/features";
const LLM = "/api/addons/intelligence/admin/llm";
const EXPOSURE = "/api/addons/intelligence/admin/llm/exposure";

const SUMMARIES = "Profile for AI summary (short)";

type Doc = {
  profiles: Record<string, Record<string, unknown>>;
  routing: Record<string, unknown>;
  output_language: string;
};

const LOCAL = {
  provider: "ollama",
  base_url: "http://host.docker.internal:11434",
  model: "qwen3:14b",
  vision_model: "",
  offhost: false,
  agentic: false,
};
const CLAUDE = {
  provider: "openai_compatible",
  base_url: "https://openrouter.ai/api/v1",
  model: "claude",
  vision_model: "",
  offhost: true,
  agentic: false,
  api_key_env: "LLM_API_KEY_CLAUDE",
};

const MODES = {
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
};

let doc: Doc;
const mockFetch = vi.fn();

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function llmView() {
  return {
    profiles: Object.fromEntries(
      Object.entries(doc.profiles).map(([name, p]) => [
        name,
        { ...p, api_key_present: typeof p.api_key_env === "string" },
      ]),
    ),
    routing: doc.routing,
    legacy: false,
    error: null,
    output_language: doc.output_language,
    output_language_restart_pending: false,
    features: ["rag", "summaries"],
    available_providers: ["disabled", "ollama", "openai_compatible"],
    available_output_languages: ["auto", "ja", "en"],
    overrides_present: true,
  };
}

function serveDocument(initial: Doc) {
  doc = initial;
  mockFetch.mockImplementation(async (url: string, init?: RequestInit) => {
    const method = init?.method ?? "GET";
    if (url === LLM && method === "GET") return json(llmView());
    if (url === LLM && method === "PUT") {
      const body = JSON.parse(String(init?.body)) as Doc;
      doc = {
        ...body,
        profiles: Object.fromEntries(
          Object.entries(body.profiles).map(([name, p]) => [
            name,
            Object.fromEntries(Object.entries(p).filter(([k]) => k !== "api_key_present")),
          ]),
        ),
      };
      return json({ status: "saved", restart_required: false });
    }
    if (url === EXPOSURE) return json({ features: {}, local_fallback: null });
    if (url === FEATURES && method === "GET") return json(MODES);
    if (url === FEATURES && method === "PUT") return json({ status: "saved", restart_required: true });
    throw new Error(`unexpected ${method} ${url}`);
  });
}

beforeEach(() => {
  vi.stubGlobal("fetch", mockFetch);
});

afterEach(() => {
  mockFetch.mockReset();
  vi.unstubAllGlobals();
});

async function renderPage(initial: Doc) {
  serveDocument(initial);
  render(
    <>
      <AdminFeaturesSettingsSection />
      <AdminLLMSettingsSection />
    </>,
  );
  const llm = await screen.findByRole("region", { name: "LLM profiles" });
  await within(llm).findByRole("button", { name: "Save" });
  const features = (await screen.findByRole("heading", { name: "Feature toggles" })).closest(
    "section",
  ) as HTMLElement;
  await within(features).findByRole("combobox", { name: SUMMARIES });
  return { llm, features };
}

async function saveFeatures(features: HTMLElement) {
  fireEvent.click(within(features).getByRole("button", { name: "Save" }));
  await within(features).findByRole("status");
}

async function saveLLM(llm: HTMLElement) {
  fireEvent.click(within(llm).getByRole("button", { name: "Save" }));
  await within(llm).findByText("Saved. Applied now.");
}

const TWO: Doc = {
  profiles: { local: LOCAL, claude: CLAUDE },
  routing: { default: "local" },
  output_language: "auto",
};

describe("the Features and LLM sections on one page", () => {
  it("an LLM save after a Features save keeps the feature choice", async () => {
    const { llm, features } = await renderPage(TWO);
    fireEvent.change(within(features).getByRole("combobox", { name: SUMMARIES }), {
      target: { value: "claude" },
    });
    await saveFeatures(features);
    fireEvent.change(within(llm).getByRole("combobox", { name: "Output language" }), {
      target: { value: "ja" },
    });
    await saveLLM(llm);
    expect(doc.routing).toEqual({ default: "local", features: { summaries: "claude" } });
    expect(doc.output_language).toBe("ja");
  });

  it("a Features save after an LLM save keeps the profile edit", async () => {
    const { llm, features } = await renderPage(TWO);
    fireEvent.click(within(llm).getByRole("button", { name: "Edit profile local" }));
    fireEvent.click(within(llm).getByRole("checkbox", { name: "Send off-host" }));
    await saveLLM(llm);
    const select = within(features).getByRole("combobox", { name: SUMMARIES });
    await waitFor(() =>
      expect(within(select).getAllByRole("option").map((o) => o.textContent)).toContain(
        "local (off-host)",
      ),
    );
    fireEvent.change(select, { target: { value: "claude" } });
    await saveFeatures(features);
    expect(doc.profiles.local).toEqual({ ...LOCAL, offhost: true });
    expect(doc.routing).toEqual({ default: "local", features: { summaries: "claude" } });
  });

  it("a rename saved in the LLM section survives a later Features save", async () => {
    const { llm, features } = await renderPage({
      ...TWO,
      routing: { default: "local", features: { rag: "claude" } },
    });
    fireEvent.click(within(llm).getByRole("button", { name: "Edit profile claude" }));
    fireEvent.change(within(llm).getByLabelText("Name"), { target: { value: "sonnet" } });
    await saveLLM(llm);
    expect(doc.routing).toEqual({ default: "local", features: { rag: "sonnet" } });
    const select = within(features).getByRole("combobox", { name: SUMMARIES });
    await waitFor(() =>
      expect(within(select).getAllByRole("option").map((o) => o.textContent)).toContain(
        "sonnet (off-host)",
      ),
    );
    fireEvent.change(select, { target: { value: "sonnet" } });
    await saveFeatures(features);
    expect(Object.keys(doc.profiles)).toEqual(["local", "sonnet"]);
    expect(doc.routing).toEqual({
      default: "local",
      features: { rag: "sonnet", summaries: "sonnet" },
    });
  });

  it("an unsaved LLM edit outlives a Features save and then saves on top of it", async () => {
    const { llm, features } = await renderPage(TWO);
    fireEvent.click(within(llm).getByRole("button", { name: "Edit profile local" }));
    fireEvent.change(within(llm).getByLabelText("Model"), { target: { value: "edited" } });
    fireEvent.change(within(features).getByRole("combobox", { name: SUMMARIES }), {
      target: { value: "claude" },
    });
    await saveFeatures(features);
    expect(within(llm).getByLabelText("Model")).toHaveValue("edited");
    await saveLLM(llm);
    expect(doc.profiles.local.model).toBe("edited");
    expect(doc.routing).toEqual({ default: "local", features: { summaries: "claude" } });
  });

  it("an unsaved Features choice outlives an LLM save and then saves on top of it", async () => {
    const { llm, features } = await renderPage(TWO);
    fireEvent.change(within(features).getByRole("combobox", { name: SUMMARIES }), {
      target: { value: "claude" },
    });
    fireEvent.change(within(llm).getByRole("combobox", { name: "Output language" }), {
      target: { value: "ja" },
    });
    await saveLLM(llm);
    expect(within(features).getByRole("combobox", { name: SUMMARIES })).toHaveValue("claude");
    await saveFeatures(features);
    expect(doc.routing).toEqual({ default: "local", features: { summaries: "claude" } });
    expect(doc.output_language).toBe("ja");
  });
});
