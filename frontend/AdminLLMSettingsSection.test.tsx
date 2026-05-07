import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";

import AdminLLMSettingsSection from "./AdminLLMSettingsSection";

const mockFetch = vi.fn();

beforeEach(() => {
  vi.stubGlobal("fetch", mockFetch);
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

function jsonResponse(data: unknown, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

const ENDPOINT = "/api/addons/intelligence/admin/llm";

function defaultPayload(overrides: Record<string, unknown> = {}) {
  return {
    provider: "disabled",
    base_url: "",
    model: "",
    output_language: "auto",
    vision_model: "",
    available_providers: ["disabled", "ollama", "openai_compatible"],
    available_output_languages: ["auto", "ja", "en"],
    api_key_present: false,
    api_key_env_var: "LLM_API_KEY",
    overrides_present: false,
    ...overrides,
  };
}

describe("AdminLLMSettingsSection", () => {
  it("renders provider radios with the current selection", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse(defaultPayload({ provider: "ollama" })),
    );
    render(<AdminLLMSettingsSection />);
    await waitFor(() =>
      expect(screen.getByRole("radio", { name: /Ollama/i })).toBeChecked(),
    );
  });

  it("disables save when an openai-compatible provider is selected without an API key", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse(
        defaultPayload({
          provider: "openai_compatible",
          api_key_present: false,
        }),
      ),
    );
    render(<AdminLLMSettingsSection />);
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /save/i })).toBeDisabled(),
    );
  });

  it("enables save for ollama (no API key needed)", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse(defaultPayload({ provider: "ollama" })),
    );
    render(<AdminLLMSettingsSection />);
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /save/i })).not.toBeDisabled(),
    );
  });

  it("PUTs the form values on save", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse(defaultPayload({ provider: "ollama" })),
    );
    mockFetch.mockResolvedValueOnce(jsonResponse({ status: "saved" }));
    mockFetch.mockResolvedValueOnce(
      jsonResponse(
        defaultPayload({
          provider: "ollama",
          model: "gemma4:e4b",
          overrides_present: true,
        }),
      ),
    );

    render(<AdminLLMSettingsSection />);
    await waitFor(() => screen.getByRole("button", { name: /save/i }));

    const modelInput = screen.getByLabelText(/text model/i);
    fireEvent.change(modelInput, { target: { value: "gemma4:e4b" } });
    fireEvent.click(screen.getByRole("button", { name: /save/i }));

    await waitFor(() => {
      const putCall = mockFetch.mock.calls.find((c) => c[1]?.method === "PUT");
      expect(putCall?.[0]).toBe(ENDPOINT);
      const body = JSON.parse(String(putCall?.[1]?.body ?? "{}"));
      expect(body.provider).toBe("ollama");
      expect(body.model).toBe("gemma4:e4b");
    });
  });

  it("renders the overrides banner with reset when overrides exist", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse(defaultPayload({ overrides_present: true, provider: "ollama" })),
    );
    render(<AdminLLMSettingsSection />);
    await waitFor(() =>
      expect(screen.getByTestId("llm-overrides-banner")).toBeInTheDocument(),
    );
  });

  it("DELETEs on reset and refetches", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse(defaultPayload({ overrides_present: true, provider: "ollama" })),
    );
    mockFetch.mockResolvedValueOnce(
      jsonResponse({ status: "reset", removed: true }),
    );
    mockFetch.mockResolvedValueOnce(
      jsonResponse(defaultPayload({ overrides_present: false })),
    );

    render(<AdminLLMSettingsSection />);
    await waitFor(() => screen.getByTestId("llm-overrides-banner"));
    fireEvent.click(screen.getByRole("button", { name: /reset/i }));

    await waitFor(() => {
      const deleteCall = mockFetch.mock.calls.find((c) => c[1]?.method === "DELETE");
      expect(deleteCall?.[0]).toBe(ENDPOINT);
    });
    await waitFor(() =>
      expect(screen.queryByTestId("llm-overrides-banner")).toBeNull(),
    );
  });
});
