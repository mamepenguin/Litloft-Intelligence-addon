import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";

import AdminRAGSettingsSection from "./AdminRAGSettingsSection";

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

const ENDPOINT = "/api/addons/intelligence/admin/rag";

function defaultPayload(overrides: Record<string, unknown> = {}) {
  return {
    personal_history_enabled: true,
    category_expansion_enabled: false,
    overrides_present: false,
    ...overrides,
  };
}

describe("AdminRAGSettingsSection", () => {
  it("renders both checkboxes with the current state", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse(defaultPayload()));
    render(<AdminRAGSettingsSection />);
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /save/i })).toBeInTheDocument(),
    );
    const boxes = screen.getAllByRole("checkbox");
    expect(boxes).toHaveLength(2);
    expect(boxes[0]).toBeChecked();
    expect(boxes[1]).not.toBeChecked();
  });

  it("PUTs the form values on save", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse(defaultPayload()));
    mockFetch.mockResolvedValueOnce(jsonResponse({ status: "saved" }));
    mockFetch.mockResolvedValueOnce(
      jsonResponse(
        defaultPayload({
          personal_history_enabled: false,
          category_expansion_enabled: true,
          overrides_present: true,
        }),
      ),
    );

    render(<AdminRAGSettingsSection />);
    await waitFor(() => screen.getByRole("button", { name: /save/i }));

    const boxes = screen.getAllByRole("checkbox");
    fireEvent.click(boxes[0]); // toggle off
    fireEvent.click(boxes[1]); // toggle on
    fireEvent.click(screen.getByRole("button", { name: /save/i }));

    await waitFor(() => {
      const putCall = mockFetch.mock.calls.find((c) => c[1]?.method === "PUT");
      expect(putCall?.[0]).toBe(ENDPOINT);
      const body = JSON.parse(String(putCall?.[1]?.body ?? "{}"));
      expect(body.personal_history_enabled).toBe(false);
      expect(body.category_expansion_enabled).toBe(true);
    });
  });

  it("renders the overrides banner with reset when overrides exist", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse(defaultPayload({ overrides_present: true })),
    );
    render(<AdminRAGSettingsSection />);
    await waitFor(() =>
      expect(screen.getByTestId("rag-overrides-banner")).toBeInTheDocument(),
    );
  });

  it("DELETEs on reset and refetches", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse(defaultPayload({ overrides_present: true })),
    );
    mockFetch.mockResolvedValueOnce(
      jsonResponse({ status: "reset", removed: true }),
    );
    mockFetch.mockResolvedValueOnce(
      jsonResponse(defaultPayload({ overrides_present: false })),
    );

    render(<AdminRAGSettingsSection />);
    await waitFor(() => screen.getByTestId("rag-overrides-banner"));
    fireEvent.click(screen.getByRole("button", { name: /reset/i }));

    await waitFor(() => {
      const deleteCall = mockFetch.mock.calls.find((c) => c[1]?.method === "DELETE");
      expect(deleteCall?.[0]).toBe(ENDPOINT);
    });
    await waitFor(() =>
      expect(screen.queryByTestId("rag-overrides-banner")).toBeNull(),
    );
  });
});
