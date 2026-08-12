import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";

import AdminFeaturesSettingsSection from "./AdminFeaturesSettingsSection";

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

const ENDPOINT = "/api/addons/intelligence/admin/features";

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
    tristate_values: ["false", "manual", "on_index"],
    overrides_present: false,
    ...overrides,
  };
}

describe("AdminFeaturesSettingsSection", () => {
  it("renders bool checkboxes and tristate radios", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse(defaultPayload()));
    render(<AdminFeaturesSettingsSection />);
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /save/i })).toBeInTheDocument(),
    );
    expect(screen.getAllByRole("checkbox").length).toBe(3);
    expect(screen.getAllByRole("radio").length).toBe(21); // 7 enums × 3 values
    expect(
      screen.getByText(/AI chapter candidates|settings\.features\.fields\.chapter_suggestions\.label/),
    ).toBeInTheDocument();
  });

  it("PUTs the form values on save", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse(defaultPayload()));
    mockFetch.mockResolvedValueOnce(jsonResponse({ status: "saved" }));
    mockFetch.mockResolvedValueOnce(
      jsonResponse(defaultPayload({ overrides_present: true, indexing: false })),
    );

    render(<AdminFeaturesSettingsSection />);
    await waitFor(() => screen.getByRole("button", { name: /save/i }));

    const indexingBox = screen.getAllByRole("checkbox")[0];
    fireEvent.click(indexingBox);
    fireEvent.click(screen.getByRole("button", { name: /save/i }));

    await waitFor(() => {
      const putCall = mockFetch.mock.calls.find((c) => c[1]?.method === "PUT");
      expect(putCall?.[0]).toBe(ENDPOINT);
      const body = JSON.parse(String(putCall?.[1]?.body ?? "{}"));
      expect(body.indexing).toBe(false);
      expect(body.chapter_suggestions).toBe("manual");
    });
  });

  it("hides the overrides banner when no override is active", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse(defaultPayload({ overrides_present: false })),
    );
    render(<AdminFeaturesSettingsSection />);
    await waitFor(() => screen.getByRole("button", { name: /save/i }));
    expect(screen.queryByTestId("features-overrides-banner")).toBeNull();
  });

  it("renders the overrides banner with reset button when overrides exist", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse(defaultPayload({ overrides_present: true })),
    );
    render(<AdminFeaturesSettingsSection />);
    await waitFor(() =>
      expect(screen.getByTestId("features-overrides-banner")).toBeInTheDocument(),
    );
    expect(screen.getByRole("button", { name: /reset/i })).toBeInTheDocument();
  });

  it("DELETEs on reset and refetches", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse(defaultPayload({ overrides_present: true })),
    );
    mockFetch.mockResolvedValueOnce(
      jsonResponse({ status: "reset", removed: true, restart_required: true }),
    );
    mockFetch.mockResolvedValueOnce(
      jsonResponse(defaultPayload({ overrides_present: false })),
    );

    render(<AdminFeaturesSettingsSection />);
    await waitFor(() => screen.getByTestId("features-overrides-banner"));
    fireEvent.click(screen.getByRole("button", { name: /reset/i }));

    await waitFor(() => {
      const deleteCall = mockFetch.mock.calls.find((c) => c[1]?.method === "DELETE");
      expect(deleteCall?.[0]).toBe(ENDPOINT);
    });
    await waitFor(() =>
      expect(screen.queryByTestId("features-overrides-banner")).toBeNull(),
    );
  });

  it("renders load error inline when GET fails", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse({ detail: "boom" }, 500));
    render(<AdminFeaturesSettingsSection />);
    await waitFor(() =>
      expect(screen.getByText(/boom|HTTP 500|Failed/)).toBeInTheDocument(),
    );
  });
});
