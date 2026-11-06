/**
 * SPEC-ADDON-050 I10: `features.image_ocr` is a three-state setting on the admin
 * screen, like the other features, and saving it writes the mode.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";

import AdminFeaturesSettingsSection from "./AdminFeaturesSettingsSection";

const ENDPOINT = "/api/addons/intelligence/admin/features";

type Reply = { status?: number; body: unknown };

const mockFetch = vi.fn();

function jsonResponse({ status = 200, body }: Reply) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function serve(routes: Record<string, Reply | Reply[]>) {
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

function bodyOf(method: string, url: string) {
  const calls = mockFetch.mock.calls.filter(
    (c) => (c[1]?.method ?? "GET") === method && c[0] === url,
  );
  return JSON.parse(String(calls[calls.length - 1][1].body));
}

function payload(overrides: Record<string, unknown> = {}) {
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
    image_ocr: "on_index",
    video_frame_ocr: "on_index",
    scanned_pdf_ocr: "on_index",
    tristate_values: ["false", "manual", "on_index"],
    overrides_present: false,
    ...overrides,
  };
}

beforeEach(() => {
  vi.stubGlobal("fetch", mockFetch);
});

afterEach(() => {
  mockFetch.mockReset();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("AdminFeaturesSettingsSection OCR (SPEC-ADDON-050 I10)", () => {
  it("offers the three modes for text in pictures and saves the chosen one", async () => {
    serve({
      [`GET ${ENDPOINT}`]: { body: payload() },
      [`PUT ${ENDPOINT}`]: { body: { status: "saved", restart_required: true } },
    });
    render(<AdminFeaturesSettingsSection />);
    await screen.findByRole("button", { name: "Save" });

    expect(screen.getAllByRole("combobox", { name: /^When .* runs$/ })).toHaveLength(14);

    const select = screen.getByRole("combobox", { name: "When Text in image runs" });
    expect((select as HTMLSelectElement).value).toBe("on_index");
    expect(
      Array.from((select as HTMLSelectElement).options).map((option) => option.value),
    ).toEqual(["false", "manual", "on_index"]);

    fireEvent.change(select, { target: { value: "manual" } });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() =>
      expect(
        screen.getByText("Saved. Restart required for the change to take effect"),
      ).toBeInTheDocument(),
    );
    expect(bodyOf("PUT", ENDPOINT).image_ocr).toBe("manual");
  });

  it("shows the saved mode it reads back from the server", async () => {
    serve({ [`GET ${ENDPOINT}`]: { body: payload({ image_ocr: "false" }) } });
    render(<AdminFeaturesSettingsSection />);

    const select = await screen.findByRole("combobox", {
      name: "When Text in image runs",
    });
    expect((select as HTMLSelectElement).value).toBe("false");
  });
});
