// AdminEmbeddingSettingsSection test (Plan Phase 4 RED-phase)
//
// Covers Phase 4 of the GUI text-embedding-model feature (spec
// docs/superpowers/specs/2026-05-20-gui-text-embedding-model.md §4 and
// plan Phase 4). Mirrors AdminTranscriptionSettingsSection.test.tsx for
// fetch mocking, render, and assertion idioms.
//
// API surface under test (host-proxy URL, Group B already implemented):
//   GET    /api/addons/intelligence/admin/embedding
//   PUT    /api/addons/intelligence/admin/embedding
//   DELETE /api/addons/intelligence/admin/embedding
//
// Locale-tolerant matchers
// ------------------------
// The addon's own ja/en strings will live under `settings.embedding.*`
// inside addons/intelligence/frontend/messages/{ja,en}.json and are
// merged into frontend/src/messages/{locale}.json by
// scripts/merge-addon-messages.mjs. The merge does NOT run on `pnpm
// test`, so tests must tolerate two states:
//   (1) en.json already contains the addon's `settings.embedding.*` keys
//       → the rendered text is the human string (e.g. "Reset to default")
//   (2) en.json has not been re-merged yet
//       → the next-intl mock in src/test/setup.ts falls back to the raw
//         key path "settings.embedding.<key>"
// Regex matchers therefore use the loose patterns the transcription
// test established (e.g. /reset|settings\.embedding\.reset/i).

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor, within } from "@testing-library/react";

import AdminEmbeddingSettingsSection from "./AdminEmbeddingSettingsSection";

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

const ENDPOINT = "/api/addons/intelligence/admin/embedding";

// Catalog mirrors the shape returned by GET (Group B):
//   { effective, recorded, reindex_pending, catalog: [{id, family, dim, weight}] }
//
// `family: "ja" | "multi"` drives the language group headings; `weight:
// "light" | "normal" | "heavy"` drives the weight-hint label mapping.
function defaultPayload(overrides: Record<string, unknown> = {}) {
  return {
    effective: "cl-nagoya/ruri-v3-130m",
    recorded: "cl-nagoya/ruri-v3-130m",
    reindex_pending: false,
    catalog: [
      { id: "cl-nagoya/ruri-v3-130m", family: "ja", dim: 768, weight: "normal" },
      { id: "cl-nagoya/ruri-v3-310m", family: "ja", dim: 1024, weight: "heavy" },
      { id: "ibm-granite/granite-embedding-97m-multilingual-r2", family: "multi", dim: 384, weight: "light" },
      { id: "ibm-granite/granite-embedding-311m-multilingual-r2", family: "multi", dim: 768, weight: "normal" },
    ],
    ...overrides,
  };
}

// A loose regex over both the eventual English label and the raw
// next-intl key path. Use this everywhere the addon i18n file is the
// source of truth (so missing keys do not turn the test red for the
// wrong reason).
const re = (...patterns: string[]) =>
  new RegExp(patterns.map((p) => `(?:${p})`).join("|"), "i");

describe("AdminEmbeddingSettingsSection", () => {
  // 1. GET success → heading, effective model, both group headings,
  //    one radio per catalog entry, the effective one is checked.
  it("renders catalog with language groups and selects the effective model", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse(defaultPayload()));
    render(<AdminEmbeddingSettingsSection />);

    // Heading + every catalog id appears as a radio
    await waitFor(() =>
      expect(
        screen.getByRole("radio", { name: /ruri-v3-130m/i }),
      ).toBeInTheDocument(),
    );
    expect(screen.getByRole("radio", { name: /ruri-v3-310m/i })).toBeInTheDocument();
    expect(
      screen.getByRole("radio", { name: /granite-embedding-97m-multilingual-r2/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("radio", { name: /granite-embedding-311m-multilingual-r2/i }),
    ).toBeInTheDocument();

    // Effective is the radio that comes back checked
    expect(screen.getByRole("radio", { name: /ruri-v3-130m/i })).toBeChecked();

    // Both language group headings present (Japanese-specialised / multilingual)
    expect(
      screen.getByText(
        re(
          "日本語特化",
          "Japanese",
          "settings\\.embedding\\.groups\\.ja",
        ),
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        re(
          "多言語",
          "Multilingual",
          "settings\\.embedding\\.groups\\.multi",
        ),
      ),
    ).toBeInTheDocument();
  });

  // 2a. reindex_pending true → recorded-mismatch badge appears.
  it("shows the recorded-mismatch badge when reindex_pending is true", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse(
        defaultPayload({
          effective: "ibm-granite/granite-embedding-97m-multilingual-r2",
          recorded: "cl-nagoya/ruri-v3-130m",
          reindex_pending: true,
        }),
      ),
    );
    render(<AdminEmbeddingSettingsSection />);

    await waitFor(() =>
      expect(
        screen.getByText(
          re(
            "記録モデル不一致",
            "Recorded model mismatch",
            "reindex",
            "pending",
            "settings\\.embedding\\.reindexPending",
          ),
        ),
      ).toBeInTheDocument(),
    );
  });

  // 2b. reindex_pending false → no badge.
  it("hides the recorded-mismatch badge when reindex_pending is false", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse(defaultPayload()));
    render(<AdminEmbeddingSettingsSection />);
    await waitFor(() =>
      expect(screen.getByRole("radio", { name: /ruri-v3-130m/i })).toBeChecked(),
    );
    expect(
      screen.queryByText(
        re(
          "記録モデル不一致",
          "Recorded model mismatch",
          "settings\\.embedding\\.reindexPending",
        ),
      ),
    ).toBeNull();
  });

  // 3. Selecting the effective model → no dialog (no-op).
  it("does not open the confirm dialog when selecting the current effective model", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse(defaultPayload()));
    render(<AdminEmbeddingSettingsSection />);

    await waitFor(() =>
      expect(screen.getByRole("radio", { name: /ruri-v3-130m/i })).toBeChecked(),
    );
    fireEvent.click(screen.getByRole("radio", { name: /ruri-v3-130m/i }));

    expect(screen.queryByRole("dialog")).toBeNull();
    // No PUT issued either
    expect(
      mockFetch.mock.calls.find((c) => c[1]?.method === "PUT"),
    ).toBeUndefined();
  });

  // 4. Selecting a different model → dialog appears with all four cost items.
  it("opens a confirmation dialog with the four reindex-cost items", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse(defaultPayload()));
    render(<AdminEmbeddingSettingsSection />);

    await waitFor(() =>
      expect(screen.getByRole("radio", { name: /granite-embedding-97m-multilingual-r2/i })).toBeInTheDocument(),
    );
    fireEvent.click(screen.getByRole("radio", { name: /granite-embedding-97m-multilingual-r2/i }));

    const dialog = await screen.findByRole("dialog");
    // (a) Reindex required + Ask/text-search degraded until done
    expect(
      within(dialog).getByText(
        re(
          "再 ?index",
          "reindex",
          "Ask",
          "settings\\.embedding\\.confirm\\.reindex",
        ),
      ),
    ).toBeInTheDocument();
    // (b) Container restart required
    expect(
      within(dialog).getByText(
        re(
          "再起動",
          "restart",
          "settings\\.embedding\\.confirm\\.restart",
        ),
      ),
    ).toBeInTheDocument();
    // (c) detailed_summary_citations must be backfilled separately
    //     (spec §2.3 / Group A MEDIUM #4)
    expect(
      within(dialog).getByText(
        re(
          "detailed_summary_citations",
          "backfill_detailed_citations",
          "詳細要約.*引用|citations.*backfill",
          "settings\\.embedding\\.confirm\\.detailedCitations",
        ),
      ),
    ).toBeInTheDocument();
    // (d) min_score_text and other calibrated thresholds are not auto-adjusted
    //     (spec §5 / §4.3 caveat)
    expect(
      within(dialog).getByText(
        re(
          "min_score_text",
          "校正",
          "calibrat",
          "settings\\.embedding\\.confirm\\.minScore",
        ),
      ),
    ).toBeInTheDocument();
  });

  // 5. Dialog cancel → no PUT, dialog closes.
  it("does not PUT when the confirmation dialog is cancelled", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse(defaultPayload()));
    render(<AdminEmbeddingSettingsSection />);

    await waitFor(() =>
      expect(
        screen.getByRole("radio", { name: /granite-embedding-97m-multilingual-r2/i }),
      ).toBeInTheDocument(),
    );
    fireEvent.click(screen.getByRole("radio", { name: /granite-embedding-97m-multilingual-r2/i }));

    const dialog = await screen.findByRole("dialog");
    fireEvent.click(
      within(dialog).getByRole("button", {
        name: re("キャンセル", "cancel", "settings\\.embedding\\.confirm\\.cancel"),
      }),
    );

    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(
      mockFetch.mock.calls.find((c) => c[1]?.method === "PUT"),
    ).toBeUndefined();
  });

  // 6. Dialog confirm → PUT issued, GET refresh, displayed effective updates.
  it("PUTs the selected model on confirm and refreshes from GET", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse(defaultPayload()));
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        effective: "ibm-granite/granite-embedding-97m-multilingual-r2",
        recorded: "cl-nagoya/ruri-v3-130m",
        reindex_pending: true,
        catalog: defaultPayload().catalog,
        status: "saved",
        restart_required: true,
        core_notified: "ok",
      }),
    );
    // Re-GET after PUT — effective now matches the new selection
    mockFetch.mockResolvedValueOnce(
      jsonResponse(
        defaultPayload({
          effective: "ibm-granite/granite-embedding-97m-multilingual-r2",
          recorded: "cl-nagoya/ruri-v3-130m",
          reindex_pending: true,
        }),
      ),
    );

    render(<AdminEmbeddingSettingsSection />);
    await waitFor(() =>
      expect(
        screen.getByRole("radio", { name: /granite-embedding-97m-multilingual-r2/i }),
      ).toBeInTheDocument(),
    );

    fireEvent.click(screen.getByRole("radio", { name: /granite-embedding-97m-multilingual-r2/i }));

    const dialog = await screen.findByRole("dialog");
    fireEvent.click(
      within(dialog).getByRole("button", {
        name: re("確認", "confirm", "settings\\.embedding\\.confirm\\.confirm"),
      }),
    );

    // PUT body asserts {text_embedding: selected}
    await waitFor(() => {
      const putCall = mockFetch.mock.calls.find((c) => c[1]?.method === "PUT");
      expect(putCall?.[0]).toBe(ENDPOINT);
      const body = JSON.parse(String(putCall?.[1]?.body ?? "{}"));
      expect(body.text_embedding).toBe("ibm-granite/granite-embedding-97m-multilingual-r2");
    });

    // Dialog closes after success
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());

    // Re-render reflects the new effective + reindex_pending badge
    await waitFor(() =>
      expect(
        screen.getByRole("radio", { name: /granite-embedding-97m-multilingual-r2/i }),
      ).toBeChecked(),
    );
    expect(
      screen.getByText(
        re(
          "記録モデル不一致",
          "Recorded model mismatch",
          "settings\\.embedding\\.reindexPending",
        ),
      ),
    ).toBeInTheDocument();
  });

  // 7. PUT 422 → server detail surfaced, no state update, dialog stays.
  it("shows the server detail when PUT returns 422 and keeps the dialog open", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse(defaultPayload()));
    mockFetch.mockResolvedValueOnce(
      jsonResponse(
        { detail: "Unknown text_embedding model 'foo/bar'" },
        422,
      ),
    );

    render(<AdminEmbeddingSettingsSection />);
    await waitFor(() =>
      expect(
        screen.getByRole("radio", { name: /granite-embedding-97m-multilingual-r2/i }),
      ).toBeInTheDocument(),
    );

    fireEvent.click(screen.getByRole("radio", { name: /granite-embedding-97m-multilingual-r2/i }));

    const dialog = await screen.findByRole("dialog");
    fireEvent.click(
      within(dialog).getByRole("button", {
        name: re("確認", "confirm", "settings\\.embedding\\.confirm\\.confirm"),
      }),
    );

    await waitFor(() =>
      expect(
        screen.getByText(/Unknown text_embedding model 'foo\/bar'/i),
      ).toBeInTheDocument(),
    );

    // Effective still the original; dialog still open so the operator can correct.
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: /ruri-v3-130m/i })).toBeChecked();
  });

  // 8. Reset → DELETE issued, GET refresh fires.
  it("DELETEs and then refetches when the reset button is clicked", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse(
        defaultPayload({
          effective: "ibm-granite/granite-embedding-97m-multilingual-r2",
          recorded: "cl-nagoya/ruri-v3-130m",
          reindex_pending: true,
        }),
      ),
    );
    mockFetch.mockResolvedValueOnce(
      jsonResponse({
        status: "reset",
        removed: true,
        restart_required: true,
        core_notified: "ok",
        effective: "cl-nagoya/ruri-v3-130m",
        recorded: "cl-nagoya/ruri-v3-130m",
        reindex_pending: false,
        catalog: defaultPayload().catalog,
      }),
    );
    mockFetch.mockResolvedValueOnce(jsonResponse(defaultPayload()));

    render(<AdminEmbeddingSettingsSection />);
    await waitFor(() =>
      expect(
        screen.getByRole("radio", { name: /granite-embedding-97m-multilingual-r2/i }),
      ).toBeChecked(),
    );

    fireEvent.click(
      screen.getByRole("button", {
        name: re(
          "デフォルトに戻す",
          "Reset to default",
          "reset",
          "settings\\.embedding\\.reset",
        ),
      }),
    );

    await waitFor(() => {
      const deleteCall = mockFetch.mock.calls.find(
        (c) => c[1]?.method === "DELETE",
      );
      expect(deleteCall?.[0]).toBe(ENDPOINT);
    });
    // GET refresh fires (3 fetches: initial GET, DELETE, refresh GET)
    await waitFor(() => expect(mockFetch.mock.calls.length).toBeGreaterThanOrEqual(3));
    await waitFor(() =>
      expect(screen.getByRole("radio", { name: /ruri-v3-130m/i })).toBeChecked(),
    );
  });

  // 9. GET network failure → inline error + retry button → re-GET on click.
  it("shows an inline error and retry button when the initial GET fails", async () => {
    mockFetch.mockRejectedValueOnce(new Error("network down"));
    mockFetch.mockResolvedValueOnce(jsonResponse(defaultPayload()));

    render(<AdminEmbeddingSettingsSection />);
    await waitFor(() =>
      expect(
        screen.getByText(
          re(
            "network down",
            "失敗",
            "Failed",
            "loadFailed",
            "settings\\.embedding\\.loadFailed",
          ),
        ),
      ).toBeInTheDocument(),
    );

    const retry = screen.getByRole("button", {
      name: re("再試行", "retry", "settings\\.embedding\\.retry"),
    });
    fireEvent.click(retry);

    // Catalog now renders after the second GET resolves
    await waitFor(() =>
      expect(
        screen.getByRole("radio", { name: /ruri-v3-130m/i }),
      ).toBeChecked(),
    );
  });

  // 10. No emoji anywhere in the rendered UI (UI rule:
  //     `feedback_no_emoji_in_ui`).
  it("renders no emoji characters in any text node", async () => {
    mockFetch.mockResolvedValueOnce(
      jsonResponse(
        defaultPayload({
          effective: "ibm-granite/granite-embedding-97m-multilingual-r2",
          recorded: "cl-nagoya/ruri-v3-130m",
          reindex_pending: true,
        }),
      ),
    );
    render(<AdminEmbeddingSettingsSection />);
    await waitFor(() =>
      expect(
        screen.getByRole("radio", { name: /granite-embedding-97m-multilingual-r2/i }),
      ).toBeInTheDocument(),
    );

    // Pictographic block (covers 🔄 ⚠ ✓ ✅ 📦 etc.)
    expect(document.body.textContent ?? "").not.toMatch(
      /[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}]/u,
    );
  });

  // 11. Weight label mapping: each `weight` value renders a translated label.
  it("renders a weight-hint label for each weight value", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse(defaultPayload()));
    render(<AdminEmbeddingSettingsSection />);

    await waitFor(() =>
      expect(
        screen.getByRole("radio", { name: /ruri-v3-130m/i }),
      ).toBeInTheDocument(),
    );

    // light (used by granite-97m-r2) → "軽量" / "Light"
    expect(
      screen.getAllByText(
        re("軽量", "light", "settings\\.embedding\\.weight\\.light"),
      ).length,
    ).toBeGreaterThan(0);
    // normal (used by ruri-v3-130m + granite-311m-r2) → "標準" / "Normal"
    expect(
      screen.getAllByText(
        re("標準", "normal", "settings\\.embedding\\.weight\\.normal"),
      ).length,
    ).toBeGreaterThan(0);
    // heavy (used by ruri-v3-310m) → "重い" / "Heavy"
    expect(
      screen.getAllByText(
        re("重い", "heavy", "settings\\.embedding\\.weight\\.heavy"),
      ).length,
    ).toBeGreaterThan(0);
  });
});
