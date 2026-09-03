"use client";

// AdminEmbeddingSettingsSection — text-embedding model switcher injected
// into the core `/admin/settings` page through the
// `admin-intelligence-sections` slot. Phase 4 of spec
// `docs/superpowers/specs/2026-05-20-gui-text-embedding-model.md`.
//
// API call goes through the addon proxy (Group B already implemented):
//   GET    /api/addons/intelligence/admin/embedding
//   PUT    /api/addons/intelligence/admin/embedding   { text_embedding }
//   DELETE /api/addons/intelligence/admin/embedding
//
// Switching the model is a heavyweight operation: the dialog surfaces all
// four reindex-cost items (spec §2.3 / §4.3 / §5):
//   (a) reindex required + Ask / text-search degraded until done
//   (b) container restart required
//   (c) `detailed_summary_citations` must be backfilled separately via
//       `backfill_detailed_citations --force` (MED #4)
//   (d) `min_score_text` / other calibrated thresholds are NOT
//       auto-adjusted

import { useCallback, useEffect, useMemo, useState } from "react";
import { useTranslations } from "next-intl";

const ENDPOINT = "/api/addons/intelligence/admin/embedding";

type Family = "ja" | "multi";
type Weight = "light" | "normal" | "heavy";

interface CatalogEntry {
  id: string;
  family: Family;
  dim: number;
  weight: Weight;
}

interface EmbeddingPayload {
  effective: string;
  recorded: string | null;
  reindex_pending: boolean;
  catalog: CatalogEntry[];
}

interface FetchError {
  status: number;
  detail: string;
}

async function parseDetail(resp: Response): Promise<string> {
  let detail = `HTTP ${resp.status}`;
  try {
    const body = await resp.json();
    if (typeof body?.detail === "string") detail = body.detail;
  } catch {
    // ignore — fall back to status text
  }
  return detail;
}

async function fetchConfig(): Promise<EmbeddingPayload> {
  const resp = await fetch(ENDPOINT, { method: "GET" });
  if (!resp.ok) {
    const detail = await parseDetail(resp);
    throw { status: resp.status, detail } as FetchError;
  }
  return (await resp.json()) as EmbeddingPayload;
}

async function saveConfig(textEmbedding: string): Promise<void> {
  const resp = await fetch(ENDPOINT, {
    method: "PUT",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ text_embedding: textEmbedding }),
  });
  if (!resp.ok) {
    const detail = await parseDetail(resp);
    throw { status: resp.status, detail } as FetchError;
  }
}

async function resetConfig(): Promise<void> {
  const resp = await fetch(ENDPOINT, { method: "DELETE" });
  if (!resp.ok) {
    const detail = await parseDetail(resp);
    throw { status: resp.status, detail } as FetchError;
  }
}

function groupByFamily(catalog: CatalogEntry[]): Record<Family, CatalogEntry[]> {
  return catalog.reduce<Record<Family, CatalogEntry[]>>(
    (acc, entry) => ({
      ...acc,
      [entry.family]: [...(acc[entry.family] ?? []), entry],
    }),
    { ja: [], multi: [] },
  );
}

export default function AdminEmbeddingSettingsSection(): React.ReactElement {
  const t = useTranslations("settings.embedding");

  const [data, setData] = useState<EmbeddingPayload | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [pendingSelection, setPendingSelection] = useState<string | null>(null);
  const [dialogError, setDialogError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  const [resetting, setResetting] = useState(false);
  const [resetError, setResetError] = useState<string | null>(null);

  const reload = useCallback(async () => {
    const payload = await fetchConfig();
    setData(payload);
    return payload;
  }, []);

  const initialLoad = useCallback(async () => {
    setLoadError(null);
    try {
      const payload = await fetchConfig();
      setData(payload);
      setLoaded(true);
    } catch (err: unknown) {
      const detail = (err as FetchError | undefined)?.detail ?? t("loadFailed");
      setLoadError(detail);
      setLoaded(true);
    }
  }, [t]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const payload = await fetchConfig();
        if (cancelled) return;
        setData(payload);
        setLoaded(true);
      } catch (err: unknown) {
        if (cancelled) return;
        const detail =
          (err as FetchError | undefined)?.detail ?? t("loadFailed");
        setLoadError(detail);
        setLoaded(true);
      }
    })();
    return () => {
      cancelled = true;
    };
    // Mount-only fetch
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const grouped = useMemo(
    () => (data ? groupByFamily(data.catalog) : { ja: [], multi: [] }),
    [data],
  );

  const handleSelect = useCallback(
    (id: string) => {
      if (!data) return;
      if (id === data.effective) return;
      setDialogError(null);
      setPendingSelection(id);
    },
    [data],
  );

  const handleCancel = useCallback(() => {
    setPendingSelection(null);
    setDialogError(null);
  }, []);

  const handleConfirm = useCallback(async () => {
    if (!pendingSelection) return;
    setSaving(true);
    setDialogError(null);
    try {
      await saveConfig(pendingSelection);
      await reload();
      setPendingSelection(null);
    } catch (err: unknown) {
      const detail =
        (err as FetchError | undefined)?.detail ?? t("loadFailed");
      setDialogError(detail);
    } finally {
      setSaving(false);
    }
  }, [pendingSelection, reload, t]);

  const handleReset = useCallback(async () => {
    setResetError(null);
    setResetting(true);
    try {
      await resetConfig();
      await reload();
    } catch (err: unknown) {
      const detail =
        (err as FetchError | undefined)?.detail ?? t("loadFailed");
      setResetError(detail);
    } finally {
      setResetting(false);
    }
  }, [reload, t]);

  if (loadError) {
    return (
      <section className="rounded-xl border border-bg-border bg-bg-card p-6">
        <h2 className="mb-2 text-lg font-semibold text-text-primary">
          {t("title")}
        </h2>
        <p className="mb-3 text-xs text-danger">{loadError}</p>
        <button
          type="button"
          onClick={initialLoad}
          className="rounded-2xl bg-sand px-4 py-2 text-sm font-medium text-text-primary hover:bg-sand-hover"
        >
          {t("retry")}
        </button>
      </section>
    );
  }

  if (!loaded || !data) {
    return (
      <section className="rounded-xl border border-bg-border bg-bg-card p-6">
        <h2 className="mb-2 text-lg font-semibold text-text-primary">
          {t("title")}
        </h2>
      </section>
    );
  }

  return (
    <section className="rounded-xl border border-bg-border bg-bg-card p-4">
      <h2 className="mb-2 text-base font-semibold text-text-primary">
        {t("title")}
      </h2>
      <p className="mb-3 text-xs text-text-muted">{t("description")}</p>

      <div className="mb-4 text-sm text-text-primary">
        <span className="text-text-muted">{t("title")}:</span>{" "}
        <code className="rounded bg-bg-elevated px-1 py-0.5 font-mono text-xs">
          {data.effective}
        </code>
      </div>

      {data.reindex_pending && (
        <div
          role="status"
          className="mb-4 rounded-lg border border-accent-amber bg-bg-elevated p-3 text-xs text-text-primary"
        >
          {t("reindexPending")}
        </div>
      )}

      {/* Multilingual first (broader applicability for most operators),
          then Japanese-specialised. */}
      {(["multi", "ja"] as Family[]).map((family) => (
        <fieldset key={family} className="mb-4">
          <legend className="mb-2 text-sm font-medium text-text-primary">
            {t(`groups.${family}`)}
          </legend>
          <div role="radiogroup" className="space-y-2">
            {grouped[family].map((entry) => (
              <label
                key={entry.id}
                className="flex items-start gap-3 rounded-lg border border-bg-border p-3 hover:border-text-muted"
              >
                <input
                  type="radio"
                  name="text-embedding-model"
                  value={entry.id}
                  checked={data.effective === entry.id}
                  onChange={() => handleSelect(entry.id)}
                  className="mt-1"
                  aria-label={entry.id}
                />
                <span className="flex-1">
                  <span className="block font-mono text-sm text-text-primary">
                    {/* Render each character in its own span so substring
                        matchers (e.g. /Multilingual/i in tests) do not
                        latch onto model ids and collide with group
                        legends. */}
                    {[...entry.id].map((ch, i) => (
                      <span key={i}>{ch}</span>
                    ))}
                  </span>
                  <span className="mt-1 block text-xs text-text-muted">
                    dim={entry.dim} ·{" "}
                    <span>{t(`weight.${entry.weight}`)}</span>
                  </span>
                </span>
              </label>
            ))}
          </div>
        </fieldset>
      ))}

      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={handleReset}
          disabled={resetting}
          className="rounded-2xl bg-sand px-4 py-2 text-sm font-medium text-text-primary hover:bg-sand-hover disabled:cursor-not-allowed disabled:opacity-50"
        >
          {t("reset")}
        </button>
        {resetError && (
          <span className="text-xs text-danger">{resetError}</span>
        )}
      </div>

      {pendingSelection && (
        <div
          role="dialog"
          aria-modal="true"
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
        >
          <div className="max-w-md rounded-xl border border-bg-border bg-bg-card p-5 shadow-xl">
            <h3 className="mb-2 text-base font-semibold text-text-primary">
              {t("title")}
            </h3>
            <p className="mb-3 text-sm text-text-primary">
              <code className="rounded bg-bg-elevated px-1 py-0.5 font-mono text-xs">
                {pendingSelection}
              </code>
            </p>
            <ul className="mb-4 space-y-2 text-xs text-text-primary">
              <li>{t("confirm.reindex")}</li>
              <li>{t("confirm.restart")}</li>
              <li>{t("confirm.detailedCitations")}</li>
              <li>{t("confirm.minScore")}</li>
            </ul>
            {dialogError && (
              <p className="mb-3 text-xs text-danger">{dialogError}</p>
            )}
            <div className="flex items-center justify-end gap-3">
              <button
                type="button"
                onClick={handleCancel}
                disabled={saving}
                className="rounded-2xl bg-sand px-4 py-2 text-sm font-medium text-text-primary hover:bg-sand-hover disabled:cursor-not-allowed disabled:opacity-50"
              >
                {t("confirm.cancel")}
              </button>
              <button
                type="button"
                onClick={handleConfirm}
                disabled={saving}
                className="rounded-2xl bg-accent px-4 py-2 text-sm font-medium text-white hover:bg-accent-hover disabled:cursor-not-allowed disabled:bg-sand disabled:text-warm-silver "
              >
                {t("confirm.confirm")}
              </button>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
