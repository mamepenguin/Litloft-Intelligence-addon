"use client";

// AdminRAGSettingsSection — toggles the two RAG sub-feature gates
// (``personal_history.enabled`` and ``category_expansion.enabled``)
// without touching the rest of the heavily-tuned ``rag`` block.
// Other rag.* knobs (top_k, max_context_chars*, hierarchical
// retrieval params, …) stay file-only because operators almost
// never need to change them.

import { useCallback, useEffect, useState } from "react";
import { useTranslations } from "next-intl";

const ENDPOINT = "/api/addons/intelligence/admin/rag";

interface RagPayload {
  personal_history_enabled: boolean;
  category_expansion_enabled: boolean;
  overrides_present: boolean;
}

interface FetchError {
  status: number;
  detail: string;
}

async function fetchConfig(): Promise<RagPayload> {
  const resp = await fetch(ENDPOINT, { method: "GET" });
  if (!resp.ok) {
    const text = await resp.text().catch(() => "");
    throw {
      status: resp.status,
      detail: text || `HTTP ${resp.status}`,
    } as FetchError;
  }
  return (await resp.json()) as RagPayload;
}

async function saveConfig(payload: Record<string, unknown>): Promise<void> {
  const resp = await fetch(ENDPOINT, {
    method: "PUT",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!resp.ok) {
    let detail = `HTTP ${resp.status}`;
    try {
      const body = await resp.json();
      if (typeof body?.detail === "string") detail = body.detail;
    } catch {
      // ignore parse failures and use the HTTP status text
    }
    throw { status: resp.status, detail } as FetchError;
  }
}

async function resetConfig(): Promise<void> {
  const resp = await fetch(ENDPOINT, { method: "DELETE" });
  if (!resp.ok) {
    let detail = `HTTP ${resp.status}`;
    try {
      const body = await resp.json();
      if (typeof body?.detail === "string") detail = body.detail;
    } catch {
      // ignore parse failures and use the HTTP status text
    }
    throw { status: resp.status, detail } as FetchError;
  }
}

export default function AdminRAGSettingsSection(): React.ReactElement {
  const t = useTranslations("settings.rag");
  const [data, setData] = useState<RagPayload | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [savedOk, setSavedOk] = useState(false);
  const [saving, setSaving] = useState(false);
  const [resetting, setResetting] = useState(false);
  const [resetError, setResetError] = useState<string | null>(null);
  const [resetOk, setResetOk] = useState(false);

  const [personalHistory, setPersonalHistory] = useState(false);
  const [categoryExpansion, setCategoryExpansion] = useState(false);

  const reload = useCallback(async () => {
    const payload = await fetchConfig();
    setData(payload);
    setPersonalHistory(payload.personal_history_enabled);
    setCategoryExpansion(payload.category_expansion_enabled);
  }, []);

  useEffect(() => {
    let cancelled = false;
    reload()
      .then(() => {
        if (cancelled) return;
        setLoaded(true);
        setLoadError(null);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        const detail =
          (err as FetchError | undefined)?.detail ?? t("loadFailed");
        setLoadError(detail);
        setLoaded(true);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleSave = useCallback(async () => {
    setSaveError(null);
    setSavedOk(false);
    setResetOk(false);
    setSaving(true);
    try {
      await saveConfig({
        personal_history_enabled: personalHistory,
        category_expansion_enabled: categoryExpansion,
      });
      setSavedOk(true);
      await reload();
    } catch (err: unknown) {
      const detail =
        (err as FetchError | undefined)?.detail ?? t("saveFailed");
      setSaveError(detail);
    } finally {
      setSaving(false);
    }
  }, [personalHistory, categoryExpansion, t, reload]);

  const handleReset = useCallback(async () => {
    setResetError(null);
    setResetOk(false);
    setSavedOk(false);
    setResetting(true);
    try {
      await resetConfig();
      await reload();
      setResetOk(true);
    } catch (err: unknown) {
      const detail =
        (err as FetchError | undefined)?.detail ?? t("resetFailed");
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
        <p className="text-xs text-danger">{loadError}</p>
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
      <p className="mb-4 text-xs text-text-muted">{t("description")}</p>

      {data.overrides_present && (
        <div
          role="status"
          data-testid="rag-overrides-banner"
          className="mb-4 rounded-lg border border-accent-amber bg-bg-elevated p-3"
        >
          <p className="text-sm text-text-primary">{t("overridesActive")}</p>
          <p className="mt-1 text-xs text-text-muted">{t("overridesHelp")}</p>
          <div className="mt-2 flex items-center gap-3">
            <button
              type="button"
              onClick={handleReset}
              disabled={resetting}
              className="rounded-2xl bg-sand px-4 py-2 text-sm font-medium text-text-primary hover:bg-sand-hover disabled:cursor-not-allowed disabled:opacity-50"
            >
              {resetting ? t("resetting") : t("reset")}
            </button>
            {resetOk && (
              <span className="text-xs text-success">{t("resetSuccess")}</span>
            )}
            {resetError && (
              <span className="text-xs text-danger">{resetError}</span>
            )}
          </div>
        </div>
      )}

      <div className="space-y-3">
        <label className="flex items-start gap-3">
          <input
            type="checkbox"
            checked={personalHistory}
            onChange={(e) => setPersonalHistory(e.target.checked)}
            className="mt-1"
          />
          <span className="flex-1">
            <span className="block text-sm font-medium text-text-primary">
              {t("personalHistory.label")}
            </span>
            <span className="block text-xs text-text-muted">
              {t("personalHistory.help")}
            </span>
          </span>
        </label>

        <label className="flex items-start gap-3">
          <input
            type="checkbox"
            checked={categoryExpansion}
            onChange={(e) => setCategoryExpansion(e.target.checked)}
            className="mt-1"
          />
          <span className="flex-1">
            <span className="block text-sm font-medium text-text-primary">
              {t("categoryExpansion.label")}
            </span>
            <span className="block text-xs text-text-muted">
              {t("categoryExpansion.help")}
            </span>
          </span>
        </label>
      </div>

      <div className="mt-4 flex items-center gap-3">
        <button
          type="button"
          onClick={handleSave}
          disabled={saving}
          className="rounded-2xl bg-accent px-4 py-2 text-sm font-medium text-white hover:bg-accent-hover disabled:cursor-not-allowed disabled:bg-sand disabled:text-warm-silver"
        >
          {saving ? t("saving") : t("save")}
        </button>
        {savedOk && <span className="text-xs text-success">{t("saved")}</span>}
        {saveError && <span className="text-xs text-danger">{saveError}</span>}
      </div>
    </section>
  );
}
