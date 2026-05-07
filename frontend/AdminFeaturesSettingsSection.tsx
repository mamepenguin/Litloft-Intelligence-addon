"use client";

// AdminFeaturesSettingsSection — toggles the eight ``features.*``
// flags from the admin GUI. Save writes
// ``/intelligence-data/features-overrides.json`` (Phase 2D pattern).
// Reset deletes the file so search-config.yml becomes authoritative.
// Restart is required for changes to take effect.

import { useCallback, useEffect, useState } from "react";
import { useTranslations } from "next-intl";

const ENDPOINT = "/api/addons/intelligence/admin/features";

const TRISTATE_FIELDS = [
  "auto_tags",
  "summaries",
  "detailed_summaries",
  "transcript_refine",
  "vision_describe",
] as const;
const BOOL_FIELDS = ["indexing", "search", "rag"] as const;

type TristateField = (typeof TRISTATE_FIELDS)[number];
type BoolField = (typeof BOOL_FIELDS)[number];

interface FeaturesPayload {
  indexing: boolean;
  search: boolean;
  rag: boolean;
  auto_tags: string;
  summaries: string;
  detailed_summaries: string;
  transcript_refine: string;
  vision_describe: string;
  tristate_values: string[];
  overrides_present: boolean;
}

interface FetchError {
  status: number;
  detail: string;
}

async function fetchConfig(): Promise<FeaturesPayload> {
  const resp = await fetch(ENDPOINT, { method: "GET" });
  if (!resp.ok) {
    const text = await resp.text().catch(() => "");
    throw {
      status: resp.status,
      detail: text || `HTTP ${resp.status}`,
    } as FetchError;
  }
  return (await resp.json()) as FeaturesPayload;
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

export default function AdminFeaturesSettingsSection(): React.ReactElement {
  const t = useTranslations("settings.features");
  const [data, setData] = useState<FeaturesPayload | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [savedOk, setSavedOk] = useState(false);
  const [saving, setSaving] = useState(false);
  const [resetting, setResetting] = useState(false);
  const [resetError, setResetError] = useState<string | null>(null);
  const [resetOk, setResetOk] = useState(false);

  const [bools, setBools] = useState<Record<BoolField, boolean>>({
    indexing: true,
    search: true,
    rag: false,
  });
  const [tristates, setTristates] = useState<Record<TristateField, string>>({
    auto_tags: "false",
    summaries: "false",
    detailed_summaries: "false",
    transcript_refine: "false",
    vision_describe: "false",
  });

  const reload = useCallback(async () => {
    const payload = await fetchConfig();
    setData(payload);
    setBools({
      indexing: payload.indexing,
      search: payload.search,
      rag: payload.rag,
    });
    setTristates({
      auto_tags: payload.auto_tags,
      summaries: payload.summaries,
      detailed_summaries: payload.detailed_summaries,
      transcript_refine: payload.transcript_refine,
      vision_describe: payload.vision_describe,
    });
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
      await saveConfig({ ...bools, ...tristates });
      setSavedOk(true);
      await reload();
    } catch (err: unknown) {
      const detail =
        (err as FetchError | undefined)?.detail ?? t("saveFailed");
      setSaveError(detail);
    } finally {
      setSaving(false);
    }
  }, [bools, tristates, t, reload]);

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
      <section className="rounded-xl border border-bg-border bg-bg-card p-4">
        <h2 className="mb-2 text-base font-semibold text-text-primary">
          {t("title")}
        </h2>
        <p className="text-xs text-danger">{loadError}</p>
      </section>
    );
  }

  if (!loaded || !data) {
    return (
      <section className="rounded-xl border border-bg-border bg-bg-card p-4">
        <h2 className="mb-2 text-base font-semibold text-text-primary">
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
          data-testid="features-overrides-banner"
          className="mb-4 rounded-md border border-accent-amber bg-bg-elevated p-3"
        >
          <p className="text-sm text-text-primary">{t("overridesActive")}</p>
          <p className="mt-1 text-xs text-text-muted">{t("overridesHelp")}</p>
          <div className="mt-2 flex items-center gap-3">
            <button
              type="button"
              onClick={handleReset}
              disabled={resetting}
              className="rounded-md border border-bg-border bg-bg-card px-3 py-1 text-sm text-text-primary hover:bg-bg-light disabled:cursor-not-allowed disabled:opacity-50"
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
        {BOOL_FIELDS.map((name) => (
          <label key={name} className="flex items-start gap-3">
            <input
              type="checkbox"
              checked={bools[name]}
              onChange={(e) =>
                setBools((b) => ({ ...b, [name]: e.target.checked }))
              }
              className="mt-1"
            />
            <span className="flex-1">
              <span className="block text-sm font-medium text-text-primary">
                {t(`fields.${name}.label`)}
              </span>
              <span className="block text-xs text-text-muted">
                {t(`fields.${name}.help`)}
              </span>
            </span>
          </label>
        ))}

        {TRISTATE_FIELDS.map((name) => (
          <fieldset key={name} className="rounded-md border border-bg-border p-3">
            <legend className="px-1 text-sm font-medium text-text-primary">
              {t(`fields.${name}.label`)}
            </legend>
            <p className="mb-2 text-xs text-text-muted">
              {t(`fields.${name}.help`)}
            </p>
            <div className="flex flex-wrap gap-3">
              {data.tristate_values.map((value) => (
                <label
                  key={value}
                  className="inline-flex items-center gap-1 text-sm text-text-primary"
                >
                  <input
                    type="radio"
                    name={`feature-${name}`}
                    value={value}
                    checked={tristates[name] === value}
                    onChange={() =>
                      setTristates((s) => ({ ...s, [name]: value }))
                    }
                  />
                  {t(`tristate.${value}`)}
                </label>
              ))}
            </div>
          </fieldset>
        ))}
      </div>

      <div className="mt-4 flex items-center gap-3">
        <button
          type="button"
          onClick={handleSave}
          disabled={saving}
          className="rounded-md border border-bg-border bg-bg-card px-3 py-1 text-sm text-text-primary hover:bg-bg-light disabled:cursor-not-allowed disabled:opacity-50"
        >
          {saving ? t("saving") : t("save")}
        </button>
        {savedOk && <span className="text-xs text-success">{t("saved")}</span>}
        {saveError && <span className="text-xs text-danger">{saveError}</span>}
      </div>
    </section>
  );
}
