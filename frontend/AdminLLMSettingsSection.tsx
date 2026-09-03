"use client";

// AdminLLMSettingsSection — provider switcher and model picker for the
// LLM that drives auto_tags / summaries / RAG / vision_describe. Save
// writes ``/intelligence-data/llm-overrides.json``; reset deletes it
// so search-config.yml's ``llm`` section becomes authoritative again.
// API key (LLM_API_KEY env var) is shown as a presence flag only —
// secrets do not pass through this UI.

import { useCallback, useEffect, useState } from "react";
import { useTranslations } from "next-intl";

const ENDPOINT = "/api/addons/intelligence/admin/llm";

interface LLMPayload {
  provider: string;
  base_url: string;
  model: string;
  output_language: string;
  vision_model: string;
  available_providers: string[];
  available_output_languages: string[];
  api_key_present: boolean;
  api_key_env_var: string;
  overrides_present: boolean;
}

interface FetchError {
  status: number;
  detail: string;
}

async function fetchConfig(): Promise<LLMPayload> {
  const resp = await fetch(ENDPOINT, { method: "GET" });
  if (!resp.ok) {
    const text = await resp.text().catch(() => "");
    throw {
      status: resp.status,
      detail: text || `HTTP ${resp.status}`,
    } as FetchError;
  }
  return (await resp.json()) as LLMPayload;
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

export default function AdminLLMSettingsSection(): React.ReactElement {
  const t = useTranslations("settings.llm");
  const [data, setData] = useState<LLMPayload | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [savedOk, setSavedOk] = useState(false);
  const [saving, setSaving] = useState(false);
  const [resetting, setResetting] = useState(false);
  const [resetError, setResetError] = useState<string | null>(null);
  const [resetOk, setResetOk] = useState(false);

  const [provider, setProvider] = useState("disabled");
  const [baseUrl, setBaseUrl] = useState("");
  const [model, setModel] = useState("");
  const [outputLanguage, setOutputLanguage] = useState("auto");
  const [visionModel, setVisionModel] = useState("");

  const reload = useCallback(async () => {
    const payload = await fetchConfig();
    setData(payload);
    setProvider(payload.provider);
    setBaseUrl(payload.base_url);
    setModel(payload.model);
    setOutputLanguage(payload.output_language);
    setVisionModel(payload.vision_model);
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
        provider,
        base_url: baseUrl,
        model,
        output_language: outputLanguage,
        vision_model: visionModel,
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
  }, [provider, baseUrl, model, outputLanguage, visionModel, t, reload]);

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

  const apiKeyMissingForCloud =
    provider !== "disabled" && provider !== "ollama" && !data.api_key_present;

  return (
    <section className="rounded-xl border border-bg-border bg-bg-card p-4">
      <h2 className="mb-2 text-base font-semibold text-text-primary">
        {t("title")}
      </h2>
      <p className="mb-4 text-xs text-text-muted">{t("description")}</p>

      {data.overrides_present && (
        <div
          role="status"
          data-testid="llm-overrides-banner"
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

      <fieldset className="mb-4">
        <legend className="mb-2 text-sm font-medium text-text-primary">
          {t("provider")}
        </legend>
        <div className="space-y-2">
          {data.available_providers.map((name) => (
            <label
              key={name}
              className="flex items-start gap-3 rounded-lg border border-bg-border p-3 hover:border-text-muted"
            >
              <input
                type="radio"
                name="llm-provider"
                value={name}
                checked={provider === name}
                onChange={() => setProvider(name)}
                className="mt-1"
              />
              <span className="flex-1">
                <span className="block text-sm text-text-primary">
                  {t(`providers.${name}`)}
                </span>
                <span className="block text-xs text-text-muted">
                  {t(`providerHelp.${name}`)}
                </span>
              </span>
            </label>
          ))}
        </div>
      </fieldset>

      <div className="mb-3">
        <label className="block">
          <span className="text-sm font-medium text-text-primary">
            {t("baseUrl")}
          </span>
          <input
            type="text"
            value={baseUrl}
            onChange={(e) => setBaseUrl(e.target.value)}
            placeholder="http://host.docker.internal:11434"
            className="mt-1 block w-full rounded-lg border border-bg-border bg-bg-card px-2 py-1 font-mono text-sm text-text-primary"
          />
        </label>
        <p className="mt-1 text-xs text-text-muted">{t("baseUrlHelp")}</p>
      </div>

      <div className="mb-3">
        <label className="block">
          <span className="text-sm font-medium text-text-primary">
            {t("model")}
          </span>
          <input
            type="text"
            value={model}
            onChange={(e) => setModel(e.target.value)}
            placeholder="gemma4:e4b"
            className="mt-1 block w-full rounded-lg border border-bg-border bg-bg-card px-2 py-1 font-mono text-sm text-text-primary"
          />
        </label>
        <p className="mt-1 text-xs text-text-muted">{t("modelHelp")}</p>
      </div>

      <div className="mb-3">
        <label className="block">
          <span className="text-sm font-medium text-text-primary">
            {t("visionModel")}
          </span>
          <input
            type="text"
            value={visionModel}
            onChange={(e) => setVisionModel(e.target.value)}
            placeholder="llava:13b"
            className="mt-1 block w-full rounded-lg border border-bg-border bg-bg-card px-2 py-1 font-mono text-sm text-text-primary"
          />
        </label>
        <p className="mt-1 text-xs text-text-muted">{t("visionModelHelp")}</p>
      </div>

      <div className="mb-3">
        <label className="block">
          <span className="text-sm font-medium text-text-primary">
            {t("outputLanguage")}
          </span>
          <select
            value={outputLanguage}
            onChange={(e) => setOutputLanguage(e.target.value)}
            className="mt-1 block w-40 rounded-lg border border-bg-border bg-bg-card px-2 py-1 text-sm text-text-primary"
          >
            {data.available_output_languages.map((lang) => (
              <option key={lang} value={lang}>
                {t(`outputLanguages.${lang}`)}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="mb-4 rounded-lg border border-bg-border bg-bg-elevated p-3">
        <p className="text-xs">
          {data.api_key_env_var}:{" "}
          <span className={data.api_key_present ? "text-success" : "text-danger"}>
            {data.api_key_present ? t("apiKeyPresent") : t("apiKeyMissing")}
          </span>
        </p>
        <p className="mt-1 text-xs text-text-muted">{t("apiKeyHelp")}</p>
        {apiKeyMissingForCloud && (
          <p className="mt-1 text-xs text-danger">{t("apiKeyRequired")}</p>
        )}
      </div>

      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={handleSave}
          disabled={saving || apiKeyMissingForCloud}
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
