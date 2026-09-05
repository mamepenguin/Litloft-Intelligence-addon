"use client";

// AdminTranscriptionSettingsSection — provider switcher injected into
// the core ``/admin/settings`` page through the ``admin-settings-sections``
// slot. Phase 2D writes runtime overrides into the intelligence container's
// data dir; the search-config.yml ship file stays untouched (per hako
// EZSuSEfDHFXkz9MrHdXF9 — addon config is not directly edited by the
// core admin GUI).
//
// API call goes through the addon proxy:
//   GET  /api/addons/intelligence/admin/transcription
//   PUT  /api/addons/intelligence/admin/transcription
//
// On success the intelligence side touches the core's restart_pending
// sentinel via /api/internal/restart-pending so the existing
// RestartBanner picks up "restart required" without this component
// duplicating that UI.

import { useCallback, useEffect, useMemo, useState } from "react";
import { Button } from "@/components/Button";
import { useTranslations } from "next-intl";

const ENDPOINT = "/api/addons/intelligence/admin/transcription";

const PROVIDER_KEY_ENV: Record<string, string | null> = {
  whisper_local: null,
  openai_compatible: "OPENAI_API_KEY",
  deepgram: "DEEPGRAM_API_KEY",
  elevenlabs_scribe: "ELEVENLABS_API_KEY",
  assemblyai: "ASSEMBLYAI_API_KEY",
  gemini: "GEMINI_API_KEY",
};

interface SearchConfigSummary {
  whisper_local?: { model?: string };
  openai_compatible?: { model?: string; base_url?: string };
  deepgram?: { model?: string };
  elevenlabs_scribe?: { model_id?: string };
  assemblyai?: { model?: string };
  gemini?: { model?: string; output_language?: string };
}

interface TranscriptionPayload {
  provider: string;
  language_hint: string;
  hotwords: string[];
  available_providers: string[];
  api_keys_present: Record<string, boolean>;
  overrides_present: boolean;
  search_config_summary: SearchConfigSummary;
}

interface FetchError {
  status: number;
  detail: string;
}

async function fetchConfig(): Promise<TranscriptionPayload> {
  const resp = await fetch(ENDPOINT, { method: "GET" });
  if (!resp.ok) {
    const text = await resp.text().catch(() => "");
    throw {
      status: resp.status,
      detail: text || `HTTP ${resp.status}`,
    } as FetchError;
  }
  return (await resp.json()) as TranscriptionPayload;
}

async function saveConfig(payload: {
  provider: string;
  language_hint: string;
  hotwords: string[];
}): Promise<void> {
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

function describeSubconfig(
  provider: string,
  summary: SearchConfigSummary,
): string {
  const block = (summary as Record<string, Record<string, string>>)[provider];
  if (!block) return "—";
  return Object.entries(block)
    .map(([k, v]) => `${k}=${v}`)
    .join(", ");
}

export default function AdminTranscriptionSettingsSection(): React.ReactElement {
  const t = useTranslations("settings.transcription");
  const [data, setData] = useState<TranscriptionPayload | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [savedOk, setSavedOk] = useState(false);
  const [saving, setSaving] = useState(false);

  const [provider, setProvider] = useState("");
  const [languageHint, setLanguageHint] = useState("");
  const [hotwordsText, setHotwordsText] = useState("");

  const [resetting, setResetting] = useState(false);
  const [resetError, setResetError] = useState<string | null>(null);
  const [resetOk, setResetOk] = useState(false);

  const reload = useCallback(() => {
    return fetchConfig().then((payload) => {
      setData(payload);
      setProvider(payload.provider);
      setLanguageHint(payload.language_hint || "");
      setHotwordsText((payload.hotwords ?? []).join("\n"));
    });
  }, []);

  useEffect(() => {
    let cancelled = false;
    fetchConfig()
      .then((payload) => {
        if (cancelled) return;
        setData(payload);
        setProvider(payload.provider);
        setLanguageHint(payload.language_hint || "");
        setHotwordsText((payload.hotwords ?? []).join("\n"));
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
    // Mount-only fetch
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const apiEnv = useMemo(
    () => (provider ? PROVIDER_KEY_ENV[provider] ?? null : null),
    [provider],
  );
  const apiKeyMissing = useMemo(() => {
    if (!data || !apiEnv) return false;
    return data.api_keys_present[provider] === false;
  }, [data, apiEnv, provider]);

  const handleSave = useCallback(async () => {
    setSaveError(null);
    setSavedOk(false);
    setResetOk(false);
    setSaving(true);
    const hotwords = hotwordsText
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean);
    try {
      await saveConfig({
        provider,
        language_hint: languageHint,
        hotwords,
      });
      setSavedOk(true);
      // Refresh so overrides_present flips to true and the banner appears.
      await reload();
    } catch (err: unknown) {
      const detail =
        (err as FetchError | undefined)?.detail ?? t("saveFailed");
      setSaveError(detail);
    } finally {
      setSaving(false);
    }
  }, [provider, languageHint, hotwordsText, t, reload]);

  const handleReset = useCallback(async () => {
    setResetError(null);
    setResetOk(false);
    setSavedOk(false);
    setResetting(true);
    try {
      await resetConfig();
      // Pull the file-side baseline back into the form.
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
          data-testid="overrides-banner"
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
              <span className="text-xs text-accent-teal">{t("resetSuccess")}</span>
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
          {data.available_providers.map((name) => {
            const envKey = PROVIDER_KEY_ENV[name];
            const present = data.api_keys_present[name];
            const missing = envKey !== null && present === false;
            return (
              <label
                key={name}
                className="flex items-start gap-3 rounded-lg border border-bg-border p-3 hover:border-text-muted"
              >
                <input
                  type="radio"
                  name="transcription-provider"
                  value={name}
                  checked={provider === name}
                  onChange={() => setProvider(name)}
                  className="mt-1"
                />
                <span className="flex-1">
                  <span className="block text-sm text-text-primary">
                    {t(`providers.${name}`)}
                    {name === "whisper_local" || name === "openai_compatible" ? (
                      <span className="ml-1 text-text-muted">†</span>
                    ) : null}
                  </span>
                  <span className="mt-1 block text-xs text-text-muted">
                    {describeSubconfig(name, data.search_config_summary)}
                  </span>
                  {envKey !== null && (
                    <span className="mt-1 block text-xs">
                      {envKey}:{" "}
                      <span
                        className={
                          present ? "text-accent-teal" : "text-danger"
                        }
                      >
                        {present ? t("apiKeyPresent") : t("apiKeyMissing")}
                      </span>
                    </span>
                  )}
                  {missing && provider === name && (
                    <span className="mt-1 block text-xs text-danger">
                      {t("providerNeedsKey", { env: envKey })}
                    </span>
                  )}
                </span>
              </label>
            );
          })}
        </div>
      </fieldset>

      <div className="mb-4">
        <label className="block">
          <span className="text-sm font-medium text-text-primary">
            {t("languageHint")}
          </span>
          <input
            type="text"
            value={languageHint}
            onChange={(e) => setLanguageHint(e.target.value)}
            placeholder="ja"
            className="mt-1 block w-40 rounded-lg border border-bg-border bg-bg-card px-2 py-1 text-sm text-text-primary"
          />
        </label>
        <p className="mt-1 text-xs text-text-muted">
          {t("languageHintHelp")}
        </p>
      </div>

      <div className="mb-4">
        <label className="block">
          <span className="text-sm font-medium text-text-primary">
            {t("hotwords")}
          </span>
          <textarea
            value={hotwordsText}
            onChange={(e) => setHotwordsText(e.target.value)}
            rows={4}
            className="mt-1 block w-full rounded-lg border border-bg-border bg-bg-card px-2 py-1 font-mono text-sm text-text-primary"
          />
        </label>
        <p className="mt-1 text-xs text-text-muted">
          {t("hotwordsHelp")}
        </p>
      </div>

      <div className="flex items-center gap-3">
        <Button
          type="button"
          variant="primary"
          onClick={handleSave}
          disabled={saving || apiKeyMissing}
        >
          {saving ? t("saving") : t("save")}
        </Button>
        {savedOk && (
          <span className="text-xs text-accent-teal">{t("saved")}</span>
        )}
        {saveError && (
          <span className="text-xs text-danger">{saveError}</span>
        )}
      </div>
    </section>
  );
}
