"use client";

import { useCallback, useEffect, useState } from "react";
import { Button } from "@/components/Button";
import { useTranslations } from "next-intl";
import { errorDetail, FEATURES_ENDPOINT, RequestError } from "./llm-settings/api";
import { ApplyMarker } from "./llm-settings/Notices";

const ENDPOINT = FEATURES_ENDPOINT;

const TRISTATE_FIELDS = [
  "auto_tags",
  "summaries",
  "detailed_summaries",
  "transcript_refine",
  "vision_describe",
  "retrieval_keywords",
  "chapter_suggestions",
  "video_visual_index",
] as const;
const BOOL_FIELDS = ["indexing", "search", "rag"] as const;
const FIELDS = [...BOOL_FIELDS, ...TRISTATE_FIELDS] as const;

type TristateField = (typeof TRISTATE_FIELDS)[number];
type BoolField = (typeof BOOL_FIELDS)[number];
type Field = (typeof FIELDS)[number];
type Modes = Record<Field, string>;

interface FeaturesPayload extends Record<BoolField, boolean>, Record<TristateField, string> {
  tristate_values: string[];
  overrides_present: boolean;
}

type Outcome = "saved" | "resetSuccess" | null;

const SELECT_CLASS =
  "w-full rounded-2xl border border-warm-silver/40 bg-bg-card px-3 py-2 text-sm text-text-primary " +
  "focus:outline-none focus:ring-2 focus:ring-focus-ring";

async function request(method: "GET" | "PUT" | "DELETE", body?: unknown): Promise<unknown> {
  const resp = await fetch(ENDPOINT, {
    method,
    ...(body === undefined
      ? {}
      : { headers: { "content-type": "application/json" }, body: JSON.stringify(body) }),
  });
  if (!resp.ok) {
    let detail = `HTTP ${resp.status}`;
    try {
      const parsed = await resp.json();
      if (typeof parsed?.detail === "string") detail = parsed.detail;
    } catch {
      // A body that is not JSON carries no detail worth showing.
    }
    throw new RequestError(resp.status, detail);
  }
  return resp.json().catch(() => null);
}

function isBool(field: Field): field is BoolField {
  return (BOOL_FIELDS as readonly string[]).includes(field);
}

function modesOf(payload: FeaturesPayload): Modes {
  return Object.fromEntries(
    FIELDS.map((f) => [f, isBool(f) ? String(payload[f]) : payload[f]]),
  ) as Modes;
}

function modesBody(modes: Modes): Record<string, unknown> {
  return Object.fromEntries(FIELDS.map((f) => [f, isBool(f) ? modes[f] === "true" : modes[f]]));
}

export default function AdminFeaturesSettingsSection(): React.ReactElement {
  const t = useTranslations("settings.features");
  const [data, setData] = useState<FeaturesPayload | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [modes, setModes] = useState<Modes | null>(null);
  const [busy, setBusy] = useState<"save" | "reset" | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [reloadError, setReloadError] = useState<string | null>(null);
  const [resetError, setResetError] = useState<string | null>(null);
  const [outcome, setOutcome] = useState<Outcome>(null);

  const reload = useCallback(async () => {
    const payload = (await request("GET")) as FeaturesPayload;
    setData(payload);
    setModes(modesOf(payload));
  }, []);

  useEffect(() => {
    let cancelled = false;
    reload().catch((err: unknown) => {
      if (!cancelled) setLoadError(errorDetail(err, t("loadFailed")));
    });
    return () => {
      cancelled = true;
    };
    // `t` is left out: loading runs once, not whenever the translator is rebuilt.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reload]);

  const handleSave = useCallback(async () => {
    if (!modes) return;
    setSaveError(null);
    setReloadError(null);
    setOutcome(null);
    setBusy("save");
    try {
      await request("PUT", modesBody(modes));
      setOutcome("saved");
      try {
        await reload();
      } catch {
        setReloadError(t("reloadFailed"));
      }
    } catch (err: unknown) {
      setSaveError(errorDetail(err, t("saveFailed")));
    } finally {
      setBusy(null);
    }
  }, [modes, reload, t]);

  const handleReset = useCallback(async () => {
    setResetError(null);
    setOutcome(null);
    setBusy("reset");
    try {
      await request("DELETE");
      await reload();
      setOutcome("resetSuccess");
    } catch (err: unknown) {
      setResetError(errorDetail(err, t("resetFailed")));
    } finally {
      setBusy(null);
    }
  }, [reload, t]);

  if (loadError) {
    return (
      <section className="rounded-xl border border-bg-border bg-bg-card p-6">
        <h2 className="mb-2 text-lg font-semibold text-text-primary">{t("title")}</h2>
        <p className="text-xs text-danger">{loadError}</p>
      </section>
    );
  }

  if (!data || !modes) {
    return (
      <section className="rounded-xl border border-bg-border bg-bg-card p-6">
        <h2 className="mb-2 text-lg font-semibold text-text-primary">{t("title")}</h2>
      </section>
    );
  }

  const headClass =
    "sticky top-0 z-10 border-b border-bg-border bg-bg-card px-4 py-3 text-left align-top font-semibold";

  return (
    <section className="rounded-xl border border-bg-border bg-bg-card p-4">
      <h2 className="mb-2 text-base font-semibold text-text-primary">{t("title")}</h2>
      <p className="mb-4 text-xs text-text-muted">{t("description")}</p>

      {data.overrides_present && (
        <div
          role="status"
          data-testid="features-overrides-banner"
          className="mb-4 rounded-xl border border-accent-amber bg-bg-elevated p-3"
        >
          <p className="text-sm text-text-primary">{t("overridesActive")}</p>
          <p className="mt-1 text-xs text-text-muted">{t("overridesHelp")}</p>
          <div className="mt-2 flex items-center gap-3">
            <Button onClick={handleReset} disabled={busy !== null}>
              {busy === "reset" ? t("resetting") : t("reset")}
            </Button>
            {resetError && <span className="text-xs text-danger">{resetError}</span>}
          </div>
        </div>
      )}

      <div className="max-h-[70vh] overflow-auto">
        <table className="min-w-full border-separate border-spacing-0 text-sm text-text-primary">
          <thead>
            <tr>
              <th className={`${headClass} pl-0`}>{t("columns.feature")}</th>
              <th className={headClass}>
                <span className="flex flex-col gap-0.5">
                  {t("columns.mode")}
                  <ApplyMarker when="restart" />
                </span>
              </th>
            </tr>
          </thead>
          <tbody>
            {FIELDS.map((field) => {
              const label = t(`fields.${field}.label`);
              const values = isBool(field) ? ["true", "false"] : data.tristate_values;
              return (
                <tr key={field}>
                  <td className="border-b border-bg-border py-3 pr-4 align-top">
                    <span className="block font-medium">{label}</span>
                    <span className="mt-1 block text-xs text-text-muted">
                      {t(`fields.${field}.help`)}
                    </span>
                  </td>
                  <td className="min-w-40 border-b border-bg-border px-4 py-3 align-top">
                    <select
                      aria-label={t("modeLabel", { feature: label })}
                      value={modes[field]}
                      onChange={(e) => setModes({ ...modes, [field]: e.target.value })}
                      className={SELECT_CLASS}
                    >
                      {values.map((value) => (
                        <option key={value} value={value}>
                          {isBool(field) ? t(`bool.${value}`) : t(`tristate.${value}`)}
                        </option>
                      ))}
                    </select>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="mt-4 text-xs text-text-muted">{t("profilesNote")}</p>

      <div className="mt-4 flex flex-wrap items-center gap-3">
        <Button variant="primary" onClick={handleSave} disabled={busy !== null}>
          {busy === "save" ? t("saving") : t("save")}
        </Button>
        {outcome && (
          <span role="status" className="text-xs text-accent-teal">
            {t(outcome)}
          </span>
        )}
        {saveError && <span className="text-xs text-danger">{saveError}</span>}
        {reloadError && <span className="text-xs text-danger">{reloadError}</span>}
      </div>
    </section>
  );
}
