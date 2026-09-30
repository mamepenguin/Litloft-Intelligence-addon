"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Button } from "@/components/Button";
import { useTranslations } from "next-intl";
import {
  errorDetail,
  fetchExposure,
  RequestError,
  type ExposureView,
  type LLMView,
} from "./llm-settings/api";
import {
  loadLLMView,
  saveLLMPatch,
  subscribeLLMSaved,
  withFeatureChoices,
} from "./llm-settings/store";
import FeatureProfileCell, { SELECT_CLASS } from "./llm-settings/FeatureProfileCell";
import { ApplyMarker } from "./llm-settings/Notices";

const ENDPOINT = "/api/addons/intelligence/admin/features";

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

type Outcome = "saved" | "savedRestart" | "savedNow" | "resetSuccess" | null;

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

/** The entries of `next` that differ from `saved`, "" standing for an absent key. */
function changedEntries(
  next: Record<string, string>,
  saved: Record<string, string>,
): Record<string, string> {
  const keys = [...new Set([...Object.keys(next), ...Object.keys(saved)])];
  return Object.fromEntries(
    keys.filter((k) => (next[k] ?? "") !== (saved[k] ?? "")).map((k) => [k, next[k] ?? ""]),
  );
}

function sameRecord(a: Record<string, string>, b: Record<string, string>): boolean {
  return Object.keys(changedEntries(a, b)).length === 0;
}

export default function AdminFeaturesSettingsSection(): React.ReactElement {
  const t = useTranslations("settings.features");
  const [data, setData] = useState<FeaturesPayload | null>(null);
  const [llm, setLLM] = useState<LLMView | null>(null);
  const [exposure, setExposure] = useState<ExposureView | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [modes, setModes] = useState<Modes | null>(null);
  const [assigned, setAssigned] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState<"save" | "reset" | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [reloadError, setReloadError] = useState<string | null>(null);
  const [resetError, setResetError] = useState<string | null>(null);
  const [outcome, setOutcome] = useState<Outcome>(null);

  const applyLLM = useCallback((nextLLM: LLMView | null, nextExposure: ExposureView | null) => {
    setLLM(nextLLM);
    setExposure(nextExposure);
    setAssigned({ ...(nextLLM?.routing.features ?? {}) });
  }, []);

  const reloadLLM = useCallback(async () => {
    const [nextLLM, nextExposure] = await Promise.all([
      loadLLMView().catch(() => null),
      fetchExposure().catch(() => null),
    ]);
    applyLLM(nextLLM, nextExposure);
  }, [applyLLM]);

  const reload = useCallback(async () => {
    const [payload, nextLLM, nextExposure] = await Promise.all([
      request("GET") as Promise<FeaturesPayload>,
      loadLLMView().catch(() => null),
      fetchExposure().catch(() => null),
    ]);
    setData(payload);
    setModes(modesOf(payload));
    applyLLM(nextLLM, nextExposure);
  }, [applyLLM]);

  const routingDraft = useRef<{ llm: LLMView | null; assigned: Record<string, string> }>({
    llm: null,
    assigned: {},
  });
  useEffect(() => {
    routingDraft.current = { llm, assigned };
  });

  useEffect(() => {
    let cancelled = false;
    reload().catch((err: unknown) => {
      if (!cancelled) setLoadError(errorDetail(err, t("loadFailed")));
    });
    const unsubscribe = subscribeLLMSaved((writer) => {
      if (writer === "features") return;
      const current = routingDraft.current;
      const saved = current.llm?.routing.features ?? {};
      if (current.llm && !sameRecord(current.assigned, saved)) return;
      reloadLLM().catch(() => undefined);
    });
    return () => {
      cancelled = true;
      unsubscribe();
    };
    // `t` is left out: loading runs once, not whenever the translator is rebuilt.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reload, reloadLLM]);

  const handleSave = useCallback(async () => {
    if (!data || !modes) return;
    setSaveError(null);
    setReloadError(null);
    setOutcome(null);
    setBusy("save");
    const modesChanged = !sameRecord(modes, modesOf(data));
    const choices = llm ? changedEntries(assigned, llm.routing.features ?? {}) : {};
    const routingChanged = Object.keys(choices).length > 0;
    let routingSaved = false;
    let failure: unknown = null;
    try {
      if (routingChanged) {
        await saveLLMPatch("features", (latest) => withFeatureChoices(latest, choices));
        routingSaved = true;
      }
      if (modesChanged || !routingChanged) await request("PUT", modesBody(modes));
    } catch (err: unknown) {
      failure = err;
    }
    if (failure === null || routingSaved) {
      try {
        await reload();
      } catch {
        setReloadError(t("reloadFailed"));
      }
    }
    if (failure === null) {
      setOutcome(modesChanged ? "savedRestart" : routingChanged ? "savedNow" : "saved");
    } else if (routingSaved) {
      setSaveError(t("routingSavedModesFailed", { detail: errorDetail(failure, t("saveFailed")) }));
    } else {
      setSaveError(errorDetail(failure, t("saveFailed")));
    }
    setBusy(null);
  }, [data, modes, llm, assigned, reload, t]);

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

  const profileColumn = llm !== null && Object.keys(llm.profiles).length >= 2;
  const savedFeatures = llm?.routing.features ?? {};
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
              {profileColumn && (
                <th className={headClass}>
                  <span className="flex flex-col gap-0.5">
                    {t("columns.profile")}
                    <ApplyMarker when="now" />
                  </span>
                </th>
              )}
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
                  {profileColumn && llm && (
                    <td className="min-w-48 border-b border-bg-border px-4 py-3 align-top">
                      {llm.features.includes(field) && (
                        <FeatureProfileCell
                          featureLabel={label}
                          llm={llm}
                          value={assigned[field] ?? ""}
                          savedValue={savedFeatures[field] ?? ""}
                          exposure={exposure?.features[field]}
                          fallback={exposure?.local_fallback ?? null}
                          onChange={(value) => setAssigned({ ...assigned, [field]: value })}
                        />
                      )}
                    </td>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {profileColumn && (
        <p className="mt-4 text-xs text-text-muted">{t("destinationsNote")}</p>
      )}

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
