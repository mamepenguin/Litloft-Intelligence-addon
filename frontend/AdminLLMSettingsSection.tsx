"use client";

import { useCallback, useEffect, useId, useState } from "react";
import { Plus } from "lucide-react";
import { useTranslations } from "next-intl";
import { Button } from "@/components/Button";
import {
  errorDetail,
  fetchExposure,
  fetchLLM,
  fetchOffFeatures,
  resetLLM,
  saveLLM,
  type ExposureView,
  type LLMView,
} from "./llm-settings/api";
import { Card, OutputLanguageCard, RoutingCard } from "./llm-settings/Cards";
import FeatureRoutingTable from "./llm-settings/FeatureRoutingTable";
import { ErrorBlock, WarningBlock } from "./llm-settings/Notices";
import ProfileFields from "./llm-settings/ProfileFields";
import ProfileList from "./llm-settings/ProfileList";
import {
  bodyOf,
  draftFromView,
  drivesLeftWithoutAI,
  invalidOf,
  nextProfile,
  removeProfile,
  setFeatureRoute,
  updateProfile,
  type Draft,
  type Invalid,
  type ProfileDraft,
} from "./llm-settings/model";

type Outcome = { kind: "saved" | "reset"; restart: boolean } | null;
type ActionError = { title: string; detail: string | null } | null;

function Shell({
  titleId,
  intro,
  children,
}: {
  titleId: string;
  intro?: string;
  children: React.ReactNode;
}): React.ReactElement {
  const t = useTranslations("settings.llm");
  return (
    <section aria-labelledby={titleId} className="flex flex-col gap-6">
      <div className="flex flex-col gap-1.5">
        <h2 id={titleId} className="text-lg font-semibold text-text-primary">
          {t("title")}
        </h2>
        {intro && <p className="text-sm text-text-muted">{intro}</p>}
      </div>
      {children}
    </section>
  );
}

function useInvalidText(): (invalid: Invalid) => string {
  const t = useTranslations("settings.llm.invalid");
  return (invalid) => t(invalid.kind, { name: invalid.name });
}

export default function AdminLLMSettingsSection(): React.ReactElement {
  const t = useTranslations("settings.llm");
  const titleId = useId();
  const reasonId = useId();
  const invalidText = useInvalidText();

  const [view, setView] = useState<LLMView | null>(null);
  const [exposure, setExposure] = useState<ExposureView | null>(null);
  const [offFeatures, setOffFeatures] = useState<Set<string> | null>(null);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [seq, setSeq] = useState(0);
  const [busy, setBusy] = useState<"save" | "reset" | null>(null);
  const [actionError, setActionError] = useState<ActionError>(null);
  const [outcome, setOutcome] = useState<Outcome>(null);
  const reload = useCallback(async () => {
    const [next, nextExposure, nextOff] = await Promise.all([
      fetchLLM(),
      fetchExposure().catch(() => null),
      fetchOffFeatures().catch(() => null),
    ]);
    setView(next);
    setExposure(nextExposure);
    setOffFeatures(nextOff);
    setDraft(draftFromView(next));
    setExpandedId(null);
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

  const run = useCallback(
    async (kind: "save" | "reset", base: LLMView, current: Draft) => {
      setActionError(null);
      setOutcome(null);
      setBusy(kind);
      try {
        const result = kind === "save" ? await saveLLM(bodyOf(base, current)) : await resetLLM();
        setOutcome({ kind: kind === "save" ? "saved" : "reset", restart: result.restart_required });
        try {
          await reload();
        } catch (err: unknown) {
          const title = kind === "save" ? t("reloadFailed") : t("resetReloadFailed");
          setActionError({ title, detail: errorDetail(err, "") || null });
        }
      } catch (err: unknown) {
        const title = kind === "save" ? t("saveFailed") : t("resetFailed");
        setActionError({ title, detail: errorDetail(err, title) });
      } finally {
        setBusy(null);
      }
    },
    [reload, t],
  );

  if (loadError) {
    return (
      <Shell titleId={titleId}>
        <ErrorBlock title={t("loadFailed")}>
          <p className="text-sm text-text-primary">{loadError}</p>
        </ErrorBlock>
      </Shell>
    );
  }
  if (!view || !draft) return <Shell titleId={titleId}>{null}</Shell>;

  const single = draft.profiles.length === 1;
  const invalid = invalidOf(draft.profiles);
  const skipDrives = drivesLeftWithoutAI(view, exposure);
  const yamlKey = view.legacy && view.profiles.default?.api_key_source === "yaml";

  const change = (id: string, patch: Partial<Omit<ProfileDraft, "id">>) =>
    setDraft((d) => (d ? updateProfile(d, id, patch) : d));
  const add = () => {
    const profile = nextProfile(draft.profiles, seq);
    setSeq((n) => n + 1);
    setDraft((d) => (d ? { ...d, profiles: [...d.profiles, profile] } : d));
    setExpandedId(profile.id);
  };
  const remove = (id: string) => setDraft((d) => (d ? removeProfile(d, id) : d));
  const setRouting = (patch: Partial<Draft["routing"]>) =>
    setDraft((d) => (d ? { ...d, routing: { ...d.routing, ...patch } } : d));

  return (
    <Shell titleId={titleId} intro={single ? t("introSingle") : t("introList")}>

      {view.error && (
        <ErrorBlock title={t("routingError.title")} testId="llm-routing-error">
          <p className="text-sm text-text-primary">{view.error}</p>
          <p className="text-sm text-text-muted">{t("routingError.hint")}</p>
        </ErrorBlock>
      )}

      {skipDrives.length > 0 && (
        <WarningBlock
          title={t("offhostOnly.title", { drives: skipDrives.join(t("offhostOnly.separator")) })}
          testId="llm-offhost-only-warning"
        >
          <p className="text-sm text-text-muted">{t("offhostOnly.body")}</p>
        </WarningBlock>
      )}

      {yamlKey && (
        <WarningBlock title={t("yamlKey.title")} testId="llm-yaml-key-warning">
          <p className="text-sm text-text-muted">{t("yamlKey.body")}</p>
        </WarningBlock>
      )}

      {single ? (
        <Card title={t("connection")} marker="now">
          <ProfileFields
            profile={draft.profiles[0]}
            variant="single"
            providers={view.available_providers}
            onChange={(patch) => change(draft.profiles[0].id, patch)}
          />
          <Button className="self-start" onClick={add}>
            <Plus size={16} aria-hidden="true" />
            {t("addToSplit")}
          </Button>
        </Card>
      ) : (
        <>
          <Card title={t("profiles")} marker="now">
            <ProfileList
              profiles={draft.profiles}
              defaultId={draft.routing.defaultId}
              expandedId={expandedId}
              providers={view.available_providers}
              onExpand={setExpandedId}
              onChange={change}
              onDelete={remove}
              onAdd={add}
            />
          </Card>
          <RoutingCard
            profiles={draft.profiles}
            defaultId={draft.routing.defaultId}
            fallbackId={draft.routing.fallbackId}
            fallbackMissing={draft.routing.fallbackMissing}
            onDefault={(defaultId) => setRouting({ defaultId })}
            onFallback={(fallbackId) => setRouting({ fallbackId, fallbackMissing: null })}
          >
            <FeatureRoutingTable
              features={view.features}
              draft={draft}
              offFeatures={offFeatures}
              exposure={exposure}
              onChange={(feature, profileId) =>
                setDraft((d) => (d ? setFeatureRoute(d, feature, profileId) : d))
              }
            />
          </RoutingCard>
        </>
      )}

      <OutputLanguageCard
        value={draft.outputLanguage}
        languages={view.available_output_languages}
        restartPending={view.output_language_restart_pending}
        onChange={(outputLanguage) => setDraft((d) => (d ? { ...d, outputLanguage } : d))}
      />

      {actionError && (
        <ErrorBlock title={actionError.title} testId="llm-save-error">
          {actionError.detail && actionError.detail !== actionError.title && (
            <p className="text-sm text-text-primary">{actionError.detail}</p>
          )}
        </ErrorBlock>
      )}

      <div className="flex flex-wrap items-center gap-3">
        <Button
          variant="primary"
          onClick={() => run("save", view, draft)}
          disabled={busy !== null || invalid !== null}
          aria-describedby={invalid ? reasonId : undefined}
        >
          {busy === "save" ? t("saving") : t("save")}
        </Button>
        {view.overrides_present && (
          <>
            <Button onClick={() => run("reset", view, draft)} disabled={busy !== null}>
              {busy === "reset" ? t("resetting") : t("reset")}
            </Button>
            <span className="text-sm text-text-muted">{t("overridesActive")}</span>
          </>
        )}
        {invalid && (
          <span id={reasonId} className="text-sm text-danger">
            {invalidText(invalid)}
          </span>
        )}
        {outcome && (
          <span role="status" className="text-sm text-accent-teal">
            {t(`outcome.${outcome.kind}${outcome.restart ? "Restart" : "Now"}`)}
          </span>
        )}
      </div>
    </Shell>
  );
}
