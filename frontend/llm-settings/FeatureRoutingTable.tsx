"use client";

import { useTranslations } from "next-intl";
import type { Destination, ExposureView } from "./api";
import { effectiveProfileName, type Draft } from "./model";

const DESTINATION_ORDER: Destination[] = ["sends", "falls_back", "skips", "unknown"];
const MISSING = "__missing__";

const SELECT_CLASS =
  "w-full rounded-2xl border border-warm-silver/40 bg-bg-card px-3 py-2 text-sm text-text-primary " +
  "focus:outline-none focus:ring-2 focus:ring-focus-ring";

function Destinations({
  drives,
  fallback,
}: {
  drives: Record<string, Destination>;
  fallback: string | null;
}): React.ReactElement {
  const t = useTranslations("settings.llm.routing");
  const groups = DESTINATION_ORDER.map((destination) => ({
    destination,
    names: Object.entries(drives)
      .filter(([, d]) => d === destination)
      .map(([drive]) => drive),
  })).filter((g) => g.names.length > 0);
  return (
    <div className="flex flex-col text-xs text-text-muted" data-testid="feature-destinations">
      {groups.map(({ destination, names }) => (
        <span key={destination}>
          <span className="font-semibold text-text-primary">
            {t(`destinations.${destination}`, { fallback: fallback ?? "" })}
          </span>{" "}
          {names.join(t("driveSeparator"))}
        </span>
      ))}
    </div>
  );
}

function ProfileSelect({
  feature,
  label,
  draft,
  onChange,
}: {
  feature: string;
  label: string;
  draft: Draft;
  onChange: (profileId: string) => void;
}): React.ReactElement {
  const t = useTranslations("settings.llm.routing");
  const route = draft.routing.features[feature];
  const value = route === undefined ? "" : "id" in route ? route.id : MISSING;
  const defaultName = draft.profiles.find((p) => p.id === draft.routing.defaultId)?.name ?? "";
  return (
    <select
      aria-label={t("profileLabel", { feature: label })}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className={SELECT_CLASS}
    >
      <option value="">{t("profileDefault", { name: defaultName })}</option>
      {route !== undefined && "missing" in route && (
        <option value={MISSING} disabled>
          {t("profileMissing", { name: route.missing })}
        </option>
      )}
      {draft.profiles.map((p) => (
        <option key={p.id} value={p.id}>
          {p.offhost ? t("profileOffhost", { name: p.name }) : p.name}
        </option>
      ))}
    </select>
  );
}

export default function FeatureRoutingTable({
  features,
  draft,
  offFeatures,
  exposure,
  onChange,
}: {
  features: string[];
  draft: Draft;
  /** `null` when the feature modes could not be read. */
  offFeatures: Set<string> | null;
  exposure: ExposureView | null;
  onChange: (feature: string, profileId: string) => void;
}): React.ReactElement {
  const t = useTranslations("settings.llm.routing");
  const tf = useTranslations("settings.features.fields");
  const headClass = "border-b border-bg-border px-4 py-3 text-left align-top font-semibold";
  return (
    <div className="flex flex-col gap-4">
      <table className="w-full border-separate border-spacing-0 text-sm text-text-primary">
          <thead>
            <tr>
              <th className={`${headClass} pl-0`}>{t("columns.feature")}</th>
              <th className={`${headClass} pr-0`}>{t("columns.profile")}</th>
            </tr>
          </thead>
          <tbody>
            {features.map((feature) => {
              const label = tf(`${feature}.label`);
              const off = offFeatures?.has(feature) === true;
              const seen = exposure?.features[feature];
              // Exposure describes the saved routing; a changed row has no answer until saved.
              const drives =
                seen?.offhost && seen.profile === effectiveProfileName(draft, feature)
                  ? seen.drives
                  : undefined;
              return (
                <tr key={feature} data-testid={`feature-route-${feature}`}>
                  <td className="border-b border-bg-border py-3 pr-4 align-top">
                    <span className={off ? "text-text-muted" : undefined}>{label}</span>
                    {off && (
                      <span className="ml-2 text-xs text-text-muted">{t("featureOff")}</span>
                    )}
                  </td>
                  <td className="w-1/2 border-b border-bg-border py-3 pl-4 align-top">
                    <div className="flex flex-col gap-2">
                      <ProfileSelect
                        feature={feature}
                        label={label}
                        draft={draft}
                        onChange={(profileId) => onChange(feature, profileId)}
                      />
                      {drives && Object.keys(drives).length > 0 && (
                        <Destinations drives={drives} fallback={exposure?.local_fallback ?? null} />
                      )}
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
      </table>
      <p className="text-xs text-text-muted">{t("destinationsNote")}</p>
    </div>
  );
}
