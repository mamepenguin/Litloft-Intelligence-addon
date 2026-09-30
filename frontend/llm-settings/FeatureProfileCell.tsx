"use client";

import { useTranslations } from "next-intl";
import type { Destination, FeatureExposure, LLMView } from "./api";

const DESTINATION_ORDER: Destination[] = ["sends", "falls_back", "skips", "unknown"];

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
  const t = useTranslations("settings.features");
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

export default function FeatureProfileCell({
  featureLabel,
  llm,
  value,
  savedValue,
  exposure,
  fallback,
  onChange,
}: {
  featureLabel: string;
  llm: LLMView;
  /** A profile name, or "" for the default profile. */
  value: string;
  savedValue: string;
  exposure: FeatureExposure | undefined;
  fallback: string | null;
  onChange: (value: string) => void;
}): React.ReactElement {
  const t = useTranslations("settings.features");
  const names = Object.keys(llm.profiles);
  const defaultName = llm.routing.default ?? names[0] ?? "";
  // Exposure describes the saved routing; a changed choice has no answer until saved.
  const drives = value === savedValue && exposure?.offhost ? exposure.drives : undefined;
  return (
    <div className="flex flex-col gap-2">
      <select
        aria-label={t("profileLabel", { feature: featureLabel })}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className={SELECT_CLASS}
      >
        <option value="">{t("profileDefault", { name: defaultName })}</option>
        {value !== "" && !names.includes(value) && (
          <option value={value} disabled>
            {t("profileMissing", { name: value })}
          </option>
        )}
        {names.map((name) => (
          <option key={name} value={name}>
            {llm.profiles[name].offhost === false ? name : t("profileOffhost", { name })}
          </option>
        ))}
      </select>
      {drives && Object.keys(drives).length > 0 && (
        <Destinations drives={drives} fallback={fallback} />
      )}
    </div>
  );
}

export { SELECT_CLASS };
