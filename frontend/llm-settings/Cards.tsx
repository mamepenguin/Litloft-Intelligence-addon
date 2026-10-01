"use client";

import { useId, type ReactNode } from "react";
import { useTranslations } from "next-intl";
import { ApplyMarker } from "./Notices";
import { INPUT_CLASS } from "./ProfileFields";
import type { ProfileDraft } from "./model";

export function Card({
  title,
  marker,
  children,
}: {
  title: string;
  marker?: "now" | "restart";
  children: ReactNode;
}): React.ReactElement {
  return (
    <div className="flex flex-col gap-4 rounded-xl border border-bg-border bg-bg-card p-6">
      <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <h3 className="text-base font-semibold text-text-primary">{title}</h3>
        {marker && <ApplyMarker when={marker} />}
      </div>
      {children}
    </div>
  );
}

function RoutingSelect({
  label,
  help,
  value,
  onChange,
  children,
}: {
  label: string;
  help: string;
  value: string;
  onChange: (value: string) => void;
  children: ReactNode;
}): React.ReactElement {
  const id = useId();
  return (
    <div className="flex flex-col gap-1.5">
      <label htmlFor={id} className="text-sm font-semibold text-text-primary">
        {label}
      </label>
      <select
        id={id}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        aria-describedby={`${id}-help`}
        className={INPUT_CLASS}
      >
        {children}
      </select>
      <span id={`${id}-help`} className="text-xs text-text-muted">
        {help}
      </span>
    </div>
  );
}

const MISSING = "__missing__";

export function RoutingCard({
  profiles,
  defaultId,
  fallbackId,
  fallbackMissing,
  onDefault,
  onFallback,
  children,
}: {
  profiles: ProfileDraft[];
  defaultId: string | null;
  fallbackId: string | null;
  fallbackMissing: string | null;
  onDefault: (id: string) => void;
  onFallback: (id: string | null) => void;
  children: ReactNode;
}): React.ReactElement {
  const t = useTranslations("settings.llm.routing");
  const local = profiles.filter((p) => !p.offhost);
  const offhostFallback = profiles.find((p) => p.id === fallbackId && p.offhost);
  return (
    <Card title={t("title")} marker="now">
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <RoutingSelect
          label={t("default")}
          help={t("defaultHelp")}
          value={defaultId ?? ""}
          onChange={onDefault}
        >
          {profiles.map((p) => (
            <option key={p.id} value={p.id}>
              {p.name}
            </option>
          ))}
        </RoutingSelect>
        <RoutingSelect
          label={t("fallback")}
          help={t("fallbackHelp")}
          value={fallbackMissing !== null ? MISSING : (fallbackId ?? "")}
          onChange={(value) => onFallback(value || null)}
        >
          {offhostFallback && (
            <option value={offhostFallback.id} disabled>
              {t("fallbackOffhost", { name: offhostFallback.name })}
            </option>
          )}
          {fallbackMissing !== null && (
            <option value={MISSING} disabled>
              {t("fallbackMissing", { name: fallbackMissing })}
            </option>
          )}
          {local.map((p) => (
            <option key={p.id} value={p.id}>
              {p.name}
            </option>
          ))}
          <option value="">{t("fallbackNone")}</option>
        </RoutingSelect>
      </div>
      {children}
    </Card>
  );
}

export function OutputLanguageCard({
  value,
  languages,
  restartPending,
  onChange,
}: {
  value: string;
  languages: string[];
  restartPending: boolean;
  onChange: (value: string) => void;
}): React.ReactElement {
  const t = useTranslations("settings.llm.outputLanguage");
  return (
    <Card title={t("title")} marker="restart">
      <select
        aria-label={t("title")}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className={`${INPUT_CLASS} sm:w-80`}
      >
        {languages.map((lang) => (
          <option key={lang} value={lang}>
            {t(`options.${lang}`)}
          </option>
        ))}
      </select>
      {restartPending && <p className="text-xs text-text-muted">{t("pending")}</p>}
    </Card>
  );
}
