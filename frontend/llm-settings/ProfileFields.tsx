"use client";

import { useId } from "react";
import { Check, X } from "lucide-react";
import { useTranslations } from "next-intl";
import { SegmentedControl } from "@/components/SegmentedControl";
import {
  KEY_ENV_PREFIX,
  keyChoiceOf,
  keyEnvFor,
  keyPresenceKnown,
  keySuffixOf,
  type KeyChoice,
  type ProfileDraft,
} from "./model";

export const INPUT_CLASS =
  "w-full rounded-2xl border border-warm-silver/40 bg-bg-card px-3.5 py-2.5 text-sm text-text-primary " +
  "focus:outline-none focus:ring-2 focus:ring-focus-ring";
const MONO_INPUT_CLASS = `${INPUT_CLASS} font-mono`;
const LABEL_CLASS = "text-sm font-semibold text-text-primary";

const PROVIDER_ORDER = ["ollama", "openai_compatible", "disabled"];

type Patch = Partial<Omit<ProfileDraft, "id">>;

interface Props {
  profile: ProfileDraft;
  /** `single` hides the name, the key editor and the Agentic switch. */
  variant: "single" | "list";
  providers: string[];
  onChange: (patch: Patch) => void;
}

function TextField({
  label,
  hint,
  value,
  onChange,
}: {
  label: string;
  hint?: string;
  value: string;
  onChange: (value: string) => void;
}): React.ReactElement {
  return (
    <label className="flex flex-col gap-1.5">
      <span className={LABEL_CLASS}>
        {label}
        {hint && <span className="font-normal text-text-muted"> {hint}</span>}
      </span>
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className={MONO_INPUT_CLASS}
      />
    </label>
  );
}

function Switch({
  label,
  help,
  checked,
  onChange,
}: {
  label: string;
  help: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
}): React.ReactElement {
  const id = useId();
  return (
    <div className="flex items-start gap-2.5">
      <input
        id={id}
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        aria-describedby={`${id}-help`}
        className="mt-1 h-4 w-4 shrink-0 accent-accent"
      />
      <span className="flex flex-col gap-0.5">
        <label htmlFor={id} className="text-sm font-semibold text-text-primary">
          {label}
        </label>
        <span id={`${id}-help`} className="text-xs text-text-muted">
          {help}
        </span>
      </span>
    </div>
  );
}

export function KeyPresence({ profile }: { profile: ProfileDraft }): React.ReactElement | null {
  const t = useTranslations("settings.llm.profile");
  if (!keyPresenceKnown(profile)) return null;
  if (profile.apiKeyPresent) {
    return (
      <span className="flex items-center gap-1.5 whitespace-nowrap text-sm text-accent-teal">
        <Check size={14} aria-hidden="true" />
        {t("keyPresent")}
      </span>
    );
  }
  if (profile.provider === "ollama") {
    return <span className="text-sm text-text-muted">{t("keyMissingOllama")}</span>;
  }
  if (profile.provider !== "openai_compatible") {
    return <span className="text-sm text-text-muted">{t("keyMissing")}</span>;
  }
  return (
    <span className="flex items-center gap-1.5 text-sm text-danger">
      <X size={14} aria-hidden="true" />
      {t("keyMissingFails")}
    </span>
  );
}

const KEY_CHOICES: KeyChoice[] = ["none", "shared", "named"];

function KeyEditor({ profile, onChange }: Pick<Props, "profile" | "onChange">): React.ReactElement {
  const t = useTranslations("settings.llm.profile");
  const id = useId();
  const choice = keyChoiceOf(profile.keyEnv);
  const suffix = keySuffixOf(profile.keyEnv);
  return (
    <div className="flex flex-col gap-1.5">
      <label htmlFor={`${id}-choice`} className={LABEL_CLASS}>
        {t("apiKey")}
      </label>
      <div className="flex flex-wrap items-center gap-3">
        <select
          id={`${id}-choice`}
          value={choice}
          onChange={(e) =>
            onChange({ keyEnv: keyEnvFor(e.target.value as KeyChoice, suffix) })
          }
          className={`${INPUT_CLASS} sm:w-64`}
        >
          {KEY_CHOICES.map((value) => (
            <option key={value} value={value}>
              {t(`keyChoice.${value}`)}
            </option>
          ))}
        </select>
        {choice === "named" && (
          <div className="flex min-w-0 flex-1 items-center overflow-hidden rounded-2xl border border-warm-silver/40 bg-bg-card focus-within:ring-2 focus-within:ring-focus-ring">
            <span className="py-2.5 pl-3.5 font-mono text-sm text-text-muted" aria-hidden="true">
              {`${KEY_ENV_PREFIX}_`}
            </span>
            <input
              id={id}
              type="text"
              aria-label={t("keyEnv")}
              value={suffix}
              onChange={(e) => onChange({ keyEnv: keyEnvFor("named", e.target.value) })}
              aria-describedby={`${id}-prefix ${id}-help`}
              className="min-w-0 flex-1 bg-transparent py-2.5 pr-3.5 font-mono text-sm text-text-primary focus:outline-none"
            />
            <span id={`${id}-prefix`} className="sr-only">
              {t("keyEnvPrefix", { env: `${KEY_ENV_PREFIX}_${suffix}` })}
            </span>
          </div>
        )}
        <KeyPresence profile={profile} />
      </div>
      {choice !== "none" && (
        <span id={`${id}-help`} className="text-xs text-text-muted">
          {t("keyEnvHelp")}
        </span>
      )}
    </div>
  );
}

function StaticKey({ profile }: { profile: ProfileDraft }): React.ReactElement {
  const t = useTranslations("settings.llm.profile");
  return (
    <div className="flex flex-col gap-1.5">
      <span className={LABEL_CLASS}>{t("apiKey")}</span>
      {profile.keyEnv === null ? (
        <span className="text-sm text-text-muted">{t("keyChoice.none")}</span>
      ) : (
        <span className="flex flex-wrap items-center gap-x-2 gap-y-1 text-sm text-text-muted">
          <span>
            {t("envVar")}{" "}
            <span className="font-mono text-text-primary">{profile.keyEnv}</span> ·
          </span>
          <KeyPresence profile={profile} />
        </span>
      )}
    </div>
  );
}

export default function ProfileFields({
  profile,
  variant,
  providers,
  onChange,
}: Props): React.ReactElement {
  const t = useTranslations("settings.llm.profile");
  const options = PROVIDER_ORDER.filter((p) => providers.includes(p)).map((value) => ({
    value,
    label: t(`providers.${value}`),
  }));
  const provider = (
    <div className="flex flex-col gap-1.5">
      <span className={LABEL_CLASS} aria-hidden="true">
        {t("provider")}
      </span>
      <SegmentedControl
        label={t("provider")}
        options={options}
        value={profile.provider}
        onChange={(value) => onChange({ provider: value })}
      />
    </div>
  );

  return (
    <div className="flex flex-col gap-4">
      {variant === "list" ? (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          <TextField label={t("name")} value={profile.name} onChange={(name) => onChange({ name })} />
          {provider}
        </div>
      ) : (
        provider
      )}

      <TextField
        label={t("baseUrl")}
        value={profile.baseUrl}
        onChange={(baseUrl) => onChange({ baseUrl })}
      />

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <TextField label={t("model")} value={profile.model} onChange={(model) => onChange({ model })} />
        <TextField
          label={t("visionModel")}
          hint={t("visionModelHint")}
          value={profile.visionModel}
          onChange={(visionModel) => onChange({ visionModel })}
        />
      </div>

      {variant === "list" ? (
        <KeyEditor profile={profile} onChange={onChange} />
      ) : (
        <StaticKey profile={profile} />
      )}

      <div className="flex flex-col gap-2.5 border-t border-bg-border pt-4">
        <Switch
          label={t("offhost")}
          help={variant === "list" ? t("offhostHelpList") : t("offhostHelpSingle")}
          checked={profile.offhost}
          onChange={(offhost) => onChange({ offhost })}
        />
        {variant === "list" && (
          <Switch
            label={t("agentic")}
            help={t("agenticHelp")}
            checked={profile.agentic}
            onChange={(agentic) => onChange({ agentic })}
          />
        )}
      </div>
    </div>
  );
}
