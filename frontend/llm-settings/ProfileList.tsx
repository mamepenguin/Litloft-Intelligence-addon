"use client";

import { Cloud, Monitor, Plus, Trash2 } from "lucide-react";
import { useTranslations } from "next-intl";
import { Button } from "@/components/Button";
import ProfileFields, { KeyPresence } from "./ProfileFields";
import { keyPresenceKnown, type ProfileDraft } from "./model";

type Patch = Partial<Omit<ProfileDraft, "id">>;

const BADGE_CLASS = "rounded-full px-2.5 py-0.5 text-xs";

function ProfileIcon({ offhost }: { offhost: boolean }): React.ReactElement {
  const Icon = offhost ? Cloud : Monitor;
  return (
    <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-warm-light text-text-primary">
      <Icon size={18} aria-hidden="true" />
    </span>
  );
}

function Badges({ profile, isDefault }: { profile: ProfileDraft; isDefault: boolean }) {
  const t = useTranslations("settings.llm.list");
  return (
    <>
      <span className="font-mono text-sm font-semibold text-text-primary">{profile.name}</span>
      <span className={`${BADGE_CLASS} bg-sand text-text-primary`}>
        {profile.offhost ? t("offhost") : t("local")}
      </span>
      {isDefault && (
        <span className={`${BADGE_CLASS} border border-warm-silver/40 text-text-muted`}>
          {t("default")}
        </span>
      )}
    </>
  );
}

function Summary({ profile }: { profile: ProfileDraft }): React.ReactElement {
  const t = useTranslations("settings.llm");
  const parts = [t(`profile.providers.${profile.provider}`)];
  if (profile.model) parts.push(profile.model);
  if (profile.visionModel) parts.push(t("list.vision", { model: profile.visionModel }));
  const keyMissing =
    profile.provider === "openai_compatible" && keyPresenceKnown(profile) && !profile.apiKeyPresent;
  return (
    <div className="flex min-w-0 flex-col gap-1">
      <span className="truncate text-sm text-text-muted">{parts.join(" · ")}</span>
      {keyMissing && (
        <span className="flex flex-wrap items-center gap-x-3 gap-y-1">
          <span className="whitespace-nowrap font-mono text-sm text-text-primary">{profile.keyEnv}</span>
          <KeyPresence profile={profile} />
        </span>
      )}
    </div>
  );
}

interface ListProps {
  profiles: ProfileDraft[];
  defaultId: string | null;
  expandedId: string | null;
  providers: string[];
  onExpand: (id: string) => void;
  onChange: (id: string, patch: Patch) => void;
  onDelete: (id: string) => void;
  onAdd: () => void;
}

export default function ProfileList({
  profiles,
  defaultId,
  expandedId,
  providers,
  onExpand,
  onChange,
  onDelete,
  onAdd,
}: ListProps): React.ReactElement {
  const t = useTranslations("settings.llm.list");
  return (
    <div className="flex flex-col gap-4">
      {profiles.map((profile) =>
        profile.id === expandedId ? (
          <div
            key={profile.id}
            data-testid={`llm-profile-editor-${profile.id}`}
            className="flex flex-col gap-4 rounded-xl border border-accent p-5"
          >
            <div className="flex items-center gap-4">
              <ProfileIcon offhost={profile.offhost} />
              <div className="flex min-w-0 flex-1 flex-wrap items-center gap-2">
                <Badges profile={profile} isDefault={profile.id === defaultId} />
                <span className="text-xs text-text-muted">{t("editing")}</span>
              </div>
              <Button
                iconOnly
                variant="danger"
                aria-label={t("delete", { name: profile.name })}
                onClick={() => onDelete(profile.id)}
              >
                <Trash2 size={16} aria-hidden="true" />
              </Button>
            </div>
            <ProfileFields
              profile={profile}
              variant="list"
              providers={providers}
              onChange={(patch) => onChange(profile.id, patch)}
            />
          </div>
        ) : (
          <div
            key={profile.id}
            data-testid={`llm-profile-row-${profile.id}`}
            className="flex flex-col gap-3 rounded-xl border border-bg-border p-4 sm:flex-row sm:items-center sm:gap-4"
          >
            <div className="flex min-w-0 flex-1 items-center gap-4">
              <ProfileIcon offhost={profile.offhost} />
              <div className="flex min-w-0 flex-1 flex-col gap-1">
                <div className="flex flex-wrap items-center gap-2">
                  <Badges profile={profile} isDefault={profile.id === defaultId} />
                </div>
                <Summary profile={profile} />
              </div>
            </div>
            <Button
              className="self-end sm:self-auto"
              variant="ghost"
              aria-label={t("edit", { name: profile.name })}
              onClick={() => onExpand(profile.id)}
            >
              {t("editLabel")}
            </Button>
          </div>
        ),
      )}
      <Button className="self-start" onClick={onAdd}>
        <Plus size={16} aria-hidden="true" />
        {t("add")}
      </Button>
    </div>
  );
}
