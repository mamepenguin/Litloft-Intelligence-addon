import type { LLMUpdateBody, LLMView, ProfileView, RoutingView } from "./api";

export const KEY_ENV_PREFIX = "LLM_API_KEY";

const NAME_RE = /^[a-z0-9][a-z0-9_-]{0,31}$/;
const KEY_SUFFIX_RE = /^[A-Z0-9]+(_[A-Z0-9]+)*$/;

const PROFILE_FIELDS = new Set([
  "provider",
  "base_url",
  "model",
  "vision_model",
  "offhost",
  "agentic",
  "api_key_env",
  "api_key_present",
  "api_key",
]);
const ROUTING_FIELDS = new Set(["default", "local_fallback", "features"]);

export interface ProfileDraft {
  id: string;
  name: string;
  provider: string;
  baseUrl: string;
  model: string;
  visionModel: string;
  offhost: boolean;
  agentic: boolean;
  keySuffix: string;
  /** The env var `apiKeyPresent` was measured for. */
  savedKeyEnv: string | null;
  apiKeyPresent: boolean;
  extra: Record<string, unknown>;
}

export interface RoutingDraft {
  defaultId: string | null;
  fallbackId: string | null;
  /** feature -> profile id */
  features: Record<string, string>;
  extra: Record<string, unknown>;
}

export interface Draft {
  profiles: ProfileDraft[];
  routing: RoutingDraft;
  outputLanguage: string;
}

export type Invalid =
  | { kind: "name"; name: string }
  | { kind: "duplicate"; name: string }
  | { kind: "key"; name: string };

export function keyEnvOf(suffix: string): string {
  return suffix === "" ? KEY_ENV_PREFIX : `${KEY_ENV_PREFIX}_${suffix}`;
}

export function keySuffixOf(env: string | undefined): string {
  if (!env || env === KEY_ENV_PREFIX) return "";
  return env.startsWith(`${KEY_ENV_PREFIX}_`) ? env.slice(KEY_ENV_PREFIX.length + 1) : env;
}

function without(record: Record<string, unknown>, keys: Set<string>): Record<string, unknown> {
  return Object.fromEntries(Object.entries(record).filter(([k]) => !keys.has(k)));
}

function str(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function profileDraft(id: string, name: string, view: ProfileView): ProfileDraft {
  return {
    id,
    name,
    provider: str(view.provider) || "disabled",
    baseUrl: str(view.base_url),
    model: str(view.model),
    visionModel: str(view.vision_model),
    offhost: view.offhost !== false,
    agentic: view.agentic === true,
    keySuffix: keySuffixOf(view.api_key_env),
    savedKeyEnv: typeof view.api_key_env === "string" ? view.api_key_env : null,
    apiKeyPresent: view.api_key_present === true,
    extra: without(view, PROFILE_FIELDS),
  };
}

export function draftFromView(view: LLMView): Draft {
  const profiles = Object.entries(view.profiles).map(([name, p], i) =>
    profileDraft(`p${i}`, name, p),
  );
  const idOf = (name: unknown): string | null =>
    profiles.find((p) => p.name === name)?.id ?? null;
  const features = Object.fromEntries(
    Object.entries(view.routing.features ?? {})
      .map(([feature, name]) => [feature, idOf(name)] as const)
      .filter((entry): entry is readonly [string, string] => entry[1] !== null),
  );
  return {
    profiles,
    routing: {
      defaultId: idOf(view.routing.default) ?? profiles[0]?.id ?? null,
      fallbackId: idOf(view.routing.local_fallback),
      features,
      extra: without(view.routing, ROUTING_FIELDS),
    },
    outputLanguage: view.output_language,
  };
}

export function nextProfile(existing: ProfileDraft[], seq: number): ProfileDraft {
  const taken = new Set(existing.map((p) => p.name));
  let n = existing.length + 1;
  while (taken.has(`profile-${n}`)) n += 1;
  return {
    id: `new${seq}`,
    name: `profile-${n}`,
    provider: "openai_compatible",
    baseUrl: "",
    model: "",
    visionModel: "",
    offhost: true,
    agentic: false,
    keySuffix: "",
    savedKeyEnv: null,
    apiKeyPresent: false,
    extra: {},
  };
}

export function removeProfile(draft: Draft, id: string): Draft {
  const profiles = draft.profiles.filter((p) => p.id !== id);
  const { routing } = draft;
  return {
    ...draft,
    profiles,
    routing: {
      ...routing,
      defaultId: routing.defaultId === id ? (profiles[0]?.id ?? null) : routing.defaultId,
      fallbackId: routing.fallbackId === id ? null : routing.fallbackId,
      features: Object.fromEntries(
        Object.entries(routing.features).filter(([, pid]) => pid !== id),
      ),
    },
  };
}

export function updateProfile(
  draft: Draft,
  id: string,
  patch: Partial<Omit<ProfileDraft, "id">>,
): Draft {
  const profiles = draft.profiles.map((p) => (p.id === id ? { ...p, ...patch } : p));
  const fallbackNowOffhost =
    draft.routing.fallbackId === id && patch.offhost === true;
  return {
    ...draft,
    profiles,
    routing: fallbackNowOffhost ? { ...draft.routing, fallbackId: null } : draft.routing,
  };
}

export function invalidOf(profiles: ProfileDraft[]): Invalid | null {
  const seen = new Set<string>();
  for (const p of profiles) {
    if (!NAME_RE.test(p.name)) return { kind: "name", name: p.name };
    if (seen.has(p.name)) return { kind: "duplicate", name: p.name };
    seen.add(p.name);
    if (p.keySuffix !== "" && !KEY_SUFFIX_RE.test(p.keySuffix)) {
      return { kind: "key", name: p.name };
    }
  }
  return null;
}

/** Whether the saved key presence still describes the env var in the input. */
export function keyPresenceKnown(p: ProfileDraft): boolean {
  return p.savedKeyEnv === keyEnvOf(p.keySuffix);
}

export function bodyFromDraft(draft: Draft): LLMUpdateBody {
  const nameOf = (id: string | null): string | undefined =>
    draft.profiles.find((p) => p.id === id)?.name;
  const profiles = Object.fromEntries(
    draft.profiles.map((p) => [
      p.name,
      {
        ...p.extra,
        provider: p.provider,
        base_url: p.baseUrl,
        model: p.model,
        vision_model: p.visionModel,
        offhost: p.offhost,
        agentic: p.agentic,
        api_key_env: keyEnvOf(p.keySuffix),
      },
    ]),
  );
  const routing: RoutingView = { ...draft.routing.extra };
  const defaultName = nameOf(draft.routing.defaultId);
  if (defaultName !== undefined) routing.default = defaultName;
  const fallbackName = nameOf(draft.routing.fallbackId);
  if (fallbackName !== undefined) routing.local_fallback = fallbackName;
  const features = Object.fromEntries(
    Object.entries(draft.routing.features).flatMap(([feature, id]) => {
      const name = nameOf(id);
      return name === undefined ? [] : [[feature, name]];
    }),
  );
  if (Object.keys(features).length > 0) routing.features = features;
  return { profiles, routing, output_language: draft.outputLanguage };
}

/** Drives where some feature does not run because the only profile is off-host. */
export function drivesLeftWithoutAI(
  view: LLMView,
  exposure: { features: Record<string, { drives?: Record<string, string> }> } | null,
): string[] {
  const profiles = Object.values(view.profiles);
  if (profiles.length !== 1 || profiles[0].offhost === false || !exposure) return [];
  const drives = new Set<string>();
  for (const entry of Object.values(exposure.features)) {
    for (const [drive, destination] of Object.entries(entry.drives ?? {})) {
      if (destination === "skips") drives.add(drive);
    }
  }
  return [...drives].sort();
}
