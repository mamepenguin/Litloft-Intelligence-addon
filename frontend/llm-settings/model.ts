import type { LLMUpdateBody, LLMView, ProfileView, RoutingView } from "./api";

export const KEY_ENV_PREFIX = "LLM_API_KEY";
const NAMED_KEY_PREFIX = `${KEY_ENV_PREFIX}_`;

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
  "api_key_source",
  "api_key",
]);
const ROUTING_FIELDS = new Set(["default", "local_fallback", "features"]);
const VIEW_ONLY_KEYS = new Set(["api_key_present", "api_key_source"]);

export interface ProfileDraft {
  id: string;
  name: string;
  provider: string;
  baseUrl: string;
  model: string;
  visionModel: string;
  offhost: boolean;
  agentic: boolean;
  /** `null`: the profile reads no key. */
  keyEnv: string | null;
  /** The env var `apiKeyPresent` was measured for. */
  savedKeyEnv: string | null;
  apiKeyPresent: boolean;
  extra: Record<string, unknown>;
}

/** A saved feature choice: a profile, or a name that matches none (kept until changed). */
export type FeatureRoute = { id: string } | { missing: string };

export interface RoutingDraft {
  defaultId: string | null;
  fallbackId: string | null;
  /** A saved fallback that names no profile, kept until the user picks another. */
  fallbackMissing: string | null;
  /** A feature absent here uses the default profile. */
  features: Record<string, FeatureRoute>;
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

export type KeyChoice = "none" | "shared" | "named";

export function keyChoiceOf(keyEnv: string | null): KeyChoice {
  if (keyEnv === null) return "none";
  return keyEnv === KEY_ENV_PREFIX ? "shared" : "named";
}

export function keySuffixOf(keyEnv: string | null): string {
  return keyEnv?.startsWith(NAMED_KEY_PREFIX) ? keyEnv.slice(NAMED_KEY_PREFIX.length) : "";
}

export function keyEnvFor(choice: KeyChoice, suffix = ""): string | null {
  if (choice === "none") return null;
  return choice === "shared" ? KEY_ENV_PREFIX : `${NAMED_KEY_PREFIX}${suffix}`;
}

function validKeyEnv(keyEnv: string | null): boolean {
  if (keyEnv === null || keyEnv === KEY_ENV_PREFIX) return true;
  return keyEnv.startsWith(NAMED_KEY_PREFIX) && KEY_SUFFIX_RE.test(keySuffixOf(keyEnv));
}

function without(record: Record<string, unknown>, keys: Set<string>): Record<string, unknown> {
  return Object.fromEntries(Object.entries(record).filter(([k]) => !keys.has(k)));
}

function str(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function profileDraft(id: string, name: string, view: ProfileView): ProfileDraft {
  const keyEnv = typeof view.api_key_env === "string" ? view.api_key_env : null;
  return {
    id,
    name,
    provider: str(view.provider) || "disabled",
    baseUrl: str(view.base_url),
    model: str(view.model),
    visionModel: str(view.vision_model),
    offhost: view.offhost !== false,
    agentic: view.agentic === true,
    keyEnv,
    savedKeyEnv: keyEnv,
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
  const fallback = view.routing.local_fallback;
  const fallbackId = idOf(fallback);
  const features = Object.fromEntries(
    Object.entries(view.routing.features ?? {}).map(([feature, name]): [string, FeatureRoute] => {
      const id = idOf(name);
      return [feature, id === null ? { missing: name } : { id }];
    }),
  );
  return {
    profiles,
    routing: {
      defaultId: idOf(view.routing.default) ?? profiles[0]?.id ?? null,
      fallbackId,
      fallbackMissing: fallbackId === null && typeof fallback === "string" ? fallback : null,
      features,
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
    keyEnv: null,
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
    if (!validKeyEnv(p.keyEnv)) return { kind: "key", name: p.name };
  }
  return null;
}

/** Whether the saved key presence still describes the env var the profile names. */
export function keyPresenceKnown(p: ProfileDraft): boolean {
  return p.keyEnv !== null && p.savedKeyEnv === p.keyEnv;
}

function profileBody(p: ProfileDraft): ProfileView {
  return {
    ...p.extra,
    provider: p.provider,
    base_url: p.baseUrl,
    model: p.model,
    vision_model: p.visionModel,
    offhost: p.offhost,
    agentic: p.agentic,
    ...(p.keyEnv === null ? {} : { api_key_env: p.keyEnv }),
  };
}

/** `profileId`: a profile's id, or "" for the default profile. */
export function setFeatureRoute(draft: Draft, feature: string, profileId: string): Draft {
  const rest = Object.fromEntries(
    Object.entries(draft.routing.features).filter(([f]) => f !== feature),
  );
  const features = profileId === "" ? rest : { ...rest, [feature]: { id: profileId } };
  return { ...draft, routing: { ...draft.routing, features } };
}

function nameOfId(draft: Draft, id: string | null): string | undefined {
  return draft.profiles.find((p) => p.id === id)?.name;
}

/** The profile name a feature runs on in `draft`, or undefined when unresolved. */
export function effectiveProfileName(draft: Draft, feature: string): string | undefined {
  const route = draft.routing.features[feature];
  if (route === undefined) return nameOfId(draft, draft.routing.defaultId);
  return "id" in route ? nameOfId(draft, route.id) : route.missing;
}

export function storedProfile(view: ProfileView): ProfileView {
  return Object.fromEntries(Object.entries(view).filter(([k]) => !VIEW_ONLY_KEYS.has(k)));
}

function sameJSON(a: unknown, b: unknown): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

/**
 * The PUT body for `draft`, loaded from `base`. A profile the user did not
 * touch is written as `base` stored it; routing keys this screen does not edit
 * are kept.
 */
export function bodyOf(base: LLMView, draft: Draft): LLMUpdateBody {
  const baseProfiles = new Map(draftFromView(base).profiles.map((p) => [p.id, p]));
  const profiles = Object.fromEntries(
    draft.profiles.map((p) => {
      const before = baseProfiles.get(p.id);
      const untouched = before !== undefined && sameJSON(before, p);
      return [p.name, untouched ? storedProfile(base.profiles[before.name]) : profileBody(p)];
    }),
  );
  const features = Object.fromEntries(
    Object.entries(draft.routing.features).flatMap(([feature, route]) => {
      const name = "id" in route ? nameOfId(draft, route.id) : route.missing;
      return name === undefined ? [] : [[feature, name]];
    }),
  );
  const rest = Object.fromEntries(
    Object.entries(base.routing).filter(([k]) => !ROUTING_FIELDS.has(k)),
  );
  const defaultName = nameOfId(draft, draft.routing.defaultId);
  const fallbackName = nameOfId(draft, draft.routing.fallbackId) ?? draft.routing.fallbackMissing;
  const routing: RoutingView = {
    ...rest,
    ...(defaultName === undefined ? {} : { default: defaultName }),
    ...(fallbackName == null ? {} : { local_fallback: fallbackName }),
    ...(Object.keys(features).length > 0 ? { features } : {}),
  };
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
