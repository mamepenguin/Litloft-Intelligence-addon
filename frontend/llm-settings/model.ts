import type { LLMUpdateBody, LLMView, ProfileView, RoutingView } from "./api";
import { documentOf, storedProfile } from "./store";

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

export interface RoutingDraft {
  defaultId: string | null;
  fallbackId: string | null;
  /** A saved fallback that names no profile, kept until the user picks another. */
  fallbackMissing: string | null;
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
  return {
    profiles,
    routing: {
      defaultId: idOf(view.routing.default) ?? profiles[0]?.id ?? null,
      fallbackId,
      fallbackMissing: fallbackId === null && typeof fallback === "string" ? fallback : null,
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

function sameJSON(a: unknown, b: unknown): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

/** Whether `draft` differs from what `base` loads as. */
export function isDirty(base: LLMView, draft: Draft): boolean {
  return !sameJSON(draft, draftFromView(base));
}

/**
 * The body that writes this section's edits onto `latest`: its profiles
 * (renamed and deleted by id), the default, the fallback and the output
 * language. `routing.features` is taken from `latest`, following the renames
 * and deletions; other profiles and routing keys in `latest` are kept.
 */
export function bodyOnto(latest: LLMView, base: LLMView, draft: Draft): LLMUpdateBody {
  const doc = documentOf(latest);
  const baseDraft = draftFromView(base);
  const baseNameOf = new Map(baseDraft.profiles.map((p) => [p.id, p.name]));
  const idOfBaseName = new Map(baseDraft.profiles.map((p) => [p.name, p.id]));
  const draftById = new Map(draft.profiles.map((p) => [p.id, p]));
  const draftNames = new Set(draft.profiles.map((p) => p.name));

  const entryOf = (p: ProfileDraft): ProfileView => {
    const before = baseDraft.profiles.find((b) => b.id === p.id);
    const baseName = baseNameOf.get(p.id);
    if (before && baseName !== undefined && sameJSON(before, p)) {
      return doc.profiles[baseName] ?? storedProfile(base.profiles[baseName]);
    }
    return profileBody(p);
  };

  const placed = new Set<string>();
  const profiles: [string, ProfileView][] = [];
  for (const [name, stored] of Object.entries(doc.profiles)) {
    const id = idOfBaseName.get(name);
    if (id === undefined) {
      if (!draftNames.has(name)) profiles.push([name, stored]);
      continue;
    }
    const p = draftById.get(id);
    if (!p) continue;
    profiles.push([p.name, entryOf(p)]);
    placed.add(id);
  }
  for (const p of draft.profiles) {
    if (!placed.has(p.id)) profiles.push([p.name, entryOf(p)]);
  }

  const nameOf = (id: string | null): string | undefined =>
    draft.profiles.find((p) => p.id === id)?.name;
  const renamed = (name: string): string | null => {
    const id = idOfBaseName.get(name);
    if (id === undefined) return name;
    return draftById.get(id)?.name ?? null;
  };
  const features = Object.fromEntries(
    Object.entries(latest.routing.features ?? {}).flatMap(([feature, name]) => {
      const next = renamed(name);
      return next === null ? [] : [[feature, next]];
    }),
  );
  const rest = Object.fromEntries(
    Object.entries(doc.routing).filter(([k]) => !ROUTING_FIELDS.has(k)),
  );
  const defaultName = nameOf(draft.routing.defaultId);
  const fallbackName = nameOf(draft.routing.fallbackId) ?? draft.routing.fallbackMissing;
  const routing: RoutingView = {
    ...rest,
    ...(defaultName === undefined ? {} : { default: defaultName }),
    ...(fallbackName == null ? {} : { local_fallback: fallbackName }),
    ...(Object.keys(features).length > 0 ? { features } : {}),
  };
  const outputLanguage =
    draft.outputLanguage !== base.output_language ? draft.outputLanguage : latest.output_language;
  return { profiles: Object.fromEntries(profiles), routing, output_language: outputLanguage };
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
