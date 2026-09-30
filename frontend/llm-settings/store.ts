import {
  fetchLLM,
  resetLLM,
  saveLLM,
  type LLMUpdateBody,
  type LLMView,
  type ProfileView,
  type SaveResult,
} from "./api";

/** The sections on the settings page that write `/admin/llm`. */
export type Writer = "llm" | "features";

type Listener = (writer: Writer) => void;

const listeners = new Set<Listener>();

export function subscribeLLMSaved(listener: Listener): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

function notify(writer: Writer): void {
  for (const listener of [...listeners]) listener(writer);
}

export const loadLLMView = fetchLLM;

const VIEW_ONLY_KEYS = new Set(["api_key_present", "api_key_source"]);

export function storedProfile(view: ProfileView): ProfileView {
  return Object.fromEntries(Object.entries(view).filter(([k]) => !VIEW_ONLY_KEYS.has(k)));
}

export function documentOf(view: LLMView): LLMUpdateBody {
  return {
    profiles: Object.fromEntries(
      Object.entries(view.profiles).map(([name, p]) => [name, storedProfile(p)]),
    ),
    routing: { ...view.routing },
    output_language: view.output_language,
  };
}

/**
 * Reads the document as it is now and writes `patch` applied to it, so a save
 * from one section does not carry the other section's stale copy.
 */
export async function saveLLMPatch(
  writer: Writer,
  patch: (latest: LLMView) => LLMUpdateBody,
): Promise<SaveResult> {
  const latest = await fetchLLM();
  const result = await saveLLM(patch(latest));
  notify(writer);
  return result;
}

export async function resetLLMDocument(writer: Writer): Promise<SaveResult> {
  const result = await resetLLM();
  notify(writer);
  return result;
}

/** `choices`: feature -> profile name, or "" to fall back to the default profile. */
export function withFeatureChoices(
  latest: LLMView,
  choices: Record<string, string>,
): LLMUpdateBody {
  const doc = documentOf(latest);
  const merged = { ...(latest.routing.features ?? {}), ...choices };
  const features = Object.fromEntries(Object.entries(merged).filter(([, name]) => name !== ""));
  const routing = Object.fromEntries(
    Object.entries(doc.routing).filter(([k]) => k !== "features"),
  );
  return {
    ...doc,
    routing: Object.keys(features).length > 0 ? { ...routing, features } : routing,
  };
}
