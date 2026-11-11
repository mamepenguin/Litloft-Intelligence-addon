import { vi } from "vitest";

const messages = vi.hoisted(() => ({
  scenesTitle: "Scenes",
  scenesCount: "{count} scenes",
  scenesProcessing: "Processing {progress}",
  scenesUpdating: "Updating {progress}",
  scenesCreate: "Create scenes",
  scenesRegenerate: "Regenerate scenes",
  scenesRetry: "Retry the {count} failed",
  scenesFailedNote: "{count} failed",
  scenesRuler: "Frame timeline",
  scenesPlayFrom: "Play from {time}",
  visualIndexStale: "The source scenes have changed since this index was built.",
  visualIndexGenerating: "Starting…",
  visualIndexWaitingPrerequisite:
    "Waiting on scene indexing to finish before this can start.",
  visualIndexActionError: "Could not start visual index generation.",
}));

vi.mock("next-intl", () => ({
  useTranslations: () => (key: string, values?: Record<string, unknown>) => {
    const template = (messages as Record<string, string>)[key] ?? key;
    return Object.entries(values ?? {}).reduce(
      (text, [name, value]) => text.replace(`{${name}}`, String(value)),
      template,
    );
  },
  useLocale: () => "en",
}));
