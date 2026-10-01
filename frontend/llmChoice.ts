"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import {
  fetchLLMChoices,
  type LLMChoiceFeature,
  type LLMChoicesResponse,
} from "./api";

/**
 * With one choice that is also what routing picks, the menu would only
 * repeat the plain button. With `auto` null the plain button cannot
 * generate at all, so a single choice is worth offering.
 */
export function choiceOffered(choices: LLMChoicesResponse | null): boolean {
  if (choices === null) return false;
  return (
    choices.choices.length >= 2
    || (choices.auto === null && choices.choices.length >= 1)
  );
}

export function choiceLabel(
  choice: { name: string; model: string; offhost: boolean },
  offhostText: string,
): string {
  const base = `${choice.name} — ${choice.model}`;
  return choice.offhost ? `${base} (${offhostText})` : base;
}

/**
 * The profiles this drive may use for `feature`, fetched while `enabled`.
 * `null` until loaded, after a failed request, and while disabled.
 */
export function useLLMChoices(
  feature: LLMChoiceFeature,
  drive: string,
  enabled: boolean,
): { choices: LLMChoicesResponse | null; refetch: () => void } {
  const [choices, setChoices] = useState<LLMChoicesResponse | null>(null);
  const [generation, setGeneration] = useState(0);
  const tokenRef = useRef(0);

  useEffect(() => {
    const token = ++tokenRef.current;
    setChoices(null);
    if (!enabled || !drive) return;
    void (async () => {
      try {
        const result = await fetchLLMChoices(feature, drive);
        if (token !== tokenRef.current) return;
        setChoices(Array.isArray(result?.choices) ? result : null);
      } catch {
        if (token === tokenRef.current) setChoices(null);
      }
    })();
  }, [feature, drive, enabled, generation]);

  const refetch = useCallback(() => setGeneration((g) => g + 1), []);

  return { choices, refetch };
}
