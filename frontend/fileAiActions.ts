"use client";

/**
 * The one place a file's "make something for me" actions collect.
 *
 * Five sections on the file detail page can each produce something —
 * tag candidates, a summary, a detailed summary, chapter candidates, an
 * image description. Before a file has any of them, each section still
 * drew a heading and a button, so a freshly indexed video opened onto
 * five rows that said only that five things could be made. The rows are
 * gone; the offers arrive here instead and the action row shows one
 * "AI" menu.
 *
 * A section keeps owning its own generation: it already knows the
 * endpoint, the polling, and what to do with the result, and it is the
 * component that has to re-render when the content lands. So an offer
 * carries a callback back into the section rather than a copy of its
 * logic, and the section withdraws the offer the moment it has content
 * to show — which is exactly when its own heading takes over.
 *
 * A module-level store rather than a context: the menu and the sections
 * are mounted by two different `AddonSlot`s with no shared ancestor
 * inside this addon, so there is no element a provider could wrap.
 */

import { useCallback, useEffect, useRef, useSyncExternalStore } from "react";

export type FileAiActionKind =
  | "tags"
  | "summary"
  | "detailedSummary"
  | "chapters"
  | "visualDescription";

/**
 * Menu order. Fixed rather than mount order, which varies with how fast
 * each section's fetch resolves and would otherwise reshuffle the menu
 * between page loads.
 */
const ORDER: readonly FileAiActionKind[] = [
  "tags",
  "summary",
  "detailedSummary",
  "chapters",
  "visualDescription",
];

export interface FileAiAction {
  kind: FileAiActionKind;
  /** Message key in the `file` namespace. */
  labelKey: string;
  /** A run of this action is already in flight. */
  busy: boolean;
  run: () => void;
}

const EMPTY: readonly FileAiAction[] = [];

const registry = new Map<string, FileAiAction>();
const listeners = new Set<() => void>();
let snapshots = new Map<string, readonly FileAiAction[]>();

function slot(fileId: string, kind: FileAiActionKind): string {
  return `${fileId} ${kind}`;
}

function publish(): void {
  // Snapshots are what `useSyncExternalStore` compares by identity, so
  // they must be rebuilt on change and stable in between — returning a
  // freshly filtered array on every read would re-render forever.
  snapshots = new Map();
  for (const listener of listeners) listener();
}

function offer(fileId: string, action: FileAiAction): void {
  registry.set(slot(fileId, action.kind), action);
  publish();
}

function withdraw(fileId: string, kind: FileAiActionKind): void {
  if (registry.delete(slot(fileId, kind))) publish();
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

function read(fileId: string): readonly FileAiAction[] {
  const cached = snapshots.get(fileId);
  if (cached) return cached;
  const built = ORDER.map((kind) => registry.get(slot(fileId, kind))).filter(
    (action): action is FileAiAction => action !== undefined,
  );
  const result = built.length > 0 ? built : EMPTY;
  snapshots.set(fileId, result);
  return result;
}

/** What the "AI" menu should currently list for this file. */
export function useFileAiActions(fileId: string): readonly FileAiAction[] {
  const getSnapshot = useCallback(() => read(fileId), [fileId]);
  const getServerSnapshot = useCallback(() => EMPTY, []);
  return useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);
}

/**
 * Offer this file's "AI" menu a way to generate something, for as long
 * as `active` holds.
 *
 * Call it unconditionally, above the section's early returns — a
 * section that has nothing to show still has an offer to make, and that
 * is precisely the state in which it renders nothing.
 */
export function useOfferFileAiAction(params: {
  fileId: string;
  kind: FileAiActionKind;
  labelKey: string;
  active: boolean;
  busy?: boolean;
  run: () => void;
}): void {
  const { fileId, kind, labelKey, active, busy = false, run } = params;

  // The callback is re-created on most renders; the registry entry must
  // not be, or every render would republish and re-render the menu.
  const runRef = useRef(run);
  useEffect(() => {
    runRef.current = run;
  });

  useEffect(() => {
    if (!active) return;
    offer(fileId, {
      kind,
      labelKey,
      busy,
      run: () => runRef.current(),
    });
    return () => withdraw(fileId, kind);
  }, [fileId, kind, labelKey, active, busy]);
}

/** Test seam: drop every registration. */
export function resetFileAiActions(): void {
  registry.clear();
  publish();
}
