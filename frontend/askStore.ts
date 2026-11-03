import type { FileCitation } from "./askStream";

export type AskStatus = "idle" | "streaming" | "answered" | "failed";

/** The line the tab shows under the question when the answer cannot be made. */
export type AskFailure =
  | "no_model"
  | "profile_unavailable"
  | "busy"
  | "generic";

export interface AskEntry {
  /** The box's current text: the question that was sent, or the next one being typed. */
  question: string;
  /** The question shown in the conversation, set when it is sent. */
  asked: string;
  answer: string;
  citations: FileCitation[] | null;
  partial: boolean;
  status: AskStatus;
  failure: AskFailure | null;
  controller: AbortController | null;
}

const EMPTY: AskEntry = Object.freeze({
  question: "",
  asked: "",
  answer: "",
  citations: null,
  partial: false,
  status: "idle",
  failure: null,
  controller: null,
});

const entries = new Map<string, AskEntry>();
const listeners = new Set<() => void>();

function emit(): void {
  for (const listener of listeners) listener();
}

function write(fileId: string, entry: AskEntry): void {
  entries.set(fileId, entry);
  emit();
}

export function subscribeAskStore(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function getAskEntry(fileId: string): AskEntry {
  return entries.get(fileId) ?? EMPTY;
}

export function setAskQuestion(fileId: string, question: string): void {
  write(fileId, { ...getAskEntry(fileId), question });
}

export function beginAsk(
  fileId: string,
  question: string,
  controller: AbortController,
): void {
  write(fileId, {
    question,
    asked: question,
    answer: "",
    citations: null,
    partial: false,
    status: "streaming",
    failure: null,
    controller,
  });
}

function update(fileId: string, patch: (entry: AskEntry) => Partial<AskEntry>): void {
  write(fileId, { ...getAskEntry(fileId), ...patch(getAskEntry(fileId)) });
}

export function appendAskAnswer(fileId: string, delta: string): void {
  update(fileId, (entry) => ({ answer: entry.answer + delta }));
}

export function setAskCitations(fileId: string, citations: FileCitation[]): void {
  update(fileId, () => ({ citations }));
}

export function setAskPartial(fileId: string, partial: boolean): void {
  update(fileId, () => ({ partial }));
}

export function settleAsk(
  fileId: string,
  status: "answered" | "failed",
  failure: AskFailure | null,
): void {
  update(fileId, () => ({ status, failure, controller: null }));
}

/** Abort a stream and forget the file's question and answer. */
export function dropAskEntry(fileId: string): void {
  const entry = entries.get(fileId);
  if (!entry) return;
  entry.controller?.abort();
  entries.delete(fileId);
  emit();
}

/**
 * Abort a stream when the tab goes away but keep a settled answer, so a
 * tab switch or a closed sheet finds the same conversation.
 */
export function releaseAskEntry(fileId: string): void {
  const entry = entries.get(fileId);
  if (!entry) return;
  entry.controller?.abort();
  if (entry.status === "streaming") {
    entries.delete(fileId);
  } else if (entry.controller !== null) {
    entries.set(fileId, { ...entry, controller: null });
  }
  emit();
}
