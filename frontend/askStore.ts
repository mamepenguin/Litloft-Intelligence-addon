import {
  FILE_PAGE_CLOSED_EVENT,
  type FilePageClosedDetail,
} from "@/lib/addonEvents";

import type { AskImage, FileCitation } from "./askStream";

export type AskStatus = "idle" | "streaming" | "answered" | "failed";

/** The line the tab shows under the question when the answer cannot be made. */
export type AskFailure =
  | "no_model"
  | "profile_unavailable"
  | "busy"
  | "picture_needs_vision"
  | "image_unusable"
  | "generic";

export interface AskEntry {
  /** The box's current text: the question that was sent, or the next one being typed. */
  question: string;
  /** The question shown in the conversation, set when it is sent. */
  asked: string;
  answer: string;
  citations: FileCitation[] | null;
  partial: boolean;
  /** `meta.image` of the last question; read only while `answered`. */
  image: AskImage | null;
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
  image: null,
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
    image: null,
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

export function setAskImage(fileId: string, image: AskImage): void {
  update(fileId, () => ({ image }));
}

export function settleAsk(
  fileId: string,
  status: "answered" | "failed",
  failure: AskFailure | null,
): void {
  update(fileId, (entry) => ({
    status,
    failure,
    controller: null,
    image: status === "failed" ? null : entry.image,
  }));
}

/** Abort a stream and forget the file's question and answer. */
export function dropAskEntry(fileId: string): void {
  const entry = entries.get(fileId);
  if (!entry) return;
  entry.controller?.abort();
  entries.delete(fileId);
  emit();
}

/** Registered here rather than from the tab, which a sheet close unmounts while its answer streams. */
function dropOnPageClosed(event: Event): void {
  const fileId = (event as CustomEvent<FilePageClosedDetail>).detail?.fileId;
  if (typeof fileId === "string") dropAskEntry(fileId);
}

if (typeof window !== "undefined") {
  window.addEventListener(FILE_PAGE_CLOSED_EVENT, dropOnPageClosed);
}
