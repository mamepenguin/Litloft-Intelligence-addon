import {
  FILE_PAGE_CLOSED_EVENT,
  type FilePageClosedDetail,
} from "@/lib/addonEvents";

import type { AskImage, FileCitation } from "./askStream";

export type AskTurnStatus = "streaming" | "answered" | "failed";

/** The line the tab shows under the question when the answer cannot be made. */
export type AskFailure =
  | "no_model"
  | "profile_unavailable"
  | "busy"
  | "picture_needs_vision"
  | "image_unusable"
  | "video_needs_vision"
  | "frames_unusable"
  | "frames_rejected"
  | "generic";

export interface AskTurn {
  /** The question as it was sent. */
  question: string;
  answer: string;
  citations: FileCitation[] | null;
  partial: boolean;
  /** `meta.image` of this turn; read only while `answered`. */
  image: AskImage | null;
  status: AskTurnStatus;
  failure: AskFailure | null;
}

/** A note saved from this thread, and how many turns it held at the save. */
export interface SavedAskNote {
  fileId: string;
  path: string;
  answeredCount: number;
}

export interface AskThread {
  turns: AskTurn[];
  /** The text box's current contents: empty after a send, the draft otherwise. */
  text: string;
  controller: AbortController | null;
  savedNote: SavedAskNote | null;
}

const EMPTY: AskThread = Object.freeze({
  turns: [],
  text: "",
  controller: null,
  savedNote: null,
});

const threads = new Map<string, AskThread>();
const listeners = new Set<() => void>();

function emit(): void {
  for (const listener of listeners) listener();
}

function write(fileId: string, thread: AskThread): void {
  threads.set(fileId, thread);
  emit();
}

export function subscribeAskStore(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function getAskThread(fileId: string): AskThread {
  return threads.get(fileId) ?? EMPTY;
}

export function setAskText(fileId: string, text: string): void {
  write(fileId, { ...getAskThread(fileId), text });
}

function updateTurn(
  fileId: string,
  patch: (turn: AskTurn) => Partial<AskTurn>,
): void {
  const thread = getAskThread(fileId);
  const index = thread.turns.length - 1;
  if (index < 0) return;
  const turns = thread.turns.slice();
  turns[index] = { ...turns[index], ...patch(turns[index]) };
  write(fileId, { ...thread, turns });
}

export function beginAsk(
  fileId: string,
  question: string,
  controller: AbortController,
): void {
  const thread = getAskThread(fileId);
  write(fileId, {
    ...thread,
    turns: [
      ...thread.turns,
      {
        question,
        answer: "",
        citations: null,
        partial: false,
        image: null,
        status: "streaming",
        failure: null,
      },
    ],
    text: "",
    controller,
  });
}

export function appendAskAnswer(fileId: string, delta: string): void {
  updateTurn(fileId, (turn) => ({ answer: turn.answer + delta }));
}

export function setAskCitations(
  fileId: string,
  citations: FileCitation[],
): void {
  updateTurn(fileId, () => ({ citations }));
}

export function setAskPartial(fileId: string, partial: boolean): void {
  updateTurn(fileId, () => ({ partial }));
}

export function setAskImage(fileId: string, image: AskImage): void {
  updateTurn(fileId, () => ({ image }));
}

export function settleAsk(
  fileId: string,
  status: "answered" | "failed",
  failure: AskFailure | null,
): void {
  const thread = getAskThread(fileId);
  const index = thread.turns.length - 1;
  if (index < 0) return;
  const turns = thread.turns.slice();
  const turn = turns[index];
  turns[index] = {
    ...turn,
    status,
    failure,
    image: status === "failed" ? null : turn.image,
  };
  write(fileId, {
    ...thread,
    turns,
    controller: null,
    // A failed question goes back into an empty box so one press resends it;
    // text typed while it streamed is the viewer's and is left alone.
    text: status === "failed" && thread.text === "" ? turn.question : thread.text,
  });
}

/** Empty the thread and forget its saved note, keeping the text box. */
export function clearAskThread(fileId: string): void {
  const thread = getAskThread(fileId);
  write(fileId, {
    ...thread,
    turns: [],
    controller: null,
    savedNote: null,
  });
}

export function recordAskSave(
  fileId: string,
  note: { fileId: string; path: string },
): void {
  const thread = getAskThread(fileId);
  write(fileId, {
    ...thread,
    savedNote: {
      fileId: note.fileId,
      path: note.path,
      answeredCount: thread.turns.filter((turn) => turn.status === "answered")
        .length,
    },
  });
}

/** Abort a stream and forget the file's whole thread. */
export function dropAskEntry(fileId: string): void {
  const thread = threads.get(fileId);
  if (!thread) return;
  thread.controller?.abort();
  threads.delete(fileId);
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
