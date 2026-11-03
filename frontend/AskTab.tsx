"use client";

import {
  Fragment,
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
  type FormEvent,
} from "react";
import { useTranslations } from "next-intl";

import { useWebSocket } from "@/hooks/useWebSocket";
import { Button } from "@/components/Button";
import type { DocumentController } from "@/lib/documentController";
import type { MediaController } from "@/lib/mediaController";

import { isProfileUnavailable, LLMRequestError } from "./api";
import { choiceLabel, choiceOffered, useLLMChoices } from "./llmChoice";
import {
  fetchFileAskAvailability,
  streamFileQuestion,
  type CitationPosition,
  type FileCitation,
} from "./askStream";
import {
  appendAskAnswer,
  beginAsk,
  dropAskEntry,
  getAskEntry,
  releaseAskEntry,
  setAskCitations,
  setAskPartial,
  setAskQuestion,
  settleAsk,
  subscribeAskStore,
  type AskFailure,
} from "./askStore";

interface AskTabProps {
  fileId: string;
  drive: string;
  fileType?: string;
  mediaController?: MediaController | null;
  documentController?: DocumentController | null;
  lowerInspector?: () => void;
  fillHeight?: boolean;
  labelledByHost?: boolean;
  onAvailability?: (available: boolean) => void;
}

const MIN_QUESTION_CHARS = 3;
const MAX_QUESTION_CHARS = 1000;
/** `/ask`'s answer for a full semaphore; any other 503 is a generic failure. */
const BUSY_DETAIL = "Too many concurrent Ask requests, please retry shortly";

const useIsomorphicLayoutEffect =
  typeof window !== "undefined" ? useLayoutEffect : useEffect;

function nonSpaceLength(text: string): number {
  let count = 0;
  for (const char of text) {
    if (!/\s/.test(char)) count += 1;
  }
  return count;
}

/** `M:SS` below one hour, `H:MM:SS` from one hour. */
function formatSeconds(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const rest = total % 60;
  const mm = String(minutes).padStart(2, "0");
  const ss = String(rest).padStart(2, "0");
  return hours > 0 ? `${hours}:${mm}:${ss}` : `${minutes}:${ss}`;
}

function citationLabel(position: CitationPosition): string | null {
  if (!position) return null;
  if ("seconds" in position) return formatSeconds(position.seconds);
  if ("page" in position) return `p.${position.page}`;
  return position.title ?? `§${position.section}`;
}

function jumpAction(
  position: CitationPosition,
  mediaController: MediaController | null,
  documentController: DocumentController | null,
): (() => void) | null {
  if (!position) return null;
  if ("seconds" in position) {
    return mediaController ? () => mediaController.seek(position.seconds) : null;
  }
  if ("page" in position) {
    return documentController
      ? () => documentController.goTo({ page: position.page })
      : null;
  }
  return documentController
    ? () => documentController.goTo({ section: position.section })
    : null;
}

function failureFor(err: unknown): AskFailure {
  if (isProfileUnavailable(err)) return "profile_unavailable";
  if (err instanceof LLMRequestError) {
    if (err.status === 400) return "no_model";
    if (err.status === 503 && err.detail === BUSY_DETAIL) return "busy";
  }
  return "generic";
}

const PILL_CLASS =
  "mx-0.5 inline rounded-full px-1.5 py-0.5 align-baseline text-text-muted transition-colors hover:bg-accent/10 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-focus-ring";

interface AnswerTextProps {
  text: string;
  citations: FileCitation[] | null;
  mediaController: MediaController | null;
  documentController: DocumentController | null;
  onJump: () => void;
}

/**
 * The answer with each `[n]` replaced by its cited position once the
 * citations arrive. Before that the markers are shown exactly as typed.
 */
function AnswerText({
  text,
  citations,
  mediaController,
  documentController,
  onJump,
}: AnswerTextProps) {
  if (citations === null) return <>{text}</>;

  const positions = new Map(citations.map((citation) => [citation.n, citation.position]));

  return (
    <>
      {text.split(/(\[\d+\])/g).map((part, index) => {
        const marker = /^\[(\d+)\]$/.exec(part);
        if (!marker) return <Fragment key={index}>{part}</Fragment>;
        const position = positions.get(Number(marker[1]));
        if (position === undefined) return null;
        const label = citationLabel(position);
        if (label === null) return null;
        const action = jumpAction(position, mediaController, documentController);
        if (!action) {
          return (
            <span key={index} className="mx-0.5 inline text-text-muted">
              {label}
            </span>
          );
        }
        return (
          <button
            key={index}
            type="button"
            onClick={() => {
              action();
              onJump();
            }}
            className={PILL_CLASS}
          >
            {label}
          </button>
        );
      })}
    </>
  );
}

export default function AskTab({
  fileId,
  drive,
  fileType,
  mediaController = null,
  documentController = null,
  lowerInspector,
  fillHeight = false,
  onAvailability,
}: AskTabProps) {
  const t = useTranslations("askTab");
  const tc = useTranslations("llmChoice");

  const entry = useSyncExternalStore(
    subscribeAskStore,
    () => getAskEntry(fileId),
    () => getAskEntry(fileId),
  );

  const availabilityRef = useRef(onAvailability);
  useIsomorphicLayoutEffect(() => {
    availabilityRef.current = onAvailability;
  });

  const [available, setAvailable] = useState(false);
  const [availabilityGeneration, setAvailabilityGeneration] = useState(0);

  // The host lists an `inspector-tabs` entry only after it says yes, so the
  // first answer of every file is "not yet".
  useIsomorphicLayoutEffect(() => {
    availabilityRef.current?.(false);
    setAvailable(false);
  }, [fileId, drive]);

  useEffect(() => {
    const controller = new AbortController();
    let cancelled = false;
    void fetchFileAskAvailability(fileId, drive, controller.signal).then(
      (canAsk) => {
        if (cancelled) return;
        setAvailable(canAsk);
        availabilityRef.current?.(canAsk);
      },
    );
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [fileId, drive, availabilityGeneration]);

  const isMedia = fileType === "video" || fileType === "audio";
  const transcriptionEvent = useWebSocket(
    "intelligence.transcription.completed",
  );
  useEffect(() => {
    if (!isMedia || transcriptionEvent?.data.file_id !== fileId) return;
    setAvailabilityGeneration((generation) => generation + 1);
  }, [isMedia, transcriptionEvent, fileId]);

  const previousFileRef = useRef(fileId);
  useIsomorphicLayoutEffect(() => {
    const previous = previousFileRef.current;
    if (previous !== fileId) {
      dropAskEntry(previous);
      previousFileRef.current = fileId;
    }
  }, [fileId]);

  useEffect(() => {
    return () => releaseAskEntry(fileId);
  }, [fileId]);

  const { choices, refetch: refetchChoices } = useLLMChoices(
    "rag",
    drive,
    available,
  );
  const profileOffered = choiceOffered(choices);
  const [chosenProfile, setChosenProfile] = useState<string | null>(null);
  useEffect(() => {
    setChosenProfile(null);
  }, [drive]);

  const askProfile = useMemo(() => {
    if (!choices || !profileOffered) return undefined;
    if (
      chosenProfile !== null
      && choices.choices.some((choice) => choice.name === chosenProfile)
    ) {
      return chosenProfile;
    }
    return choices.auto === null ? choices.choices[0]?.name : undefined;
  }, [choices, profileOffered, chosenProfile]);

  const ask = useCallback(
    async (question: string) => {
      const trimmed = question.trim();
      if (nonSpaceLength(trimmed) < MIN_QUESTION_CHARS) return;
      if (getAskEntry(fileId).status === "streaming") return;

      const controller = new AbortController();
      beginAsk(fileId, trimmed, controller);
      let sawDone = false;
      let doneError: string | undefined;

      try {
        stream: for await (const event of streamFileQuestion({
          fileId,
          drive,
          question: trimmed,
          profile: askProfile,
          signal: controller.signal,
        })) {
          if (controller.signal.aborted) return;
          switch (event.kind) {
            case "answer_chunk":
              appendAskAnswer(fileId, event.delta);
              break;
            case "citations":
              setAskCitations(fileId, event.citations);
              break;
            case "meta":
              setAskPartial(fileId, event.partial);
              break;
            case "done":
              sawDone = true;
              doneError = event.error;
              break stream;
          }
        }
        if (controller.signal.aborted) return;
        if (sawDone && doneError === undefined) {
          settleAsk(fileId, "answered", null);
        } else {
          settleAsk(fileId, "failed", "generic");
        }
      } catch (err) {
        if (controller.signal.aborted) return;
        const failure = failureFor(err);
        if (failure === "profile_unavailable") refetchChoices();
        settleAsk(fileId, "failed", failure);
      }
    },
    [fileId, drive, askProfile, refetchChoices],
  );

  const submit = useCallback(
    (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      if (entry.status === "streaming") return;
      if (nonSpaceLength(entry.question) < MIN_QUESTION_CHARS) return;
      void ask(entry.question);
    },
    [ask, entry.question, entry.status],
  );

  const canSubmit =
    entry.status !== "streaming"
    && nonSpaceLength(entry.question) >= MIN_QUESTION_CHARS;

  return (
    <div
      className={
        fillHeight ? "flex h-full min-h-0 flex-col" : "flex flex-col"
      }
    >
      <form
        onSubmit={submit}
        className={
          fillHeight
            ? "shrink-0 space-y-2 bg-bg-card pb-2"
            : "space-y-2"
        }
      >
        <textarea
          value={entry.question}
          onChange={(event) => setAskQuestion(fileId, event.target.value)}
          placeholder={t("placeholder")}
          maxLength={MAX_QUESTION_CHARS}
          rows={3}
          aria-label={t("placeholder")}
          className="w-full resize-y rounded-2xl border border-bg-border bg-bg-card p-3 text-sm text-text-primary placeholder:text-text-muted focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-focus-ring"
        />
        <div className="flex flex-wrap items-center justify-end gap-2">
          {profileOffered && choices && (
            <select
              value={askProfile ?? ""}
              onChange={(event) => setChosenProfile(event.target.value || null)}
              disabled={entry.status === "streaming"}
              aria-label={tc("model")}
              className="max-w-36 truncate rounded-lg border border-bg-border bg-bg-card px-2 py-1.5 text-xs text-text-primary focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-focus-ring disabled:opacity-50"
            >
              {choices.auto !== null && (
                <option value="">{tc("auto")}</option>
              )}
              {choices.choices.map((choice) => (
                <option key={choice.name} value={choice.name}>
                  {choiceLabel(choice, tc("offhost"))}
                </option>
              ))}
            </select>
          )}
          <Button
            type="submit"
            variant="primary"
            size="md"
            disabled={!canSubmit}
          >
            {t("submit")}
          </Button>
        </div>
      </form>

      <div
        className={
          fillHeight
            ? "min-h-0 flex-1 space-y-2 overflow-y-auto pt-2"
            : "space-y-2 pt-2"
        }
      >
        {entry.status !== "idle" && (
          <section aria-live="polite" className="space-y-2">
            <p className="whitespace-pre-wrap text-sm font-medium text-text-primary">
              {entry.asked}
            </p>
            {entry.answer !== "" && (
              <div className="whitespace-pre-wrap text-sm leading-relaxed text-text-primary">
                <AnswerText
                  text={entry.answer}
                  citations={entry.citations}
                  mediaController={mediaController}
                  documentController={documentController}
                  onJump={() => lowerInspector?.()}
                />
              </div>
            )}
            {entry.partial && (
              <p className="text-xs text-text-muted">{t("partial")}</p>
            )}
            {entry.failure && (
              <p role="alert" className="text-sm text-danger">
                {entry.failure === "no_model"
                  ? t("noModel")
                  : entry.failure === "profile_unavailable"
                    ? tc("profileUnavailable")
                    : entry.failure === "busy"
                      ? t("busy")
                      : t("failed")}
              </p>
            )}
          </section>
        )}
      </div>
    </div>
  );
}
