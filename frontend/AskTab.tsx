"use client";

import {
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
import { AnswerMarkdown } from "./AnswerMarkdown";
import { choiceLabel, choiceOffered, useLLMChoices } from "./llmChoice";
import { fetchFileAskAvailability, streamFileQuestion } from "./askStream";
import {
  appendAskAnswer,
  beginAsk,
  dropAskEntry,
  getAskEntry,
  setAskCitations,
  setAskImage,
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

/** The position the player shows at the moment the question is submitted. */
function positionAtSubmit(
  mediaController: MediaController | null,
): number | undefined {
  if (!mediaController) return undefined;
  if (mediaController.isInterrupted?.()) return undefined;
  const seconds = mediaController.getCurrentTime?.();
  if (typeof seconds !== "number" || !Number.isFinite(seconds) || seconds < 0) {
    return undefined;
  }
  return seconds;
}

function failureFor(err: unknown): AskFailure {
  if (isProfileUnavailable(err)) return "profile_unavailable";
  if (err instanceof LLMRequestError) {
    if (err.status === 400) return "no_model";
    if (err.status === 409 && err.detail === "picture_needs_vision") {
      return "picture_needs_vision";
    }
    if (err.status === 503 && err.detail === BUSY_DETAIL) return "busy";
  }
  return "generic";
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

      const positionSeconds = positionAtSubmit(mediaController);
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
          positionSeconds,
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
              if (event.image !== undefined) setAskImage(fileId, event.image);
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
          settleAsk(
            fileId,
            "failed",
            doneError === "image_unusable" ? "image_unusable" : "generic",
          );
        }
      } catch (err) {
        if (controller.signal.aborted) return;
        const failure = failureFor(err);
        if (failure === "profile_unavailable") refetchChoices();
        settleAsk(fileId, "failed", failure);
      }
    },
    [fileId, drive, askProfile, mediaController, refetchChoices],
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
              <AnswerMarkdown
                source={entry.answer}
                citations={entry.citations}
                mediaController={mediaController}
                documentController={documentController}
                onJump={() => lowerInspector?.()}
              />
            )}
            {entry.status === "answered" && entry.image === "unavailable" && (
              <p className="text-xs text-text-muted">{t("imageUnavailable")}</p>
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
                      : entry.failure === "picture_needs_vision"
                        ? t("pictureNeedsVision")
                        : entry.failure === "image_unusable"
                          ? t("imageUnusable")
                          : t("failed")}
              </p>
            )}
          </section>
        )}
      </div>
    </div>
  );
}
