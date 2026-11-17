"use client";

import { useCallback, useEffect, useId, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { X } from "lucide-react";
import { useTranslations } from "next-intl";

import { correctTranscriptUtterance, LLMRequestError } from "./api";
import { Button } from "@/components/Button";
import { useDialogPortalTarget } from "@/components/DialogPortal";
import { useShortcuts } from "@/hooks/useShortcuts";
import { formatDuration } from "@/lib/format";
import { useImeKeyGuard } from "@/lib/ime";

/** The utterance a dialog opened on, as it was when it opened. */
export interface TranscriptEditTarget {
  index: number;
  start: number;
  text: string;
}

interface TranscriptEditDialogProps {
  fileId: string;
  drive: string;
  target: TranscriptEditTarget;
  /** The panel's code-to-message table, so both editors say the same thing. */
  errorText: (err: unknown, fallbackKey: string) => string;
  onClose: () => void;
  onSaved: () => void;
  onChanged: (message: string) => void;
}

/**
 * Correcting one utterance on a coarse pointer: where the inline field is a
 * single line that scrolls sideways, this one wraps and grows.
 *
 * The text and start it opened with are kept for the request's `expected_*`,
 * so a reload underneath does not move the target the correction is checked
 * against.
 */
export function TranscriptEditDialog({
  fileId,
  drive,
  target,
  errorText,
  onClose,
  onSaved,
  onChanged,
}: TranscriptEditDialogProps) {
  const t = useTranslations("searchIndex");
  const tc = useTranslations("common");
  const ime = useImeKeyGuard();
  const host = useDialogPortalTarget();
  const titleId = useId();
  const fieldRef = useRef<HTMLTextAreaElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  // The draft lives in the field itself and is read once, by 保存; putting the
  // text in and leaving the caret at its end is the whole of its setup.
  useEffect(() => {
    const field = fieldRef.current;
    if (!field) return;
    field.value = target.text;
    field.focus();
    field.setSelectionRange(target.text.length, target.text.length);
  }, [target.text]);

  const close = useCallback(() => {
    if (saving) return;
    onClose();
  }, [onClose, saving]);

  useShortcuts(
    "transcript-edit-dialog",
    "Dialog",
    [
      {
        key: "escape",
        label: "Cancel",
        hidden: true,
        // The field has focus the whole time, and a dismissal from a field is
        // still a dismissal. Keys the IME owns never reach the provider.
        editingOnly: false,
        handler: close,
      },
    ],
    true,
  );

  const save = useCallback(async () => {
    if (saving) return;
    setSaving(true);
    setError(null);
    try {
      await correctTranscriptUtterance(
        fileId,
        target.index,
        fieldRef.current?.value ?? "",
        target.text,
        target.start,
        drive,
      );
      onSaved();
    } catch (err) {
      const code =
        err instanceof LLMRequestError &&
        typeof err.detail === "object" &&
        err.detail !== null
          ? (err.detail as { code?: string }).code
          : undefined;
      // `changed` means the row is not the one this draft was written
      // against; the panel reloads and says so, as the inline editor does.
      if (code === "changed") onChanged(errorText(err, "transcriptSaveFailed"));
      else setError(errorText(err, "transcriptSaveFailed"));
    } finally {
      setSaving(false);
    }
  }, [
    saving,
    fileId,
    drive,
    target.index,
    target.text,
    target.start,
    onSaved,
    onChanged,
    errorText,
  ]);

  if (!host) return null;

  return createPortal(
    <div className="fixed inset-0 z-50">
      {/* No click handler: a stray tap must not throw away a typed draft. */}
      <div className="absolute inset-0 animate-fade-in bg-black/60" />
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        className="absolute inset-x-4 top-14 mx-auto flex max-w-md animate-fade-in flex-col gap-4 rounded-2xl bg-bg-card p-5 shadow-lg"
      >
        <div className="flex items-center gap-2">
          <h2
            id={titleId}
            className="text-lg font-semibold text-text-primary"
          >
            {t("transcriptEdit")}
            <span className="ml-2 font-mono text-xs font-normal text-text-muted">
              {formatDuration(target.start)}
            </span>
          </h2>
          <button
            type="button"
            onClick={close}
            disabled={saving}
            aria-label={tc("close")}
            className="ml-auto flex h-11 w-11 shrink-0 items-center justify-center rounded-lg text-text-muted transition-colors enabled:hover:text-text-primary disabled:opacity-50"
          >
            <X size={18} />
          </button>
        </div>
        <textarea
          ref={fieldRef}
          aria-label={t("transcriptEdit")}
          readOnly={saving}
          onChange={(event) => {
            // A line break belongs to the single line this replaces; one that
            // arrives by paste is dropped rather than stored.
            const text = event.target.value.replace(/[\r\n]/g, "");
            if (text !== event.target.value) event.target.value = text;
            setError(null);
          }}
          onCompositionEnd={ime.onCompositionEnd}
          onKeyDown={(event) => {
            if (ime.isImeKeystroke(event)) return;
            // Return saves nowhere but 保存, and a textarea would otherwise
            // take it as a line break.
            if (event.key === "Enter") {
              event.preventDefault();
              event.stopPropagation();
            }
          }}
          className="field-sizing-content max-h-[142px] min-h-[118px] w-full resize-none overflow-y-auto rounded-2xl border border-bg-border bg-bg-elevated px-3 py-2.5 text-base leading-6 text-text-primary outline-none focus-visible:ring-2 focus-visible:ring-focus-ring"
        />
        {error && (
          <p className="break-words text-xs text-danger" role="status">
            {error}
          </p>
        )}
        <div className="flex justify-end gap-3">
          <Button variant="secondary" onClick={close} disabled={saving}>
            {t("transcriptCancel")}
          </Button>
          <Button
            variant="primary"
            onClick={() => void save()}
            disabled={saving}
          >
            {t("transcriptSave")}
          </Button>
        </div>
      </div>
    </div>,
    host,
  );
}
