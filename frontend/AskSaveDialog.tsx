"use client";

/**
 * Simplified save dialog for Ask → Knowledge.
 *
 * Unlike KnowledgeSaveDialog (which is tied to a single source file
 * and checks for existing distill notes), this dialog accepts
 * pre-formatted Markdown content with multiple source_file_ids and
 * posts to POST /addons/knowledge/notes.
 */
import { useCallback, useEffect, useState } from "react";
import { useTranslations } from "next-intl";
import { X } from "lucide-react";

import { saveAskToKnowledge } from "./knowledgeBridge";

interface Props {
  open: boolean;
  drive: string;
  defaultFilename: string;
  content: string;
  sourceFileIds: string[];
  onClose: () => void;
  onSaved: (result: { noteFileId: string; notePath: string }) => void;
}

const DEFAULT_FOLDER = "Ask";

export function AskSaveDialog({
  open,
  drive,
  defaultFilename,
  content,
  sourceFileIds,
  onClose,
  onSaved,
}: Props) {
  const t = useTranslations("askSave");
  const tc = useTranslations("common");

  const [folder, setFolder] = useState(DEFAULT_FOLDER);
  const [filename, setFilename] = useState(defaultFilename);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    setFilename(defaultFilename);
    setFolder(DEFAULT_FOLDER);
    setSubmitting(false);
    setSubmitError(null);
  }, [open, defaultFilename]);

  useEffect(() => {
    if (!open) return;
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  const handleSubmit = useCallback(async () => {
    setSubmitting(true);
    setSubmitError(null);
    try {
      const res = await saveAskToKnowledge(drive, {
        folder: folder.trim() || DEFAULT_FOLDER,
        filename: filename.trim() || defaultFilename,
        content,
        source_file_ids: sourceFileIds,
      });
      onSaved({ noteFileId: res.note_file_id, notePath: res.note_path });
      onClose();
    } catch (err) {
      setSubmitError((err as Error).message);
    } finally {
      setSubmitting(false);
    }
  }, [drive, folder, filename, defaultFilename, content, sourceFileIds, onSaved, onClose]);

  if (!open) return null;

  const inputClass =
    "w-full rounded-lg border border-bg-border bg-bg-primary px-3 py-2 text-sm text-text-primary focus:border-focus-ring focus:outline-none";
  const labelClass = "block text-xs font-medium text-text-muted mb-1";

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
      role="dialog"
      aria-modal
      aria-label={t("title")}
    >
      <div className="flex w-full max-w-md flex-col gap-4 rounded-xl bg-bg-card p-5 shadow-lg">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-text-primary">{t("title")}</h2>
          <button
            type="button"
            onClick={onClose}
            className="flex h-7 w-7 items-center justify-center rounded-lg text-text-muted hover:bg-bg-elevated hover:text-text-primary"
            aria-label={tc("close")}
          >
            <X size={14} />
          </button>
        </div>

        <div className="flex flex-col gap-3">
          <div>
            <label className={labelClass}>{t("folder")}</label>
            <input
              className={inputClass}
              value={folder}
              onChange={(e) => setFolder(e.target.value)}
              placeholder="Ask"
            />
          </div>
          <div>
            <label className={labelClass}>{t("filename")}</label>
            <input
              className={inputClass}
              value={filename}
              onChange={(e) => setFilename(e.target.value)}
            />
          </div>
          {submitError && <p className="text-xs text-danger">{submitError}</p>}
          <div className="flex justify-end gap-2">
            <button
              type="button"
              onClick={onClose}
              className="rounded-lg border border-bg-border px-4 py-2 text-sm text-text-muted hover:bg-bg-elevated"
            >
              {tc("cancel")}
            </button>
            <button
              type="button"
              disabled={submitting}
              onClick={handleSubmit}
              className="rounded-2xl bg-accent px-4 py-2 text-sm font-medium text-white hover:bg-accent-hover disabled:opacity-50"
            >
              {submitting ? t("saving") : t("save")}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
