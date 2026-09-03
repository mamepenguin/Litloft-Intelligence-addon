"use client";

/**
 * Dialog that promotes an edited detailed_summary into a Knowledge ``.md``
 * via the knowledge addon's ``/distill`` endpoint.
 *
 * Opens on "knowledge に保存" button click in DetailedSummarySection.
 * Handles three cases before showing the save form:
 *
 *   A. Exactly one prior promotion from this source file — default to
 *      "open existing", offering "create new" as a secondary action
 *      (保守的 default — avoid accidentally growing .md count).
 *   B. Multiple prior promotions — same as A, with a disclosure for
 *      the older notes.
 *   C. No prior promotion — show the save form (folder + filename + title).
 *
 * On success, closes itself and calls `onSaved(noteFileId)` so the
 * parent component can refresh `useActiveSummary` (or rely on the WS
 * event `core.file_active_summary.changed`).
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import { useTranslations } from "next-intl";
import { X } from "lucide-react";

import { FolderPicker } from "@/components/FolderPicker";
import { useShortcuts } from "@/hooks/useShortcuts";
import { OVERLAY_PRIORITY } from "@/lib/shortcuts";
import {
  distillToKnowledge,
  getNotesBySourceFile,
  type NoteOrigin,
} from "./knowledgeBridge";

export interface KnowledgeSaveDialogProps {
  open: boolean;
  fileId: string;
  drive: string;
  /** The full detailed_summary markdown body being promoted. */
  content: string;
  /** Source file's filename (with extension); used to seed the default
   * note filename stem and the H1 title. */
  sourceFilename: string;
  onClose: () => void;
  onSaved: (result: { noteFileId: string; notePath: string }) => void;
}

type ViewState =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "chooseExisting"; notes: NoteOrigin[] }
  | { kind: "form" };

const DEFAULT_FOLDER = "AI-Drafts";

function stemOf(name: string): string {
  const idx = name.lastIndexOf(".");
  return idx > 0 ? name.slice(0, idx) : name;
}

export function KnowledgeSaveDialog({
  open,
  fileId,
  drive,
  content,
  sourceFilename,
  onClose,
  onSaved,
}: KnowledgeSaveDialogProps) {
  const t = useTranslations("knowledgeSave");
  const tc = useTranslations("common");

  const defaultStem = useMemo(() => stemOf(sourceFilename), [sourceFilename]);

  const [state, setState] = useState<ViewState>({ kind: "loading" });

  const [folder, setFolder] = useState(DEFAULT_FOLDER);
  const [filename, setFilename] = useState(`${defaultStem}-summary.md`);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const reset = useCallback(() => {
    setFolder(DEFAULT_FOLDER);
    setFilename(`${defaultStem}-summary.md`);
    setSubmitting(false);
    setSubmitError(null);
  }, [defaultStem]);

  const loadDialogState = useCallback(async () => {
    setState({ kind: "loading" });
    reset();
    try {
      const notes = await getNotesBySourceFile(drive, fileId);
      if (notes.length > 0) {
        setState({ kind: "chooseExisting", notes });
        return;
      }
      setState({ kind: "form" });
    } catch (err) {
      setState({ kind: "error", message: (err as Error).message });
    }
  }, [drive, fileId, reset]);

  useEffect(() => {
    if (!open) return;
    void loadDialogState();
  }, [open, loadDialogState]);

  // Escape goes through the shortcut stack instead of a listener of its
  // own, so a layer pushed on top wins the key rather than both closing
  // on one press. `editingOnly: false` is load-bearing: the provider
  // counts a focused input as "editing", and the default means "only
  // when not editing" — which in a dialog that focuses its filename
  // field is never.
  useShortcuts(
    "intelligence-knowledge-save-dialog",
    "Dialog",
    [
      {
        key: "escape",
        label: "Close",
        editingOnly: false,
        hidden: true,
        handler: onClose,
      },
    ],
    open,
    OVERLAY_PRIORITY,
  );

  const handleSubmit = useCallback(async () => {
    setSubmitting(true);
    setSubmitError(null);
    try {
      const cleanFilename = filename.trim();
      const res = await distillToKnowledge(drive, {
        source_file_id: fileId,
        folder: folder.trim() || DEFAULT_FOLDER,
        filename: cleanFilename,
        title: stemOf(cleanFilename),
        content,
        origin: "detailed_summary",
      });
      onSaved({ noteFileId: res.note_file_id, notePath: res.note_path });
      onClose();
    } catch (err) {
      setSubmitError((err as Error).message);
    } finally {
      setSubmitting(false);
    }
  }, [drive, fileId, folder, filename, content, onSaved, onClose]);

  const openExisting = useCallback((note: NoteOrigin) => {
    const url = `/drive/${encodeURIComponent(drive)}/addons/knowledge?edit=${encodeURIComponent(note.note_file_id)}`;
    window.location.href = url;
  }, [drive]);

  const proceedToFormFromExisting = useCallback(() => {
    setState({ kind: "form" });
  }, []);

  const inputClass =
    "w-full rounded-lg border border-bg-border bg-bg-primary px-3 py-2 text-sm text-text-primary placeholder:text-text-muted focus:border-focus-ring focus:outline-none focus:ring-1 focus:ring-focus-ring";

  const primaryLatestNote = useMemo(() => {
    if (state.kind !== "chooseExisting") return null;
    return [...state.notes].sort((a, b) => {
      const at = a.approved_at ? Date.parse(a.approved_at) : 0;
      const bt = b.approved_at ? Date.parse(b.approved_at) : 0;
      return bt - at;
    })[0];
  }, [state]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm animate-fade-in"
        onClick={onClose}
      />
      <div className="relative mx-4 w-full max-w-md rounded-2xl bg-bg-card p-6 shadow-lg animate-fade-in-scale">
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-lg font-semibold text-text-primary">
            {t("title")}
          </h2>
          <button
            onClick={onClose}
            className="rounded-xl p-1 text-text-muted hover:text-text-primary"
            aria-label={tc("close")}
          >
            <X size={18} />
          </button>
        </div>

        {state.kind === "loading" && (
          <p className="text-sm text-text-muted">{t("loading")}</p>
        )}

        {state.kind === "error" && (
          <div className="rounded-lg border border-danger/30 bg-danger/10 px-3 py-2 text-sm text-danger">
            {state.message}
          </div>
        )}

        {state.kind === "chooseExisting" && primaryLatestNote && (
          <div className="flex flex-col gap-3">
            <p className="text-sm text-text-muted">
              {t("existing.description")}
            </p>
            <div className="rounded-lg border border-bg-border bg-bg-elevated px-3 py-2 font-mono text-[13px] text-text-primary break-anywhere">
              {primaryLatestNote.path}
            </div>
            {state.notes.length > 1 && (
              <details className="text-xs text-text-muted">
                <summary className="cursor-pointer">
                  {t("existing.moreNotes", { count: state.notes.length - 1 })}
                </summary>
                <ul className="mt-2 flex flex-col gap-1">
                  {state.notes
                    .filter((n) => n.note_file_id !== primaryLatestNote.note_file_id)
                    .map((n) => (
                      <li key={n.note_file_id}>
                        <button
                          onClick={() => openExisting(n)}
                          className="text-left font-mono text-text-muted hover:text-text-primary break-anywhere"
                        >
                          {n.path}
                        </button>
                      </li>
                    ))}
                </ul>
              </details>
            )}
            <div className="mt-2 flex flex-wrap justify-end gap-2">
              <button
                onClick={onClose}
                className="rounded-2xl bg-sand px-4 py-2 text-sm font-medium text-text-primary hover:bg-sand-hover"
              >
                {tc("cancel")}
              </button>
              <button
                onClick={proceedToFormFromExisting}
                className="rounded-2xl border border-bg-border bg-bg-elevated px-4 py-2 text-sm font-medium text-text-primary hover:border-accent/40"
              >
                {t("existing.createNew")}
              </button>
              <button
                onClick={() => openExisting(primaryLatestNote)}
                className="rounded-2xl bg-accent px-4 py-2 text-sm font-medium text-white hover:bg-accent-hover"
              >
                {t("existing.openExisting")}
              </button>
            </div>
          </div>
        )}

        {state.kind === "form" && (
          <div className="flex flex-col gap-3">
            <div className="flex flex-col gap-1">
              <span className="text-xs text-text-muted">{t("form.folder")}</span>
              <FolderPicker drive={drive} value={folder} onChange={setFolder} />
            </div>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-text-muted">{t("form.filename")}</span>
              <input
                type="text"
                value={filename}
                onChange={(e) => setFilename(e.target.value)}
                className={`${inputClass} font-mono text-[13px]`}
              />
            </label>
            {submitError && (
              <div className="rounded-lg border border-danger/30 bg-danger/10 px-3 py-2 text-sm text-danger">
                {submitError}
              </div>
            )}
            <div className="mt-2 flex justify-end gap-2">
              <button
                onClick={onClose}
                disabled={submitting}
                className="rounded-2xl bg-sand px-4 py-2 text-sm font-medium text-text-primary hover:bg-sand-hover disabled:opacity-50"
              >
                {tc("cancel")}
              </button>
              <button
                onClick={handleSubmit}
                disabled={submitting || !filename.trim()}
                className="rounded-2xl bg-accent px-4 py-2 text-sm font-medium text-white hover:bg-accent-hover disabled:bg-sand disabled:text-warm-silver disabled:cursor-not-allowed"
              >
                {submitting ? t("form.submitting") : t("form.submit")}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
