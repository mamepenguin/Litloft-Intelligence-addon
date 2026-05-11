"use client";

/**
 * Dialog that promotes an edited detailed_summary into a Vault ``.md``
 * via the knowledge addon's ``/distill`` endpoint.
 *
 * Opens on "knowledge に保存" button click in DetailedSummarySection.
 * Handles four cases before showing the save form:
 *
 *   A. No Vault exists on this drive — show an inline create form
 *      (mirrors VaultSetup.tsx's knobs but cheaper than bouncing the
 *      user to the knowledge page and losing the summary context).
 *   B. Exactly one prior promotion from this source file — default to
 *      "open existing", offering "create new" as a secondary action
 *      (保守的 default — avoid accidentally growing .md count).
 *   C. Multiple prior promotions — same as B, with a disclosure for
 *      the older notes.
 *   D. No prior promotion — show the full save form with Vault
 *      select, folder, filename, title.
 *
 * On success, closes itself and calls `onSaved(noteFileId)` so the
 * parent component can refresh `useActiveSummary` (or rely on the WS
 * event `core.file_active_summary.changed`).
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import { useTranslations } from "next-intl";
import { X } from "lucide-react";

import {
  createKnowledgeVault,
  distillToKnowledge,
  getNotesBySourceFile,
  listKnowledgeVaults,
  type KnowledgeVault,
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
  | { kind: "createVault"; reason: "none" | "user" }
  | { kind: "chooseExisting"; notes: NoteOrigin[]; vaults: KnowledgeVault[]; activeId: number | null }
  | { kind: "form"; vaults: KnowledgeVault[]; activeId: number | null };

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
  const defaultTitle = useMemo(
    () => t("form.defaultTitle", { filename: sourceFilename }),
    [t, sourceFilename],
  );

  const [state, setState] = useState<ViewState>({ kind: "loading" });

  const [selectedVaultId, setSelectedVaultId] = useState<number | null>(null);
  const [folder, setFolder] = useState(DEFAULT_FOLDER);
  const [filename, setFilename] = useState(`${defaultStem}-summary.md`);
  const [title, setTitle] = useState(defaultTitle);
  const [newVaultLabel, setNewVaultLabel] = useState("MyVault");
  const [newVaultPath, setNewVaultPath] = useState("Knowledge");
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const reset = useCallback(() => {
    setFolder(DEFAULT_FOLDER);
    setFilename(`${defaultStem}-summary.md`);
    setTitle(defaultTitle);
    setNewVaultLabel("MyVault");
    setNewVaultPath("Knowledge");
    setSubmitting(false);
    setSubmitError(null);
  }, [defaultStem, defaultTitle]);

  const loadDialogState = useCallback(async () => {
    setState({ kind: "loading" });
    reset();
    try {
      const [vaultRes, notes] = await Promise.all([
        listKnowledgeVaults(drive),
        getNotesBySourceFile(drive, fileId),
      ]);
      if (vaultRes.vaults.length === 0) {
        setState({ kind: "createVault", reason: "none" });
        return;
      }
      const activeId = vaultRes.active_vault_id ?? vaultRes.vaults[0].id;
      setSelectedVaultId(activeId);
      if (notes.length > 0) {
        setState({
          kind: "chooseExisting",
          notes,
          vaults: vaultRes.vaults,
          activeId,
        });
        return;
      }
      setState({ kind: "form", vaults: vaultRes.vaults, activeId });
    } catch (err) {
      setState({ kind: "error", message: (err as Error).message });
    }
  }, [drive, fileId, reset]);

  useEffect(() => {
    if (!open) return;
    void loadDialogState();
  }, [open, loadDialogState]);

  useEffect(() => {
    if (!open) return;
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [open, onClose]);

  const handleCreateVault = useCallback(async () => {
    setSubmitting(true);
    setSubmitError(null);
    try {
      const vault = await createKnowledgeVault(drive, {
        label: newVaultLabel.trim() || "MyVault",
        path: newVaultPath.trim(),
      });
      setSelectedVaultId(vault.id);
      setState({ kind: "form", vaults: [vault], activeId: vault.id });
    } catch (err) {
      setSubmitError((err as Error).message);
    } finally {
      setSubmitting(false);
    }
  }, [drive, newVaultLabel, newVaultPath]);

  const handleSubmit = useCallback(async () => {
    if (!selectedVaultId) return;
    setSubmitting(true);
    setSubmitError(null);
    try {
      const res = await distillToKnowledge(drive, {
        source_file_id: fileId,
        vault_id: selectedVaultId,
        folder: folder.trim() || DEFAULT_FOLDER,
        filename: filename.trim(),
        title: title.trim() || defaultTitle,
        content,
        origin: "detailed_summary",
        origin_ref: `intelligence:${fileId}/detailed_summary`,
      });
      onSaved({ noteFileId: res.note_file_id, notePath: res.note_path });
      onClose();
    } catch (err) {
      setSubmitError((err as Error).message);
    } finally {
      setSubmitting(false);
    }
  }, [
    selectedVaultId,
    drive,
    fileId,
    folder,
    filename,
    title,
    defaultTitle,
    content,
    onSaved,
    onClose,
  ]);

  const openExisting = useCallback((note: NoteOrigin) => {
    const url = `/drive/${encodeURIComponent(drive)}/addons/knowledge?edit=${encodeURIComponent(note.note_file_id)}`;
    window.location.href = url;
  }, [drive]);

  const proceedToFormFromExisting = useCallback(() => {
    setState((prev) => {
      if (prev.kind !== "chooseExisting") return prev;
      return { kind: "form", vaults: prev.vaults, activeId: prev.activeId };
    });
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

        {state.kind === "createVault" && (
          <div className="flex flex-col gap-3">
            <p className="text-sm text-text-muted">{t("createVault.description")}</p>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-text-muted">
                {t("createVault.labelField")}
              </span>
              <input
                type="text"
                value={newVaultLabel}
                onChange={(e) => setNewVaultLabel(e.target.value)}
                className={inputClass}
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-text-muted">
                {t("createVault.pathField")}
              </span>
              <input
                type="text"
                value={newVaultPath}
                onChange={(e) => setNewVaultPath(e.target.value)}
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
                onClick={handleCreateVault}
                disabled={submitting || !newVaultLabel.trim()}
                className="rounded-2xl bg-accent px-4 py-2 text-sm font-medium text-white hover:bg-accent-hover disabled:opacity-50"
              >
                {submitting ? t("createVault.submitting") : t("createVault.submit")}
              </button>
            </div>
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
            {state.vaults.length > 1 ? (
              <label className="flex flex-col gap-1">
                <span className="text-xs text-text-muted">{t("form.vault")}</span>
                <select
                  value={selectedVaultId ?? ""}
                  onChange={(e) => setSelectedVaultId(Number(e.target.value))}
                  className={inputClass}
                >
                  {state.vaults.map((v) => (
                    <option key={v.id} value={v.id}>
                      {v.label}
                    </option>
                  ))}
                </select>
              </label>
            ) : (
              <div className="text-xs text-text-muted">
                {t("form.vault")}:{" "}
                <span className="font-medium text-text-primary">
                  {state.vaults[0]?.label}
                </span>
              </div>
            )}
            <label className="flex flex-col gap-1">
              <span className="text-xs text-text-muted">{t("form.folder")}</span>
              <input
                type="text"
                value={folder}
                onChange={(e) => setFolder(e.target.value)}
                className={`${inputClass} font-mono text-[13px]`}
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-text-muted">{t("form.filename")}</span>
              <input
                type="text"
                value={filename}
                onChange={(e) => setFilename(e.target.value)}
                className={`${inputClass} font-mono text-[13px]`}
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-text-muted">{t("form.noteTitle")}</span>
              <input
                type="text"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                className={inputClass}
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
                disabled={
                  submitting
                  || !selectedVaultId
                  || !filename.trim()
                  || !title.trim()
                }
                className="rounded-2xl bg-accent px-4 py-2 text-sm font-medium text-white hover:bg-accent-hover disabled:opacity-50"
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
