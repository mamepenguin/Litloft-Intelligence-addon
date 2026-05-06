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

import {
  createKnowledgeVault,
  listKnowledgeVaults,
  saveAskToKnowledge,
  type KnowledgeVault,
} from "./knowledgeBridge";

interface Props {
  open: boolean;
  drive: string;
  defaultFilename: string;
  content: string;
  sourceFileIds: string[];
  onClose: () => void;
  onSaved: (result: { noteFileId: string; notePath: string }) => void;
}

type ViewState =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "createVault" }
  | { kind: "form"; vaults: KnowledgeVault[]; activeId: number | null };

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

  const [state, setState] = useState<ViewState>({ kind: "loading" });
  const [selectedVaultId, setSelectedVaultId] = useState<number | null>(null);
  const [folder, setFolder] = useState(DEFAULT_FOLDER);
  const [filename, setFilename] = useState(defaultFilename);
  const [newVaultLabel, setNewVaultLabel] = useState("MyVault");
  const [newVaultPath, setNewVaultPath] = useState("Knowledge");
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setState({ kind: "loading" });
    setFilename(defaultFilename);
    setFolder(DEFAULT_FOLDER);
    setSubmitting(false);
    setSubmitError(null);
    try {
      const res = await listKnowledgeVaults(drive);
      if (res.vaults.length === 0) {
        setState({ kind: "createVault" });
        return;
      }
      const activeId = res.active_vault_id ?? res.vaults[0].id;
      setSelectedVaultId(activeId);
      setState({ kind: "form", vaults: res.vaults, activeId });
    } catch (err) {
      setState({ kind: "error", message: (err as Error).message });
    }
  }, [drive, defaultFilename]);

  useEffect(() => {
    if (!open) return;
    void load();
  }, [open, load]);

  useEffect(() => {
    if (!open) return;
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
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
      const res = await saveAskToKnowledge(drive, {
        vault_id: selectedVaultId,
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
  }, [selectedVaultId, drive, folder, filename, defaultFilename, content, sourceFileIds, onSaved, onClose]);

  if (!open) return null;

  const inputClass =
    "w-full rounded-lg border border-bg-border bg-bg-primary px-3 py-2 text-sm text-text-primary focus:border-accent-blue focus:outline-none";
  const labelClass = "block text-xs font-medium text-text-muted mb-1";

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
      role="dialog"
      aria-modal
      aria-label={t("title")}
    >
      <div className="flex w-full max-w-md flex-col gap-4 rounded-xl bg-bg-card p-5 shadow-2xl">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-text-primary">{t("title")}</h2>
          <button
            type="button"
            onClick={onClose}
            className="flex h-7 w-7 items-center justify-center rounded-md text-text-muted hover:bg-bg-elevated hover:text-text-primary"
            aria-label={tc("close")}
          >
            <X size={14} />
          </button>
        </div>

        {state.kind === "loading" && (
          <p className="text-sm text-text-muted">{t("loading")}</p>
        )}

        {state.kind === "error" && (
          <p className="text-sm text-red-400">{state.message}</p>
        )}

        {state.kind === "createVault" && (
          <div className="flex flex-col gap-3">
            <p className="text-xs text-text-muted">{t("noVault")}</p>
            <div>
              <label className={labelClass}>{t("vaultLabel")}</label>
              <input
                className={inputClass}
                value={newVaultLabel}
                onChange={(e) => setNewVaultLabel(e.target.value)}
              />
            </div>
            <div>
              <label className={labelClass}>{t("vaultPath")}</label>
              <input
                className={inputClass}
                value={newVaultPath}
                onChange={(e) => setNewVaultPath(e.target.value)}
              />
            </div>
            {submitError && <p className="text-xs text-red-400">{submitError}</p>}
            <button
              type="button"
              disabled={submitting}
              onClick={handleCreateVault}
              className="self-end rounded-lg bg-accent-blue px-4 py-2 text-sm font-medium text-white hover:opacity-90 disabled:opacity-50"
            >
              {submitting ? t("saving") : t("createVault")}
            </button>
          </div>
        )}

        {state.kind === "form" && (
          <div className="flex flex-col gap-3">
            <div>
              <label className={labelClass}>{t("vault")}</label>
              <select
                className={inputClass}
                value={selectedVaultId ?? ""}
                onChange={(e) => setSelectedVaultId(Number(e.target.value))}
              >
                {state.vaults.map((v) => (
                  <option key={v.id} value={v.id}>{v.label}</option>
                ))}
              </select>
            </div>
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
            {submitError && <p className="text-xs text-red-400">{submitError}</p>}
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
                disabled={submitting || !selectedVaultId}
                onClick={handleSubmit}
                className="rounded-lg bg-accent-blue px-4 py-2 text-sm font-medium text-white hover:opacity-90 disabled:opacity-50"
              >
                {submitting ? t("saving") : t("save")}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
