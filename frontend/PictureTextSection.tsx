"use client";

/**
 * The text the OCR worker read from one picture (SPEC-ADDON-067).
 *
 * It shows the stored text line by line, lets the owner correct it, and
 * offers the kept original once one exists. It is a `file-detail-sections`
 * entry, so the page draws it only when the manifest lists it.
 */

import { useCallback, useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { useTranslations } from "next-intl";
import { Pencil } from "lucide-react";

import { getPictureText, restorePictureText, updatePictureText } from "./api";
import type { PictureTextResponse } from "./api";
import { LLMRequestError } from "./api";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import { useDialogPortalTarget } from "@/components/DialogPortal";
import { Button } from "@/components/Button";
import { useWebSocket } from "@/hooks/useWebSocket";
import { useImeKeyGuard } from "@/lib/ime";

interface PictureTextSectionProps {
  fileId: string;
  drive: string;
}

export default function PictureTextSection({
  fileId,
  drive,
}: PictureTextSectionProps) {
  const t = useTranslations("searchIndex");
  const ime = useImeKeyGuard();
  const dialogHost = useDialogPortalTarget();
  const [data, setData] = useState<PictureTextResponse | null>(null);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  const [expectedText, setExpectedText] = useState("");
  const [saving, setSaving] = useState(false);
  const [confirmRestore, setConfirmRestore] = useState(false);
  const [restoring, setRestoring] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [refreshTick, setRefreshTick] = useState(0);

  // The draft and the confirmation belong to one picture. The host reuses
  // this mount for the next file, so they must not carry over.
  useEffect(() => {
    setEditing(false);
    setDraft("");
    setExpectedText("");
    setConfirmRestore(false);
    setError(null);
    setData(null);
  }, [fileId]);

  useEffect(() => {
    let cancelled = false;
    getPictureText(fileId, drive)
      .then((res) => {
        if (!cancelled) setData(res);
      })
      .catch(() => {
        if (!cancelled) setData(null);
      });
    return () => {
      cancelled = true;
    };
  }, [fileId, drive, refreshTick]);

  const updated = useWebSocket("intelligence.picture_text.updated");
  useEffect(() => {
    if (!updated) return;
    const body = updated.data as { file_id?: string } | null;
    if (!body || !body.file_id || body.file_id === fileId) {
      setRefreshTick((tick) => tick + 1);
    }
  }, [updated, fileId]);

  const beginEdit = useCallback(() => {
    setDraft(data?.text ?? "");
    setExpectedText(data?.text ?? "");
    setEditing(true);
    setError(null);
  }, [data]);

  const cancelEdit = useCallback(() => {
    setEditing(false);
    setDraft("");
    setError(null);
  }, []);

  const save = useCallback(async () => {
    if (!data || saving) return;
    setSaving(true);
    setError(null);
    try {
      const res = await updatePictureText(fileId, draft, expectedText, drive);
      setData(res);
      setEditing(false);
    } catch (err) {
      if (err instanceof LLMRequestError) {
        const detail = err.detail as { code?: string } | string | undefined;
        const code = typeof detail === "object" ? detail?.code : undefined;
        if (code === "busy") setError(t("transcriptBusy"));
        else if (code === "changed") {
          setError(t("transcriptChanged"));
          setEditing(false);
          setRefreshTick((tick) => tick + 1);
        } else if (code === "too_long") setError(t("transcriptTooLong"));
        else if (err.status === 404) setError(t("transcriptUnavailable"));
        else setError(t("transcriptSaveFailed"));
      } else {
        setError(t("transcriptSaveFailed"));
      }
    } finally {
      setSaving(false);
    }
  }, [data, saving, fileId, draft, expectedText, drive, t]);

  const doRestore = useCallback(async () => {
    if (restoring) return;
    setRestoring(true);
    setError(null);
    setConfirmRestore(false);
    try {
      const res = await restorePictureText(fileId, drive);
      setData(res);
      setEditing(false);
    } catch (err) {
      setError(
        err instanceof LLMRequestError && err.status === 404
          ? t("transcriptUnavailable")
          : t("transcriptRestoreFailed"),
      );
    } finally {
      setRestoring(false);
    }
  }, [restoring, fileId, drive, t]);

  if (!data) return null;
  if (!data.text && !data.has_original) return null;

  const lines = data.text ? data.text.split("\n") : [];

  return (
    <section className="rounded-xl border border-bg-border bg-bg-card p-4">
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <h2 className="text-sm font-semibold text-text-primary">
          {t("pictureTextTitle")}
        </h2>
        {data.corrected && (
          <span className="rounded-lg bg-warm-light px-1.5 py-0.5 text-[10px] font-medium text-text-muted">
            {t("transcriptCorrectedBadge")}
          </span>
        )}
      </div>

      {editing ? (
        <div>
          <textarea
            aria-label={t("transcriptEdit")}
            value={draft}
            autoFocus
            onCompositionEnd={ime.onCompositionEnd}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              event.stopPropagation();
              if (ime.isImeKeystroke(event)) return;
              if (event.key === "Escape") {
                event.preventDefault();
                cancelEdit();
              } else if (
                event.key === "Enter" &&
                (event.ctrlKey || event.metaKey)
              ) {
                event.preventDefault();
                void save();
              }
            }}
            rows={Math.max(3, lines.length + 1)}
            className="w-full rounded-2xl border border-bg-border bg-bg-primary p-2 text-base text-text-primary outline-none focus-visible:ring-2 focus-visible:ring-focus-ring"
          />
          <p className="mt-1 text-[10px] text-text-muted">Esc / ⌘↵</p>
          {error && (
            <p className="mt-1 text-xs text-danger" role="status">
              {error}
            </p>
          )}
          <div className="mt-2 flex flex-wrap justify-end gap-2 text-xs">
            <Button
              variant="primary"
              size="sm"
              onClick={() => void save()}
              disabled={saving}
              className="pointer-coarse:min-h-11 pointer-coarse:flex-1"
            >
              {t("transcriptSave")}
            </Button>
            <Button
              variant="secondary"
              size="sm"
              onClick={cancelEdit}
              className="pointer-coarse:min-h-11 pointer-coarse:flex-1"
            >
              {t("transcriptCancel")}
            </Button>
          </div>
        </div>
      ) : (
        <>
          <div className="space-y-0.5 text-base leading-relaxed text-text-primary">
            {lines.map((line, index) => (
              <p key={`${index}-${line}`}>{line}</p>
            ))}
          </div>
          <div className="mt-2 flex flex-wrap items-center gap-2 text-xs">
            <button
              type="button"
              onClick={beginEdit}
              disabled={data.busy}
              className="flex items-center gap-1 rounded-lg px-2 py-0.5 text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary disabled:opacity-50 pointer-coarse:min-h-11"
            >
              <Pencil size={11} />
              {t("transcriptEdit")}
            </button>
            {data.has_original && (
              <button
                type="button"
                onClick={() => setConfirmRestore(true)}
                disabled={data.busy || restoring}
                className="rounded-lg px-2 py-0.5 text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary disabled:opacity-50 pointer-coarse:min-h-11"
              >
                {t("transcriptRestoreOriginal")}
              </button>
            )}
            {error && (
              <span className="text-danger" role="status">
                {error}
              </span>
            )}
          </div>
        </>
      )}

      {dialogHost &&
        createPortal(
          <ConfirmDialog
            open={confirmRestore}
            title={t("transcriptRestoreOriginal")}
            message={t("transcriptRestoreConfirm")}
            confirmLabel={t("transcriptRestoreConfirmAction")}
            onConfirm={() => void doRestore()}
            onCancel={() => setConfirmRestore(false)}
          />,
          dialogHost,
        )}
    </section>
  );
}
