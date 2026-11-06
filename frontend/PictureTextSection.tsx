"use client";

/**
 * The text the OCR worker read from one picture (SPEC-ADDON-067).
 *
 * It shows the stored text line by line, lets the owner correct it, and
 * offers the kept original once one exists. It is a `file-detail-sections`
 * entry, so the page draws it only when the manifest lists it.
 */

import { useCallback, useEffect, useState } from "react";
import { useTranslations } from "next-intl";
import { Pencil } from "lucide-react";

import { getPictureText, restorePictureText, updatePictureText } from "./api";
import type { PictureTextResponse } from "./api";
import { LLMRequestError } from "./api";
import { Button } from "@/components/Button";

interface PictureTextSectionProps {
  fileId: string;
  drive: string;
}

export default function PictureTextSection({
  fileId,
  drive,
}: PictureTextSectionProps) {
  const t = useTranslations("searchIndex");
  const [data, setData] = useState<PictureTextResponse | null>(null);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  const [saving, setSaving] = useState(false);
  const [confirmRestore, setConfirmRestore] = useState(false);
  const [restoring, setRestoring] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
  }, [fileId, drive]);

  const beginEdit = useCallback(() => {
    setDraft(data?.text ?? "");
    setEditing(true);
    setError(null);
  }, [data]);

  const cancelEdit = useCallback(() => {
    setEditing(false);
    setError(null);
  }, []);

  const save = useCallback(async () => {
    if (!data || saving) return;
    setSaving(true);
    setError(null);
    try {
      const res = await updatePictureText(fileId, draft, data.text, drive);
      setData(res);
      setEditing(false);
    } catch (err) {
      if (err instanceof LLMRequestError) {
        const detail = err.detail as { code?: string } | string | undefined;
        const code = typeof detail === "object" ? detail?.code : undefined;
        if (code === "busy") setError(t("transcriptBusy"));
        else if (code === "changed") setError(t("transcriptChanged"));
        else if (code === "too_long") setError(t("transcriptTooLong"));
        else if (err.status === 404) setError(t("transcriptUnavailable"));
        else setError(t("transcriptSaveFailed"));
      } else {
        setError(t("transcriptSaveFailed"));
      }
    } finally {
      setSaving(false);
    }
  }, [data, saving, fileId, draft, drive, t]);

  const doRestore = useCallback(async () => {
    if (restoring) return;
    setRestoring(true);
    setError(null);
    try {
      const res = await restorePictureText(fileId, drive);
      setData(res);
      setConfirmRestore(false);
    } catch (err) {
      setConfirmRestore(false);
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
          <span className="rounded-lg bg-accent-amber/15 px-1.5 py-0.5 text-[10px] font-medium text-accent-amber">
            {t("transcriptCorrectedBadge")}
          </span>
        )}
        <div className="ml-auto flex flex-wrap justify-end gap-1 text-xs">
          {data.has_original && (
            <button
              type="button"
              onClick={() => setConfirmRestore(true)}
              disabled={data.busy || restoring}
              className="rounded-lg px-2 py-0.5 text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary disabled:opacity-50"
            >
              {t("transcriptRestoreOriginal")}
            </button>
          )}
          {!editing && (
            <button
              type="button"
              onClick={beginEdit}
              disabled={data.busy}
              className="flex items-center gap-1 rounded-lg px-2 py-0.5 text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary disabled:opacity-50"
            >
              <Pencil size={11} />
              {t("transcriptEdit")}
            </button>
          )}
        </div>
      </div>

      {error && (
        <p className="mb-2 text-xs text-danger" role="status">
          {error}
        </p>
      )}

      {confirmRestore && (
        <div className="mb-2 rounded-lg border border-bg-border bg-bg-elevated p-3 text-xs">
          <p className="mb-2 text-text-primary">{t("transcriptRestoreConfirm")}</p>
          <div className="flex justify-end gap-2">
            <Button
              variant="primary"
              size="sm"
              onClick={() => void doRestore()}
              disabled={restoring}
            >
              {t("transcriptRestoreConfirmAction")}
            </Button>
            <Button
              variant="secondary"
              size="sm"
              onClick={() => setConfirmRestore(false)}
            >
              {t("transcriptCancel")}
            </Button>
          </div>
        </div>
      )}

      {editing ? (
        <div>
          <textarea
            aria-label={t("transcriptEdit")}
            value={draft}
            autoFocus
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              event.stopPropagation();
              if (event.key === "Escape") {
                event.preventDefault();
                cancelEdit();
              } else if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
                event.preventDefault();
                void save();
              }
            }}
            rows={Math.max(3, lines.length + 1)}
            className="w-full rounded-lg border border-bg-border bg-bg-primary p-2 text-sm text-text-primary outline-none focus:border-focus-ring"
          />
          <div className="mt-2 flex justify-end gap-2 text-xs">
            <Button
              variant="primary"
              size="sm"
              onClick={() => void save()}
              disabled={saving}
            >
              {t("transcriptSave")}
            </Button>
            <Button
              variant="secondary"
              size="sm"
              onClick={cancelEdit}
            >
              {t("transcriptCancel")}
            </Button>
          </div>
        </div>
      ) : (
        <div className="space-y-0.5 text-sm text-text-primary">
          {lines.map((line, index) => (
            <p key={`${index}-${line}`}>{line}</p>
          ))}
        </div>
      )}
    </section>
  );
}
