"use client";

/**
 * `folder-actions-menu` rows for the folder toolbar's `Add` menu.
 *
 * These were a labelled `AI ▾` button of their own, sitting on the toolbar
 * beside `Add`. 案 2 gives that bar four exposed controls plus a conditional
 * `Play`, and an addon's own dropdown is a fifth — the shape
 * `2026-08-30-file-actions-menu-addon-slot.md` §6 settles by having addons
 * draw `ActionMenuItem` rows the host cannot tell from its own.
 *
 * The host closes its menu for us: none of these opens a dialog that would
 * unmount with it, unlike `IndexDetailsMenuItem`, which asks for the close
 * afterwards instead.
 */

import { useCallback, useState } from "react";
import { useTranslations } from "next-intl";
import { BookOpen, ImageIcon, Sparkles } from "lucide-react";

import { ActionMenuItem } from "@/components/ActionMenuItem";
import { useToast } from "@/components/ToastProvider";
import {
  batchSuggestedTags,
  batchSummaries,
  generateFolderVisualDescription,
} from "./api";
import type { FolderVisualDescriptionTooManyError } from "./api";

interface FolderAIActionsMenuItemsProps {
  fileIds: string[];
  drive: string;
  onRequestClose?: () => void;
}

type Pending = "tags" | "summaries" | "vision" | null;

export default function FolderAIActionsMenuItems({
  fileIds,
  drive,
  onRequestClose,
}: FolderAIActionsMenuItemsProps) {
  const t = useTranslations("file");
  const toast = useToast();
  const [pending, setPending] = useState<Pending>(null);

  const handleTags = useCallback(async () => {
    if (fileIds.length === 0 || pending) return;
    onRequestClose?.();
    setPending("tags");
    try {
      const result = await batchSuggestedTags(fileIds, drive);
      if (result.queued === 0 && result.skipped > 0) {
        toast.info(t("tagsBatchEmpty"));
      } else {
        toast.success(
          t("tagsBatchQueued", { queued: result.queued, skipped: result.skipped }),
        );
      }
    } catch {
      // non-critical
    } finally {
      setPending(null);
    }
  }, [fileIds, drive, pending, t, toast, onRequestClose]);

  const handleSummaries = useCallback(async () => {
    if (fileIds.length === 0 || pending) return;
    onRequestClose?.();
    setPending("summaries");
    try {
      const result = await batchSummaries(fileIds, drive);
      if (result.queued === 0 && result.skipped > 0) {
        toast.info(t("summariesBatchEmpty"));
      } else {
        toast.success(
          t("summariesBatchQueued", {
            queued: result.queued,
            skipped: result.skipped,
          }),
        );
      }
    } catch {
      // non-critical
    } finally {
      setPending(null);
    }
  }, [fileIds, drive, pending, t, toast, onRequestClose]);

  const handleVision = useCallback(async () => {
    if (fileIds.length === 0 || pending) return;
    onRequestClose?.();
    const confirmed = window.confirm(
      t("visionFolderConfirm", {
        defaultMessage:
          "Generate AI descriptions for every image in this folder? This may incur LLM costs.",
      }),
    );
    if (!confirmed) return;
    setPending("vision");
    try {
      const result = await generateFolderVisualDescription(drive, fileIds);
      toast.success(
        t("visionFolderQueued", {
          queued: result.queued,
          defaultMessage: "{queued} images queued",
        }),
      );
    } catch (e) {
      const info = (e as { info?: FolderVisualDescriptionTooManyError }).info;
      // `toast.error`, where this used to be a plain message held for 8s
      // instead of 5. The kind carries what the extra seconds were doing,
      // and it carries it to a reader who has already looked away.
      if (info?.kind === "too_many_files") {
        toast.error(
          t("visionFolderTooMany", {
            max: info.max,
            requested: info.requested,
            defaultMessage:
              "Too many files ({requested}). Max per batch: {max}.",
          }),
        );
      } else {
        toast.error(
          t("visionFolderError", {
            defaultMessage: "Failed to queue folder — please retry.",
          }),
        );
      }
    } finally {
      setPending(null);
    }
  }, [drive, fileIds, pending, t, toast, onRequestClose]);

  // Nothing to act on, so no rows — and `AddButton` hides the separator
  // above them with `empty:hidden` when that happens.
  if (fileIds.length === 0) return null;

  const busy = pending !== null;

  return (
    <>
      <ActionMenuItem
        icon={Sparkles}
        label={t("generateFolderTags")}
        onClick={handleTags}
        disabled={busy}
      />
      <ActionMenuItem
        icon={BookOpen}
        label={t("generateFolderSummaries")}
        onClick={handleSummaries}
        disabled={busy}
      />
      <ActionMenuItem
        icon={ImageIcon}
        label={t("visionFolderButton", {
          defaultMessage: "Generate AI descriptions for folder images",
        })}
        onClick={handleVision}
        disabled={busy}
      />
    </>
  );
}
