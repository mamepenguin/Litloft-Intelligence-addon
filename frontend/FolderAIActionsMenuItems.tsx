"use client";

/**
 * `folder-actions-menu` rows for the folder toolbar's `Add` menu.
 *
 * 案 2 gives that bar four exposed controls plus a conditional `Play`; an
 * addon's own dropdown would be a fifth.
 * `2026-08-30-file-actions-menu-addon-slot.md` §6 is the contract that
 * avoids it — addons draw `ActionMenuItem` rows the host cannot tell from
 * its own.
 *
 * The host closes its menu as soon as a row is pressed, unlike
 * `IndexDetailsMenuItem`, which asks for the close only after its dialog is
 * dismissed. Nothing here opens one.
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

type Action = "tags" | "summaries" | "vision";

/**
 * Which batches are running, at module scope rather than in state.
 *
 * The host unmounts these rows the moment one is pressed, so a guard held
 * in this component is gone before the request it guards returns — and the
 * next time the menu opens, a fresh instance starts with nothing in flight.
 * These are batch LLM jobs; queueing one twice costs money.
 *
 * Keyed by drive as well as action, because two drives are two batches.
 * Read during render, so reopening the menu while one is running draws that
 * row disabled.
 */
const inFlight = new Set<string>();
const keyFor = (action: Action, drive: string) => `${action}:${drive}`;

export default function FolderAIActionsMenuItems({
  fileIds,
  drive,
  onRequestClose,
}: FolderAIActionsMenuItemsProps) {
  const t = useTranslations("file");
  const toast = useToast();
  // Re-render once a batch settles, so a menu that is still open stops
  // showing a row disabled when it no longer is.
  const [, setTick] = useState(0);

  const claim = useCallback(
    (action: Action) => {
      const key = keyFor(action, drive);
      if (fileIds.length === 0 || inFlight.has(key)) return null;
      inFlight.add(key);
      return () => {
        inFlight.delete(key);
        setTick((n) => n + 1);
      };
    },
    [drive, fileIds],
  );

  const handleTags = useCallback(async () => {
    const done = claim("tags");
    if (!done) return;
    onRequestClose?.();
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
      toast.error(
        t("tagsBatchError", { defaultMessage: "Could not queue — please retry." }),
      );
    } finally {
      done();
    }
  }, [claim, drive, fileIds, t, toast, onRequestClose]);

  const handleSummaries = useCallback(async () => {
    const done = claim("summaries");
    if (!done) return;
    onRequestClose?.();
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
      toast.error(
        t("summariesBatchError", { defaultMessage: "Could not queue — please retry." }),
      );
    } finally {
      done();
    }
  }, [claim, drive, fileIds, t, toast, onRequestClose]);

  const handleVision = useCallback(async () => {
    if (fileIds.length === 0 || inFlight.has(keyFor("vision", drive))) return;
    onRequestClose?.();
    const confirmed = window.confirm(
      t("visionFolderConfirm", {
        defaultMessage:
          "Generate AI descriptions for every image in this folder? This may incur LLM costs.",
      }),
    );
    if (!confirmed) return;
    // Claimed after the confirm, so declining holds nothing.
    const done = claim("vision");
    if (!done) return;
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
      done();
    }
  }, [claim, drive, fileIds, t, toast, onRequestClose]);

  // Nothing to act on, so no rows — and `AddButton` hides the separator
  // above them with `empty:hidden` when that happens.
  if (fileIds.length === 0) return null;

  const busy = (action: Action) => inFlight.has(keyFor(action, drive));

  return (
    <>
      <ActionMenuItem
        icon={Sparkles}
        label={t("generateFolderTags")}
        onClick={handleTags}
        disabled={busy("tags")}
      />
      <ActionMenuItem
        icon={BookOpen}
        label={t("generateFolderSummaries")}
        onClick={handleSummaries}
        disabled={busy("summaries")}
      />
      <ActionMenuItem
        icon={ImageIcon}
        label={t("visionFolderButton", {
          defaultMessage: "Generate AI descriptions for folder images",
        })}
        onClick={handleVision}
        disabled={busy("vision")}
      />
    </>
  );
}
