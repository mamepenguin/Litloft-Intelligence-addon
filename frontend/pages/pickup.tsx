"use client";

/**
 * Intelligence → Pickup feed.
 *
 * The drive-home carousel is the entrance; this is the thing itself.
 * A few hundred files the viewer has never opened, ordered by how well
 * they match the interests their watch history describes.
 *
 * Paged in stable rank order, deliberately. The carousel rotates its
 * twelve cards daily, but a paged list cannot: reshuffling between
 * pages duplicates rows on one and skips them on the next.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { useTranslations } from "next-intl";
import { PageFrame } from "@/components/PageFrame";
import { PageHeader } from "@/components/PageHeader";
import { CircleSlash, Sparkles } from "lucide-react";

import { useCurrentDrive } from "@/components/CurrentDriveProvider";
import { useProfile } from "@/components/ProfileProvider";
import { useToast } from "@/components/ToastProvider";
import { FileGrid } from "@/components/FileGrid";
import { batchGetFiles } from "@/lib/api";
import type { FileItem } from "@/types";
import {
  fetchPickup,
  markPickupNotInterested,
  unmarkPickupNotInterested,
} from "../api";

const PAGE_SIZE = 40;

/** A card this page removed, and everything needed to put it back. */
interface RemovedCard {
  item: FileItem;
  index: number;
  drive: string;
  viewer: string;
  /** The page's list when the card left; a different one holds no place for it. */
  generation: number;
}

export default function PickupPage() {
  const t = useTranslations("intelligence");
  const tc = useTranslations("common");
  const drive = useCurrentDrive();
  const profile = useProfile();
  const toast = useToast();
  const [files, setFiles] = useState<FileItem[]>([]);
  const [total, setTotal] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [failed, setFailed] = useState(false);
  const [pending, setPending] = useState<ReadonlySet<string>>(() => new Set());
  const sentinel = useRef<HTMLDivElement | null>(null);
  // Read inside the observer callback, which would otherwise close over
  // a stale count and ask for the same page forever.
  const loadedRef = useRef(0);
  // Bumped whenever the drive changes. A request in flight when that
  // happens resolves against the old drive, and without this its files
  // would be appended to the new drive's feed. Undo reads it too: a card
  // belongs to the list that held it.
  const generation = useRef(0);
  // A page that answered with nothing while it claimed more remain must
  // not make the "list is empty" effect ask again forever.
  const stalled = useRef(false);
  const viewer = useRef<string | null>(profile.nickname);
  useEffect(() => {
    viewer.current = profile.nickname;
  }, [profile.nickname]);

  const loadMore = useCallback(async () => {
    // Not gated on ``failed``: retry calls this directly, and the
    // callback it holds was built while the flag was still set. The
    // sentinel is unmounted on failure, so nothing else can re-enter.
    if (!drive || loading) return;
    const offset = loadedRef.current;
    if (total !== null && offset >= total) return;

    const mine = generation.current;
    setLoading(true);
    try {
      const page = await fetchPickup(drive, { limit: PAGE_SIZE, offset });
      if (mine !== generation.current) return;
      setTotal(page.total);
      if (page.file_ids.length === 0) {
        stalled.current = true;
        return;
      }
      const items = await batchGetFiles(page.file_ids);
      if (mine !== generation.current) return;
      loadedRef.current = offset + page.file_ids.length;
      stalled.current = false;
      setFiles((prev) => {
        const seen = new Set(prev.map((file) => file.id));
        return [...prev, ...items.filter((item) => !seen.has(item.id))];
      });
    } catch {
      // Hydration can fail after the page itself succeeded. Stopping is
      // the point: the sentinel is re-observed as soon as loading ends,
      // so retrying from the same offset would spin on the same failure
      // for as long as the page is open.
      if (mine === generation.current) setFailed(true);
    } finally {
      if (mine === generation.current) setLoading(false);
    }
  }, [drive, loading, total]);

  // Clearing the flag is not enough on its own: the sentinel only
  // re-triggers a load when it crosses back into view, and after a
  // failure it is usually already sitting there.
  const retry = useCallback(() => {
    setFailed(false);
    void loadMore();
  }, [loadMore]);

  useEffect(() => {
    generation.current += 1;
    setFiles([]);
    setTotal(null);
    setFailed(false);
    // Also cleared here, and not only in the finally above. A request
    // belonging to the previous drive must not write state on the way
    // out, so it leaves this flag set — and the new drive's first load
    // would then find the page busy and never start.
    setLoading(false);
    setPending(new Set());
    loadedRef.current = 0;
    stalled.current = false;
  }, [drive]);

  useEffect(() => {
    if (files.length === 0 && total === null && !failed) void loadMore();
  }, [files.length, total, failed, loadMore]);

  // A mark can empty the page while the feed still holds more. Asking
  // for the next page keeps the reader looking at the head of the feed
  // rather than at an empty state that is not true.
  useEffect(() => {
    if (files.length !== 0 || total === null || loading || failed) return;
    if (stalled.current || loadedRef.current >= total) return;
    void loadMore();
  }, [files.length, total, loading, failed, loadMore]);

  useEffect(() => {
    const node = sentinel.current;
    if (!node) return;
    const observer = new IntersectionObserver((entries) => {
      if (entries.some((e) => e.isIntersecting)) void loadMore();
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, [loadMore]);

  const undo = async (removed: RemovedCard) => {
    if (viewer.current !== removed.viewer) return;
    try {
      await unmarkPickupNotInterested(removed.item.id, removed.drive);
    } catch {
      toast.error(tc("saveFailed"));
      return;
    }
    if (generation.current !== removed.generation) return;
    setFiles((prev) => {
      if (prev.some((file) => file.id === removed.item.id)) return prev;
      const at = Math.min(removed.index, prev.length);
      return [...prev.slice(0, at), removed.item, ...prev.slice(at)];
    });
    loadedRef.current += 1;
    setTotal((prev) => (prev === null ? prev : prev + 1));
  };

  const mark = async (file: FileItem) => {
    if (!drive || pending.has(file.id)) return;
    const index = files.findIndex((candidate) => candidate.id === file.id);
    const removed: RemovedCard = {
      item: file,
      index: index < 0 ? 0 : index,
      drive,
      viewer: profile.nickname ?? "",
      generation: generation.current,
    };
    setPending((prev) => new Set(prev).add(file.id));
    try {
      await markPickupNotInterested(file.id, drive);
    } catch {
      toast.error(tc("saveFailed"));
      return;
    } finally {
      setPending((prev) => {
        const next = new Set(prev);
        next.delete(file.id);
        return next;
      });
    }

    if (generation.current === removed.generation) {
      setFiles((prev) => prev.filter((candidate) => candidate.id !== file.id));
      // The offset counts feed ids, so a marked id that was inside the
      // loaded window moves the next request back by one.
      loadedRef.current = Math.max(0, loadedRef.current - 1);
      setTotal((prev) => (prev === null ? prev : Math.max(0, prev - 1)));
    }
    toast.success(t("pickup.notInterestedToast"), {
      action: { label: tc("undo"), onClick: () => void undo(removed) },
    });
  };

  const exhausted = total !== null && loadedRef.current >= total;

  const extraMenuItems = profile.nickname
    ? (file: FileItem) => [
        {
          id: "pickup-not-interested",
          label: t("pickup.notInterested"),
          icon: CircleSlash,
          disabled: pending.has(file.id),
          onSelect: () => void mark(file),
        },
      ]
    : undefined;

  return (
    <PageFrame
      width="wide"
      header={
        <PageHeader
          titleIcon={Sparkles}
          title={t("pickup.heading")}
          scope={t("pickup.description")}
        />
      }
    >
      {/* `px-4`, matching PageHeader's own padding. */}
      <div className="px-4 pb-6">

        {files.length > 0 && (
          <FileGrid files={files} extraMenuItems={extraMenuItems} />
        )}

        {files.length === 0 && !loading && total !== null && (
          <p className="py-12 text-center text-sm text-text-muted">
            {t("pickup.empty")}
          </p>
        )}

        {loading && (
          <p className="py-6 text-center text-sm text-text-muted">
            {t("pickup.loading")}
          </p>
        )}

        {failed && !loading && (
          <div className="py-6 text-center">
            <button
              type="button"
              onClick={retry}
              className="text-sm text-text-muted transition-colors hover:text-accent"
            >
              {t("pickup.retry")}
            </button>
          </div>
        )}

        {!exhausted && !failed && (
          <div ref={sentinel} className="h-px" aria-hidden />
        )}
      </div>
    </PageFrame>
  );
}
