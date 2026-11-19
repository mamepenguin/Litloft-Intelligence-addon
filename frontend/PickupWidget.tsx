"use client";

/**
 * Pickup → the drive-home carousel.
 *
 * An entrance, not the feed. It asks for the day's window rather than
 * the head of the feed, and the difference is not cosmetic: lanes emit
 * at positions spaced by the reciprocal of their weight, so the first
 * dozen rows belong to the heaviest lanes and a quiet interest has not
 * appeared at all. Twelve sampled from the top forty track the
 * proportions the weighting intends.
 *
 * The row draws as many of the day's twelve as fit its width, so the
 * link through to the full feed is the only way to the rest — it is
 * offered whenever there is a feed at all, and the count on it says how
 * much is behind it. It used to appear only above forty, back when the
 * row drew all twelve and forty was the smallest feed the page was
 * worth opening for; with the row showing four on a phone, that gate
 * left twenty files unreachable.
 */

import { useEffect, useRef, useState } from "react";
import { CircleSlash, Sparkles } from "lucide-react";
import { useTranslations } from "next-intl";
import { batchGetFiles } from "@/lib/api";
import type { FileItem } from "@/types";
import { CarouselSection } from "@/components/CarouselSection";
import { useProfile } from "@/components/ProfileProvider";
import { useToast } from "@/components/ToastProvider";
import {
  fetchPickup,
  markPickupNotInterested,
  unmarkPickupNotInterested,
} from "./api";

const CAROUSEL_LIMIT = 12;

interface PickupWidgetProps {
  drive?: string;
}

/** A card this row removed, and everything needed to put it back. */
interface RemovedCard {
  item: FileItem;
  index: number;
  drive: string;
  viewer: string;
  /** The row's list when the card left; a different one holds no place for it. */
  generation: number;
}

export default function PickupWidget({ drive }: PickupWidgetProps) {
  const t = useTranslations("drive");
  const ti = useTranslations("intelligence");
  const tc = useTranslations("common");
  const profile = useProfile();
  const toast = useToast();
  const [files, setFiles] = useState<FileItem[]>([]);
  const [total, setTotal] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [pending, setPending] = useState<ReadonlySet<string>>(() => new Set());

  // A list replaced by a drive change must not receive an undo's card,
  // and a toast made by another profile must not write for this one.
  const generation = useRef(0);
  const viewer = useRef<string | null>(profile.nickname);
  useEffect(() => {
    viewer.current = profile.nickname;
  }, [profile.nickname]);

  useEffect(() => {
    generation.current += 1;

    if (!drive) {
      setLoading(false);
      return;
    }

    let cancelled = false;

    const load = async () => {
      setLoading(true);
      setTotal(null);
      setFiles([]);
      try {
        const page = await fetchPickup(drive, {
          limit: CAROUSEL_LIMIT,
          daily: true,
        });
        if (cancelled) return;
        setTotal(page.total);
        if (page.file_ids.length === 0) {
          setFiles([]);
          return;
        }
        const items = await batchGetFiles(page.file_ids);
        if (!cancelled) setFiles(items);
      } catch {
        // Silently fail — the slot simply won't render.
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    load();
    return () => {
      cancelled = true;
    };
  }, [drive]);

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
      setTotal((prev) => (prev === null ? prev : Math.max(0, prev - 1)));
    }
    toast.success(ti("pickup.notInterestedToast"), {
      action: { label: tc("undo"), onClick: () => void undo(removed) },
    });
  };

  // Nothing until the feed has answered: on a drive with no pickup, a
  // skeleton row that then vanishes is a section that was never there.
  if (total === null || (!loading && files.length === 0)) return null;

  const seeAllHref = drive
    ? `/drive/${encodeURIComponent(drive)}/addons/intelligence/pickup`
    : undefined;

  const extraMenuItems = profile.nickname
    ? (file: FileItem) => [
        {
          id: "pickup-not-interested",
          label: ti("pickup.notInterested"),
          icon: CircleSlash,
          disabled: pending.has(file.id),
          onSelect: () => void mark(file),
        },
      ]
    : undefined;

  return (
    <CarouselSection
      title={t("pickup")}
      icon={<Sparkles size={20} className="text-text-muted" />}
      files={files}
      loading={loading}
      seeAllHref={seeAllHref}
      // The size of the feed, not of the day's window: the link leads to
      // the whole thing, and the number beside it has to be the number
      // the reader arrives at.
      totalCount={total ?? undefined}
      extraMenuItems={extraMenuItems}
    />
  );
}
