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

import { useEffect, useState } from "react";
import { Sparkles } from "lucide-react";
import { useTranslations } from "next-intl";
import { batchGetFiles } from "@/lib/api";
import type { FileItem } from "@/types";
import { CarouselSection } from "@/components/CarouselSection";
import { fetchPickup } from "./api";

const CAROUSEL_LIMIT = 12;

interface PickupWidgetProps {
  drive?: string;
}

export default function PickupWidget({ drive }: PickupWidgetProps) {
  const t = useTranslations("drive");
  const [files, setFiles] = useState<FileItem[]>([]);
  // `null`, not `0`, until a fetch has answered. The row's link carries
  // this number, and core's contract is that an unknown total means an
  // unqualified "See all" rather than a claimed one — `DriveHome` says
  // so where it threads the same field. With `0` the row spent its whole
  // load saying "See all (0)" beside a set of skeletons.
  //
  // The reset in the effect below covers every load after the first, and
  // is the half a test can see; this initial value covers the one frame
  // before the effect runs, which React has committed past by the time a
  // test can look.
  const [total, setTotal] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!drive) {
      setLoading(false);
      return;
    }

    let cancelled = false;

    const load = async () => {
      setLoading(true);
      setTotal(null);
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

  if (!loading && files.length === 0) return null;

  const seeAllHref = drive
    ? `/drive/${encodeURIComponent(drive)}/addons/intelligence/pickup`
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
    />
  );
}
