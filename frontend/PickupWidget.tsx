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
 * The link through to the full feed appears only once there is enough
 * behind it to be worth the trip.
 */

import { useEffect, useState } from "react";
import { Sparkles } from "lucide-react";
import { useTranslations } from "next-intl";
import { batchGetFiles } from "@/lib/api";
import type { FileItem } from "@/types";
import { CarouselSection } from "@/components/CarouselSection";
import { fetchPickup } from "./api";

/** Below this the feed page has too little to show; carousel only. */
const FEED_MIN_STOCK = 40;

const CAROUSEL_LIMIT = 12;

interface PickupWidgetProps {
  drive?: string;
}

export default function PickupWidget({ drive }: PickupWidgetProps) {
  const t = useTranslations("drive");
  const [files, setFiles] = useState<FileItem[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!drive) {
      setLoading(false);
      return;
    }

    let cancelled = false;

    const load = async () => {
      setLoading(true);
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

  const seeAllHref =
    drive && total >= FEED_MIN_STOCK
      ? `/drive/${encodeURIComponent(drive)}/addons/intelligence/pickup`
      : undefined;

  return (
    <CarouselSection
      title={t("pickup")}
      icon={<Sparkles size={20} className="text-accent-cta" />}
      files={files}
      loading={loading}
      seeAllHref={seeAllHref}
    />
  );
}
