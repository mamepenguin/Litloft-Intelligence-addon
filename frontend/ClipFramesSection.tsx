"use client";

import { useCallback, useEffect, useState } from "react";
import { Film } from "lucide-react";
import { useTranslations } from "next-intl";

import { getClipTimestamps, getFrameUrl } from "./api";
import type { ClipTimestampItem } from "./api";
import { formatDuration } from "@/lib/format";
import type { MediaController } from "@/lib/mediaController";

interface ClipFramesSectionProps {
  fileId: string;
  drive: string;
  // Legacy: still accepted for callers that pass a native video ref.
  videoRef?: React.RefObject<HTMLVideoElement | null>;
  // Preferred: works for native video AND HvLink (YouTube) embeds.
  mediaController?: MediaController | null;
}

const INITIAL_SHOW = 20;

export default function ClipFramesSection({ fileId, drive, videoRef, mediaController }: ClipFramesSectionProps) {
  const t = useTranslations("searchIndex");
  const [timestamps, setTimestamps] = useState<ClipTimestampItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [available, setAvailable] = useState(false);
  const [showCount, setShowCount] = useState(INITIAL_SHOW);

  useEffect(() => {
    setLoading(true);
    getClipTimestamps(fileId, drive).then((res) => {
      if (res.available && res.timestamps && res.timestamps.length > 0) {
        setTimestamps(res.timestamps);
        setAvailable(true);
      } else {
        setAvailable(false);
      }
      setLoading(false);
    });
  }, [fileId, drive]);

  const seekTo = useCallback(
    (time: number) => {
      // Prefer the unified controller (covers HvLink/YouTube) and fall
      // back to the legacy native video ref for callers that haven't
      // migrated yet.
      if (mediaController) {
        mediaController.seek(time);
        mediaController.play();
        return;
      }
      const video = videoRef?.current;
      if (video) {
        video.currentTime = time;
      }
    },
    [videoRef, mediaController]
  );

  if (loading || !available) return null;

  const visible = timestamps.slice(0, showCount);
  const hasMore = timestamps.length > showCount;

  return (
    <div>
      <div className="mb-2 flex items-center gap-2 text-sm text-text-muted">
        <Film size={14} />
        <span>{t("clipTitle")}</span>
        <span className="text-xs">({timestamps.length})</span>
      </div>
      <div className="grid grid-cols-3 gap-2 sm:grid-cols-4 md:grid-cols-5">
        {visible.map((ts) => (
          <button
            key={ts.start}
            onClick={() => seekTo(ts.start)}
            className="group cursor-pointer overflow-hidden rounded-lg bg-bg-card transition-colors hover:ring-2 hover:ring-accent"
          >
            <img
              src={getFrameUrl(fileId, ts.start)}
              alt={ts.content_preview}
              loading="lazy"
              className="aspect-video w-full object-cover"
            />
            <div className="px-1.5 py-1 text-center text-xs text-text-muted group-hover:text-accent">
              {formatDuration(ts.start)}
            </div>
          </button>
        ))}
      </div>
      {hasMore && (
        <button
          onClick={() => setShowCount((c) => c + INITIAL_SHOW)}
          className="mt-2 w-full cursor-pointer rounded-lg bg-bg-card py-1.5 text-center text-sm text-text-muted transition-colors hover:text-text-primary"
        >
          {t("showMore")} (+{Math.min(INITIAL_SHOW, timestamps.length - showCount)})
        </button>
      )}
    </div>
  );
}
