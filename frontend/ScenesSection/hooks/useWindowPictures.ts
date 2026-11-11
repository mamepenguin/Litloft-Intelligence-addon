import { useEffect, useState } from "react";

/**
 * Frames whose picture may be fetched. A sweep across the ruler changes the
 * window on every pointer move, and each frame-cache miss runs ffmpeg, so the
 * window has to stay put before its pictures are asked for.
 */
export const WINDOW_SETTLE_MS = 150;

export function useWindowPictures(
  times: number[],
  active: boolean,
): number[] {
  const key = active ? times.join(",") : "";
  const [requested, setRequested] = useState<number[]>([]);

  useEffect(() => {
    if (!key) {
      setRequested((current) => (current.length === 0 ? current : []));
      return;
    }
    const value = key.split(",").map(Number);
    const timer = setTimeout(() => setRequested(value), WINDOW_SETTLE_MS);
    return () => clearTimeout(timer);
  }, [key]);

  return requested;
}
