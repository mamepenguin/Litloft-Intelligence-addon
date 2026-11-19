import { useEffect, useRef, useState } from "react";

/** How long a row must stay put before its picture is asked for. */
export const PICTURE_SETTLE_MS = 150;

export interface ScenePictureOptions {
  /** Frames within one row step of the visible area, in time order. */
  times: number[];
  active: boolean;
  /** Changes when the showing, the file or the rows change. */
  resetKey: string;
  /** Changes on every scroll of the scroller, so a still list is required. */
  settleKey: number;
}

/**
 * The frames whose pictures may be fetched. A drawn picture is never dropped
 * while the panel stays shown, and a reader's scroll restarts the settle, so a
 * sweep through the list asks for nothing until it stops.
 */
export function useScenePictures({
  times,
  active,
  resetKey,
  settleKey,
}: ScenePictureOptions): number[] {
  const [requested, setRequested] = useState<number[]>([]);
  const requestedRef = useRef(new Set<number>());

  useEffect(() => {
    requestedRef.current = new Set();
    setRequested((current) => (current.length === 0 ? current : []));
  }, [resetKey, active]);

  const timesRef = useRef(times);
  useEffect(() => {
    timesRef.current = times;
  }, [times]);

  const key = active ? times.join(",") : "";
  useEffect(() => {
    if (!active || !key) return;
    const timer = setTimeout(() => {
      let changed = false;
      for (const time of timesRef.current) {
        if (!requestedRef.current.has(time)) {
          requestedRef.current.add(time);
          changed = true;
        }
      }
      if (changed) {
        setRequested([...requestedRef.current].sort((a, b) => a - b));
      }
    }, PICTURE_SETTLE_MS);
    return () => clearTimeout(timer);
    // `times` is read through `key`, which changes only when the frames do;
    // `settleKey` restarts the settle without changing them.
  }, [key, active, resetKey, settleKey]);

  return requested;
}
