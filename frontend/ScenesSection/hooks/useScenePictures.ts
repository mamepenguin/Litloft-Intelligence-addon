import { useCallback, useEffect, useRef, useState } from "react";

/** How long a tile must stay near the visible area before its picture is asked for. */
export const PICTURE_DWELL_MS = 200;

export interface ScenePictureOptions {
  /** Frames near the visible area, in time order. */
  times: number[];
  active: boolean;
  /** Until the reader first scrolls the list or uses the ruler after a start event. */
  opening: boolean;
  /** False until the start placement's own reading has been taken. */
  ready: boolean;
  /** Changes when the showing, the file or the frames change. */
  resetKey: string;
}

/**
 * The frames whose pictures may be fetched. Opening asks for every near tile at
 * once; afterwards each tile is asked for once it has stayed near for the dwell
 * time, however much the list moves around it.
 */
export function useScenePictures({
  times,
  active,
  opening,
  ready,
  resetKey,
}: ScenePictureOptions): number[] {
  const [requested, setRequested] = useState<number[]>([]);
  const requestedRef = useRef(new Set<number>());
  const dwellsRef = useRef(new Map<number, number>());
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const timesRef = useRef(times);
  timesRef.current = times;
  const stateRef = useRef({ active, opening, ready });
  stateRef.current = { active, opening, ready };

  const request = useCallback((wanted: number[]) => {
    let changed = false;
    for (const time of wanted) {
      if (!requestedRef.current.has(time)) {
        requestedRef.current.add(time);
        changed = true;
      }
    }
    if (changed) {
      setRequested([...requestedRef.current].sort((a, b) => a - b));
    }
  }, []);

  const clearTimer = useCallback(() => {
    if (timerRef.current !== null) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const read = useCallback(
    (now: number) => {
      const state = stateRef.current;
      if (!state.active || !state.ready) return;
      const near = new Set(timesRef.current);
      for (const time of Array.from(dwellsRef.current.keys())) {
        if (!near.has(time)) dwellsRef.current.delete(time);
      }
      if (state.opening) {
        dwellsRef.current.clear();
        request(timesRef.current);
        return;
      }
      for (const time of near) {
        if (!dwellsRef.current.has(time)) dwellsRef.current.set(time, now);
      }
      const due: number[] = [];
      let next = Number.POSITIVE_INFINITY;
      for (const [time, start] of dwellsRef.current) {
        const deadline = start + PICTURE_DWELL_MS;
        if (deadline <= now) due.push(time);
        else if (deadline < next) next = deadline;
      }
      request(due);
      clearTimer();
      if (Number.isFinite(next)) {
        timerRef.current = setTimeout(
          () => {
            timerRef.current = null;
            read(Date.now());
          },
          Math.max(0, next - now),
        );
      }
    },
    [clearTimer, request],
  );

  useEffect(() => {
    requestedRef.current = new Set();
    dwellsRef.current.clear();
    clearTimer();
    setRequested((current) => (current.length === 0 ? current : []));
  }, [resetKey, active, clearTimer]);

  useEffect(() => {
    if (!active) {
      dwellsRef.current.clear();
      clearTimer();
      return;
    }
    read(Date.now());
  }, [times, active, opening, ready, read, clearTimer]);

  useEffect(() => clearTimer, [clearTimer]);

  return requested;
}
