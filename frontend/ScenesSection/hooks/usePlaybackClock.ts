import { useEffect, useState } from "react";

import { getMediaClockSnapshot, subscribeMediaClock } from "@/lib/mediaClock";
import type { MediaController } from "@/lib/mediaController";

export interface PlaybackState {
  time: number;
  duration: number;
  paused: boolean;
}

const IDLE_PLAYBACK: PlaybackState = { time: 0, duration: 0, paused: true };

function snapshotOf(controller: MediaController): PlaybackState {
  const snapshot = getMediaClockSnapshot(controller);
  return {
    time: snapshot.currentTime,
    duration: snapshot.duration,
    paused: snapshot.paused,
  };
}

export function usePlaybackClock(
  controller: MediaController | null,
): PlaybackState {
  const [playback, setPlayback] = useState<PlaybackState>(() =>
    controller ? snapshotOf(controller) : IDLE_PLAYBACK,
  );

  useEffect(() => {
    if (!controller) {
      setPlayback(IDLE_PLAYBACK);
      return;
    }
    const update = () => {
      const next = snapshotOf(controller);
      setPlayback((current) =>
        current.time === next.time &&
        current.duration === next.duration &&
        current.paused === next.paused
          ? current
          : next,
      );
    };
    const unsubscribe = subscribeMediaClock(controller, update);
    update();
    return unsubscribe;
  }, [controller]);

  return playback;
}
