import { useEffect, useRef, useState } from "react";

import { getMediaClockSnapshot, subscribeMediaClock } from "@/lib/mediaClock";
import type { MediaController } from "@/lib/mediaController";

export interface PlaybackState {
  time: number;
  duration: number;
  paused: boolean;
}

const IDLE_PLAYBACK: PlaybackState = { time: 0, duration: 0, paused: true };

/** Steady playback at the current rate; a larger miss is a jump. */
const PLAYING_JUMP_SECONDS = 2;
/** A paused player can still be scrubbed by a shortcut or a seek. */
const PAUSED_JUMP_SECONDS = 0.5;

function readSnapshot(controller: MediaController): PlaybackState {
  try {
    const snapshot = getMediaClockSnapshot(controller);
    return {
      time: snapshot.currentTime,
      duration: snapshot.duration,
      paused: snapshot.paused,
    };
  } catch {
    return IDLE_PLAYBACK;
  }
}

/**
 * The shared media clock, followed locally so the section alone decides what
 * a jump is. `onJump` fires between two samples whose distance is larger than
 * playback would explain.
 */
export function usePlaybackClock(
  controller: MediaController | null,
  onJump: () => void,
): PlaybackState {
  const [playback, setPlayback] = useState<PlaybackState>(() =>
    controller ? readSnapshot(controller) : IDLE_PLAYBACK,
  );
  const onJumpRef = useRef(onJump);
  onJumpRef.current = onJump;

  useEffect(() => {
    if (!controller) {
      setPlayback(IDLE_PLAYBACK);
      return;
    }

    let previous = { ...readSnapshot(controller), at: Date.now() };

    const update = () => {
      const next = readSnapshot(controller);
      const now = Date.now();
      const rate = controller.getPlaybackRate() || 1;
      const expected = previous.paused
        ? previous.time
        : previous.time + ((now - previous.at) / 1000) * rate;
      const limit = previous.paused
        ? PAUSED_JUMP_SECONDS
        : PLAYING_JUMP_SECONDS;
      if (next.time !== previous.time && Math.abs(next.time - expected) > limit) {
        onJumpRef.current();
      }
      previous = { ...next, at: now };
      setPlayback(next);
    };

    const unsubscribe = subscribeMediaClock(controller, update);
    update();
    return unsubscribe;
  }, [controller]);

  return playback;
}
