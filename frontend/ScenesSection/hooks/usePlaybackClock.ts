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

function isJump(
  previous: PlaybackState,
  next: PlaybackState,
  elapsedSeconds: number,
  rate: number,
): boolean {
  const step = next.time - previous.time;
  if (previous.paused) {
    if (next.paused) {
      return step !== 0 && Math.abs(step) > PAUSED_JUMP_SECONDS;
    }
    // A paused sample does not say when playback started, so forward travel is
    // bounded by the whole interval; backwards travel cannot be playback.
    return (
      -step > PAUSED_JUMP_SECONDS ||
      step > elapsedSeconds * rate + PLAYING_JUMP_SECONDS
    );
  }
  const expected = previous.time + elapsedSeconds * rate;
  return step !== 0 && Math.abs(next.time - expected) > PLAYING_JUMP_SECONDS;
}

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
      if (isJump(previous, next, (now - previous.at) / 1000, rate)) {
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
