import type { ClipTimestampItem } from "../api";

export const LOFT_MIME_TYPE = "application/vnd.litloft.loft+json";

/**
 * Step sizes for the time labels under the ruler, in seconds. The first one
 * that yields at most four interior labels wins; the list is short on purpose,
 * so a 4-hour video does not fall back to a step measured in weeks.
 */
const LABEL_STEPS_SECONDS = [
  60, 120, 300, 600, 900, 1800, 3600, 7200, 10800, 21600,
] as const;

/** An interior label closer than this to either end is omitted. */
const LABEL_EDGE_FRACTION = 0.08;

const MAX_INTERIOR_LABELS = 4;

export interface TimeLabel {
  time: number;
  fraction: number;
}

/**
 * The ruler's length. The player's duration wins when it can hold every
 * frame; otherwise the last frame plus the median gap keeps the track from
 * collapsing around a duration the player has not reported yet.
 */
export function rulerLength(
  rows: ClipTimestampItem[],
  duration: number,
): number {
  if (rows.length === 0) return 0;
  const last = rows[rows.length - 1].start;
  if (Number.isFinite(duration) && duration > 0 && duration >= last) {
    return duration;
  }
  if (rows.length === 1) return rows[0].start + 1;

  const gaps: number[] = [];
  for (let i = 1; i < rows.length; i += 1) {
    gaps.push(rows[i].start - rows[i - 1].start);
  }
  gaps.sort((a, b) => a - b);
  const middle = Math.floor(gaps.length / 2);
  const median =
    gaps.length % 2 === 1
      ? gaps[middle]
      : (gaps[middle - 1] + gaps[middle]) / 2;
  return last + median;
}

/** The last frame at or before `point`, or frame 0 before the first frame. */
export function frameIndexAt(
  rows: ClipTimestampItem[],
  point: number,
): number {
  if (rows.length === 0) return 0;
  let index = 0;
  for (let i = 0; i < rows.length; i += 1) {
    if (rows[i].start <= point) index = i;
    else break;
  }
  return index;
}

/** First frame of the 3×3 grid for a frame index. */
export function windowStartFor(index: number, count: number): number {
  const maxStart = Math.max(0, count - 9);
  return Math.max(0, Math.min(3 * Math.floor(index / 3) - 3, maxStart));
}

/**
 * Left-edge positions for the ruler's interior labels: 0:00 and the length
 * are drawn separately at the ends.
 */
export function timeLabels(length: number): TimeLabel[] {
  if (!(length > 0)) return [];
  for (const step of LABEL_STEPS_SECONDS) {
    const labels: TimeLabel[] = [];
    for (let time = step; time < length; time += step) {
      const fraction = time / length;
      if (fraction < LABEL_EDGE_FRACTION) continue;
      if (fraction > 1 - LABEL_EDGE_FRACTION) continue;
      labels.push({ time, fraction });
    }
    if (labels.length <= MAX_INTERIOR_LABELS) return labels;
  }
  return [];
}

/** Clamped fraction of the ruler for a time, for the ticks, marks and markers. */
export function rulerFraction(time: number, length: number): number {
  if (!(length > 0)) return 0;
  return Math.min(Math.max(time, 0), length) / length;
}
