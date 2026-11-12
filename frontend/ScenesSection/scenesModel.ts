import { formatDuration } from "@/lib/format";
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

/** A kept label's drawn box keeps at least this much room to its neighbour. */
const LABEL_GAP_PX = 4;

const MAX_INTERIOR_LABELS = 4;

export interface TimeLabel {
  time: number;
  fraction: number;
  /** Left edge of the drawn box, in px from the ruler's left edge. */
  left: number;
  /** Width of the drawn box in px. */
  width: number;
}

/**
 * A conservative width for a label at 11px with tabular digits: 7px per
 * character is wider than the drawn span, so the collision check never keeps
 * a label that touches its neighbour.
 */
export function estimateLabelWidth(text: string): number {
  return text.length * 7;
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

const DEFAULT_ROOT_FONT_SIZE_PX = 16;

/** The narrowest tile, its width at three columns in the 384px inspector. */
const MIN_TILE_REM = 7;

/** The gap between columns. */
const COLUMN_GAP_REM = 0.5;

const MIN_COLUMNS = 3;

const MIN_ROWS = 3;

/**
 * Columns the section draws at a measured width. The root font size is read
 * here rather than assumed to be 16px so scaled text keeps the same layout.
 */
export function sceneColumns(width: number): number {
  const rem = rootFontSizePx();
  const fits = Math.floor(
    (width + COLUMN_GAP_REM * rem) /
      ((MIN_TILE_REM + COLUMN_GAP_REM) * rem),
  );
  return Math.max(MIN_COLUMNS, fits);
}

function rootFontSizePx(): number {
  if (typeof document === "undefined") return DEFAULT_ROOT_FONT_SIZE_PX;
  const size = Number.parseFloat(
    getComputedStyle(document.documentElement).fontSize,
  );
  return Number.isFinite(size) && size > 0 ? size : DEFAULT_ROOT_FONT_SIZE_PX;
}

/**
 * Rows the grid can draw: the drawn rows plus the whole steps the slack still
 * holds, never fewer than three. Without a measurable step or a finite slack,
 * returns the drawn rows, never fewer than three.
 */
export function sceneRows(
  drawnRows: number,
  slack: number,
  step: number,
): number {
  if (!(step > 0) || !Number.isFinite(slack)) {
    return Math.max(MIN_ROWS, drawnRows);
  }
  return Math.max(MIN_ROWS, drawnRows + Math.floor(slack / step));
}

/**
 * The tile's height at a scaled width: the picture grows with the width, the
 * time label under it keeps its height.
 */
export function scaledTileHeight(
  tileHeight: number,
  pictureHeight: number,
  scale: number,
): number {
  return tileHeight - pictureHeight + pictureHeight * scale;
}

/** First frame of the window of `rows` rows, which holds `columns` frames a row. */
export function windowStartFor(
  index: number,
  count: number,
  columns: number,
  rows: number,
): number {
  const maxStart = Math.max(0, count - rows * columns);
  return Math.max(
    0,
    Math.min(columns * Math.floor(index / columns) - columns, maxStart),
  );
}

/**
 * The ruler's interior labels, with the drawn box each one gets: 0:00 and the
 * length are drawn separately at the ends.
 */
export function timeLabels(
  length: number,
  rulerWidth: number,
  labelWidth: (text: string) => number,
): TimeLabel[] {
  if (!(length > 0) || !(rulerWidth > 0)) return [];
  const times = interiorLabelTimes(length);
  const endLeft = rulerWidth - labelWidth(formatDuration(length));
  let previousRight = labelWidth(formatDuration(0));
  const kept: TimeLabel[] = [];
  for (const time of times) {
    const fraction = time / length;
    const width = labelWidth(formatDuration(time));
    const left = fraction * rulerWidth - width / 2;
    const right = left + width;
    if (left - previousRight < LABEL_GAP_PX) continue;
    if (endLeft - right < LABEL_GAP_PX) continue;
    kept.push({ time, fraction, left, width });
    previousRight = right;
  }
  return kept;
}

/**
 * The step is the first one that yields at most four labels, before the boxes
 * are known. It is deliberately independent of the width so the labels keep
 * their times when the section is drawn at another size.
 */
function interiorLabelTimes(length: number): number[] {
  for (const step of LABEL_STEPS_SECONDS) {
    const times: number[] = [];
    for (let time = step; time < length; time += step) {
      times.push(time);
    }
    if (times.length <= MAX_INTERIOR_LABELS) return times;
  }
  return [];
}

/** Clamped fraction of the ruler for a time, for the ticks, marks and markers. */
export function rulerFraction(time: number, length: number): number {
  if (!(length > 0)) return 0;
  return Math.min(Math.max(time, 0), length) / length;
}
