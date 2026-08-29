/**
 * Display-layer transforms for a passage.
 *
 * A passage arrives as a whole chunk — up to 400 characters, cut where
 * the chunker cut, which is routinely mid-word. Nothing here rewrites
 * the string: the section's guarantee is that what it shows is
 * byte-identical to the text whose vector produced the score, so these
 * functions only choose *which* slice to show.
 *
 * Spec ``2026-08-30-related-passages-recognition-ui.md`` §7.5.
 */

/** Run-up kept in front of the term the window is centred on. */
const LEAD_CHARS = 30;

/**
 * How far forward to look for a sentence boundary.
 *
 * Only the *first* terminator is a candidate. Everything after it is a
 * complete sentence and therefore real content — snapping past those
 * would discard the passage rather than tidy its opening.
 */
const SNAP_LOOKAHEAD = 60;

/** Below this many characters left, snapping would leave a stub. */
const MIN_REMAINDER = 12;

const CJK_TERMINATORS = "。．！？…\n";
const ASCII_TERMINATORS = ".!?";
/** Punctuation that trails a terminator and belongs to the sentence before it. */
const TRAILING_MARKS = "」』）】〉》\"'";

/** Punctuation a sentence can open with, before its first real character. */
const OPENING_MARKS = "「『（【〈《\"'“‘([";

export interface PassageWindow {
  text: string;
  /** True when the slice starts mid-passage and needs a leading ellipsis. */
  truncatedStart: boolean;
}

export interface PassageSegment {
  text: string;
  /** Whether this run is one of the terms both passages share. */
  marked: boolean;
}

/**
 * Split a passage around the terms it shares with the other one.
 *
 * The point is not decoration: an unpunctuated transcript chunk is a
 * wall, and marks are what turn it into something the eye can land in.
 * Concatenating every segment reproduces the input exactly — the row
 * promises the words are the passage's own, so this may only choose
 * where to draw, never what to show.
 *
 * Longest term first, because a short term inside a longer one would
 * otherwise cut it in half and leave the remainder unmarked beside it.
 */
export function highlightSegments(
  text: string,
  terms: string[],
): PassageSegment[] {
  const wanted = terms
    .filter(Boolean)
    .slice()
    .sort((a, b) => b.length - a.length);
  if (wanted.length === 0) return [{ text, marked: false }];

  const haystack = text.toLowerCase();
  const segments: PassageSegment[] = [];
  let plainFrom = 0;
  let i = 0;

  while (i < text.length) {
    const hit = wanted.find(
      (term) => haystack.startsWith(term.toLowerCase(), i), // case-folded, like the backend's intersection
    );
    if (!hit) {
      i += 1;
      continue;
    }
    if (i > plainFrom) {
      segments.push({ text: text.slice(plainFrom, i), marked: false });
    }
    // Sliced from the passage, not from the term: the mark shows the
    // casing the author used.
    segments.push({ text: text.slice(i, i + hit.length), marked: true });
    i += hit.length;
    plainFrom = i;
  }

  if (plainFrom < text.length) {
    segments.push({ text: text.slice(plainFrom), marked: false });
  }
  return segments;
}

/**
 * Where a passage should start being shown.
 *
 * `line-clamp` still decides where it ends; this only moves the left
 * edge. Two things move it:
 *
 * 1. **The first matched term**, so a term buried at the end of a chunk
 *    is on screen at all. Measured on real pairs, that happens often
 *    enough that highlighting without windowing would leave rows
 *    unchanged.
 * 2. **The nearest sentence boundary**, which is what stops a row
 *    opening on the severed word a chunk boundary left behind. This one
 *    is punctuation-based and so works in any language — and it is the
 *    only one of the two that applies when there are no terms, which is
 *    every non-Japanese row (§8) and 14% of Japanese ones.
 */
export function passageWindow(text: string, terms: string[]): PassageWindow {
  if (!text) return { text: "", truncatedStart: false };

  const matchAt = earliestMatch(text, terms);
  const base = matchAt >= 0 ? Math.max(0, matchAt - LEAD_CHARS) : 0;
  const snapped = opensMidSentence(text, base)
    ? sentenceStartFrom(text, base)
    : null;

  let start = base;
  if (
    snapped !== null &&
    // Never step past the term the window exists to show.
    (matchAt < 0 || snapped <= matchAt) &&
    text.length - snapped >= MIN_REMAINDER
  ) {
    start = snapped;
  }

  return { text: text.slice(start), truncatedStart: start > 0 };
}

/**
 * Whether the text at ``from`` looks like the middle of a sentence.
 *
 * Snapping forward discards whatever precedes the boundary, so it is
 * only justified where that text is a remnant. Two cases:
 *
 * - **``from`` is an offset we chose** (a term's run-up). It lands
 *   wherever arithmetic put it, so it is mid-sentence by construction.
 * - **``from`` is the start of the chunk.** Here the passage may well
 *   begin on a real sentence, and snapping would throw a whole one
 *   away. A capital or a digit is the evidence that it does: a chunker
 *   splitting on size lands inside a word — `ing chunks…` — and a
 *   severed word begins with neither. The test looks past any quote or
 *   bracket the sentence opened with, and reads case through Unicode
 *   rather than ASCII, so `"The claim."` and `Ärger` both count.
 *   Japanese has no case to read and no spaces for a chunker to
 *   respect, so a fragment there is the common case and the snap
 *   stands.
 */
function opensMidSentence(text: string, from: number): boolean {
  if (from > 0) return true;

  // Whatever a sentence opens with sits behind any quotes or brackets
  // that open with it.
  let i = 0;
  while (i < text.length && OPENING_MARKS.includes(text[i])) i += 1;
  const first = text[i];
  if (!first) return false;

  // A capital in any cased script, or a digit: neither is where a
  // severed word begins. Japanese has no case, so this finds nothing
  // and the snap stands — which is right, since a chunker with no
  // spaces to respect is exactly what leaves fragments.
  const isCapital = first !== first.toLowerCase() && first === first.toUpperCase();
  return !isCapital && !/\d/.test(first);
}

/**
 * Index of the earliest occurrence of any term, or -1.
 *
 * Case-folded, because the terms are the words the *source* passage
 * used and the window is applied to the *match* — the two may spell a
 * shared latin word differently, and the backend's intersection and the
 * highlighter both already fold. Comparing exactly here would leave a
 * term chipped and marked yet outside the two visible lines.
 */
function earliestMatch(text: string, terms: string[]): number {
  const haystack = text.toLowerCase();
  let best = -1;
  for (const term of terms) {
    if (!term) continue;
    const at = haystack.indexOf(term.toLowerCase());
    if (at >= 0 && (best < 0 || at < best)) best = at;
  }
  return best;
}

/**
 * Start of the first whole sentence at or after ``from``, or null.
 */
function sentenceStartFrom(text: string, from: number): number | null {
  const limit = Math.min(from + SNAP_LOOKAHEAD, text.length - 1);
  for (let i = from; i <= limit; i += 1) {
    const ch = text[i];
    if (!CJK_TERMINATORS.includes(ch) && !isAsciiSentenceEnd(text, i)) continue;

    let start = i + 1;
    while (
      start < text.length &&
      (/\s/.test(text[start]) || TRAILING_MARKS.includes(text[start]))
    ) {
      start += 1;
    }
    return start;
  }
  return null;
}

/**
 * Whether the ASCII terminator at ``i`` really ends a sentence.
 *
 * Whitespace must follow, so a decimal point does not split the line —
 * but a closing quote or bracket may sit between the two, and a period
 * is no less final for being inside the quotation it ends.
 *
 * `Dr. Smith` is knowingly accepted here. Excluding it needs a list of
 * abbreviations, which is per-language data of the kind this feature
 * declines to carry, and any list-free heuristic ("short word before
 * the period") would also suppress real boundaries. The cost of being
 * wrong is one word missing from the front of an excerpt whose full
 * text is one press away.
 */
function isAsciiSentenceEnd(text: string, i: number): boolean {
  if (!ASCII_TERMINATORS.includes(text[i])) return false;
  let after = i + 1;
  while (after < text.length && TRAILING_MARKS.includes(text[after])) {
    after += 1;
  }
  return after >= text.length || /\s/.test(text[after]);
}
