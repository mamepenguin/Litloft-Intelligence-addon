/**
 * Where the reader had got to in a transcript, kept across unmounts.
 *
 * On a phone the inspector is a bottom sheet that rests at a 56px strip,
 * and the drawer below that strip is mounted only while it is expanded —
 * core has to, because vaul renders a modal Radix dialog and one left
 * mounted at rest puts `aria-hidden` on the whole application. So
 * collapsing the sheet unmounts this panel, and raising it again mounts
 * a new one.
 *
 * Almost everything that costs is bought back by the refetch: the cues,
 * the language, which source is showing, the highlight. The scroll
 * position is the one thing the refetch cannot return, because it is not
 * a fact about the file — it is a fact about the reader. Without this a
 * reader who has followed a transcript for twenty minutes, glanced at
 * the file's tags and come back lands at the top of it.
 *
 * `following` travels with it and is not a separate nicety. Restoring
 * only the offset on a playing file puts the reader back where they were
 * and then, a second later, drags them to the cue that is playing —
 * which is the state they left precisely by scrolling away from.
 *
 * Module-level rather than a store with subscribers: nothing re-renders
 * on a change, and the only reader is the next mount for the same file.
 */

export interface TranscriptScrollState {
  /** `scrollTop` of the cue list, in px. */
  top: number;
  /** Whether the highlight was still allowed to drag the list around. */
  following: boolean;
}

/**
 * How many files are remembered at once.
 *
 * A session moves through more files than a reader returns to, and the
 * cost of a miss is landing at the top — the behaviour before this
 * existed. Bounded rather than unbounded because this is module state
 * that nothing ever clears: a tab left open for a week browsing a large
 * drive would otherwise keep an entry per file opened.
 */
const REMEMBERED_FILES = 20;

const positions = new Map<string, TranscriptScrollState>();

export function rememberTranscriptScroll(
  fileId: string,
  state: TranscriptScrollState,
): void {
  // Delete first so a re-set moves the entry to the end: `Map` iterates
  // in insertion order, which is what makes the eviction below the
  // least-recently-written one rather than the oldest-ever.
  positions.delete(fileId);
  positions.set(fileId, state);
  if (positions.size > REMEMBERED_FILES) {
    const oldest = positions.keys().next();
    if (!oldest.done) positions.delete(oldest.value);
  }
}

export function recallTranscriptScroll(
  fileId: string,
): TranscriptScrollState | undefined {
  return positions.get(fileId);
}

/** Test seam. Nothing in the app clears this. */
export function clearTranscriptScroll(): void {
  positions.clear();
}
