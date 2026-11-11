export type BrowseSource = "mouse" | "touch" | "keyboard";

/** Something that happened to the section, as far as a browse point cares. */
export type BrowseClearEvent =
  | "pointerleave"
  | "focusleave"
  | "jump"
  | "seek"
  | "collapse"
  | "filechange";

/**
 * The events that clear a browse point, by the input that set it. An event is
 * listed only under the inputs whose point it ends, so an omission here is the
 * rule and not a missing entry.
 */
const CLEARED_BY: Record<BrowseSource, readonly BrowseClearEvent[]> = {
  mouse: ["pointerleave", "seek", "collapse", "filechange"],
  touch: ["seek", "jump", "collapse", "filechange"],
  keyboard: ["focusleave", "jump", "seek", "collapse", "filechange"],
};

/** No source means no point is set, so no event can clear it. */
export function browseClearedBy(
  source: BrowseSource | null,
  event: BrowseClearEvent,
): boolean {
  return source !== null && CLEARED_BY[source].includes(event);
}
