import type { CitationPosition } from "./askStream";

/** `M:SS` below one hour, `H:MM:SS` from one hour. */
export function formatSeconds(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const rest = total % 60;
  const mm = String(minutes).padStart(2, "0");
  const ss = String(rest).padStart(2, "0");
  return hours > 0 ? `${hours}:${mm}:${ss}` : `${minutes}:${ss}`;
}

/** The label a citation pill shows for a position. */
export function citationLabel(position: CitationPosition): string | null {
  if (!position) return null;
  if ("seconds" in position) return formatSeconds(position.seconds);
  if ("page" in position) return `p.${position.page}`;
  return position.title ?? `§${position.section}`;
}
