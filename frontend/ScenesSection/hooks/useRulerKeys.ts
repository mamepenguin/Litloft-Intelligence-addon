import { useEffect, type RefObject } from "react";

/** The ruler's native key listener, attached only while the ruler is enabled. */
export function useRulerKeys(
  ref: RefObject<HTMLDivElement | null>,
  onKeyDown: (event: KeyboardEvent) => void,
  enabled: boolean,
): void {
  useEffect(() => {
    if (!enabled) return;
    const el = ref.current;
    if (!el) return;
    // A React handler runs in the root container under the App Router, where
    // stopping propagation cannot keep the key from the document listener
    // ShortcutsProvider adds for the same event.
    const listener = (event: KeyboardEvent) => onKeyDown(event);
    el.addEventListener("keydown", listener);
    return () => el.removeEventListener("keydown", listener);
  }, [enabled, onKeyDown, ref]);
}
