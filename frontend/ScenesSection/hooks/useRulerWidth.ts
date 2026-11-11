import { useLayoutEffect, useState, type RefObject } from "react";

/**
 * The ruler's measured width, which decides how many time labels fit.
 * `active` reruns the measurement when the ruler element mounts.
 */
export function useRulerWidth(
  ref: RefObject<HTMLDivElement | null>,
  active: boolean,
): number {
  const [width, setWidth] = useState(0);

  useLayoutEffect(() => {
    if (!active) return;
    const el = ref.current;
    if (!el) return;
    const measure = () => setWidth(el.getBoundingClientRect().width);
    measure();
    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  }, [active, ref]);

  return width;
}
