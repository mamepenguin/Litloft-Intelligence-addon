"use client";

/**
 * Shared state for the detailed-summary citation marker + inline
 * overlay panel. Markers scattered across the summary dispatch
 * `setActive(citation)` / `scheduleClose()` / `cancelClose()`; the
 * panel subscribes and renders the excerpt for whichever citation is
 * currently active.
 *
 * Two activation modes coexist:
 *   - Hover: `setActive(citation)` with no `pin` flag — the marker
 *     and panel cooperate via `scheduleClose`/`cancelClose` so the
 *     panel auto-dismisses on mouseleave with a short grace period
 *     (lets the cursor hop between the trigger and the panel body).
 *   - Pin: `setActive(citation, { pin: true })` from a click — the
 *     panel stays open through scheduleClose fires until the user
 *     re-clicks the marker, clicks outside, or hits Escape.
 *
 * Excerpts are fetched lazily on first activation and cached by
 * chunk_id so repeated switches reuse the previous fetch.
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";

import { getCitationChunkExcerpt } from "./api";
import type { CitationChunkExcerpt, DetailedSummaryCitation } from "./api";

// Grace period between a mouseleave and auto-dismissal. Long enough to
// tolerate the sub-pixel vertical travel between the 🔗 icon and the
// panel body; short enough that an actual intent-to-leave still feels
// responsive.
const CLOSE_GRACE_MS = 160;

export type CitationFetchState =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "ready"; excerpt: CitationChunkExcerpt | null }
  | { kind: "error" };

interface ActiveCitation {
  citation: DetailedSummaryCitation;
  chunkId: string;
  // True when the active citation was opened by an explicit click.
  // `scheduleClose` is a no-op while pinned, so moving the cursor away
  // from the marker/panel doesn't dismiss it — only a re-click,
  // click-outside, Escape, or explicit `clearActive` does.
  pinned: boolean;
}

interface CitationRailContextValue {
  active: ActiveCitation | null;
  state: CitationFetchState;
  setActive: (
    citation: DetailedSummaryCitation,
    opts?: { pin?: boolean },
  ) => void;
  clearActive: () => void;
  // Start a grace timer that dismisses the active citation when it
  // fires, unless the active is pinned. Calling `cancelClose` before
  // the timer fires aborts it.
  scheduleClose: () => void;
  cancelClose: () => void;
}

const CitationRailContext = createContext<CitationRailContextValue | null>(
  null,
);

interface CitationRailProviderProps {
  fileId: string;
  drive: string;
  children: ReactNode;
}

export function CitationRailProvider({
  fileId,
  drive,
  children,
}: CitationRailProviderProps) {
  const [active, setActiveState] = useState<ActiveCitation | null>(null);
  const [state, setFetchState] = useState<CitationFetchState>({ kind: "idle" });
  // Cache keyed by chunkId — file/drive are stable for this provider.
  const cacheRef = useRef<Map<string, CitationChunkExcerpt | null>>(new Map());
  // Token to discard stale fetch results when the user switches markers
  // mid-request. Without this, a slow first fetch could overwrite the
  // state for the second (faster) marker.
  const fetchTokenRef = useRef(0);
  const closeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Sync'd mirror of `active` so the grace-timer callback can inspect
  // the latest pinned flag without re-registering the timer on every
  // render.
  const activeRef = useRef<ActiveCitation | null>(null);
  useEffect(() => {
    activeRef.current = active;
  }, [active]);

  const cancelClose = useCallback(() => {
    if (closeTimerRef.current != null) {
      clearTimeout(closeTimerRef.current);
      closeTimerRef.current = null;
    }
  }, []);

  const clearActive = useCallback(() => {
    cancelClose();
    fetchTokenRef.current += 1;
    setActiveState(null);
    setFetchState({ kind: "idle" });
  }, [cancelClose]);

  const scheduleClose = useCallback(() => {
    cancelClose();
    closeTimerRef.current = setTimeout(() => {
      closeTimerRef.current = null;
      // Pinned activations survive the grace window — they can only be
      // dismissed via `clearActive` (re-click, outside-click, Escape).
      if (activeRef.current?.pinned) return;
      fetchTokenRef.current += 1;
      setActiveState(null);
      setFetchState({ kind: "idle" });
    }, CLOSE_GRACE_MS);
  }, [cancelClose]);

  useEffect(() => () => cancelClose(), [cancelClose]);

  const setActive = useCallback(
    (citation: DetailedSummaryCitation, opts?: { pin?: boolean }) => {
      cancelClose();
      const pinned = opts?.pin ?? false;
      const chunkId = citation.chunk_ids[0] ?? null;
      if (!citation.has_citation || !chunkId) {
        // Missing-citation markers don't fetch — the panel branches on
        // `has_citation` and renders the warning copy directly.
        setActiveState({ citation, chunkId: "", pinned });
        setFetchState({ kind: "ready", excerpt: null });
        return;
      }
      setActiveState({ citation, chunkId, pinned });
      const cached = cacheRef.current.get(chunkId);
      if (cached !== undefined) {
        setFetchState({ kind: "ready", excerpt: cached });
        return;
      }
      const token = ++fetchTokenRef.current;
      setFetchState({ kind: "loading" });
      void (async () => {
        try {
          const excerpt = await getCitationChunkExcerpt(fileId, chunkId, drive);
          cacheRef.current.set(chunkId, excerpt);
          if (token !== fetchTokenRef.current) return;
          setFetchState({ kind: "ready", excerpt });
        } catch {
          if (token !== fetchTokenRef.current) return;
          setFetchState({ kind: "error" });
        }
      })();
    },
    [fileId, drive, cancelClose],
  );

  const value = useMemo<CitationRailContextValue>(
    () => ({ active, state, setActive, clearActive, scheduleClose, cancelClose }),
    [active, state, setActive, clearActive, scheduleClose, cancelClose],
  );

  return (
    <CitationRailContext.Provider value={value}>
      {children}
    </CitationRailContext.Provider>
  );
}

export function useCitationRail(): CitationRailContextValue {
  const ctx = useContext(CitationRailContext);
  if (!ctx) {
    throw new Error(
      "useCitationRail must be used inside a CitationRailProvider",
    );
  }
  return ctx;
}
