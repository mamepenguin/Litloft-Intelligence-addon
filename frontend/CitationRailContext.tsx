"use client";

/**
 * Shared state for the detailed-summary citation accordion.
 *
 * Design shift (Phase 2 UI overhaul): hover + pin + single-active was
 * retired because it coupled two concerns (intent-to-verify and
 * intent-to-read-excerpt) to cursor micro-motion. The new model is a
 * per-section toggle: clicking a marker (or pressing Enter on a
 * focused segment) flips that one ``section_path`` in the
 * ``expanded`` Set, fetching + rendering the excerpt in-flow beneath
 * the citing line. Multiple citations can be open simultaneously so
 * users can compare sources side-by-side while scrolling.
 *
 * Bulk operations exposed by the context — ``collapseAll``,
 * ``expandAll``, ``expandWeakOnly`` — live here (not in
 * DetailedSummarySection) because they need access to the chunkId
 * cache and fetchToken plumbing, which the context already owns. The
 * DetailedSummarySection header calls them directly.
 *
 * Excerpts are fetched lazily on first expansion and cached by
 * chunk_id so re-opens reuse the previous fetch.
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

// Confidence tier threshold. Calibrated against citation eval baseline
// (ruri-v3-30m, N=69, 2026-04-19): top_score ≥ 0.90 hits location
// offset 0 at 86% vs ~68% for [0.80, 0.90). See
// docs/CITATION-PIPELINE.md Stage 5 and hako Uxs06_pOPfbkGtvwIK_Vq.
export const CITATION_STRONG_THRESHOLD = 0.9;

export type CitationFetchState =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "ready"; excerpt: CitationChunkExcerpt | null }
  | { kind: "error" };

interface CitationRailContextValue {
  /**
   * Verify mode — global ON/OFF for this provider. When OFF the
   * citation markers (dots) hide via ``visibility: hidden`` (they
   * keep their layout slot so copy doesn't reflow) and expanded
   * panels are collapsed. Persisted to localStorage via
   * ``hv.intelligence.verify`` so it survives navigation.
   */
  verify: boolean;
  /** Toggle verify mode. Turns off collapses all open panels. */
  setVerify: (next: boolean) => void;
  /** The full set of section_paths currently expanded. */
  expanded: ReadonlySet<string>;
  /** Convenience predicate for components that only care about one section. */
  isExpanded: (sectionPath: string) => boolean;
  /**
   * Flip one citation. Opening lazily fetches the excerpt (cached by
   * chunk_id) and returns immediately; closing is synchronous.
   * A no-op for citations with `has_citation=false` since the UI
   * hides markers for those entirely.
   */
  toggle: (citation: DetailedSummaryCitation) => void;
  /** Explicitly collapse one citation. Safe to call when already closed. */
  close: (sectionPath: string) => void;
  /** Collapse every expanded citation. */
  collapseAll: () => void;
  /**
   * Expand every citation in the provided list that has a citation
   * (has_citation=true). Idempotent. Used by the Verify header's
   * "All expanded" action.
   */
  expandAll: (citations: readonly DetailedSummaryCitation[]) => void;
  /**
   * Expand only the weak-tier citations (top_score < 0.90 && has_citation).
   * Used by the Verify header's "Needs check" badge.
   */
  expandWeakOnly: (citations: readonly DetailedSummaryCitation[]) => void;
  /** Fetch state for one expanded section (idle when not expanded). */
  excerptState: (sectionPath: string) => CitationFetchState;
}

const CitationRailContext = createContext<CitationRailContextValue | null>(
  null,
);

interface CitationRailProviderProps {
  fileId: string;
  drive: string;
  children: ReactNode;
}

interface ExpandedEntry {
  citation: DetailedSummaryCitation;
  chunkId: string;
}

const VERIFY_STORAGE_KEY = "hv.intelligence.verify";

export function CitationRailProvider({
  fileId,
  drive,
  children,
}: CitationRailProviderProps) {
  // Verify mode. Initialised OFF on first render to avoid SSR
  // hydration mismatch; a useEffect below reads localStorage on mount
  // and applies the saved value.
  const [verify, setVerifyState] = useState<boolean>(false);
  useEffect(() => {
    try {
      const saved = window.localStorage.getItem(VERIFY_STORAGE_KEY);
      if (saved === "true") setVerifyState(true);
    } catch {
      // localStorage unavailable (privacy mode, SSR, etc.) — stay OFF.
    }
  }, []);

  // section_path → ExpandedEntry (citation + chosen chunkId). Using
  // Map rather than Set lets us recover the citation later without an
  // auxiliary lookup — handy for bulk operations.
  const [entries, setEntries] = useState<
    ReadonlyMap<string, ExpandedEntry>
  >(() => new Map());
  // section_path → fetch state. Strictly tracks excerpts for expanded
  // sections; collapsing drops the entry.
  const [states, setStates] = useState<
    ReadonlyMap<string, CitationFetchState>
  >(() => new Map());
  // chunkId → cached excerpt. Shared across re-opens; survives toggles.
  const cacheRef = useRef<Map<string, CitationChunkExcerpt | null>>(new Map());
  // Per-section token to discard stale fetches when a user rapidly
  // toggles the same section mid-request.
  const fetchTokensRef = useRef<Map<string, number>>(new Map());

  // Stable identity for the exposed `expanded` view.
  const expandedSet = useMemo(() => {
    const set = new Set<string>();
    for (const k of entries.keys()) set.add(k);
    return set;
  }, [entries]);

  const isExpanded = useCallback(
    (sectionPath: string) => expandedSet.has(sectionPath),
    [expandedSet],
  );

  const excerptState = useCallback(
    (sectionPath: string): CitationFetchState => {
      return states.get(sectionPath) ?? { kind: "idle" };
    },
    [states],
  );

  const startFetch = useCallback(
    (entry: ExpandedEntry) => {
      const sectionPath = entry.citation.section_path;
      const chunkId = entry.chunkId;
      const token = (fetchTokensRef.current.get(sectionPath) ?? 0) + 1;
      fetchTokensRef.current.set(sectionPath, token);

      const cached = cacheRef.current.get(chunkId);
      if (cached !== undefined) {
        setStates((prev) => {
          const next = new Map(prev);
          next.set(sectionPath, { kind: "ready", excerpt: cached });
          return next;
        });
        return;
      }

      setStates((prev) => {
        const next = new Map(prev);
        next.set(sectionPath, { kind: "loading" });
        return next;
      });

      void (async () => {
        try {
          const excerpt = await getCitationChunkExcerpt(fileId, chunkId, drive);
          cacheRef.current.set(chunkId, excerpt);
          if (fetchTokensRef.current.get(sectionPath) !== token) return;
          setStates((prev) => {
            const next = new Map(prev);
            next.set(sectionPath, { kind: "ready", excerpt });
            return next;
          });
        } catch {
          if (fetchTokensRef.current.get(sectionPath) !== token) return;
          setStates((prev) => {
            const next = new Map(prev);
            next.set(sectionPath, { kind: "error" });
            return next;
          });
        }
      })();
    },
    [fileId, drive],
  );

  const close = useCallback((sectionPath: string) => {
    // Bump the fetch token so any in-flight excerpt request for this
    // section is ignored on completion.
    fetchTokensRef.current.set(
      sectionPath,
      (fetchTokensRef.current.get(sectionPath) ?? 0) + 1,
    );
    setEntries((prev) => {
      if (!prev.has(sectionPath)) return prev;
      const next = new Map(prev);
      next.delete(sectionPath);
      return next;
    });
    setStates((prev) => {
      if (!prev.has(sectionPath)) return prev;
      const next = new Map(prev);
      next.delete(sectionPath);
      return next;
    });
  }, []);

  const toggle = useCallback(
    (citation: DetailedSummaryCitation) => {
      const sectionPath = citation.section_path;
      const chunkId = citation.chunk_ids[0] ?? "";
      // Missing-citation segments have no marker in the UI, but guard
      // defensively in case a caller wires this to an alternative
      // trigger: flipping a has_citation=false is a no-op.
      if (!citation.has_citation || !chunkId) return;

      if (entries.has(sectionPath)) {
        close(sectionPath);
        return;
      }
      const entry: ExpandedEntry = { citation, chunkId };
      setEntries((prev) => {
        const next = new Map(prev);
        next.set(sectionPath, entry);
        return next;
      });
      startFetch(entry);
    },
    [entries, close, startFetch],
  );

  const setVerify = useCallback((next: boolean) => {
    setVerifyState(next);
    try {
      window.localStorage.setItem(VERIFY_STORAGE_KEY, next ? "true" : "false");
    } catch {
      // Ignore — ephemeral session is acceptable.
    }
    // Turning Verify OFF must drop every expanded panel so the UI
    // returns to the quiet "just text" reading state.
    if (!next) {
      setEntries((prev) => (prev.size === 0 ? prev : new Map()));
      setStates((prev) => (prev.size === 0 ? prev : new Map()));
      for (const [k] of fetchTokensRef.current) {
        fetchTokensRef.current.set(
          k,
          (fetchTokensRef.current.get(k) ?? 0) + 1,
        );
      }
    }
  }, []);

  const collapseAll = useCallback(() => {
    setEntries((prev) => (prev.size === 0 ? prev : new Map()));
    setStates((prev) => (prev.size === 0 ? prev : new Map()));
    // Bump every outstanding token so in-flight fetches are discarded.
    for (const [k] of fetchTokensRef.current) {
      fetchTokensRef.current.set(
        k,
        (fetchTokensRef.current.get(k) ?? 0) + 1,
      );
    }
  }, []);

  const expandAll = useCallback(
    (citations: readonly DetailedSummaryCitation[]) => {
      const openable = citations.filter(
        (c) => c.has_citation && c.chunk_ids.length > 0,
      );
      if (openable.length === 0) return;
      const additions: ExpandedEntry[] = [];
      setEntries((prev) => {
        const next = new Map(prev);
        for (const c of openable) {
          if (next.has(c.section_path)) continue;
          const entry: ExpandedEntry = {
            citation: c,
            chunkId: c.chunk_ids[0],
          };
          next.set(c.section_path, entry);
          additions.push(entry);
        }
        return next.size === prev.size ? prev : next;
      });
      for (const entry of additions) startFetch(entry);
    },
    [startFetch],
  );

  const expandWeakOnly = useCallback(
    (citations: readonly DetailedSummaryCitation[]) => {
      const weak = citations.filter(
        (c) =>
          c.has_citation
          && c.chunk_ids.length > 0
          && c.top_score < CITATION_STRONG_THRESHOLD,
      );
      if (weak.length === 0) return;
      const additions: ExpandedEntry[] = [];
      setEntries((prev) => {
        const next = new Map(prev);
        for (const c of weak) {
          if (next.has(c.section_path)) continue;
          const entry: ExpandedEntry = {
            citation: c,
            chunkId: c.chunk_ids[0],
          };
          next.set(c.section_path, entry);
          additions.push(entry);
        }
        return next.size === prev.size ? prev : next;
      });
      for (const entry of additions) startFetch(entry);
    },
    [startFetch],
  );

  // Reset cache + open state when the provider remounts for a new file.
  useEffect(() => {
    return () => {
      cacheRef.current.clear();
      fetchTokensRef.current.clear();
    };
  }, [fileId, drive]);

  const value = useMemo<CitationRailContextValue>(
    () => ({
      verify,
      setVerify,
      expanded: expandedSet,
      isExpanded,
      toggle,
      close,
      collapseAll,
      expandAll,
      expandWeakOnly,
      excerptState,
    }),
    [
      verify,
      setVerify,
      expandedSet,
      isExpanded,
      toggle,
      close,
      collapseAll,
      expandAll,
      expandWeakOnly,
      excerptState,
    ],
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
