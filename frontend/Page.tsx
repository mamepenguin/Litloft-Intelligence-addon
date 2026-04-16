"use client";

/**
 * Intelligence → Ask page.
 *
 * Dedicated chat-style UI for natural-language question answering
 * over the indexed corpus. Consumes the SSE `POST /ask` stream via
 * `askQuestionStream` and renders:
 *
 *  1. The transformed search keywords (so the user sees what was
 *     actually looked up when their question got noisy-word-stripped).
 *  2. The list of retrieved source files (shown as soon as they come
 *     in, before the answer finishes generating).
 *  3. The answer text, with inline `[N]` citation chips that link to
 *     the referenced file / timestamp.
 *  4. The final citations sidebar with quotes.
 *
 * Seed query: when the URL contains `?q=<query>`, the page auto-fires
 * the Ask pipeline on mount. The input stays editable so the user can
 * refine the question without leaving the page.
 *
 * State is intentionally ephemeral (Phase 1 of the RAG redesign):
 * nothing is persisted, nothing survives a reload. The only shareable
 * state is the question itself in the URL.
 */

import {
  type ChangeEvent,
  type FormEvent,
  type KeyboardEvent as ReactKeyboardEvent,
  type ReactNode,
  Suspense,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { flushSync } from "react-dom";
import { useTranslations } from "next-intl";
import { useSearchParams } from "next/navigation";
import { AlertCircle, Send, Sparkles, Square, X } from "lucide-react";

import { useCurrentDrive } from "@/components/CurrentDriveProvider";
import {
  askQuestionStream,
  getIntelligenceStatus,
  type AskStreamEvent,
  type Citation,
  type IntelligenceStatus,
  type Source,
} from "./api";

// Minimum allowed query length after trimming. Matches the backend
// gate so we never send a request the server will reject.
const MIN_QUERY_LENGTH = 3;

// Non-terminal states describe the live request; terminal states
// describe what the user should see once the stream has ended.
type AskState =
  | { kind: "idle" }
  | {
      kind: "streaming";
      keywords: string | null;
      sources: Source[];
      answerBuffer: string;
      // Progressive citations — populated by per-citation events as
      // they arrive. The terminal `citations` (plural) event replaces
      // this list with the server's canonical final ordering.
      citations: Citation[];
    }
  | {
      kind: "answered";
      keywords: string | null;
      sources: Source[];
      answer: string;
      citations: Citation[];
      tookMs: number | null;
    }
  | {
      kind: "error";
      message: string;
      retryable: boolean;
    };

/**
 * Parse backend `segment_location` into a display label + optional
 * seek-second. Format is one of `"m:ss"` (video/audio) or `"page 3"`.
 */
function parseSegmentLocation(
  loc: string | null,
): { label: string; seconds: number | null } | null {
  if (!loc) return null;
  const timeMatch = loc.match(/^(\d+):(\d{2})$/);
  if (timeMatch) {
    const m = parseInt(timeMatch[1], 10);
    const s = parseInt(timeMatch[2], 10);
    if (Number.isFinite(m) && Number.isFinite(s)) {
      return { label: loc, seconds: m * 60 + s };
    }
  }
  return { label: loc, seconds: null };
}

/**
 * Build a file-detail URL from a citation, appending `?t=` when we
 * have a seekable timestamp so the detail page auto-scrubs there.
 */
function buildCitationUrl(citation: Citation): string {
  const parsed = parseSegmentLocation(
    // segment_location is optional in the streaming Citation payload
    // (the service dataclass always emits it, but TS treats it as
    // possibly null / undefined for defensive rendering).
    (citation as Citation & { segment_location?: string | null }).segment_location ?? null,
  );
  if (parsed?.seconds != null) {
    return `/files/${citation.file_id}?t=${parsed.seconds}`;
  }
  return `/files/${citation.file_id}`;
}

/**
 * Render an answer string with `[N]` markers converted to interactive
 * chip buttons. Out-of-range indices are left as plain text so a
 * mid-stream partial answer never crashes the render.
 */
function renderAnswerWithCitations(
  answer: string,
  citations: Citation[],
  onCitationClick: (index: number) => void,
): ReactNode[] {
  const parts = answer.split(/(\[\d+\])/g);
  return parts.map((part, i) => {
    const match = part.match(/^\[(\d+)\]$/);
    if (!match) return <span key={i}>{part}</span>;
    const n = parseInt(match[1], 10);
    if (n < 1 || n > citations.length) {
      return (
        <span key={i} className="text-text-muted/60">
          {part}
        </span>
      );
    }
    return (
      <button
        key={i}
        type="button"
        onClick={() => onCitationClick(n)}
        className="mx-0.5 inline-flex items-center rounded px-1 py-0 align-super text-[10px] font-semibold text-accent bg-accent/10 hover:bg-accent/20 transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent"
        aria-label={`Jump to citation ${n}`}
      >
        {n}
      </button>
    );
  });
}

function CitationCard({
  index,
  citation,
}: {
  index: number;
  citation: Citation;
}) {
  const parsed = parseSegmentLocation(
    (citation as Citation & { segment_location?: string | null }).segment_location ?? null,
  );
  return (
    <a
      id={`ask-citation-${index}`}
      href={buildCitationUrl(citation)}
      className="group flex w-full items-start gap-2 rounded-md border border-bg-border bg-bg-card px-3 py-2 text-left transition-colors hover:bg-bg-elevated focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent"
    >
      <span className="mt-0.5 inline-flex h-5 min-w-[1.25rem] items-center justify-center rounded px-1 text-[11px] font-semibold text-accent bg-accent/10">
        {index}
      </span>
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-1.5">
          <span className="truncate text-sm font-medium text-text-primary">
            {citation.filename}
          </span>
          {parsed && (
            <span className="flex-shrink-0 rounded px-1 py-0.5 text-[10px] font-medium text-accent">
              {parsed.label}
            </span>
          )}
        </div>
        {citation.quote && (
          <p className="mt-1 line-clamp-3 text-xs italic text-text-muted">
            “{citation.quote}”
          </p>
        )}
      </div>
    </a>
  );
}

/**
 * "Thinking" placeholder shown while the backend is still retrieving
 * / warming up the LLM and no `answer_chunk` has arrived yet. Uses
 * three bouncing dots (staggered `animate-pulse`) next to the i18n
 * label. Stable `data-testid="ask-thinking"` is how unit tests pin
 * the indicator without depending on the label copy.
 */
function ThinkingIndicator({ label }: { label: string }) {
  return (
    <span
      data-testid="ask-thinking"
      role="status"
      aria-live="polite"
      className="ml-0.5 inline-flex items-center gap-1 align-baseline text-xs text-text-muted"
    >
      <span className="inline-flex items-center gap-0.5">
        <span
          className="inline-block h-1 w-1 animate-pulse rounded-full bg-accent/70"
          style={{ animationDelay: "0ms" }}
        />
        <span
          className="inline-block h-1 w-1 animate-pulse rounded-full bg-accent/70"
          style={{ animationDelay: "150ms" }}
        />
        <span
          className="inline-block h-1 w-1 animate-pulse rounded-full bg-accent/70"
          style={{ animationDelay: "300ms" }}
        />
      </span>
      <span className="ml-1">{label}</span>
    </span>
  );
}

function SourceCard({ source }: { source: Source }) {
  return (
    <a
      href={`/files/${source.file_id}`}
      className="flex min-w-0 items-center gap-2 rounded-md border border-bg-border bg-bg-card px-2 py-1.5 text-xs text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary"
    >
      <span className="truncate">{source.filename}</span>
    </a>
  );
}

function IntelligenceAskPageInner() {
  const t = useTranslations("askSearch");
  const searchParams = useSearchParams();
  const seedQuery = searchParams?.get("q") ?? "";
  // The page lives at /drive/{drive}/addons/intelligence so this is
  // always populated when the wrapper renders us. Guard anyway in case
  // of future scope changes.
  const drive = useCurrentDrive();

  const [input, setInput] = useState(seedQuery);
  const [state, setState] = useState<AskState>({ kind: "idle" });
  const [ragAvailable, setRagAvailable] = useState<boolean | null>(null);
  const [composing, setComposing] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  // Guard so the seed-query auto-fire runs exactly once even when the
  // status check re-renders the component. Without this an upstream
  // router update could accidentally re-trigger the request.
  const autoFiredRef = useRef(false);

  // --- Status probe: gate the Ask button when the backend has RAG off
  //     or the LLM isn't configured. Runs once on mount. ---
  useEffect(() => {
    if (!drive) {
      setRagAvailable(false);
      return;
    }
    const controller = new AbortController();
    let cancelled = false;
    getIntelligenceStatus(drive, controller.signal).then(
      (status: IntelligenceStatus | null) => {
        if (cancelled) return;
        const enabled =
          status?.features?.rag === true && status?.llm?.enabled === true;
        setRagAvailable(enabled);
      },
    );
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [drive]);

  // --- Cleanup: abort any inflight stream on unmount. ---
  useEffect(() => {
    return () => {
      abortRef.current?.abort();
      abortRef.current = null;
    };
  }, []);

  const runAsk = useCallback(
    async (rawQuery: string) => {
      const trimmed = rawQuery.trim();
      if (trimmed.length < MIN_QUERY_LENGTH) {
        setState({
          kind: "error",
          message: t("queryTooShort"),
          retryable: false,
        });
        return;
      }

      // Cancel any previous stream first so we never have two open
      // fetches racing into the same state setter.
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      setState({
        kind: "streaming",
        keywords: null,
        sources: [],
        answerBuffer: "",
        citations: [],
      });

      if (!drive) {
        setState({
          kind: "error",
          message: t("queryTooShort"),
          retryable: false,
        });
        return;
      }

      try {
        const stream = askQuestionStream(trimmed, drive, {
          signal: controller.signal,
        });

        // Running state accumulators. We keep them in locals so
        // back-to-back yields don't fight React's state batching — the
        // setState call at the bottom of each loop iteration uses the
        // functional form to merge into whatever the latest render saw.
        let liveKeywords: string | null = null;
        let liveSources: Source[] = [];
        let liveAnswer = "";
        // Progressive citations accumulated from per-event updates.
        // Replaced wholesale by the terminal `citations` (plural) event
        // so the server's final ordering wins.
        let liveCitations: Citation[] = [];
        let finalCitations: Citation[] = [];
        let finalTookMs: number | null = null;
        let finalError: string | null = null;

        for await (const event of stream as AsyncIterable<AskStreamEvent>) {
          if (controller.signal.aborted) break;

          switch (event.kind) {
            case "keywords":
              liveKeywords = event.keywords;
              setState({
                kind: "streaming",
                keywords: liveKeywords,
                sources: liveSources,
                answerBuffer: liveAnswer,
                citations: liveCitations,
              });
              break;
            case "sources":
              liveSources = event.sources;
              setState({
                kind: "streaming",
                keywords: liveKeywords,
                sources: liveSources,
                answerBuffer: liveAnswer,
                citations: liveCitations,
              });
              break;
            case "answer_chunk":
              liveAnswer = liveAnswer + event.delta;
              // flushSync is required because React 18+ automatically
              // batches setState calls that happen across await points
              // in the same async task. A single browser reader.read()
              // typically delivers many SSE frames in one TCP packet,
              // so the `for await` loop below dispatches dozens of
              // answer_chunk updates back-to-back — auto-batching
              // would coalesce them into a single render at the end,
              // making the UI look like "all at once" even though the
              // network actually streamed the tokens progressively.
              // Forcing a synchronous flush per chunk restores the
              // character-by-character typewriter effect. The other
              // event kinds (keywords/sources) fire at most once each,
              // so they keep the default batched behavior.
              flushSync(() => {
                setState({
                  kind: "streaming",
                  keywords: liveKeywords,
                  sources: liveSources,
                  answerBuffer: liveAnswer,
                  citations: liveCitations,
                });
              });
              break;
            case "citation":
              // Append — immutable spread to respect the repo's
              // immutability rule. flushSync keeps the citation from
              // being swallowed by a back-to-back answer_chunk batch:
              // if the server interleaves a citation between two token
              // chunks, the auto-batch would otherwise paint the chip
              // and the next chunk in one render and the user never
              // sees the progressive reveal.
              liveCitations = [...liveCitations, event.citation];
              flushSync(() => {
                setState({
                  kind: "streaming",
                  keywords: liveKeywords,
                  sources: liveSources,
                  answerBuffer: liveAnswer,
                  citations: liveCitations,
                });
              });
              break;
            case "citations":
              // Terminal list is canonical — replace whatever we built
              // from per-citation events. Also push the replacement to
              // the visible state so the DOM mirrors the final order
              // before the `answered` transition (tests assert on the
              // streaming-time replacement).
              finalCitations = event.citations;
              liveCitations = event.citations;
              setState({
                kind: "streaming",
                keywords: liveKeywords,
                sources: liveSources,
                answerBuffer: liveAnswer,
                citations: liveCitations,
              });
              break;
            case "done":
              if (event.error) finalError = event.error;
              if (typeof event.took_ms === "number")
                finalTookMs = event.took_ms;
              break;
          }
        }

        // If the stream ended without a terminal `citations` frame
        // (older backends, or a test stub), fall through with whatever
        // progressive citations we accumulated.
        if (finalCitations.length === 0 && liveCitations.length > 0) {
          finalCitations = liveCitations;
        }

        if (controller.signal.aborted) return;

        if (finalError) {
          setState({
            kind: "error",
            message: finalError,
            retryable: true,
          });
          return;
        }

        if (!liveAnswer) {
          // Retrieval may have been empty, or the LLM returned no
          // output. Show the "no answer" state rather than a blank
          // panel — the UX must never fall into a silent dead end.
          setState({
            kind: "error",
            message: t("noAnswer"),
            retryable: true,
          });
          return;
        }

        setState({
          kind: "answered",
          keywords: liveKeywords,
          sources: liveSources,
          answer: liveAnswer,
          citations: finalCitations,
          tookMs: finalTookMs,
        });
      } catch (err) {
        if (controller.signal.aborted) return;
        if (err instanceof DOMException && err.name === "AbortError") return;
        const message =
          err instanceof Error && err.message
            ? err.message
            : t("errorGenerate");
        setState({
          kind: "error",
          message,
          retryable: true,
        });
      } finally {
        if (abortRef.current === controller) {
          abortRef.current = null;
        }
      }
    },
    [t, drive],
  );

  // --- Auto-fire on mount when the URL carries a seed query. ---
  useEffect(() => {
    if (autoFiredRef.current) return;
    if (!seedQuery.trim()) return;
    if (ragAvailable !== true) return;
    autoFiredRef.current = true;
    void runAsk(seedQuery);
  }, [seedQuery, ragAvailable, runAsk]);

  const canSubmit = useMemo(() => {
    return (
      ragAvailable === true &&
      input.trim().length >= MIN_QUERY_LENGTH &&
      state.kind !== "streaming"
    );
  }, [input, ragAvailable, state.kind]);

  const handleSubmit = useCallback(
    (e: FormEvent<HTMLFormElement>) => {
      e.preventDefault();
      if (!canSubmit) return;
      void runAsk(input);
    },
    [canSubmit, input, runAsk],
  );

  const handleInputKeyDown = useCallback(
    (e: ReactKeyboardEvent<HTMLTextAreaElement>) => {
      // Skip while IME composition is active (e.g. Japanese conversion),
      // otherwise the conversion-confirming Enter would submit the form.
      if (composing) return;
      // Enter submits; Shift+Enter inserts a newline. Matches the
      // convention used by the main search input.
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        if (canSubmit) void runAsk(input);
      }
    },
    [canSubmit, composing, input, runAsk],
  );

  const handleCitationClick = useCallback((index: number) => {
    const el = document.getElementById(`ask-citation-${index}`);
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "center" });
      el.focus();
    }
  }, []);

  const handleAbort = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setState({ kind: "idle" });
  }, []);

  const handleInputChange = useCallback(
    (e: ChangeEvent<HTMLTextAreaElement>) => {
      setInput(e.target.value);
    },
    [],
  );

  return (
    <div className="mx-auto flex w-full max-w-3xl flex-col gap-4 p-4 sm:p-6">
      <header className="flex items-center gap-2">
        <Sparkles size={18} className="text-accent-teal" />
        <h1 className="text-lg font-semibold text-text-primary">
          {t("answerTitle")}
        </h1>
      </header>

      {ragAvailable === false && (
        <div
          role="alert"
          className="flex items-start gap-2 rounded-md border border-bg-border bg-bg-card p-3"
        >
          <AlertCircle
            size={16}
            className="mt-0.5 flex-shrink-0 text-text-muted"
          />
          <p className="text-sm text-text-muted">{t("llmDisabled")}</p>
        </div>
      )}

      <form onSubmit={handleSubmit} className="flex flex-col gap-2">
        <textarea
          value={input}
          onChange={handleInputChange}
          onKeyDown={handleInputKeyDown}
          onCompositionStart={() => setComposing(true)}
          onCompositionEnd={() => setComposing(false)}
          placeholder={seedQuery || ""}
          rows={3}
          disabled={ragAvailable === false}
          className="w-full resize-y rounded-md border border-bg-border bg-bg-card p-3 text-sm text-text-primary placeholder:text-text-muted focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent"
          aria-label="Question input"
        />
        <div className="flex items-center justify-between gap-2">
          <p className="text-xs text-text-muted">
            {state.kind === "streaming"
              ? t("loading")
              : t("loadingHint")}
          </p>
          {state.kind === "streaming" ? (
            <button
              type="button"
              onClick={handleAbort}
              className="inline-flex items-center gap-1.5 rounded-md border border-bg-border bg-bg-card px-3 py-1.5 text-xs font-medium text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary"
            >
              <Square size={12} /> {t("close")}
            </button>
          ) : (
            <button
              type="submit"
              disabled={!canSubmit}
              className="inline-flex items-center gap-1.5 rounded-md bg-accent px-3 py-1.5 text-xs font-medium text-white transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
            >
              <Send size={12} /> {t("poweredByLlm")}
            </button>
          )}
        </div>
      </form>

      {(state.kind === "streaming" || state.kind === "answered") &&
        state.keywords && (
          <div className="flex flex-wrap items-center gap-2 rounded-md border border-bg-border bg-bg-elevated px-3 py-2 text-xs text-text-muted">
            <span>🔎</span>
            <span className="truncate">{state.keywords}</span>
          </div>
        )}

      {state.kind === "error" && (
        <div
          role="alert"
          className="flex items-start gap-2 rounded-md border border-bg-border bg-bg-card p-3"
        >
          <AlertCircle
            size={16}
            className="mt-0.5 flex-shrink-0 text-text-muted"
          />
          <div className="min-w-0 flex-1">
            <p className="text-sm text-text-primary">{state.message}</p>
            {state.retryable && (
              <button
                type="button"
                onClick={() => {
                  void runAsk(input);
                }}
                className="mt-2 inline-flex items-center gap-1 rounded px-2 py-1 text-xs text-text-muted hover:bg-bg-elevated hover:text-text-primary"
              >
                {t("retryHint")}
              </button>
            )}
          </div>
          <button
            type="button"
            aria-label={t("close")}
            onClick={() => setState({ kind: "idle" })}
            className="flex-shrink-0 rounded p-1 text-text-muted hover:bg-bg-elevated hover:text-text-primary"
          >
            <X size={12} />
          </button>
        </div>
      )}

      {(state.kind === "streaming" || state.kind === "answered") && (
        <section
          aria-live="polite"
          className="rounded-md border border-bg-border bg-bg-card p-4"
        >
          <div className="whitespace-pre-wrap text-sm leading-relaxed text-text-primary">
            {state.kind === "streaming"
              ? renderAnswerWithCitations(
                  state.answerBuffer,
                  // Use progressively-accumulated citations so `[N]`
                  // chips become interactive the moment their citation
                  // arrives. Unseen indices stay greyed-out via the
                  // out-of-range branch in renderAnswerWithCitations.
                  state.citations,
                  handleCitationClick,
                )
              : renderAnswerWithCitations(
                  state.answer,
                  state.citations,
                  handleCitationClick,
                )}
            {state.kind === "streaming" && state.answerBuffer === "" ? (
              // "Thinking" indicator — shown while retrieval / LLM
              // warm-up is happening and no answer tokens have been
              // emitted yet. Stable `data-testid` keeps the unit test
              // decoupled from the visual / i18n choice.
              <ThinkingIndicator label={t("thinking")} />
            ) : (
              state.kind === "streaming" && (
                <span className="ml-0.5 inline-block h-3 w-1 animate-pulse bg-accent align-baseline" />
              )
            )}
          </div>
          {state.kind === "answered" && state.tookMs != null && (
            <p className="mt-2 text-[10px] text-text-muted/70">
              {t("takenMs", { ms: state.tookMs })}
            </p>
          )}
        </section>
      )}

      {((state.kind === "streaming" && state.citations.length > 0) ||
        (state.kind === "answered" && state.citations.length > 0)) && (
        <section>
          <h2 className="mb-2 text-sm font-semibold text-text-primary">
            {t("citationsTitle")}
          </h2>
          <ul className="flex flex-col gap-2">
            {state.citations.map((citation, i) => (
              <li key={`${citation.file_id}-${i}`}>
                <CitationCard index={i + 1} citation={citation} />
              </li>
            ))}
          </ul>
        </section>
      )}

      {(state.kind === "streaming" || state.kind === "answered") &&
        state.sources.length > 0 && (
          <section>
            <h2 className="mb-2 text-xs font-semibold text-text-muted">
              Sources
            </h2>
            <ul className="flex flex-wrap gap-2">
              {state.sources.map((source) => (
                <li key={source.file_id} className="min-w-0">
                  <SourceCard source={source} />
                </li>
              ))}
            </ul>
          </section>
        )}
    </div>
  );
}

/**
 * Suspense wrapper around the inner component.
 *
 * ``useSearchParams`` in the App Router opts the component into
 * client-side rendering *and* requires a Suspense boundary up the
 * tree during prerender, or Next.js fails the build with
 * "useSearchParams() should be wrapped in a suspense boundary". The
 * addon page wrapper auto-generated by ``frontend/Dockerfile`` just
 * re-exports this default, so the Suspense boundary has to live here.
 */
export default function IntelligenceAskPage() {
  return (
    <Suspense
      fallback={
        <div className="mx-auto flex w-full max-w-3xl flex-col gap-4 p-4 sm:p-6">
          <div className="h-6 w-32 animate-pulse rounded bg-bg-elevated" />
        </div>
      }
    >
      <IntelligenceAskPageInner />
    </Suspense>
  );
}
