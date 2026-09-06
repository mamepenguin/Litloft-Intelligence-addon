"use client";

/**
 * Intelligence → Find page.
 *
 * Spec: ``docs/superpowers/specs/2026-04-30-intelligence-find-mode.md``
 *
 * Single-shot file-list output mode. Posts to
 * ``POST /api/addons/intelligence/find`` and renders the structured
 * decomposition (chips) + retrieved file cards (thumbnail, name,
 * file_type, score, hit-text snippet, viewed_at).
 *
 * Chip × clicks rebuild ``overrides`` from the current decomposed
 * snapshot with the cleared slot reset to ``"none"`` (or ``""`` for
 * ``semantic_query``) and re-POST. The backend returns a fresh
 * decomposition each call so chips stay in sync.
 *
 * Stateless: nothing is persisted, nothing survives a reload. The
 * shareable surface is the question itself in the URL (``?q=``).
 */

import {
  type ChangeEvent,
  type FormEvent,
  Suspense,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { useTranslations } from "next-intl";
import { useSearchParams } from "next/navigation";
import { AlertCircle, ListFilter, Search } from "lucide-react";

import { useCurrentDrive } from "@/components/CurrentDriveProvider";
import { findFiles } from "../api";
import type {
  FindDecomposed,
  FindOverrides,
  FindResponse,
  FindResultEntry,
} from "../api";
import FindChip from "../FindChip";
import type { FindChipSlot } from "../FindChip";
import { Button } from "@/components/Button";
import { PageHeader } from "@/components/PageHeader";
import { DriveScopeLine } from "../DriveScopeLine";
import ModeTabs from "../ModeTabs";

const FIND_LIMIT = 20;

type FindPageState =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "loaded"; response: FindResponse }
  | { kind: "error"; message: string };

/**
 * Build the chip list from a decomposed snapshot. Returns null entries
 * for slots that should not render a chip (``"none"`` / empty
 * ``semantic_query``).
 */
interface ChipDescriptor {
  slot: FindChipSlot;
  label: string;
}

function buildChips(
  decomposed: FindDecomposed,
  t: (key: string, values?: Record<string, string | number>) => string,
): ChipDescriptor[] {
  const chips: ChipDescriptor[] = [];
  const tr = decomposed.time_range;
  if (tr && tr.value && tr.value !== "none") {
    // Prefer a half-resolved range label when both ends are known so
    // the user sees the concrete dates the LLM resolved to.
    const after = tr.after ?? null;
    const before = tr.before ?? null;
    let label = tr.value;
    if (after && before) {
      label = `${tr.value} (${formatDateLabel(after)}-${formatDateLabel(before)})`;
    }
    chips.push({ slot: "time_range", label });
  }
  if (decomposed.personal_scope && decomposed.personal_scope !== "none") {
    chips.push({
      slot: "personal_scope",
      label:
        decomposed.personal_scope === "viewed"
          ? t("chipPersonalViewed")
          : decomposed.personal_scope === "not_viewed"
            ? t("chipPersonalNotViewed")
            : decomposed.personal_scope,
    });
  }
  if (decomposed.file_type_hint && decomposed.file_type_hint !== "none") {
    chips.push({ slot: "file_type_hint", label: decomposed.file_type_hint });
  }
  if (decomposed.semantic_query && decomposed.semantic_query.length > 0) {
    chips.push({ slot: "semantic_query", label: decomposed.semantic_query });
  }
  return chips;
}

/**
 * Format an ISO timestamp as a compact ``M/d`` label for chip display.
 * Falls back to the raw string when parsing fails.
 */
function formatDateLabel(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return `${d.getUTCMonth() + 1}/${d.getUTCDate()}`;
}

/**
 * Build the override snapshot for a chip × click. The cleared slot
 * resets to ``"none"`` (or ``""`` for ``semantic_query``) and the
 * other slots carry their current resolved values forward.
 */
function buildOverrides(
  decomposed: FindDecomposed,
  cleared: FindChipSlot,
): FindOverrides {
  const overrides: FindOverrides = {
    time_range: decomposed.time_range?.value ?? "none",
    personal_scope: decomposed.personal_scope || "none",
    file_type_hint: decomposed.file_type_hint || "none",
    semantic_query: decomposed.semantic_query ?? "",
  };
  if (cleared === "semantic_query") {
    overrides.semantic_query = "";
  } else {
    overrides[cleared] = "none";
  }
  return overrides;
}

function ResultCard({ entry }: { entry: FindResultEntry }) {
  const { file, hit, score, file_id } = entry;
  const viewedAt = file.viewed_at ? new Date(file.viewed_at) : null;
  return (
    <a
      href={`/files/${file_id}`}
      className="flex items-start gap-3 rounded-xl border border-bg-border bg-bg-card p-3 transition-colors hover:bg-bg-elevated"
    >
      <img
        src={file.thumbnail_url}
        alt={file.name}
        loading="lazy"
        className="h-16 w-24 flex-shrink-0 rounded-lg bg-bg-elevated object-cover"
      />
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm font-medium text-text-primary">
          {file.name}
        </p>
        <div className="mt-0.5 flex flex-wrap items-center gap-2 text-xs text-text-muted">
          <span className="rounded-lg bg-bg-elevated px-1.5 py-0.5">
            {file.file_type}
          </span>
          {viewedAt && (
            <span>
              {viewedAt.getUTCMonth() + 1}/{viewedAt.getUTCDate()}
            </span>
          )}
          <span>{score.toFixed(2)}</span>
        </div>
        {hit?.text && (
          <p className="mt-1 line-clamp-2 text-xs italic text-text-muted">
            {hit.text}
          </p>
        )}
      </div>
    </a>
  );
}

function IntelligenceFindPageInner() {
  const t = useTranslations("find");
  const searchParams = useSearchParams();
  const seedQuery = searchParams?.get("q") ?? "";
  const drive = useCurrentDrive();

  const [input, setInput] = useState(seedQuery);
  const [state, setState] = useState<FindPageState>({ kind: "idle" });
  const [composing, setComposing] = useState(false);
  const autoFiredRef = useRef(false);
  // Track the question the current decomposition came from so chip ×
  // re-POSTs can reuse it without depending on the input field (the
  // user may have started editing a follow-up query).
  const lastQuestionRef = useRef<string>("");

  const runFind = useCallback(
    async (rawQuery: string, overrides?: FindOverrides) => {
      const trimmed = rawQuery.trim();
      if (trimmed.length === 0) return;
      if (!drive) {
        setState({ kind: "error", message: t("error") });
        return;
      }

      lastQuestionRef.current = trimmed;
      setState({ kind: "loading" });
      try {
        const response = await findFiles(trimmed, drive, {
          limit: FIND_LIMIT,
          ...(overrides ? { overrides } : {}),
        });
        setState({ kind: "loaded", response });
      } catch (err) {
        const message =
          err instanceof Error && err.message ? err.message : t("error");
        setState({ kind: "error", message });
      }
    },
    [drive, t],
  );

  // Seed-query auto-fire on mount. Guarded so a re-render does not
  // re-trigger the request.
  useEffect(() => {
    if (autoFiredRef.current) return;
    if (!seedQuery.trim()) return;
    autoFiredRef.current = true;
    void runFind(seedQuery);
  }, [seedQuery, runFind]);

  const handleSubmit = useCallback(
    (e: FormEvent<HTMLFormElement>) => {
      e.preventDefault();
      // Skip Enter while IME composition is active (e.g. Japanese
      // conversion), otherwise the conversion-confirming Enter would
      // submit the form.
      if (composing) return;
      void runFind(input);
    },
    [composing, input, runFind],
  );

  const handleInputChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      setInput(e.target.value);
    },
    [],
  );

  const handleChipRemove = useCallback(
    (slot: FindChipSlot, decomposed: FindDecomposed) => {
      const overrides = buildOverrides(decomposed, slot);
      void runFind(lastQuestionRef.current, overrides);
    },
    [runFind],
  );

  const decomposed =
    state.kind === "loaded" ? state.response.decomposed : null;
  const chips = decomposed ? buildChips(decomposed, t) : [];

  return (
    <div className="mx-auto flex w-full max-w-3xl flex-col gap-4 py-4 sm:py-6">
      <PageHeader
        titleIcon={ListFilter}
        title={t("title")}
        tabs={
          drive ? <ModeTabs current="find" query={input} drive={drive} /> : undefined
        }
      />

      {/* `px-4`, matching PageHeader's own padding — see the note on the Ask
          page. */}
      <div className="flex flex-col gap-4 px-4">

        <DriveScopeLine drive={drive} />

        <form onSubmit={handleSubmit} className="flex items-center gap-2">
          <input
            type="text"
            value={input}
            onChange={handleInputChange}
            onCompositionStart={() => setComposing(true)}
            onCompositionEnd={() => setComposing(false)}
            placeholder={t("placeholder")}
            className="flex-1 rounded-2xl border border-bg-border bg-bg-card px-3 py-2 text-sm text-text-primary placeholder:text-text-muted focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-focus-ring"
          />
          <Button
            type="submit"
            variant="primary"
            disabled={state.kind === "loading" || input.trim().length === 0}
          >
            <Search size={12} /> {t("submit")}
          </Button>
        </form>

        {decomposed && chips.length > 0 && (
          <div
            data-testid="find-chips"
            className="flex flex-wrap items-center gap-2"
          >
            {chips.map((chip) => (
              <FindChip
                key={chip.slot}
                slot={chip.slot}
                label={chip.label}
                onRemove={() => handleChipRemove(chip.slot, decomposed)}
              />
            ))}
            {decomposed.category_expansion.length > 0 && (
              <span className="text-xs text-text-muted">
                {t("categoryExpansion", {
                  terms: decomposed.category_expansion.join(" / "),
                })}
              </span>
            )}
          </div>
        )}

        {state.kind === "loading" && (
          <div
            data-testid="find-loading"
            className="flex items-center justify-center py-8"
            role="status"
            aria-live="polite"
          >
            <div className="h-5 w-5 animate-spin rounded-full border-2 border-accent border-t-transparent" />
            <span className="ml-2 text-sm text-text-muted">{t("loading")}</span>
          </div>
        )}

        {state.kind === "error" && (
          <div
            data-testid="find-error"
            role="alert"
            className="flex items-start gap-2 rounded-xl border border-bg-border bg-bg-card p-3"
          >
            <AlertCircle
              size={16}
              className="mt-0.5 flex-shrink-0 text-danger"
            />
            <p className="text-sm text-text-primary">{state.message}</p>
          </div>
        )}

        {state.kind === "loaded" && (
          <>
            <p className="text-sm text-text-muted">
              {t("totalCount", { count: state.response.total })}
            </p>
            {state.response.results.length === 0 ? (
              <div
                data-testid="find-empty"
                className="rounded-xl border border-bg-border bg-bg-card p-6 text-center text-sm text-text-muted"
              >
                {t("empty")}
              </div>
            ) : (
              <ul className="flex flex-col gap-2">
                {state.response.results.map((entry) => (
                  <li key={entry.file_id}>
                    <ResultCard entry={entry} />
                  </li>
                ))}
              </ul>
            )}
          </>
        )}
      </div>
    </div>
  );
}

/**
 * Suspense wrapper. ``useSearchParams`` in the App Router opts the
 * component into client-side rendering and requires a Suspense
 * boundary up the tree during prerender. The addon page wrapper
 * auto-generated by the host re-exports this default, so the
 * Suspense boundary lives here.
 */
export default function IntelligenceFindPage() {
  return (
    <Suspense
      fallback={
        <div className="mx-auto flex w-full max-w-3xl flex-col gap-4 p-4 sm:p-6">
          <div className="h-6 w-32 animate-pulse rounded-lg bg-bg-elevated" />
        </div>
      }
    >
      <IntelligenceFindPageInner />
    </Suspense>
  );
}
