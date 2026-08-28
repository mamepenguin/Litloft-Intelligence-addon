"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { ShieldCheck, X } from "lucide-react";
import { useTranslations } from "next-intl";

import { semanticSearch } from "./api";

interface UnverifiedSourceSectionProps {
  fileId: string;
  drive: string;
  trustTier?: "verified" | "unverified";
  trustReviewedAt?: string | null;
  onFileChange?: (file: unknown) => void;
}

/** One paragraph of the clip, paired with a note it echoes. */
export interface SourcePointer {
  /** The clip's own wording, verbatim. Never a summary. */
  excerpt: string;
  noteFileId: string;
  noteFilename: string;
}

/** How many of the clip's leading paragraphs are offered as evidence. */
const MAX_POINTERS = 3;

/** Paragraphs shorter than this are headings, bylines, and nav cruft. */
const MIN_PARAGRAPH_CHARS = 80;

/**
 * Paragraphs are cut to this length *before* being used, so the string shown
 * and the string searched are the same one. A prefix on screen and the full
 * paragraph on the wire would let a match driven by the hidden tail look
 * unrelated to the evidence, and would put arbitrarily long text in a GET URL.
 */
const EXCERPT_CHARS = 240;

export function splitParagraphs(markdown: string): string[] {
  return markdown
    .replace(/^---\n[\s\S]*?\n---\n/, "")  // frontmatter is not prose
    .split(/\n\s*\n/)
    .map((p) => p.replace(/\s+/g, " ").trim())
    .filter((p) => p.length >= MIN_PARAGRAPH_CHARS && !p.startsWith("#"))
    .map((p) => p.slice(0, EXCERPT_CHARS));
}

/**
 * Asks the viewer to rule on a source they are reading.
 *
 * Shows **pointers, never generated prose**. Each row pairs one of the clip's
 * own paragraphs — reproduced verbatim, because it is the exact string that
 * was used as the query — with a note of the viewer's that it echoes. No LLM
 * is called and nothing is summarised: approving generated text would put it
 * in the verified tier, whose whole definition is content the viewer wrote or
 * vouched for.
 *
 * To distil one paragraph rather than trust the whole page, select it in the
 * document and use Knowledge's quotation basket. That affordance is owned by
 * Knowledge and is deliberately not duplicated here — Intelligence must not
 * depend on another addon.
 *
 * Spec `2026-08-29-web-clip-promotion.md` §7.
 */
export default function UnverifiedSourceSection({
  fileId,
  drive,
  trustTier,
  trustReviewedAt,
  onFileChange,
}: UnverifiedSourceSectionProps) {
  const t = useTranslations("intelligence.unverifiedSource");
  const [pointers, setPointers] = useState<SourcePointer[]>([]);
  const [pending, setPending] = useState(false);
  const [done, setDone] = useState(false);

  // Only unverified files are asked about, and only while nobody has ruled:
  // once dismissed, re-asking on every open would be nagging.
  const applies = trustTier === "unverified" && !trustReviewedAt;

  useEffect(() => {
    setDone(false);
  }, [fileId]);

  useEffect(() => {
    if (!applies) return;
    let cancelled = false;

    (async () => {
      try {
        const res = await fetch(`/api/files/${fileId}/stream`);
        if (!res.ok) return;
        const paragraphs = splitParagraphs(await res.text()).slice(0, MAX_POINTERS);

        const found: SourcePointer[] = [];
        for (const excerpt of paragraphs) {
          const hits = await semanticSearch(excerpt, drive, { limit: 5 });
          // Ordinary search deliberately returns unverified files, and a
          // clip is a .md like any note — so the extension proves nothing.
          // Only a file the viewer has actually vouched for can stand as
          // "a note of yours"; anything else would be external content
          // wearing that label. Unhydrated hits are skipped rather than
          // guessed at.
          const note = hits.results.find(
            (r) =>
              r.file_id !== fileId &&
              r.file?.trust_tier === "verified" &&
              r.filename.endsWith(".md"),
          );
          if (note) {
            found.push({
              excerpt,
              noteFileId: note.file_id,
              noteFilename: note.filename,
            });
          }
        }
        if (!cancelled) setPointers(found);
      } catch {
        // Pointers are a convenience; the decision stands without them.
        if (!cancelled) setPointers([]);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [applies, fileId, drive]);

  const rule = useCallback(
    async (tier: "verified" | "unverified") => {
      if (pending) return;
      setPending(true);
      try {
        const res = await fetch(`/api/files/${fileId}/trust-tier`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ tier }),
        });
        if (!res.ok) return;
        const updated = await res.json();
        onFileChange?.(updated);
        setDone(true);
      } finally {
        setPending(false);
      }
    },
    [fileId, onFileChange, pending],
  );

  if (!applies || done) return null;

  return (
    <section className="rounded-2xl border border-bg-border bg-bg-card p-4">
      <h3 className="text-sm font-medium text-text-primary">{t("title")}</h3>
      <p className="mt-1 text-sm text-text-muted">{t("explanation")}</p>

      {pointers.length > 0 && (
        <div className="mt-3 space-y-3">
          <p className="text-xs text-text-muted">{t("relatedHeading")}</p>
          {pointers.map((pointer) => (
            <div key={pointer.noteFileId} className="text-sm">
              {/* Selectable on purpose: selecting it is how the excerpt
                  reaches Knowledge's quotation basket. */}
              <blockquote
                data-testid="source-excerpt"
                className="border-l-2 border-bg-border pl-3 text-text-primary"
              >
                {pointer.excerpt}
              </blockquote>
              <Link
                href={`/files/${pointer.noteFileId}`}
                className="mt-1 inline-block text-text-muted underline-offset-2 hover:underline"
              >
                {pointer.noteFilename}
              </Link>
            </div>
          ))}
          <p className="text-xs text-text-muted">{t("distilHint")}</p>
        </div>
      )}

      <div className="mt-4 flex flex-wrap gap-2">
        <button
          onClick={() => rule("verified")}
          disabled={pending}
          className="inline-flex items-center gap-1.5 rounded-2xl bg-accent px-3 py-2 text-sm text-white transition-colors hover:bg-accent-hover disabled:opacity-50"
        >
          <ShieldCheck size={16} />
          {t("trust")}
        </button>
        <button
          onClick={() => rule("unverified")}
          disabled={pending}
          className="inline-flex items-center gap-1.5 rounded-2xl px-3 py-2 text-sm text-text-muted transition-colors hover:text-text-primary disabled:opacity-50"
        >
          <X size={16} />
          {t("dismiss")}
        </button>
      </div>
    </section>
  );
}
