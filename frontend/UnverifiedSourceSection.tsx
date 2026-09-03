"use client";

import { useCallback, useEffect, useState } from "react";
import { ShieldCheck, X } from "lucide-react";
import { useTranslations } from "next-intl";

interface UnverifiedSourceSectionProps {
  fileId: string;
  trustTier?: "verified" | "unverified";
  trustReviewedAt?: string | null;
  onFileChange?: (file: unknown) => void;
}

/**
 * Asks the viewer to rule on a source they are reading.
 *
 * The panel is the question and nothing else. Evidence for answering it
 * comes from *Similar files*, which is permanent and applies whatever a
 * file's trust state — unlike a panel that disappears the instant
 * anyone answers.
 *
 * To distil one paragraph rather than trust the whole page, select it in
 * the document and use Knowledge's quotation basket. That affordance is
 * owned by Knowledge and is deliberately not duplicated here —
 * Intelligence must not depend on another addon.
 */
export default function UnverifiedSourceSection({
  fileId,
  trustTier,
  trustReviewedAt,
  onFileChange,
}: UnverifiedSourceSectionProps) {
  const t = useTranslations("intelligence.unverifiedSource");
  const [pending, setPending] = useState(false);
  const [done, setDone] = useState(false);

  // Only unverified files are asked about, and only while nobody has ruled:
  // once dismissed, re-asking on every open would be nagging.
  const applies = trustTier === "unverified" && !trustReviewedAt;

  useEffect(() => {
    setDone(false);
  }, [fileId]);

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
      <p className="mt-2 text-xs text-text-muted">{t("distilHint")}</p>

      <div className="mt-4 flex flex-wrap gap-2">
        <button
          onClick={() => rule("verified")}
          disabled={pending}
          className="inline-flex items-center gap-1.5 rounded-2xl bg-accent px-3 py-2 text-sm text-white transition-colors hover:bg-accent-hover disabled:bg-sand disabled:text-warm-silver disabled:cursor-not-allowed"
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
