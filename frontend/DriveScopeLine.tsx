"use client";

import { useEffect, useState } from "react";
import { useTranslations } from "next-intl";

import { getDrives } from "@/lib/api";

/**
 * Which drive is in scope, and how big it is.
 *
 * Both search pages tell the reader nothing about their scope, and a drive
 * is a hard boundary — a result that found nothing and one that looked in
 * the wrong place read identically without this line.
 *
 * **It is the size of the drive, not the size of the index.** Core's
 * `file_count` is every active file there, whatever its type; what Ask can
 * actually retrieve from is the subset carrying a transcript, a caption or
 * an embedding, and is smaller by an amount this component cannot know.
 * The wording says "files in <drive>" for that reason and stops there —
 * "files it will read" would be a claim about the index.
 *
 * The count comes from core's public `GET /api/drives`, which returns only
 * the drives the caller may see. The addon's own `/status` counter cannot
 * answer it: `api.ts` records that it is process-wide, summed across every
 * drive.
 *
 * **Silent on failure.** A wrong number here is worse than no number: it
 * would be read as "the index holds 619 of your files" and believed.
 *
 * One string and one component for both pages, though the spec named
 * `askSearch.scope` and `find.scope` separately. Two copies of one sentence
 * is the drift this repository keeps paying for; the namespace it lives in
 * is `find` because that page had the vocabulary first. Sharing it is also
 * why the sentence names no verb: the first version said "Asking across…",
 * which is false on the page that searches.
 */
export function DriveScopeLine({ drive }: { drive: string | null }) {
  const t = useTranslations("find");
  const [count, setCount] = useState<number | null>(null);

  useEffect(() => {
    if (!drive) {
      setCount(null);
      return;
    }
    let cancelled = false;
    void (async () => {
      try {
        const drives = await getDrives();
        const match = drives.find((d) => d.name === drive);
        if (!cancelled) setCount(match ? match.file_count : null);
      } catch {
        if (!cancelled) setCount(null);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [drive]);

  if (!drive || count === null) return null;

  return (
    <p className="text-xs text-text-muted" data-testid="drive-scope">
      {t("scope", { drive, count })}
    </p>
  );
}
