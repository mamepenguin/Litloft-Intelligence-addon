"use client";

import { useEffect, useState } from "react";
import { useTranslations } from "next-intl";

import { getDrives } from "@/lib/api";

/**
 * What the question will be asked of.
 *
 * Both search pages tell the reader nothing about their scope, and a drive
 * is a hard boundary — an answer that found nothing and an answer that
 * looked in the wrong place read identically without this line.
 *
 * The count comes from core's public `GET /api/drives`, which returns only
 * the drives the caller may see and carries `file_count` per drive. The
 * addon's own `/status` counter cannot answer it: `api.ts` records that it
 * is process-wide, summed across every drive.
 *
 * **Silent on failure.** A wrong number here is worse than no number: it
 * would be read as "the index holds 619 of your files" and believed.
 *
 * One string and one component for both pages, though the spec named
 * `askSearch.scope` and `find.scope` separately. Two copies of one sentence
 * is the drift this repository keeps paying for; the namespace it lives in
 * is `find` because that page had the vocabulary first.
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
