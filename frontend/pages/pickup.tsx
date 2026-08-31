"use client";

/**
 * Intelligence → Pickup feed.
 *
 * The drive-home carousel is the entrance; this is the thing itself.
 * A few hundred files the viewer has never opened, ordered by how well
 * they match the interests their watch history describes.
 *
 * Paged in stable rank order, deliberately. The carousel rotates its
 * twelve cards daily, but a paged list cannot: reshuffling between
 * pages duplicates rows on one and skips them on the next.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { useTranslations } from "next-intl";
import { Sparkles } from "lucide-react";

import { useCurrentDrive } from "@/components/CurrentDriveProvider";
import { FileGrid } from "@/components/FileGrid";
import { batchGetFiles } from "@/lib/api";
import type { FileItem } from "@/types";
import { fetchPickup } from "../api";

const PAGE_SIZE = 40;

export default function PickupPage() {
  const t = useTranslations("intelligence");
  const drive = useCurrentDrive();
  const [files, setFiles] = useState<FileItem[]>([]);
  const [total, setTotal] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [failed, setFailed] = useState(false);
  const sentinel = useRef<HTMLDivElement | null>(null);
  // Read inside the observer callback, which would otherwise close over
  // a stale count and ask for the same page forever.
  const loadedRef = useRef(0);
  // Bumped whenever the drive changes. A request in flight when that
  // happens resolves against the old drive, and without this its files
  // would be appended to the new drive's feed.
  const generation = useRef(0);

  const loadMore = useCallback(async () => {
    // Not gated on ``failed``: retry calls this directly, and the
    // callback it holds was built while the flag was still set. The
    // sentinel is unmounted on failure, so nothing else can re-enter.
    if (!drive || loading) return;
    const offset = loadedRef.current;
    if (total !== null && offset >= total) return;

    const mine = generation.current;
    setLoading(true);
    try {
      const page = await fetchPickup(drive, { limit: PAGE_SIZE, offset });
      if (mine !== generation.current) return;
      setTotal(page.total);
      if (page.file_ids.length === 0) return;
      const items = await batchGetFiles(page.file_ids);
      if (mine !== generation.current) return;
      loadedRef.current = offset + page.file_ids.length;
      setFiles((prev) => [...prev, ...items]);
    } catch {
      // Hydration can fail after the page itself succeeded. Stopping is
      // the point: the sentinel is re-observed as soon as loading ends,
      // so retrying from the same offset would spin on the same failure
      // for as long as the page is open.
      if (mine === generation.current) setFailed(true);
    } finally {
      if (mine === generation.current) setLoading(false);
    }
  }, [drive, loading, total]);

  // Clearing the flag is not enough on its own: the sentinel only
  // re-triggers a load when it crosses back into view, and after a
  // failure it is usually already sitting there.
  const retry = useCallback(() => {
    setFailed(false);
    void loadMore();
  }, [loadMore]);

  useEffect(() => {
    generation.current += 1;
    setFiles([]);
    setTotal(null);
    setFailed(false);
    // Also cleared here, and not only in the finally above. A request
    // belonging to the previous drive must not write state on the way
    // out, so it leaves this flag set — and the new drive's first load
    // would then find the page busy and never start.
    setLoading(false);
    loadedRef.current = 0;
  }, [drive]);

  useEffect(() => {
    if (files.length === 0 && total === null && !failed) void loadMore();
  }, [files.length, total, failed, loadMore]);

  useEffect(() => {
    const node = sentinel.current;
    if (!node) return;
    const observer = new IntersectionObserver((entries) => {
      if (entries.some((e) => e.isIntersecting)) void loadMore();
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, [loadMore]);

  const exhausted = total !== null && loadedRef.current >= total;

  return (
    <div className="mx-auto w-full max-w-7xl px-4 py-6">
      <header className="mb-6">
        <h1 className="flex items-center gap-2 text-2xl font-bold text-text-primary">
          <Sparkles size={22} className="text-accent-cta" />
          {t("pickup.heading")}
        </h1>
        <p className="mt-1 text-sm text-text-muted">
          {t("pickup.description")}
        </p>
      </header>

      {files.length > 0 && <FileGrid files={files} />}

      {files.length === 0 && !loading && total !== null && (
        <p className="py-12 text-center text-sm text-text-muted">
          {t("pickup.empty")}
        </p>
      )}

      {loading && (
        <p className="py-6 text-center text-sm text-text-muted">
          {t("pickup.loading")}
        </p>
      )}

      {failed && !loading && (
        <div className="py-6 text-center">
          <button
            type="button"
            onClick={retry}
            className="text-sm text-text-muted transition-colors hover:text-accent"
          >
            {t("pickup.retry")}
          </button>
        </div>
      )}

      {!exhausted && !failed && (
        <div ref={sentinel} className="h-px" aria-hidden />
      )}
    </div>
  );
}
