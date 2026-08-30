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
  const sentinel = useRef<HTMLDivElement | null>(null);
  // Read inside the observer callback, which would otherwise close over
  // a stale count and ask for the same page forever.
  const loadedRef = useRef(0);

  const loadMore = useCallback(async () => {
    if (!drive || loading) return;
    const offset = loadedRef.current;
    if (total !== null && offset >= total) return;

    setLoading(true);
    try {
      const page = await fetchPickup(drive, { limit: PAGE_SIZE, offset });
      setTotal(page.total);
      if (page.file_ids.length === 0) return;
      const items = await batchGetFiles(page.file_ids);
      loadedRef.current = offset + page.file_ids.length;
      setFiles((prev) => [...prev, ...items]);
    } finally {
      setLoading(false);
    }
  }, [drive, loading, total]);

  useEffect(() => {
    setFiles([]);
    setTotal(null);
    loadedRef.current = 0;
  }, [drive]);

  useEffect(() => {
    if (files.length === 0 && total === null) void loadMore();
  }, [files.length, total, loadMore]);

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

      {!exhausted && <div ref={sentinel} className="h-px" aria-hidden />}
    </div>
  );
}
