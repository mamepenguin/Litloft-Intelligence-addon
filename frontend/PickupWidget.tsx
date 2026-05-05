"use client";

import { useEffect, useState } from "react";
import { Sparkles } from "lucide-react";
import { useTranslations } from "next-intl";
import { batchGetFiles } from "@/lib/api";
import type { FileItem } from "@/types";
import { CarouselSection } from "@/components/CarouselSection";

interface PickupWidgetProps {
  drive?: string;
}

export default function PickupWidget({ drive }: PickupWidgetProps) {
  const t = useTranslations("drive");
  const [files, setFiles] = useState<FileItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!drive) {
      setLoading(false);
      return;
    }

    let cancelled = false;

    const load = async () => {
      setLoading(true);
      try {
        const res = await fetch("/api/addons/intelligence/pickup", {
          credentials: "include",
          headers: { "X-Lit-Drive": encodeURIComponent(drive) },
        });
        if (!res.ok || cancelled) {
          setLoading(false);
          return;
        }
        const data = await res.json();
        const ids: string[] = Array.isArray(data.file_ids) ? data.file_ids : [];
        if (ids.length === 0 || cancelled) {
          setLoading(false);
          return;
        }
        const items = await batchGetFiles(ids);
        if (!cancelled) {
          setFiles(items);
        }
      } catch {
        // silently fail — the slot simply won't render
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    load();
    return () => {
      cancelled = true;
    };
  }, [drive]);

  if (!loading && files.length === 0) return null;

  return (
    <CarouselSection
      title={t("pickup")}
      icon={<Sparkles size={20} className="text-accent-cta" />}
      files={files}
      loading={loading}
    />
  );
}
