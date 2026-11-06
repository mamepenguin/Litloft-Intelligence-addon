"use client";

/** 「ここに質問」 in a folder's `…` menu: opens Ask scoped to that folder. */

import { useRouter } from "next/navigation";
import { useTranslations } from "next-intl";
import { MessageCircleQuestion } from "lucide-react";

import { ActionMenuItem } from "@/components/ActionMenuItem";
import { formatOperatorToken } from "@/lib/searchQuery";

interface FolderAskMenuItemProps {
  drive: string;
  path: string;
  fileIds?: string[];
  onRequestClose?: () => void;
}

function hasQuote(value: string): boolean {
  return value.includes('"') || value.includes("＂");
}

export function FolderAskMenuItem({
  drive,
  path,
  onRequestClose,
}: FolderAskMenuItemProps) {
  const t = useTranslations("askSearch");
  const router = useRouter();

  // The drive root is the whole drive (the sidebar's Ask entry), and the
  // quoting writer cannot express a name holding a quote (H6).
  if (!path || hasQuote(path)) return null;

  return (
    <ActionMenuItem
      icon={MessageCircleQuestion}
      label={t("askHere")}
      onClick={() => {
        onRequestClose?.();
        const scope = formatOperatorToken("path", path);
        router.push(
          `/drive/${encodeURIComponent(drive)}/addons/intelligence?scope=${encodeURIComponent(scope)}`,
        );
      }}
    />
  );
}

export default FolderAskMenuItem;
