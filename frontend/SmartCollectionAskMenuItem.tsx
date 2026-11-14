"use client";

/** 「ここに質問」 in a Smart Collection's `…` menu: opens Ask scoped to it. */

import { useRouter } from "next/navigation";
import { useTranslations } from "next-intl";
import { MessageCircleQuestion } from "lucide-react";

import { ActionMenuItem } from "@/components/ActionMenuItem";
import { formatOperatorToken } from "@/lib/searchQuery";

interface SmartCollectionAskMenuItemProps {
  drive: string;
  smartCollectionId: string;
  smartCollectionName: string;
  closeMenu: () => void;
}

function hasQuote(value: string): boolean {
  return value.includes('"') || value.includes("＂");
}

export function SmartCollectionAskMenuItem({
  drive,
  smartCollectionName,
  closeMenu,
}: SmartCollectionAskMenuItemProps) {
  const t = useTranslations("askSearch");
  const router = useRouter();

  // The quoting writer cannot express a name holding a quote (H6).
  if (!smartCollectionName || hasQuote(smartCollectionName)) return null;

  return (
    <ActionMenuItem
      icon={MessageCircleQuestion}
      label={t("askHere")}
      onClick={() => {
        closeMenu();
        const scope = formatOperatorToken("in", smartCollectionName);
        router.push(
          `/drive/${encodeURIComponent(drive)}/addons/intelligence?scope=${encodeURIComponent(scope)}`,
        );
      }}
    />
  );
}

export default SmartCollectionAskMenuItem;
