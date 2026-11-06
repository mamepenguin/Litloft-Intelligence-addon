"use client";

/** 「ここに質問」 on a collection page: opens Ask scoped to that collection. */

import { useRouter } from "next/navigation";
import { useTranslations } from "next-intl";
import { MessageCircleQuestion } from "lucide-react";

import { Button } from "@/components/Button";
import { formatOperatorToken } from "@/lib/searchQuery";

interface CollectionAskButtonProps {
  drive: string;
  collectionId: string;
  collectionName: string;
}

function hasQuote(value: string): boolean {
  return value.includes('"') || value.includes("＂");
}

export function CollectionAskButton({
  drive,
  collectionName,
}: CollectionAskButtonProps) {
  const t = useTranslations("askSearch");
  const router = useRouter();

  // A name the quoting writer cannot express draws nothing (H6). An empty
  // collection keeps its button: the Ask page's empty state says so.
  if (!collectionName || hasQuote(collectionName)) return null;

  return (
    <Button
      variant="secondary"
      onClick={() => {
        const scope = formatOperatorToken("in", collectionName);
        router.push(
          `/drive/${encodeURIComponent(drive)}/addons/intelligence?scope=${encodeURIComponent(scope)}`,
        );
      }}
    >
      <MessageCircleQuestion size={16} />
      {t("askHere")}
    </Button>
  );
}

export default CollectionAskButton;
