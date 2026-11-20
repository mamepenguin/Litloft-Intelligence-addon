"use client";

import { useEffect, useState } from "react";
import { MessageCircleQuestion } from "lucide-react";
import { useTranslations } from "next-intl";

import { PageHeader } from "@/components/PageHeader";
import { getDriveFiles, getDrives } from "@/lib/api";
import { filterFromParsed, parseSearchQuery } from "@/lib/searchQuery";

/**
 * The drive's own `file_count` from core, not the size of the index: what Ask
 * can retrieve from is a smaller subset this page cannot know. The addon's
 * `/status` counter is summed across every drive, so it cannot answer either.
 *
 * A scoped Ask shows the scope's count instead, read from the listing the
 * scope names (`meta.total`). The `scope` event's `matched_file_count`
 * replaces it once an answer lands, so the header and the answer agree.
 *
 * A scope naming a Smart Collection has no listing to read: its members are
 * the union of those Smart Collections' files, which the page evaluates, or —
 * beside another chip — a size only the answer's `scope` event knows.
 */
function useScopeFileCount(
  drive: string | null,
  scope: string,
  matchedCount: number | null,
  smartCollectionScope: boolean,
  smartMemberCount: number | null,
): number | null {
  const [count, setCount] = useState<number | null>(null);

  useEffect(() => {
    if (!drive) {
      setCount(null);
      return;
    }
    if (matchedCount !== null) {
      setCount(matchedCount);
      return;
    }
    if (smartCollectionScope) {
      setCount(smartMemberCount);
      return;
    }
    let cancelled = false;
    setCount(null);
    void (async () => {
      try {
        const trimmedScope = scope.trim();
        if (trimmedScope) {
          const filter = filterFromParsed(parseSearchQuery(trimmedScope), null);
          const page = await getDriveFiles(drive, {
            folder: filter.folders,
            collection: filter.collections,
            tag: filter.tags,
            type: filter.kinds,
            limit: 1,
          });
          if (!cancelled) setCount(page.meta?.total ?? null);
          return;
        }
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
  }, [drive, scope, matchedCount, smartCollectionScope, smartMemberCount]);

  return count;
}

/**
 * The Ask page's header. A count that failed to load leaves the drive alone
 * rather than a number that would be believed.
 */
export function AskFindHeader({
  drive,
  scope = "",
  matchedCount = null,
  smartCollectionScope = false,
  smartMemberCount = null,
}: {
  drive: string | null;
  scope?: string;
  matchedCount?: number | null;
  /** The Ask scope names a Smart Collection, so no listing can count it. */
  smartCollectionScope?: boolean;
  /** The union size of a scope of Smart Collection chips only. */
  smartMemberCount?: number | null;
}) {
  const tNav = useTranslations("intelligence.nav");
  const tCommon = useTranslations("common");
  const count = useScopeFileCount(
    drive,
    scope,
    matchedCount,
    smartCollectionScope,
    smartMemberCount,
  );

  const scopeLine = drive ? (
    <span data-testid="drive-scope">
      {count === null
        ? drive
        : tCommon("driveScope", { drive, detail: tCommon("items", { count }) })}
    </span>
  ) : undefined;

  return (
    <PageHeader
      titleIcon={MessageCircleQuestion}
      title={tNav("label")}
      scope={scopeLine}
    />
  );
}
