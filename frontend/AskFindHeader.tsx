"use client";

import { useEffect, useState } from "react";
import { MessageCircleQuestion } from "lucide-react";
import { useTranslations } from "next-intl";

import { PageHeader } from "@/components/PageHeader";
import { getDriveFiles, getDrives } from "@/lib/api";
import { filterFromParsed, parseSearchQuery } from "@/lib/searchQuery";

import ModeTabs from "./ModeTabs";

/**
 * The drive's own `file_count` from core, not the size of the index: what Ask
 * can retrieve from is a smaller subset this page cannot know. The addon's
 * `/status` counter is summed across every drive, so it cannot answer either.
 *
 * A scoped Ask shows the scope's count instead, read from the listing the
 * scope names (`meta.total`). The `scope` event's `matched_file_count`
 * replaces it once an answer lands, so the header and the answer agree.
 */
function useScopeFileCount(
  drive: string | null,
  scope: string,
  matchedCount: number | null,
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
  }, [drive, scope, matchedCount]);

  return count;
}

/**
 * One header for both modes, so moving between the Ask and Find tabs does not
 * move or rename the title. A count that failed to load leaves the drive alone
 * rather than a number that would be believed.
 */
export function AskFindHeader({
  current,
  query,
  drive,
  scope = "",
  matchedCount = null,
}: {
  current: "ask" | "find";
  query: string;
  drive: string | null;
  scope?: string;
  matchedCount?: number | null;
}) {
  const tNav = useTranslations("intelligence.nav");
  const tCommon = useTranslations("common");
  const count = useScopeFileCount(
    drive,
    current === "ask" ? scope : "",
    matchedCount,
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
      tabs={
        drive ? (
          <ModeTabs
            current={current}
            query={query}
            drive={drive}
            scope={scope}
          />
        ) : undefined
      }
    />
  );
}
