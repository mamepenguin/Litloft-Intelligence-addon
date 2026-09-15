"use client";

import { useEffect, useState } from "react";
import { MessageCircleQuestion } from "lucide-react";
import { useTranslations } from "next-intl";

import { PageHeader } from "@/components/PageHeader";
import { getDrives } from "@/lib/api";

import ModeTabs from "./ModeTabs";

/**
 * The drive's own `file_count` from core, not the size of the index: what Ask
 * can retrieve from is a smaller subset this page cannot know. The addon's
 * `/status` counter is summed across every drive, so it cannot answer either.
 */
function useDriveFileCount(drive: string | null): number | null {
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
}: {
  current: "ask" | "find";
  query: string;
  drive: string | null;
}) {
  const tNav = useTranslations("intelligence.nav");
  const tCommon = useTranslations("common");
  const count = useDriveFileCount(drive);

  const scope = drive ? (
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
      scope={scope}
      tabs={drive ? <ModeTabs current={current} query={query} drive={drive} /> : undefined}
    />
  );
}
