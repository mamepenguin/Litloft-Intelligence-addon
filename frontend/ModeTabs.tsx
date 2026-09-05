"use client";

/**
 * Tab row for switching between Ask and Find modes.
 *
 * Both modes are separate routes (`/addons/intelligence` and
 * `/addons/intelligence/find`) but share the same input experience:
 * choosing a tab navigates to the other route while preserving the
 * current query string as `?q=`. The destination page auto-fires its
 * pipeline on mount when `?q=` is non-empty.
 *
 * The row itself is core's `PageTabs`; what stays here is the pair of
 * destinations and the query it carries across. Two things changed with the
 * adoption, both core's contract rather than a local choice:
 *
 * - **No `role="tablist"` and no `aria-selected`.** These tabs navigate, and
 *   `PageTabs` treats a navigating row as navigation: `role="tab"` promises a
 *   screen reader that activating it swaps a panel in the same view, which a
 *   `<Link>` does not do. The state a link in a set carries is
 *   `aria-current="page"`, and this row used to carry both vocabularies at
 *   once. Media Import's adoption resolved the same pairing from the other
 *   end — its two views *are* one page, so it kept the tablist and dropped
 *   `aria-current`.
 * - **The selected tab is no longer `bg-accent text-white`.** DESIGN.md §2.2
 *   allows one accent fill per screen, and spending it on saying which tab
 *   you are already looking at leaves none for the thing the screen is for.
 *   `PageTabs` marks the selection with a 2px border instead.
 *
 * The component is presentational — it does not own the input value. Parents
 * pass the current input via `query` so the tab can attach it to the
 * destination URL.
 */

import { ListFilter, Sparkles } from "lucide-react";
import { useTranslations } from "next-intl";

import { PageTabs } from "@/components/PageTabs";

interface ModeTabsProps {
  current: "ask" | "find";
  query: string;
  drive: string;
}

export default function ModeTabs({ current, query, drive }: ModeTabsProps) {
  const t = useTranslations("modeTabs");
  const trimmed = query.trim();
  const qParam = trimmed ? `?q=${encodeURIComponent(trimmed)}` : "";
  const driveSegment = encodeURIComponent(drive);

  return (
    <PageTabs
      label={t("ariaLabel")}
      current={current}
      items={[
        {
          key: "ask",
          label: t("ask"),
          icon: Sparkles,
          href: `/drive/${driveSegment}/addons/intelligence${qParam}`,
        },
        {
          key: "find",
          label: t("find"),
          icon: ListFilter,
          href: `/drive/${driveSegment}/addons/intelligence/find${qParam}`,
        },
      ]}
    />
  );
}
