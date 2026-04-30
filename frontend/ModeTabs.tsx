"use client";

/**
 * Tab row for switching between Ask and Find modes.
 *
 * Both modes are separate routes (`/addons/intelligence` and
 * `/addons/intelligence/find`) but share the same input experience:
 * clicking a tab navigates to the other route while preserving the
 * current query string as `?q=`. The destination page auto-fires its
 * pipeline on mount when `?q=` is non-empty.
 *
 * The component is presentational — it does not own the input value.
 * Parents pass the current input via `query` so the tab can attach it
 * to the destination URL.
 */

import Link from "next/link";
import { useTranslations } from "next-intl";
import { ListFilter, Sparkles } from "lucide-react";

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
  const askHref = `/drive/${driveSegment}/addons/intelligence${qParam}`;
  const findHref = `/drive/${driveSegment}/addons/intelligence/find${qParam}`;

  return (
    <nav
      role="tablist"
      aria-label={t("ariaLabel")}
      className="inline-flex items-center gap-1 rounded-2xl bg-bg-card p-1 self-start"
    >
      <Tab
        href={askHref}
        active={current === "ask"}
        icon={<Sparkles size={14} />}
        label={t("ask")}
      />
      <Tab
        href={findHref}
        active={current === "find"}
        icon={<ListFilter size={14} />}
        label={t("find")}
      />
    </nav>
  );
}

interface TabProps {
  href: string;
  active: boolean;
  icon: React.ReactNode;
  label: string;
}

function Tab({ href, active, icon, label }: TabProps) {
  const baseClass =
    "inline-flex items-center gap-1.5 rounded-xl px-3 py-1.5 text-xs font-medium transition-colors";
  const stateClass = active
    ? "bg-accent text-white"
    : "text-text-muted hover:text-text-primary hover:bg-bg-elevated";
  return (
    <Link
      href={href}
      role="tab"
      aria-selected={active}
      aria-current={active ? "page" : undefined}
      className={`${baseClass} ${stateClass}`}
    >
      {icon}
      {label}
    </Link>
  );
}
