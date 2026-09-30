"use client";

import type { ReactNode } from "react";
import { AlertCircle, AlertTriangle, RotateCw, Zap } from "lucide-react";
import { useTranslations } from "next-intl";

/** When a control's change takes effect; shown in the heading it belongs to. */
export function ApplyMarker({ when }: { when: "now" | "restart" }): React.ReactElement {
  const t = useTranslations("settings.llm.markers");
  if (when === "now") {
    return (
      <span className="flex items-center gap-1.5 text-xs font-normal text-accent-teal">
        <Zap size={14} aria-hidden="true" />
        {t("now")}
      </span>
    );
  }
  return (
    <span className="flex items-center gap-1.5 text-xs font-normal text-accent-amber">
      <RotateCw size={14} aria-hidden="true" />
      {t("restart")}
    </span>
  );
}

export function ErrorBlock({
  title,
  children,
  testId,
}: {
  title: string;
  children: ReactNode;
  testId?: string;
}): React.ReactElement {
  return (
    <div
      role="alert"
      data-testid={testId}
      className="flex gap-3.5 rounded-xl border border-danger bg-danger/10 p-5"
    >
      <AlertCircle size={20} aria-hidden="true" className="mt-0.5 shrink-0 text-danger" />
      <div className="flex min-w-0 flex-col gap-2">
        <p className="text-sm font-semibold text-danger">{title}</p>
        {children}
      </div>
    </div>
  );
}

export function WarningBlock({
  title,
  children,
  testId,
}: {
  title: string;
  children: ReactNode;
  testId?: string;
}): React.ReactElement {
  return (
    <div
      role="status"
      data-testid={testId}
      className="flex gap-3.5 rounded-xl border border-dashed border-accent-amber bg-bg-elevated p-5"
    >
      <AlertTriangle size={20} aria-hidden="true" className="mt-0.5 shrink-0 text-accent-amber" />
      <div className="flex min-w-0 flex-col gap-2">
        <p className="text-sm font-semibold text-text-primary">{title}</p>
        {children}
      </div>
    </div>
  );
}
