"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { AlertTriangle } from "lucide-react";
import { useTranslations } from "next-intl";

import { getFailedJobs } from "./api";
import FailedJobsModal from "./FailedJobsModal";

const POLL_INTERVAL = 10_000;

/**
 * Indexing jobs that gave up, said once and above everything that is
 * fine.
 *
 * This used to be the last row of the index-status widget, three
 * sections down the dashboard — which put the one thing needing a
 * decision below everything that needed none. It also drew a "no failed
 * jobs" row when there were none, so the usual state of a healthy
 * install was a line of text reporting that there was nothing to
 * report.
 *
 * Nothing wrong renders nothing at all: `dashboard-alerts` supplies no
 * wrapper and no heading, so an entry that always renders would be a
 * permanent band above the page. That is also why the first poll
 * renders nothing rather than a skeleton — a band that appears and then
 * vanishes on every dashboard load is worse than one that arrives a
 * moment late.
 */
export default function FailedJobsAlert() {
  const t = useTranslations("semanticSearch.failedJobs");
  const [count, setCount] = useState(0);
  const [open, setOpen] = useState(false);
  const mountedRef = useRef(true);

  const poll = useCallback(async () => {
    try {
      // limit=1: only ``total`` is wanted here. The modal fetches the
      // rows when it opens.
      const failed = await getFailedJobs(1, 0);
      if (mountedRef.current) setCount(failed.total ?? 0);
    } catch {
      // The dashboard has its own way of saying the addon is
      // unreachable (the index-status widget). An alert that turned a
      // failed poll into a warning band would report the wrong problem.
      if (mountedRef.current) setCount(0);
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    poll();
    const timer = setInterval(poll, POLL_INTERVAL);
    return () => {
      mountedRef.current = false;
      clearInterval(timer);
    };
  }, [poll]);

  // Nothing wrong, and nobody looking: render nothing at all.
  //
  // The modal is deliberately outside that test. Retry and Exclude both
  // remove jobs from the failing set, so an operator clearing the last
  // two would have the poll drop `count` to 0 underneath them and the
  // dialog they are reading would vanish mid-read — nobody pressed
  // Close. The band goes; the modal stays until it is dismissed.
  if (count <= 0 && !open) return null;

  return (
    <>
      {count > 0 && (
        <button
          type="button"
          onClick={() => setOpen(true)}
          // No `aria-label`: WCAG 2.5.3 asks that the accessible name
          // contain the visible label, and the name computed from the
          // content ("3 failed jobs View") already does. An
          // `aria-label` here could only subtract from it.
          className="flex w-full items-center justify-between gap-3 rounded-xl border border-accent-amber/30 bg-accent-amber/10 px-4 py-3 text-sm font-medium text-accent-amber transition-colors hover:bg-accent-amber/20"
        >
          <span className="flex min-w-0 items-center gap-2">
            <AlertTriangle size={16} className="shrink-0" />
            <span className="truncate">{t("summary", { count })}</span>
          </span>
          <span className="shrink-0 text-xs">{t("view")}</span>
        </button>
      )}

      <FailedJobsModal open={open} onClose={() => setOpen(false)} />
    </>
  );
}
