"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { FileText, Quote, Sparkles } from "lucide-react";
import { useTranslations } from "next-intl";

import {
  getFileTranscript,
  refineFileTranscript,
} from "./api";
import type { TranscriptChunkItem } from "./api";
import { formatDuration } from "@/lib/format";
import { getSubtitleUrl } from "@/lib/api";
import type { SubtitleInfo } from "@/types";
import { useAddonStatus } from "@/components/AddonSlotsProvider";
import type { MediaController } from "@/lib/mediaController";
import { getMediaClockSnapshot, subscribeMediaClock } from "@/lib/mediaClock";
import { addSourceCapture } from "@/lib/sourceCapture";
import {
  recallTranscriptScroll,
  type TranscriptPlace,
  rememberTranscriptScroll,
} from "./transcriptScroll";

interface TranscriptSectionProps {
  fileId: string;
  drive: string;
  filename?: string;
  fileType?: string;
  /**
   * The playback handle. Drives both the active-cue highlight and
   * click-to-seek, for every backend — a native element reference is
   * no longer involved in either.
   */
  mediaController?: MediaController | null;
  subtitles?: SubtitleInfo[];
  /**
   * Set by the host when this is rendered as the companion rail beside
   * the player, where there is a real height to fill. In the stacked
   * form — audio, narrow containers, mobile — it stays a bounded box,
   * because filling the height there would mean filling the page.
   */
  fillHeight?: boolean;
  /**
   * The host has already written this panel's name above it.
   *
   * True in the inspector's tab strip, where the button the reader just
   * pressed says "Transcript" — repeating it under the button spends a
   * line saying what they can still see. False in the box below the
   * player, which has no heading of its own, so the title is the only
   * thing naming what the box holds.
   */
  labelledByHost?: boolean;
  /**
   * Whether this file has a transcript at all, reported to the host.
   *
   * The host draws a tab per `player-side` entry and cannot look inside
   * one to find out whether it has anything — asking by name would be
   * the core-to-addon dependency the rules forbid. Without an answer it
   * assumes yes, which is what a video nobody has transcribed used to
   * get: a Transcript tab opening on an empty panel.
   *
   * Answered `false` on mount and corrected when the fetches settle.
   * The host keeps this component mounted while the answer is `false` —
   * it is the thing giving the answer — so it can be taken back.
   */
  onAvailability?: (available: boolean) => void;
}

type Source = "chunks" | "words" | "external";

const CJK_LANGUAGES = /^(ja|zh|ko|th)/i;

function parseVttCues(vtt: string): TranscriptChunkItem[] {
  const lines = vtt.split(/\r?\n/);
  const cues: TranscriptChunkItem[] = [];
  const tsRe = /(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})\.(\d{3})/;
  // Extract language from VTT header (e.g. "Language: ja")
  const langMatch = vtt.match(/^Language:\s*(\S+)/m);
  const isCjk = langMatch ? CJK_LANGUAGES.test(langMatch[1]) : false;
  const joiner = isCjk ? "" : " ";
  let current: { start: number; end: number; text: string[] } | null = null;
  let idx = 0;
  const flush = () => {
    if (current) {
      cues.push({
        index: idx++,
        start: current.start,
        end: current.end,
        text: current.text.join(joiner).trim(),
      });
      current = null;
    }
  };
  for (const raw of lines) {
    const line = raw.trim();
    const m = line.match(tsRe);
    if (m) {
      flush();
      const n = m.map((x) => Number(x));
      current = {
        start: n[1] * 3600 + n[2] * 60 + n[3] + n[4] / 1000,
        end: n[5] * 3600 + n[6] * 60 + n[7] + n[8] / 1000,
        text: [],
      };
      continue;
    }
    if (!line) {
      flush();
      continue;
    }
    if (line.startsWith("WEBVTT") || line.startsWith("NOTE") || line.startsWith("Language:")) continue;
    if (current) current.text.push(line.replace(/<[^>]+>/g, ""));
  }
  flush();
  return cues.filter((c) => c.text);
}

const EMPTY_SUBTITLES: SubtitleInfo[] = [];

/**
 * The box that scrolls `list`: the host's scroller when the host has
 * marked one around it, and otherwise the list itself.
 *
 * Asked of the host rather than guessed from which ancestor overflows: on
 * the file page the box around a short list is the canvas holding the
 * video, and in the sheet the scroller may not overflow yet when asked.
 */
function scrollingBoxOf(list: HTMLElement): HTMLElement {
  return list.closest<HTMLElement>("[data-inspector-scroller]") ?? list;
}

/** The first line of `scroller` the reader can see, in viewport px. */
function viewTopOf(list: HTMLElement, scroller: HTMLElement): number {
  if (scroller === list) return list.getBoundingClientRect().top;
  return scroller.getBoundingClientRect().top + stripCoverOf(list);
}

function cueRows(list: HTMLElement): HTMLElement[] {
  return Array.from(list.querySelectorAll<HTMLElement>("[data-cue-start]"));
}

function readPlace(list: HTMLElement): TranscriptPlace | null {
  const viewTop = viewTopOf(list, scrollingBoxOf(list));
  const rows = cueRows(list);
  for (const [index, row] of rows.entries()) {
    const rect = row.getBoundingClientRect();
    if (rect.bottom > viewTop) {
      const at = Number(row.dataset.cueStart);
      const nth = rows
        .slice(0, index)
        .filter((r) => Number(r.dataset.cueStart) === at).length;
      return { at, into: viewTop - rect.top, ...(nth > 0 ? { nth } : {}) };
    }
  }
  return null;
}

function applyPlace(list: HTMLElement, place: TranscriptPlace): void {
  const rows = cueRows(list);
  if (rows.length === 0) return;
  const startOf = (r: HTMLElement) => Number(r.dataset.cueStart);
  // The latest start at or before the place; subtitles often start two
  // lines together, and `nth` says which of them.
  const latest = rows.reduce<number | null>((best, r) => {
    const start = startOf(r);
    return start <= place.at && (best === null || start > best) ? start : best;
  }, null);
  const together = latest === null ? [] : rows.filter((r) => startOf(r) === latest);
  const row =
    together.length === 0
      ? rows[0]
      : together[Math.min(place.at === latest ? (place.nth ?? 0) : 0, together.length - 1)];
  const rect = row.getBoundingClientRect();
  // A row laid out shorter than before cannot hold the old distance into
  // it; carrying it over would land on the row after.
  const into = rect.height > 0 ? Math.min(place.into, rect.height - 1) : place.into;
  const scroller = scrollingBoxOf(list);
  const viewTop = viewTopOf(list, scroller);
  scroller.scrollTop += rect.top - (viewTop - into);
}

/** How far down the host's pinned strip covers its scroller. */
function stripCoverOf(list: HTMLElement): number {
  return (
    Number.parseFloat(
      getComputedStyle(list).getPropertyValue("--inspector-sticky-top"),
    ) || 0
  );
}

export default function TranscriptSection({
  fileId,
  drive,
  filename,
  fileType = "video",
  mediaController,
  subtitles = EMPTY_SUBTITLES,
  fillHeight = false,
  labelledByHost = false,
  onAvailability,
}: TranscriptSectionProps) {
  const t = useTranslations("searchIndex");
  const addonStatus = useAddonStatus("intelligence");
  const refineFeature = addonStatus.features?.transcript_refine;
  // Backend sends either boolean false or the string "false" when the
  // feature is fully OFF. Anything else ("manual", "on_index", true)
  // counts as enabled for UI purposes.
  const refineEnabled =
    refineFeature !== false && refineFeature !== "false" && refineFeature !== undefined;
  const [refining, setRefining] = useState(false);
  const [whisperChunks, setWhisperChunks] = useState<TranscriptChunkItem[]>([]);
  const [whisperLanguage, setWhisperLanguage] = useState("");
  const [whisperWordCues, setWhisperWordCues] = useState<TranscriptChunkItem[]>([]);
  const [externalCues, setExternalCues] = useState<TranscriptChunkItem[]>([]);
  // Whether each subtitle fetch has answered, found or not.
  const [wordsSettled, setWordsSettled] = useState(false);
  const [externalSettled, setExternalSettled] = useState(false);
  const [externalLanguage, setExternalLanguage] = useState("");
  const [loading, setLoading] = useState(true);
  const [source, setSource] = useState<Source>("chunks");
  // Until the reader picks one, the source follows what is available: a
  // source that answered first is not a choice. The pick is per file.
  const [chosenSource, setChosenSource] = useState<Source | null>(
    () => (recallTranscriptScroll(fileId)?.source as Source | undefined) ?? null,
  );
  const chosenSourceRef = useRef(chosenSource);
  useEffect(() => {
    chosenSourceRef.current = chosenSource;
  }, [chosenSource]);
  useEffect(() => {
    setChosenSource(
      (recallTranscriptScroll(fileId)?.source as Source | undefined) ?? null,
    );
  }, [fileId]);
  const [activeIndex, setActiveIndex] = useState(-1);
  // Whether the highlight is still allowed to drag the list around.
  // Reading ahead has to win over following, or the reader is pulled
  // back every few seconds.
  const [following, setFollowing] = useState(true);
  const activeRef = useRef<HTMLButtonElement>(null);
  const listRef = useRef<HTMLDivElement | null>(null);
  // State as well as a ref: the list mounts only once loading ends, which
  // can be after the cue count it is keyed on has stopped changing.
  const [listEl, setListEl] = useState<HTMLDivElement | null>(null);
  const attachList = useCallback((node: HTMLDivElement | null) => {
    listRef.current = node;
    setListEl(node);
  }, []);

  const chunksAvailable = whisperChunks.length > 0;
  const wordsAvailable = whisperWordCues.length > 0;
  const externalAvailable = subtitles.length > 0 && externalCues.length > 0;

  // All three fetches abandon a response that arrives after the file
  // changed. The host reuses one mount across files and resets its own
  // per-file state on the way; a late response landing after that would
  // put one file's cues under another file's player, and — since the
  // availability answer is derived from these — would tell the host the
  // new file has a transcript because the old one did.
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    getFileTranscript(fileId, drive).then((res) => {
      if (cancelled) return;
      if (res.available && res.chunks && res.chunks.length > 0) {
        setWhisperChunks(res.chunks);
        setWhisperLanguage(res.language || "");
      } else {
        setWhisperChunks([]);
      }
      setLoading(false);
    });
    return () => {
      cancelled = true;
    };
  }, [fileId, drive]);

  useEffect(() => {
    let cancelled = false;
    setWordsSettled(false);
    fetch(`/api/addons/intelligence/files/${fileId}/subtitles.vtt`)
      .then((r) => (r.ok ? r.text() : ""))
      .then((text) => {
        if (!cancelled) setWhisperWordCues(text ? parseVttCues(text) : []);
      })
      .catch(() => {
        if (!cancelled) setWhisperWordCues([]);
      })
      .finally(() => {
        if (!cancelled) setWordsSettled(true);
      });
    return () => {
      cancelled = true;
    };
  }, [fileId]);

  useEffect(() => {
    if (subtitles.length === 0) {
      setExternalCues([]);
      setExternalSettled(true);
      return;
    }
    let cancelled = false;
    setExternalSettled(false);
    const first = subtitles[0];
    setExternalLanguage(first.language || "");
    fetch(getSubtitleUrl(fileId, first.index))
      .then((r) => (r.ok ? r.text() : ""))
      .then((text) => {
        if (!cancelled) setExternalCues(text ? parseVttCues(text) : []);
      })
      .catch(() => {
        if (!cancelled) setExternalCues([]);
      })
      .finally(() => {
        if (!cancelled) setExternalSettled(true);
      });
    return () => {
      cancelled = true;
    };
  }, [fileId, subtitles]);

  // Reset `source` to the first available one whenever availability changes.
  useEffect(() => {
    const available: Source[] = [];
    if (chunksAvailable) available.push("chunks");
    if (wordsAvailable) available.push("words");
    if (externalAvailable) available.push("external");
    if (available.length === 0) return;
    const wanted =
      chosenSource && available.includes(chosenSource) ? chosenSource : available[0];
    if (source !== wanted) {
      // Rows starting at other times are about to replace these, so the
      // reader's row is read now and put back once they have.
      const list = listRef.current;
      if (list && !list.closest("[hidden]") && !putBackPendingRef.current) {
        placeRef.current = readPlace(list) ?? placeRef.current;
        putBackPendingRef.current = true;
      }
      // The highlight indexes the old rows until the clock syncs again, and
      // would otherwise name a row at another time as the playing one.
      setActiveIndex(-1);
      setSource(wanted);
    }
  }, [chunksAvailable, wordsAvailable, externalAvailable, source, chosenSource]);

  const hasAnything = chunksAvailable || wordsAvailable || externalAvailable;

  // Held in a ref so an inline arrow from the host — which is what a
  // host naturally writes — does not re-fire this on every render of a
  // component that re-renders on every clock tick. Core's own
  // `ChaptersPanel` holds `onResolved` the same way.
  const onAvailabilityRef = useRef(onAvailability);
  useEffect(() => {
    onAvailabilityRef.current = onAvailability;
  });

  // `false` first, because on mount nothing has arrived yet and the
  // host's default is "assume it has something". Answering only when
  // there is something to report would leave the empty tab exactly
  // where it was: silence is what the host reads as yes.
  useEffect(() => {
    onAvailabilityRef.current?.(hasAnything);
  }, [hasAnything]);

  const { cues, language } = useMemo(() => {
    if (source === "external") return { cues: externalCues, language: externalLanguage };
    if (source === "words") return { cues: whisperWordCues, language: whisperLanguage };
    return { cues: whisperChunks, language: whisperLanguage };
  }, [source, externalCues, externalLanguage, whisperWordCues, whisperChunks, whisperLanguage]);

  useEffect(() => {
    // The highlight used to bind `timeupdate` on an HTMLVideoElement,
    // which a YouTube IFrame player never dispatches into the parent
    // document — so it was a native-video-only nicety and said so.
    // The shared playback clock reports position for every backend, so
    // native video, native audio and .loft now take one path.
    if (!mediaController || cues.length === 0) {
      setActiveIndex(-1);
      return;
    }
    const sync = () => {
      const { currentTime } = getMediaClockSnapshot(mediaController);
      // Writing the same index back is a no-op in React, so the list
      // only re-renders when the highlight actually moves — not four
      // times a second for the duration of the video.
      setActiveIndex(
        cues.findIndex((c) => currentTime >= c.start && currentTime < c.end),
      );
    };
    const unsubscribe = subscribeMediaClock(mediaController, sync);
    sync();
    return unsubscribe;
  }, [mediaController, cues]);

  const scrollActiveIntoView = useCallback(() => {
    const list = listRef.current;
    const target = activeRef.current;
    if (!list || !target) return;
    // A panel the host is not showing has no position to aim at, and
    // scrolling an enclosing box for it would move what the reader is
    // looking at instead.
    if (target.closest("[hidden]")) return;
    // Scroll one box — avoid scrollIntoView, which bubbles up and moves
    // the page away from the video.
    const scroller = scrollingBoxOf(list);
    const covered = scroller === list ? 0 : stripCoverOf(list);
    const scrollerRect = scroller.getBoundingClientRect();
    const targetRect = target.getBoundingClientRect();
    const viewTop = scrollerRect.top + covered;
    const above = targetRect.top < viewTop;
    const below = targetRect.bottom > scrollerRect.bottom;
    if (!above && !below) return;
    const targetOffset = targetRect.top - viewTop + scroller.scrollTop;
    const nextTop =
      targetOffset - (scroller.clientHeight - covered - target.clientHeight) / 2;
    scroller.scrollTo({ top: nextTop, behavior: "smooth" });
  }, []);

  useEffect(() => {
    if (!following) return;
    scrollActiveIntoView();
  }, [activeIndex, following, scrollActiveIntoView]);

  /**
   * Stop following when the reader takes over.
   *
   * Deliberately driven by input events rather than `scroll`: the
   * auto-scroll above emits scroll events of its own, and smooth
   * scrolling emits a stream of them with no reliable end. Anything
   * built on `scroll` has to guess which ones were its own doing.
   * `wheel` and `touchmove` only ever come from the reader.
   *
   * `pointerdown` covers dragging the scrollbar, but only when it lands
   * on the scroll container itself — on a row it is someone clicking a
   * cue, which resumes following rather than suspending it.
   */
  useEffect(() => {
    const list = listEl;
    if (!list) return;
    // The box that gets scrolled is the one the reader can scroll, from
    // anywhere on it — the host's strip and gutters included. The host's
    // scroller is shared with the other tabs, so it only counts while the
    // transcript is the one shown.
    const scroller = scrollingBoxOf(list);
    const shown = () => !list.closest("[hidden]");
    const suspend = () => {
      if (!shown()) return;
      setFollowing(false);
      // The reader has taken their place into their own hands; putting
      // back an older one later would undo that.
      putBackPendingRef.current = false;
    };
    // A finger or a pen does not drag a scrollbar; landing on the box
    // itself is a tap on its padding.
    const suspendOnScrollbar = (event: PointerEvent) => {
      const dragsScrollbars =
        event.pointerType !== "touch" && event.pointerType !== "pen";
      if (dragsScrollbars && event.target === scroller && shown()) {
        setFollowing(false);
      }
    };
    scroller.addEventListener("wheel", suspend, { passive: true });
    scroller.addEventListener("touchmove", suspend, { passive: true });
    scroller.addEventListener("pointerdown", suspendOnScrollbar);
    return () => {
      scroller.removeEventListener("wheel", suspend);
      scroller.removeEventListener("touchmove", suspend);
      scroller.removeEventListener("pointerdown", suspendOnScrollbar);
    };
  }, [listEl]);

  // Current `following` for the save below, which runs from a DOM
  // listener and on unmount — neither of which sees a re-rendered
  // closure.
  const followingRef = useRef(following);
  useEffect(() => {
    followingRef.current = following;
  }, [following]);

  // A source the reader chose that has not answered yet: the rows shown
  // meanwhile are another source's, and a place laid on them would be read
  // back off them, coarser, when the chosen one arrives.
  const settledOf: Record<Source, boolean> = {
    chunks: !loading,
    words: wordsSettled,
    external: externalSettled,
  };
  const availableOf: Record<Source, boolean> = {
    chunks: chunksAvailable,
    words: wordsAvailable,
    external: externalAvailable,
  };
  // Also while it has arrived but is not on screen yet: the switch to it is
  // a render behind its arrival.
  const awaitingChosen =
    chosenSource !== null &&
    source !== chosenSource &&
    (!settledOf[chosenSource] || availableOf[chosenSource]);
  const awaitingChosenRef = useRef(awaitingChosen);
  useEffect(() => {
    awaitingChosenRef.current = awaitingChosen;
  }, [awaitingChosen]);

  // The reader's last seen place in this file, and whether it still has to
  // be put back: after a mount, and after the host hid the panel.
  const placeRef = useRef<TranscriptPlace | null>(null);
  const putBackPendingRef = useRef(false);

  /**
   * While following a cue that is playing, "back" is that cue, not the
   * place the reader last saw: that place was only ever where following
   * had taken them.
   */
  const putBack = useCallback(
    (list: HTMLElement): boolean => {
      if (!putBackPendingRef.current) return false;
      if (list.closest("[hidden]") || awaitingChosenRef.current) return false;
      putBackPendingRef.current = false;
      if (followingRef.current && activeRef.current) {
        scrollActiveIntoView();
      } else if (placeRef.current) {
        applyPlace(list, placeRef.current);
      }
      return true;
    },
    [scrollActiveIntoView],
  );

  /**
   * A cue that changed while the host hid this panel was not scrolled to,
   * and in the sheet another tab may have moved the shared scroller since.
   * Revealing changes nothing the follow effect depends on, so the reveal
   * itself — the list going from no height to some — re-aims.
   */
  useEffect(() => {
    const list = listEl;
    if (!list || typeof ResizeObserver === "undefined") return;
    let height = list.getBoundingClientRect().height;
    const observer = new ResizeObserver((entries) => {
      const next = entries[entries.length - 1]?.contentRect.height ?? 0;
      const hidden = height > 0 && next === 0;
      const revealed = height === 0 && next > 0;
      height = next;
      // In the sheet another tab moves the shared scroller while this one
      // is hidden, so the place has to be put back, not just kept.
      if (hidden) putBackPendingRef.current = true;
      if (!revealed) return;
      if (!putBack(list) && followingRef.current) scrollActiveIntoView();
    });
    observer.observe(list);
    return () => observer.disconnect();
  }, [listEl, scrollActiveIntoView, putBack]);

  const hasCues = cues.length > 0;

  /**
   * Put the reader back where they were.
   *
   * **Keyed on the list element, not on how many cues it holds.** The count
   * changes when the reader switches source, and re-running then would take
   * the place off them. The element is what appears once loading ends,
   * which can be after the count has stopped changing.
   *
   * A file with nothing remembered starts out following: a suspension on
   * the previous file under the same mount says nothing about this one.
   */
  useEffect(() => {
    const list = listEl;
    if (!list) return;
    const saved = recallTranscriptScroll(fileId);
    setFollowing(saved ? saved.following : true);
    followingRef.current = saved ? saved.following : true;
    placeRef.current = saved?.place ?? null;
    putBackPendingRef.current = true;
    putBack(list);
  }, [fileId, listEl, putBack]);

  // A source that arrives later can replace the rows under the reader.
  useEffect(() => {
    if (listEl) putBack(listEl);
  }, [listEl, cues, awaitingChosen, putBack]);

  /**
   * Remember it, because a refetch cannot bring it back.
   *
   * Written as the reader scrolls, and only while the panel is shown and
   * nothing is waiting to be put back: in the sheet the box that scrolls is
   * shared with the other tabs, and a scroll there is not the reader's
   * place in the transcript.
   *
   * The cleanup writes the last place seen rather than reading one: a
   * browser detaches the list before passive cleanups run, and a detached
   * element has no position. It is still needed, because `following` can
   * change with no scroll — clicking a cue resumes it.
   */
  useEffect(() => {
    const list = listEl;
    if (!list) return;
    const scroller = scrollingBoxOf(list);
    const save = () =>
      rememberTranscriptScroll(fileId, {
        place: placeRef.current,
        following: followingRef.current,
        ...(chosenSourceRef.current ? { source: chosenSourceRef.current } : {}),
      });
    const onScroll = () => {
      if (list.closest("[hidden]") || putBackPendingRef.current) return;
      placeRef.current = readPlace(list) ?? placeRef.current;
      save();
    };
    scroller.addEventListener("scroll", onScroll, { passive: true });
    return () => {
      scroller.removeEventListener("scroll", onScroll);
      save();
    };
  }, [fileId, listEl]);

  const resumeFollowing = useCallback(() => {
    setFollowing(true);
    scrollActiveIntoView();
  }, [scrollActiveIntoView]);

  const seekTo = useCallback(
    (time: number) => {
      if (!mediaController) return;
      // Jumping somewhere deliberately is a statement about where the
      // reader wants to be, so it ends any suspension too.
      setFollowing(true);
      mediaController.seek(time);
      mediaController.play();
    },
    [mediaController]
  );

  const captureCue = useCallback(
    (cue: TranscriptChunkItem) => {
      addSourceCapture({
        drive,
        sourceFileId: fileId,
        filename: filename || fileId,
        fileType,
        kind: "transcript",
        locator: {
          seconds: cue.start,
          endSeconds: cue.end,
          label: formatDuration(cue.start),
        },
        quote: cue.text,
      });
    },
    [drive, fileId, fileType, filename],
  );

  const handleRefine = useCallback(async () => {
    if (refining) return;
    setRefining(true);
    try {
      await refineFileTranscript(fileId, drive);
      // Re-fetch so refinedAt renders immediately; the backend
      // processes asynchronously, so this may still show the
      // pre-refine state. WebSocket push lands in a follow-up.
      const res = await getFileTranscript(fileId, drive);
      if (res.available && res.chunks) setWhisperChunks(res.chunks);
    } catch {
      // non-critical — user can retry
    } finally {
      setRefining(false);
    }
  }, [fileId, drive, refining]);

  if (loading || !hasAnything) return null;

  const toggleOptions: { id: Source; label: string; available: boolean }[] = [
    { id: "chunks", label: t("transcriptSourceChunks"), available: chunksAvailable },
    { id: "words", label: t("transcriptSourceWords"), available: wordsAvailable },
    { id: "external", label: t("transcriptSourceExternal"), available: externalAvailable },
  ];
  const visibleOptions = toggleOptions.filter((o) => o.available);
  const showToggle = visibleOptions.length >= 2;

  return (
    <div
      // Fills its column as a flex item, not with `h-full`. The rail's
      // height comes from a max-height clamp rather than a set height,
      // and a percentage height against that resolves to auto — the
      // list then renders at full length and is silently clipped.
      className={fillHeight ? "flex min-h-0 flex-1 flex-col" : undefined}
    >
      {/* The title goes when the host has already written it — the tab
          the reader pressed says "Transcript", and saying it again
          under the button costs a line of a panel whose whole value is
          length. What stays either way is the row's other occupants:
          the language, the count and the two controls are facts about
          this transcript, not a second name for it. The row is never
          empty, because the count is unconditional. */}
      <div className="mb-2 flex items-center gap-2 text-sm text-text-muted">
        {!labelledByHost && (
          <>
            <FileText size={14} />
            <span>{t("transcriptTitle")}</span>
          </>
        )}
        {language && (
          <span className="rounded-lg bg-bg-card px-1.5 py-0.5 text-xs">
            {language}
          </span>
        )}
        <span className="text-xs">({cues.length})</span>
        {showToggle && (
          <div className="ml-2 flex gap-1 text-xs">
            {visibleOptions.map((opt) => (
              <button
                key={opt.id}
                type="button"
                onClick={() => {
                  setChosenSource(opt.id);
                  setSource(opt.id);
                }}
                className={`rounded-lg px-1.5 py-0.5 ${source === opt.id ? "bg-accent text-white" : "bg-bg-card"}`}
              >
                {opt.label}
              </button>
            ))}
          </div>
        )}
        {refineEnabled && source === "chunks" && (
          <div className="ml-auto flex gap-1 text-xs">
            <button
              type="button"
              onClick={handleRefine}
              disabled={refining}
              className="flex items-center gap-1 rounded-lg px-2 py-0.5 text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary disabled:opacity-50"
            >
              <Sparkles size={11} className={refining ? "animate-pulse" : ""} />
              {t("transcriptRefine")}
            </button>
          </div>
        )}
      </div>
      <div
        className={`relative ${fillHeight ? "flex min-h-0 flex-1 flex-col" : ""}`}
      >
        {/* Only offered when there is somewhere to go back to: with no
            cue playing, "current position" means nothing. */}
        {!following && activeIndex >= 0 && (
          // A row of no height, so the list does not move when it appears.
          // Sticky rather than absolute: when the host's scroller encloses
          // the list, an absolute chip scrolls away with the rows and over
          // the host's pinned strip.
          <div
            className="pointer-events-none sticky z-[5] flex h-0 justify-center"
            style={{ top: "var(--inspector-sticky-top, 0px)" }}
          >
            <button
              type="button"
              onClick={resumeFollowing}
              className="pointer-events-auto mt-1 h-fit w-fit rounded-full bg-accent px-3 py-1 text-xs font-medium text-white shadow-card hover:bg-accent-hover focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus-ring"
            >
              {t("transcriptResumeFollowing")}
            </button>
          </div>
        )}
      <div
        ref={attachList}
        className={`space-y-0.5 overflow-y-auto rounded-lg bg-bg-card p-2 ${
          fillHeight ? "min-h-0 flex-1" : "max-h-80"
        }`}
      >
        {cues.map((cue) => {
          const isRefined = Boolean(cue.refinedAt);
          return (
            <div
              key={cue.index}
              data-cue-start={cue.start}
              // 44px of row on a coarse pointer, and 32px on a fine one.
              // The floor is from the mobile sizing rules, so it is about
              // touch and says nothing against a dense desktop list —
              // where 32px already clears the 24px minimum for repeated
              // icon-only controls (hako Prwd_iaXmCjWfY24KjFz2). Applying
              // it everywhere would add 37% of height to a transcript
              // that runs to hundreds of lines, in an environment the
              // rule was not written for.
              //
              // It goes on the row, not on either control: the row is
              // what both of them are asking to be big enough, and a
              // 44px pitch is also what stops the quote button's
              // pseudo-element overlapping its neighbour's.
              className={`group/cue flex w-full items-start rounded-lg text-sm transition-colors hover:bg-bg-primary pointer-coarse:min-h-11 ${
                cue.index === activeIndex
                  ? "bg-accent/10 text-accent"
                  : "text-text-primary"
              }`}
            >
              <button
                type="button"
                ref={cue.index === activeIndex ? activeRef : undefined}
                // Announces the row playback is currently on, and is
                // the only handle a test has on the highlight.
                aria-current={cue.index === activeIndex ? "true" : undefined}
                onClick={() => seekTo(cue.start)}
                // The row's primary action — tapping to move the
                // playhead — so it takes the floor too. `items-start`
                // means it does not inherit the row's height, and a
                // list whose secondary control clears 44px while its
                // main one does not has bought nothing.
                className="flex min-w-0 flex-1 cursor-pointer gap-3 px-2 py-1.5 text-left pointer-coarse:min-h-11"
              >
                <span className="shrink-0 font-mono text-xs text-text-muted">
                  {formatDuration(cue.start)}
                </span>
                <span className="min-w-0 flex-1">{cue.text}</span>
                {isRefined && (
                  <span className="shrink-0 rounded-lg bg-accent-teal/15 px-1.5 py-0.5 text-[10px] font-medium text-accent-teal">
                    {t("transcriptRefinedBadge")}
                  </span>
                )}
              </button>
              {/* One of these per cue, and a transcript runs to
                  hundreds — drawn at all times they read as a grey rule
                  down the right edge of the text they are meant to
                  annotate. Revealed by the row instead, on the three
                  signals that mean someone is working on that row:
                  hovering it, focusing anything inside it (so the
                  keyboard path opens with the pointer one), or having no
                  hover to give in the first place.

                  `opacity-0` and not `hidden` / `invisible`: those two
                  take the button out of the tab order, and
                  `group-focus-within` could then never fire.

                  The accessible name carries the timestamp because the
                  name is all a screen reader gets — several hundred
                  identical "add to capture basket" leave no way to tell
                  which line is about to be quoted (hako
                  Prwd_iaXmCjWfY24KjFz2). It is the only name here: a
                  `title` alongside it becomes the accessible
                  *description*, which NVDA and JAWS read after the name,
                  so the sentence would be announced twice.

                  On a coarse pointer the target grows by overhanging
                  the box rather than by enlarging it: a taller button
                  would raise the row it sits in, and this list is capped
                  at `max-h-80`, so every 12px costs roughly a quarter of
                  the cues on a phone. Vertical space is scarcest exactly
                  where the rule applies.

                  That is 44px in both axes because the row is 44px on
                  a coarse pointer: at a 46px pitch this pseudo-element
                  ends 2px before the next row's begins, so nothing
                  overlaps and no row wins a band of its neighbour's. On
                  a fine pointer the row stays 36px and so does the
                  32px box — the floor is a mobile rule, and a desktop
                  transcript of several hundred lines is not what it was
                  written about.

                  A device reporting `pointer: fine` with `hover: none`
                  — a stylus, some TV browsers — matches neither trigger
                  and reaches the button only by focusing it.
                  `[@media(hover:none)]:opacity-100` does close that, and
                  compiles here; it is left out because it compiles *here*
                  and not in Tailwind 4.3, while `package.json` asks for
                  `^4`. A class that stops emitting on a patch bump fails
                  exactly the way this whole control already failed once:
                  silently invisible. `not-hover` is not an alternative —
                  it emits `:not(:hover)` alongside the media query, which
                  would draw the button on every desktop row the pointer
                  is not over. Both measured against the pinned compiler,
                  not assumed. */}
              <button
                type="button"
                onClick={() => captureCue(cue)}
                aria-label={t("transcriptCaptureCue", {
                  time: formatDuration(cue.start),
                })}
                className="relative m-0.5 inline-flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-lg text-text-muted opacity-0 transition-opacity hover:bg-bg-elevated hover:text-text-primary group-hover/cue:opacity-100 group-focus-within/cue:opacity-100 pointer-coarse:opacity-100 pointer-coarse:before:absolute pointer-coarse:before:-inset-1.5 pointer-coarse:before:content-['']"
              >
                <Quote size={14} />
              </button>
            </div>
          );
        })}
      </div>
      </div>
    </div>
  );
}
