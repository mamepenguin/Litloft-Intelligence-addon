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
  const [externalLanguage, setExternalLanguage] = useState("");
  const [loading, setLoading] = useState(true);
  const [source, setSource] = useState<Source>("chunks");
  const [activeIndex, setActiveIndex] = useState(-1);
  // Whether the highlight is still allowed to drag the list around.
  // Reading ahead has to win over following, or the reader is pulled
  // back every few seconds.
  const [following, setFollowing] = useState(true);
  const activeRef = useRef<HTMLButtonElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

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
    fetch(`/api/addons/intelligence/files/${fileId}/subtitles.vtt`)
      .then((r) => (r.ok ? r.text() : ""))
      .then((text) => {
        if (!cancelled) setWhisperWordCues(text ? parseVttCues(text) : []);
      })
      .catch(() => {
        if (!cancelled) setWhisperWordCues([]);
      });
    return () => {
      cancelled = true;
    };
  }, [fileId]);

  useEffect(() => {
    if (subtitles.length === 0) {
      setExternalCues([]);
      return;
    }
    let cancelled = false;
    const first = subtitles[0];
    setExternalLanguage(first.language || "");
    fetch(getSubtitleUrl(fileId, first.index))
      .then((r) => (r.ok ? r.text() : ""))
      .then((text) => {
        if (!cancelled) setExternalCues(text ? parseVttCues(text) : []);
      })
      .catch(() => {
        if (!cancelled) setExternalCues([]);
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
    if (!available.includes(source)) setSource(available[0]);
  }, [chunksAvailable, wordsAvailable, externalAvailable, source]);

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
    // Scroll only the transcript container — avoid scrollIntoView, which
    // bubbles up and moves the page away from the video.
    const listRect = list.getBoundingClientRect();
    const targetRect = target.getBoundingClientRect();
    const above = targetRect.top < listRect.top;
    const below = targetRect.bottom > listRect.bottom;
    if (!above && !below) return;
    const targetOffset = targetRect.top - listRect.top + list.scrollTop;
    const nextTop = targetOffset - (list.clientHeight - target.clientHeight) / 2;
    list.scrollTo({ top: nextTop, behavior: "smooth" });
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
    const list = listRef.current;
    if (!list) return;
    const suspend = () => setFollowing(false);
    const suspendOnScrollbar = (event: PointerEvent) => {
      if (event.target === list) setFollowing(false);
    };
    list.addEventListener("wheel", suspend, { passive: true });
    list.addEventListener("touchmove", suspend, { passive: true });
    list.addEventListener("pointerdown", suspendOnScrollbar);
    return () => {
      list.removeEventListener("wheel", suspend);
      list.removeEventListener("touchmove", suspend);
      list.removeEventListener("pointerdown", suspendOnScrollbar);
    };
  }, [cues.length]);

  // Current `following` for the save below, which runs from a DOM
  // listener and on unmount — neither of which sees a re-rendered
  // closure.
  const followingRef = useRef(following);
  useEffect(() => {
    followingRef.current = following;
  }, [following]);

  const hasCues = cues.length > 0;

  /**
   * Put the reader back where they were.
   *
   * Waits for cues because `scrollTop` on an empty list is silently
   * clamped to 0, so restoring before they render restores nothing.
   *
   * **Keyed on whether there is a list, not on how long it is.** The
   * count changes when the reader switches source, and re-running then
   * would take the position off them and pin them to an offset measured
   * against a list that no longer exists — at best a no-op, at worst
   * past the end of a shorter one. After the first restore the position
   * is the reader's.
   */
  useEffect(() => {
    const list = listRef.current;
    if (!list || !hasCues) return;
    const saved = recallTranscriptScroll(fileId);
    if (!saved) return;
    // Both, together. Not an ordering constraint — `setFollowing` is a
    // state setter queued for the next render, so the auto-scroll effect
    // sees the restored value whichever line runs first. It is that
    // restoring the offset *without* it would hand the reader back their
    // place and then, a second later, drag them to the cue that is
    // playing: the state they left by scrolling away from it.
    list.scrollTop = saved.top;
    setFollowing(saved.following);
  }, [fileId, hasCues]);

  /**
   * Remember it, because a refetch cannot bring it back.
   *
   * Everything else this panel holds is re-derived when it mounts again
   * — the cues, the language, the highlight. Where the reader had got
   * to is not a fact about the file, so nothing re-derives it.
   *
   * Two strands, and each covers what the other cannot.
   *
   * The `scroll` listener writes it down as it happens. That is the one
   * that matters in a browser: `useEffect` cleanups are passive, so on
   * unmount they run *after* React has detached the subtree, and
   * `scrollTop` on a detached element reads 0. jsdom keeps the value,
   * which is why a test cannot show this — the same class of blind spot
   * `mediaDetailTheaterCss.test.ts` exists for.
   *
   * The cleanup save covers the reverse: `following` can change with no
   * scroll of the reader's — clicking a cue resumes it — and there is
   * no event for that. It is also what saves file A's position when the
   * host swaps the file under one mount rather than unmounting.
   *
   * The auto-scroll emits scroll events of its own and that is fine:
   * unlike the follow-suspension above, this does not care who moved the
   * list, only where it is now.
   */
  useEffect(() => {
    const list = listRef.current;
    if (!list || !hasCues) return;
    const save = () =>
      rememberTranscriptScroll(fileId, {
        top: list.scrollTop,
        following: followingRef.current,
      });
    list.addEventListener("scroll", save, { passive: true });
    return () => {
      list.removeEventListener("scroll", save);
      save();
    };
  }, [fileId, hasCues]);

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
                onClick={() => setSource(opt.id)}
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
          <button
            type="button"
            onClick={resumeFollowing}
            className="absolute inset-x-0 top-1 z-10 mx-auto w-fit rounded-full bg-accent px-3 py-1 text-xs font-medium text-white shadow-card hover:bg-accent-hover focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus-ring"
          >
            {t("transcriptResumeFollowing")}
          </button>
        )}
      <div
        ref={listRef}
        className={`space-y-0.5 overflow-y-auto rounded-lg bg-bg-card p-2 ${
          fillHeight ? "min-h-0 flex-1" : "max-h-80"
        }`}
      >
        {cues.map((cue) => {
          const isRefined = Boolean(cue.refinedAt);
          return (
            <div
              key={cue.index}
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
