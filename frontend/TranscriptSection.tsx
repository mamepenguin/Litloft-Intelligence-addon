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

export default function TranscriptSection({ fileId, drive, filename, fileType = "video", mediaController, subtitles = EMPTY_SUBTITLES, fillHeight = false }: TranscriptSectionProps) {
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

  useEffect(() => {
    setLoading(true);
    getFileTranscript(fileId, drive).then((res) => {
      if (res.available && res.chunks && res.chunks.length > 0) {
        setWhisperChunks(res.chunks);
        setWhisperLanguage(res.language || "");
      } else {
        setWhisperChunks([]);
      }
      setLoading(false);
    });
  }, [fileId, drive]);

  useEffect(() => {
    fetch(`/api/addons/intelligence/files/${fileId}/subtitles.vtt`)
      .then((r) => (r.ok ? r.text() : ""))
      .then((text) => setWhisperWordCues(text ? parseVttCues(text) : []))
      .catch(() => setWhisperWordCues([]));
  }, [fileId]);

  useEffect(() => {
    if (subtitles.length === 0) {
      setExternalCues([]);
      return;
    }
    const first = subtitles[0];
    setExternalLanguage(first.language || "");
    fetch(getSubtitleUrl(fileId, first.index))
      .then((r) => (r.ok ? r.text() : ""))
      .then((text) => setExternalCues(text ? parseVttCues(text) : []))
      .catch(() => setExternalCues([]));
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

  if (loading || (!chunksAvailable && !wordsAvailable && !externalAvailable)) return null;

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
      <div className="mb-2 flex items-center gap-2 text-sm text-text-muted">
        <FileText size={14} />
        <span>{t("transcriptTitle")}</span>
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
              className={`group/cue flex w-full items-start rounded-lg text-sm transition-colors hover:bg-bg-primary ${
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
                className="flex min-w-0 flex-1 cursor-pointer gap-3 px-2 py-1.5 text-left"
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
                  Prwd_iaXmCjWfY24KjFz2). The tap target grows to 44px on
                  coarse pointers only: that is the input the 44px rule
                  is about, and spending it on every desktop row would
                  add 12px to each of hundreds of rows. */}
              <button
                type="button"
                onClick={() => captureCue(cue)}
                title={t("transcriptCaptureCue", {
                  time: formatDuration(cue.start),
                })}
                aria-label={t("transcriptCaptureCue", {
                  time: formatDuration(cue.start),
                })}
                className="m-0.5 inline-flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-lg text-text-muted opacity-0 transition-opacity hover:bg-bg-elevated hover:text-text-primary group-hover/cue:opacity-100 group-focus-within/cue:opacity-100 pointer-coarse:h-11 pointer-coarse:w-11 pointer-coarse:opacity-100"
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
