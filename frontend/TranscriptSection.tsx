"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { FileText, Sparkles } from "lucide-react";
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

interface TranscriptSectionProps {
  fileId: string;
  drive: string;
  videoRef: React.RefObject<HTMLVideoElement | null>;
  subtitles?: SubtitleInfo[];
}

type Source = "chunks" | "words" | "external";

function parseVttCues(vtt: string): TranscriptChunkItem[] {
  const lines = vtt.split(/\r?\n/);
  const cues: TranscriptChunkItem[] = [];
  const tsRe = /(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})\.(\d{3})/;
  let current: { start: number; end: number; text: string[] } | null = null;
  let idx = 0;
  const flush = () => {
    if (current) {
      cues.push({
        index: idx++,
        start: current.start,
        end: current.end,
        text: current.text.join(" ").trim(),
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

export default function TranscriptSection({ fileId, drive, videoRef, subtitles = [] }: TranscriptSectionProps) {
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
  const activeRef = useRef<HTMLButtonElement>(null);

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

  const handleTimeUpdate = useCallback(() => {
    const video = videoRef.current;
    if (!video || cues.length === 0) return;
    const currentTime = video.currentTime;
    const idx = cues.findIndex(
      (c) => currentTime >= c.start && currentTime < c.end
    );
    setActiveIndex(idx);
  }, [cues, videoRef]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    video.addEventListener("timeupdate", handleTimeUpdate);
    return () => video.removeEventListener("timeupdate", handleTimeUpdate);
  }, [videoRef, handleTimeUpdate]);

  useEffect(() => {
    if (activeRef.current) {
      activeRef.current.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  }, [activeIndex]);

  const seekTo = useCallback(
    (time: number) => {
      const video = videoRef.current;
      if (video) {
        video.currentTime = time;
      }
    },
    [videoRef]
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
    <div>
      <div className="mb-2 flex items-center gap-2 text-sm text-text-muted">
        <FileText size={14} />
        <span>{t("transcriptTitle")}</span>
        {language && (
          <span className="rounded bg-bg-card px-1.5 py-0.5 text-xs">
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
                className={`rounded px-1.5 py-0.5 ${source === opt.id ? "bg-accent text-white" : "bg-bg-card"}`}
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
              className="flex items-center gap-1 rounded px-2 py-0.5 text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary disabled:opacity-50"
            >
              <Sparkles size={11} className={refining ? "animate-pulse" : ""} />
              {t("transcriptRefine")}
            </button>
          </div>
        )}
      </div>
      <div className="max-h-80 space-y-0.5 overflow-y-auto rounded-lg bg-bg-card p-2">
        {cues.map((cue) => {
          const isRefined = Boolean(cue.refinedAt);
          return (
            <button
              key={cue.index}
              ref={cue.index === activeIndex ? activeRef : undefined}
              onClick={() => seekTo(cue.start)}
              className={`flex w-full cursor-pointer gap-3 rounded px-2 py-1.5 text-left text-sm transition-colors hover:bg-bg-primary ${
                cue.index === activeIndex
                  ? "bg-accent/10 text-accent"
                  : "text-text-primary"
              }`}
            >
              <span className="shrink-0 font-mono text-xs text-text-muted">
                {formatDuration(cue.start)}
              </span>
              <span className="min-w-0 flex-1">{cue.text}</span>
              {isRefined && (
                <span className="shrink-0 rounded bg-accent-teal/15 px-1.5 py-0.5 text-[10px] font-medium text-accent-teal">
                  {t("transcriptRefinedBadge")}
                </span>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}
