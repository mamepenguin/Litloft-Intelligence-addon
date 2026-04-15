"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { FileText } from "lucide-react";
import { useTranslations } from "next-intl";

import { getFileTranscript } from "./api";
import type { TranscriptChunkItem } from "./api";
import { formatDuration } from "@/lib/format";
import { getSubtitleUrl } from "@/lib/api";
import type { SubtitleInfo } from "@/types";

interface TranscriptSectionProps {
  fileId: string;
  drive: string;
  videoRef: React.RefObject<HTMLVideoElement | null>;
  subtitles?: SubtitleInfo[];
}

type Source = "whisper" | "external";

function parseVttCues(vtt: string): TranscriptChunkItem[] {
  const lines = vtt.split(/\r?\n/);
  const cues: TranscriptChunkItem[] = [];
  const tsRe = /(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})\.(\d{3})/;
  let current: { start: number; end: number; text: string[] } | null = null;
  let idx = 0;
  for (const raw of lines) {
    const line = raw.trim();
    const m = line.match(tsRe);
    if (m) {
      if (current) {
        cues.push({
          index: idx++,
          start: current.start,
          end: current.end,
          text: current.text.join(" ").trim(),
        });
      }
      const n = m.map((x) => Number(x));
      current = {
        start: n[1] * 3600 + n[2] * 60 + n[3] + n[4] / 1000,
        end: n[5] * 3600 + n[6] * 60 + n[7] + n[8] / 1000,
        text: [],
      };
      continue;
    }
    if (!line) {
      if (current) {
        cues.push({
          index: idx++,
          start: current.start,
          end: current.end,
          text: current.text.join(" ").trim(),
        });
        current = null;
      }
      continue;
    }
    if (line.startsWith("WEBVTT") || line.startsWith("NOTE") || line.startsWith("Language:")) continue;
    if (current) current.text.push(line.replace(/<[^>]+>/g, ""));
  }
  if (current) {
    cues.push({
      index: idx,
      start: current.start,
      end: current.end,
      text: current.text.join(" ").trim(),
    });
  }
  return cues.filter((c) => c.text);
}

export default function TranscriptSection({ fileId, drive, videoRef, subtitles = [] }: TranscriptSectionProps) {
  const t = useTranslations("searchIndex");
  const [whisperChunks, setWhisperChunks] = useState<TranscriptChunkItem[]>([]);
  const [whisperLanguage, setWhisperLanguage] = useState("");
  const [externalCues, setExternalCues] = useState<TranscriptChunkItem[]>([]);
  const [externalLanguage, setExternalLanguage] = useState("");
  const [loading, setLoading] = useState(true);
  const [source, setSource] = useState<Source>("whisper");
  const [activeIndex, setActiveIndex] = useState(-1);
  const activeRef = useRef<HTMLButtonElement>(null);
  const externalAvailable = subtitles.length > 0;
  const whisperAvailable = whisperChunks.length > 0;

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
    if (!externalAvailable) {
      setExternalCues([]);
      return;
    }
    const first = subtitles[0];
    setExternalLanguage(first.language || "");
    fetch(getSubtitleUrl(fileId, first.index))
      .then((r) => (r.ok ? r.text() : ""))
      .then((text) => setExternalCues(text ? parseVttCues(text) : []))
      .catch(() => setExternalCues([]));
  }, [fileId, subtitles, externalAvailable]);

  useEffect(() => {
    if (!whisperAvailable && externalAvailable) setSource("external");
    else if (whisperAvailable && !externalAvailable) setSource("whisper");
  }, [whisperAvailable, externalAvailable]);

  const { chunks, language } = useMemo(() => {
    if (source === "external") return { chunks: externalCues, language: externalLanguage };
    return { chunks: whisperChunks, language: whisperLanguage };
  }, [source, externalCues, externalLanguage, whisperChunks, whisperLanguage]);

  const handleTimeUpdate = useCallback(() => {
    const video = videoRef.current;
    if (!video || chunks.length === 0) return;
    const currentTime = video.currentTime;
    const idx = chunks.findIndex(
      (c) => currentTime >= c.start && currentTime < c.end
    );
    setActiveIndex(idx);
  }, [chunks, videoRef]);

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

  if (loading || (!whisperAvailable && !externalAvailable)) return null;

  const showToggle = whisperAvailable && externalAvailable;

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
        <span className="text-xs">({chunks.length})</span>
        {showToggle && (
          <div className="ml-auto flex gap-1 text-xs">
            <button
              type="button"
              onClick={() => setSource("whisper")}
              className={`rounded px-1.5 py-0.5 ${source === "whisper" ? "bg-accent text-white" : "bg-bg-card"}`}
            >
              {t("transcriptSourceWhisper")}
            </button>
            <button
              type="button"
              onClick={() => setSource("external")}
              className={`rounded px-1.5 py-0.5 ${source === "external" ? "bg-accent text-white" : "bg-bg-card"}`}
            >
              {t("transcriptSourceExternal")}
            </button>
          </div>
        )}
      </div>
      <div className="max-h-80 space-y-0.5 overflow-y-auto rounded-lg bg-bg-card p-2">
        {chunks.map((chunk) => (
          <button
            key={chunk.index}
            ref={chunk.index === activeIndex ? activeRef : undefined}
            onClick={() => seekTo(chunk.start)}
            className={`flex w-full cursor-pointer gap-3 rounded px-2 py-1.5 text-left text-sm transition-colors hover:bg-bg-primary ${
              chunk.index === activeIndex
                ? "bg-accent/10 text-accent"
                : "text-text-primary"
            }`}
          >
            <span className="shrink-0 font-mono text-xs text-text-muted">
              {formatDuration(chunk.start)}
            </span>
            <span className="min-w-0 flex-1">{chunk.text}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
