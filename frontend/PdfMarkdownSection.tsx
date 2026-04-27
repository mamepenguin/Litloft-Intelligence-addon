"use client";

/**
 * PdfMarkdownSection — renders the PyMuPDF4LLM-generated Markdown body
 * for an indexed PDF inside the file-detail-sections slot.
 *
 * The section gates on the file's MIME type so it stays hidden for
 * every non-PDF file. PDFs that fell back to fitz extraction (no
 * Markdown persisted) surface 404 from the API and we render a quiet
 * placeholder rather than nothing at all — DESIGN.md §3.3 / §3.4 lay
 * out the read surface conventions, and the parent file-detail layout
 * already provides chrome so MarkdownPreview is mounted with
 * `chrome=false` (matches the Ask answer panel pattern noted in spec
 * §影響範囲 / hako).
 */

import { useEffect, useRef, useState } from "react";
import { useTranslations } from "next-intl";
import { FileText } from "lucide-react";

import { getFile } from "@/lib/api";
import { MarkdownPreview } from "@/components/MarkdownPreview";
import type { FileItem } from "@/types";

import { getPdfMarkdown } from "./api";
import type { PdfMarkdownPayload } from "./api";

interface PdfMarkdownSectionProps {
  fileId: string;
  drive: string;
}

type LoadState =
  | { kind: "loading" }
  | { kind: "ready"; payload: PdfMarkdownPayload }
  | { kind: "unavailable" }
  | { kind: "error" };

function isPdfFile(file: FileItem | null): boolean {
  if (!file) return false;
  const mime = file.mime_type;
  return typeof mime === "string" && mime.toLowerCase() === "application/pdf";
}

export default function PdfMarkdownSection({
  fileId,
  drive,
}: PdfMarkdownSectionProps) {
  const t = useTranslations("file");
  const [file, setFile] = useState<FileItem | null>(null);
  const [fileLoaded, setFileLoaded] = useState(false);
  const [state, setState] = useState<LoadState>({ kind: "loading" });

  // Guard against late responses clobbering fresh state when the
  // parent navigates between files (matches VisualDescriptionSection).
  const requestIdRef = useRef(0);

  useEffect(() => {
    const requestId = requestIdRef.current + 1;
    requestIdRef.current = requestId;
    setFile(null);
    setFileLoaded(false);
    setState({ kind: "loading" });

    (async () => {
      const fileResult = await getFile(fileId).catch(() => null);
      if (requestId !== requestIdRef.current) return;
      setFile(fileResult);
      setFileLoaded(true);
      if (!isPdfFile(fileResult)) {
        // Non-PDFs never render the section; skip the API call.
        setState({ kind: "unavailable" });
        return;
      }
      try {
        const payload = await getPdfMarkdown(fileId, drive);
        if (requestId !== requestIdRef.current) return;
        if (payload === null) {
          setState({ kind: "unavailable" });
        } else {
          setState({ kind: "ready", payload });
        }
      } catch {
        if (requestId !== requestIdRef.current) return;
        setState({ kind: "error" });
      }
    })();
  }, [fileId, drive]);

  // Hide the section entirely until we know the file is a PDF. Other
  // file types must not see "loading…" placeholders.
  if (!fileLoaded) return null;
  if (!isPdfFile(file)) return null;

  return (
    <div>
      <div className="mb-2 flex items-center gap-2 text-sm text-text-muted">
        <FileText size={14} />
        <span>{t("pdfMarkdownTitle")}</span>
        {state.kind === "ready" && (
          <span className="text-xs">
            {t("pdfMarkdownPageCount", { count: state.payload.page_count })}
          </span>
        )}
      </div>

      {state.kind === "loading" && (
        <div
          aria-label={t("pdfMarkdownLoading")}
          role="status"
          className="space-y-2 rounded-lg bg-bg-card p-4"
        >
          <div className="h-3 w-3/4 animate-pulse rounded bg-bg-elevated" />
          <div className="h-3 w-2/3 animate-pulse rounded bg-bg-elevated" />
          <div className="h-3 w-1/2 animate-pulse rounded bg-bg-elevated" />
        </div>
      )}

      {state.kind === "ready" && (
        <div className="rounded-lg bg-bg-card p-4">
          <MarkdownPreview
            source={state.payload.markdown}
            chrome={false}
            mermaid={false}
            className="markdown-segment max-h-[60vh] overflow-y-auto"
          />
        </div>
      )}

      {state.kind === "unavailable" && (
        <p className="text-xs text-text-muted/70">
          {t("pdfMarkdownUnavailable")}
        </p>
      )}

      {state.kind === "error" && (
        <p className="text-xs text-text-muted/70">
          {t("pdfMarkdownError")}
        </p>
      )}
    </div>
  );
}
