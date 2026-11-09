"use client";

import { useEffect, useLayoutEffect, useMemo, useRef } from "react";

import { MarkdownPreview } from "@/components/MarkdownPreview";
import type { DocumentController } from "@/lib/documentController";
import type { MediaController } from "@/lib/mediaController";

import type { CitationPosition, FileCitation } from "./askStream";
import { citationLabel } from "./citationLabel";

const PILL_CLASS =
  "mx-0.5 inline rounded-full px-1.5 py-0.5 align-baseline text-text-muted transition-colors hover:bg-accent/10 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-focus-ring";
const PLAIN_CLASS = "mx-0.5 inline text-text-muted";
const LINK_SCHEMES = new Set(["http:", "https:", "mailto:"]);

const useIsomorphicLayoutEffect =
  typeof window !== "undefined" ? useLayoutEffect : useEffect;

function jumpAction(
  position: CitationPosition,
  mediaController: MediaController | null,
  documentController: DocumentController | null,
): (() => void) | null {
  if (!position) return null;
  if ("seconds" in position) {
    return mediaController ? () => mediaController.seek(position.seconds) : null;
  }
  if ("page" in position) {
    return documentController
      ? () => documentController.goTo({ page: position.page })
      : null;
  }
  return documentController
    ? () => documentController.goTo({ section: position.section })
    : null;
}

function closingBracket(
  text: string,
  start: number,
  open: string,
  close: string,
): number {
  let depth = 0;
  for (let i = start; i < text.length; i += 1) {
    const char = text[i];
    if (char === "\\") {
      i += 1;
      continue;
    }
    if (char === open) depth += 1;
    else if (char === close) {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

function readImage(
  text: string,
  start: number,
): { label: string; end: number } | null {
  const labelEnd = closingBracket(text, start + 1, "[", "]");
  if (labelEnd < 0) return null;
  const label = text.slice(start + 2, labelEnd);
  const next = text[labelEnd + 1];
  if (next === "(") {
    const end = closingBracket(text, labelEnd + 1, "(", ")");
    return end < 0 ? null : { label, end: end + 1 };
  }
  if (next === "[") {
    const end = closingBracket(text, labelEnd + 1, "[", "]");
    return end < 0 ? null : { label, end: end + 1 };
  }
  return null;
}

function linkScheme(href: string): string | null {
  const match = /^([a-z][a-z0-9+.-]*):/i.exec(href.trim());
  return match ? `${match[1].toLowerCase()}:` : null;
}

function destination(raw: string): string {
  const trimmed = raw.trim();
  if (trimmed.startsWith("<")) {
    const end = trimmed.indexOf(">");
    return end < 0 ? trimmed.slice(1) : trimmed.slice(1, end);
  }
  return trimmed.split(/\s+/)[0] ?? "";
}

function readLink(
  text: string,
  start: number,
): { label: string; href: string; end: number } | null {
  const labelEnd = closingBracket(text, start, "[", "]");
  if (labelEnd < 0) return null;
  if (text[labelEnd + 1] !== "(") return null;
  const destinationEnd = closingBracket(text, labelEnd + 1, "(", ")");
  if (destinationEnd < 0) return null;
  return {
    label: text.slice(start + 1, labelEnd),
    href: destination(text.slice(labelEnd + 2, destinationEnd)),
    end: destinationEnd + 1,
  };
}

/** A destination markdown-it would refuse to make a link from. */
function isRefusedLink(href: string): boolean {
  const scheme = linkScheme(href);
  return scheme !== null && !LINK_SCHEMES.has(scheme);
}

function stripInlineMarkup(line: string): string {
  let out = "";
  let i = 0;
  while (i < line.length) {
    const char = line[i];
    if (char === "\\" && i + 1 < line.length) {
      out += line.slice(i, i + 2);
      i += 2;
      continue;
    }
    if (char === "`") {
      let run = 1;
      while (line[i + run] === "`") run += 1;
      const fence = "`".repeat(run);
      const close = line.indexOf(fence, i + run);
      if (close < 0) {
        out += line.slice(i);
        break;
      }
      out += line.slice(i, close + run);
      i = close + run;
      continue;
    }
    if (char === "!" && line[i + 1] === "[") {
      const image = readImage(line, i);
      if (image) {
        out += image.label;
        i = image.end;
        continue;
      }
    }
    if (char === "[") {
      const link = readLink(line, i);
      if (link && isRefusedLink(link.href)) {
        out += stripInlineMarkup(link.label);
        i = link.end;
        continue;
      }
    }
    out += char;
    i += 1;
  }
  return out;
}

/**
 * Drops the parts of the source the answer must not draw: an image becomes
 * its alternative text, and a link markdown-it refuses to render becomes the
 * words of its label. Both are done before rendering because an `<img>` that
 * reaches the DOM starts its request at once, too early for an effect to take
 * it back.
 */
function withoutImagesAndRefusedLinks(source: string): string {
  if (!source.includes("![") && !source.includes("](")) return source;
  let fence: string | null = null;
  return source
    .split("\n")
    .map((line) => {
      const opening = /^ {0,3}(`{3,}|~{3,})/.exec(line);
      if (fence !== null) {
        if (
          opening &&
          opening[1][0] === fence[0] &&
          opening[1].length >= fence.length
        ) {
          fence = null;
        }
        return line;
      }
      if (opening) {
        fence = opening[1];
        return line;
      }
      return stripInlineMarkup(line);
    })
    .join("\n");
}

/**
 * MarkdownPreview renders every scheme it accepts as an anchor; the answer
 * keeps only http, https and mailto as links, so the rest lose their href.
 */
function normalizeLinks(root: HTMLElement): void {
  for (const anchor of Array.from(root.querySelectorAll("a"))) {
    const scheme = linkScheme(anchor.getAttribute("href") ?? "");
    if (scheme !== null && LINK_SCHEMES.has(scheme)) continue;
    const span = document.createElement("span");
    while (anchor.firstChild) span.appendChild(anchor.firstChild);
    anchor.replaceWith(span);
  }
}

const MARKER = /\[(\d+)\]/g;

/**
 * A pill has to sit inside the block that holds the marker — a heading, a
 * cell, a link — and MarkdownPreview renders its output as one HTML string,
 * so the marker is replaced in that output rather than around it.
 */
function applyCitations(
  root: HTMLElement,
  citations: FileCitation[],
  mediaController: MediaController | null,
  documentController: DocumentController | null,
  onJump: () => void,
): void {
  const positions = new Map(
    citations.map((citation) => [citation.n, citation.position]),
  );
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  const nodes: Text[] = [];
  while (walker.nextNode()) {
    nodes.push(walker.currentNode as Text);
  }

  for (const node of nodes) {
    if (node.parentElement?.closest("code, pre")) continue;
    const text = node.data;
    if (!text.includes("[")) continue;

    const fragment = document.createDocumentFragment();
    let cursor = 0;
    let replaced = false;
    for (const match of text.matchAll(MARKER)) {
      const index = match.index ?? 0;
      if (index > cursor) {
        fragment.appendChild(document.createTextNode(text.slice(cursor, index)));
      }
      cursor = index + match[0].length;
      replaced = true;
      const n = Number(match[1]);
      if (!positions.has(n)) continue;
      const position = positions.get(n) ?? null;
      const label = citationLabel(position);
      if (label === null) continue;
      const action = jumpAction(position, mediaController, documentController);
      if (!action) {
        const span = document.createElement("span");
        span.className = PLAIN_CLASS;
        span.textContent = label;
        fragment.appendChild(span);
        continue;
      }
      const button = document.createElement("button");
      button.type = "button";
      button.className = PILL_CLASS;
      button.textContent = label;
      button.addEventListener("click", () => {
        action();
        onJump();
      });
      fragment.appendChild(button);
    }
    if (!replaced) continue;
    if (cursor < text.length) {
      fragment.appendChild(document.createTextNode(text.slice(cursor)));
    }
    node.replaceWith(fragment);
  }
}

export interface AnswerMarkdownProps {
  source: string;
  citations: FileCitation[] | null;
  mediaController: MediaController | null;
  documentController: DocumentController | null;
  onJump: () => void;
}

export function AnswerMarkdown({
  source,
  citations,
  mediaController,
  documentController,
  onJump,
}: AnswerMarkdownProps) {
  const hostRef = useRef<HTMLDivElement>(null);
  const rendered = useMemo(
    () => withoutImagesAndRefusedLinks(source),
    [source],
  );

  useIsomorphicLayoutEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    normalizeLinks(host);
    if (citations !== null) {
      applyCitations(
        host,
        citations,
        mediaController,
        documentController,
        onJump,
      );
    }
  }, [rendered, citations, mediaController, documentController, onJump]);

  return (
    <div ref={hostRef}>
      <MarkdownPreview
        source={rendered}
        chrome={false}
        mermaid={false}
        showFrontmatter={false}
        className="markdown-segment text-base leading-relaxed text-text-primary"
      />
    </div>
  );
}
