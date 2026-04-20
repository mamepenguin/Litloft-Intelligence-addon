"use client";

/**
 * Inline-only Markdown renderer for contexts where a block element
 * would break the surrounding layout — most notably table cells inside
 * DetailedSummarySection, where ``<p>`` / ``<ul>`` would split the cell
 * mid-flow. Runs markdown-it's ``renderInline`` (no block grammar —
 * strong / em / code / link / image / etc. only) then DOMPurifies the
 * result to strip any stray <script> / <iframe> that a future
 * markdown-it upgrade could leak. Links are forced to
 * ``rel="noopener noreferrer"`` + ``target="_blank"`` to match
 * MarkdownPreview.tsx's hardening.
 */

import { useMemo } from "react";
import MarkdownIt from "markdown-it";
import DOMPurify from "isomorphic-dompurify";

const md = new MarkdownIt({
  html: false,
  linkify: true,
  typographer: false,
  breaks: false,
});

const defaultLinkRender =
  md.renderer.rules.link_open ??
  ((tokens, idx, options, _env, self) => self.renderToken(tokens, idx, options));

md.renderer.rules.link_open = function (tokens, idx, options, env, self) {
  const token = tokens[idx];
  const href = token.attrGet("href") ?? "";
  const lower = href.trim().toLowerCase();
  if (lower.startsWith("javascript:") || lower.startsWith("data:")) {
    token.attrSet("href", "#");
  }
  token.attrSet("target", "_blank");
  token.attrSet("rel", "noopener noreferrer");
  return defaultLinkRender(tokens, idx, options, env, self);
};

export function InlineMarkdown({ source }: { source: string }) {
  const html = useMemo(() => {
    const raw = md.renderInline(source);
    return DOMPurify.sanitize(raw, {
      FORBID_TAGS: ["script", "iframe", "object", "embed", "form", "style"],
      FORBID_ATTR: ["onload", "onerror", "onclick", "onmouseover"],
      ADD_ATTR: ["target"],
    });
  }, [source]);

  return <span dangerouslySetInnerHTML={{ __html: html }} />;
}
