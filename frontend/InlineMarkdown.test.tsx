/**
 * Security + behavioural contract tests for the inline-only Markdown
 * renderer used by the detailed-summary table cells. InlineMarkdown
 * runs markdown-it's ``renderInline`` (no block grammar) and
 * DOMPurifies the result — this test suite nails down both properties
 * so a future markdown-it / DOMPurify upgrade can't silently
 * regress either.
 */

import { describe, it, expect } from "vitest";
import { render } from "@testing-library/react";
import React from "react";

import { InlineMarkdown } from "@/addons/intelligence/InlineMarkdown";

describe("InlineMarkdown", () => {
  it("renders inline emphasis markers as HTML", () => {
    const { container } = render(<InlineMarkdown source="**bold** and *em* and `code`" />);
    const html = container.innerHTML;
    expect(html).toContain("<strong>bold</strong>");
    expect(html).toContain("<em>em</em>");
    expect(html).toContain("<code>code</code>");
  });

  it("does NOT produce block elements for block markdown", () => {
    // ATX headings and list markers are block grammar; renderInline
    // must leave them as literal characters so nothing becomes an
    // <h1> / <ul> / <li> that would break table-cell layout.
    const { container } = render(
      <InlineMarkdown source={"# Heading\n\n- item 1\n- item 2"} />,
    );
    expect(container.querySelector("h1")).toBeNull();
    expect(container.querySelector("ul")).toBeNull();
    expect(container.querySelector("ol")).toBeNull();
    expect(container.querySelector("li")).toBeNull();
    expect(container.querySelector("p")).toBeNull();
  });

  it("strips <script> / <iframe> via DOMPurify", () => {
    // markdown-it with html:false already ignores raw HTML, but the
    // DOMPurify layer is a second guard — verify both are active by
    // feeding an already-HTML payload. (linkify may or may not pick
    // it up; either way the tags must NOT survive.)
    const { container } = render(
      <InlineMarkdown source={"safe <script>alert(1)</script> <iframe src=x></iframe> text"} />,
    );
    expect(container.querySelector("script")).toBeNull();
    expect(container.querySelector("iframe")).toBeNull();
    // The literal "safe" / "text" content still renders.
    expect(container.textContent).toContain("safe");
    expect(container.textContent).toContain("text");
  });

  it("neutralises javascript: / data: link hrefs and hardens legit links with target+rel", () => {
    const { container } = render(
      <InlineMarkdown source="[x](javascript:alert(1)) [y](data:text/html,hi) [ok](https://example.com)" />,
    );
    // markdown-it's built-in validateLink rejects javascript: / data:
    // URLs outright — the link syntax decays to plain text and no
    // anchor is produced. Our own link_open hook is a belt-and-braces
    // layer for any future validateLink relaxation. Either way the
    // contract we care about is: NO anchor in the DOM ever carries a
    // javascript: or data: href.
    const anchors = container.querySelectorAll("a");
    for (const a of anchors) {
      const href = (a.getAttribute("href") ?? "").trim().toLowerCase();
      expect(href.startsWith("javascript:")).toBe(false);
      expect(href.startsWith("data:")).toBe(false);
      expect(a.getAttribute("target")).toBe("_blank");
      expect(a.getAttribute("rel")).toBe("noopener noreferrer");
    }
    // The legit https link must survive with its original URL.
    const legit = Array.from(anchors).find(
      (a) => a.getAttribute("href") === "https://example.com",
    );
    expect(legit).toBeTruthy();
  });
});
