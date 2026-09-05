/**
 * Where the mode row goes, and what it carries there.
 *
 * The component's own doc comment promises two things — that choosing a tab
 * preserves the current query as `?q=`, and that the two tabs are different
 * routes — and until this file nothing checked either. Removing `?q=`,
 * dropping `encodeURIComponent` around the drive, and pointing Find at Ask's
 * own route all left the addon's 432 tests green.
 *
 * The page-level tests assert the row's vocabulary (a link marked
 * `aria-current`, no tablist). They cannot see a destination: a Find tab
 * pointing at Ask still renders a link named "Find" that is not current.
 */

import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import React from "react";

import ModeTabs from "./ModeTabs";

const ASK = "/drive/family/addons/intelligence";
const FIND = "/drive/family/addons/intelligence/find";

function hrefs(current: "ask" | "find", query: string, drive = "family") {
  render(<ModeTabs current={current} query={query} drive={drive} />);
  return {
    ask: screen.getByRole("link", { name: /ask/i }).getAttribute("href"),
    find: screen.getByRole("link", { name: /find/i }).getAttribute("href"),
  };
}

describe("ModeTabs", () => {
  it("sends each tab to its own route", () => {
    // The two destinations differ. Without this, a row whose tabs both point
    // at the page you are already on satisfies every other assertion in the
    // suite — the labels are still right and neither is wrongly current.
    const { ask, find } = hrefs("ask", "");
    expect(ask).toBe(ASK);
    expect(find).toBe(FIND);
  });

  it("carries the current query across the switch", () => {
    // The destination auto-fires its pipeline on mount when `?q=` is
    // non-empty, so losing this loses the question the reader just typed.
    const { ask, find } = hrefs("ask", "what is the plot?");
    expect(ask).toBe(`${ASK}?q=what%20is%20the%20plot%3F`);
    expect(find).toBe(`${FIND}?q=what%20is%20the%20plot%3F`);
  });

  it("carries nothing when the query is only whitespace", () => {
    const { ask, find } = hrefs("find", "   ");
    expect(ask).toBe(ASK);
    expect(find).toBe(FIND);
  });

  it("escapes the drive name", () => {
    // Drive names are user-chosen and reach this component raw. A name with a
    // space or a `#` builds a broken path unescaped, and the failure is a
    // link that silently goes somewhere else rather than an error.
    const { ask, find } = hrefs("ask", "", "家族 の #写真");
    const drive = encodeURIComponent("家族 の #写真");
    expect(ask).toBe(`/drive/${drive}/addons/intelligence`);
    expect(find).toBe(`/drive/${drive}/addons/intelligence/find`);
    expect(ask).not.toContain("#");
    expect(ask).not.toContain(" ");
  });

  it("escapes the query too", () => {
    const { ask } = hrefs("ask", "a&b=c");
    expect(ask).toBe(`${ASK}?q=a%26b%3Dc`);
  });
});
