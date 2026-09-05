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

/** The `?q=` a destination will actually read, rather than how it was spelt. */
function queryOf(href: string): string | null {
  return new URL(href, "http://x").searchParams.get("q");
}

function hrefs(current: "ask" | "find", query: string, drive = "family") {
  render(<ModeTabs current={current} query={query} drive={drive} />);
  // Two, exactly. Naming the tabs one at a time says nothing about how many
  // there are, and a third destination grown into this row would be invisible
  // to every assertion below.
  expect(screen.getAllByRole("link")).toHaveLength(2);
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
    //
    // Compared after decoding, because what the destination reads is what
    // matters. Pinning the encoded spelling instead makes an equivalent
    // implementation fail: `URLSearchParams` writes `+` for a space, which
    // `useSearchParams()` decodes back to a space, and the first version of
    // this test went red on it while the destination was unchanged.
    const { ask, find } = hrefs("ask", "what is the plot?");
    expect(queryOf(ask!)).toBe("what is the plot?");
    expect(queryOf(find!)).toBe("what is the plot?");
    expect(ask!.split("?")[0]).toBe(ASK);
    expect(find!.split("?")[0]).toBe(FIND);
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
    // The whole string, and nothing after it: a `not.toContain("#")` beside
    // this cannot change any verdict `toBe` does not already reach, which is
    // the "sentence that reads like a second defence" `button-adoption.test.ts`
    // names.
    expect(ask).toBe(`/drive/${drive}/addons/intelligence`);
    expect(find).toBe(`/drive/${drive}/addons/intelligence/find`);
  });

  it("escapes the query too", () => {
    // `&` and `=` would end the parameter early if they went in raw, so the
    // decoded value is the thing to check — and it is checked through the same
    // parser the destination uses.
    const { ask } = hrefs("ask", "a&b=c");
    expect(queryOf(ask!)).toBe("a&b=c");
  });
});
