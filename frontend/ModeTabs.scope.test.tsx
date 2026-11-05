/** SPEC-ADDON-064 step 11 (H8): ModeTabs carries the scope on both links. */

import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";

import ModeTabs from "@/addons/intelligence/ModeTabs";

function hrefs() {
  return screen.getAllByRole("link").map((link) => link.getAttribute("href")!);
}

function query(url: string): URLSearchParams {
  return new URL(url, "http://localhost").searchParams;
}

describe("SPEC-ADDON-064 ModeTabs scope", () => {
  it("puts the scope after the question on both links", () => {
    render(
      <ModeTabs
        current="ask"
        query="cats"
        drive="family"
        scope="path:Work"
      />,
    );

    for (const url of hrefs()) {
      expect(query(url).get("q")).toBe("cats");
      expect(query(url).get("scope")).toBe("path:Work");
    }
  });

  it("uses ?scope= when there is no question", () => {
    render(
      <ModeTabs current="find" query="" drive="family" scope="tag:会議" />,
    );

    for (const url of hrefs()) {
      expect(query(url).get("scope")).toBe("tag:会議");
    }
  });
});
