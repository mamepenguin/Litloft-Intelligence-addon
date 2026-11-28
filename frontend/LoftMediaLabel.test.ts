/**
 * SPEC-ADDON-193 I15: the loft_media queue row is provider-neutral in both
 * languages.
 */

import { describe, it, expect } from "vitest";

import en from "@/addons/intelligence/messages/en.json";
import ja from "@/addons/intelligence/messages/ja.json";

describe("IndexStatusWidget loft_media label (SPEC-ADDON-193 I15)", () => {
  it("names the linked video in Japanese", () => {
    expect(ja.semanticSearch.tasks.loft_media.label).toBe("リンク先の映像解析");
  });

  it("names the linked video in English", () => {
    expect(en.semanticSearch.tasks.loft_media.label).toBe("Linked video analysis");
  });
});
