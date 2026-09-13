import { describe, expect, it } from "vitest";

import en from "./messages/en.json";
import ja from "./messages/ja.json";

describe("the Ask navigation label", () => {
  it("is carried under the key the manifest names, in both locales", () => {
    expect(en.intelligence.nav.label).toBe("Ask");
    expect(ja.intelligence.nav.label).toBe("質問");
  });
});
