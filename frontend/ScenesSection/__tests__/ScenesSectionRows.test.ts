import { describe, it, expect } from "vitest";

import { sceneRows } from "@/addons/intelligence/ScenesSection/scenesModel";

describe("SPEC-ADDON-104 I1 — R is the largest whole number of at least 3 for which the slack is at least 0", () => {
  it.each([
    { name: "no slack keeps the drawn rows", drawn: 3, slack: 0, step: 120, rows: 3 },
    { name: "slack one pixel short of a step adds nothing", drawn: 3, slack: 119, step: 120, rows: 3 },
    { name: "slack of exactly one step adds a row", drawn: 3, slack: 120, step: 120, rows: 4 },
    { name: "slack of two and a half steps adds two rows", drawn: 3, slack: 300, step: 120, rows: 5 },
    { name: "a fractional step", drawn: 4, slack: 250.5, step: 125.25, rows: 6 },
    { name: "an overflow of one pixel drops a row", drawn: 5, slack: -1, step: 120, rows: 4 },
    { name: "an overflow of exactly one step drops a row", drawn: 5, slack: -120, step: 120, rows: 4 },
    { name: "an overflow of one step and a pixel drops two rows", drawn: 6, slack: -121, step: 120, rows: 4 },
    { name: "an overflow never goes below 3", drawn: 4, slack: -1, step: 120, rows: 3 },
    { name: "a deep overflow at 3 rows stays at 3", drawn: 3, slack: -900, step: 120, rows: 3 },
    { name: "fewer drawn rows than 3 with little slack is still 3", drawn: 2, slack: 10, step: 120, rows: 3 },
    { name: "fewer drawn rows than 3 with a large slack follows the formula", drawn: 2, slack: 600, step: 120, rows: 7 },
    { name: "one drawn row with a large slack follows the formula", drawn: 1, slack: 480, step: 120, rows: 5 },
  ])("SPEC-ADDON-104 I1 — $name", ({ drawn, slack, step, rows }) => {
    expect(sceneRows(drawn, slack, step)).toBe(rows);
  });

  it("SPEC-ADDON-104 I6 — the R it returns, drawn, measures as the same R", () => {
    const step = 132;
    for (const extra of [0, 1, 2, 5]) {
      for (const leftover of [0, 1, step - 1]) {
        const atThree = extra * step + leftover;
        const rows = sceneRows(3, atThree, step);
        expect(rows).toBe(3 + extra);
        const atRows = atThree - (rows - 3) * step;
        expect(sceneRows(rows, atRows, step)).toBe(rows);
      }
    }
  });
});
