import { describe, it, expect } from "vitest";

import { highlightSegments, passageWindow } from "./passageText";

// The two pairs from the shipped screenshot that started this redesign.
// A chunk begins wherever the chunker cut, which is routinely mid-word:
// `回対象` is the tail of `回転対象`.
const MID_WORD_OPENING =
  "回対象になっています。対角線を軸として120°回転させると元と同じ立法体の" +
  "見た目になりますがこの時XYZの３つの軸";
const TERM_AT_THE_TAIL =
  "除きますこれは３つの要素を順に入れ替えるようなものですねそして同じ性質を" +
  "持つ８つの回転が立方体にもありますそれぞれの対角線について";
// Whisper leaves long stretches with no terminator at all.
const UNPUNCTUATED =
  "合うように働いてどれだけ遠くのスリットでも位相のずれが一定の範囲に収まり" +
  "続けることが知られていますこうした方向では無数のスリットからの波がほとんど" +
  "揃って届くのでスクリーンに";

describe("passageWindow", () => {
  describe("with no terms", () => {
    it("drops the partial sentence a chunk boundary left behind", () => {
      const { text, truncatedStart } = passageWindow(MID_WORD_OPENING, []);

      // `回対象` is a severed word. Two clamped lines that open on it are
      // what made the shipped section unreadable.
      expect(text.startsWith("対角線を軸として")).toBe(true);
      expect(truncatedStart).toBe(true);
    });

    it("drops a one-character orphan", () => {
      const { text } = passageWindow(
        "ね。なのでこの点では強め合うことになります。",
        [],
      );

      expect(text.startsWith("なのでこの点では")).toBe(true);
    });

    it("starts at zero when there is no terminator to snap to", () => {
      const { text, truncatedStart } = passageWindow(UNPUNCTUATED, []);

      // Never invent a boundary: an unpunctuated transcript has none.
      expect(text).toBe(UNPUNCTUATED);
      expect(truncatedStart).toBe(false);
    });

    it("leaves a passage that already opens on a sentence alone", () => {
      const clean = "対角線を軸として回転させると元と同じ立方体になります。";

      expect(passageWindow(clean, [])).toEqual({
        text: clean,
        truncatedStart: false,
      });
    });

    it("does not snap past the end into an empty remainder", () => {
      const trailing = "これで終わりです。";

      expect(passageWindow(trailing, []).text).toBe(trailing);
    });
  });

  describe("with terms", () => {
    it("opens near a term buried at the end", () => {
      const { text, truncatedStart } = passageWindow(TERM_AT_THE_TAIL, [
        "対角線",
      ]);

      // The whole reason windowing exists: `対角線` sits past the point a
      // head-anchored clamp would ever reach, so highlighting alone would
      // have changed nothing on this row.
      expect(text).toContain("対角線");
      expect(text.length).toBeLessThan(TERM_AT_THE_TAIL.length);
      expect(truncatedStart).toBe(true);
    });

    it("keeps some run-up before the term", () => {
      const { text } = passageWindow(TERM_AT_THE_TAIL, ["対角線"]);

      // A term flush against the left edge reads as a fragment, not a
      // sentence.
      expect(text.indexOf("対角線")).toBeGreaterThan(0);
    });

    it("stays at zero when the term is already near the head", () => {
      const { text, truncatedStart } = passageWindow(
        "対角線を軸として回転させると元と同じ立方体になります。",
        ["対角線"],
      );

      expect(truncatedStart).toBe(false);
      expect(text.startsWith("対角線")).toBe(true);
    });

    it("never snaps forward past the term it centred on", () => {
      // Terminators sit between the window's start and the match, so a
      // greedy snap would cut off the very word the row is about.
      const text =
        "はい。そうですね。ええ。まったくそのとおりです。ところで立方体の対角線の話ですが";
      const out = passageWindow(text, ["対角線"]);

      expect(out.text).toContain("対角線");
    });

    it("uses the first occurrence when a term repeats", () => {
      const text =
        "スリットの話をします。" +
        "あいだにいろいろな説明が入りますがここでは省略します。" +
        "もう一度スリットに戻ります。";
      const out = passageWindow(text, ["スリット"]);

      expect(out.text.startsWith("スリット")).toBe(true);
      expect(out.truncatedStart).toBe(false);
    });

    it("falls back to the no-term behaviour when no term is present", () => {
      expect(passageWindow(MID_WORD_OPENING, ["まったく無関係な語"])).toEqual(
        passageWindow(MID_WORD_OPENING, []),
      );
    });

    it("prefers the earliest term, not the first one listed", () => {
      const text = "立方体の話からはじめて、あとのほうで対角線の話をします。";
      const out = passageWindow(text, ["対角線", "立方体"]);

      expect(out.text).toContain("立方体");
      expect(out.truncatedStart).toBe(false);
    });
  });

  describe("latin punctuation", () => {
    it("treats a period as a boundary only when a space follows", () => {
      const { text } = passageWindow(
        "ing chunks. The claim it was part of is on the other side.",
        [],
      );

      expect(text.startsWith("The claim")).toBe(true);
    });

    it("does not break on a decimal point", () => {
      const text = "3.14 is the ratio we keep coming back to";

      expect(passageWindow(text, []).text).toBe(text);
    });

    it("sees a sentence that ends inside a quotation mark", () => {
      const { text } = passageWindow(
        'was part of." The next sentence carries the claim it belonged to.',
        [],
      );

      // The terminator is not the last character of the sentence when a
      // closer follows it, so testing only the character immediately
      // after the period leaves the severed prefix on screen.
      expect(text.startsWith("The next sentence")).toBe(true);
    });
  });
});


describe("highlightSegments", () => {
  it("marks every occurrence of a term", () => {
    const out = highlightSegments("回転と対角線と回転の話", ["回転"]);

    expect(out.filter((s) => s.marked).map((s) => s.text)).toEqual([
      "回転",
      "回転",
    ]);
    expect(out.map((s) => s.text).join("")).toBe("回転と対角線と回転の話");
  });

  it("never lets a short term split a longer one", () => {
    // `回転` is inside `回転対称`. Marking the short one first would cut
    // the long one in half and leave `対称` unmarked beside it.
    const out = highlightSegments("回転対称の話", ["回転", "回転対称"]);

    expect(out.filter((s) => s.marked).map((s) => s.text)).toEqual(["回転対称"]);
  });

  it("returns the passage untouched when there are no terms", () => {
    const out = highlightSegments("対角線の話", []);

    expect(out).toEqual([{ text: "対角線の話", marked: false }]);
  });

  it("ignores a term that is not present", () => {
    const out = highlightSegments("対角線の話", ["立方体"]);

    expect(out.some((s) => s.marked)).toBe(false);
  });

  it("preserves the passage byte for byte", () => {
    // The row's whole guarantee: highlighting is a display choice, not
    // an edit.
    const text = "同じ性質を持つ8つの回転が立方体にもあり対角線について";
    const out = highlightSegments(text, ["対角線", "回転", "立方体"]);

    expect(out.map((s) => s.text).join("")).toBe(text);
  });

  it("matches case-insensitively but shows the passage's own casing", () => {
    const out = highlightSegments("これは Diagonal の話", ["diagonal"]);

    expect(out.filter((s) => s.marked).map((s) => s.text)).toEqual(["Diagonal"]);
  });
});
