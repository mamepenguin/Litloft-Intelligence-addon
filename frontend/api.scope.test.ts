/** SPEC-ADDON-054: the Ask client puts the scope on the wire and reads it back. */

import { afterEach, describe, expect, it, vi } from "vitest";

import { askQuestionStream, parseSseFrame } from "./api";
import type { AskScopePayload } from "./api";

const SCOPE: AskScopePayload = {
  folders: ["Work"],
  collections: ["読書 メモ"],
  tags: ["会議"],
  kinds: ["pdf"],
};

function frame(name: string, data: unknown): string {
  return `event: ${name}\ndata: ${JSON.stringify(data)}`;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("SPEC-ADDON-054 the Ask client wire", () => {
  it("parses a scope event into its four lists and count", () => {
    expect(
      parseSseFrame(
        frame("scope", { scope: SCOPE, matched_file_count: 2 }),
      ),
    ).toEqual({
      kind: "scope",
      scope: SCOPE,
      matched_file_count: 2,
    });
  });

  it("parses scope_empty on the done event", () => {
    expect(
      parseSseFrame(
        frame("done", {
          retrieved_count: 0,
          scope_empty: true,
          took_ms: 3,
        }),
      ),
    ).toEqual({
      kind: "done",
      retrieved_count: 0,
      scope_empty: true,
      took_ms: 3,
      error: undefined,
    });
  });

  it("puts scope in the request body only when it was given", async () => {
    const fetchMock = vi.fn(async () => ({ ok: true, body: null }));
    vi.stubGlobal("fetch", fetchMock);

    await askQuestionStream("a question", "family", { scope: SCOPE }).next();
    const scopedBody = JSON.parse(
      (fetchMock.mock.calls[0][1] as RequestInit).body as string,
    ) as Record<string, unknown>;
    expect(scopedBody.scope).toEqual(SCOPE);

    await askQuestionStream("a question", "family").next();
    const unscopedBody = JSON.parse(
      (fetchMock.mock.calls[1][1] as RequestInit).body as string,
    ) as Record<string, unknown>;
    expect("scope" in unscopedBody).toBe(false);
  });
});
