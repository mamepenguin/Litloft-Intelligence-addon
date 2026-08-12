import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/api", () => ({
  fetchJSON: vi.fn(),
}));

import { fetchJSON } from "@/lib/api";
import {
  approveSuggestedChapters,
  dismissSuggestedChapters,
  generateSuggestedChapters,
  getSuggestedChapters,
} from "@/addons/intelligence/api";

describe("suggested chapters API", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(fetchJSON).mockResolvedValue(undefined);
  });

  it("uses the confirmed chapter-suggestions routes and encoded drive header", async () => {
    await getSuggestedChapters("f1", "動画");
    await generateSuggestedChapters("f1", "動画");
    await approveSuggestedChapters("f1", "動画");
    await dismissSuggestedChapters("f1", "動画");

    expect(vi.mocked(fetchJSON).mock.calls).toEqual([
      ["/api/addons/intelligence/files/f1/chapter-suggestions", {
        headers: { "X-Lit-Drive": encodeURIComponent("動画") },
      }],
      ["/api/addons/intelligence/files/f1/chapter-suggestions/generate", {
        method: "POST",
        headers: { "X-Lit-Drive": encodeURIComponent("動画") },
      }],
      ["/api/addons/intelligence/files/f1/chapter-suggestions/approve", {
        method: "POST",
        headers: { "X-Lit-Drive": encodeURIComponent("動画") },
      }],
      ["/api/addons/intelligence/files/f1/chapter-suggestions/dismiss", {
        method: "POST",
        headers: { "X-Lit-Drive": encodeURIComponent("動画") },
      }],
    ]);
  });

  it("does not hide GET failures from the UI", async () => {
    vi.mocked(fetchJSON).mockRejectedValue(new Error("offline"));

    await expect(getSuggestedChapters("f1", "media")).rejects.toThrow("offline");
  });
});
