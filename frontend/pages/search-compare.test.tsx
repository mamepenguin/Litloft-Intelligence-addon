import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { COMPOSITION_GRACE_MS } from "@/lib/ime";

vi.mock("@/components/CurrentDriveProvider", () => ({
  useCurrentDrive: () => "family",
}));

const searchCompareMock = vi.hoisted(() =>
  vi.fn(async () => ({ source_counts: null })),
);
vi.mock("@/addons/intelligence/api", async () => {
  const actual = await vi.importActual<
    typeof import("@/addons/intelligence/api")
  >("@/addons/intelligence/api");
  return { ...actual, searchCompare: searchCompareMock };
});

import SearchComparePage from "@/addons/intelligence/pages/search-compare";

afterEach(() => {
  cleanup();
  searchCompareMock.mockClear();
  vi.restoreAllMocks();
});

function renderWithConvertedQuery(query: string) {
  render(<SearchComparePage />);
  const input = screen.getByPlaceholderText("Search query...") as HTMLInputElement;
  fireEvent.compositionStart(input);
  fireEvent.change(input, { target: { value: query } });
  fireEvent.compositionEnd(input, { data: query });
  return input;
}

describe("SearchComparePage IME composition", () => {
  it("does not search on the Enter that confirms a conversion", () => {
    const now = vi.spyOn(Date, "now").mockReturnValue(1_000_000);
    const input = renderWithConvertedQuery("会議の記録");
    now.mockReturnValue(1_000_000 + COMPOSITION_GRACE_MS - 1);

    fireEvent.keyDown(input, { key: "Enter", keyCode: 13 });

    expect(searchCompareMock).not.toHaveBeenCalled();
  });

  it("does not search on an Enter the IME still owns", () => {
    const input = renderWithConvertedQuery("会議の記録");
    fireEvent.compositionStart(input);

    fireEvent.keyDown(input, { key: "Enter", isComposing: true });
    fireEvent.keyDown(input, { key: "Enter", keyCode: 229 });

    expect(searchCompareMock).not.toHaveBeenCalled();
  });

  it("searches once on an Enter pressed after the grace window", async () => {
    const now = vi.spyOn(Date, "now").mockReturnValue(1_000_000);
    const input = renderWithConvertedQuery("会議の記録");
    now.mockReturnValue(1_000_000 + COMPOSITION_GRACE_MS);

    fireEvent.keyDown(input, { key: "Enter", keyCode: 13 });

    expect(searchCompareMock).toHaveBeenCalledTimes(1);
    expect(searchCompareMock).toHaveBeenCalledWith("会議の記録", "family", {
      limit: 30,
    });
    await screen.findByText("RRF (Reciprocal Rank Fusion)", { exact: false });
  });
});
