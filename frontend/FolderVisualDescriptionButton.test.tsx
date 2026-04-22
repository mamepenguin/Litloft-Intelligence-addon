/**
 * Tests for FolderVisualDescriptionButton — confirm dialog + 413 handling.
 *
 * Spec: docs/superpowers/specs/2026-04-23-intelligence-vision-describe.md §UI
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor, act } from "@testing-library/react";
import { NextIntlClientProvider } from "next-intl";

vi.mock("@/addons/intelligence/api", () => ({
  generateFolderVisualDescription: vi.fn(),
}));

import FolderVisualDescriptionButton from "@/addons/intelligence/FolderVisualDescriptionButton";
import { generateFolderVisualDescription } from "@/addons/intelligence/api";

function renderButton(fileIds: string[] = ["f1", "f2"]) {
  return render(
    <NextIntlClientProvider locale="en" messages={{}}>
      <FolderVisualDescriptionButton
        drive="family"
        path="photos/2024"
        fileIds={fileIds}
      />
    </NextIntlClientProvider>,
  );
}

describe("FolderVisualDescriptionButton", () => {
  const originalConfirm = window.confirm;

  beforeEach(() => {
    vi.clearAllMocks();
    window.confirm = vi.fn(() => true);
  });

  afterEach(() => {
    window.confirm = originalConfirm;
  });

  it("renders nothing when fileIds is empty", () => {
    const { container } = renderButton([]);
    expect(container).toBeEmptyDOMElement();
  });

  it("confirms then posts with drive and path", async () => {
    (generateFolderVisualDescription as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValue({ queued: 7, file_ids: ["f1", "f2"] });
    renderButton();
    const btn = screen.getByRole("button");
    await act(async () => {
      fireEvent.click(btn);
    });
    expect(window.confirm).toHaveBeenCalled();
    await waitFor(() => {
      expect(generateFolderVisualDescription).toHaveBeenCalledWith(
        "family",
        "photos/2024",
      );
    });
    // Success feedback
    expect(
      await screen.findByText(/7 件の画像をキューに追加/),
    ).toBeInTheDocument();
  });

  it("does not call the API when the user cancels the confirm dialog", async () => {
    window.confirm = vi.fn(() => false);
    renderButton();
    await act(async () => {
      fireEvent.click(screen.getByRole("button"));
    });
    expect(generateFolderVisualDescription).not.toHaveBeenCalled();
  });

  it("surfaces the 413 too_many_files error", async () => {
    const err = new Error("too_many_files") as Error & {
      info: { kind: "too_many_files"; max: number; requested: number };
    };
    err.info = { kind: "too_many_files", max: 500, requested: 832 };
    (generateFolderVisualDescription as unknown as ReturnType<typeof vi.fn>)
      .mockRejectedValue(err);
    renderButton();
    await act(async () => {
      fireEvent.click(screen.getByRole("button"));
    });
    expect(
      await screen.findByText(/選択範囲が多すぎます（832 件）/),
    ).toBeInTheDocument();
  });

  it("shows a generic error message on other failures", async () => {
    (generateFolderVisualDescription as unknown as ReturnType<typeof vi.fn>)
      .mockRejectedValue(new Error("network"));
    renderButton();
    await act(async () => {
      fireEvent.click(screen.getByRole("button"));
    });
    expect(
      await screen.findByText(/フォルダの処理に失敗しました/),
    ).toBeInTheDocument();
  });
});
