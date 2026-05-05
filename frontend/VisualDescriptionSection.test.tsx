/**
 * Tests for VisualDescriptionSection state branches.
 *
 * Spec: docs/superpowers/specs/2026-04-23-intelligence-vision-describe.md §UI
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent, act } from "@testing-library/react";
import { NextIntlClientProvider } from "next-intl";

vi.mock("@/addons/intelligence/api", () => ({
  getVisualDescription: vi.fn(),
  generateVisualDescription: vi.fn(),
  deleteVisualDescription: vi.fn(),
}));

vi.mock("@/lib/api", () => ({
  getFile: vi.fn(),
}));

import VisualDescriptionSection from "@/addons/intelligence/VisualDescriptionSection";
import {
  getVisualDescription,
  generateVisualDescription,
} from "@/addons/intelligence/api";
import { getFile } from "@/lib/api";

const imageFile = {
  id: "f1",
  drive: "family",
  filename: "photo.jpg",
  file_type: "image",
  mime_type: "image/jpeg",
};

function renderSection() {
  return render(
    <NextIntlClientProvider locale="en" messages={{}}>
      <VisualDescriptionSection fileId="f1" drive="family" />
    </NextIntlClientProvider>,
  );
}

describe("VisualDescriptionSection", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    (getFile as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(imageFile);
  });

  it("renders nothing when the feature is unavailable (GET returned 404 / null)", async () => {
    (getVisualDescription as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(null);
    const { container } = renderSection();
    await waitFor(() => {
      expect(getVisualDescription).toHaveBeenCalled();
    });
    // No section rendered when backend said the feature is unreachable.
    expect(container).toBeEmptyDOMElement();
  });

  it("renders nothing for non-image files", async () => {
    (getFile as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({
      ...imageFile,
      file_type: "video",
      mime_type: "video/mp4",
    });
    (getVisualDescription as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({
      file_id: "f1",
      visual_description: null,
      status: null,
      model: null,
      generated_at: null,
    });
    const { container } = renderSection();
    await waitFor(() => expect(getFile).toHaveBeenCalled());
    // Even if the backend would serve data, non-image files don't show
    // the section in Phase 1.
    expect(container).toBeEmptyDOMElement();
  });

  it("renders the generate button when status is null", async () => {
    (getVisualDescription as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({
      file_id: "f1",
      visual_description: null,
      status: null,
      model: null,
      generated_at: null,
    });
    renderSection();
    expect(
      await screen.findByRole("button", { name: /Generate AI description/ }),
    ).toBeInTheDocument();
  });

  it("renders the unsupported notice when the backend reports unsupported", async () => {
    (getVisualDescription as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({
      file_id: "f1",
      visual_description: null,
      status: "unsupported",
      model: null,
      generated_at: null,
    });
    renderSection();
    expect(
      await screen.findByText(/No vision-capable LLM is configured/),
    ).toBeInTheDocument();
  });

  it("renders the description and regenerate button when status is success", async () => {
    (getVisualDescription as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({
      file_id: "f1",
      visual_description: "A detailed description of the image.",
      status: "success",
      model: "llava:13b",
      generated_at: "2026-04-23T12:00:00Z",
    });
    renderSection();
    expect(
      await screen.findByText("A detailed description of the image."),
    ).toBeInTheDocument();
    expect(screen.getByText("llava:13b")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /Regenerate/ }),
    ).toBeInTheDocument();
  });

  it("renders the retry button when status is failed", async () => {
    (getVisualDescription as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({
      file_id: "f1",
      visual_description: null,
      status: "failed",
      model: null,
      generated_at: null,
    });
    renderSection();
    expect(
      await screen.findByText(/Description generation failed/),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Retry/ })).toBeInTheDocument();
  });

  it("posts to generate when the generate button is clicked", async () => {
    (getVisualDescription as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({
      file_id: "f1",
      visual_description: null,
      status: null,
      model: null,
      generated_at: null,
    });
    (generateVisualDescription as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValue({ status: "accepted", file_id: "f1" });
    renderSection();
    const button = await screen.findByRole("button", {
      name: /Generate AI description/,
    });
    await act(async () => {
      fireEvent.click(button);
    });
    await waitFor(() => {
      expect(generateVisualDescription).toHaveBeenCalledWith("f1", "family");
    });
  });

  it("shows the pending indicator when status is pending", async () => {
    (getVisualDescription as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({
      file_id: "f1",
      visual_description: null,
      status: "pending",
      model: null,
      generated_at: null,
    });
    renderSection();
    expect(
      await screen.findByText(/Generating description/),
    ).toBeInTheDocument();
  });
});
