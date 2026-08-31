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
      reason: null,
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
      reason: null,
      model: null,
      generated_at: null,
    });
    renderSection();
    expect(
      await screen.findByRole("button", { name: /Create description/ }),
    ).toBeInTheDocument();
  });

  it("shows the configuration notice, and no retry, when no vision model is set", async () => {
    // Nothing to run, so nothing to offer: the fix is in search-config.yml.
    (getVisualDescription as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({
      file_id: "f1",
      visual_description: null,
      status: "unsupported",
      reason: "not_configured",
      model: null,
      generated_at: null,
    });
    renderSection();
    expect(
      await screen.findByText(/No LLM capable of image description is configured/),
    ).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Retry/ })).toBeNull();
  });

  it("offers a retry when a real attempt found the model cannot see", async () => {
    // The reported bug: this branch used to be a dead end.
    (getVisualDescription as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({
      file_id: "f1",
      visual_description: null,
      status: "unsupported",
      reason: "vision_unsupported",
      model: "qwen3:8b",
      generated_at: null,
    });
    renderSection();
    expect(
      await screen.findByText(/qwen3:8b does not accept images/),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /Retry/ }),
    ).toBeInTheDocument();
  });

  it("offers a retry for a verdict recorded before reasons were kept", async () => {
    // Legacy rows carry no reason and were latched by the guessing the
    // backend no longer does, so they are the likeliest to be wrong.
    (getVisualDescription as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({
      file_id: "f1",
      visual_description: null,
      status: "unsupported",
      reason: null,
      model: "llava:13b",
      generated_at: null,
    });
    renderSection();
    expect(
      await screen.findByRole("button", { name: /Retry/ }),
    ).toBeInTheDocument();
  });

  it("retrying an unsupported file posts to generate", async () => {
    (getVisualDescription as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({
      file_id: "f1",
      visual_description: null,
      status: "unsupported",
      reason: "vision_unsupported",
      model: "qwen3:8b",
      generated_at: null,
    });
    (generateVisualDescription as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValue({ status: "accepted", file_id: "f1" });
    renderSection();
    const button = await screen.findByRole("button", { name: /Retry/ });
    await act(async () => {
      fireEvent.click(button);
    });
    await waitFor(() => {
      expect(generateVisualDescription).toHaveBeenCalledWith("f1", "family");
    });
  });

  it.each([
    ["model_missing", /was not found on the LLM provider/],
    ["image_rejected", /could not read this image/],
    ["token_budget", /cut off by the token limit/],
  ])("explains a %s failure specifically", async (reason, expected) => {
    (getVisualDescription as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({
      file_id: "f1",
      visual_description: null,
      status: "failed",
      reason,
      model: "llava:13b",
      generated_at: null,
    });
    renderSection();
    expect(await screen.findByText(expected)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Retry/ })).toBeInTheDocument();
  });

  it("falls back to the generic message for an unknown reason", async () => {
    (getVisualDescription as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({
      file_id: "f1",
      visual_description: null,
      status: "failed",
      reason: "something_added_later",
      model: null,
      generated_at: null,
    });
    renderSection();
    expect(
      await screen.findByText(/Failed to create description/),
    ).toBeInTheDocument();
  });

  it("says so when the backend declines to queue the file", async () => {
    (getVisualDescription as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({
      file_id: "f1",
      visual_description: null,
      status: "failed",
      reason: null,
      model: null,
      generated_at: null,
    });
    const declined = Object.assign(new Error("not_queued"), {
      info: { kind: "not_queued", reason: "policy_off" },
    });
    (generateVisualDescription as unknown as ReturnType<typeof vi.fn>)
      .mockRejectedValue(declined);
    renderSection();
    const button = await screen.findByRole("button", { name: /Retry/ });
    await act(async () => {
      fireEvent.click(button);
    });
    expect(
      await screen.findByText(/could not be queued/),
    ).toBeInTheDocument();
  });

  it("renders the description and regenerate button when status is success", async () => {
    (getVisualDescription as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({
      file_id: "f1",
      visual_description: "A detailed description of the image.",
      status: "success",
      reason: null,
      model: "llava:13b",
      generated_at: "2026-04-23T12:00:00Z",
    });
    renderSection();
    expect(
      await screen.findByText("A detailed description of the image."),
    ).toBeInTheDocument();
    expect(screen.getByText("llava:13b")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /Create again/ }),
    ).toBeInTheDocument();
  });

  it("renders the retry button when status is failed", async () => {
    (getVisualDescription as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({
      file_id: "f1",
      visual_description: null,
      status: "failed",
      reason: null,
      model: null,
      generated_at: null,
    });
    renderSection();
    expect(
      await screen.findByText(/Failed to create description/),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Retry/ })).toBeInTheDocument();
  });

  it("posts to generate when the generate button is clicked", async () => {
    (getVisualDescription as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({
      file_id: "f1",
      visual_description: null,
      status: null,
      reason: null,
      model: null,
      generated_at: null,
    });
    (generateVisualDescription as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValue({ status: "accepted", file_id: "f1" });
    renderSection();
    const button = await screen.findByRole("button", {
      name: /Create description/,
    });
    await act(async () => {
      fireEvent.click(button);
    });
    await waitFor(() => {
      expect(generateVisualDescription).toHaveBeenCalledWith("f1", "family");
    });
  });

  it("offers a way out of a pending row it did not start", async () => {
    // A worker that died mid-flight (a restart, say) leaves the row
    // exactly like this. A spinner with no button is the same dead end
    // this section had for unsupported.
    (getVisualDescription as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({
      file_id: "f1",
      visual_description: null,
      status: "pending",
      reason: null,
      model: "llava:13b",
      generated_at: null,
    });
    renderSection();
    expect(await screen.findByText(/Creating description/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Retry/ })).toBeInTheDocument();
  });

  it("treats an already-queued answer as success, not an error", async () => {
    // Retrying a row that is genuinely running is answered by the
    // worker recognising the same work, not by refusing it.
    (getVisualDescription as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({
      file_id: "f1",
      visual_description: null,
      status: "pending",
      reason: null,
      model: "llava:13b",
      generated_at: null,
    });
    (generateVisualDescription as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValue({ status: "already_queued", file_id: "f1" });
    renderSection();
    const button = await screen.findByRole("button", { name: /Retry/ });
    await act(async () => {
      fireEvent.click(button);
    });
    await waitFor(() => expect(generateVisualDescription).toHaveBeenCalled());
    expect(screen.queryByText(/could not be queued/)).toBeNull();
    expect(screen.queryByText(/Could not start/)).toBeNull();
  });

  it("shows the pending indicator when status is pending", async () => {
    (getVisualDescription as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({
      file_id: "f1",
      visual_description: null,
      status: "pending",
      reason: null,
      model: null,
      generated_at: null,
    });
    renderSection();
    expect(
      await screen.findByText(/Creating description/),
    ).toBeInTheDocument();
  });
});
