/**
 * RED-phase tests for the folder-level transcript refine button.
 *
 * Spec: docs/superpowers/specs/2026-04-15-intelligence-transcript-refine.md
 *
 * The component is not yet implemented. Import is expected to fail
 * during the RED phase — Vitest reports the test file as "failed to
 * collect", which is a failing test, which is what we want.
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";

vi.mock("@/addons/intelligence/api", () => ({
  refineFolderTranscripts: vi.fn().mockResolvedValue({
    queued: 2,
    jobs: ["job-a", "job-b"],
  }),
}));

// The component lives next to the other Folder* buttons in the
// intelligence addon tree so the per-addon slot wiring can pick it up.
import FolderRefineButton from "@/addons/intelligence/FolderRefineButton";
import { refineFolderTranscripts } from "@/addons/intelligence/api";

describe("FolderRefineButton", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders the folder-level button", () => {
    render(
      <FolderRefineButton
        drive="family"
        path="videos/2024"
        fileIds={["f1", "f2"]}
      />
    );
    // Button label is something like "Refine folder transcripts with AI"
    expect(
      screen.getByRole("button", { name: /Refine folder transcripts with AI/ })
    ).toBeInTheDocument();
  });

  it("calls refineFolderTranscripts with drive and fileIds on click", async () => {
    render(
      <FolderRefineButton
        drive="family"
        path="videos/2024"
        fileIds={["f1", "f2"]}
      />
    );
    fireEvent.click(screen.getByRole("button", { name: /Refine folder transcripts with AI/ }));

    await waitFor(() => {
      expect(refineFolderTranscripts).toHaveBeenCalledWith(
        "family",
        ["f1", "f2"],
      );
    });
  });

  it("shows a progress indicator while the request is in flight", async () => {
    // Delay the mock so we can observe the intermediate loading state.
    let resolveFn: (v: unknown) => void = () => {};
    (refineFolderTranscripts as unknown as ReturnType<typeof vi.fn>)
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolveFn = resolve;
          })
      );

    render(
      <FolderRefineButton
        drive="family"
        path="videos/2024"
        fileIds={["f1", "f2"]}
      />
    );
    fireEvent.click(screen.getByRole("button", { name: /Refine folder transcripts with AI/ }));

    // While pending, the button is disabled (matching FolderSummariesButton).
    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: /Refine folder transcripts with AI/ })
      ).toBeDisabled();
    });

    resolveFn({ queued: 2, jobs: ["job-a", "job-b"] });
  });

  it("returns null when there are no file ids", () => {
    const { container } = render(
      <FolderRefineButton drive="family" path="videos/2024" fileIds={[]} />
    );
    // Match the FolderSummariesButton / FolderAutoTagsButton convention:
    // the button is hidden when there's nothing to refine.
    expect(container).toBeEmptyDOMElement();
  });
});
