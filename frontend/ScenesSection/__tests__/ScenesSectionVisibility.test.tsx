import "./intlMock";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render } from "@testing-library/react";

vi.mock("@/addons/intelligence/api", () => ({
  getClipTimestamps: vi.fn(),
  getVideoVisualIndex: vi.fn(),
  generateVideoVisualIndex: vi.fn(),
  retryVideoVisualIndex: vi.fn(),
  getFrameUrl: vi.fn(),
  getSceneTileUrl: vi.fn(),
}));

vi.mock("@/hooks/usePolicy", () => ({
  usePolicy: vi.fn(),
}));

import ScenesSection from "@/addons/intelligence/ScenesSection";
import {
  getClipTimestamps,
  getVideoVisualIndex,
} from "@/addons/intelligence/api";
import { usePolicy } from "@/hooks/usePolicy";
import { flush, SceneSectionProps } from "./helpers";

function renderSection(props: Partial<SceneSectionProps>) {
  return render(
    <ScenesSection
      fileId="f1"
      drive="d1"
      fileType="video"
      mimeType="video/mp4"
      {...props}
    />,
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(usePolicy).mockReturnValue({ enabled: true, isLoading: false });
});

describe("SPEC-ADDON-089 I5 — file types", () => {
  it.each([
    { fileType: "image", mimeType: "image/jpeg" },
    { fileType: "document", mimeType: "application/pdf" },
    { fileType: "audio", mimeType: "audio/mpeg" },
  ])(
    "SPEC-ADDON-089 I5 — a $fileType renders nothing and requests nothing",
    async ({ fileType, mimeType }) => {
      const { container } = renderSection({ fileType, mimeType });
      await flush();

      expect(container.firstChild).toBeNull();
      expect(getClipTimestamps).not.toHaveBeenCalled();
      expect(getVideoVisualIndex).not.toHaveBeenCalled();
    },
  );
});
