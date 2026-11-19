import "./intlMock";
import { describe, it, expect, vi, beforeEach, afterEach, beforeAll } from "vitest";
import { fireEvent, render } from "@testing-library/react";

vi.mock("@/addons/intelligence/api", () => ({
  getClipTimestamps: vi.fn(),
  getVideoVisualIndex: vi.fn(),
  generateVideoVisualIndex: vi.fn(),
  retryVideoVisualIndex: vi.fn(),
  getFrameUrl: vi.fn((fileId: string, t: number) => `/frame/${fileId}?t=${t}`),
  getSceneTileUrl: vi.fn((fileId: string, t: number) => `/scene-tile/${fileId}?t=${t}`),
}));

vi.mock("@/hooks/usePolicy", () => ({
  usePolicy: vi.fn(),
}));

vi.mock("@/hooks/useWebSocket", () => ({
  useWebSocket: vi.fn(() => null),
}));

import ScenesSection from "@/addons/intelligence/ScenesSection";
import { getClipTimestamps } from "@/addons/intelligence/api";
import { usePolicy } from "@/hooks/usePolicy";
import {
  advance,
  flush,
  installPointerEvents,
  makeController,
  makeRows,
  pointerX,
  ruler,
  SceneSectionProps,
  timestamps,
} from "./helpers";
import {
  WIDTH,
  installLayout,
  resetLayout,
  scroller,
  scrollTopOf,
  visibleRows,
} from "./scrollListLayout";

const N = 30;
const ROWS = makeRows(N);
const L = 150;

function section(props: Partial<SceneSectionProps> = {}) {
  return (
    <ScenesSection
      fileId="f1"
      drive="d1"
      fileType="video"
      mimeType="video/mp4"
      labelledByHost
      active
      onAvailability={() => {}}
      {...props}
    />
  );
}

async function settle(ms = 0): Promise<void> {
  await flush(8);
  await advance(ms);
  await flush(8);
}

beforeAll(() => {
  installPointerEvents();
  installLayout();
});

beforeEach(() => {
  vi.clearAllMocks();
  vi.useFakeTimers();
  resetLayout();
  vi.mocked(getClipTimestamps).mockResolvedValue(timestamps(ROWS));
  vi.mocked(usePolicy).mockReturnValue({ enabled: true, isLoading: false });
});

afterEach(() => {
  vi.useRealTimers();
});

describe("SPEC-ADDON-153 I13 — only the primary mouse button drives the ruler", () => {
  it("SPEC-ADDON-153 I13 — a press outside the ruler and a drag onto it set no point and move no list", async () => {
    const { controller } = makeController({ duration: L, currentTime: 10 });
    render(section({ mediaController: controller }));
    await settle(0);
    await settle(0);

    const el = ruler();
    const placed = scrollTopOf(scroller());
    const rows = visibleRows();

    fireEvent.pointerDown(document.body, {
      pointerType: "mouse",
      button: 0,
      buttons: 1,
      clientX: 1,
      clientY: 400,
    });
    fireEvent.pointerMove(el, {
      pointerType: "mouse",
      button: 0,
      buttons: 1,
      clientX: pointerX(120, L, WIDTH),
    });
    fireEvent.pointerMove(el, {
      pointerType: "mouse",
      button: 0,
      buttons: 1,
      clientX: pointerX(140, L, WIDTH),
    });
    fireEvent.pointerUp(el, {
      pointerType: "mouse",
      button: 0,
      buttons: 0,
      clientX: pointerX(140, L, WIDTH),
    });
    await settle(0);

    expect(ruler(), "no press on the ruler, no ruler point").toHaveAttribute(
      "aria-valuenow",
      "10",
    );
    expect(scrollTopOf(scroller()), "the list does not move").toBe(placed);
    expect(visibleRows()).toEqual(rows);
  });

  it("SPEC-ADDON-153 I13 — a right-button press and drag leave the point unset and the list where it was", async () => {
    const { controller } = makeController({ duration: L, currentTime: 10 });
    render(section({ mediaController: controller }));
    await settle(0);
    await settle(0);

    const el = ruler();
    expect(ruler()).toHaveAttribute("aria-valuenow", "10");
    const placed = scrollTopOf(scroller());
    const rows = visibleRows();

    fireEvent.pointerDown(el, {
      pointerType: "mouse",
      button: 2,
      buttons: 2,
      clientX: pointerX(120, L, WIDTH),
    });
    fireEvent.pointerMove(el, {
      pointerType: "mouse",
      button: 2,
      buttons: 2,
      clientX: pointerX(140, L, WIDTH),
    });
    fireEvent.pointerUp(el, {
      pointerType: "mouse",
      button: 2,
      buttons: 0,
      clientX: pointerX(140, L, WIDTH),
    });
    await settle(0);

    expect(ruler(), "a right press sets no ruler point").toHaveAttribute(
      "aria-valuenow",
      "10",
    );
    expect(scrollTopOf(scroller()), "the list does not move").toBe(placed);
    expect(visibleRows()).toEqual(rows);
  });
});
