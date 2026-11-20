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
  firePointer,
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

  it("SPEC-ADDON-153 I13 — after a ruler drag released off the ruler, a press elsewhere dragged onto it sets no point", async () => {
    const { controller } = makeController({ duration: L, currentTime: 10 });
    render(section({ mediaController: controller }));
    await settle(0);
    await settle(0);

    const el = ruler();
    const mouse = { pointerType: "mouse", button: 0 } as const;
    fireEvent.pointerDown(el, { ...mouse, buttons: 1, clientX: pointerX(60, L, WIDTH) });
    fireEvent.pointerMove(el, { ...mouse, buttons: 1, clientX: pointerX(80, L, WIDTH) });
    fireEvent.pointerUp(document.body, { ...mouse, buttons: 0, clientX: 1, clientY: 400 });
    await settle(0);
    const pointAfterDrag = ruler().getAttribute("aria-valuenow");
    const placed = scrollTopOf(scroller());

    fireEvent.pointerDown(document.body, { ...mouse, buttons: 1, clientX: 1, clientY: 400 });
    fireEvent.pointerMove(el, { ...mouse, buttons: 1, clientX: pointerX(140, L, WIDTH) });
    fireEvent.pointerUp(document.body, { ...mouse, buttons: 0, clientX: 1, clientY: 400 });
    await settle(0);

    expect(ruler().getAttribute("aria-valuenow"), "the earlier drag's point stays").toBe(pointAfterDrag);
    expect(scrollTopOf(scroller()), "the list does not move").toBe(placed);
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

describe("SPEC-ADDON-153 I9 — a cancelled touch drag on the ruler in flow form", () => {
  it("SPEC-ADDON-153 I9 — touchmoves of the cancelled gesture keep the ruler point and following it put back", async () => {
    const { controller } = makeController({ duration: L, currentTime: 10 });
    render(<div data-inspector-scroller="">{section({ mediaController: controller })}</div>);
    await settle(0);
    await settle(0);
    const el = ruler();
    const x = (time: number) => pointerX(time, L, WIDTH);

    firePointer(el, "pointerDown", { pointerType: "touch", clientX: x(12), buttons: 1 });
    firePointer(el, "pointerMove", { pointerType: "touch", clientX: x(72), buttons: 1 });
    fireEvent.touchMove(el, { touches: [{ clientX: x(132), clientY: 10 }] });
    firePointer(el, "pointerMove", { pointerType: "touch", clientX: x(132), buttons: 1 });
    await settle(0);
    expect(ruler()).toHaveAttribute("aria-valuenow", "130");

    firePointer(el, "pointerCancel", { pointerType: "touch", clientX: x(132) });
    await settle(0);
    expect(ruler()).toHaveAttribute("aria-valuenow", "10");
    const restored = scrollTopOf(scroller());

    fireEvent.touchMove(el, { touches: [{ clientX: x(132), clientY: 40 }] });
    fireEvent.touchMove(el, { touches: [{ clientX: x(132), clientY: 80 }] });
    await settle(0);
    expect(ruler(), "the gesture's touchmoves are not a reader's scroll").toHaveAttribute(
      "aria-valuenow",
      "10",
    );
    expect(scrollTopOf(scroller())).toBe(restored);

    fireEvent.touchEnd(el, { changedTouches: [{ clientX: x(132), clientY: 80 }] });
    const list = scroller();
    fireEvent.touchMove(list, { touches: [{ clientX: 10, clientY: 200 }] });
    await settle(0);
    expect(ruler(), "a later touch on the list is a reader's scroll again").not.toHaveAttribute(
      "aria-valuenow",
      "10",
    );
  });
});
