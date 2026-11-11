import "./intlMock";
import { describe, it, expect, vi, beforeEach, beforeAll } from "vitest";
import { useEffect } from "react";
import {
  render,
  screen,
  fireEvent,
  createEvent,
  act,
} from "@testing-library/react";

vi.mock("@/addons/intelligence/api", () => ({
  getClipTimestamps: vi.fn(),
  getVideoVisualIndex: vi.fn(),
  generateVideoVisualIndex: vi.fn(),
  retryVideoVisualIndex: vi.fn(),
  getFrameUrl: vi.fn(
    (fileId: string, t: number) => `/frame/${fileId}?t=${t}`,
  ),
  getSceneTileUrl: vi.fn(
    (fileId: string, t: number) => `/scene-tile/${fileId}?t=${t}`,
  ),
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
import {
  ShortcutsProvider,
  useShortcutsContext,
} from "@/components/ShortcutsProvider";
import {
  flush,
  header,
  installPointerEvents,
  makeController,
  makeRows,
  ruler,
  SceneSectionProps,
  tile,
  timestamps,
} from "./helpers";

function PageShortcuts({ onShortcut }: { onShortcut: () => void }) {
  const { push, pop } = useShortcutsContext();

  useEffect(() => {
    push({
      id: "page-test",
      label: "page test",
      shortcuts: ["arrowleft", "arrowright", "home", "end", "enter"].map(
        (key) => ({ key, label: key, handler: onShortcut }),
      ),
    });
    return () => pop("page-test");
  }, [push, pop, onShortcut]);

  return null;
}

function section(props: Partial<SceneSectionProps> = {}) {
  return (
    <ScenesSection
      fileId="f1"
      drive="d1"
      fileType="video"
      mimeType="video/mp4"
      {...props}
    />
  );
}

function renderKeyboard(
  onShortcut: () => void,
  props: Partial<SceneSectionProps> = {},
) {
  return render(
    <ShortcutsProvider>
      <PageShortcuts onShortcut={onShortcut} />
      <button>outside</button>
      {section(props)}
    </ShortcutsProvider>,
  );
}

async function openWithFocus(): Promise<HTMLElement> {
  fireEvent.click(header());
  const el = await screen.findByRole("slider");
  el.focus();
  return el;
}

beforeAll(installPointerEvents);

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(getClipTimestamps).mockResolvedValue(timestamps(makeRows(15)));
  vi.mocked(getVideoVisualIndex).mockResolvedValue(null);
  vi.mocked(usePolicy).mockReturnValue({ enabled: true, isLoading: false });
});

describe("SPEC-ADDON-089 I4 — the ruler as a slider", () => {
  it("registers the page shortcuts used by this suite", async () => {
    const pageShortcut = vi.fn();
    const { controller } = makeController({ duration: 70, currentTime: 10 });
    const { container } = renderKeyboard(pageShortcut, {
      mediaController: controller,
    });
    await flush();

    fireEvent.keyDown(container.querySelector("button")!, { key: "ArrowLeft" });

    expect(pageShortcut).toHaveBeenCalled();
  });

  it.each([
    { key: "ArrowLeft" },
    { key: "ArrowRight" },
    { key: "Home" },
    { key: "End" },
    { key: "Enter" },
  ])(
    "SPEC-ADDON-089 I4 — $key on the focused ruler never reaches the page shortcut",
    async ({ key }) => {
      const pageShortcut = vi.fn();
      const { controller } = makeController({ duration: 70, currentTime: 10 });
      renderKeyboard(pageShortcut, { mediaController: controller });
      await flush();
      const el = await openWithFocus();

      if (key === "Enter") {
        fireEvent.keyDown(el, { key: "ArrowRight" });
      }

      const event = createEvent.keyDown(el, { key });
      fireEvent(el, event);

      expect(event.defaultPrevented).toBe(true);
      expect(pageShortcut).not.toHaveBeenCalled();
    },
  );

  it("keeps the five keys off the React root and the document", async () => {
    const pageShortcut = vi.fn();
    const { controller } = makeController({ duration: 70, currentTime: 10 });
    const { container } = renderKeyboard(pageShortcut, {
      mediaController: controller,
    });
    await flush();
    const el = await openWithFocus();

    const onRoot = vi.fn();
    const onDocument = vi.fn();
    container.addEventListener("keydown", onRoot);
    document.addEventListener("keydown", onDocument);
    try {
      for (const key of ["ArrowLeft", "ArrowRight", "Home", "End", "Enter"]) {
        fireEvent.keyDown(el, { key });
      }
    } finally {
      container.removeEventListener("keydown", onRoot);
      document.removeEventListener("keydown", onDocument);
    }

    expect(onRoot).not.toHaveBeenCalled();
    expect(onDocument).not.toHaveBeenCalled();
  });

  it("moves by one frame from the playing frame", async () => {
    const pageShortcut = vi.fn();
    const { controller } = makeController({ duration: 70, currentTime: 10 });
    renderKeyboard(pageShortcut, { mediaController: controller });
    await flush();
    const el = await openWithFocus();

    fireEvent.keyDown(el, { key: "ArrowRight" });
    expect(ruler()).toHaveAttribute("aria-valuenow", "15");

    fireEvent.keyDown(el, { key: "ArrowLeft" });
    expect(ruler()).toHaveAttribute("aria-valuenow", "10");
  });

  it("Home and End reach the first and last frame", async () => {
    const pageShortcut = vi.fn();
    const { controller } = makeController({ duration: 70, currentTime: 10 });
    renderKeyboard(pageShortcut, { mediaController: controller });
    await flush();
    const el = await openWithFocus();

    fireEvent.keyDown(el, { key: "Home" });
    expect(ruler()).toHaveAttribute("aria-valuenow", "0");

    fireEvent.keyDown(el, { key: "End" });
    expect(ruler()).toHaveAttribute("aria-valuenow", "70");
  });

  it("Enter seeks the browse frame, plays and clears the point", async () => {
    const pageShortcut = vi.fn();
    const { controller, seek, play } = makeController({
      duration: 70,
      currentTime: 10,
    });
    renderKeyboard(pageShortcut, { mediaController: controller });
    await flush();
    const el = await openWithFocus();

    fireEvent.keyDown(el, { key: "ArrowRight" });
    expect(ruler()).toHaveAttribute("aria-valuenow", "15");

    fireEvent.keyDown(el, { key: "Enter" });

    expect(seek).toHaveBeenCalledWith(15);
    expect(play).toHaveBeenCalled();
    expect(ruler()).toHaveAttribute("aria-valuenow", "10");
    expect(pageShortcut).not.toHaveBeenCalled();
  });

  it("Enter with no browse point does nothing", async () => {
    const pageShortcut = vi.fn();
    const { controller, seek, play } = makeController({
      duration: 70,
      currentTime: 10,
    });
    renderKeyboard(pageShortcut, { mediaController: controller });
    await flush();
    const el = await openWithFocus();

    fireEvent.keyDown(el, { key: "Enter" });

    expect(seek).not.toHaveBeenCalled();
    expect(play).not.toHaveBeenCalled();
    expect(pageShortcut).not.toHaveBeenCalled();
    expect(ruler()).toHaveAttribute("aria-valuenow", "10");
  });

  it("reports the point in whole seconds and formatted text", async () => {
    const pageShortcut = vi.fn();
    const { controller } = makeController({ duration: 70, currentTime: 10 });
    renderKeyboard(pageShortcut, { mediaController: controller });
    await flush();
    const el = await openWithFocus();

    expect(ruler()).toHaveAttribute("aria-valuemin", "0");
    expect(ruler()).toHaveAttribute("aria-valuemax", "70");
    expect(ruler()).toHaveAttribute("aria-valuenow", "10");
    expect(ruler()).toHaveAttribute("aria-valuetext", "0:10");

    fireEvent.keyDown(el, { key: "ArrowRight" });
    expect(ruler()).toHaveAttribute("aria-valuenow", "15");
    expect(ruler()).toHaveAttribute("aria-valuetext", "0:15");

    fireEvent.keyDown(el, { key: "End" });
    expect(ruler()).toHaveAttribute("aria-valuenow", "70");
    expect(ruler()).toHaveAttribute("aria-valuetext", "1:10");
  });

  it("keeps the point when focus moves to a tile and clears it when focus leaves", async () => {
    const pageShortcut = vi.fn();
    const { controller } = makeController({ duration: 70, currentTime: 10 });
    renderKeyboard(pageShortcut, { mediaController: controller });
    await flush();
    const el = await openWithFocus();

    fireEvent.keyDown(el, { key: "ArrowRight" });
    expect(ruler()).toHaveAttribute("aria-valuenow", "15");

    act(() => tile(15).focus());
    expect(ruler()).toHaveAttribute("aria-valuenow", "15");

    act(() => screen.getByRole("button", { name: "outside" }).focus());
    await flush();
    expect(ruler()).toHaveAttribute("aria-valuenow", "10");
  });
});
