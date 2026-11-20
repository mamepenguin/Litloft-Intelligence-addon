import { describe, it, expect, vi, beforeEach, afterEach, beforeAll } from "vitest";
import { render, screen } from "@testing-library/react";

vi.mock("next-intl", async () => {
  const en = (await import("@/addons/intelligence/messages/en.json")).default as unknown as Record<
    string,
    Record<string, string>
  >;
  return {
    useTranslations: (namespace?: string) => (key: string, values?: Record<string, unknown>) => {
      const table = (namespace ? en[namespace] : undefined) ?? en.file;
      const template = table?.[key] ?? key;
      return Object.entries(values ?? {}).reduce(
        (text, [name, value]) => text.replace(`{${name}}`, String(value)),
        template,
      );
    },
    useLocale: () => "en",
  };
});

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

const wsEvent = vi.hoisted(() => ({
  value: null as null | { event: string; data: Record<string, unknown> },
}));

vi.mock("@/hooks/useWebSocket", () => ({
  useWebSocket: vi.fn((filter?: string) => {
    const event = wsEvent.value;
    if (!event) return null;
    if (filter && event.event !== filter) return null;
    return event;
  }),
}));

import ScenesSection from "@/addons/intelligence/ScenesSection";
import {
  getClipTimestamps,
  getVideoVisualIndex,
} from "@/addons/intelligence/api";
import { usePolicy } from "@/hooks/usePolicy";
import {
  advance,
  flush,
  indexState,
  installPointerEvents,
  LOFT_MIME,
  makeController,
  makeRows,
  ruler,
  SceneSectionProps,
  timestamps,
} from "./helpers";
import {
  CHIP,
  ROW,
  chip,
  installLayout,
  resetLayout,
  scroller,
  scrollTopOf,
  tileElements,
  wheelTo,
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

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

beforeAll(() => {
  installPointerEvents();
  installLayout();
});

beforeEach(() => {
  vi.clearAllMocks();
  wsEvent.value = null;
  vi.useFakeTimers();
  resetLayout();
  vi.mocked(getClipTimestamps).mockResolvedValue(timestamps(ROWS));
  vi.mocked(getVideoVisualIndex).mockResolvedValue(indexState());
  vi.mocked(usePolicy).mockReturnValue({ enabled: true, isLoading: false });
});

afterEach(() => {
  vi.useRealTimers();
});

describe("SPEC-ADDON-152 §3 — the frames until they arrive", () => {
  it("SPEC-ADDON-152 §1.1 — the 3c skeleton is drawn inside the list's box and no tile is drawn", async () => {
    const pending = deferred<ReturnType<typeof timestamps>>();
    vi.mocked(getClipTimestamps).mockReturnValue(pending.promise);
    const { container } = render(section({ mediaController: makeController({ duration: L }).controller }));
    await settle(0);

    expect(tileElements()).toHaveLength(0);
    const list = container.querySelector("[data-scene-list]");
    const skeleton = list?.querySelector(".grid");
    expect(skeleton).not.toBeNull();
    expect(skeleton!.children).toHaveLength(9);

    pending.resolve(timestamps(ROWS));
    await settle(0);
    await settle(0);
    expect(tileElements()).toHaveLength(N);
  });
});

describe("SPEC-ADDON-152 §3 — a .loft whose rows arrive while shown", () => {
  it("SPEC-ADDON-152 I2 — the arriving rows are all drawn and the playing row is placed", async () => {
    const pending = deferred<ReturnType<typeof timestamps>>();
    vi.mocked(getClipTimestamps).mockReturnValue(pending.promise);
    const { controller } = makeController({ duration: L, currentTime: 100 });
    render(section({ mimeType: LOFT_MIME, mediaController: controller }));
    await settle(0);

    expect(tileElements()).toHaveLength(0);
    expect(screen.getByRole("slider", { name: "Frame timeline" })).toBeInTheDocument();

    pending.resolve(timestamps(ROWS));
    await settle(0);
    await settle(0);

    expect(tileElements()).toHaveLength(N);
    expect(scrollTopOf(scroller())).toBe(5 * ROW);
    expect(chip()).toBeNull();
    expect(ruler()).toHaveAttribute("aria-valuenow", "100");
  });
});

describe("SPEC-ADDON-152 I2 — the playback clock", () => {
  it("SPEC-ADDON-152 I2 — a controller that appears while shown places the playing row then", async () => {
    const view = render(section({ mediaController: null }));
    await settle(0);
    await settle(0);
    expect(tileElements()).toHaveLength(N);
    expect(scrollTopOf(scroller())).toBe(0);

    const { controller } = makeController({ duration: L, currentTime: 100 });
    view.rerender(section({ mediaController: controller }));
    await settle(0);
    await settle(0);

    expect(scrollTopOf(scroller())).toBe(5 * ROW);
    expect(chip()).toBeNull();
  });
});

describe("SPEC-ADDON-152 I3 — a Visual Index reload while scrolled", () => {
  it("SPEC-ADDON-152 I3 — the reload leaves the list where the reader left it and following clear", async () => {
    const { controller } = makeController({ duration: L, currentTime: 0 });
    const view = render(section({ mediaController: controller }));
    await settle(0);
    await settle(0);
    expect(tileElements()).toHaveLength(N);

    await wheelTo(6 * ROW);
    await settle(0);
    expect(chip(), `no "${CHIP}" chip`).not.toBeNull();
    const top = scrollTopOf(scroller());
    expect(getVideoVisualIndex).toHaveBeenCalledTimes(1);

    wsEvent.value = {
      event: "intelligence.video_visual.succeeded",
      data: { file_id: "f1" },
    };
    view.rerender(section({ mediaController: controller }));
    await settle(0);
    await settle(0);

    expect(getVideoVisualIndex).toHaveBeenCalledTimes(2);
    expect(scrollTopOf(scroller())).toBe(top);
    expect(chip()).not.toBeNull();
  });
});
