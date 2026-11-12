import "./intlMock";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { useCallback, useState } from "react";

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

vi.mock("@/hooks/useWebSocket", () => ({
  useWebSocket: vi.fn(() => null),
}));

import ScenesSection from "@/addons/intelligence/ScenesSection";
import {
  getClipTimestamps,
  getFrameUrl,
  getSceneTileUrl,
  getVideoVisualIndex,
} from "@/addons/intelligence/api";
import { usePolicy } from "@/hooks/usePolicy";
import {
  flush,
  images,
  indexState,
  LOFT_MIME,
  makeRows,
  timestamps,
} from "./helpers";

const Section = ScenesSection as unknown as React.ComponentType<
  Record<string, unknown>
>;

interface PanelProps {
  fileId?: string;
  mimeType?: string;
  active?: boolean;
  labelledByHost?: boolean;
  onAvailability?: (available: boolean) => void;
}

function panel({
  fileId = "f1",
  mimeType = "video/mp4",
  ...rest
}: PanelProps = {}) {
  return (
    <Section
      fileId={fileId}
      drive="d1"
      fileType="video"
      mimeType={mimeType}
      fillHeight
      {...rest}
    />
  );
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

const answers = (report: ReturnType<typeof vi.fn>) =>
  report.mock.calls.map(([available]) => available);

/**
 * Availability is wired back into `active` the way the core does: a `false`
 * hides the panel, and a file change drops every report. Both are deliberate
 * setup for the loop the host and the panel form together.
 */
function Host({
  fileId,
  on,
  log,
  mimeType = "video/mp4",
}: {
  fileId: string;
  on: boolean;
  log: boolean[];
  mimeType?: string;
}) {
  const [reported, setReported] = useState<boolean | null>(null);
  const [reportedFor, setReportedFor] = useState(fileId);
  if (reportedFor !== fileId) {
    setReportedFor(fileId);
    setReported(null);
  }
  const report = useCallback(
    (available: boolean) => {
      log.push(available);
      setReported(available);
    },
    [log],
  );
  return (
    <Section
      fileId={fileId}
      drive="d1"
      fileType="video"
      mimeType={mimeType}
      fillHeight
      labelledByHost
      active={on && reported !== false}
      onAvailability={report}
    />
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(usePolicy).mockReturnValue({ enabled: true, isLoading: false });
  vi.mocked(getVideoVisualIndex).mockResolvedValue(indexState());
});

describe("SPEC-ADDON-099 Scenes as a player-side panel", () => {
  it("SPEC-ADDON-099 I10 answers false on mount, before its request returns", async () => {
    vi.mocked(getClipTimestamps).mockReturnValue(new Promise(() => {}));
    const report = vi.fn();

    render(panel({ active: false, onAvailability: report }));

    expect(answers(report)[0]).toBe(false);
    expect(answers(report)).not.toContain(true);
  });

  it("SPEC-ADDON-099 I10 makes one clip-timestamps request for a native video before it is first active, and nothing else", async () => {
    vi.mocked(getClipTimestamps).mockResolvedValue(timestamps(makeRows(3)));
    const report = vi.fn();

    render(panel({ active: false, onAvailability: report }));
    await flush(8);

    expect(getClipTimestamps).toHaveBeenCalledTimes(1);
    expect(getVideoVisualIndex).not.toHaveBeenCalled();
    expect(getFrameUrl).not.toHaveBeenCalled();
    expect(getSceneTileUrl).not.toHaveBeenCalled();
    expect(images()).toHaveLength(0);
    expect(answers(report).at(-1)).toBe(true);
  });

  it("SPEC-ADDON-099 I10 asks for the Visual Index once it is first active", async () => {
    vi.mocked(getClipTimestamps).mockResolvedValue(timestamps(makeRows(3)));
    const { rerender } = render(panel({ active: false, onAvailability: vi.fn() }));
    await flush(8);
    expect(getVideoVisualIndex).not.toHaveBeenCalled();

    rerender(panel({ active: true, onAvailability: vi.fn() }));
    await flush(8);

    expect(getVideoVisualIndex).toHaveBeenCalled();
  });

  it.each([
    ["zero rows", () => Promise.resolve(timestamps([]))],
    ["a failed request", () => Promise.reject(new Error("boom"))],
    [
      "a 404 (the policy off)",
      () => Promise.reject(Object.assign(new Error("Not Found"), { status: 404 })),
    ],
  ])("SPEC-ADDON-099 I10 never answers true for a native video on %s", async (_name, answer) => {
    vi.mocked(getClipTimestamps).mockImplementation(answer as never);
    const report = vi.fn();

    render(panel({ active: false, onAvailability: report }));
    await flush(8);

    expect(getClipTimestamps).toHaveBeenCalledTimes(1);
    expect(answers(report)).toContain(false);
    expect(answers(report)).not.toContain(true);
  });

  it("SPEC-ADDON-099 answers true for a .loft once its clip-timestamps has a row", async () => {
    vi.mocked(getClipTimestamps).mockResolvedValue(timestamps(makeRows(2)));
    const report = vi.fn();

    render(panel({ mimeType: LOFT_MIME, active: false, onAvailability: report }));
    await flush(8);

    expect(answers(report)[0]).toBe(false);
    expect(answers(report).at(-1)).toBe(true);
  });

  it("SPEC-ADDON-099 I10 answers false again on a file change and applies no answer for the earlier file", async () => {
    const first = deferred<ReturnType<typeof timestamps>>();
    const second = deferred<ReturnType<typeof timestamps>>();
    vi.mocked(getClipTimestamps)
      .mockReturnValueOnce(first.promise as never)
      .mockReturnValueOnce(second.promise as never);
    const report = vi.fn();
    const { rerender } = render(
      panel({ fileId: "f1", active: false, onAvailability: report }),
    );

    rerender(panel({ fileId: "f2", active: false, onAvailability: report }));
    expect(answers(report).at(-1)).toBe(false);
    const afterChange = report.mock.calls.length;

    first.resolve(timestamps(makeRows(3), "f1"));
    await flush(8);
    second.resolve(timestamps([], "f2"));
    await flush(8);

    expect(answers(report).slice(afterChange)).not.toContain(true);
  });

  it("SPEC-ADDON-099 I10 draws no header or collapse of its own when the host labels it, and shows its body while active", async () => {
    vi.mocked(getClipTimestamps).mockResolvedValue(timestamps(makeRows(3)));

    render(panel({ active: true, labelledByHost: true, onAvailability: vi.fn() }));

    expect(
      await screen.findByRole("slider", { name: "Frame timeline" }),
    ).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /^Scenes/ })).toBeNull();
    expect(
      screen.queryAllByRole("button").filter((button) =>
        button.hasAttribute("aria-expanded"),
      ),
    ).toEqual([]);
  });
});

describe("SPEC-ADDON-099 I10 — a host that hides the entry while it answers false", () => {
  it.each([
    ["a native video", "video/mp4"],
    ["a .loft", LOFT_MIME],
  ])("ends available with %s on at mount", async (_name, mimeType) => {
    vi.mocked(getClipTimestamps).mockResolvedValue(timestamps(makeRows(3)));
    const log: boolean[] = [];

    render(<Host fileId="f1" on log={log} mimeType={mimeType} />);
    await flush(12);

    expect(log.at(-1)).toBe(true);
    expect(
      await screen.findByRole("slider", { name: "Frame timeline" }),
    ).toBeInTheDocument();
  });

  it.each([
    ["a native video", "video/mp4"],
    ["a .loft", LOFT_MIME],
  ])(
    "ends available with %s on across a file change",
    async (_name, mimeType) => {
      vi.mocked(getClipTimestamps).mockResolvedValue(timestamps(makeRows(3)));
      const log: boolean[] = [];
      const host = (fileId: string, on: boolean) => (
        <Host fileId={fileId} on={on} log={log} mimeType={mimeType} />
      );

      const { rerender } = render(host("f1", false));
      await flush(12);
      rerender(host("f1", true));
      await flush(12);
      expect(log.at(-1)).toBe(true);

      log.length = 0;
      rerender(host("f2", true));
      await flush(12);

      expect(log.at(-1)).toBe(true);
      expect(
        await screen.findByRole("slider", { name: "Frame timeline" }),
      ).toBeInTheDocument();
    },
  );
});
