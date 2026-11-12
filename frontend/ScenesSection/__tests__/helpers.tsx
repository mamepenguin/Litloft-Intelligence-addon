import type { RefObject } from "react";
import { act, fireEvent, screen, waitFor } from "@testing-library/react";
import { expect, vi, type Mock } from "vitest";

import { formatDuration } from "@/lib/format";
import type { MediaController } from "@/lib/mediaController";
import {
  getVideoVisualIndex,
  type ClipTimestampItem,
  type VideoVisualIndexResponse,
  type VideoVisualRunSummary,
  type VideoVisualSceneItem,
} from "@/addons/intelligence/api";

export const LOFT_MIME = "application/vnd.litloft.loft+json";

class PointerEventPolyfill extends MouseEvent {
  readonly pointerId: number;
  readonly pointerType: string;
  readonly isPrimary: boolean;

  constructor(type: string, params: PointerEventInit = {}) {
    super(type, params);
    this.pointerId = params.pointerId ?? 1;
    this.pointerType = params.pointerType ?? "mouse";
    this.isPrimary = params.isPrimary ?? true;
  }
}

/**
 * jsdom has no PointerEvent, so fireEvent falls back to a plain Event and
 * drops pointerType/clientX. Install a MouseEvent-backed stand-in before
 * dispatching any pointer gesture.
 */
export function installPointerEvents(): void {
  (globalThis as unknown as { PointerEvent: typeof PointerEvent }).PointerEvent =
    PointerEventPolyfill as unknown as typeof PointerEvent;

  // jsdom lacks pointer capture. A drag handler that captures the ruler
  // would otherwise throw before it can move the browse point.
  const proto = Element.prototype as Element & {
    setPointerCapture?: (id: number) => void;
    releasePointerCapture?: (id: number) => void;
    hasPointerCapture?: (id: number) => boolean;
  };
  if (typeof proto.setPointerCapture !== "function") {
    proto.setPointerCapture = () => {};
  }
  if (typeof proto.releasePointerCapture !== "function") {
    proto.releasePointerCapture = () => {};
  }
  if (typeof proto.hasPointerCapture !== "function") {
    proto.hasPointerCapture = () => false;
  }
}

export interface SceneSectionProps {
  fileId: string;
  drive: string;
  fileType?: string;
  mimeType?: string;
  videoRef?: RefObject<HTMLVideoElement | null>;
  mediaController?: MediaController | null;
}

export interface ClockState {
  currentTime: number;
  duration: number;
  paused: boolean;
  playbackRate: number;
}

export function makeRows(n: number, step = 5, offset = 0): ClipTimestampItem[] {
  return Array.from({ length: n }, (_, i) => ({
    start: offset + i * step,
    content_preview: `Frame at ${offset + i * step}s`,
  }));
}

export function timestamps(
  rows: ClipTimestampItem[],
  fileId = "f1",
  drive = "d1",
): { file_id: string; drive: string; timestamps: ClipTimestampItem[] } {
  return { file_id: fileId, drive, timestamps: rows };
}

export function makeController(initial: Partial<ClockState> = {}): {
  controller: MediaController;
  state: ClockState;
  seek: Mock;
  play: Mock;
} {
  const state: ClockState = {
    currentTime: 0,
    duration: 0,
    paused: true,
    playbackRate: 1,
    ...initial,
  };
  const seek = vi.fn();
  const play = vi.fn();
  const controller: MediaController = {
    seek,
    play,
    pause: vi.fn(),
    togglePlay: vi.fn(),
    toggleMute: vi.fn(),
    toggleFullscreen: vi.fn(),
    getCurrentTime: () => state.currentTime,
    getDuration: () => state.duration,
    isPaused: () => state.paused,
    isMuted: () => false,
    getVolume: () => 1,
    setVolume: () => {},
    getPlaybackRate: () => state.playbackRate,
    setPlaybackRate: (rate) => {
      state.playbackRate = rate;
    },
    getBufferedFraction: () => 0,
  };
  return { controller, state, seek, play };
}

export function stubRect(el: Element, width = 700): void {
  vi.spyOn(el, "getBoundingClientRect").mockReturnValue({
    left: 0,
    top: 0,
    width,
    height: 44,
    right: width,
    bottom: 44,
    x: 0,
    y: 0,
    toJSON: () => ({}),
  } as DOMRect);
}

export function pointerX(time: number, length: number, width = 700): number {
  return (time / length) * width;
}

export async function advance(
  ms: number,
  mutate?: () => void,
): Promise<void> {
  await act(async () => {
    mutate?.();
    vi.advanceTimersByTime(ms);
  });
}

export async function flush(rounds = 4): Promise<void> {
  await act(async () => {
    for (let i = 0; i < rounds; i += 1) await Promise.resolve();
  });
}

/**
 * The ruler is committed with the timestamps; the Visual Index answer is a
 * separate request drawn in a later commit, so the slider alone does not mean
 * the index part is on screen yet.
 */
export async function waitForVisualIndex(): Promise<void> {
  await waitFor(() => expect(getVideoVisualIndex).toHaveBeenCalled());
  await waitFor(() =>
    expect(screen.queryByTestId("scenes-visual-placeholder")).toBeNull(),
  );
}

export function header(): HTMLElement {
  return screen.getByRole("button", { name: /^Scenes/ });
}

export function ruler(): HTMLElement {
  return screen.getByRole("slider", { name: "Frame timeline" });
}

export function allTiles(): HTMLElement[] {
  return screen.getAllByRole("button", { name: /^Play from / });
}

export function tile(time: number): HTMLElement {
  return screen.getByRole("button", {
    name: `Play from ${formatDuration(time)}`,
  });
}

function tileLabel(el: HTMLElement): string {
  const label = el.getAttribute("aria-label") ?? el.textContent ?? "";
  return label.replace(/^Play from /, "").trim();
}

export function tileLabels(): string[] {
  return allTiles().map(tileLabel);
}

export function classTokens(root: Element): string {
  return [root, ...Array.from(root.querySelectorAll("*"))]
    .map((el) => el.getAttribute("class") ?? "")
    .join(" ");
}

export function hasAccentOutline(el: Element): boolean {
  return /(?:outline|ring|border)-accent\b/.test(classTokens(el));
}

export function accentTiles(): string[] {
  return allTiles().filter(hasAccentOutline).map(tileLabel);
}

export function isFilledAction(button: HTMLElement): boolean {
  return /\bbg-bg-elevated\b/.test(button.className);
}

const ACTION_NAME = /^(Create scenes|Regenerate scenes|Retry the \d+ failed)$/;

function buttonName(el: HTMLElement): string {
  return (el.getAttribute("aria-label") ?? el.textContent ?? "")
    .replace(/\s+/g, " ")
    .trim();
}

export function actionNames(): string[] {
  return screen
    .queryAllByRole("button")
    .filter((el) => ACTION_NAME.test(buttonName(el)))
    .map(buttonName);
}

export function action(name: string): HTMLElement | null {
  return screen.queryByRole("button", { name });
}

export function nameVisible(pattern: RegExp): boolean {
  return (
    screen.queryAllByLabelText(pattern).length > 0 ||
    screen.queryAllByText(pattern).length > 0
  );
}

/**
 * `alt=""` pictures map to the presentation role, so `getAllByRole("img")`
 * misses exactly the decorative tiles this section draws.
 */
export function images(): HTMLImageElement[] {
  return Array.from(document.querySelectorAll("img"));
}

export function imgTimes(): number[] {
  return images()
    .map((img) => img.getAttribute("src") ?? "")
    .map((src) => /[?&]t=([^&]+)/.exec(src)?.[1])
    .filter((value): value is string => value !== undefined)
    .map(Number);
}

export function firePointer(
  el: Element,
  type:
    | "pointerMove"
    | "pointerDown"
    | "pointerUp"
    | "pointerCancel"
    | "pointerLeave",
  opts: {
    pointerType?: string;
    clientX?: number;
    clientY?: number;
    buttons?: number;
  },
): void {
  fireEvent[type](el, {
    pointerType: opts.pointerType ?? "mouse",
    clientX: opts.clientX ?? 0,
    clientY: opts.clientY ?? 10,
    buttons: opts.buttons ?? 0,
  });
}

export function makeRun(
  over: Partial<VideoVisualRunSummary> = {},
): VideoVisualRunSummary {
  return {
    run_id: "run-1",
    status: "succeeded",
    selected_count: 2,
    completed_count: 2,
    succeeded_count: 2,
    failed_count: 0,
    created_at: null,
    completed_at: null,
    ...over,
  };
}

export function makeScene(
  over: Partial<VideoVisualSceneItem> = {},
): VideoVisualSceneItem {
  return {
    ordering: 0,
    start_time: 10,
    end_time: null,
    status: "succeeded",
    scene_type: "slide",
    scene_label: "Scene label",
    visible_text: null,
    transcript_excerpt: null,
    ...over,
  };
}

export function indexState(
  over: Partial<VideoVisualIndexResponse> = {},
): VideoVisualIndexResponse {
  return {
    eligible: true,
    available: true,
    file_id: "f1",
    active_run: null,
    scenes: [],
    staged_run: null,
    stale: false,
    ...over,
  };
}

export function sceneCardNamed(label: string | RegExp): HTMLElement | null {
  return screen.queryByRole("button", { name: label });
}
