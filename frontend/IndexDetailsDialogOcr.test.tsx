/**
 * SPEC-ADDON-050 I4: the Index details dialog lists OCR for a picture, with its
 * status, and offers to run it again.
 *
 * The mode rides on the index-details payload (`image_ocr`), so a dialog opened
 * on an install with OCR off has no row to offer.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { cleanup, render, screen, fireEvent, waitFor, within } from "@testing-library/react";

vi.mock("@/addons/intelligence/api", () => ({
  getIndexDetails: vi.fn(),
  reindexFile: vi.fn(),
}));

vi.mock("@/components/ConfirmDialog", () => ({
  ConfirmDialog: ({
    open,
    onConfirm,
    onCancel,
  }: {
    open: boolean;
    onConfirm: () => void;
    onCancel: () => void;
  }) =>
    open ? (
      <div data-testid="confirm-dialog">
        <button onClick={onConfirm}>dialog-confirm</button>
        <button onClick={onCancel}>dialog-cancel</button>
      </div>
    ) : null,
}));

import IndexDetailsMenuItem from "@/addons/intelligence/IndexDetailsMenuItem";
import { ShortcutsProvider } from "@/components/ShortcutsProvider";
import { getIndexDetails, reindexFile } from "@/addons/intelligence/api";

const mockedGet = getIndexDetails as unknown as ReturnType<typeof vi.fn>;
const mockedReindex = reindexFile as unknown as ReturnType<typeof vi.fn>;

const ENTRY_LABEL = /index details|インデックス詳細|semanticSearch\.indexDetails\.title/i;
const OCR_LABEL = /^Text in image$|semanticSearch\.tasks\.ocr\.label/;

function details(overrides: Record<string, unknown> = {}) {
  return {
    file_id: "abc12345",
    drive: "drive1",
    filename: "slide.png",
    image_ocr: "on_index",
    status: { metadata: true, clip: true, whisper: true, text: false, ocr: false },
    indexed_at: "2026-11-03T10:00:00Z",
    embeddings: {},
    provider_stats: {},
    ...overrides,
  };
}

function open(props: { mimeType?: string; fileType?: string } = {}) {
  // A test opens twice to compare two payloads; the first tree has to go
  // or both menu entries match.
  cleanup();
  render(
    <ShortcutsProvider>
      <IndexDetailsMenuItem
        fileId="abc12345"
        drive="drive1"
        mimeType={props.mimeType ?? "image/png"}
        fileType={props.fileType ?? "image"}
      />
    </ShortcutsProvider>,
  );
  fireEvent.click(screen.getByRole("menuitem", { name: ENTRY_LABEL }));
}

async function ocrRow(): Promise<HTMLElement | null> {
  const label = await screen.findByText(OCR_LABEL);
  return label.closest("[data-task-row]") as HTMLElement | null;
}

beforeEach(() => {
  vi.clearAllMocks();
  mockedGet.mockResolvedValue(details());
  mockedReindex.mockResolvedValue({ status: "accepted" });
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("IndexDetailsDialog OCR row (SPEC-ADDON-050 I4)", () => {
  it("lists OCR for a picture and hides it when image_ocr is false", async () => {
    open({ mimeType: "image/png", fileType: "image" });

    const row = await ocrRow();
    expect(row).not.toBeNull();
    expect(within(row as HTMLElement).getByText(/Pending|未処理/)).toBeInTheDocument();

    screen.getByRole("button", { name: /close|閉じる/i }).click();
    mockedGet.mockResolvedValue(details({ image_ocr: "false" }));
    open({ mimeType: "image/png", fileType: "image" });

    await screen.findByText(/^Metadata extraction$|semanticSearch\.tasks\.metadata/);
    await waitFor(() => expect(screen.queryByText(OCR_LABEL)).toBeNull());
  });

  it("lists OCR for a picture and not for a file that is not one", async () => {
    open({ mimeType: "image/heic", fileType: "image" });
    expect(await ocrRow()).not.toBeNull();

    screen.getByRole("button", { name: /close|閉じる/i }).click();
    mockedGet.mockResolvedValue(details({ filename: "notes.md", image_ocr: "on_index" }));
    open({ mimeType: "text/markdown", fileType: "text" });

    await screen.findByText(/^Metadata extraction$|semanticSearch\.tasks\.metadata/);
    await waitFor(() => expect(screen.queryByText(OCR_LABEL)).toBeNull());
  });

  it("shows the OCR status and reindexes it on request", async () => {
    mockedGet.mockResolvedValue(
      details({ status: { metadata: true, clip: true, whisper: true, text: false, ocr: true } }),
    );
    open({ mimeType: "image/png", fileType: "image" });

    const row = await ocrRow();
    expect(row).not.toBeNull();
    expect(within(row as HTMLElement).getByText(/Indexed|インデックス済/)).toBeInTheDocument();

    fireEvent.click(
      within(row as HTMLElement).getByRole("button", { name: /regenerate|再生成/i }),
    );
    const confirm = await screen.findByTestId("confirm-dialog");
    fireEvent.click(within(confirm).getByText("dialog-confirm"));

    await waitFor(() => expect(mockedReindex).toHaveBeenCalled());
    expect(mockedReindex.mock.calls[0][0]).toBe("abc12345");
    expect(mockedReindex.mock.calls[0][1]).toEqual(["ocr"]);
  });
});
