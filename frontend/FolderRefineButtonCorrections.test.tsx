/**
 * SPEC-ID: SPEC-ADDON-068
 *
 * Folder Refine reports the corrected files it left out.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";

vi.mock("next-intl", async () => await import("./testI18nJa"));

import FolderRefineButton from "@/addons/intelligence/FolderRefineButton";

let skipped: string[];

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
  const url =
    typeof input === "string"
      ? input
      : input instanceof URL
        ? input.toString()
        : input.url;
  if (url.includes("/refine/folders")) {
    return json({
      queued: 1,
      file_ids: ["normal"],
      skipped_corrected: skipped,
    });
  }
  return json({ detail: "not found" }, 404);
});

vi.stubGlobal("fetch", fetchMock);

function renderButton() {
  return render(
    <FolderRefineButton drive="family" fileIds={["normal", "corrected"]} />,
  );
}

beforeEach(() => {
  skipped = [];
  fetchMock.mockClear();
});

afterEach(() => {
  cleanup();
});

describe("SPEC-ADDON-068 — folder Refine", () => {
  it("reports how many corrected files it skipped", async () => {
    skipped = ["corrected"];
    renderButton();

    fireEvent.click(screen.getByRole("button"));

    expect(
      await screen.findByText("手で直したファイル 1 件は対象外にしました"),
    ).toBeInTheDocument();
  });

  it("shows no skip line when none were corrected", async () => {
    skipped = [];
    renderButton();

    fireEvent.click(screen.getByRole("button"));

    expect(
      await screen.findByText(/1 件を処理待ちに追加しました/),
    ).toBeInTheDocument();
    expect(
      screen.queryByText(/手で直したファイル/),
    ).not.toBeInTheDocument();
  });
});
