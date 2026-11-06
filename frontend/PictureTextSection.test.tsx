/**
 * SPEC-ID: SPEC-ADDON-067
 *
 * The picture page's 画像の文字 section. It is found through the slot table,
 * because the spec fixes the slot id and not the component's file name.
 */

import React, { Suspense } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";

vi.mock("next-intl", async () => await import("./testI18nJa"));
vi.mock("@/components/AddonSlotsProvider", () => ({
  useAddonStatus: () => ({ features: { image_ocr: "false" } }),
  useAddonSlots: () => ({ slots: {} }),
}));

import { slotComponents } from "@/addons/intelligence/slots";

type PictureText = {
  text: string;
  corrected: boolean;
  has_original: boolean;
  busy: boolean;
};

let state: PictureText;
let requests: { method: string; url: string; body: unknown }[];

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
  const url =
    typeof input === "string"
      ? input
      : input instanceof URL
        ? input.toString()
        : input.url;
  const method = init?.method ?? "GET";
  const body = init?.body ? JSON.parse(String(init.body)) : null;
  requests.push({ method, url, body });

  if (url.includes("/picture-text/restore") && method === "POST") {
    state = {
      text: "機械が読んだ一行目\n機械が読んだ二行目",
      corrected: false,
      has_original: false,
      busy: false,
    };
    return json({ file_id: "pic1", ...state });
  }
  if (url.includes("/picture-text") && method === "PUT") {
    state = {
      text: body.text,
      corrected: true,
      has_original: true,
      busy: false,
    };
    return json({ file_id: "pic1", ...state });
  }
  if (url.includes("/picture-text")) {
    return json({ file_id: "pic1", ...state });
  }
  if (/\/files\/pic1$/.test(url)) {
    return json({
      id: "pic1",
      filename: "sign.png",
      file_type: "image",
      mime_type: "image/png",
      drive: "family",
    });
  }
  return json({ detail: "not found" }, 404);
});

vi.stubGlobal("fetch", fetchMock);

function renderSection() {
  const Section = slotComponents["picture-text"];
  if (!Section) {
    throw new Error("the picture-text slot is not registered");
  }
  return render(
    <Suspense fallback={<div>loading</div>}>
      <Section fileId="pic1" drive="family" />
    </Suspense>,
  );
}

beforeEach(() => {
  state = {
    text: "機械が読んだ一行目\n機械が読んだ二行目",
    corrected: false,
    has_original: false,
    busy: false,
  };
  requests = [];
  fetchMock.mockClear();
});

afterEach(() => {
  cleanup();
});

describe("SPEC-ADDON-067 — the section", () => {
  it("shows the stored text line by line", async () => {
    renderSection();

    expect(await screen.findByText("画像の文字")).toBeInTheDocument();
    expect(screen.getByText("機械が読んだ一行目")).toBeInTheDocument();
    expect(screen.getByText("機械が読んだ二行目")).toBeInTheDocument();
  });

  it("shows no section when the text is empty and there is no original", async () => {
    state = { text: "", corrected: false, has_original: false, busy: false };
    renderSection();

    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    expect(screen.queryByText("画像の文字")).not.toBeInTheDocument();
  });

  it("stays visible for an emptied corrected text with an original", async () => {
    state = { text: "", corrected: true, has_original: true, busy: false };
    renderSection();

    expect(await screen.findByText("画像の文字")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "元の結果に戻す" }),
    ).toBeInTheDocument();
  });

  it("marks a corrected text 修正済み", async () => {
    state = {
      text: "直した一行目\n直した二行目",
      corrected: true,
      has_original: true,
      busy: false,
    };
    renderSection();

    expect(await screen.findByText("修正済み")).toBeInTheDocument();
    expect(screen.getByText("直した一行目")).toBeInTheDocument();
    expect(screen.getByText("直した二行目")).toBeInTheDocument();
  });

  it("disables 修正 and 元の結果に戻す while busy", async () => {
    state = {
      text: "機械が読んだ一行目",
      corrected: true,
      has_original: true,
      busy: true,
    };
    renderSection();
    await screen.findByText("画像の文字");

    expect(screen.getByRole("button", { name: "修正" })).toBeDisabled();
    expect(
      screen.getByRole("button", { name: "元の結果に戻す" }),
    ).toBeDisabled();
  });
});

describe("SPEC-ADDON-067 — editing", () => {
  it("keeps line breaks in a multi-line field and saves with Cmd/Ctrl+Enter", async () => {
    renderSection();
    await screen.findByText("画像の文字");
    fireEvent.click(screen.getByRole("button", { name: "修正" }));

    const field = screen.getByRole("textbox");
    expect(field).toHaveValue("機械が読んだ一行目\n機械が読んだ二行目");
    expect(screen.getByRole("button", { name: "保存" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "キャンセル" })).toBeInTheDocument();

    fireEvent.change(field, {
      target: { value: "直した一行目\n直した二行目" },
    });
    fireEvent.keyDown(field, { key: "Enter", ctrlKey: true });

    await waitFor(() => {
      const put = requests.find((request) => request.method === "PUT");
      expect(put).toBeTruthy();
      expect(put!.body).toEqual({
        text: "直した一行目\n直した二行目",
        expected_text: "機械が読んだ一行目\n機械が読んだ二行目",
      });
    });
    expect(await screen.findByText("修正済み")).toBeInTheDocument();
  });

  it("cancels with Escape without writing", async () => {
    renderSection();
    await screen.findByText("画像の文字");
    fireEvent.click(screen.getByRole("button", { name: "修正" }));

    fireEvent.keyDown(screen.getByRole("textbox"), { key: "Escape" });

    expect(screen.queryByRole("textbox")).not.toBeInTheDocument();
    expect(requests.some((request) => request.method === "PUT")).toBe(false);
    expect(screen.getByText("機械が読んだ一行目")).toBeInTheDocument();
  });
});

describe("SPEC-ADDON-067 — restore", () => {
  it("confirms in Japanese and posts the restore", async () => {
    state = {
      text: "直した一行目",
      corrected: true,
      has_original: true,
      busy: false,
    };
    renderSection();
    await screen.findByText("画像の文字");

    fireEvent.click(screen.getByRole("button", { name: "元の結果に戻す" }));
    expect(
      screen.getByText(
        "修正をすべて取り消し、機械が出した元の結果に戻します。",
      ),
    ).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "戻す" }));

    await waitFor(() =>
      expect(
        requests.some(
          (request) =>
            request.method === "POST" &&
            request.url.includes("/picture-text/restore"),
        ),
      ).toBe(true),
    );
    expect(await screen.findByText("機械が読んだ一行目")).toBeInTheDocument();
    expect(screen.queryByText("修正済み")).not.toBeInTheDocument();
  });

  it("closes the confirmation on キャンセル without posting", async () => {
    state = {
      text: "直した一行目",
      corrected: true,
      has_original: true,
      busy: false,
    };
    renderSection();
    await screen.findByText("画像の文字");
    fireEvent.click(screen.getByRole("button", { name: "元の結果に戻す" }));

    fireEvent.click(screen.getByRole("button", { name: "キャンセル" }));

    expect(
      requests.some(
        (request) =>
          request.method === "POST" &&
          request.url.includes("/picture-text/restore"),
      ),
    ).toBe(false);
  });
});
