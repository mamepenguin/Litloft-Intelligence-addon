import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";

vi.mock("next-intl", () => ({
  useTranslations: () => (key: string) => key,
}));

import UnverifiedSourceSection from "./UnverifiedSourceSection";

function trustCall() {
  return (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls.find(
    (c) => String(c[0]).endsWith("/trust-tier"),
  );
}

describe("UnverifiedSourceSection", () => {
  beforeEach(() => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: true,
        json: async () => ({ id: "f1", trust_tier: "verified" }),
      })),
    );
  });

  it("asks only about unverified files nobody has ruled on", () => {
    const { container, rerender } = render(
      <UnverifiedSourceSection
        fileId="f1" trustTier="verified" trustReviewedAt={null}
      />,
    );
    expect(container.firstChild).toBeNull();

    // Already dismissed once: re-asking on every open would be nagging.
    rerender(
      <UnverifiedSourceSection
        fileId="f1" trustTier="unverified"
        trustReviewedAt="2026-08-29T00:00:00Z"
      />,
    );
    expect(container.firstChild).toBeNull();
  });

  it("renders for an unreviewed unverified file", async () => {
    render(
      <UnverifiedSourceSection
        fileId="f1" trustTier="unverified" trustReviewedAt={null}
      />,
    );
    expect(await screen.findByText("title")).toBeTruthy();
  });

  it("promotes on trust", async () => {
    const onFileChange = vi.fn();
    render(
      <UnverifiedSourceSection
        fileId="f1" trustTier="unverified" trustReviewedAt={null}
        onFileChange={onFileChange}
      />,
    );

    fireEvent.click(await screen.findByText("trust"));

    await waitFor(() => expect(onFileChange).toHaveBeenCalled());
    const call = trustCall();
    expect(call?.[0]).toBe("/api/files/f1/trust-tier");
    expect(JSON.parse(call![1].body)).toEqual({ tier: "verified" });
  });

  it("dismiss records the judgement without granting trust", async () => {
    render(
      <UnverifiedSourceSection
        fileId="f1" trustTier="unverified" trustReviewedAt={null}
      />,
    );

    fireEvent.click(await screen.findByText("dismiss"));

    await waitFor(() => expect(trustCall()).toBeTruthy());
    // Same tier it already had: the write exists to stamp the review, which
    // is what stops the panel coming back.
    expect(JSON.parse(trustCall()![1].body)).toEqual({ tier: "unverified" });
  });

  it("fetches nothing until the viewer answers", async () => {
    render(
      <UnverifiedSourceSection
        fileId="f1" trustTier="unverified" trustReviewedAt={null}
      />,
    );
    await screen.findByText("title");

    // The panel is the question and nothing else. Evidence-gathering
    // the panel no longer fetches evidence of its own.
    expect((globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls).toHaveLength(0);
  });
});
