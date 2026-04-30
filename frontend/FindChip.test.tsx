/**
 * Unit tests for the FindChip component.
 *
 * Spec: ``2026-04-30-intelligence-find-mode.md`` §3.1 (chip の意味と
 * 編集セマンティクス). Each chip shows a label + a × button. Clicking
 * × invokes ``onRemove`` so the parent page can rebuild ``overrides``
 * and re-POST. The chip is also a visible accessibility surface — the
 * × button must have an aria-label that names the slot being cleared.
 */

import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup } from "@testing-library/react";
import React from "react";

import FindChip from "./FindChip";

afterEach(() => cleanup());

describe("FindChip", () => {
  it("renders the supplied label", () => {
    render(
      <FindChip
        label="先週 (4/23-4/30)"
        slot="time_range"
        onRemove={() => {}}
      />,
    );
    expect(screen.getByText(/先週 \(4\/23-4\/30\)/)).toBeInTheDocument();
  });

  it("calls onRemove when the × button is clicked", () => {
    const onRemove = vi.fn();
    render(
      <FindChip label="視聴済み" slot="personal_scope" onRemove={onRemove} />,
    );

    fireEvent.click(
      screen.getByRole("button", { name: /視聴済み|personal_scope|remove/i }),
    );
    expect(onRemove).toHaveBeenCalledTimes(1);
  });

  it("the × button has an accessible name that mentions the chip label", () => {
    render(
      <FindChip label="video" slot="file_type_hint" onRemove={() => {}} />,
    );
    const btn = screen.getByRole("button");
    const accessibleName =
      btn.getAttribute("aria-label") ?? btn.textContent ?? "";
    expect(accessibleName.toLowerCase()).toContain("video");
  });
});
