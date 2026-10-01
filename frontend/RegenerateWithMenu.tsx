"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useTranslations } from "next-intl";
import { ChevronDown } from "lucide-react";

import { DismissScrim } from "@/components/DismissScrim";
import { useMenuSurface } from "@/components/ToolbarMenu";
import { useShortcuts } from "@/hooks/useShortcuts";
import { OVERLAY_PRIORITY } from "@/lib/shortcuts";

import type { LLMChoice } from "./api";
import { choiceLabel } from "./llmChoice";

interface RegenerateWithMenuProps {
  choices: LLMChoice[];
  disabled: boolean;
  onChoose: (profile: string) => void;
}

/**
 * The "▾" beside a regenerate button. Portalled on a phone because the
 * file page's inspector is a transformed sheet, where the menu's `fixed`
 * bottom sheet would resolve against the sheet rather than the screen.
 */
export function RegenerateWithMenu({
  choices,
  disabled,
  onChoose,
}: RegenerateWithMenuProps) {
  const t = useTranslations("llmChoice");
  const [open, setOpen] = useState(false);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const surface = useMenuSurface(open, "end", undefined, {
    portalOnPhone: true,
  });
  const label = t("regenerateWith");

  const close = useCallback(() => {
    setOpen(false);
    triggerRef.current?.focus();
  }, []);

  const items = useCallback(
    () =>
      Array.from(
        surface.panelRef.current?.querySelectorAll<HTMLButtonElement>(
          '[role="menuitem"]:not(:disabled)',
        ) ?? [],
      ),
    [surface.panelRef],
  );

  useEffect(() => {
    if (open) items()[0]?.focus();
  }, [open, items]);

  const moveTo = (pick: (at: number, count: number) => number) => {
    const list = items();
    if (list.length === 0) return;
    const at = list.indexOf(document.activeElement as HTMLButtonElement);
    list[pick(at, list.length)].focus();
  };

  // On the shortcut stack rather than in `onKeyDown`: React and
  // `ShortcutsProvider` both listen on `document`, so `stopPropagation`
  // cannot keep these keys from the page's own bindings (video seek on the
  // arrows, collapsing citations on Escape). The open menu answers first.
  useShortcuts(
    "intelligence-regenerate-with",
    "Menu",
    [
      { key: "escape", label: "Close", editingOnly: false, hidden: true, handler: close },
      {
        key: "arrowdown",
        label: "Next",
        editingOnly: false,
        hidden: true,
        handler: () => moveTo((at, n) => (at < 0 ? 0 : (at + 1) % n)),
      },
      {
        key: "arrowup",
        label: "Previous",
        editingOnly: false,
        hidden: true,
        handler: () => moveTo((at, n) => (at < 0 ? n - 1 : (at - 1 + n) % n)),
      },
      {
        key: "home",
        label: "First",
        editingOnly: false,
        hidden: true,
        handler: () => moveTo(() => 0),
      },
      {
        key: "end",
        label: "Last",
        editingOnly: false,
        hidden: true,
        handler: () => moveTo((_, n) => n - 1),
      },
    ],
    open,
    OVERLAY_PRIORITY,
  );

  return (
    <div
      ref={surface.wrapperRef}
      className="relative flex items-center"
    >
      <button
        ref={triggerRef}
        type="button"
        onClick={() => setOpen((s) => !s)}
        disabled={disabled}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label={label}
        title={label}
        className="flex items-center rounded-lg px-1 py-1 text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary disabled:opacity-50 pointer-coarse:min-h-11 pointer-coarse:min-w-11 pointer-coarse:justify-center"
      >
        <ChevronDown size={11} />
      </button>
      {open &&
        surface.layer(
          <DismissScrim onDismiss={() => setOpen(false)}>
            <div
              ref={surface.panelRef}
              role="menu"
              aria-label={label}
              className={surface.className}
            >
              {choices.map((choice) => (
                <button
                  key={choice.name}
                  type="button"
                  role="menuitem"
                  disabled={disabled}
                  aria-label={choiceLabel(choice, t("offhost"))}
                  onClick={() => {
                    close();
                    onChoose(choice.name);
                  }}
                  className="flex w-full flex-wrap items-baseline gap-x-2 px-3 py-2 text-left text-sm text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary focus-visible:-outline-offset-2 disabled:opacity-50 disabled:hover:bg-transparent"
                >
                  <span className="min-w-0 break-words">
                    {choice.name} — {choice.model}
                  </span>
                  {choice.offhost && (
                    <span className="text-xs text-text-muted/80">
                      {t("offhost")}
                    </span>
                  )}
                </button>
              ))}
            </div>
          </DismissScrim>,
        )}
    </div>
  );
}
