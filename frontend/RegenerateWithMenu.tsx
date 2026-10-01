"use client";

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
} from "react";
import { useTranslations } from "next-intl";
import { ChevronDown } from "lucide-react";

import { DismissScrim } from "@/components/DismissScrim";
import { useMenuSurface } from "@/components/ToolbarMenu";

import type { LLMChoice } from "./api";

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

  const handleKeyDown = (e: ReactKeyboardEvent<HTMLDivElement>) => {
    if (!open) return;
    if (e.key === "Escape") {
      e.stopPropagation();
      close();
      return;
    }
    const list = items();
    if (list.length === 0) return;
    const at = list.indexOf(document.activeElement as HTMLButtonElement);
    let next: number | null = null;
    if (e.key === "ArrowDown") next = at < 0 ? 0 : (at + 1) % list.length;
    else if (e.key === "ArrowUp")
      next = at < 0 ? list.length - 1 : (at - 1 + list.length) % list.length;
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = list.length - 1;
    if (next === null) return;
    e.preventDefault();
    list[next].focus();
  };

  return (
    <div
      ref={surface.wrapperRef}
      className="relative flex items-center"
      onKeyDown={handleKeyDown}
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
