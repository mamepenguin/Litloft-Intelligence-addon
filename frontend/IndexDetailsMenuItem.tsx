"use client";

/**
 * `file-actions-menu` entry that opens `IndexDetailsDialog`.
 *
 * Indexing state is operator-facing: a reader opening a file never needs
 * it, so it sits in the `[...]` menu rather than in the inspector.
 *
 * The host does not close its menu for us — doing so would unmount this
 * component and take the dialog with it. We report the dialog through
 * `onDialogOpenChange` and ask for the close only once it is dismissed.
 */

import { useCallback, useState } from "react";
import { useTranslations } from "next-intl";
import { Database } from "lucide-react";

import { ActionMenuItem } from "@/components/ActionMenuItem";
import IndexDetailsDialog from "./IndexDetailsDialog";

interface IndexDetailsMenuItemProps {
  fileId: string;
  drive: string;
  mimeType?: string;
  fileType?: string;
  onRequestClose?: () => void;
  onDialogOpenChange?: (open: boolean) => void;
}

export default function IndexDetailsMenuItem({
  fileId,
  drive,
  mimeType,
  fileType,
  onRequestClose,
  onDialogOpenChange,
}: IndexDetailsMenuItemProps) {
  const tDetails = useTranslations("semanticSearch.indexDetails");
  const [open, setOpen] = useState(false);

  const handleOpen = useCallback(() => {
    setOpen(true);
    onDialogOpenChange?.(true);
  }, [onDialogOpenChange]);

  const handleClose = useCallback(() => {
    setOpen(false);
    onDialogOpenChange?.(false);
    onRequestClose?.();
  }, [onDialogOpenChange, onRequestClose]);

  return (
    <>
      <ActionMenuItem
        icon={Database}
        label={tDetails("title")}
        onClick={handleOpen}
      />
      <IndexDetailsDialog
        open={open}
        fileId={fileId}
        drive={drive}
        mimeType={mimeType}
        fileType={fileType}
        onClose={handleClose}
      />
    </>
  );
}
