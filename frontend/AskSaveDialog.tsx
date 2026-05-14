"use client";

import { useTranslations } from "next-intl";

import { FileSaveDialog } from "@/components/FileSaveDialog";
import { saveAskToKnowledge } from "./knowledgeBridge";

interface Props {
  open: boolean;
  drive: string;
  defaultFilename: string;
  content: string;
  sourceFileIds: string[];
  onClose: () => void;
  onSaved: (result: { noteFileId: string; notePath: string }) => void;
}

const DEFAULT_FOLDER = "Ask";

export function AskSaveDialog({
  open,
  drive,
  defaultFilename,
  content,
  sourceFileIds,
  onClose,
  onSaved,
}: Props) {
  const t = useTranslations("askSave");

  async function handleConfirm({
    folder,
    filename,
  }: {
    folder: string;
    filename: string;
  }) {
    const res = await saveAskToKnowledge(drive, {
      folder: folder || DEFAULT_FOLDER,
      filename,
      content,
      source_file_ids: sourceFileIds,
    });
    onSaved({ noteFileId: res.note_file_id, notePath: res.note_path });
    onClose();
  }

  return (
    <FileSaveDialog
      open={open}
      title={t("title")}
      drive={drive}
      defaultFolder={DEFAULT_FOLDER}
      defaultFilename={defaultFilename}
      confirmLabel={t("save")}
      onConfirm={handleConfirm}
      onCancel={onClose}
    />
  );
}
