import { lazy } from "react";
import { CircleSlash } from "lucide-react";

import type { ShelfItemActions } from "@/lib/addons";
import {
  markPickupNotInterested,
  unmarkPickupNotInterested,
} from "./api";
import {
  pickupPageProvider,
  pickupShelfProvider,
  type ShelfPageProvider,
  type ShelfProvider,
} from "./pickupShelfProvider";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const slotComponents: Record<string, React.LazyExoticComponent<React.ComponentType<any>>> = {
  "find-mode": lazy(() => import("./FindModeSlot")),
  // "ask" slot removed in the RAG redesign: the Ask UI lives on a
  // dedicated page at /addons/intelligence. See manifest.json (href)
  // and frontend/Page.tsx.
  "similar-files": lazy(() => import("./SimilarFilesSection")),
  "pickup": lazy(() => import("./PickupWidget")),
  "index-status": lazy(() => import("./IndexStatusWidget")),
  "failed-jobs": lazy(() => import("./FailedJobsAlert")),
  "scenes": lazy(() => import("./ScenesSection")),
  "scene-segments": lazy(() => import("./SceneCandidatesSection")),
  "index-details": lazy(() => import("./IndexDetailsMenuItem")),
"transcript": lazy(() => import("./TranscriptSection")),
  "ask": lazy(() => import("./AskTab")),
  "unverified-source": lazy(() => import("./UnverifiedSourceSection")),
  "suggested-tags": lazy(() => import("./SuggestedTagsSection")),
  "suggested-chapters": lazy(() => import("./SuggestedChaptersSection")),
  "summary": lazy(() => import("./SummarySection")),
  "detailed-summary": lazy(() => import("./DetailedSummarySection")),
  "visual-description": lazy(() => import("./VisualDescriptionSection")),
  "picture-text": lazy(() => import("./PictureTextSection")),
  "folder-ai-actions": lazy(() => import("./FolderAIActionsMenuItems")),
  "folder-ask": lazy(() => import("./FolderAskMenuItem")),
  "collection-ask": lazy(() => import("./CollectionAskButton")),
  "smart-collection-ask": lazy(() => import("./SmartCollectionAskMenuItem")),
  "file-ai-actions": lazy(() => import("./FileAIActionsButton")),
  "folder-refine-transcripts": lazy(() => import("./FolderRefineButton")),
  "admin-transcription": lazy(() => import("./AdminTranscriptionSettingsSection")),
  "admin-features": lazy(() => import("./AdminFeaturesSettingsSection")),
  "admin-llm": lazy(() => import("./AdminLLMSettingsSection")),
  "admin-rag": lazy(() => import("./AdminRAGSettingsSection")),
  "admin-embedding": lazy(() => import("./AdminEmbeddingSettingsSection")),
};

export const slotProviders: Record<string, ShelfProvider> = {
  pickup: pickupShelfProvider,
};

export const slotPageProviders: Record<string, ShelfPageProvider> = {
  pickup: pickupPageProvider,
};

/**
 * The video app's `pickup` cards get one action: mark the video "not
 * interested" for this viewer and drive, with an undo that removes the
 * mark. Labels come through the core's translator over the merged
 * messages, so the action reads in the viewer's language.
 */
export const slotItemActions: Record<string, ShelfItemActions> = {
  pickup: ({ drive, t }) => [
    {
      id: "pickup-not-interested",
      label: t("intelligence.pickup.notInterested"),
      icon: CircleSlash,
      run: async (fileId: string) => {
        await markPickupNotInterested(fileId, drive);
        return {
          message: t("intelligence.pickup.notInterestedToast"),
          undo: async () => {
            await unmarkPickupNotInterested(fileId, drive);
          },
        };
      },
    },
  ],
};
