import { lazy } from "react";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const slotComponents: Record<string, React.LazyExoticComponent<React.ComponentType<any>>> = {
  "semantic-search": lazy(() => import("./SemanticSearchSlot")),
  "find-mode": lazy(() => import("./FindModeSlot")),
  // "ask" slot removed in the RAG redesign: the Ask UI lives on a
  // dedicated page at /addons/intelligence. See manifest.json (href)
  // and frontend/Page.tsx.
  "similar-files": lazy(() => import("./SimilarFilesSection")),
  "pickup": lazy(() => import("./PickupWidget")),
  "index-status": lazy(() => import("./IndexStatusWidget")),
  "clip-frames": lazy(() => import("./ClipFramesSection")),
  "index-details": lazy(() => import("./IndexDetailsSection")),
  "transcript": lazy(() => import("./TranscriptSection")),
  "suggested-tags": lazy(() => import("./SuggestedTagsSection")),
  "summary": lazy(() => import("./SummarySection")),
  "detailed-summary": lazy(() => import("./DetailedSummarySection")),
  "visual-description": lazy(() => import("./VisualDescriptionSection")),
  "folder-ai-actions": lazy(() => import("./FolderAIActionsButton")),
  "folder-refine-transcripts": lazy(() => import("./FolderRefineButton")),
};
