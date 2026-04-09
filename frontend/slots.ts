import { lazy } from "react";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const slotComponents: Record<string, React.LazyExoticComponent<React.ComponentType<any>>> = {
  "semantic-search": lazy(() => import("./SemanticSearchSlot")),
  "similar-files": lazy(() => import("./SimilarFilesSection")),
  "index-status": lazy(() => import("./IndexStatusWidget")),
  "clip-frames": lazy(() => import("./ClipFramesSection")),
  "index-details": lazy(() => import("./IndexDetailsSection")),
  "transcript": lazy(() => import("./TranscriptSection")),
};
