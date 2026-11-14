/**
 * An Ask scope's `in:` chips, resolved and evaluated (SPEC-ADDON-110).
 *
 * A chip names a collection when the drive has one whose name equals it
 * ignoring case, and a Smart Collection when only that matches. A Smart
 * Collection's members are `evaluateSmartCollection`'s answer for its saved
 * query and type, which the server cannot compute (meaning search lives in
 * this addon).
 */

import { getDriveFiles } from "@/lib/api";
import { getSemanticAvailability } from "@/lib/addons";
import { fetchSemanticHits } from "@/lib/semanticSearch";
import { evaluateSmartCollection } from "@/lib/smartCollectionEval";
import type { SmartFolder } from "@/types/smartFolder";

/** The server refuses more; so does the page before sending. */
export const SCOPE_FILE_ID_LIMIT = 10_000;

export interface SmartScopeResolution {
  /** The chips to send in `collections`: a collection's, or one that names nothing. */
  collections: string[];
  /** The chips that name only a Smart Collection, in the order they appear. */
  smartCollections: SmartFolder[];
}

export function resolveSmartScope(
  names: readonly string[],
  collections: readonly { name: string }[],
  smartFolders: readonly SmartFolder[],
): SmartScopeResolution {
  const collectionNames = new Set(
    collections.map((collection) => collection.name.toLowerCase()),
  );
  const smartByName = new Map(
    smartFolders.map((folder) => [folder.name.toLowerCase(), folder]),
  );
  const sent: string[] = [];
  const smart: SmartFolder[] = [];
  for (const name of names) {
    const key = name.toLowerCase();
    const folder = smartByName.get(key);
    if (collectionNames.has(key) || folder === undefined) {
      sent.push(name);
      continue;
    }
    smart.push(folder);
  }
  return { collections: sent, smartCollections: smart };
}

/**
 * The union of the Smart Collections' members in the order first seen, or
 * `null` when one of them could not be read completely: a partial union would
 * silently answer within fewer files than the chips name.
 */
export async function evaluateSmartMembers(
  drive: string,
  smartCollections: readonly SmartFolder[],
): Promise<string[] | null> {
  const ids: string[] = [];
  const seen = new Set<string>();
  for (const smart of smartCollections) {
    let result;
    try {
      result = await evaluateSmartCollection(
        {
          drive,
          query: smart.query,
          type: smart.file_type,
          sort: "relevance",
          order: "desc",
          includeSceneClip: false,
        },
        { getDriveFiles, getSemanticAvailability, fetchSemanticHits },
      );
    } catch {
      return null;
    }
    if (!result.complete) return null;
    for (const file of result.files) {
      if (seen.has(file.id)) continue;
      seen.add(file.id);
      ids.push(file.id);
    }
  }
  return ids;
}
