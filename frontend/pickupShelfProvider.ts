import { fetchPickup, localDateStamp } from "./api";

export type ShelfProvider = (args: {
  drive: string;
  limit: number;
  restrict: (ids: string[]) => Promise<string[]>;
}) => Promise<string[]>;

const PAGE_SIZE = 60;
const POOL_SIZE = 40;
// The feed holds at most 300 rows, so this covers it in pages of 60.
const MAX_PAGES = 5;

function hashSeed(seed: string): number {
  let hash = 2166136261;
  for (let i = 0; i < seed.length; i += 1) {
    hash ^= seed.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function mulberry32(seed: number): () => number {
  let state = seed;
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function seededShuffle<T>(items: T[], seed: string): T[] {
  const out = items.slice();
  const random = mulberry32(hashSeed(seed));
  for (let i = out.length - 1; i > 0; i -= 1) {
    const j = Math.floor(random() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

export async function pickupShelfProvider({
  drive,
  limit,
  restrict,
}: Parameters<ShelfProvider>[0]): Promise<string[]> {
  const matched: string[] = [];
  let offset = 0;

  for (let page = 0; page < MAX_PAGES && matched.length < POOL_SIZE; page += 1) {
    const { file_ids, total } = await fetchPickup(drive, {
      limit: PAGE_SIZE,
      offset,
    });
    matched.push(...(await restrict(file_ids)));
    offset += file_ids.length;
    if (file_ids.length < PAGE_SIZE || offset >= total) break;
  }

  // The home carousel samples twelve from the top forty; the shelf keeps
  // that window, applied after the target filter rather than before it.
  const pool = matched.slice(0, POOL_SIZE);
  const seed = `${drive}:${localDateStamp()}`;
  return seededShuffle(pool, seed).slice(0, limit);
}
