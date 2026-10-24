import type { DatasetFacets, DatasetFilters, LengthRange } from './use-fish-dataset';

/** The chapters filter says "no chapter at all" with this one empty name (an empty list means every chapter). */
export const NO_CHAPTERS = '';

export type TagMode = 'any' | 'all';

/**
 * What the owner has picked in the Explorer but not applied yet. A missing key means "untouched": it
 * follows the applied filters, so a typed filter elsewhere never leaves a stale copy behind.
 */
export interface Staged {
  chapters?: ReadonlySet<string>;
  include?: readonly string[];
  exclude?: readonly string[];
  mode?: TagMode;
  ranges?: readonly LengthRange[];
}

export const rangeKey = (r: LengthRange) => `${r[0]}-${r[1] ?? ''}`;

export const appliedTagMode = (f: DatasetFilters): TagMode => (f.tagsAll.length > 0 ? 'all' : 'any');
export const appliedInclude = (f: DatasetFilters) => (f.tagsAll.length > 0 ? f.tagsAll : f.tagsAny);

/** Chapters ticked in the applied filters: every listed one when there is no chapter filter. */
export function appliedChapters(f: DatasetFilters, facets: DatasetFacets): Set<string> {
  const names = facets.chapters.map((c) => c.chapter);
  if (f.chapters.length === 0) return new Set(names);
  const chosen = new Set(f.chapters);
  return new Set(names.filter((n) => chosen.has(n)));
}

const symmetricDifference = (a: Iterable<string>, b: Iterable<string>) => {
  const left = new Set(a);
  const right = new Set(b);
  let n = 0;
  left.forEach((x) => right.has(x) || n++);
  right.forEach((x) => left.has(x) || n++);
  return n;
};

/** How many individual choices differ from the applied filters ("3 changes not applied"). */
export function countChanges(staged: Staged, f: DatasetFilters, facets: DatasetFacets): number {
  let n = 0;
  if (staged.chapters) n += symmetricDifference(staged.chapters, appliedChapters(f, facets));
  const include = staged.include ?? appliedInclude(f);
  if (staged.include) n += symmetricDifference(staged.include, appliedInclude(f));
  if (staged.exclude) n += symmetricDifference(staged.exclude, f.tagsNone);
  if (staged.mode && staged.mode !== appliedTagMode(f) && include.length > 0) n += 1;
  if (staged.ranges) n += symmetricDifference(staged.ranges.map(rangeKey), f.durRanges.map(rangeKey));
  return n;
}

/** The filter patch for the staged choices. Ticking every listed chapter clears the filter again. */
export function toPatch(staged: Staged, f: DatasetFilters, facets: DatasetFacets): Partial<DatasetFilters> {
  const patch: Partial<DatasetFilters> = {};
  if (staged.chapters) {
    const listed = facets.chapters.map((c) => c.chapter);
    const ticked = listed.filter((name) => staged.chapters!.has(name));
    patch.chapters = ticked.length === listed.length ? [] : ticked.length === 0 ? [NO_CHAPTERS] : ticked;
  }
  if (staged.include || staged.mode || staged.exclude) {
    const include = [...(staged.include ?? appliedInclude(f))];
    const mode = staged.mode ?? appliedTagMode(f);
    patch.tagsAll = mode === 'all' ? include : [];
    patch.tagsAny = mode === 'any' ? include : [];
    if (staged.exclude) patch.tagsNone = [...staged.exclude];
  }
  if (staged.ranges) patch.durRanges = [...staged.ranges].sort((a, b) => a[0] - b[0]).map((r) => [r[0], r[1]] as LengthRange);
  return patch;
}

/** "1–2, 4–6, 20+": adjacent bars read as one range. */
export function describeRanges(ranges: readonly LengthRange[]): string {
  const sorted = [...ranges].sort((a, b) => a[0] - b[0]);
  const merged: LengthRange[] = [];
  for (const [from, to] of sorted) {
    const last = merged[merged.length - 1];
    if (last && last[1] !== null && last[1] >= from) last[1] = to === null ? null : Math.max(last[1], to);
    else merged.push([from, to]);
  }
  return merged.map(([from, to]) => (to === null ? `${from}+` : `${from}–${to}`)).join(', ');
}

export function median(values: number[]): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}
