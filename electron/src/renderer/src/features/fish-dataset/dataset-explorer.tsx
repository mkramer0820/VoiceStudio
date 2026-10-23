import { useTranslation } from 'react-i18next';
import { cn } from '@/lib/utils';
import type { DatasetFacets, DatasetFilters } from './use-fish-dataset';

const minutes = (seconds: number) => (seconds / 60).toFixed(seconds >= 6000 ? 0 : 1);

function median(values: number[]): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/**
 * Clickable breakdown of whatever the filters currently match: lengths, tags and chapters.
 * Clicking narrows the filters, so zooming into a character, a chapter or a kind of line is
 * a few clicks and every number on screen follows.
 */
export function DatasetExplorer({
  facets,
  filters,
  onPatch,
}: {
  facets: DatasetFacets;
  filters: DatasetFilters;
  onPatch: (patch: Partial<DatasetFilters>) => void;
}) {
  const { t } = useTranslation();
  const maxClips = Math.max(1, ...facets.duration.map((b) => b.clips));
  const lo = filters.minDur.trim() === '' ? null : Number(filters.minDur);
  const hi = filters.maxDur.trim() === '' ? null : Number(filters.maxDur);
  const inRange = (from: number, to: number | null) =>
    (lo !== null || hi !== null) && (lo ?? 0) <= from && (hi === null || (to !== null && to <= hi));

  const pickBin = (from: number, to: number | null) => {
    const same = lo === from && (to === null ? hi === null : hi === to);
    onPatch(same ? { minDur: '', maxDur: '' } : { minDur: String(from), maxDur: to === null ? '' : String(to) });
  };

  const tagState = (tag: string) =>
    filters.tagsAll.includes(tag) ? 'all' : filters.tagsNone.includes(tag) ? 'none' : 'off';
  const cycleTag = (tag: string) => {
    const state = tagState(tag);
    onPatch({
      tagsAll: state === 'off' ? [...filters.tagsAll, tag] : filters.tagsAll.filter((x) => x !== tag),
      tagsNone:
        state === 'all'
          ? [...filters.tagsNone, tag]
          : filters.tagsNone.filter((x) => x !== tag),
    });
  };
  const shownTags = [
    ...facets.tags.slice(0, 40),
    ...[...filters.tagsAll, ...filters.tagsNone]
      .filter((tag) => !facets.tags.slice(0, 40).some(([name]) => name === tag))
      .map((tag): [string, number] => [tag, 0]),
  ];

  const toggleChapter = (chapter: string) =>
    onPatch({
      chapters: filters.chapters.includes(chapter)
        ? filters.chapters.filter((c) => c !== chapter)
        : [...filters.chapters, chapter],
    });
  const maxSeconds = Math.max(1, ...facets.chapters.map((c) => c.seconds));
  const middle = median(facets.chapters.flatMap((c) => (c.rms_db === null ? [] : [c.rms_db])));

  return (
    <div className="grid gap-4 border-b border-border/50 p-3 text-xs @3xl:grid-cols-2">
      <section aria-label={t('fishDataset.chart_length')} className="space-y-1.5">
        <h3 className="font-medium">{t('fishDataset.chart_length')}</h3>
        <div className="flex h-20 items-end gap-0.5" role="group">
          {facets.duration.map((bin) => (
            <button
              key={bin.from}
              type="button"
              aria-pressed={inRange(bin.from, bin.to)}
              title={t('fishDataset.bin_title', {
                range: bin.to === null ? `${bin.from}+` : `${bin.from}–${bin.to}`,
                clips: bin.clips,
              })}
              onClick={() => pickBin(bin.from, bin.to)}
              className="group flex h-full min-w-0 flex-1 flex-col justify-end outline-none"
            >
              <span
                className={cn(
                  'w-full rounded-t-sm transition-colors group-focus-visible:ring-2 group-focus-visible:ring-ring',
                  inRange(bin.from, bin.to) ? 'bg-primary' : 'bg-primary/35 group-hover:bg-primary/60',
                )}
                style={{ height: `${Math.max(bin.clips ? 4 : 1, (bin.clips / maxClips) * 100)}%` }}
              />
            </button>
          ))}
        </div>
        <div className="flex gap-0.5 text-[9px] text-muted-foreground tabular-nums" aria-hidden>
          {facets.duration.map((bin) => (
            <span key={bin.from} className="min-w-0 flex-1 text-center">
              {bin.from}
            </span>
          ))}
        </div>
        <p className="text-[11px] text-muted-foreground">
          {t('fishDataset.chart_length_hint')}
          {facets.words &&
            ` · ${t('fishDataset.words_stats', { min: facets.words.min, avg: facets.words.avg, max: facets.words.max })}`}
        </p>
      </section>

      <section aria-label={t('fishDataset.chart_tags')} className="space-y-1.5">
        <h3 className="font-medium">{t('fishDataset.chart_tags')}</h3>
        <div className="flex max-h-28 flex-wrap gap-1 overflow-y-auto">
          {shownTags.map(([tag, count]) => {
            const state = tagState(tag);
            return (
              <button
                key={tag}
                type="button"
                data-state={state}
                aria-label={t(`fishDataset.tag_${state}`, { tag })}
                onClick={() => cycleTag(tag)}
                className={cn(
                  'rounded-full px-2 py-0.5 text-[10px] outline-none focus-visible:ring-2 focus-visible:ring-ring',
                  state === 'all' && 'bg-emerald-500/20 text-emerald-500',
                  state === 'none' && 'bg-red-500/20 text-red-500 line-through',
                  state === 'off' && 'bg-muted/50 text-muted-foreground hover:bg-muted',
                )}
              >
                [{tag}] {count}
              </button>
            );
          })}
        </div>
        <p className="text-[11px] text-muted-foreground">{t('fishDataset.chart_tags_hint')}</p>
      </section>

      <section aria-label={t('fishDataset.chart_chapters')} className="space-y-1.5 @3xl:col-span-2">
        <h3 className="font-medium">{t('fishDataset.chart_chapters')}</h3>
        <ul className="grid max-h-44 gap-x-4 gap-y-0.5 overflow-y-auto @3xl:grid-cols-2">
          {facets.chapters.map((chapter) => {
            const on = filters.chapters.includes(chapter.chapter);
            const off = middle !== null && chapter.rms_db !== null && Math.abs(chapter.rms_db - middle) >= 6;
            return (
              <li key={chapter.chapter}>
                <button
                  type="button"
                  aria-pressed={on}
                  onClick={() => toggleChapter(chapter.chapter)}
                  className={cn(
                    'relative flex w-full items-center gap-2 overflow-hidden rounded px-1.5 py-0.5 text-left outline-none focus-visible:ring-2 focus-visible:ring-ring',
                    on ? 'bg-primary/15' : 'hover:bg-muted/50',
                  )}
                >
                  <span
                    aria-hidden
                    className="absolute inset-y-0 left-0 bg-primary/10"
                    style={{ width: `${(chapter.seconds / maxSeconds) * 100}%` }}
                  />
                  <span className="relative min-w-0 flex-1 truncate">{chapter.chapter}</span>
                  <span className="relative tabular-nums text-muted-foreground">
                    {chapter.clips} · {minutes(chapter.seconds)} min
                  </span>
                  {chapter.rms_db !== null && (
                    <span
                      className={cn('relative w-14 text-right tabular-nums', off ? 'text-amber-500' : 'text-muted-foreground')}
                      title={off ? t('fishDataset.level_differs') : t('fishDataset.level_title')}
                    >
                      {chapter.rms_db} dB
                    </span>
                  )}
                </button>
              </li>
            );
          })}
        </ul>
        {middle !== null &&
          facets.chapters.some((c) => c.rms_db !== null && Math.abs(c.rms_db - middle) >= 6) && (
            <p role="status" className="text-[11px] text-amber-500">
              {t('fishDataset.level_warning')}
            </p>
          )}
      </section>
    </div>
  );
}
