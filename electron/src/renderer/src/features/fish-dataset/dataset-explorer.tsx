import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { ChapterPicker } from './chapter-picker';
import {
  appliedChapters,
  appliedInclude,
  appliedTagMode,
  countChanges,
  rangeKey,
  toPatch,
  type Staged,
  type TagMode,
} from './explorer-staging';
import type { DatasetFacets, DatasetFilters, LengthRange } from './use-fish-dataset';

/**
 * Breakdown of whatever the filters currently match: lengths, tags and chapters. Each list shows
 * every option (it ignores its own filter), clicks only STAGE a choice, and Apply sends all the
 * staged choices as one change. The charts and counts keep describing the applied filters meanwhile.
 */
export function DatasetExplorer({
  facets,
  filters,
  onApply,
  onMeasure,
  measuring,
}: {
  facets: DatasetFacets;
  filters: DatasetFilters;
  onApply: (patch: Partial<DatasetFilters>) => void;
  onMeasure?: () => void;
  measuring?: boolean;
}) {
  const { t } = useTranslation();
  const [staged, setStaged] = useState<Staged>({});
  const edit = (patch: Staged) => setStaged((current) => ({ ...current, ...patch }));
  const changes = countChanges(staged, filters, facets);

  const apply = () => {
    onApply(toPatch(staged, filters, facets));
    setStaged({});
  };

  // ── length ──
  const ranges = staged.ranges ?? filters.durRanges;
  const picked = new Set(ranges.map(rangeKey));
  const maxClips = Math.max(1, ...facets.duration.map((b) => b.clips));
  const toggleBin = (range: LengthRange) =>
    edit({
      ranges: picked.has(rangeKey(range)) ? ranges.filter((r) => rangeKey(r) !== rangeKey(range)) : [...ranges, range],
    });

  // ── tags ──
  const include = staged.include ?? appliedInclude(filters);
  const exclude = staged.exclude ?? filters.tagsNone;
  const mode: TagMode = staged.mode ?? appliedTagMode(filters);
  const tagState = (tag: string) => (include.includes(tag) ? 'include' : exclude.includes(tag) ? 'exclude' : 'off');
  const cycleTag = (tag: string) => {
    const state = tagState(tag);
    edit({
      include: state === 'off' ? [...include, tag] : include.filter((x) => x !== tag),
      exclude: state === 'include' ? [...exclude, tag] : exclude.filter((x) => x !== tag),
    });
  };
  const top = facets.tags.slice(0, 40);
  const counts = new Map(facets.tags);
  const shownTags: Array<[string, number]> = [
    ...top,
    ...[...appliedInclude(filters), ...filters.tagsNone, ...include, ...exclude]
      .filter((tag, i, all) => all.indexOf(tag) === i && !top.some(([name]) => name === tag))
      .map((tag): [string, number] => [tag, counts.get(tag) ?? 0]),
  ];

  // ── chapters ──
  const ticked = staged.chapters ?? appliedChapters(filters, facets);

  return (
    <div className="grid gap-4 border-b border-border/50 p-3 text-xs @3xl:grid-cols-2">
      <div
        data-testid="explorer-apply-bar"
        className="sticky top-0 z-10 -mx-1 flex flex-wrap items-center gap-2 border-b border-border/50 bg-background/90 px-1 pb-2 backdrop-blur @3xl:col-span-2"
      >
        <Button size="xs" disabled={changes === 0} onClick={apply}>
          {t('fishDataset.explorer_apply')}
        </Button>
        <Button size="xs" variant="ghost" disabled={changes === 0} onClick={() => setStaged({})}>
          {t('fishDataset.explorer_discard')}
        </Button>
        <p role="status" className={cn('text-[11px] leading-snug', changes ? 'text-amber-500' : 'text-muted-foreground')}>
          {changes ? t('fishDataset.explorer_pending', { count: changes }) : t('fishDataset.explorer_in_effect')}
        </p>
      </div>

      <section aria-label={t('fishDataset.chart_length')} className="space-y-1.5">
        <h3 className="font-medium">{t('fishDataset.chart_length')}</h3>
        <div className="flex h-20 items-end gap-0.5" role="group">
          {facets.duration.map((bin) => {
            const range: LengthRange = [bin.from, bin.to];
            const on = picked.has(rangeKey(range));
            return (
              <button
                key={bin.from}
                type="button"
                aria-pressed={on}
                title={t('fishDataset.bin_title', {
                  range: bin.to === null ? `${bin.from}+` : `${bin.from}–${bin.to}`,
                  clips: bin.clips,
                })}
                onClick={() => toggleBin(range)}
                className="group flex h-full min-w-0 flex-1 flex-col justify-end outline-none"
              >
                <span
                  className={cn(
                    'w-full rounded-t-sm transition-colors group-focus-visible:ring-2 group-focus-visible:ring-ring',
                    on ? 'bg-primary' : 'bg-primary/35 group-hover:bg-primary/60',
                  )}
                  style={{ height: `${Math.max(bin.clips ? 4 : 1, (bin.clips / maxClips) * 100)}%` }}
                />
              </button>
            );
          })}
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
        <div className="flex flex-wrap items-center gap-2">
          <h3 className="font-medium">{t('fishDataset.chart_tags')}</h3>
          <div role="group" aria-label={t('fishDataset.tag_mode_label')} className="ml-auto flex overflow-hidden rounded border border-border">
            {(['any', 'all'] as const).map((m) => (
              <button
                key={m}
                type="button"
                aria-pressed={mode === m}
                onClick={() => edit({ mode: m })}
                className={cn(
                  'px-2 py-0.5 text-[10px] outline-none focus-visible:ring-2 focus-visible:ring-ring',
                  mode === m ? 'bg-primary text-primary-foreground' : 'text-muted-foreground hover:bg-muted',
                )}
              >
                {t(`fishDataset.tag_mode_${m}`)}
              </button>
            ))}
          </div>
        </div>
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
                  state === 'include' && 'bg-emerald-500/20 text-emerald-500',
                  state === 'exclude' && 'bg-red-500/20 text-red-500 line-through',
                  state === 'off' && 'bg-muted/50 text-muted-foreground hover:bg-muted',
                  count === 0 && 'opacity-50',
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
        <ChapterPicker
          chapters={facets.chapters}
          ticked={ticked}
          onChange={(next) => edit({ chapters: next })}
          onMeasure={onMeasure}
          measuring={measuring}
        />
      </section>
    </div>
  );
}
