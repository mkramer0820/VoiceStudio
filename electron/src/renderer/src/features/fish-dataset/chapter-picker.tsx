import { useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { cn } from '@/lib/utils';
import { median } from './explorer-staging';
import type { ChapterFacet } from './use-fish-dataset';

const minutes = (seconds: number) => (seconds / 60).toFixed(seconds >= 6000 ? 0 : 1);
const WARN_DB = 6;

/**
 * Checklist of every chapter, with its clips, minutes and level under the current OTHER filters.
 * Ticks are staged: the parent applies them. A chapter you left out stays listed so it can be ticked back.
 */
export function ChapterPicker({
  chapters,
  ticked,
  onChange,
  onMeasure,
  measuring,
}: {
  chapters: ChapterFacet[];
  ticked: ReadonlySet<string>;
  onChange: (next: Set<string>) => void;
  onMeasure?: () => void;
  measuring?: boolean;
}) {
  const { t } = useTranslation();
  const [find, setFind] = useState('');
  const [byLevel, setByLevel] = useState(false);
  const [within, setWithin] = useState('3');

  const middle = useMemo(
    () => median(chapters.flatMap((c) => (c.rms_db === null ? [] : [c.rms_db]))),
    [chapters],
  );
  const rows = useMemo(() => {
    const needle = find.trim().toLowerCase();
    const shown = chapters.filter((c) => !needle || c.chapter.toLowerCase().includes(needle));
    return [...shown].sort((a, b) =>
      byLevel
        ? (a.rms_db ?? Infinity) - (b.rms_db ?? Infinity) || a.chapter.localeCompare(b.chapter)
        : a.chapter.localeCompare(b.chapter),
    );
  }, [chapters, find, byLevel]);

  const far = (c: ChapterFacet) => middle !== null && c.rms_db !== null && Math.abs(c.rms_db - middle) >= WARN_DB;
  const maxSeconds = Math.max(1, ...chapters.map((c) => c.seconds));
  const tickedRows = chapters.filter((c) => ticked.has(c.chapter));
  const unmeasured = chapters.filter((c) => c.clips > 0 && c.measured < c.clips).length;

  // The quick controls act on the rows you can see, so a name filter narrows what they touch.
  const setRows = (pick: (c: ChapterFacet, on: boolean) => boolean) => {
    const next = new Set(ticked);
    for (const c of rows) {
      if (pick(c, ticked.has(c.chapter))) next.add(c.chapter);
      else next.delete(c.chapter);
    }
    onChange(next);
  };
  const limit = Number(within);
  const limitOk = within.trim() !== '' && Number.isFinite(limit) && limit >= 0;
  const toggle = (name: string) => {
    const next = new Set(ticked);
    if (!next.delete(name)) next.add(name);
    onChange(next);
  };

  return (
    <div className="space-y-1.5">
      <div className="flex flex-wrap items-center gap-1.5">
        <Button size="xs" variant="outline" onClick={() => setRows(() => true)}>
          {t('fishDataset.chap_all')}
        </Button>
        <Button size="xs" variant="outline" onClick={() => setRows(() => false)}>
          {t('fishDataset.chap_none')}
        </Button>
        <Button size="xs" variant="outline" onClick={() => setRows((_, on) => !on)}>
          {t('fishDataset.chap_invert')}
        </Button>
        <span className="mx-1 h-4 w-px bg-border" aria-hidden />
        <Button
          size="xs"
          variant="outline"
          disabled={!limitOk || middle === null}
          title={t('fishDataset.chap_within_hint')}
          onClick={() => setRows((c) => c.rms_db !== null && middle !== null && Math.abs(c.rms_db - middle) <= limit)}
        >
          {t('fishDataset.chap_within')}
        </Button>
        <Input
          type="number"
          min={0}
          step={0.5}
          aria-label={t('fishDataset.chap_within_db')}
          className="h-6 w-14 px-1.5 text-xs"
          value={within}
          onChange={(e) => setWithin(e.target.value)}
        />
        <span className="text-muted-foreground">{t('fishDataset.chap_within_unit')}</span>
        <Input
          type="search"
          aria-label={t('fishDataset.chap_find')}
          placeholder={t('fishDataset.chap_find')}
          className="ml-auto h-6 w-36 px-1.5 text-xs"
          value={find}
          onChange={(e) => setFind(e.target.value)}
        />
        <label className="flex items-center gap-1 text-[11px] text-muted-foreground">
          <input type="checkbox" checked={byLevel} onChange={(e) => setByLevel(e.target.checked)} />
          {t('fishDataset.chap_sort_level')}
        </label>
      </div>

      <p role="status" data-testid="chapter-summary" className="text-[11px] tabular-nums text-muted-foreground">
        {t('fishDataset.chap_summary', {
          selected: tickedRows.length,
          total: chapters.length,
          minutes: minutes(tickedRows.reduce((sum, c) => sum + c.seconds, 0)),
          clips: tickedRows.reduce((sum, c) => sum + c.clips, 0),
        })}
      </p>

      {unmeasured > 0 && (
        <p className="flex flex-wrap items-center gap-2 text-[11px] text-muted-foreground">
          {t('fishDataset.chap_unmeasured_note', { count: unmeasured })}
          {onMeasure && (
            <Button size="xs" variant="outline" disabled={measuring} onClick={onMeasure}>
              {t('fishDataset.analyze')}
            </Button>
          )}
        </p>
      )}

      <ul className="grid max-h-56 gap-x-4 gap-y-0.5 overflow-y-auto @3xl:grid-cols-2" aria-label={t('fishDataset.chart_chapters')}>
        {rows.map((c) => {
          const on = ticked.has(c.chapter);
          const dimmed = c.clips === 0;
          return (
            <li key={c.chapter} className={cn(dimmed && 'opacity-50')} title={dimmed ? t('fishDataset.chap_no_match') : undefined}>
              <label
                className={cn(
                  'relative flex cursor-pointer items-center gap-2 overflow-hidden rounded px-1.5 py-0.5',
                  on ? 'bg-primary/15' : 'hover:bg-muted/50',
                )}
              >
                <span
                  aria-hidden
                  className="absolute inset-y-0 left-0 bg-primary/10"
                  style={{ width: `${(c.seconds / maxSeconds) * 100}%` }}
                />
                <input
                  type="checkbox"
                  className="relative"
                  checked={on}
                  aria-label={t('fishDataset.chap_check', { chapter: c.chapter })}
                  onChange={() => toggle(c.chapter)}
                />
                <span className="relative min-w-0 flex-1 truncate">{c.chapter}</span>
                <span className="relative tabular-nums text-muted-foreground">
                  {c.clips} · {minutes(c.seconds)} min
                </span>
                {c.rms_db !== null && c.rms_min !== null && c.rms_max !== null ? (
                  <span
                    className={cn('relative text-right tabular-nums', far(c) ? 'text-amber-500' : 'text-muted-foreground')}
                    title={
                      (far(c) ? `${t('fishDataset.level_differs')}. ` : '') +
                      t('fishDataset.chap_level_title', {
                        median: c.rms_db,
                        p10: c.rms_p10,
                        p90: c.rms_p90,
                        measured: c.measured,
                        clips: c.clips,
                      })
                    }
                  >
                    {far(c) && <span aria-label={t('fishDataset.level_differs')}>⚠ </span>}
                    {t('fishDataset.chap_level', { median: c.rms_db, low: Math.round(c.rms_min), high: Math.round(c.rms_max) })}
                  </span>
                ) : (
                  c.clips > 0 && <span className="relative text-muted-foreground/70">{t('fishDataset.chap_unmeasured')}</span>
                )}
              </label>
            </li>
          );
        })}
        {rows.length === 0 && (
          <li className="text-muted-foreground">{t('fishDataset.chap_nothing', { q: find.trim() })}</li>
        )}
      </ul>

      {middle !== null && chapters.some(far) && (
        <p role="note" className="text-[11px] text-amber-500">
          {t('fishDataset.level_warning')}
        </p>
      )}
    </div>
  );
}
