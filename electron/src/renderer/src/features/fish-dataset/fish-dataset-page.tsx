import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from 'react';
import { useTranslation } from 'react-i18next';
import {
  ArrowLeftIcon,
  CheckCheckIcon,
  CpuIcon,
  DatabaseIcon,
  DownloadIcon,
  FolderOpenIcon,
  LoaderCircleIcon,
  PauseIcon,
  PlayIcon,
  RefreshCwIcon,
  ShuffleIcon,
  SquareIcon,
  StarIcon,
  XIcon,
  ZoomInIcon,
} from 'lucide-react';
import { WorkspaceHeader } from '@/components/app-shell/workspace-header';
import { SecondarySidebar } from '@/components/workspace-sidebar';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Switch } from '@/components/ui/switch';
import { getBridge } from '@/components/bridge';
import { SettingsActionError } from '@/features/settings/settings-action-error';
import { cn } from '@/lib/utils';
import { useClipPlayer } from './clip-player';
import { DatasetExplorer } from './dataset-explorer';
import { ExportDialog } from './export-dialog';
import { TrainDialog } from './train-dialog';
import {
  EMPTY_FILTERS,
  PAGE_SIZE,
  activeFilterCount,
  useFishDataset,
  type DatasetFilters,
  type DatasetLine,
  type SortKey,
} from './use-fish-dataset';

const selectClass =
  'h-8 w-full rounded-md border border-input bg-input/20 px-2 text-sm outline-none focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/30 dark:bg-input/30';

const minutes = (seconds: number) => (seconds / 60).toFixed(seconds >= 6000 ? 0 : 1);
const SORTS: SortKey[] = ['order', 'duration_desc', 'duration_asc', 'words_desc', 'words_asc', 'random'];

function Num({
  label,
  value,
  onChange,
  step,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  step?: number;
}) {
  return (
    <label className="space-y-1 text-[11px]">
      {label}
      <Input type="number" min={0} step={step} value={value} onChange={(e) => onChange(e.target.value)} />
    </label>
  );
}

function StatChips({ line }: { line: DatasetLine }) {
  const { t } = useTranslation();
  const s = line.stats;
  if (!s) return null;
  const hot = s.clip > 0.0005 || s.peak >= 0.999;
  return (
    <span className="flex flex-wrap gap-1 tabular-nums" data-testid="clip-stats">
      <span className={cn('rounded-full px-1.5', hot ? 'bg-red-500/20 text-red-500' : s.peak > 0.8 ? 'bg-amber-500/15 text-amber-500' : 'bg-muted/50')}>
        {t('fishDataset.stat_peak', { value: s.peak.toFixed(2) })}
      </span>
      <span className="rounded-full bg-muted/50 px-1.5">{s.rms_db} dB</span>
      {s.pauses > 0 && (
        <span className={cn('rounded-full px-1.5', s.pauses >= 3 ? 'bg-amber-500/15 text-amber-500' : 'bg-muted/50')}>
          {t('fishDataset.stat_pauses', { count: s.pauses })}
        </span>
      )}
      {(s.lead_ms > 500 || s.tail_ms > 800) && (
        <span className="rounded-full bg-amber-500/15 px-1.5 text-amber-500">
          {t('fishDataset.stat_edges', { lead: s.lead_ms, tail: s.tail_ms })}
        </span>
      )}
    </span>
  );
}

export function FishDatasetPage() {
  const { t } = useTranslation();
  const ds = useFishDataset();
  const [folder, setFolder] = useState('');
  const [exportOpen, setExportOpen] = useState(false);
  const [reference, setReference] = useState<{ key: string; label: string } | null>(null);
  const [trainOpen, setTrainOpen] = useState(false);
  const [trainDir, setTrainDir] = useState('');
  const [auto, setAuto] = useState(true);
  const [active, setActive] = useState<string | null>(null);
  const { overview, filters, facets, analysis } = ds;
  const pages = Math.max(1, Math.ceil(ds.total / PAGE_SIZE));
  const totals = overview?.totals;
  const maxSeconds = Math.max(1, ...(overview?.speakers.map((s) => s.seconds) ?? [1]));
  const issues = (overview?.runs ?? []).filter((r) => r.failed > 0 || r.missing > 0);
  const focused =
    ds.voices.size === 1 ? overview?.speakers.find((s) => ds.voices.has(s.voice_id)) : undefined;
  const nFilters = activeFilterCount(filters) + (ds.voices.size > 0 ? 1 : 0);

  // Auto-advance reads fresh values through refs: the player's callback outlives a render.
  const live = useRef({ lines: ds.lines, page: ds.page, pages, auto });
  live.current = { lines: ds.lines, page: ds.page, pages, auto };
  const pendingAuto = useRef(false);
  const listRef = useRef<HTMLDivElement | null>(null);
  const player = useClipPlayer(ds.setError, (key) => {
    const { lines, page, pages: count, auto: on } = live.current;
    if (!on) return;
    const next = lines[lines.findIndex((l) => l.key === key) + 1];
    if (next) {
      setActive(next.key);
      void player.play(next.key);
    } else if (page + 1 < count) {
      pendingAuto.current = true;
      ds.setPage(page + 1);
    }
  });

  useEffect(() => {
    if (pendingAuto.current && ds.lines.length > 0 && !ds.loading) {
      pendingAuto.current = false;
      setActive(ds.lines[0].key);
      void player.play(ds.lines[0].key);
    }
  }, [ds.lines, ds.loading, player]);

  useEffect(() => {
    if (!active) return;
    listRef.current?.querySelector(`[data-key="${CSS.escape(active)}"]`)?.scrollIntoView?.({ block: 'nearest' });
  }, [active]);

  const set = (patch: Partial<DatasetFilters>) => ds.patchFilters(patch);
  const tagLabel = (tag: string, how: 'all' | 'none') => t(`fishDataset.chip_tag_${how}`, { tag });
  const chips = useMemo(() => {
    const out: Array<{ id: string; label: string; clear: () => void }> = [];
    const f = filters;
    if (f.q) out.push({ id: 'q', label: t('fishDataset.chip_q', { q: f.q }), clear: () => set({ q: '' }) });
    f.tagsAll.forEach((tag) =>
      out.push({ id: `a${tag}`, label: tagLabel(tag, 'all'), clear: () => set({ tagsAll: f.tagsAll.filter((x) => x !== tag) }) }),
    );
    f.tagsNone.forEach((tag) =>
      out.push({ id: `n${tag}`, label: tagLabel(tag, 'none'), clear: () => set({ tagsNone: f.tagsNone.filter((x) => x !== tag) }) }),
    );
    f.chapters.forEach((c) =>
      out.push({ id: `c${c}`, label: c, clear: () => set({ chapters: f.chapters.filter((x) => x !== c) }) }),
    );
    if (f.minDur || f.maxDur)
      out.push({
        id: 'dur',
        label: t('fishDataset.chip_dur', { min: f.minDur || '0', max: f.maxDur || '∞' }),
        clear: () => set({ minDur: '', maxDur: '' }),
      });
    if (f.minWords || f.maxWords)
      out.push({
        id: 'words',
        label: t('fishDataset.chip_words', { min: f.minWords || '0', max: f.maxWords || '∞' }),
        clear: () => set({ minWords: '', maxWords: '' }),
      });
    if (f.hideClipped) out.push({ id: 'clip', label: t('fishDataset.chip_noclip'), clear: () => set({ hideClipped: false }) });
    if (f.maxPeak) out.push({ id: 'peak', label: t('fishDataset.chip_peak', { value: f.maxPeak }), clear: () => set({ maxPeak: '' }) });
    if (f.minRms || f.maxRms)
      out.push({
        id: 'rms',
        label: t('fishDataset.chip_rms', { min: f.minRms || '−∞', max: f.maxRms || '0' }),
        clear: () => set({ minRms: '', maxRms: '' }),
      });
    if (f.maxLead) out.push({ id: 'lead', label: t('fishDataset.chip_lead', { value: f.maxLead }), clear: () => set({ maxLead: '' }) });
    if (f.maxTail) out.push({ id: 'tail', label: t('fishDataset.chip_tail', { value: f.maxTail }), clear: () => set({ maxTail: '' }) });
    if (f.maxPauses) out.push({ id: 'pauses', label: t('fishDataset.chip_pauses', { value: f.maxPauses }), clear: () => set({ maxPauses: '' }) });
    if (f.maxGap) out.push({ id: 'gap', label: t('fishDataset.chip_gap', { value: f.maxGap }), clear: () => set({ maxGap: '' }) });
    if (f.excludedOnly) out.push({ id: 'ex', label: t('fishDataset.chip_excluded'), clear: () => set({ excludedOnly: false }) });
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filters, t]);

  const onListKey = (event: KeyboardEvent<HTMLDivElement>) => {
    if ((event.target as HTMLElement).closest('input,select,textarea')) return;
    const index = ds.lines.findIndex((l) => l.key === active);
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault();
      const next = ds.lines[Math.max(0, Math.min(ds.lines.length - 1, index + (event.key === 'ArrowDown' ? 1 : -1)))];
      if (next) setActive(next.key);
    } else if ((event.key === ' ' || event.key === 'Enter') && active) {
      event.preventDefault();
      void player.toggle(active);
    } else if ((event.key === 'x' || event.key === 'X') && active) {
      event.preventDefault();
      ds.toggleExcluded(active);
    }
  };

  return (
    <div className="flex h-full min-h-0 flex-col">
      <WorkspaceHeader>
        <h1 className="text-sm font-medium">{t('fishDataset.title')}</h1>
        {totals && (
          <span role="status" className="text-xs text-muted-foreground tabular-nums">
            {t('fishDataset.totals', {
              clips: totals.lines.toLocaleString(),
              hours: (totals.seconds / 3600).toFixed(1),
              runs: totals.runs,
            })}
          </span>
        )}
      </WorkspaceHeader>

      <div className="flex min-h-0 flex-1 @max-[40rem]:flex-col">
        <SecondarySidebar title={t('fishDataset.settings')} icon={DatabaseIcon} size="wide" variant="controls">
          <div className="space-y-4 p-1 text-sm">
            <div className="space-y-1">
              <label htmlFor="ds-root" className="text-xs font-medium">
                {t('fishDataset.folder')}
              </label>
              <Input
                id="ds-root"
                className="font-mono text-xs"
                placeholder="D:\\fish_tts\\input"
                value={folder || overview?.root || ''}
                onChange={(e) => setFolder(e.target.value)}
              />
              <p className="text-[11px] leading-snug text-muted-foreground">{t('fishDataset.folder_hint')}</p>
              <div className="flex flex-wrap gap-2">
                <Button
                  size="xs"
                  variant="outline"
                  disabled={!folder.trim() || folder.trim() === overview?.root}
                  onClick={() => void ds.saveRoot(folder.trim()).then((ok) => ok && setFolder(''))}
                >
                  {t('fishDataset.folder_save')}
                </Button>
                <Button size="xs" variant="ghost" disabled={ds.scanning} onClick={() => void ds.refresh(true)}>
                  {ds.scanning ? <LoaderCircleIcon className="animate-spin" /> : <RefreshCwIcon />}
                  {t('fishDataset.rescan')}
                </Button>
                {overview?.root && (
                  <Button
                    size="xs"
                    variant="ghost"
                    onClick={() =>
                      void getBridge()
                        ?.files.revealPath(overview.root ?? '')
                        .catch((reason) => ds.setError(String(reason?.message ?? reason)))
                    }
                  >
                    <FolderOpenIcon /> {t('fishStory.reveal')}
                  </Button>
                )}
              </div>
            </div>

            <fieldset className="space-y-3 border-t border-border/50 pt-3" disabled={!overview?.exists}>
              <legend className="mb-2 text-xs font-semibold tracking-wide text-muted-foreground uppercase">
                {t('fishDataset.filters')}
              </legend>
              <Input
                aria-label={t('fishDataset.search')}
                placeholder={t('fishDataset.search')}
                value={filters.q}
                onChange={(e) => set({ q: e.target.value })}
              />
              <div className="grid grid-cols-2 gap-2">
                <Num label={t('fishDataset.min_dur')} step={0.5} value={filters.minDur} onChange={(minDur) => set({ minDur })} />
                <Num label={t('fishDataset.max_dur')} step={0.5} value={filters.maxDur} onChange={(maxDur) => set({ maxDur })} />
                <Num label={t('fishDataset.min_words')} value={filters.minWords} onChange={(minWords) => set({ minWords })} />
                <Num label={t('fishDataset.max_words')} value={filters.maxWords} onChange={(maxWords) => set({ maxWords })} />
              </div>
              <label className="flex items-center justify-between gap-3 text-xs">
                <span>
                  {t('fishDataset.duplicates')}
                  <span className="block text-[11px] text-muted-foreground">{t('fishDataset.duplicates_hint')}</span>
                </span>
                <Switch checked={filters.duplicates} onCheckedChange={(duplicates) => set({ duplicates })} />
              </label>
              <label className="space-y-1 text-[11px]">
                {t('fishDataset.sort')}
                <select
                  className={selectClass}
                  value={filters.sort}
                  onChange={(e) => set({ sort: e.target.value as SortKey, seed: filters.seed || 1 })}
                >
                  {SORTS.map((s) => (
                    <option key={s} value={s}>
                      {t(`fishDataset.sort_${s}`)}
                    </option>
                  ))}
                </select>
              </label>
              <label className="space-y-1 text-[11px]">
                {t('fishDataset.view')}
                <select
                  className={selectClass}
                  value={filters.excludedOnly ? 'excluded' : 'all'}
                  onChange={(e) => set({ excludedOnly: e.target.value === 'excluded' })}
                >
                  <option value="all">{t('fishDataset.view_all')}</option>
                  <option value="excluded">{t('fishDataset.view_excluded', { count: ds.excluded.size })}</option>
                </select>
              </label>
            </fieldset>

            <fieldset className="space-y-3 border-t border-border/50 pt-3" disabled={!overview?.exists}>
              <legend className="mb-2 text-xs font-semibold tracking-wide text-muted-foreground uppercase">
                {t('fishDataset.quality')}
              </legend>
              <p className="text-[11px] leading-snug text-muted-foreground">{t('fishDataset.quality_hint')}</p>
              <div className="flex flex-wrap items-center gap-2">
                <Button size="xs" variant="outline" disabled={analysis?.running} onClick={() => void ds.startAnalysis()}>
                  {analysis?.running ? <LoaderCircleIcon className="animate-spin" /> : null}
                  {analysis && analysis.cached > 0 ? t('fishDataset.analyze_again') : t('fishDataset.analyze')}
                </Button>
                {analysis?.running && (
                  <span role="status" className="text-[11px] tabular-nums">
                    {t('fishDataset.analyzing', { done: analysis.done, total: analysis.total })}
                  </span>
                )}
                {!analysis?.running && facets && (
                  <span className="text-[11px] text-muted-foreground tabular-nums">
                    {t('fishDataset.analyzed', { done: facets.quality.analyzed, total: facets.quality.total })}
                  </span>
                )}
              </div>
              <label className="flex items-center justify-between gap-3 text-xs">
                <span>{t('fishDataset.hide_clipped')}</span>
                <Switch checked={filters.hideClipped} onCheckedChange={(hideClipped) => set({ hideClipped })} />
              </label>
              <div className="grid grid-cols-2 gap-2">
                <Num label={t('fishDataset.max_peak')} step={0.05} value={filters.maxPeak} onChange={(maxPeak) => set({ maxPeak })} />
                <Num label={t('fishDataset.max_pauses')} value={filters.maxPauses} onChange={(maxPauses) => set({ maxPauses })} />
                <label className="space-y-1 text-[11px]">
                  {t('fishDataset.min_rms')}
                  <Input type="number" max={0} value={filters.minRms} onChange={(e) => set({ minRms: e.target.value })} />
                </label>
                <label className="space-y-1 text-[11px]">
                  {t('fishDataset.max_rms')}
                  <Input type="number" max={0} value={filters.maxRms} onChange={(e) => set({ maxRms: e.target.value })} />
                </label>
                <Num label={t('fishDataset.max_lead')} step={50} value={filters.maxLead} onChange={(maxLead) => set({ maxLead })} />
                <Num label={t('fishDataset.max_tail')} step={50} value={filters.maxTail} onChange={(maxTail) => set({ maxTail })} />
                <Num label={t('fishDataset.max_gap')} step={50} value={filters.maxGap} onChange={(maxGap) => set({ maxGap })} />
              </div>
            </fieldset>

            <Button
              size="xs"
              variant="ghost"
              disabled={nFilters === 0 && ds.excluded.size === 0}
              onClick={() => {
                ds.setFilters(EMPTY_FILTERS);
                ds.clearVoices();
                ds.clearExcluded();
              }}
            >
              {t('fishDataset.reset')}
            </Button>
            <Button
              className="w-full"
              disabled={!overview?.exists || ds.selected === 0}
              onClick={() => setExportOpen(true)}
            >
              <DownloadIcon /> {t('fishDataset.export_open', { count: ds.selected })}
            </Button>
            <Button className="w-full" variant="outline" onClick={() => setTrainOpen(true)}>
              <CpuIcon /> {t('fishTraining.runs_open')}
            </Button>
          </div>
        </SecondarySidebar>

        <section className="flex min-w-0 flex-1 flex-col">
          {ds.error && (
            <div className="p-3">
              <SettingsActionError title={t('fishDataset.error')} detail={ds.error} onDismiss={ds.clearError} />
            </div>
          )}

          {!overview?.root || !overview.exists ? (
            <p className="p-6 text-center text-sm text-muted-foreground">
              {overview?.root ? t('fishDataset.missing_folder') : t('fishDataset.empty')}
            </p>
          ) : (
            <div className="min-h-0 flex-1 overflow-y-auto">
              {focused ? (
                <div data-testid="character-view" className="space-y-1 border-b border-border/50 p-3">
                  <Button size="xs" variant="ghost" onClick={() => ds.clearVoices()}>
                    <ArrowLeftIcon /> {t('fishDataset.all_speakers')}
                  </Button>
                  <div className="flex flex-wrap items-baseline gap-x-3">
                    <h2 className="text-base font-semibold">{focused.name}</h2>
                    {focused.aliases.length > 0 && (
                      <span className="text-xs text-muted-foreground">
                        {t('fishDataset.also_called', { names: focused.aliases.join(', ') })}
                      </span>
                    )}
                    <span className="text-xs tabular-nums text-muted-foreground">
                      {t('fishDataset.speaker_stats', {
                        clips: focused.lines.toLocaleString(),
                        minutes: minutes(focused.seconds),
                        avg: focused.avg_s.toFixed(1),
                        chapters: focused.chapters,
                      })}
                    </span>
                  </div>
                </div>
              ) : (
                <div className="space-y-2 border-b border-border/50 p-3">
                  <div className="flex items-baseline gap-2">
                    <h2 className="text-sm font-medium">{t('fishDataset.speakers')}</h2>
                    <p className="text-xs text-muted-foreground">{t('fishDataset.speakers_hint')}</p>
                  </div>
                  <ul className="space-y-1">
                    {overview.speakers.map((s) => {
                      const on = ds.voices.has(s.voice_id);
                      return (
                        <li key={s.voice_id} className="flex items-stretch gap-1">
                          <button
                            type="button"
                            aria-pressed={on}
                            data-testid="speaker-row"
                            onClick={() => ds.toggleVoice(s.voice_id)}
                            className={cn(
                              'relative min-w-0 flex-1 overflow-hidden rounded-lg border px-3 py-2 text-left text-xs outline-none transition-colors focus-visible:ring-2 focus-visible:ring-ring',
                              on ? 'border-primary bg-primary/10' : 'border-border/60 bg-card/30 hover:bg-card/60',
                            )}
                          >
                            <span
                              aria-hidden
                              className="absolute inset-y-0 left-0 bg-primary/10"
                              style={{ width: `${(s.seconds / maxSeconds) * 100}%` }}
                            />
                            <span className="relative flex flex-wrap items-center gap-x-3 gap-y-1">
                              <b className="text-sm">{s.name}</b>
                              {s.aliases.length > 0 && (
                                <span className="text-muted-foreground">
                                  {t('fishDataset.also_called', { names: s.aliases.join(', ') })}
                                </span>
                              )}
                              <span className="ml-auto tabular-nums">
                                {t('fishDataset.speaker_stats', {
                                  clips: s.lines.toLocaleString(),
                                  minutes: minutes(s.seconds),
                                  avg: s.avg_s.toFixed(1),
                                  chapters: s.chapters,
                                })}
                              </span>
                            </span>
                            {s.top_tags.length > 0 && (
                              <span className="relative mt-1 flex flex-wrap gap-1 text-[10px] text-muted-foreground">
                                {s.top_tags.map(([tag, count]) => (
                                  <span key={tag} className="rounded-full bg-muted/50 px-1.5">
                                    [{tag}] {count}
                                  </span>
                                ))}
                              </span>
                            )}
                          </button>
                          <Button
                            size="icon-sm"
                            variant="outline"
                            className="h-auto"
                            aria-label={t('fishDataset.zoom', { name: s.name })}
                            title={t('fishDataset.zoom', { name: s.name })}
                            onClick={() => ds.setOnlyVoice(s.voice_id)}
                          >
                            <ZoomInIcon />
                          </Button>
                        </li>
                      );
                    })}
                  </ul>
                  {issues.length > 0 && (
                    <p className="text-[11px] text-amber-500">
                      {t('fishDataset.run_issues', {
                        missing: issues.reduce((n, r) => n + r.missing, 0),
                        failed: issues.reduce((n, r) => n + r.failed, 0),
                        runs: issues.length,
                      })}
                    </p>
                  )}
                  {totals && totals.duplicates > 0 && (
                    <p className="text-[11px] text-muted-foreground">
                      {t('fishDataset.duplicates_note', { count: totals.duplicates })}
                    </p>
                  )}
                </div>
              )}

              {facets && <DatasetExplorer facets={facets} filters={filters} onPatch={set} />}

              <div className="sticky top-0 z-10 space-y-1.5 border-b border-border/50 bg-background/90 px-3 py-2 text-xs backdrop-blur">
                <div className="flex flex-wrap items-center gap-2">
                  <span role="status" className="tabular-nums">
                    {ds.loading ? <LoaderCircleIcon className="inline size-3 animate-spin" /> : null}{' '}
                    {t('fishDataset.selected_of', {
                      selected: ds.selected.toLocaleString(),
                      total: ds.total.toLocaleString(),
                      minutes: minutes(ds.selectedSeconds),
                    })}
                  </span>
                  <Button
                    size="xs"
                    variant="outline"
                    disabled={ds.lines.length === 0}
                    onClick={() => {
                      if (player.playing) return player.stop();
                      setActive(ds.lines[0].key);
                      void player.play(ds.lines[0].key);
                    }}
                  >
                    {player.playing ? <PauseIcon /> : <PlayIcon />}
                    {player.playing ? t('fishDataset.stop') : t('fishDataset.play_all')}
                  </Button>
                  <label className="flex items-center gap-1.5">
                    <Switch checked={auto} onCheckedChange={setAuto} aria-label={t('fishDataset.auto_advance')} />
                    {t('fishDataset.auto_advance')}
                  </label>
                  <Button
                    size="xs"
                    variant="outline"
                    disabled={ds.total === 0}
                    onClick={() => set({ sort: 'random', seed: Math.floor(Math.random() * 1e6) + 1 })}
                  >
                    <ShuffleIcon /> {t('fishDataset.shuffle')}
                  </Button>
                  <Button
                    size="xs"
                    variant="outline"
                    disabled={ds.total === 0 || ds.selected === ds.total}
                    onClick={() => void ds.setMatchedExcluded(false)}
                  >
                    <CheckCheckIcon /> {t('fishDataset.select_all')}
                  </Button>
                  <Button
                    size="xs"
                    variant="outline"
                    disabled={ds.selected === 0}
                    onClick={() => void ds.setMatchedExcluded(true)}
                  >
                    <SquareIcon /> {t('fishDataset.select_none')}
                  </Button>
                  <span className="ml-auto flex items-center gap-1">
                    <Button size="xs" variant="outline" disabled={ds.page === 0} onClick={() => ds.setPage(ds.page - 1)}>
                      {t('fishDataset.prev')}
                    </Button>
                    <span className="tabular-nums">
                      {ds.page + 1} / {pages}
                    </span>
                    <Button
                      size="xs"
                      variant="outline"
                      disabled={ds.page + 1 >= pages}
                      onClick={() => ds.setPage(ds.page + 1)}
                    >
                      {t('fishDataset.next')}
                    </Button>
                  </span>
                </div>
                {(chips.length > 0 || ds.voices.size > 1) && (
                  <ul className="flex flex-wrap gap-1" aria-label={t('fishDataset.active_filters')}>
                    {chips.map((chip) => (
                      <li key={chip.id}>
                        <button
                          type="button"
                          data-testid="filter-chip"
                          onClick={chip.clear}
                          aria-label={t('fishDataset.remove_filter', { filter: chip.label })}
                          className="inline-flex items-center gap-1 rounded-full bg-primary/15 px-2 py-0.5 text-[10px] text-primary hover:bg-primary/25"
                        >
                          {chip.label} <XIcon className="size-3" />
                        </button>
                      </li>
                    ))}
                  </ul>
                )}
                <p className="text-[10px] text-muted-foreground">{t('fishDataset.keys_hint')}</p>
              </div>

              <div
                ref={listRef}
                role="listbox"
                tabIndex={0}
                aria-label={t('fishDataset.clips')}
                onKeyDown={onListKey}
                className="outline-none focus-visible:ring-2 focus-visible:ring-ring/40"
              >
                {ds.lines.map((line) => {
                  const out = ds.excluded.has(line.key);
                  return (
                    <div
                      key={line.key}
                      role="option"
                      aria-selected={active === line.key}
                      data-testid="clip-row"
                      data-key={line.key}
                      onClick={() => setActive(line.key)}
                      className={cn(
                        'flex items-start gap-2 border-b border-border/30 px-3 py-2 text-xs',
                        active === line.key && 'bg-primary/10',
                        player.playing === line.key && 'bg-primary/15',
                        out && 'opacity-45',
                      )}
                    >
                      <Button
                        size="icon-sm"
                        variant="ghost"
                        tabIndex={-1}
                        aria-label={t('fishDataset.play', { index: line.index })}
                        onClick={(e) => {
                          e.stopPropagation();
                          setActive(line.key);
                          void player.toggle(line.key);
                        }}
                      >
                        {player.playing === line.key ? <PauseIcon /> : <PlayIcon />}
                      </Button>
                      <div className="min-w-0 flex-1 space-y-1 pt-1">
                        <p className="leading-snug">{line.plain}</p>
                        <p className="flex flex-wrap items-center gap-1 text-[10px] text-muted-foreground">
                          <b className="text-foreground/80">{line.voice_name}</b>
                          {line.tags.map((tag, i) => (
                            <span key={`${tag}${i}`} className="rounded-full bg-muted/50 px-1.5">
                              [{tag}]
                            </span>
                          ))}
                          <span>
                            {line.chapter} · #{line.index}
                          </span>
                          {line.duplicate && (
                            <span className="rounded-full bg-amber-500/15 px-1.5 text-amber-500">
                              {t('fishDataset.older_copy')}
                            </span>
                          )}
                        </p>
                        <p className="text-[10px] text-muted-foreground">
                          <StatChips line={line} />
                        </p>
                      </div>
                      <span className="pt-1.5 tabular-nums text-muted-foreground">{line.duration_s.toFixed(1)}s</span>
                      <Button
                        size="icon-sm"
                        variant="ghost"
                        tabIndex={-1}
                        aria-pressed={reference?.key === line.key}
                        aria-label={t(reference?.key === line.key ? 'fishDataset.reference_unset' : 'fishDataset.reference_set', { index: line.index })}
                        title={t('fishDataset.reference_title')}
                        onClick={(e) => {
                          e.stopPropagation();
                          setReference(reference?.key === line.key ? null : { key: line.key, label: `${line.chapter} #${line.index}` });
                        }}
                      >
                        <StarIcon className={reference?.key === line.key ? 'fill-amber-500 text-amber-500' : undefined} />
                      </Button>
                      <label className="flex items-center gap-1 pt-1.5 text-muted-foreground">
                        <input
                          type="checkbox"
                          tabIndex={-1}
                          checked={!out}
                          aria-label={t('fishDataset.include', { index: line.index })}
                          onClick={(e) => e.stopPropagation()}
                          onChange={() => ds.toggleExcluded(line.key)}
                        />
                      </label>
                    </div>
                  );
                })}
              </div>
              {!ds.loading && ds.lines.length === 0 && (
                <p className="p-6 text-center text-sm text-muted-foreground">{t('fishDataset.no_match')}</p>
              )}
            </div>
          )}
        </section>
      </div>

      <ExportDialog
        open={exportOpen}
        onOpenChange={setExportOpen}
        clips={ds.selected}
        seconds={ds.selectedSeconds}
        excluded={ds.total - ds.selected}
        knownTags={overview?.tags ?? facets?.tags ?? []}
        omnivoiceTags={overview?.omnivoice_tags}
        referenceKey={reference?.key ?? null}
        referenceLabel={reference?.label ?? null}
        onClearReference={() => setReference(null)}
        speakerCount={ds.voices.size || overview?.speakers.length || 1}
        speakerName={overview?.speakers.find((s) => ds.voices.has(s.voice_id))?.name ?? overview?.speakers[0]?.name ?? ''}
        onExport={ds.runExport}
        onError={ds.setError}
        onTrain={(dir) => {
          setTrainDir(dir);
          setExportOpen(false);
          setTrainOpen(true);
        }}
      />
      <TrainDialog open={trainOpen} onOpenChange={setTrainOpen} initialExportDir={trainDir} />
    </div>
  );
}
