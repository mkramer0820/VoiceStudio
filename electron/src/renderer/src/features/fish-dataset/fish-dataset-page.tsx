import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  DatabaseIcon,
  DownloadIcon,
  FolderOpenIcon,
  LoaderCircleIcon,
  PauseIcon,
  PlayIcon,
  RefreshCwIcon,
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
import { ExportDialog } from './export-dialog';
import { EMPTY_FILTERS, PAGE_SIZE, useFishDataset } from './use-fish-dataset';

const selectClass =
  'h-8 w-full rounded-md border border-input bg-input/20 px-2 text-sm outline-none focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/30 dark:bg-input/30';

const minutes = (seconds: number) => (seconds / 60).toFixed(seconds >= 6000 ? 0 : 1);

export function FishDatasetPage() {
  const { t } = useTranslation();
  const ds = useFishDataset();
  const player = useClipPlayer(ds.setError);
  const [folder, setFolder] = useState('');
  const [exportOpen, setExportOpen] = useState(false);
  const { overview, filters } = ds;
  const totals = overview?.totals;
  const maxSeconds = Math.max(1, ...(overview?.speakers.map((s) => s.seconds) ?? [1]));
  const pages = Math.max(1, Math.ceil(ds.total / PAGE_SIZE));
  const set = (patch: Partial<typeof filters>) => ds.setFilters({ ...filters, ...patch });
  const filtered = JSON.stringify(filters) !== JSON.stringify(EMPTY_FILTERS) || ds.voices.size > 0;
  const issues = (overview?.runs ?? []).filter((r) => r.failed > 0 || r.missing > 0);
  const tags = Array.from(
    new Set((overview?.speakers ?? []).flatMap((s) => s.top_tags.map(([tag]) => tag))),
  ).sort();

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
              <select
                aria-label={t('fishDataset.tag')}
                className={selectClass}
                value={filters.tag}
                onChange={(e) => set({ tag: e.target.value })}
              >
                <option value="">{t('fishDataset.any_tag')}</option>
                {tags.map((tag) => (
                  <option key={tag} value={tag}>
                    [{tag}]
                  </option>
                ))}
              </select>
              <select
                aria-label={t('fishDataset.chapter')}
                className={selectClass}
                value={filters.chapter}
                onChange={(e) => set({ chapter: e.target.value })}
              >
                <option value="">{t('fishDataset.any_chapter')}</option>
                {overview?.chapters.map((chapter) => (
                  <option key={chapter} value={chapter}>
                    {chapter}
                  </option>
                ))}
              </select>
              <div className="grid grid-cols-3 gap-2">
                <label className="space-y-1 text-[11px]">
                  {t('fishDataset.min_dur')}
                  <Input
                    type="number"
                    min={0}
                    step={0.5}
                    value={filters.minDur}
                    onChange={(e) => set({ minDur: e.target.value })}
                  />
                </label>
                <label className="space-y-1 text-[11px]">
                  {t('fishDataset.max_dur')}
                  <Input
                    type="number"
                    min={0}
                    step={0.5}
                    value={filters.maxDur}
                    onChange={(e) => set({ maxDur: e.target.value })}
                  />
                </label>
                <label className="space-y-1 text-[11px]">
                  {t('fishDataset.min_words')}
                  <Input
                    type="number"
                    min={0}
                    value={filters.minWords}
                    onChange={(e) => set({ minWords: e.target.value })}
                  />
                </label>
              </div>
              <label className="flex items-center justify-between gap-3 text-xs">
                <span>
                  {t('fishDataset.duplicates')}
                  <span className="block text-[11px] text-muted-foreground">{t('fishDataset.duplicates_hint')}</span>
                </span>
                <Switch checked={filters.duplicates} onCheckedChange={(duplicates) => set({ duplicates })} />
              </label>
              <Button
                size="xs"
                variant="ghost"
                disabled={!filtered && ds.excluded.size === 0}
                onClick={() => {
                  ds.setFilters(EMPTY_FILTERS);
                  ds.clearVoices();
                  ds.clearExcluded();
                }}
              >
                {t('fishDataset.reset')}
              </Button>
            </fieldset>

            <Button
              className="w-full"
              disabled={!overview?.exists || ds.total === 0}
              onClick={() => setExportOpen(true)}
            >
              <DownloadIcon /> {t('fishDataset.export_open', { count: ds.total })}
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
              <div className="space-y-2 border-b border-border/50 p-3">
                <div className="flex items-baseline gap-2">
                  <h2 className="text-sm font-medium">{t('fishDataset.speakers')}</h2>
                  <p className="text-xs text-muted-foreground">{t('fishDataset.speakers_hint')}</p>
                </div>
                <ul className="space-y-1">
                  {overview.speakers.map((s) => {
                    const on = ds.voices.has(s.voice_id);
                    return (
                      <li key={s.voice_id}>
                        <button
                          type="button"
                          aria-pressed={on}
                          data-testid="speaker-row"
                          onClick={() => ds.toggleVoice(s.voice_id)}
                          className={cn(
                            'relative w-full overflow-hidden rounded-lg border px-3 py-2 text-left text-xs outline-none transition-colors focus-visible:ring-2 focus-visible:ring-ring',
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

              <div className="sticky top-0 z-10 flex flex-wrap items-center gap-2 border-b border-border/50 bg-background/90 px-3 py-2 text-xs backdrop-blur">
                <span role="status" className="tabular-nums">
                  {ds.loading ? <LoaderCircleIcon className="inline size-3 animate-spin" /> : null}{' '}
                  {t('fishDataset.matching', { clips: ds.total.toLocaleString(), minutes: minutes(ds.seconds) })}
                  {ds.excluded.size > 0 && ` · ${t('fishDataset.excluded', { count: ds.excluded.size })}`}
                </span>
                <span className="ml-auto flex items-center gap-1">
                  <Button
                    size="xs"
                    variant="outline"
                    disabled={ds.page === 0}
                    onClick={() => ds.setPage(ds.page - 1)}
                  >
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

              <ul>
                {ds.lines.map((line) => {
                  const out = ds.excluded.has(line.key);
                  return (
                    <li
                      key={line.key}
                      data-testid="clip-row"
                      className={cn(
                        'flex items-start gap-2 border-b border-border/30 px-3 py-2 text-xs',
                        out && 'opacity-45',
                      )}
                    >
                      <Button
                        size="icon-sm"
                        variant="ghost"
                        aria-label={t('fishDataset.play', { index: line.index })}
                        onClick={() => void player.toggle(line.key)}
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
                      </div>
                      <span className="pt-1.5 tabular-nums text-muted-foreground">{line.duration_s.toFixed(1)}s</span>
                      <label className="flex items-center gap-1 pt-1.5 text-muted-foreground">
                        <input
                          type="checkbox"
                          checked={!out}
                          aria-label={t('fishDataset.include', { index: line.index })}
                          onChange={() => ds.toggleExcluded(line.key)}
                        />
                      </label>
                    </li>
                  );
                })}
              </ul>
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
        clips={ds.total}
        seconds={ds.seconds}
        excluded={ds.excluded.size}
        knownTags={overview?.tags ?? []}
        onExport={ds.runExport}
        onError={ds.setError}
      />
    </div>
  );
}
