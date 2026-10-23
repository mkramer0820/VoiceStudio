import { useCallback, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  CheckCheckIcon,
  CloudDownloadIcon,
  CloudIcon,
  FilePlusIcon,
  ListChecksIcon,
  LoaderCircleIcon,
  MousePointerClickIcon,
  TagsIcon,
  UsersRoundIcon,
  Trash2Icon,
  Undo2Icon,
  WalletIcon,
  XIcon,
} from 'lucide-react';
import { WorkspaceHeader } from '@/components/app-shell/workspace-header';
import { SecondarySidebar } from '@/components/workspace-sidebar';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { SettingsActionError } from '@/features/settings/settings-action-error';
import { ChunkList } from './chunk-list';
import { JobsDialog, useRunningJobs } from './jobs-dialog';
import { RenderPanel } from './render-panel';
import { SettingsPanel } from './settings-panel';
import { TagsDialog } from './tags-dialog';
import { VoicesDialog } from './voices-dialog';
import {
  applyVoice,
  idsBetween,
  idsInRange,
  insertAfter,
  makeChunk,
  mergeWithNext,
  nextBlock,
  removeChunks,
  splitText,
  summarize,
  voiceHue,
  voiceOrder,
  type SplitMode,
} from './story-model';
import { detectSpeakers } from './tag-tools';
import { useFishStory } from './use-fish-story';

const selectClass =
  'h-8 min-w-0 rounded-md border border-input bg-input/20 px-2 text-sm outline-none focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/30 dark:bg-input/30';

export function FishStoryPage() {
  const { t } = useTranslation();
  const story = useFishStory();
  const { draft, setChunks } = story;
  const [selected, setSelected] = useState<ReadonlySet<string>>(new Set());
  const anchor = useRef<string | null>(null);
  const chunksRef = useRef(draft.chunks);
  chunksRef.current = draft.chunks;
  const [from, setFrom] = useState('1');
  const [to, setTo] = useState('15');
  const [pick, setPick] = useState('');
  const [paste, setPaste] = useState('');
  const [mode, setMode] = useState<SplitMode>('lines');
  const [maxChars, setMaxChars] = useState('');
  const [showImport, setShowImport] = useState(false);
  const [tagsOpen, setTagsOpen] = useState(false);
  const [jobsOpen, setJobsOpen] = useState(false);
  const [voicesOpen, setVoicesOpen] = useState(false);
  const [notice, setNotice] = useState('');
  const runningJobs = useRunningJobs(story.ready);

  const order = useMemo(() => voiceOrder(draft.chunks), [draft.chunks]);
  const hueFor = useCallback((voice: string | null | undefined) => voiceHue(voice, order), [order]);
  const summary = useMemo(() => summarize(draft.chunks), [draft.chunks]);
  const selectedCount = selected.size;

  const onSelect = useCallback((id: string, how: 'single' | 'range' | 'toggle') => {
    setSelected((prev) => {
      if (how === 'range' && anchor.current) {
        return new Set(idsBetween(chunksRef.current, anchor.current, id));
      }
      anchor.current = id;
      if (how === 'toggle') {
        const next = new Set(prev);
        if (next.has(id)) next.delete(id);
        else next.add(id);
        return next;
      }
      return new Set([id]);
    });
  }, []);

  const selectRange = () => {
    const ids = idsInRange(chunksRef.current, Number(from), Number(to));
    setSelected(new Set(ids));
    anchor.current = ids[0] ?? null;
  };

  const apply = (voice: string | null, advance = false) => {
    if (selected.size === 0) return;
    const advanceTo = advance ? nextBlock(draft.chunks, selected) : null;
    setChunks((chunks) => applyVoice(chunks, selected, voice));
    if (advanceTo) {
      setSelected(new Set(advanceTo));
      anchor.current = advanceTo[0] ?? null;
    }
  };

  const addText = (replace: boolean) => {
    const cap = Number(maxChars);
    const added = splitText(paste, mode, Number.isFinite(cap) && cap > 0 ? cap : 0);
    if (added.length === 0) return;
    setChunks((chunks) => (replace ? added : [...chunks, ...added]));
    setPaste('');
    setShowImport(false);
    setSelected(new Set());
    // AI-directed scripts name their speakers in [brackets]: offer to map them right away.
    if (detectSpeakers(added).size > 0) setTagsOpen(true);
  };

  const importOpen = showImport || draft.chunks.length === 0;
  const narratorId = draft.settings.default_voice ?? story.defaults?.reference_id ?? null;
  const narratorName = narratorId ? (draft.voices[narratorId] ?? narratorId) : '';

  return (
    <div
      className="flex h-full min-h-0 flex-col"
      onKeyDown={(event) => {
        const target = event.target as HTMLElement;
        if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z' && !target.closest('input,textarea,select')) {
          event.preventDefault();
          story.undo();
        }
      }}
    >
      <WorkspaceHeader>
        <h1 className="text-sm font-medium">{t('fishStory.title')}</h1>
        <Input
          aria-label={t('fishStory.story_title')}
          value={draft.title}
          placeholder={t('fishStory.untitled')}
          className="h-7 w-56 max-w-[40vw]"
          onChange={(e) => story.setTitle(e.target.value)}
        />
        <span role="status" className="text-xs text-muted-foreground">
          {story.saveState === 'saving'
            ? t('fishStory.saving')
            : story.saveState === 'saved'
              ? t('fishStory.saved_local')
              : ''}
        </span>
      </WorkspaceHeader>

      <JobsDialog
        open={jobsOpen}
        onOpenChange={setJobsOpen}
        onError={story.setError}
        hasBoard={draft.chunks.length > 0}
        onLoad={async (jobId) => {
          const loaded = await story.loadRun(jobId);
          if (!loaded) return;
          setSelected(new Set());
          setJobsOpen(false);
          setNotice(
            t('fishStory.jobs_loaded', {
              title: loaded.title || t('fishStory.untitled'),
              count: loaded.lines,
            }),
          );
        }}
      />

      <VoicesDialog
        open={voicesOpen}
        onOpenChange={setVoicesOpen}
        voiceNames={draft.voices}
        library={story.library}
        onOpen={story.refreshLibrary}
        onAdd={story.addVoice}
        onRename={story.renameVoice}
        onNotes={(id, notes) => void story.saveVoice(id, { notes })}
        onForget={story.forgetVoice}
        onSearch={story.searchLibrary}
        onImportCollection={story.importCollection}
      />

      <TagsDialog
        open={tagsOpen}
        onOpenChange={setTagsOpen}
        chunks={draft.chunks}
        voiceNames={draft.voices}
        cast={draft.cast}
        narrator={draft.settings.default_voice ?? null}
        inheritedNarrator={story.defaults?.reference_id ?? null}
        onChunks={(next) => setChunks(next)}
        onCast={story.setCast}
        onNarrator={(voice) => story.setSettings({ default_voice: voice })}
      />

      <div className="flex min-h-0 flex-1 @max-[40rem]:flex-col">
        <SecondarySidebar title={t('fishStory.settings')} icon={CloudIcon} size="wide" variant="controls">
          <SettingsPanel
            settings={draft.settings}
            defaults={story.defaults}
            voiceNames={draft.voices}
            onChange={story.setSettings}
            onSaveDefaults={story.saveDefaults}
            onError={story.setError}
          />
        </SecondarySidebar>

        <section className="flex min-w-0 flex-1 flex-col">
          {story.error && (
            <div className="p-3">
              <SettingsActionError
                title={t('fishStory.error')}
                detail={story.error}
                onDismiss={story.clearError}
              />
            </div>
          )}

          <div className="space-y-2 border-b border-border/50 bg-background/40 p-3">
            <p className="text-xs text-muted-foreground">{t('fishStory.local_note')}</p>
            {notice && (
              <p role="status" className="text-xs text-emerald-500">
                {notice}
              </p>
            )}
            <div className="flex flex-wrap items-center gap-2">
              <Button variant="outline" size="sm" onClick={() => setShowImport((v) => !v)}>
                <FilePlusIcon /> {t('fishStory.add_text')}
              </Button>
              <Button
                variant="outline"
                size="sm"
                disabled={draft.chunks.length === 0}
                onClick={() => setTagsOpen(true)}
              >
                <TagsIcon /> {t('fishStory.tags_button')}
              </Button>
              <Button variant="outline" size="sm" onClick={() => setVoicesOpen(true)}>
                <UsersRoundIcon /> {t('fishStory.more_voices')}
              </Button>
              <Button variant="outline" size="sm" onClick={() => setJobsOpen(true)}>
                <ListChecksIcon /> {t('fishStory.jobs_button')}
                {runningJobs > 0 && (
                  <span className="rounded-full bg-primary/20 px-1.5 text-[10px]">
                    {t('fishStory.jobs_running_count', { count: runningJobs })}
                  </span>
                )}
              </Button>
              <Button variant="outline" size="sm" disabled={!story.canUndo} onClick={story.undo}>
                <Undo2Icon /> {t('fishStory.undo')}
              </Button>
              <Button
                variant="outline"
                size="sm"
                disabled={story.voicesBusy}
                onClick={() => void story.loadVoices()}
              >
                {story.voicesBusy ? <LoaderCircleIcon className="animate-spin" /> : <CloudDownloadIcon />}
                {t('fishStory.load_voices')}
              </Button>
              <Button
                variant="outline"
                size="sm"
                disabled={story.balanceBusy}
                onClick={() => void story.checkBalance()}
              >
                {story.balanceBusy ? <LoaderCircleIcon className="animate-spin" /> : <WalletIcon />}
                {t('fishStory.check_balance')}
              </Button>
              {story.balance && (
                <span role="status" className="text-xs">
                  {t('fishStory.balance', { credit: story.balance.credit })}
                </span>
              )}
              <Button variant="ghost" size="sm" className="ml-auto" onClick={story.newStory}>
                {t('fishStory.new_story')}
              </Button>
            </div>

            {importOpen && (
              <div className="space-y-2 rounded-xl border border-border/60 bg-card/30 p-3">
                <textarea
                  aria-label={t('fishStory.paste')}
                  value={paste}
                  rows={6}
                  placeholder={t('fishStory.paste_placeholder')}
                  className="w-full resize-y rounded-md border border-input bg-input/20 p-2 text-sm"
                  onChange={(e) => setPaste(e.target.value)}
                />
                <div className="flex flex-wrap items-center gap-2 text-xs">
                  <label className="flex items-center gap-1">
                    {t('fishStory.split_by')}
                    <select
                      className={selectClass}
                      value={mode}
                      onChange={(e) => setMode(e.target.value as SplitMode)}
                    >
                      <option value="lines">{t('fishStory.split_lines')}</option>
                      <option value="paragraphs">{t('fishStory.split_paragraphs')}</option>
                      <option value="sentences">{t('fishStory.split_sentences')}</option>
                    </select>
                  </label>
                  <label className="flex items-center gap-1">
                    {t('fishStory.max_chars')}
                    <Input
                      type="number"
                      min={0}
                      className="h-8 w-24"
                      placeholder="∞"
                      value={maxChars}
                      onChange={(e) => setMaxChars(e.target.value)}
                    />
                  </label>
                  <Button size="sm" disabled={!paste.trim()} onClick={() => addText(false)}>
                    {t('fishStory.append')}
                  </Button>
                  <Button size="sm" variant="outline" disabled={!paste.trim()} onClick={() => addText(true)}>
                    {t('fishStory.replace')}
                  </Button>
                </div>
              </div>
            )}
          </div>

          {draft.chunks.length > 0 && (
            <div
              role="toolbar"
              aria-label={t('fishStory.assign_bar')}
              className="sticky top-0 z-10 flex flex-wrap items-center gap-2 border-b border-border/50 bg-background/90 px-3 py-2 backdrop-blur"
            >
              <MousePointerClickIcon className="size-4 text-muted-foreground" />
              <label className="flex items-center gap-1 text-xs">
                {t('fishStory.lines_from')}
                <Input
                  type="number"
                  min={1}
                  aria-label={t('fishStory.range_from')}
                  className="h-8 w-16"
                  value={from}
                  onChange={(e) => setFrom(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && selectRange()}
                />
                {t('fishStory.lines_to')}
                <Input
                  type="number"
                  min={1}
                  aria-label={t('fishStory.range_to')}
                  className="h-8 w-16"
                  value={to}
                  onChange={(e) => setTo(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && selectRange()}
                />
              </label>
              <Button size="sm" variant="outline" onClick={selectRange}>
                {t('fishStory.select')}
              </Button>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => setSelected(new Set(draft.chunks.map((c) => c.id)))}
              >
                <CheckCheckIcon /> {t('fishStory.select_all')}
              </Button>
              <Button
                size="sm"
                variant="ghost"
                onClick={() =>
                  setSelected(new Set(draft.chunks.filter((c) => !c.voice).map((c) => c.id)))
                }
              >
                {t('fishStory.select_unassigned')}
              </Button>
              <Button size="sm" variant="ghost" disabled={selectedCount === 0} onClick={() => setSelected(new Set())}>
                <XIcon /> {t('fishStory.select_none')}
              </Button>

              <span className="mx-1 h-5 w-px bg-border" aria-hidden />

              <select
                aria-label={t('fishStory.voice')}
                className={`${selectClass} w-44`}
                value={pick}
                onChange={(e) => setPick(e.target.value)}
              >
                <option value="">{t('fishStory.default_voice_name')}</option>
                {Object.entries(draft.voices).map(([id, name]) => (
                  <option key={id} value={id}>
                    {name}
                  </option>
                ))}
              </select>
              <Button size="sm" disabled={selectedCount === 0} onClick={() => apply(pick || null)}>
                {t('fishStory.apply', { count: selectedCount })}
              </Button>
              <Button
                size="sm"
                variant="outline"
                disabled={selectedCount === 0}
                onClick={() => apply(pick || null, true)}
              >
                {t('fishStory.apply_next')}
              </Button>
              <Button
                size="icon-sm"
                variant="ghost"
                disabled={selectedCount === 0}
                aria-label={t('fishStory.delete_selected')}
                onClick={() => {
                  setChunks((chunks) => removeChunks(chunks, selected));
                  setSelected(new Set());
                }}
              >
                <Trash2Icon />
              </Button>

              {order.length > 0 && (
                <div className="flex w-full flex-wrap items-center gap-1.5 text-xs">
                  <span className="text-muted-foreground">{t('fishStory.used_voices')}</span>
                  {order.map((voice) => {
                    const hue = hueFor(voice) ?? 0;
                    return (
                      <button
                        key={voice}
                        type="button"
                        disabled={selectedCount === 0}
                        title={t('fishStory.apply_to_selection')}
                        className="rounded-full px-2.5 py-0.5 disabled:opacity-50"
                        style={{ background: `hsl(${hue} 70% 55% / 0.18)`, color: `hsl(${hue} 80% 72%)` }}
                        onClick={() => {
                          setPick(voice);
                          apply(voice);
                        }}
                      >
                        {draft.voices[voice] ?? voice}
                      </button>
                    );
                  })}
                </div>
              )}
              {Object.keys(draft.voices).length === 0 && (
                <p className="w-full text-xs text-amber-500">{t('fishStory.no_voices_hint')}</p>
              )}
            </div>
          )}

          <div className="min-h-0 flex-1 overflow-y-auto">
            <ChunkList
              chunks={draft.chunks}
              selected={selected}
              voiceNames={draft.voices}
              defaultVoiceName={narratorName}
              hueFor={hueFor}
              onSelect={onSelect}
              onEdit={(id, text) =>
                setChunks((chunks) => chunks.map((c) => (c.id === id ? { ...c, text } : c)))
              }
              onMerge={(id) => setChunks((chunks) => mergeWithNext(chunks, id))}
              onInsert={(id) => setChunks((chunks) => insertAfter(chunks, id, makeChunk('')))}
              onRemove={(id) => setChunks((chunks) => removeChunks(chunks, new Set([id])))}
            />
          </div>

          <footer className="space-y-2 border-t border-border/50 bg-background/80 p-3">
            <p className="text-xs text-muted-foreground tabular-nums">
              {t('fishStory.summary', {
                lines: summary.chunks,
                chars: summary.chars.toLocaleString(),
                voices: summary.perVoice.filter((v) => v.voice).length,
                unassigned: summary.unassigned,
              })}
            </p>
            <RenderPanel
              draft={draft}
              draftId={story.draftId}
              selectedIds={selected}
              disabled={summary.chunks === 0}
              onError={story.setError}
            />
          </footer>
        </section>
      </div>
    </div>
  );
}
