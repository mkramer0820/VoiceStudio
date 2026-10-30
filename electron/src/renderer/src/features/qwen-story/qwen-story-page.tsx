import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Link } from '@tanstack/react-router';
import { AlertTriangleIcon, LoaderCircleIcon, PlayIcon, SaveIcon, SquareIcon, TagsIcon, WandSparklesIcon } from 'lucide-react';
import { WorkspaceHeader } from '@/components/app-shell/workspace-header';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { SettingsActionError } from '@/features/settings/settings-action-error';
import { BulkBar } from './bulk-bar';
import { CharacterTable } from './character-table';
import { ChunkAudio, ChunkList } from './chunk-list';
import { selectClass } from './emotion-picker';
import { ProfileLibrary } from './profile-library';
import { TagsDialog } from './tags-dialog';
import {
  applyEmotion,
  applySpeaker,
  characterRows,
  idsBetween,
  idsOfSpeaker,
  incompleteChunkIds,
  mappingWarnings,
  modelLoads,
  type Emotion,
} from './story-model';
import { useQwenStory } from './use-qwen-story';

export function QwenStoryPage() {
  const { t } = useTranslation();
  const s = useQwenStory();
  const { draft, job, state } = s;
  const models = state?.models ?? [];
  const warnings = mappingWarnings(draft, s.profiles, models);
  const loads = modelLoads(draft, s.profiles);
  const failed = incompleteChunkIds(job);
  const canRender = draft.chunks.some((c) => c.text.trim()) && !s.running;
  const [selected, setSelected] = useState<ReadonlySet<string>>(new Set());
  const [anchor, setAnchor] = useState<string | null>(null);
  const [tagsOpen, setTagsOpen] = useState(false);
  const speakers = characterRows(s.chunks).flatMap((r) => (r.name ? [r.name] : []));
  const select = (id: string, on: boolean, range: boolean) => {
    const ids = range && anchor ? idsBetween(s.chunks, anchor, id) : [id];
    setSelected((cur) => {
      const next = new Set(cur);
      for (const x of ids) {
        if (on) next.add(x);
        else next.delete(x);
      }
      return next;
    });
    setAnchor(id);
  };

  const setCharacterEmotion = (name: string | null, emotion: Emotion | null) => {
    if (name === null) return;
    const next = { ...s.emotions };
    if (emotion) next[name] = emotion;
    else delete next[name];
    s.setEmotions(next);
  };

  return (
    <div className="flex h-full min-h-0 flex-col">
      <WorkspaceHeader>
        <h1 className="text-sm font-medium">{t('qwenStory.title')}</h1>
        <span className="text-xs text-muted-foreground">{t('qwenStory.local_note')}</span>
        <div className="ml-auto flex items-center gap-2">
          <Link to="/qwen-playground" className="text-xs underline">
            {t('qwenStory.to_playground')}
          </Link>
        </div>
      </WorkspaceHeader>

      <div className="min-h-0 flex-1 space-y-3 overflow-y-auto p-3">
        <p role="status" className={`text-xs ${state?.engine.available ? 'text-emerald-500' : 'text-amber-500'}`}>
          {state
            ? state.engine.available
              ? t('qwenStory.engine_ready')
              : t('qwenStory.engine_unavailable', { reason: state.engine.reason })
            : t('qwenStory.engine_loading')}
        </p>
        {s.error && <SettingsActionError title={t('qwenStory.error')} detail={s.error} onDismiss={() => s.setError('')} />}

        <section aria-label={t('qwenStory.script_title')} className="space-y-2 rounded-xl border border-border/60 bg-card/30 p-3">
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="text-sm font-medium">{t('qwenStory.script_title')}</h2>
            <select
              aria-label={t('qwenStory.open_story')}
              className={`${selectClass} ml-auto w-52`}
              value={s.storyId ?? ''}
              onChange={(e) => e.target.value && void s.open(e.target.value)}
            >
              <option value="">{t('qwenStory.open_story')}</option>
              {s.stories.map((row) => (
                <option key={row.id} value={row.id}>
                  {row.title} ({row.chunks})
                </option>
              ))}
            </select>
            <Input aria-label={t('qwenStory.story_title')} className="h-8 w-52" value={s.title} onChange={(e) => s.setTitle(e.target.value)} />
            <Button size="sm" variant="outline" onClick={() => void s.save()}>
              <SaveIcon /> {t('qwenStory.save')}
            </Button>
          </div>
          <textarea
            rows={6}
            aria-label={t('qwenStory.script_title')}
            placeholder={t('qwenStory.script_placeholder')}
            className="w-full resize-y rounded-md border border-input bg-input/20 p-2 font-mono text-xs"
            value={s.script}
            onChange={(e) => s.setScript(e.target.value)}
          />
          <div className="flex items-center gap-2">
            <Button size="sm" disabled={!s.script.trim() || s.busy} onClick={() => void s.parse()}>
              <WandSparklesIcon /> {t('qwenStory.split')}
            </Button>
            <span className="text-xs text-muted-foreground">{t('qwenStory.split_hint')}</span>
          </div>
        </section>

        <ProfileLibrary
          profiles={s.profiles}
          models={models}
          catalog={s.catalog}
          onSave={async (form, clip) => (await s.saveProfile(form, clip)) !== null}
          onTranscribe={s.transcribeClip}
          onPrepare={(id) => void s.prepareProfile(id)}
          onDelete={(id) => void s.deleteProfile(id)}
          onImport={(id, file) => void s.importEmbedding(id, file)}
          onExport={(id) => void s.exportEmbedding(id)}
        />

        <CharacterTable
          draft={draft}
          profiles={s.profiles}
          catalog={s.catalog}
          warnings={warnings}
          onNarrator={s.setNarrator}
          onCast={(name, id) => {
            const next = { ...s.cast };
            if (id) next[name] = id;
            else delete next[name];
            s.setCast(next);
          }}
          onEmotion={setCharacterEmotion}
          onSelectLines={(name) => setSelected(new Set(idsOfSpeaker(s.chunks, name)))}
        />

        <section aria-label={t('qwenStory.render_title')} className="space-y-2 rounded-xl border border-border/60 bg-card/30 p-3">
          <h2 className="text-sm font-medium">{t('qwenStory.render_title')}</h2>
          <div className="flex flex-wrap items-center gap-4 text-xs">
            <label className="flex items-center gap-1.5" title={t('qwenStory.warmup_hint')}>
              <input
                type="checkbox"
                checked={s.settings.warmup}
                onChange={(e) => s.setSettings({ ...s.settings, warmup: e.target.checked })}
              />
              {t('qwenStory.warmup')}
            </label>
            <label className="flex items-center gap-1.5">
              {t('qwenStory.pad_s')}
              <Input
                className="h-8 w-20"
                inputMode="decimal"
                value={String(s.settings.pad_s)}
                onChange={(e) => {
                  const v = Number(e.target.value);
                  if (!Number.isNaN(v) && v >= 0 && v <= 5) s.setSettings({ ...s.settings, pad_s: v });
                }}
              />
            </label>
            <label className="flex items-center gap-1.5">
              {t('qwenStory.speaker_pause_ms')}
              <Input
                className="h-8 w-20"
                inputMode="numeric"
                value={String(s.settings.speaker_change_pause_ms)}
                onChange={(e) => {
                  const v = Math.round(Number(e.target.value));
                  if (!Number.isNaN(v) && v >= 0 && v <= 5000) s.setSettings({ ...s.settings, speaker_change_pause_ms: v });
                }}
              />
            </label>
            <span className="text-muted-foreground">{t('qwenStory.model_loads', { count: loads })}</span>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Button size="sm" disabled={!canRender} onClick={() => void s.render()}>
              <PlayIcon /> {t('qwenStory.render_all')}
            </Button>
            <Button size="sm" variant="outline" disabled={!canRender || failed.length === 0} onClick={() => void s.render(failed)}>
              {t('qwenStory.render_missing', { count: failed.length })}
            </Button>
            {s.running && (
              <Button size="sm" variant="outline" onClick={() => void s.cancel()}>
                <SquareIcon /> {t('qwenStory.cancel_render')}
              </Button>
            )}
          </div>
          <p className="flex items-start gap-1 text-xs text-amber-500">
            <AlertTriangleIcon className="mt-0.5 size-3.5 shrink-0" />
            {s.catalog?.unverified_on_narrator ?? t('qwenStory.unverified_narrator')}
          </p>
          {job && (
            <div className="space-y-1 text-xs" role="status">
              <div className="flex flex-wrap items-center gap-3">
                <span className="font-medium">
                  {job.status === 'running'
                    ? t('qwenStory.job_running', { done: job.done, total: job.total })
                    : job.status === 'done'
                      ? t('qwenStory.job_done', { done: job.done, total: job.total })
                      : job.status === 'cancelled'
                        ? t('qwenStory.job_cancelled', { done: job.done, total: job.total })
                        : t('qwenStory.job_error', { message: job.message })}
                </span>
                <span className="break-all font-mono text-muted-foreground">{job.folder}</span>
              </div>
              <progress className="h-1.5 w-full" value={job.done} max={Math.max(1, job.total)} aria-label={t('qwenStory.progress')} />
              {(job.preparing ?? []).length > 0 && (
                <p role="status" className="flex items-center gap-1 text-muted-foreground">
                  <LoaderCircleIcon className="size-3.5 animate-spin" />
                  {t('qwenStory.job_preparing', { names: (job.preparing ?? []).join(', ') })}
                </p>
              )}
              {job.warnings.map((w) => (
                <p key={w} className="text-amber-500">
                  {w}
                </p>
              ))}
              {job.joined && job.status !== 'running' && (
                <div className="flex items-center gap-2">
                  <span>{t('qwenStory.full_story')}</span>
                  <ChunkAudio job={job} file={job.joined} />
                </div>
              )}
            </div>
          )}
        </section>

        {s.chunks.length > 0 && (
          <div className="flex flex-wrap items-center gap-2">
            <BulkBar
              count={selected.size}
              total={s.chunks.length}
              speakers={speakers}
              catalog={s.catalog}
              running={s.running}
              onSelectAll={() => setSelected(new Set(s.chunks.map((c) => c.id)))}
              onClear={() => setSelected(new Set())}
              onEmotion={(emotion) => s.setChunks(applyEmotion(s.chunks, selected, emotion))}
              onSpeaker={(speaker) => s.setChunks(applySpeaker(s.chunks, selected, speaker))}
              onRender={() => void s.render([...selected])}
              onRemove={() => {
                s.setChunks(s.chunks.filter((c) => !selected.has(c.id)));
                setSelected(new Set());
              }}
            />
            <Button size="sm" variant="outline" onClick={() => setTagsOpen(true)}>
              <TagsIcon /> {t('qwenStory.tags_open')}
            </Button>
          </div>
        )}
        <TagsDialog open={tagsOpen} onOpenChange={setTagsOpen} draft={draft} onDraft={s.applyDraft} />
        <ChunkList
          selected={selected}
          onSelect={select}
          chunks={s.chunks}
          catalog={s.catalog}
          job={job}
          running={s.running}
          onChange={(id, patch) =>
            s.setChunks(
              s.chunks.map((c) => (c.id === id ? { ...c, ...patch } : c)),
            )
          }
          onRemove={(id) => s.setChunks(s.chunks.filter((c) => c.id !== id))}
          onRerender={(id) => void s.render([id])}
        />
        {s.chunks.length === 0 && <p className="text-sm text-muted-foreground">{t('qwenStory.no_chunks')}</p>}
      </div>
    </div>
  );
}
