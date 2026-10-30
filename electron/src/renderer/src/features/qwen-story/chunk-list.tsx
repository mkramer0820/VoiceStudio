import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { AlertTriangleIcon, LoaderCircleIcon, RotateCcwIcon, Trash2Icon } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { apiFetch, describeError } from '@/lib/api/client';
import { EmotionPicker, selectClass } from './emotion-picker';
import { characterRows, type ChunkResult, type EmotionCatalog, type StoryChunk, type StoryJob } from './story-model';

// Audio routes are authenticated, so a plain <audio src> cannot reach them: fetch a blob once per file.
const cache = new Map<string, Promise<string>>();

function loadAudio(job: StoryJob, file: string): Promise<string> {
  const key = `${job.story}/${job.run}/${file}`;
  let hit = cache.get(key);
  if (!hit) {
    hit = (async () => {
      const res = await apiFetch(
        `/api/qwen/story/runs/${encodeURIComponent(job.story)}/${encodeURIComponent(job.run)}/audio/${encodeURIComponent(file)}`,
      );
      return URL.createObjectURL(await res.blob());
    })();
    hit.catch(() => cache.delete(key));
    cache.set(key, hit);
  }
  return hit;
}

export function ChunkAudio({ job, file }: { job: StoryJob; file: string }) {
  const { t } = useTranslation();
  const [url, setUrl] = useState('');
  const [error, setError] = useState('');
  useEffect(() => {
    let alive = true;
    setUrl('');
    setError('');
    loadAudio(job, file)
      .then((next) => alive && setUrl(next))
      .catch((reason) => alive && setError(describeError(reason)));
    return () => {
      alive = false;
    };
  }, [job.story, job.run, file]); // eslint-disable-line react-hooks/exhaustive-deps
  if (error) return <span className="text-xs text-destructive">{error}</span>;
  if (!url) return <LoaderCircleIcon className="size-4 animate-spin text-muted-foreground" aria-label={t('qwenStory.loading_audio')} />;
  return <audio controls preload="metadata" src={url} aria-label={file} className="h-8 w-56 max-w-full" />;
}

/** The chunks of the story: edit text, speaker and emotion; play a render; render this chunk again. */
export function ChunkList({
  chunks,
  catalog,
  job,
  running,
  selected,
  onSelect,
  onChange,
  onRemove,
  onRerender,
}: {
  selected: ReadonlySet<string>;
  /** Shift-click selects the range from the last ticked chunk. */
  onSelect: (id: string, on: boolean, range: boolean) => void;
  chunks: StoryChunk[];
  catalog: EmotionCatalog | null;
  job: StoryJob | null;
  running: boolean;
  onChange: (id: string, patch: Partial<StoryChunk>) => void;
  onRemove: (id: string) => void;
  onRerender: (id: string) => void;
}) {
  const { t } = useTranslation();
  const speakers = characterRows(chunks).flatMap((r) => (r.name ? [r.name] : []));
  const result = (id: string): ChunkResult | undefined => job?.chunks.find((c) => c.id === id);
  return (
    <ol className="space-y-2" aria-label={t('qwenStory.chunks_title')}>
      {chunks.map((chunk, i) => {
        const r = result(chunk.id);
        return (
          <li key={chunk.id} aria-selected={selected.has(chunk.id)} className={`space-y-1.5 rounded-xl border border-border/50 p-2 text-xs ${selected.has(chunk.id) ? 'bg-primary/10' : 'bg-card/20'}`}>
            <div className="flex flex-wrap items-center gap-2">
              <input
                type="checkbox"
                aria-label={t('qwenStory.select_chunk', { n: i + 1 })}
                checked={selected.has(chunk.id)}
                onChange={() => undefined}
                onClick={(e) => onSelect(chunk.id, !selected.has(chunk.id), e.shiftKey)}
              />
              <span className="w-8 tabular-nums text-muted-foreground">{i + 1}</span>
              <select
                aria-label={t('qwenStory.speaker_of', { n: i + 1 })}
                className={`${selectClass} w-36`}
                value={chunk.speaker ?? ''}
                onChange={(e) => onChange(chunk.id, { speaker: e.target.value || null })}
              >
                <option value="">{t('qwenStory.narration')}</option>
                {speakers.map((s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ))}
              </select>
              <EmotionPicker
                catalog={catalog}
                allowInherit
                label={t('qwenStory.emotion_of', { n: i + 1 })}
                value={chunk.emotion}
                onChange={(emotion) => onChange(chunk.id, { emotion })}
              />
              <span className="ml-auto flex items-center gap-1">
                {r?.status === 'running' && <LoaderCircleIcon className="size-4 animate-spin" aria-label={t('qwenStory.rendering')} />}
                {r?.status === 'queued' && running && <span className="text-muted-foreground">{t('qwenStory.queued')}</span>}
                <Button size="icon-xs" variant="ghost" disabled={running} aria-label={t('qwenStory.rerender_chunk', { n: i + 1 })} onClick={() => onRerender(chunk.id)}>
                  <RotateCcwIcon />
                </Button>
                <Button size="icon-xs" variant="ghost" disabled={running} aria-label={t('qwenStory.remove_chunk', { n: i + 1 })} onClick={() => onRemove(chunk.id)}>
                  <Trash2Icon />
                </Button>
              </span>
            </div>
            <textarea
              rows={2}
              aria-label={t('qwenStory.text_of', { n: i + 1 })}
              className="w-full resize-y rounded-md border border-input bg-input/20 p-2 text-sm"
              value={chunk.text}
              onChange={(e) => onChange(chunk.id, { text: e.target.value })}
            />
            {job && r?.status === 'done' && r.file && (
              <div className="flex flex-wrap items-center gap-2">
                <ChunkAudio job={job} file={r.file} />
                <span className="tabular-nums text-muted-foreground">
                  {t('qwenStory.seconds', { s: r.seconds ?? 0 })}
                  {r.profile_name ? ` · ${r.profile_name}` : ''}
                  {r.history.length > 1 ? ` · ${t('qwenStory.take_n', { n: r.history.length })}` : ''}
                </span>
              </div>
            )}
            {r?.status === 'error' && (
              <p role="alert" className="text-destructive">
                {t('qwenStory.chunk_failed', { error: r.error ?? '' })}
              </p>
            )}
            {r?.status === 'done' && r.complete === false && (
              <p role="alert" className="flex items-start gap-1 text-amber-500">
                <AlertTriangleIcon className="mt-0.5 size-3.5 shrink-0" />
                {t('qwenStory.incomplete')}
              </p>
            )}
            {(r?.warnings ?? []).map((w) => (
              <p key={w} role="alert" className="flex items-start gap-1 text-amber-500">
                <AlertTriangleIcon className="mt-0.5 size-3.5 shrink-0" />
                {w}
              </p>
            ))}
          </li>
        );
      })}
    </ol>
  );
}
