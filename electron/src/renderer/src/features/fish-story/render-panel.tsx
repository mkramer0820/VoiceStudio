import { useCallback, useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { AlertCircleIcon, CheckCircle2Icon, EyeIcon, FolderOpenIcon, LoaderCircleIcon, SendIcon, SquareIcon } from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { getBridge } from '@/components/bridge';
import { apiJson, describeError } from '@/lib/api/client';
import { PreviewDialog } from './preview-dialog';
import type { StoryDraft } from './story-model';

interface PlanResult {
  chunks: number;
  empty: number;
  chars: number;
  bytes: number;
  unassigned: number;
  voices: Record<string, { chunks: number; chars: number }>;
  warnings: string[];
  model: string;
  format: string;
  sample_rate: number;
  max_concurrency: number;
  request_mode: 'single' | 'multi' | null;
}

interface JobChunk {
  id: string;
  index: number;
  status: 'queued' | 'running' | 'done' | 'failed' | 'skipped';
  file: string | null;
  error: string | null;
}

export interface RenderJob {
  id: string;
  status: 'running' | 'done' | 'partial' | 'stopped' | 'cancelled' | 'error';
  message: string;
  out_dir: string;
  joined: string | null;
  total: number;
  done: number;
  failed: number;
  chunks: JobChunk[];
}

const POLL_MS = 1200;

export function RenderPanel({
  draft,
  draftId,
  selectedIds,
  disabled,
  onError,
}: {
  draft: StoryDraft;
  draftId: string | null;
  selectedIds: ReadonlySet<string>;
  disabled: boolean;
  onError: (message: string) => void;
}) {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  const [previewOpen, setPreviewOpen] = useState(false);
  const [plan, setPlan] = useState<PlanResult | null>(null);
  const [planning, setPlanning] = useState(false);
  const [onlySelected, setOnlySelected] = useState(false);
  const [retryIds, setRetryIds] = useState<string[] | null>(null);
  const [starting, setStarting] = useState(false);
  const [job, setJob] = useState<RenderJob | null>(null);

  const scoped = useCallback(
    (): StoryDraft => {
      const ids = retryIds ? new Set(retryIds) : onlySelected ? selectedIds : null;
      return ids ? { ...draft, chunks: draft.chunks.filter((c) => ids.has(c.id)) } : draft;
    },
    [draft, onlySelected, retryIds, selectedIds],
  );

  // The staging numbers come from the backend plan: local counting only, no Fish call.
  useEffect(() => {
    if (!open) return;
    let alive = true;
    setPlanning(true);
    apiJson<PlanResult>('/api/fish/story/plan', { method: 'POST', body: JSON.stringify(scoped()) })
      .then((result) => alive && setPlan(result))
      .catch((reason) => alive && onError(describeError(reason)))
      .finally(() => alive && setPlanning(false));
    return () => {
      alive = false;
    };
  }, [open, scoped, onError]);

  // What the preview and the render both cover: a retry list, the selection, or everything.
  const previewIds = useMemo(
    () => retryIds ?? (onlySelected && selectedIds.size > 0 ? [...selectedIds] : undefined),
    [retryIds, onlySelected, selectedIds],
  );

  const running = job?.status === 'running';
  useEffect(() => {
    if (!job || job.status !== 'running') return;
    const timer = setInterval(() => {
      apiJson<RenderJob>(`/api/fish/story/render/${job.id}`)
        .then(setJob)
        .catch((reason) => onError(describeError(reason)));
    }, POLL_MS);
    return () => clearInterval(timer);
  }, [job, onError]);

  const submit = async () => {
    setStarting(true);
    try {
      const started = await apiJson<RenderJob>('/api/fish/story/render', {
        method: 'POST',
        body: JSON.stringify({
          confirm: true,
          draft: draft,
          draft_id: draftId,
          chunk_ids: retryIds ?? (onlySelected ? [...selectedIds] : undefined),
        }),
      });
      setJob(started);
      setOpen(false);
      setRetryIds(null);
    } catch (reason) {
      onError(describeError(reason));
    } finally {
      setStarting(false);
    }
  };

  const cancel = () => {
    if (job) void apiJson(`/api/fish/story/render/${job.id}/cancel`, { method: 'POST' }).catch(() => undefined);
  };

  const reveal = () => {
    if (job) void getBridge()?.files.revealPath(job.out_dir).catch((reason) => onError(describeError(reason)));
  };

  const failed = job?.chunks.filter((c) => c.status === 'failed' || c.status === 'skipped') ?? [];
  const pct = job && job.total ? Math.round(((job.done + job.failed) / job.total) * 100) : 0;

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <Button
          disabled={disabled || running || !draft.settings.request_mode}
          onClick={() => {
            setRetryIds(null);
            setOpen(true);
          }}
        >
          <SendIcon />
          {t('fishStory.review_submit')}
        </Button>
        <Button
          variant="outline"
          disabled={disabled || !draft.settings.request_mode}
          onClick={() => setPreviewOpen(true)}
        >
          <EyeIcon />
          {t('fishStory.preview_button')}
        </Button>
        {selectedIds.size > 0 && (
          <span className="text-xs text-muted-foreground">
            {t('fishStory.selected_count', { count: selectedIds.size })}
          </span>
        )}
      </div>

      {!draft.settings.request_mode && (
        <p role="alert" className="text-xs text-amber-500">
          {t('fishStory.request_mode_required')}
        </p>
      )}

      {job && (
        <section
          aria-label={t('fishStory.render_status')}
          className="space-y-2 rounded-xl border border-border/60 bg-card/30 p-3 text-sm"
        >
          <div className="flex items-center gap-2">
            {running ? (
              <LoaderCircleIcon className="size-4 animate-spin" />
            ) : job.failed || job.status !== 'done' ? (
              <AlertCircleIcon className="size-4 text-amber-500" />
            ) : (
              <CheckCircle2Icon className="size-4 text-emerald-500" />
            )}
            <span className="font-medium">{t(`fishStory.job_${job.status}`)}</span>
            <span className="text-muted-foreground tabular-nums">
              {t('fishStory.job_progress', { done: job.done, failed: job.failed, total: job.total })}
            </span>
            <span className="ml-auto flex gap-1">
              {running && (
                <Button size="xs" variant="outline" onClick={cancel}>
                  <SquareIcon /> {t('fishStory.cancel_render')}
                </Button>
              )}
              <Button size="xs" variant="outline" onClick={reveal}>
                <FolderOpenIcon /> {t('fishStory.reveal')}
              </Button>
            </span>
          </div>
          <div
            role="progressbar"
            aria-valuenow={pct}
            aria-valuemin={0}
            aria-valuemax={100}
            className="h-1.5 overflow-hidden rounded-full bg-muted"
          >
            <div className="h-full bg-primary transition-[width]" style={{ width: `${pct}%` }} />
          </div>
          {job.message && <p className="text-xs text-amber-500">{job.message}</p>}
          {job.joined && (
            <p className="text-xs text-muted-foreground">{t('fishStory.joined_file', { file: job.joined })}</p>
          )}
          <p className="break-all font-mono text-[11px] text-muted-foreground">{job.out_dir}</p>
          {!running && failed.length > 0 && (
            <div className="space-y-1">
              <ul className="max-h-32 space-y-0.5 overflow-y-auto text-xs">
                {failed.map((c) => (
                  <li key={c.id} className="text-destructive">
                    #{c.index}: {c.error ?? t(`fishStory.chunk_${c.status}`)}
                  </li>
                ))}
              </ul>
              <Button
                size="xs"
                variant="outline"
                onClick={() => {
                  setRetryIds(failed.map((c) => c.id));
                  setOpen(true);
                }}
              >
                {t('fishStory.retry_failed', { count: failed.length })}
              </Button>
            </div>
          )}
        </section>
      )}

      <PreviewDialog
        open={previewOpen}
        onOpenChange={setPreviewOpen}
        draft={draft}
        draftId={draftId}
        chunkIds={previewIds}
        onError={onError}
      />

      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{t('fishStory.confirm_title')}</DialogTitle>
            <DialogDescription>{t('fishStory.confirm_body')}</DialogDescription>
          </DialogHeader>
          {planning || !plan ? (
            <LoaderCircleIcon className="mx-auto size-5 animate-spin" />
          ) : (
            <div className="space-y-3 text-sm">
              <dl className="grid grid-cols-2 gap-x-4 gap-y-1">
                <dt className="text-muted-foreground">{t('fishStory.plan_lines')}</dt>
                <dd className="tabular-nums">{plan.chunks}</dd>
                <dt className="text-muted-foreground">{t('fishStory.plan_chars')}</dt>
                <dd className="tabular-nums">{plan.chars.toLocaleString()}</dd>
                <dt className="text-muted-foreground">{t('fishStory.plan_model')}</dt>
                <dd>{plan.model}</dd>
                <dt className="text-muted-foreground">{t('fishStory.plan_format')}</dt>
                <dd>
                  {plan.format.toUpperCase()}
                  {plan.format !== 'opus' ? ` · ${plan.sample_rate} Hz` : ''}
                </dd>
                <dt className="text-muted-foreground">{t('fishStory.plan_concurrency')}</dt>
                <dd className="tabular-nums">{plan.max_concurrency}</dd>
                <dt className="text-muted-foreground">{t('fishStory.request_mode')}</dt>
                <dd>
                  {plan.request_mode ? t(`fishStory.request_mode_${plan.request_mode}`) : '—'}
                </dd>
              </dl>
              <ul className="space-y-0.5 text-xs">
                {Object.entries(plan.voices).map(([voice, v]) => (
                  <li key={voice} className="flex justify-between gap-3">
                    <span className="truncate">{draft.voices[voice] ?? voice}</span>
                    <span className="text-muted-foreground tabular-nums">
                      {t('fishStory.plan_voice_row', { chunks: v.chunks, chars: v.chars })}
                    </span>
                  </li>
                ))}
              </ul>
              {plan.warnings.includes('unassigned_no_default') && (
                <p role="alert" className="text-xs text-amber-500">
                  {t('fishStory.warn_unassigned', { count: plan.unassigned })}
                </p>
              )}
              {!retryIds && selectedIds.size > 0 && (
                <label className="flex items-center gap-2 text-xs">
                  <input
                    type="checkbox"
                    checked={onlySelected}
                    onChange={(e) => setOnlySelected(e.target.checked)}
                  />
                  {t('fishStory.only_selected', { count: selectedIds.size })}
                </label>
              )}
            </div>
          )}
          <DialogFooter>
            <Button variant="outline" onClick={() => setPreviewOpen(true)}>
              <EyeIcon /> {t('fishStory.preview_button')}
            </Button>
            <DialogClose render={<Button variant="outline" />}>{t('common.cancel')}</DialogClose>
            <Button disabled={starting || planning || !plan || plan.chunks === 0} onClick={() => void submit()}>
              {starting ? <LoaderCircleIcon className="animate-spin" /> : <SendIcon />}
              {t('fishStory.submit_to_fish')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
