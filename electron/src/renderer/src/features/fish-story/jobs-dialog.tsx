import { useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  AlertCircleIcon,
  CheckCircle2Icon,
  FileTextIcon,
  FolderOpenIcon,
  LoaderCircleIcon,
  RotateCcwIcon,
} from 'lucide-react';
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

export interface JobRow {
  id: string;
  title: string;
  status: 'running' | 'done' | 'partial' | 'stopped' | 'cancelled' | 'error' | 'interrupted';
  message: string;
  total: number;
  done: number;
  failed: number;
  started_at: number;
  finished_at: number | null;
  out_dir: string;
  joined: string | null;
  format: string;
  model: string;
  sample_rate: number;
  audio_seconds: number;
  warnings: string[];
}

const POLL_MS = 3000;
const RESUMABLE = new Set(['partial', 'stopped', 'cancelled', 'interrupted', 'error']);

export function formatDuration(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  const m = Math.floor(s / 60);
  return m ? `${m}m ${s % 60}s` : `${s}s`;
}

/** How many jobs are running right now, for the toolbar badge. */
export function useRunningJobs(enabled: boolean): number {
  const [count, setCount] = useState(0);
  useEffect(() => {
    if (!enabled) return;
    let alive = true;
    const tick = () =>
      apiJson<{ jobs: JobRow[] }>('/api/fish/story/jobs?limit=20')
        .then((r) => alive && setCount(r.jobs.filter((j) => j.status === 'running').length))
        .catch(() => undefined);
    void tick();
    const timer = setInterval(tick, POLL_MS * 2);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, [enabled]);
  return count;
}

export function JobsDialog({
  open,
  onOpenChange,
  onError,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onError: (message: string) => void;
}) {
  const { t } = useTranslation();
  const [jobs, setJobs] = useState<JobRow[] | null>(null);
  const [logFor, setLogFor] = useState<string | null>(null);
  const [log, setLog] = useState('');
  const [confirming, setConfirming] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setJobs((await apiJson<{ jobs: JobRow[] }>('/api/fish/story/jobs')).jobs);
    } catch (reason) {
      onError(describeError(reason));
    }
  }, [onError]);

  const anyRunning = jobs?.some((j) => j.status === 'running') ?? false;
  useEffect(() => {
    if (!open) return;
    void refresh();
    const timer = setInterval(() => void refresh(), anyRunning ? 1500 : POLL_MS);
    return () => clearInterval(timer);
  }, [open, refresh, anyRunning]);

  // Keep an open log fresh while its job is still running.
  useEffect(() => {
    if (!open || !logFor) return;
    let alive = true;
    const load = () =>
      apiJson<{ log: string }>(`/api/fish/story/jobs/${logFor}/log`)
        .then((r) => alive && setLog(r.log))
        .catch((reason) => alive && onError(describeError(reason)));
    void load();
    const running = jobs?.find((j) => j.id === logFor)?.status === 'running';
    const timer = running ? setInterval(load, 2000) : undefined;
    return () => {
      alive = false;
      if (timer) clearInterval(timer);
    };
  }, [open, logFor, jobs, onError]);

  const resume = async (id: string) => {
    setBusy(id);
    try {
      await apiJson(`/api/fish/story/jobs/${id}/resume`, {
        method: 'POST',
        body: JSON.stringify({ confirm: true }),
      });
      setConfirming(null);
      await refresh();
    } catch (reason) {
      onError(describeError(reason));
    } finally {
      setBusy(null);
    }
  };

  const reveal = (path: string) =>
    void getBridge()?.files.revealPath(path).catch((reason) => onError(describeError(reason)));

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[88vh] overflow-y-auto sm:max-w-3xl">
        <DialogHeader>
          <DialogTitle>{t('fishStory.jobs_title')}</DialogTitle>
          <DialogDescription>{t('fishStory.jobs_desc')}</DialogDescription>
        </DialogHeader>

        {jobs === null ? (
          <LoaderCircleIcon className="mx-auto size-5 animate-spin" />
        ) : jobs.length === 0 ? (
          <p className="p-4 text-center text-sm text-muted-foreground">{t('fishStory.jobs_empty')}</p>
        ) : (
          <ul className="space-y-2">
            {jobs.map((job) => {
              const left = job.total - job.done;
              const pct = job.total ? Math.round(((job.done + job.failed) / job.total) * 100) : 0;
              return (
                <li
                  key={job.id}
                  data-testid="job-row"
                  className="space-y-2 rounded-xl border border-border/60 bg-card/30 p-3 text-sm"
                >
                  <div className="flex flex-wrap items-center gap-2">
                    {job.status === 'running' ? (
                      <LoaderCircleIcon className="size-4 animate-spin" />
                    ) : job.status === 'done' ? (
                      <CheckCircle2Icon className="size-4 text-emerald-500" />
                    ) : (
                      <AlertCircleIcon className="size-4 text-amber-500" />
                    )}
                    <span className="font-medium">{job.title}</span>
                    <span className="rounded-full bg-muted/50 px-2 py-0.5 text-xs">
                      {t(`fishStory.job_${job.status}`)}
                    </span>
                    <span className="ml-auto text-xs text-muted-foreground">
                      {new Date(job.started_at * 1000).toLocaleString()}
                    </span>
                  </div>
                  <div
                    role="progressbar"
                    aria-valuenow={pct}
                    aria-valuemin={0}
                    aria-valuemax={100}
                    className="h-1 overflow-hidden rounded-full bg-muted"
                  >
                    <div className="h-full bg-primary" style={{ width: `${pct}%` }} />
                  </div>
                  <p className="text-xs text-muted-foreground tabular-nums">
                    {t('fishStory.jobs_progress', { done: job.done, total: job.total, failed: job.failed })}
                    {' · '}
                    {job.model} · {job.format.toUpperCase()}
                    {job.format !== 'opus' ? ` ${job.sample_rate} Hz` : ''}
                    {job.audio_seconds > 0
                      ? ` · ${t('fishStory.jobs_audio', { time: formatDuration(job.audio_seconds) })}`
                      : ''}
                    {job.finished_at
                      ? ` · ${t('fishStory.jobs_took', { time: formatDuration(job.finished_at - job.started_at) })}`
                      : ''}
                  </p>
                  {job.message && <p className="text-xs text-amber-500">{job.message}</p>}
                  {job.warnings?.map((w) => (
                    <p key={w} className="text-xs text-amber-500">
                      {w}
                    </p>
                  ))}
                  <p className="break-all font-mono text-[11px] text-muted-foreground">{job.out_dir}</p>
                  <div className="flex flex-wrap items-center gap-1.5">
                    <Button size="xs" variant="outline" onClick={() => reveal(job.out_dir)}>
                      <FolderOpenIcon /> {t('fishStory.reveal')}
                    </Button>
                    <Button
                      size="xs"
                      variant="outline"
                      onClick={() => setLogFor(logFor === job.id ? null : job.id)}
                    >
                      <FileTextIcon /> {logFor === job.id ? t('fishStory.jobs_hide_log') : t('fishStory.jobs_log')}
                    </Button>
                    {RESUMABLE.has(job.status) && left > 0 && confirming !== job.id && (
                      <Button size="xs" variant="outline" onClick={() => setConfirming(job.id)}>
                        <RotateCcwIcon /> {t('fishStory.jobs_resume', { count: left })}
                      </Button>
                    )}
                    {confirming === job.id && (
                      <span className="flex flex-wrap items-center gap-1.5 text-xs">
                        <span className="text-amber-500">{t('fishStory.jobs_resume_confirm', { count: left })}</span>
                        <Button size="xs" disabled={busy === job.id} onClick={() => void resume(job.id)}>
                          {t('fishStory.jobs_resume_yes')}
                        </Button>
                        <Button size="xs" variant="ghost" onClick={() => setConfirming(null)}>
                          {t('common.cancel')}
                        </Button>
                      </span>
                    )}
                  </div>
                  {logFor === job.id && (
                    <pre
                      aria-label={t('fishStory.jobs_log')}
                      className="max-h-56 overflow-auto rounded-md bg-background/70 p-2 text-[11px] leading-snug whitespace-pre-wrap"
                    >
                      {log || t('fishStory.jobs_log_empty')}
                    </pre>
                  )}
                </li>
              );
            })}
          </ul>
        )}

        <DialogFooter>
          <DialogClose render={<Button variant="outline" />}>{t('fishStory.done')}</DialogClose>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
