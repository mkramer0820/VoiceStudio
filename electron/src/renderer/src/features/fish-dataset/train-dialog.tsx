import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  AlertCircleIcon,
  CheckCircle2Icon,
  FileTextIcon,
  FolderOpenIcon,
  LoaderCircleIcon,
  PlayIcon,
  RotateCcwIcon,
  SquareIcon,
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
import { Input } from '@/components/ui/input';
import { getBridge } from '@/components/bridge';
import {
  useFishTraining,
  type DryRunResult,
  type TrainOptions,
  type TrainingRun,
} from './use-fish-training';

const RESUMABLE = new Set(['failed', 'cancelled', 'interrupted']);
const num = (raw: string, fallback: number) => (raw.trim() !== '' && Number.isFinite(Number(raw)) ? Number(raw) : fallback);

function Run({
  run,
  onCancel,
  onResume,
  onLog,
  onReveal,
}: {
  run: TrainingRun;
  onCancel: () => void;
  onResume: () => void;
  onLog: () => void;
  onReveal: () => void;
}) {
  const { t } = useTranslation();
  const pct = run.steps ? Math.min(100, Math.round((run.step / run.steps) * 100)) : 0;
  return (
    <div data-testid="train-run" className="space-y-2 rounded-xl border border-border/60 bg-card/30 p-3 text-sm">
      <div className="flex flex-wrap items-center gap-2">
        {run.status === 'running' ? (
          <LoaderCircleIcon className="size-4 animate-spin" />
        ) : run.status === 'done' || run.status === 'dry_run' ? (
          <CheckCircle2Icon className="size-4 text-emerald-500" />
        ) : (
          <AlertCircleIcon className="size-4 text-amber-500" />
        )}
        <span className="font-medium">{run.name}</span>
        <span className="rounded-full bg-muted/50 px-2 py-0.5 text-xs">{t(`fishTraining.status_${run.status}`)}</span>
        {run.status === 'running' && (
          <span className="rounded-full bg-muted/50 px-2 py-0.5 text-xs">{t(`fishTraining.phase_${run.phase}`)}</span>
        )}
        {run.verdict && (
          <span
            className={`rounded-full px-2 py-0.5 text-xs ${
              run.ship_ready ? 'bg-emerald-500/15 text-emerald-500' : 'bg-amber-500/15 text-amber-500'
            }`}
          >
            {t(`fishTraining.verdict_${run.verdict}`)}
          </span>
        )}
        <span className="ml-auto text-xs text-muted-foreground">{new Date(run.created_at * 1000).toLocaleString()}</span>
      </div>
      {!run.dry_run && (
        <>
          <div
            role="progressbar"
            aria-valuenow={pct}
            aria-valuemin={0}
            aria-valuemax={100}
            className="h-1 overflow-hidden rounded-full bg-muted"
          >
            <div className="h-full bg-primary" style={{ width: `${pct}%` }} />
          </div>
          <p className="text-xs tabular-nums text-muted-foreground">
            {t('fishTraining.progress', { step: run.step, steps: run.steps })}
            {run.loss != null ? ` · ${t('fishTraining.loss', { loss: run.loss.toFixed(3) })}` : ''}
            {run.eval_loss != null ? ` · ${t('fishTraining.eval_loss', { loss: run.eval_loss.toFixed(3) })}` : ''}
            {run.resumes ? ` · ${t('fishTraining.resumed', { count: run.resumes })}` : ''}
          </p>
        </>
      )}
      {run.message && <p className="text-xs text-amber-500">{run.message}</p>}
      {run.status === 'done' && run.adapter && (
        <p className="break-all font-mono text-[11px] text-muted-foreground">{run.adapter}</p>
      )}
      <div className="flex flex-wrap items-center gap-1.5">
        <Button size="xs" variant="outline" onClick={onReveal}>
          <FolderOpenIcon /> {t('fishStory.reveal')}
        </Button>
        <Button size="xs" variant="outline" onClick={onLog}>
          <FileTextIcon /> {t('fishTraining.log')}
        </Button>
        {run.status === 'running' && (
          <Button size="xs" variant="outline" onClick={onCancel}>
            <SquareIcon /> {t('fishTraining.cancel')}
          </Button>
        )}
        {RESUMABLE.has(run.status) && !run.dry_run && (
          <Button size="xs" variant="outline" onClick={onResume}>
            <RotateCcwIcon /> {t('fishTraining.resume')}
          </Button>
        )}
      </div>
    </div>
  );
}

export function TrainDialog({
  open,
  onOpenChange,
  initialExportDir,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Folder of the export that was just written, if the dialog was opened from the export dialog. */
  initialExportDir?: string;
}) {
  const { t } = useTranslation();
  const tr = useFishTraining(open);
  const { overview } = tr;
  const [exportDir, setExportDir] = useState(initialExportDir ?? '');
  const [steps, setSteps] = useState('');
  const [rank, setRank] = useState('');
  const [batch, setBatch] = useState('');
  const [heads, setHeads] = useState(false);
  const [download, setDownload] = useState(false);
  const [python, setPython] = useState('');
  const [agree, setAgree] = useState(false);
  const [checking, setChecking] = useState(false);
  const [report, setReport] = useState<DryRunResult | null>(null);
  const [logFor, setLogFor] = useState<string | null>(null);
  const [log, setLog] = useState('');

  useEffect(() => {
    if (initialExportDir) setExportDir(initialExportDir);
  }, [initialExportDir]);

  // Keep an open log fresh while its run is going.
  useEffect(() => {
    if (!open || !logFor) return;
    let alive = true;
    const load = () => void tr.loadLog(logFor).then((text) => alive && setLog(text));
    load();
    const running = overview?.runs?.find((r) => r.id === logFor)?.status === 'running';
    const timer = running ? setInterval(load, 2000) : undefined;
    return () => {
      alive = false;
      if (timer) clearInterval(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, logFor, overview]);

  const defaults = overview?.defaults;
  const options = (): Partial<TrainOptions> => ({
    steps: num(steps, defaults?.steps ?? 2000),
    lora_r: num(rank, defaults?.lora_r ?? 16),
    batch_tokens: num(batch, defaults?.batch_tokens ?? 4096),
    train_heads: heads,
    allow_download: download,
    python: python.trim() || null,
  });

  const check = async () => {
    setChecking(true);
    setReport(null);
    const result = await tr.dryRun(exportDir.trim(), options());
    setChecking(false);
    if (result) setReport(result);
  };
  const start = async () => {
    const run = await tr.start(exportDir.trim(), options());
    if (run) {
      setReport(null);
      setAgree(false);
    }
  };
  const reveal = (path: string) =>
    void getBridge()?.files.revealPath(path).catch((reason) => tr.setError(String(reason?.message ?? reason)));
  const anyRunning = overview?.runs?.some((r) => r.status === 'running') ?? false;
  const field = 'space-y-1';
  const hint = 'text-[11px] text-muted-foreground';

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[88vh] overflow-y-auto sm:max-w-3xl">
        <DialogHeader>
          <DialogTitle>{t('fishTraining.title')}</DialogTitle>
          <DialogDescription>{t('fishTraining.desc')}</DialogDescription>
        </DialogHeader>

        {tr.error && (
          <p role="alert" className="rounded-md bg-destructive/10 p-2 text-xs text-destructive">
            {tr.error}
          </p>
        )}

        {overview && !overview.enabled && (
          <div data-testid="train-off" className="space-y-2 rounded-xl border border-amber-500/40 bg-amber-500/5 p-3 text-sm">
            <p className="font-medium">{t('fishTraining.off_title')}</p>
            <p className="text-xs text-muted-foreground">{t('fishTraining.off_body')}</p>
            <Button size="sm" onClick={() => void tr.setEnabled(true)}>
              {t('fishTraining.off_enable')}
            </Button>
          </div>
        )}

        {overview?.enabled && (
          <div className="space-y-3 text-sm">
            <div className={field}>
              <label htmlFor="tr-export" className="text-xs font-medium">
                {t('fishTraining.export_dir')}
              </label>
              <Input
                id="tr-export"
                className="font-mono text-xs"
                placeholder="D:\\datasets\\narrator"
                value={exportDir}
                onChange={(e) => setExportDir(e.target.value)}
              />
              <p className={hint}>{t('fishTraining.export_dir_hint')}</p>
            </div>
            <div className="grid grid-cols-3 gap-3">
              <div className={field}>
                <label htmlFor="tr-steps" className="text-xs font-medium">
                  {t('fishTraining.steps')}
                </label>
                <Input id="tr-steps" type="number" min={10} placeholder={String(defaults?.steps ?? 2000)} value={steps} onChange={(e) => setSteps(e.target.value)} />
              </div>
              <div className={field}>
                <label htmlFor="tr-rank" className="text-xs font-medium">
                  {t('fishTraining.rank')}
                </label>
                <Input id="tr-rank" type="number" min={1} placeholder={String(defaults?.lora_r ?? 16)} value={rank} onChange={(e) => setRank(e.target.value)} />
              </div>
              <div className={field}>
                <label htmlFor="tr-batch" className="text-xs font-medium">
                  {t('fishTraining.batch_tokens')}
                </label>
                <Input id="tr-batch" type="number" min={512} placeholder={String(defaults?.batch_tokens ?? 4096)} value={batch} onChange={(e) => setBatch(e.target.value)} />
              </div>
            </div>
            <p className={hint}>{t('fishTraining.numbers_hint')}</p>
            <label className="flex items-start gap-2 text-xs">
              <input type="checkbox" className="mt-0.5" checked={heads} onChange={(e) => setHeads(e.target.checked)} />
              <span>
                <b>{t('fishTraining.train_heads')}</b>
                <span className="block text-muted-foreground">{t('fishTraining.train_heads_hint')}</span>
              </span>
            </label>
            <label className="flex items-start gap-2 text-xs">
              <input type="checkbox" className="mt-0.5" checked={download} onChange={(e) => setDownload(e.target.checked)} />
              <span>
                <b>{t('fishTraining.allow_download')}</b>
                <span className="block text-muted-foreground">{t('fishTraining.allow_download_hint')}</span>
              </span>
            </label>
            <details className="text-xs">
              <summary className="cursor-pointer font-medium">{t('fishTraining.advanced')}</summary>
              <div className={`${field} mt-2`}>
                <label htmlFor="tr-python" className="text-xs font-medium">
                  {t('fishTraining.python')}
                </label>
                <Input
                  id="tr-python"
                  className="font-mono text-xs"
                  placeholder={defaults?.python ?? ''}
                  value={python}
                  onChange={(e) => setPython(e.target.value)}
                />
                <p className={hint}>{t('fishTraining.python_hint')}</p>
              </div>
            </details>

            {report?.report && (
              <div data-testid="train-report" className="space-y-1 rounded-xl border border-border/60 bg-card/30 p-3 text-xs">
                <p className="font-medium">
                  {report.report.ready ? t('fishTraining.report_ready') : t('fishTraining.report_blocked')}
                </p>
                <p className="text-muted-foreground">
                  {t('fishTraining.report_clips', {
                    train: report.report.export.train_clips,
                    val: report.report.export.val_clips,
                    minutes: Math.round(report.report.export.seconds / 60),
                  })}
                  {' · '}
                  {t('fishTraining.report_device', { device: report.report.hardware.device })}
                </p>
                {report.report.hardware.blockers.map((b) => (
                  <p key={b} className="text-destructive">
                    {b}
                  </p>
                ))}
                {[...report.report.export.warnings, ...report.report.hardware.warnings, ...report.warnings].map((w, i) => (
                  <p key={`${i}-${w}`} className="text-amber-500">
                    {w}
                  </p>
                ))}
              </div>
            )}

            <label className="flex items-start gap-2 text-xs">
              <input type="checkbox" className="mt-0.5" checked={agree} onChange={(e) => setAgree(e.target.checked)} />
              <span>{t('fishTraining.start_confirm')}</span>
            </label>
            <div className="flex flex-wrap gap-2">
              <Button variant="outline" disabled={checking || !exportDir.trim()} onClick={() => void check()}>
                {checking && <LoaderCircleIcon className="animate-spin" />} {t('fishTraining.check')}
              </Button>
              <Button disabled={!agree || anyRunning || !exportDir.trim()} onClick={() => void start()}>
                <PlayIcon /> {t('fishTraining.start')}
              </Button>
              <Button className="ml-auto" size="sm" variant="ghost" onClick={() => void tr.setEnabled(false)}>
                {t('fishTraining.on_disable')}
              </Button>
            </div>
            {anyRunning && <p className={hint}>{t('fishTraining.one_at_a_time')}</p>}
          </div>
        )}

        {overview && (
          <div className="space-y-2">
            <h3 className="text-sm font-medium">{t('fishTraining.runs_title')}</h3>
            <p className={hint}>{t('fishTraining.verdict_hint')}</p>
            {(overview.runs?.length ?? 0) === 0 ? (
              <p className="p-3 text-center text-sm text-muted-foreground">{t('fishTraining.runs_empty')}</p>
            ) : (
              <div className="space-y-2">
                {(overview.runs ?? []).map((run) => (
                  <div key={run.id} className="space-y-2">
                    <Run
                      run={run}
                      onCancel={() => void tr.cancel(run.id)}
                      onResume={() => void tr.resume(run.id)}
                      onLog={() => setLogFor(logFor === run.id ? null : run.id)}
                      onReveal={() => reveal(run.run_dir)}
                    />
                    {logFor === run.id && (
                      <pre
                        data-testid="train-log"
                        className="max-h-64 overflow-auto whitespace-pre-wrap rounded-md bg-background/70 p-2 font-mono text-[11px]"
                      >
                        {log || t('fishTraining.log_empty')}
                      </pre>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        <DialogFooter>
          <DialogClose render={<Button variant="outline" />}>{t('fishStory.done')}</DialogClose>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
