import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { CheckCircle2Icon, AlertTriangleIcon, PlayIcon, SquareIcon, ZapOffIcon, LoaderCircleIcon } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { loadsFor } from './qwen-model';
import type { useQwenPlayground } from './use-qwen-playground';

type Playground = ReturnType<typeof useQwenPlayground>;

function Labeled({ label, hint, children, className }: { label: string; hint?: string; children: React.ReactNode; className?: string }) {
  return (
    <label className={`block space-y-1 ${className ?? ''}`} title={hint}>
      <span className="block text-xs text-muted-foreground">{label}</span>
      {children}
    </label>
  );
}

export function SetupBar({
  pg,
  selectedCount,
  onRunSelected,
}: {
  pg: Playground;
  selectedCount: number;
  onRunSelected: () => void;
}) {
  const { t } = useTranslation();
  const { state } = pg;
  const [python, setPython] = useState('');
  const [root, setRoot] = useState('');
  const [saved, setSaved] = useState(false);
  const settings = state?.settings;
  useEffect(() => {
    if (!settings) return;
    setPython(settings.python ?? '');
    setRoot(settings.checkpoint_root ?? '');
  }, [settings]);

  const dirty = !!settings && (python !== (settings.python ?? '') || root !== (settings.checkpoint_root ?? ''));
  const loadsOn = loadsFor(pg.takes, true);
  const loadsOff = loadsFor(pg.takes, false);
  const resolvedDir = pg.outputDir.trim() || state?.output_dir_default || '';
  const running = pg.jobRunning;

  return (
    <div className="space-y-3 border-b border-border/50 bg-background/40 p-3">
      <p role="status" className={`flex items-start gap-1.5 text-xs ${state?.engine.available ? 'text-emerald-500' : 'text-amber-500'}`}>
        {state?.engine.available ? <CheckCircle2Icon className="mt-0.5 size-3.5 shrink-0" /> : <AlertTriangleIcon className="mt-0.5 size-3.5 shrink-0" />}
        {state
          ? state.engine.available
            ? t('qwenPlayground.engine_ready')
            : t('qwenPlayground.engine_unavailable', { reason: state.engine.reason })
          : t('qwenPlayground.engine_loading')}
      </p>

      <div className="grid gap-3 md:grid-cols-2">
        <Labeled label={t('qwenPlayground.output_folder')} hint={t('qwenPlayground.output_folder_hint')}>
          <Input
            className="h-8 font-mono text-xs"
            value={pg.outputDir}
            placeholder={state?.output_dir_default ?? ''}
            onChange={(e) => pg.setOutputDir(e.target.value)}
          />
          <span className="block break-all text-[11px] text-muted-foreground">
            {t('qwenPlayground.output_folder_resolved', { path: resolvedDir })}
          </span>
        </Labeled>
        <Labeled label={t('qwenPlayground.session_name')}>
          <Input className="h-8" value={pg.sessionName} maxLength={80} onChange={(e) => pg.setSessionName(e.target.value)} />
        </Labeled>
        <Labeled label={t('qwenPlayground.python')} hint={t('qwenPlayground.python_hint')}>
          <Input className="h-8 font-mono text-xs" placeholder={state?.python_default ?? ''} value={python} onChange={(e) => { setPython(e.target.value); setSaved(false); }} />
        </Labeled>
        <Labeled label={t('qwenPlayground.checkpoint_root')} hint={t('qwenPlayground.checkpoint_root_hint')}>
          <Input className="h-8 font-mono text-xs" value={root} onChange={(e) => { setRoot(e.target.value); setSaved(false); }} />
        </Labeled>
      </div>

      <div className="flex flex-wrap items-end gap-3">
        <Button
          size="sm"
          variant="outline"
          disabled={!dirty}
          onClick={() => void pg.saveSettings({ python, checkpoint_root: root }).then((ok) => setSaved(ok))}
        >
          {t('qwenPlayground.save_settings')}
        </Button>
        {saved && !dirty && <span role="status" className="text-xs text-emerald-500">{t('qwenPlayground.settings_saved')}</span>}
        <Labeled label={t('qwenPlayground.warmup_runs')} hint={t('qwenPlayground.warmup_hint')} className="w-28">
          <Input
            type="number"
            className="h-8"
            min={0}
            max={10}
            value={pg.warmupRuns}
            onChange={(e) => pg.setWarmupRuns(Math.min(10, Math.max(0, Math.round(Number(e.target.value)) || 0)))}
          />
        </Labeled>
        <Labeled label={t('qwenPlayground.warmup_text')} className="min-w-48 flex-1">
          <Input className="h-8" value={pg.warmupText} maxLength={500} onChange={(e) => pg.setWarmupText(e.target.value)} />
        </Labeled>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <label className="flex items-center gap-2 text-xs">
          <input type="checkbox" checked={pg.groupByModel} onChange={(e) => pg.setGroupByModel(e.target.checked)} />
          {t('qwenPlayground.group_by_model')}
        </label>
        <span className="text-xs tabular-nums text-muted-foreground" data-testid="loads">
          {pg.groupByModel
            ? t('qwenPlayground.loads_grouped', { count: loadsOn, ungrouped: loadsOff })
            : t('qwenPlayground.loads_ungrouped', { count: loadsOff, grouped: loadsOn })}
        </span>
        <div className="ml-auto flex flex-wrap items-center gap-2">
          <Button size="sm" disabled={running || pg.busy || pg.takes.length === 0} onClick={() => void pg.run()}>
            {pg.busy ? <LoaderCircleIcon className="animate-spin" /> : <PlayIcon />}
            {t('qwenPlayground.run_all')}
          </Button>
          <Button size="sm" variant="outline" disabled={running || pg.busy || selectedCount === 0} onClick={onRunSelected}>
            {t('qwenPlayground.run_selected', { count: selectedCount })}
          </Button>
          <Button size="sm" variant="outline" disabled={!running} onClick={() => void pg.cancel()}>
            <SquareIcon /> {t('qwenPlayground.cancel')}
          </Button>
          <Button size="sm" variant="ghost" disabled={running} title={t('qwenPlayground.unload_hint')} onClick={() => void pg.unload()}>
            <ZapOffIcon /> {t('qwenPlayground.unload')}
          </Button>
        </div>
      </div>
    </div>
  );
}
