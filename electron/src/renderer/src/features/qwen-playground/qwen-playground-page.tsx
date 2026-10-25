import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { HistoryIcon, PlusIcon, UserPlusIcon } from 'lucide-react';
import { WorkspaceHeader } from '@/components/app-shell/workspace-header';
import { Button } from '@/components/ui/button';
import { SettingsActionError } from '@/features/settings/settings-action-error';
import { AbBar, type AbPick } from './audio-player';
import { CloneVoiceDialog } from './clone-voice-dialog';
import { ResultsView } from './results-view';
import { SessionsDialog } from './sessions-dialog';
import { SetupBar } from './setup-bar';
import { TakeCard } from './take-card';
import { useQwenPlayground } from './use-qwen-playground';

export function QwenPlaygroundPage() {
  const { t } = useTranslation();
  const pg = useQwenPlayground();
  const { job, state } = pg;
  const [selected, setSelected] = useState<ReadonlySet<string>>(new Set());
  const [cloneOpen, setCloneOpen] = useState(false);
  const [sessionsOpen, setSessionsOpen] = useState(false);
  const [ab, setAb] = useState<{ a: AbPick | null; b: AbPick | null }>({ a: null, b: null });
  const pick = (slot: 'a' | 'b', value: AbPick) => setAb((cur) => ({ ...cur, [slot]: value }));
  const toggle = (id: string, on: boolean) =>
    setSelected((cur) => {
      const next = new Set(cur);
      if (on) next.add(id);
      else next.delete(id);
      return next;
    });
  const profiles = state?.profiles ?? [];

  return (
    <div className="flex h-full min-h-0 flex-col">
      <WorkspaceHeader>
        <h1 className="text-sm font-medium">{t('qwenPlayground.title')}</h1>
        <span className="text-xs text-muted-foreground">{t('qwenPlayground.local_note')}</span>
        <div className="ml-auto flex items-center gap-2">
          <Button size="sm" variant="outline" onClick={() => setSessionsOpen(true)}>
            <HistoryIcon /> {t('qwenPlayground.past_sessions')}
          </Button>
          <Button size="sm" variant="outline" onClick={() => setCloneOpen(true)}>
            <UserPlusIcon /> {t('qwenPlayground.add_voice')}
          </Button>
        </div>
      </WorkspaceHeader>

      <CloneVoiceDialog open={cloneOpen} onOpenChange={setCloneOpen} onSave={pg.addProfile} />
      <SessionsDialog open={sessionsOpen} onOpenChange={setSessionsOpen} ab={ab} onPick={pick} />

      <SetupBar pg={pg} selectedCount={selected.size} onRunSelected={() => void pg.run(selected)} />

      {pg.error && (
        <div className="p-3">
          <SettingsActionError title={t('qwenPlayground.error')} detail={pg.error} onDismiss={() => pg.setError('')} />
        </div>
      )}

      {job && (
        <div className="space-y-1 border-b border-border/50 px-3 py-2 text-xs" role="status">
          <div className="flex flex-wrap items-center gap-3">
            <span className="font-medium">
              {job.status === 'running'
                ? t('qwenPlayground.job_running', { done: job.done, total: job.total })
                : job.status === 'done'
                  ? t('qwenPlayground.job_done', { done: job.done, total: job.total })
                  : job.status === 'cancelled'
                    ? t('qwenPlayground.job_cancelled', { done: job.done, total: job.total })
                    : t('qwenPlayground.job_error', { message: job.message ?? '' })}
            </span>
            <span className="text-muted-foreground">{t('qwenPlayground.job_loads', { count: job.model_swaps })}</span>
            <span className="break-all font-mono text-muted-foreground">{job.folder}</span>
          </div>
          <progress className="h-1.5 w-full" value={job.done} max={Math.max(1, job.total)} aria-label={t('qwenPlayground.progress')} />
        </div>
      )}

      <div className="min-h-0 flex-1 space-y-3 overflow-y-auto p-3">
        {pg.takes.length === 0 && <p className="text-sm text-muted-foreground">{t('qwenPlayground.no_takes')}</p>}
        {pg.takes.map((take, index) => {
          const view = job?.takes.find((x) => x.id === take.id);
          return (
            <TakeCard
              key={take.id}
              take={take}
              index={index}
              total={pg.takes.length}
              models={pg.models}
              profiles={profiles}
              selected={selected.has(take.id)}
              onSelect={(on) => toggle(take.id, on)}
              onChange={(change) => pg.updateTake(take.id, change)}
              onDuplicate={() => pg.copyTake(take.id)}
              onRemove={() => pg.removeTake(take.id)}
              onMove={(by) => pg.moveTake(take.id, by)}
              onAddVoice={() => setCloneOpen(true)}
            >
              {job && view && (
                <ResultsView
                  key={`${job.id}-${take.id}`}
                  session={job.session}
                  take={view}
                  ab={ab}
                  onPick={pick}
                  onAnnotate={(takeId, patch) => pg.annotate(job.session, takeId, patch)}
                />
              )}
            </TakeCard>
          );
        })}
        <Button variant="outline" size="sm" onClick={pg.addTake}>
          <PlusIcon /> {t('qwenPlayground.add_take')}
        </Button>
      </div>

      <footer className="border-t border-border/50 bg-background/80 p-2">
        <AbBar a={ab.a} b={ab.b} onClear={() => setAb({ a: null, b: null })} />
      </footer>
    </div>
  );
}
