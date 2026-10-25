import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { ArrowLeftIcon, LoaderCircleIcon } from 'lucide-react';
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
import { apiJson, describeError } from '@/lib/api/client';
import type { AbPick } from './audio-player';
import type { QwenModel, SessionRow, TakeView } from './qwen-model';
import { ResultsView } from './results-view';
import { listSessions, readSession } from './use-qwen-playground';

interface Opened {
  session: string;
  name: string;
  folder: string;
  takes: TakeView[];
}

/** Past sessions from the output folder; opening one shows its manifest with the same players and notes. */
export function SessionsDialog({
  open,
  onOpenChange,
  ab,
  onPick,
  models,
}: {
  models?: QwenModel[];
  open: boolean;
  onOpenChange: (open: boolean) => void;
  ab: { a: AbPick | null; b: AbPick | null };
  onPick: (slot: 'a' | 'b', pick: AbPick) => void;
}) {
  const { t } = useTranslation();
  const [rows, setRows] = useState<SessionRow[] | null>(null);
  const [opened, setOpened] = useState<Opened | null>(null);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!open) return;
    let alive = true;
    setOpened(null);
    setError('');
    listSessions()
      .then((next) => alive && setRows(next))
      .catch((reason) => alive && setError(describeError(reason)));
    return () => {
      alive = false;
    };
  }, [open]);

  const openSession = async (row: SessionRow) => {
    setError('');
    try {
      const data = await readSession(row.session);
      setOpened({ session: row.session, name: row.name, folder: row.folder, takes: data.takes ?? [] });
    } catch (reason) {
      setError(describeError(reason));
    }
  };

  // Notes and stars are the one edit a manifest takes.
  const annotate = async (takeId: string, patch: { notes?: string; starred?: boolean }) => {
    if (!opened) return false;
    try {
      await apiJson(
        `/api/qwen/playground/sessions/${encodeURIComponent(opened.session)}/takes/${encodeURIComponent(takeId)}`,
        { method: 'PATCH', body: JSON.stringify(patch) },
      );
      setOpened({ ...opened, takes: opened.takes.map((x) => (x.id === takeId ? { ...x, ...patch } : x)) });
      return true;
    } catch (reason) {
      setError(describeError(reason));
      return false;
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[88vh] overflow-y-auto sm:max-w-3xl">
        <DialogHeader>
          <DialogTitle>{t('qwenPlayground.sessions_title')}</DialogTitle>
          <DialogDescription>{t('qwenPlayground.sessions_desc')}</DialogDescription>
        </DialogHeader>
        {error && (
          <p role="alert" className="text-xs text-destructive">
            {error}
          </p>
        )}
        {opened ? (
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <Button size="sm" variant="ghost" onClick={() => setOpened(null)}>
                <ArrowLeftIcon /> {t('qwenPlayground.sessions_back')}
              </Button>
              <span className="text-sm font-medium">{opened.name}</span>
            </div>
            <p className="break-all font-mono text-[11px] text-muted-foreground">{opened.folder}</p>
            {opened.takes.map((take) => (
              <section key={take.id} className="space-y-1 rounded-lg border border-border/50 p-2">
                <p className="text-xs">
                  <span className="font-medium">{t('qwenPlayground.take_n', { n: take.number })}</span>
                  {take.label ? ` · ${take.label}` : ''}
                  {take.model ? ` · ${take.model.label} (${take.model.type})` : ''}
                </p>
                <p className="line-clamp-2 text-xs text-muted-foreground">{take.text}</p>
                <ResultsView session={opened.session} take={take} models={models} ab={ab} onPick={onPick} onAnnotate={annotate} />
              </section>
            ))}
          </div>
        ) : rows === null && !error ? (
          <LoaderCircleIcon className="mx-auto size-4 animate-spin" />
        ) : rows && rows.length === 0 ? (
          <p className="text-sm text-muted-foreground">{t('qwenPlayground.sessions_empty')}</p>
        ) : (
          <ul className="space-y-1">
            {rows?.map((row) => (
              <li key={row.session}>
                <button
                  type="button"
                  className="flex w-full items-center gap-3 rounded-md px-2 py-1.5 text-left text-sm hover:bg-muted/50"
                  onClick={() => void openSession(row)}
                >
                  <span className="min-w-0 flex-1 truncate">{row.name}</span>
                  <span className="text-xs tabular-nums text-muted-foreground">
                    {t('qwenPlayground.sessions_counts', { takes: row.takes, renders: row.renders })}
                  </span>
                  <span className="text-xs text-muted-foreground">
                    {row.created ? new Date(row.created * 1000).toLocaleString() : ''}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}
        <DialogFooter>
          <DialogClose render={<Button variant="outline" />}>{t('qwenPlayground.close')}</DialogClose>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
