import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { AlertCircleIcon, CheckCircle2Icon, LoaderCircleIcon } from 'lucide-react';
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
import type { StoryDraft } from './story-model';

interface PreviewRequest {
  index: number;
  indices: number[];
  id: string;
  ids: string[];
  mode: 'single' | 'multi';
  speaker: string | null;
  speakers: string[];
  voice: string | string[] | null;
  voice_name: string | null;
  file: string;
  chars: number;
  body: Record<string, unknown>;
  issues: string[];
  notes: string[];
}

interface PreviewResult {
  ok: boolean;
  total: number;
  total_requests: number;
  invalid: number;
  shown: number;
  endpoint: string;
  headers: Record<string, string>;
  folder: string | null;
  requests: PreviewRequest[];
  curl: string | null;
  first_invalid: number | null;
  lines: number;
  single_requests: number;
  multi_requests: number;
}

const LIMITS = [3, 10, 50];

/**
 * Dry run: shows the exact requests a render would send, checked against the
 * request schema. It is computed locally; nothing is sent to Fish and no files are made.
 */
export function PreviewDialog({
  open,
  onOpenChange,
  draft,
  draftId,
  chunkIds,
  onError,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  draft: StoryDraft;
  draftId: string | null;
  chunkIds?: string[];
  onError: (message: string) => void;
}) {
  const { t } = useTranslation();
  const [result, setResult] = useState<PreviewResult | null>(null);
  const [limit, setLimit] = useState(LIMITS[0]);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!open) return;
    let alive = true;
    setBusy(true);
    apiJson<PreviewResult>('/api/fish/story/preview', {
      method: 'POST',
      body: JSON.stringify({ draft, draft_id: draftId, chunk_ids: chunkIds, limit }),
    })
      .then((r) => alive && setResult(r))
      .catch((reason) => alive && onError(describeError(reason)))
      .finally(() => alive && setBusy(false));
    return () => {
      alive = false;
    };
  }, [open, draft, draftId, chunkIds, limit, onError]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[90vh] overflow-y-auto sm:max-w-3xl">
        <DialogHeader>
          <DialogTitle>{t('fishStory.preview_title')}</DialogTitle>
          <DialogDescription>{t('fishStory.preview_desc')}</DialogDescription>
        </DialogHeader>

        {busy && !result ? (
          <LoaderCircleIcon className="mx-auto size-5 animate-spin" />
        ) : result ? (
          <div className="space-y-3 text-sm">
            <p
              role="status"
              className={`flex items-center gap-2 font-medium ${result.ok ? 'text-emerald-500' : 'text-amber-500'}`}
            >
              {result.ok ? <CheckCircle2Icon className="size-4" /> : <AlertCircleIcon className="size-4" />}
              {result.ok
                ? t('fishStory.preview_ok', { count: result.total })
                : t('fishStory.preview_bad', { count: result.invalid, line: result.first_invalid ?? 0 })}
            </p>

            <dl className="space-y-1 text-xs">
              <div className="flex gap-2">
                <dt className="text-muted-foreground">{t('fishStory.preview_endpoint')}</dt>
                <dd className="font-mono break-all">POST {result.endpoint}</dd>
              </div>
              <div className="flex gap-2">
                <dt className="text-muted-foreground">{t('fishStory.preview_headers')}</dt>
                <dd className="font-mono break-all">
                  {Object.entries(result.headers)
                    .map(([k, v]) => `${k}: ${v}`)
                    .join(' · ')}
                </dd>
              </div>
              {result.folder && (
                <div className="flex gap-2">
                  <dt className="text-muted-foreground">{t('fishStory.preview_folder')}</dt>
                  <dd className="font-mono break-all">{result.folder}</dd>
                </div>
              )}
            </dl>

            <label className="flex items-center gap-2 text-xs">
              {t('fishStory.preview_show')}
              <select
                className="h-7 rounded-md border border-input bg-input/20 px-2"
                value={limit}
                onChange={(e) => setLimit(Number(e.target.value))}
              >
                {LIMITS.map((n) => (
                  <option key={n} value={n}>
                    {t('fishStory.preview_first', { count: n })}
                  </option>
                ))}
              </select>
              <span className="text-muted-foreground">
                {result.shown} / {result.total_requests} {t('fishStory.group_requests')}
              </span>
            </label>

            <ol className="space-y-2">
              {result.requests.map((r) => (
                <li key={r.id} className="space-y-1 rounded-lg border border-border/50 p-2">
                  <p className="text-xs">
                    <span className="font-medium">
                      #{r.index}
                      {r.indices.length > 1 ? `–${r.indices[r.indices.length - 1]}` : ''}
                    </span>{' '}
                    {t(`fishStory.request_mode_${r.mode}`)} ·{' '}
                    {r.voice_name ??
                      (Array.isArray(r.voice)
                        ? `${new Set(r.speakers).size} ${t('fishStory.speakers')}`
                        : t('fishStory.fish_default'))}
                    {' · '}
                    {r.chars} {t('fishStory.preview_chars')}
                    {' → '}
                    <span className="font-mono">{r.file}</span>
                  </p>
                  {r.issues.map((issue) => (
                    <p key={issue} role="alert" className="text-xs text-destructive">
                      {issue}
                    </p>
                  ))}
                  {r.notes.map((note) => (
                    <p key={note} className="text-xs text-amber-500">
                      {note}
                    </p>
                  ))}
                  <pre
                    aria-label={t('fishStory.preview_body', { n: r.index })}
                    className="max-h-48 overflow-auto rounded-md bg-background/70 p-2 text-[11px] leading-snug"
                  >
                    {JSON.stringify(r.body, null, 2)}
                  </pre>
                </li>
              ))}
            </ol>

            {result.curl && (
              <details>
                <summary className="cursor-pointer text-xs text-muted-foreground">
                  {t('fishStory.preview_curl')}
                </summary>
                <pre className="mt-1 max-h-40 overflow-auto rounded-md bg-background/70 p-2 text-[11px] whitespace-pre-wrap">
                  {result.curl}
                </pre>
              </details>
            )}
          </div>
        ) : null}

        <DialogFooter>
          <DialogClose render={<Button variant="outline" />}>{t('fishStory.done')}</DialogClose>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
