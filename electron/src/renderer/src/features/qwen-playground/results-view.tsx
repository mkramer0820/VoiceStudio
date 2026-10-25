import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { AlertTriangleIcon, StarIcon } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { RenderPlayer, type AbPick } from './audio-player';
import { modelIsMissing, resultWarnings, type QwenModel, type RenderResult, type TakeView } from './qwen-model';

interface Props {
  session: string;
  take: TakeView;
  /** Models the app can find now; a take whose model is not among them gets a warning. */
  models?: QwenModel[];
  ab: { a: AbPick | null; b: AbPick | null };
  onPick: (slot: 'a' | 'b', pick: AbPick) => void;
  onAnnotate: (takeId: string, patch: { notes?: string; starred?: boolean }) => Promise<boolean> | void;
}

function ResultRow({ session, take, result, ab, onPick }: Pick<Props, 'session' | 'take' | 'ab' | 'onPick'> & { result: RenderResult }) {
  const { t } = useTranslation();
  const file = result.file;
  const warnings = resultWarnings(result);
  const pick = (slot: 'a' | 'b') =>
    file && onPick(slot, { session, file, label: `#${take.number}${take.repeat > 1 ? ` r${result.rep}` : ''} ${take.label || take.model?.label || ''}`.trim() });
  return (
    <li className="space-y-1 rounded-md bg-background/60 p-2 text-xs">
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-medium">{t('qwenPlayground.render_n', { n: result.rep })}</span>
        {result.status === 'running' && <span className="text-muted-foreground">{t('qwenPlayground.rendering')}</span>}
        {result.status === 'done' && file && (
          <>
            <RenderPlayer session={session} file={file} />
            <span className="tabular-nums text-muted-foreground">
              {t('qwenPlayground.seconds', { s: result.seconds ?? 0 })}
              {result.realtime != null && ` · ${t('qwenPlayground.realtime', { x: result.realtime })}`}
              {result.seed != null && ` · ${t('qwenPlayground.seed_used', { seed: result.seed })}`}
            </span>
            <Button size="xs" variant={ab.a?.file === file && ab.a.session === session ? 'default' : 'outline'} onClick={() => pick('a')}>
              {t('qwenPlayground.ab_set_a')}
            </Button>
            <Button size="xs" variant={ab.b?.file === file && ab.b.session === session ? 'default' : 'outline'} onClick={() => pick('b')}>
              {t('qwenPlayground.ab_set_b')}
            </Button>
          </>
        )}
      </div>
      {result.status === 'error' && (
        <p role="alert" className="text-destructive">
          {t('qwenPlayground.render_failed', { error: result.error ?? '' })}
        </p>
      )}
      {result.status === 'done' && result.complete === false && (
        <p role="alert" className="flex items-start gap-1 text-amber-500">
          <AlertTriangleIcon className="mt-0.5 size-3.5 shrink-0" />
          {t('qwenPlayground.incomplete')}
        </p>
      )}
      {warnings.map((w) => (
        <p key={w} role="alert" className="flex items-start gap-1 text-amber-500">
          <AlertTriangleIcon className="mt-0.5 size-3.5 shrink-0" />
          {w}
        </p>
      ))}
    </li>
  );
}

/** One take's renders, notes and star: used for the live run and for a past session alike. */
export function ResultsView({ session, take, models, ab, onPick, onAnnotate }: Props) {
  const { t } = useTranslation();
  const [notes, setNotes] = useState(take.notes ?? '');
  const [saved, setSaved] = useState(take.notes ?? '');
  const saveNotes = async () => {
    if (notes === saved) return;
    if ((await onAnnotate(take.id, { notes })) !== false) setSaved(notes);
  };
  const doneCount = take.results.filter((r) => r.status === 'done' || r.status === 'error').length;
  return (
    <div className="space-y-2 border-t border-border/40 pt-2">
      {modelIsMissing(take.model_id, models) && (
        <p role="alert" className="flex items-start gap-1 text-xs text-amber-500">
          <AlertTriangleIcon className="mt-0.5 size-3.5 shrink-0" />
          {t('modelLocations.model_not_found', { label: take.model?.label ?? take.model_id, path: take.model?.path ?? '' })}
        </p>
      )}
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <span>{t('qwenPlayground.take_progress', { done: doneCount, total: take.repeat })}</span>
        <Button
          size="icon-xs"
          variant="ghost"
          aria-pressed={take.starred}
          aria-label={take.starred ? t('qwenPlayground.unstar') : t('qwenPlayground.star')}
          onClick={() => void onAnnotate(take.id, { starred: !take.starred })}
        >
          <StarIcon className={take.starred ? 'fill-amber-400 text-amber-400' : ''} />
        </Button>
      </div>
      {take.results.length > 0 && (
        <ul className="space-y-1.5">
          {take.results.map((r) => (
            <ResultRow key={r.rep} session={session} take={take} result={r} ab={ab} onPick={onPick} />
          ))}
        </ul>
      )}
      <textarea
        aria-label={t('qwenPlayground.notes')}
        placeholder={t('qwenPlayground.notes_placeholder')}
        rows={2}
        value={notes}
        className="w-full resize-y rounded-md border border-input bg-input/20 p-2 text-xs"
        onChange={(e) => setNotes(e.target.value)}
        onBlur={() => void saveNotes()}
      />
    </div>
  );
}
