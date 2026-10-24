import { useCallback, useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { LoaderCircleIcon, PauseIcon, PlayIcon, SparklesIcon } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { describeError } from '@/lib/api/client';
import { cn } from '@/lib/utils';
import { useClipPlayer } from './clip-player';
import type {
  AnalysisStatus,
  ClipStats,
  ReferenceCandidate,
  ReferenceSuggestion,
  ShortlistClip,
} from './use-fish-dataset';

/** The clip chosen as the Qwen3-TTS reference, with what the panel shows about it. */
export interface ReferenceClip {
  key: string;
  label: string;
  text: string;
  duration_s: number;
  stats?: ClipStats | null;
}

export const SUGGESTIONS = 5;

export const fromCandidate = (c: ReferenceCandidate): ReferenceClip => ({
  key: c.key,
  label: `${c.chapter} #${c.index}`,
  text: c.text,
  duration_s: c.duration_s,
  stats: c.stats,
});

function Details({ clip }: { clip: { duration_s: number; stats?: ClipStats | null } }) {
  const { t } = useTranslation();
  return (
    <span className="tabular-nums text-muted-foreground">
      {clip.stats
        ? t('fishDataset.ref_stats', { duration: clip.duration_s.toFixed(1), level: clip.stats.rms_db, peak: clip.stats.peak.toFixed(2) })
        : t('fishDataset.ref_stats_short', { duration: clip.duration_s.toFixed(1) })}
    </span>
  );
}

/**
 * Qwen3-TTS reference clip: the current choice, the shortlist to pick from, and ranked suggestions for the
 * selection being exported. Exactly one reference (or Automatic) leaves this panel.
 */
export function ReferencePanel({
  open,
  speakerId,
  shortlist,
  reference,
  onChange,
  onSuggest,
  minDuration,
  maxDuration,
  analysis,
  onMeasure,
  onRemoveShortlist,
  onError,
}: {
  open: boolean;
  /** The one speaker being exported, or '' while it is not known. */
  speakerId: string;
  shortlist: ShortlistClip[];
  reference: ReferenceClip | null;
  onChange: (clip: ReferenceClip | null) => void;
  onSuggest: (limit: number, minDuration: number, maxDuration: number) => Promise<ReferenceSuggestion>;
  minDuration: number;
  maxDuration: number;
  analysis?: AnalysisStatus | null;
  onMeasure?: () => void;
  onRemoveShortlist?: (keys: string[]) => void;
  onError: (message: string) => void;
}) {
  const { t } = useTranslation();
  const player = useClipPlayer(onError);
  const [suggestion, setSuggestion] = useState<ReferenceSuggestion | null>(null);
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState('');
  const latest = useRef({ onSuggest, minDuration, maxDuration });
  latest.current = { onSuggest, minDuration, maxDuration };
  const run = useRef(0);

  const suggest = useCallback(async () => {
    const id = ++run.current;
    setBusy(true);
    setProblem('');
    try {
      const { onSuggest: ask, minDuration: lo, maxDuration: hi } = latest.current;
      const result = await ask(SUGGESTIONS, lo, hi);
      if (id === run.current) setSuggestion(result);
    } catch (reason) {
      if (id === run.current) {
        setSuggestion(null);
        setProblem(describeError(reason));
      }
    } finally {
      if (id === run.current) setBusy(false);
    }
  }, []);

  // Look at the selection when the dialog opens, and again after a measurement run ends.
  useEffect(() => {
    if (open) void suggest();
    else player.stop();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, suggest]);
  const wasRunning = useRef(false);
  useEffect(() => {
    if (wasRunning.current && !analysis?.running && open) void suggest();
    wasRunning.current = Boolean(analysis?.running);
  }, [analysis?.running, open, suggest]);

  const mine = shortlist.filter((c) => !speakerId || c.voice_id === speakerId);
  const foreign = shortlist.filter((c) => speakerId && c.voice_id !== speakerId);
  const choices = reference && !mine.some((c) => c.key === reference.key) ? [reference, ...mine] : mine;
  const top = suggestion?.candidates[0];
  const unmeasured = suggestion?.unmeasured ?? 0;
  const measuring = Boolean(analysis?.running);

  const play = (key: string, label: string) => (
    <Button
      size="icon-sm"
      variant="ghost"
      aria-label={t(player.playing === key ? 'fishDataset.ref_stop' : 'fishDataset.ref_play', { label })}
      onClick={() => void player.toggle(key)}
    >
      {player.playing === key ? <PauseIcon /> : <PlayIcon />}
    </Button>
  );

  return (
    <div className="space-y-2" data-testid="reference-panel">
      <span className="text-xs font-medium">{t('fishDataset.qwen_reference')}</span>

      <div className="rounded-md bg-muted/30 p-2 text-xs" data-testid="reference-current">
        {reference ? (
          <div className="flex items-start gap-1">
            {play(reference.key, reference.label)}
            <div className="min-w-0 flex-1 space-y-0.5">
              <p className="font-medium">{t('fishDataset.ref_current', { label: reference.label })}</p>
              <p className="leading-snug">{reference.text}</p>
              <Details clip={reference} />
            </div>
          </div>
        ) : (
          <div className="flex items-start gap-1">
            {top && play(top.key, `${top.chapter} #${top.index}`)}
            <div className="min-w-0 flex-1 space-y-0.5">
              <p className="font-medium">{t('fishDataset.ref_auto_now')}</p>
              {top ? (
                <>
                  <p className="leading-snug">{top.text}</p>
                  <Details clip={top} />
                  <p className="text-[11px] text-muted-foreground">
                    {top.reasons.map((r) => t(`fishDataset.ref_reason_${r}`)).join(' · ')}
                  </p>
                </>
              ) : (
                <p className="text-[11px] text-muted-foreground">{t('fishDataset.ref_auto_unknown')}</p>
              )}
            </div>
          </div>
        )}
      </div>

      <fieldset className="space-y-1" data-testid="reference-choices">
        <legend className="text-[11px] font-medium text-muted-foreground">{t('fishDataset.ref_choose')}</legend>
        <label className="flex items-center gap-2 text-xs">
          <input
            type="radio"
            name="qwen-reference"
            checked={reference === null}
            onChange={() => onChange(null)}
          />
          {t('fishDataset.ref_automatic')}
        </label>
        {choices.map((clip) => (
          <div key={clip.key} className={cn('flex items-start gap-1 text-xs', reference?.key === clip.key && 'font-medium')}>
            <input
              type="radio"
              name="qwen-reference"
              className="mt-2"
              aria-label={t('fishDataset.ref_choose_clip', { label: clip.label })}
              checked={reference?.key === clip.key}
              onChange={() => onChange(clip)}
            />
            {play(clip.key, clip.label)}
            <div className="min-w-0 flex-1 pt-1">
              <p className="leading-snug">{clip.text}</p>
              <p className="text-[11px]">
                {clip.label} · <Details clip={clip} />
              </p>
            </div>
          </div>
        ))}
        {mine.length === 0 && <p className="text-[11px] text-muted-foreground">{t('fishDataset.ref_shortlist_empty')}</p>}
        {foreign.length > 0 && (
          <p role="alert" className="flex flex-wrap items-center gap-2 text-[11px] text-amber-500">
            {t('fishDataset.ref_other_speaker', { count: foreign.length })}
            {onRemoveShortlist && (
              <Button size="xs" variant="outline" onClick={() => onRemoveShortlist(foreign.map((c) => c.key))}>
                {t('fishDataset.ref_remove_other')}
              </Button>
            )}
          </p>
        )}
      </fieldset>

      <div className="space-y-1.5 border-t border-border/50 pt-2">
        <div className="flex flex-wrap items-center gap-2">
          <Button size="xs" variant="outline" disabled={busy} onClick={() => void suggest()}>
            {busy ? <LoaderCircleIcon className="animate-spin" /> : <SparklesIcon />}
            {t('fishDataset.ref_suggest')}
          </Button>
          <span className="text-[11px] text-muted-foreground">{t('fishDataset.ref_suggest_hint', { count: SUGGESTIONS })}</span>
        </div>
        {problem && (
          <p role="alert" className="text-[11px] text-amber-500">
            {problem}
          </p>
        )}
        {suggestion && unmeasured > 0 && (
          <div className="flex flex-wrap items-center gap-2 text-[11px] text-amber-500" data-testid="reference-unmeasured">
            <span>{t('fishDataset.ref_unmeasured', { unmeasured, clips: suggestion.clips })}</span>
            {onMeasure && (
              <Button size="xs" variant="outline" disabled={measuring} onClick={onMeasure}>
                {measuring && analysis
                  ? t('fishDataset.analyzing', { done: analysis.done, total: analysis.total })
                  : t('fishDataset.analyze')}
              </Button>
            )}
          </div>
        )}
        {suggestion && suggestion.candidates.length === 0 && unmeasured === 0 && (
          <p className="text-[11px] text-muted-foreground">{t('fishDataset.ref_none')}</p>
        )}
        <ol className="space-y-1.5" data-testid="reference-suggestions">
          {(suggestion?.candidates ?? []).map((c, i) => (
            <li key={c.key} className="flex items-start gap-1 text-xs" data-testid="reference-suggestion">
              {play(c.key, `${c.chapter} #${c.index}`)}
              <div className="min-w-0 flex-1 space-y-0.5 pt-1">
                <p className="leading-snug">
                  <span className="tabular-nums text-muted-foreground">{i + 1}. </span>
                  {c.text}
                </p>
                <p className="text-[11px]">
                  {c.chapter} #{c.index} · <Details clip={c} />
                </p>
                <p className="flex flex-wrap gap-1 text-[10px]">
                  {c.reasons.map((r) => (
                    <span key={r} className="rounded-full bg-emerald-500/15 px-1.5 text-emerald-500">
                      {t(`fishDataset.ref_reason_${r}`)}
                    </span>
                  ))}
                  {c.cautions.map((r) => (
                    <span key={r} className="rounded-full bg-amber-500/15 px-1.5 text-amber-500">
                      {t(`fishDataset.ref_caution_${r}`)}
                    </span>
                  ))}
                </p>
              </div>
              <Button
                size="xs"
                variant={reference?.key === c.key ? 'secondary' : 'outline'}
                aria-label={t('fishDataset.ref_use_clip', { label: `${c.chapter} #${c.index}` })}
                onClick={() => onChange(fromCandidate(c))}
              >
                {t('fishDataset.ref_use')}
              </Button>
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
}
