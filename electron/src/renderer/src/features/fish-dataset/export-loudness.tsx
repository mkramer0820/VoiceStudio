import { useTranslation } from 'react-i18next';
import { LoaderCircleIcon } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { LEVEL_PRESETS, LEVEL_RANGE, levelInvalid, levelSplit } from './export-options';
import type { AnalysisStatus, LevelSummary } from './use-fish-dataset';

/**
 * "Match loudness": every exported clip is brought to one whole-clip RMS level. The levels of the
 * current selection sit beside the target, so the value can be chosen from data instead of guessed.
 */
export function LoudnessSection({
  enabled,
  onEnabledChange,
  value,
  onValueChange,
  levels,
  analysis,
  onMeasure,
}: {
  enabled: boolean;
  onEnabledChange: (on: boolean) => void;
  value: string;
  onValueChange: (value: string) => void;
  /** Level spread of the clips that would be exported; absent until the page has loaded it. */
  levels?: LevelSummary | null;
  analysis?: AnalysisStatus | null;
  onMeasure?: () => void;
}) {
  const { t } = useTranslation();
  const invalid = enabled && levelInvalid(value);
  const split = enabled && !invalid && levels?.counts ? levelSplit(levels.counts, Number(value)) : null;
  const measuring = Boolean(analysis?.running);
  const measureButton = onMeasure && (
    <Button size="xs" variant="outline" disabled={measuring} onClick={onMeasure}>
      {measuring ? <LoaderCircleIcon className="animate-spin" /> : null}
      {measuring && analysis ? t('fishDataset.analyzing', { done: analysis.done, total: analysis.total }) : t('fishDataset.analyze')}
    </Button>
  );
  return (
    <div className="space-y-1.5 rounded-md border border-border/50 p-2" data-testid="loudness-section">
      <label className="flex items-center gap-2 text-xs font-medium">
        <input type="checkbox" checked={enabled} onChange={(e) => onEnabledChange(e.target.checked)} />
        {t('fishDataset.export_loudness')}
      </label>
      {enabled && (
        <div className="flex flex-wrap items-center gap-1.5">
          <Input
            id="ds-level"
            aria-label={t('fishDataset.export_loudness_db')}
            aria-invalid={invalid}
            type="number"
            min={LEVEL_RANGE[0]}
            max={LEVEL_RANGE[1]}
            className="w-24"
            value={value}
            onChange={(e) => onValueChange(e.target.value)}
          />
          <span className="text-xs text-muted-foreground">dB</span>
          {LEVEL_PRESETS.map((preset) => (
            <Button
              key={preset}
              size="xs"
              variant={Number(value) === preset ? 'secondary' : 'outline'}
              aria-pressed={Number(value) === preset}
              aria-label={t('fishDataset.level_preset', { db: preset })}
              onClick={() => onValueChange(String(preset))}
            >
              {preset}
            </Button>
          ))}
        </div>
      )}
      {invalid && (
        <p role="alert" className="text-[11px] text-amber-500">
          {t('fishDataset.level_invalid', { min: LEVEL_RANGE[0], max: LEVEL_RANGE[1] })}
        </p>
      )}
      <p className="text-[11px] text-muted-foreground">{t('fishDataset.export_loudness_hint')}</p>
      {levels && (
        <div className="space-y-1 text-[11px]" data-testid="level-summary">
          {levels.measured > 0 && levels.min !== null && levels.max !== null && levels.median !== null ? (
            <p>{t('fishDataset.level_range', { min: levels.min, max: levels.max, median: levels.median })}</p>
          ) : (
            <p className="text-amber-500">{t('fishDataset.level_unmeasured')}</p>
          )}
          {split && levels.measured > 0 && (
            <p>
              {t('fishDataset.level_split', {
                below: split.below,
                above: split.above,
                within: split.within,
                target: Number(value),
              })}
            </p>
          )}
          {levels.measured < levels.total && levels.measured > 0 && (
            <p className="text-muted-foreground">
              {t('fishDataset.level_partial', { measured: levels.measured, total: levels.total })}
            </p>
          )}
          {levels.measured < levels.total && measureButton}
        </div>
      )}
    </div>
  );
}
