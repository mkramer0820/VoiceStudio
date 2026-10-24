import { useTranslation } from 'react-i18next';
import { StarIcon } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { qwenInvalid, type QwenForm } from './export-options';

/** The Qwen3-TTS options of the export dialog: speaker name, reference clip, ZIP, limits and Drive folder. */
export function QwenSection({
  form,
  onChange,
  referenceLabel,
  onClearReference,
  speakerCount,
}: {
  form: QwenForm;
  onChange: (form: QwenForm) => void;
  /** Set when the user picked a reference clip in the list. */
  referenceLabel: string | null;
  onClearReference: () => void;
  speakerCount: number;
}) {
  const { t } = useTranslation();
  const bad = new Set(qwenInvalid(form));
  const set = (patch: Partial<QwenForm>) => onChange({ ...form, ...patch });
  const field = (id: 'speaker' | 'minDuration' | 'maxDuration', label: string, hint?: string, narrow = false) => (
    <div className="space-y-1">
      <label htmlFor={`qwen-${id}`} className="text-xs font-medium">
        {label}
      </label>
      <Input
        id={`qwen-${id}`}
        className={narrow ? 'w-28' : undefined}
        aria-invalid={bad.has(id)}
        value={form[id]}
        onChange={(e) => set({ [id]: e.target.value } as Partial<QwenForm>)}
      />
      {hint && <p className="text-[11px] text-muted-foreground">{hint}</p>}
    </div>
  );
  return (
    <div className="space-y-3 rounded-md border border-border/50 p-2" data-testid="qwen-section">
      <p className="text-[11px] leading-snug text-muted-foreground">{t('fishDataset.qwen_note')}</p>
      {speakerCount > 1 && (
        <p role="alert" className="text-[11px] leading-snug text-amber-500">
          {t('fishDataset.qwen_multi', { count: speakerCount })}
        </p>
      )}
      {field('speaker', t('fishDataset.qwen_speaker'), t('fishDataset.qwen_speaker_hint'))}
      <div className="space-y-1">
        <span className="text-xs font-medium">{t('fishDataset.qwen_reference')}</span>
        {referenceLabel ? (
          <p className="flex flex-wrap items-center gap-2 text-xs">
            <StarIcon className="size-3 text-amber-500" aria-hidden />
            <span>{t('fishDataset.qwen_ref_clip', { label: referenceLabel })}</span>
            <Button size="xs" variant="outline" onClick={onClearReference}>
              {t('fishDataset.qwen_ref_clear')}
            </Button>
          </p>
        ) : (
          <p className="text-xs">{t('fishDataset.qwen_ref_auto')}</p>
        )}
        <p className="text-[11px] text-muted-foreground">{t('fishDataset.qwen_ref_hint')}</p>
      </div>
      <div className="grid grid-cols-2 gap-3">
        {field('minDuration', t('fishDataset.qwen_min'), undefined, true)}
        {field('maxDuration', t('fishDataset.qwen_max'), undefined, true)}
      </div>
      <p className="text-[11px] text-muted-foreground">{t('fishDataset.qwen_limits_hint')}</p>
      <label className="flex items-center gap-2 text-xs font-medium">
        <input type="checkbox" checked={form.zip} onChange={(e) => set({ zip: e.target.checked })} />
        {t('fishDataset.qwen_zip')}
      </label>
    </div>
  );
}
