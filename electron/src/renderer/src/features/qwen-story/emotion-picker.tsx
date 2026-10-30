import { useTranslation } from 'react-i18next';
import type { Emotion, EmotionCatalog, Strength } from './story-model';

export const selectClass =
  'h-8 min-w-0 rounded-md border border-input bg-input/20 px-2 text-sm outline-none focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/30 dark:bg-input/30';

/** Emotion preset + strength. `value` null = "not set" (inherit). Instruct strength is UNVERIFIED on the fine-tuned narrator. */
export function EmotionPicker({
  catalog,
  value,
  onChange,
  label,
  allowInherit = false,
}: {
  catalog: EmotionCatalog | null;
  value: Emotion | null;
  onChange: (value: Emotion | null) => void;
  label: string;
  allowInherit?: boolean;
}) {
  const { t } = useTranslation();
  const presets = catalog?.presets ?? [];
  const current = value?.preset ?? (allowInherit ? '' : 'neutral');
  const strengths: Strength[] = catalog?.strengths ?? ['normal', 'strong', 'very_strong'];
  return (
    <span className="inline-flex flex-wrap items-center gap-1">
      <select
        aria-label={label}
        className={`${selectClass} w-28`}
        value={current}
        title={catalog?.unverified_on_narrator}
        onChange={(e) => {
          if (e.target.value === '') onChange(null);
          else onChange({ preset: e.target.value, strength: value?.strength ?? 'normal', custom_text: value?.custom_text ?? null });
        }}
      >
        {allowInherit && <option value="">{t('qwenStory.emotion_inherit')}</option>}
        {presets.map((p) => (
          <option key={p.name} value={p.name}>
            {t(`qwenStory.emotion_${p.name}`)}
          </option>
        ))}
      </select>
      {value && value.preset !== 'neutral' && value.preset !== 'custom' && (
        <select
          aria-label={t('qwenStory.strength')}
          className={`${selectClass} w-28`}
          value={value.strength}
          onChange={(e) => onChange({ ...value, strength: e.target.value as Strength })}
        >
          {strengths.map((s) => (
            <option key={s} value={s}>
              {t(`qwenStory.strength_${s}`)}
            </option>
          ))}
        </select>
      )}
      {value?.preset === 'custom' && (
        <input
          aria-label={t('qwenStory.custom_instruct')}
          className={`${selectClass} w-56`}
          value={value.custom_text ?? ''}
          placeholder={t('qwenStory.custom_instruct')}
          onChange={(e) => onChange({ ...value, custom_text: e.target.value })}
        />
      )}
    </span>
  );
}
