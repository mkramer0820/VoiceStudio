import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Input } from '@/components/ui/input';
import {
  CHOICES,
  DEFAULT_CHECKPOINT,
  NUMBERS,
  QWEN_LORA_KEYS,
  QWEN_LORA_NUMBERS,
  QWEN_NUMBERS,
  estimatePasses,
  invalidKeys,
  numberOf,
  qwenDefault,
  qwenEstimate,
  qwenLoraNumber,
  qwenMode,
  qwenNumber,
  qwenText,
  qwenTrainInvalid,
  suggestSteps,
  type QwenLoraKey,
  type QwenMode,
  type QwenNumberKey,
  type QwenTextKey,
  type QwenTrainForm,
  type TrainingChoice,
  type TrainingForm,
  type TrainingKey,
} from './export-options';

const selectClass =
  'h-8 w-full rounded-md border border-input bg-input/20 px-2 text-sm outline-none focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/30 dark:bg-input/30';

/** One labelled number box with the range the backend accepts; shared by the OmniVoice and Qwen3-TTS fields. */
function NumberField({
  id, label, spec, value, invalid, onChange,
}: {
  id: string;
  label: string;
  spec: { min: number; max: number; step: string };
  value: string;
  invalid: boolean;
  onChange: (value: string) => void;
}) {
  return (
    <div className="space-y-1">
      <label htmlFor={id} className="text-xs font-medium">
        {label}
      </label>
      <Input
        id={id}
        type="number"
        min={spec.min}
        max={spec.max}
        step={spec.step}
        aria-invalid={invalid}
        className={invalid ? 'border-destructive' : ''}
        value={value}
        onChange={(e) => onChange(e.target.value)}
      />
    </div>
  );
}

/** The settings written into train_config.json, with a live estimate of how often the clips are seen. */
export function TrainingSection({
  form,
  onChange,
  seconds,
}: {
  form: TrainingForm;
  onChange: (form: TrainingForm) => void;
  /** Seconds of audio that will be trained on (the validation share is already taken out). */
  seconds: number;
}) {
  const { t } = useTranslation();
  const [passes, setPasses] = useState('');
  const bad = new Set(invalidKeys(form));
  const lora = form.mode === 'lora';
  const edit = (key: TrainingKey | TrainingChoice | 'base_checkpoint', value: string) =>
    onChange({ ...form, edits: { ...form.edits, [key]: value } });
  const shown = (key: TrainingKey) => form.edits[key] ?? String(numberOf(form, key));

  const number = (key: TrainingKey) => {
    const spec = NUMBERS[key];
    if (spec.lora && !lora) return null;
    return (
      <NumberField
        key={key}
        id={`tr-${key}`}
        label={t(`fishDataset.tr_${key}`)}
        spec={spec}
        value={shown(key)}
        invalid={bad.has(key)}
        onChange={(value) => edit(key, value)}
      />
    );
  };
  const choice = (key: TrainingChoice) => (
    <div key={key} className="space-y-1">
      <label htmlFor={`tr-${key}`} className="text-xs font-medium">
        {t(`fishDataset.tr_${key}`)}
      </label>
      <select
        id={`tr-${key}`}
        className={selectClass}
        value={form.edits[key] ?? CHOICES[key].def}
        onChange={(e) => edit(key, e.target.value)}
      >
        {CHOICES[key].options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </div>
  );

  const batch = numberOf(form, 'batch_tokens');
  const steps = numberOf(form, 'steps');
  const estimate = Number.isFinite(batch) && Number.isFinite(steps) && seconds > 0 ? estimatePasses(seconds, steps, batch) : null;
  const overfit = estimate !== null && estimate > 40 && seconds < 3600;
  const advanced = (Object.keys(NUMBERS) as TrainingKey[]).filter((key) => NUMBERS[key].advanced);
  const main = (Object.keys(NUMBERS) as TrainingKey[]).filter((key) => !NUMBERS[key].advanced);

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-3">
        <div className="space-y-1">
          <label htmlFor="tr-mode" className="text-xs font-medium">
            {t('fishDataset.tr_mode')}
          </label>
          <select
            id="tr-mode"
            className={selectClass}
            value={form.mode}
            onChange={(e) => onChange({ ...form, mode: e.target.value as TrainingForm['mode'] })}
          >
            <option value="lora">{t('fishDataset.tr_mode_lora')}</option>
            <option value="full">{t('fishDataset.tr_mode_full')}</option>
          </select>
        </div>
        {choice('mixed_precision')}
        {main.map(number)}
      </div>

      <div className="space-y-1">
        <label htmlFor="tr-passes" className="text-xs font-medium">
          {t('fishDataset.tr_passes_set')}
        </label>
        <Input
          id="tr-passes"
          type="number"
          min={0.1}
          step="any"
          className="w-24"
          value={passes}
          disabled={seconds <= 0}
          onChange={(e) => {
            setPasses(e.target.value);
            const wanted = Number(e.target.value);
            if (wanted > 0 && Number.isFinite(batch) && batch > 0 && seconds > 0) {
              edit('steps', String(suggestSteps(seconds, batch, wanted)));
            }
          }}
        />
      </div>
      {estimate !== null && (
        <div className="space-y-0.5 text-[11px] leading-snug text-muted-foreground">
          <p data-testid="passes-estimate">{t('fishDataset.tr_passes_estimate', { passes: estimate })}</p>
          <p>{t('fishDataset.tr_passes_note')}</p>
          {overfit && (
            <p role="alert" className="text-amber-500">
              {t('fishDataset.tr_overfit', { passes: estimate })}
            </p>
          )}
        </div>
      )}

      {lora && (
        <label className="flex items-start gap-2 text-xs">
          <input
            type="checkbox"
            className="mt-0.5"
            checked={form.heads}
            onChange={(e) => onChange({ ...form, heads: e.target.checked })}
          />
          <span>
            {t('fishDataset.tr_train_heads')}
            <span className="block text-muted-foreground">{t('fishDataset.tr_train_heads_hint')}</span>
          </span>
        </label>
      )}
      <div className="space-y-1">
        <label htmlFor="tr-base" className="text-xs font-medium">
          {t('fishDataset.tr_base_checkpoint')}
        </label>
        <Input
          id="tr-base"
          className="font-mono text-xs"
          value={form.edits.base_checkpoint ?? DEFAULT_CHECKPOINT}
          onChange={(e) => edit('base_checkpoint', e.target.value)}
        />
      </div>

      <details className="rounded-md border border-border/50 p-2">
        <summary className="cursor-pointer text-xs font-medium">{t('fishDataset.tr_advanced')}</summary>
        <div className="mt-2 grid grid-cols-2 gap-3">
          {advanced.slice(0, 2).map(number)}
          {choice('lr_scheduler_type')}
          {advanced.slice(2).map(number)}
        </div>
        <p className="mt-2 text-[11px] text-muted-foreground">{t('fishDataset.tr_save_hint')}</p>
      </details>
    </div>
  );
}

/** Training setup for the Qwen3-TTS format: only what the official sft_12hz.py / prepare_data.py take. */
export function QwenTrainingSection({
  form,
  onChange,
  clips,
  speaker,
}: {
  form: QwenTrainForm;
  onChange: (form: QwenTrainForm) => void;
  /** Clips that will be trained on (an estimate: clips the checks reject are not counted out). */
  clips: number;
  speaker: string;
}) {
  const { t } = useTranslation();
  const bad = new Set<string>(qwenTrainInvalid(form));
  const edit = (key: QwenNumberKey | QwenTextKey | QwenLoraKey, value: string) => onChange({ ...form, edits: { ...form.edits, [key]: value } });
  const mode = qwenMode(form);
  const number = (key: QwenNumberKey) => (
    <NumberField
      key={key}
      id={`qtr-${key}`}
      label={t(`fishDataset.qtr_${key}`)}
      spec={QWEN_NUMBERS[key]}
      value={form.edits[key] ?? String(qwenDefault(form, key))}
      invalid={bad.has(key)}
      onChange={(value) => edit(key, value)}
    />
  );
  const lora = (key: QwenLoraKey) => (
    <NumberField
      key={key}
      id={`qtr-${key}`}
      label={t(`fishDataset.qtr_${key}`)}
      spec={QWEN_LORA_NUMBERS[key]}
      value={form.edits[key] ?? String(qwenLoraNumber(form, key))}
      invalid={bad.has(key)}
      onChange={(value) => edit(key, value)}
    />
  );
  const text = (key: QwenTextKey) => (
    <div key={key} className="space-y-1">
      <label htmlFor={`qtr-${key}`} className="text-xs font-medium">
        {t(`fishDataset.qtr_${key}`)}
      </label>
      <Input
        id={`qtr-${key}`}
        className="font-mono text-xs"
        aria-invalid={bad.has(key)}
        value={qwenText(form, key)}
        onChange={(e) => edit(key, e.target.value)}
      />
    </div>
  );
  const estimate = qwenEstimate(clips, qwenNumber(form, 'batch_size'), qwenNumber(form, 'epochs'));
  const folder = qwenText(form, 'drive_folder').trim().replace(/^\/+|\/+$/g, '');
  return (
    <div className="space-y-3">
      <p className="rounded-md bg-muted/40 p-2 text-[11px] leading-snug text-muted-foreground">
        {t('fishDataset.qtr_colab_only')}
      </p>
      <div className="space-y-1">
        <label htmlFor="qtr-train_mode" className="text-xs font-medium">
          {t('fishDataset.qtr_train_mode')}
        </label>
        <select
          id="qtr-train_mode"
          className="h-8 w-full rounded-md border border-input bg-background px-2 text-xs"
          value={mode}
          onChange={(e) => onChange({ ...form, mode: e.target.value as QwenMode })}
        >
          <option value="full">{t('fishDataset.qtr_mode_full')}</option>
          <option value="lora">{t('fishDataset.qtr_mode_lora')}</option>
        </select>
        {mode === 'lora' && <p className="text-[11px] leading-snug text-muted-foreground">{t('fishDataset.qtr_lora_note')}</p>}
      </div>
      <div className="grid grid-cols-2 gap-3">{(Object.keys(QWEN_NUMBERS) as QwenNumberKey[]).map(number)}</div>
      {mode === 'lora' && <div className="grid grid-cols-3 gap-3">{QWEN_LORA_KEYS.map(lora)}</div>}
      {text('drive_folder')}
      <p className="break-all font-mono text-[11px] text-muted-foreground" data-testid="qwen-checkpoint-path">
        My Drive/{folder}/runs/{speaker.trim() || '<speaker>'}/{'<run_id>'}/checkpoints
      </p>
      <p className="text-[11px] text-muted-foreground">{t('fishDataset.qtr_checkpoint_hint')}</p>
      {estimate && (
        <div className="space-y-0.5 text-[11px] leading-snug text-muted-foreground">
          <p data-testid="qwen-estimate">
            {t('fishDataset.qtr_estimate', { batches: estimate.batches, updates: estimate.updates, total: estimate.total })}
          </p>
          <p>{t('fishDataset.qtr_estimate_note')}</p>
        </div>
      )}
      {text('base_model')}
      <details className="rounded-md border border-border/50 p-2">
        <summary className="cursor-pointer text-xs font-medium">{t('fishDataset.tr_advanced')}</summary>
        <div className="mt-2 space-y-3">
          {text('tokenizer_model')}
          <p className="text-[11px] text-muted-foreground">{t('fishDataset.qtr_fixed')}</p>
        </div>
      </details>
    </div>
  );
}

