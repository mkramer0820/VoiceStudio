import { useId, useState, type ReactNode } from 'react';
import { useTranslation } from 'react-i18next';
import { Button } from '@/components/ui/button';
import { getBridge } from '@/components/bridge';
import { Input } from '@/components/ui/input';
import { Switch } from '@/components/ui/switch';
import type { StorySettings } from './story-model';
import type { FishDefaults } from './use-fish-story';

const MODELS = ['s2.1-pro', 's2.1-pro-free', 's2-pro', 'drama-3'];
const FORMATS = ['wav', 'pcm', 'mp3', 'opus'] as const;
const SAMPLE_RATES = [24000, 32000, 44100, 48000];
const BITRATES = [64, 128, 192] as const;

const selectClass =
  'h-8 w-full rounded-md border border-input bg-input/20 px-2 text-sm outline-none focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/30 dark:bg-input/30';

function Field({
  label,
  hint,
  children,
  inline = false,
}: {
  label: string;
  hint?: string;
  children: (id: string) => ReactNode;
  inline?: boolean;
}) {
  const id = useId();
  return (
    <div className={inline ? 'flex items-center justify-between gap-3' : 'space-y-1'}>
      <label htmlFor={id} className="text-xs font-medium text-foreground/80">
        {label}
      </label>
      {children(id)}
      {hint && <p className="text-[11px] leading-snug text-muted-foreground">{hint}</p>}
    </div>
  );
}

function Group({
  title,
  children,
  disabled = false,
  note,
}: {
  title: string;
  children: ReactNode;
  /** Greys out every control in the group (a native disabled fieldset). */
  disabled?: boolean;
  note?: string;
}) {
  return (
    <fieldset
      disabled={disabled}
      className={`space-y-3 border-t border-border/50 pt-3 first:border-t-0 first:pt-0 ${disabled ? 'opacity-60' : ''}`}
    >
      <legend className="mb-2 text-xs font-semibold tracking-wide text-muted-foreground uppercase">
        {title}
      </legend>
      {disabled && note && <p className="text-[11px] leading-snug text-amber-500">{note}</p>}
      {children}
    </fieldset>
  );
}

/** Empty input = inherit the Settings/env value shown as the placeholder. */
function numberOrNull(raw: string): number | null {
  if (raw.trim() === '') return null;
  const value = Number(raw);
  return Number.isFinite(value) ? value : null;
}

export function SettingsPanel({
  settings,
  defaults,
  voiceNames,
  onChange,
  onSaveDefaults,
  onError,
}: {
  settings: StorySettings;
  defaults: FishDefaults | null;
  voiceNames: Record<string, string>;
  onChange: (patch: Partial<StorySettings>) => void;
  /** Persist to Settings; resolves true on success. */
  onSaveDefaults: (patch: Record<string, unknown>) => Promise<boolean>;
  onError: (message: string) => void;
}) {
  const { t } = useTranslation();
  const [folder, setFolder] = useState('');
  const [note, setNote] = useState('');
  const format = settings.format ?? defaults?.format ?? 'wav';
  const lossless = format === 'wav' || format === 'pcm';
  const inherit = (value: string | number | null | undefined) =>
    value === null || value === undefined ? '' : String(value);
  const triState = (value: boolean | null | undefined) =>
    value === null || value === undefined ? 'default' : value ? 'on' : 'off';
  const fromTriState = (value: string) => (value === 'default' ? null : value === 'on');

  return (
    <div className="space-y-4 p-1 text-sm">
      <Group title={t('fishStory.group_output')}>
        <Field label={t('fishStory.format')} hint={lossless ? t('fishStory.format_lossless') : t('fishStory.format_lossy')}>
          {(id) => (
            <select
              id={id}
              className={selectClass}
              value={settings.format ?? ''}
              onChange={(e) => onChange({ format: (e.target.value || null) as StorySettings['format'] })}
            >
              <option value="">{t('fishStory.inherit', { value: defaults?.format ?? 'wav' })}</option>
              {FORMATS.map((f) => (
                <option key={f} value={f}>
                  {f.toUpperCase()}
                </option>
              ))}
            </select>
          )}
        </Field>
        {format !== 'opus' && (
          <Field label={t('fishStory.sample_rate')}>
            {(id) => (
              <select
                id={id}
                className={selectClass}
                value={settings.sample_rate ?? ''}
                onChange={(e) => onChange({ sample_rate: numberOrNull(e.target.value) })}
              >
                <option value="">{t('fishStory.inherit', { value: `${defaults?.sample_rate ?? 44100} Hz` })}</option>
                {SAMPLE_RATES.map((r) => (
                  <option key={r} value={r}>
                    {r} Hz
                  </option>
                ))}
              </select>
            )}
          </Field>
        )}
        {format === 'mp3' && (
          <Field label={t('fishStory.mp3_bitrate')}>
            {(id) => (
              <select
                id={id}
                className={selectClass}
                value={settings.mp3_bitrate ?? ''}
                onChange={(e) =>
                  onChange({ mp3_bitrate: (numberOrNull(e.target.value) as StorySettings['mp3_bitrate']) })
                }
              >
                <option value="">{t('fishStory.inherit', { value: `${defaults?.mp3_bitrate ?? 192} kbps` })}</option>
                {BITRATES.map((b) => (
                  <option key={b} value={b}>
                    {b} kbps
                  </option>
                ))}
              </select>
            )}
          </Field>
        )}
        <Field
          label={t('fishStory.join')}
          hint={
            lossless
              ? t(
                  settings.request_mode === 'multi'
                    ? 'fishStory.join_hint_multi'
                    : 'fishStory.join_hint',
                )
              : t('fishStory.join_unavailable')
          }
          inline
        >
          {(id) => (
            <Switch
              id={id}
              checked={settings.join && lossless}
              disabled={!lossless}
              onCheckedChange={(join) => onChange({ join })}
            />
          )}
        </Field>
        <Field
          label={t('fishStory.pause')}
          hint={t(
            settings.request_mode === 'multi'
              ? 'fishStory.pause_hint_multi'
              : 'fishStory.pause_hint',
          )}
        >
          {(id) => (
            <Input
              id={id}
              type="number"
              min={0}
              max={5000}
              step={50}
              value={settings.pause_ms}
              onChange={(e) =>
                onChange({ pause_ms: Math.max(0, Math.min(5000, Number(e.target.value) || 0)) })
              }
            />
          )}
        </Field>
      </Group>

      <Group title={t('fishStory.group_voice')}>
        <Field label={t('fishStory.model')}>
          {(id) => (
            <select
              id={id}
              className={selectClass}
              value={settings.model ?? ''}
              onChange={(e) => onChange({ model: e.target.value || null })}
            >
              <option value="">{t('fishStory.inherit', { value: defaults?.model ?? 's2.1-pro' })}</option>
              {MODELS.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          )}
        </Field>
        <Field label={t('fishStory.default_voice')} hint={t('fishStory.default_voice_hint')}>
          {(id) => (
            <select
              id={id}
              className={selectClass}
              value={settings.default_voice ?? ''}
              onChange={(e) => onChange({ default_voice: e.target.value || null })}
            >
              <option value="">
                {defaults?.reference_id
                  ? t('fishStory.inherit', { value: voiceNames[defaults.reference_id] ?? defaults.reference_id })
                  : t('fishStory.fish_default')}
              </option>
              {Object.entries(voiceNames).map(([vid, name]) => (
                <option key={vid} value={vid}>
                  {name}
                </option>
              ))}
            </select>
          )}
        </Field>
        <Field label={`${t('fishStory.speed')} · ${settings.speed.toFixed(2)}×`} hint={t('fishStory.speed_hint')}>
          {(id) => (
            <input
              id={id}
              type="range"
              min={0.5}
              max={2}
              step={0.05}
              value={settings.speed}
              className="w-full accent-[hsl(var(--primary))]"
              onChange={(e) => onChange({ speed: Number(e.target.value) })}
            />
          )}
        </Field>
        <Field label={t('fishStory.volume')} hint={t('fishStory.volume_hint')}>
          {(id) => (
            <Input
              id={id}
              type="number"
              min={-30}
              max={30}
              step={0.5}
              placeholder={inherit(defaults?.volume)}
              value={inherit(settings.volume)}
              onChange={(e) => onChange({ volume: numberOrNull(e.target.value) })}
            />
          )}
        </Field>
        <Field label={t('fishStory.language')} hint={t('fishStory.language_hint')}>
          {(id) => (
            <Input
              id={id}
              maxLength={35}
              placeholder={defaults?.language ?? t('fishStory.auto')}
              value={settings.language ?? ''}
              onChange={(e) => onChange({ language: e.target.value.trim() || null })}
            />
          )}
        </Field>
      </Group>

      <Group
        title={t('fishStory.group_quality')}
        disabled={settings.minimal_request === true}
        note={t('fishStory.minimal_ignores')}
      >
        <Field label={t('fishStory.latency')} hint={t('fishStory.latency_hint')}>
          {(id) => (
            <select
              id={id}
              className={selectClass}
              value={settings.latency ?? ''}
              onChange={(e) => onChange({ latency: (e.target.value || null) as StorySettings['latency'] })}
            >
              <option value="">{t('fishStory.inherit', { value: defaults?.latency ?? 'normal' })}</option>
              <option value="normal">{t('fishStory.latency_normal')}</option>
              <option value="balanced">{t('fishStory.latency_balanced')}</option>
            </select>
          )}
        </Field>
        <Field label={t('fishStory.normalize')} hint={t('fishStory.normalize_hint')}>
          {(id) => (
            <select
              id={id}
              className={selectClass}
              value={triState(settings.normalize)}
              onChange={(e) => onChange({ normalize: fromTriState(e.target.value) })}
            >
              <option value="default">{t('fishStory.inherit', { value: defaults?.normalize ? 'on' : 'off' })}</option>
              <option value="on">{t('fishStory.on')}</option>
              <option value="off">{t('fishStory.off')}</option>
            </select>
          )}
        </Field>
        <Field label={t('fishStory.loudness')} hint={t('fishStory.loudness_hint')}>
          {(id) => (
            <select
              id={id}
              className={selectClass}
              value={triState(settings.normalize_loudness)}
              onChange={(e) => onChange({ normalize_loudness: fromTriState(e.target.value) })}
            >
              <option value="default">
                {t('fishStory.inherit', { value: defaults?.normalize_loudness === false ? 'off' : 'on' })}
              </option>
              <option value="on">{t('fishStory.on')}</option>
              <option value="off">{t('fishStory.off')}</option>
            </select>
          )}
        </Field>
        <div className="grid grid-cols-2 gap-2">
          <Field label={t('fishStory.temperature')}>
            {(id) => (
              <Input
                id={id}
                type="number"
                min={0}
                max={1}
                step={0.05}
                placeholder={inherit(defaults?.temperature)}
                value={inherit(settings.temperature)}
                onChange={(e) => onChange({ temperature: numberOrNull(e.target.value) })}
              />
            )}
          </Field>
          <Field label={t('fishStory.top_p')}>
            {(id) => (
              <Input
                id={id}
                type="number"
                min={0}
                max={1}
                step={0.05}
                placeholder={inherit(defaults?.top_p)}
                value={inherit(settings.top_p)}
                onChange={(e) => onChange({ top_p: numberOrNull(e.target.value) })}
              />
            )}
          </Field>
        </div>
        <Field label={t('fishStory.chunk_length')} hint={t('fishStory.chunk_length_hint')}>
          {(id) => (
            <Input
              id={id}
              type="number"
              min={100}
              max={300}
              step={10}
              placeholder={inherit(defaults?.chunk_length)}
              value={inherit(settings.chunk_length)}
              onChange={(e) => onChange({ chunk_length: numberOrNull(e.target.value) })}
            />
          )}
        </Field>
      </Group>

      <Group
        title={t('fishStory.group_advanced')}
        disabled={settings.minimal_request === true}
        note={t('fishStory.minimal_ignores')}
      >
        <p className="text-[11px] leading-snug text-muted-foreground">{t('fishStory.advanced_note')}</p>
        <div className="space-y-1">
          <Button
            size="xs"
            variant="outline"
            onClick={() => {
              const cleared = {
                temperature: null, top_p: null, chunk_length: null, repetition_penalty: null,
                max_new_tokens: null, min_chunk_length: null, condition_on_previous_chunks: null,
                early_stop_threshold: null, opus_bitrate: null,
              };
              onChange(cleared);
              void onSaveDefaults(cleared).then((ok) => ok && setNote(t('fishStory.fish_defaults_done')));
            }}
          >
            {t('fishStory.fish_defaults')}
          </Button>
          <p className="text-[11px] leading-snug text-muted-foreground">{t('fishStory.fish_defaults_hint')}</p>
        </div>
        <div className="grid grid-cols-2 gap-2">
          <Field label={t('fishStory.repetition_penalty')} hint={t('fishStory.repetition_penalty_hint')}>
            {(id) => (
              <Input
                id={id}
                type="number"
                min={0.5}
                max={2}
                step={0.05}
                placeholder={inherit(defaults?.repetition_penalty)}
                value={inherit(settings.repetition_penalty)}
                onChange={(e) => onChange({ repetition_penalty: numberOrNull(e.target.value) })}
              />
            )}
          </Field>
          <Field label={t('fishStory.max_new_tokens')} hint={t('fishStory.max_new_tokens_hint')}>
            {(id) => (
              <Input
                id={id}
                type="number"
                min={64}
                max={8192}
                step={64}
                placeholder={inherit(defaults?.max_new_tokens)}
                value={inherit(settings.max_new_tokens)}
                onChange={(e) => onChange({ max_new_tokens: numberOrNull(e.target.value) })}
              />
            )}
          </Field>
          <Field label={t('fishStory.min_chunk_length')}>
            {(id) => (
              <Input
                id={id}
                type="number"
                min={0}
                max={300}
                placeholder={inherit(defaults?.min_chunk_length)}
                value={inherit(settings.min_chunk_length)}
                onChange={(e) => onChange({ min_chunk_length: numberOrNull(e.target.value) })}
              />
            )}
          </Field>
          <Field label={t('fishStory.early_stop')}>
            {(id) => (
              <Input
                id={id}
                type="number"
                min={0}
                max={1}
                step={0.05}
                placeholder={inherit(defaults?.early_stop_threshold)}
                value={inherit(settings.early_stop_threshold)}
                onChange={(e) => onChange({ early_stop_threshold: numberOrNull(e.target.value) })}
              />
            )}
          </Field>
        </div>
        <Field label={t('fishStory.condition_previous')} hint={t('fishStory.condition_previous_hint')}>
          {(id) => (
            <select
              id={id}
              className={selectClass}
              value={triState(settings.condition_on_previous_chunks)}
              onChange={(e) => onChange({ condition_on_previous_chunks: fromTriState(e.target.value) })}
            >
              <option value="default">{t('fishStory.fish_default')}</option>
              <option value="on">{t('fishStory.on')}</option>
              <option value="off">{t('fishStory.off')}</option>
            </select>
          )}
        </Field>
        {settings.format === 'opus' && (
          <Field label={t('fishStory.opus_bitrate')} hint={t('fishStory.opus_bitrate_hint')}>
            {(id) => (
              <Input
                id={id}
                type="number"
                min={-1000}
                placeholder="-1000"
                value={inherit(settings.opus_bitrate)}
                onChange={(e) => onChange({ opus_bitrate: numberOrNull(e.target.value) })}
              />
            )}
          </Field>
        )}
      </Group>

      <Group title={t('fishStory.group_storage')}>
        <Field label={t('fishStory.output_folder')} hint={t('fishStory.output_folder_hint')}>
          {(id) => (
            <Input
              id={id}
              className="font-mono text-xs"
              placeholder={defaults?.output_dir_default ?? ''}
              value={folder || defaults?.output_dir || ''}
              onChange={(e) => {
                setFolder(e.target.value);
                setNote('');
              }}
            />
          )}
        </Field>
        <p className="break-all text-[11px] text-muted-foreground">
          {t('fishStory.output_folder_effective', { path: defaults?.output_dir_effective ?? '' })}
        </p>
        <div className="flex flex-wrap gap-2">
          <Button
            size="xs"
            variant="outline"
            disabled={!folder.trim() || folder.trim() === (defaults?.output_dir ?? '')}
            onClick={() =>
              void onSaveDefaults({ output_dir: folder.trim() }).then((ok) => {
                if (ok) {
                  setFolder('');
                  setNote(t('fishStory.output_folder_saved'));
                }
              })
            }
          >
            {t('fishStory.output_folder_save')}
          </Button>
          {defaults?.output_dir && (
            <Button
              size="xs"
              variant="ghost"
              onClick={() => void onSaveDefaults({ output_dir: null }).then(() => setFolder(''))}
            >
              {t('fishStory.output_folder_reset')}
            </Button>
          )}
          <Button
            size="xs"
            variant="ghost"
            onClick={() =>
              void getBridge()
                ?.files.revealPath(defaults?.output_dir_effective ?? '')
                .catch((reason) => onError(String(reason?.message ?? reason)))
            }
          >
            {t('fishStory.reveal')}
          </Button>
        </div>
        <div className="space-y-1">
          <Button
            size="sm"
            variant="outline"
            onClick={() => {
              const keys = [
                'model', 'format', 'sample_rate', 'mp3_bitrate', 'latency', 'normalize',
                'normalize_loudness', 'max_concurrency', 'default_voice',
              ] as const;
              // A blank tuning field means "send nothing", so saving it clears any stored value.
              const tuning = [
                'temperature', 'top_p', 'chunk_length', 'volume', 'language', 'repetition_penalty',
                'max_new_tokens', 'min_chunk_length', 'condition_on_previous_chunks',
                'early_stop_threshold', 'opus_bitrate',
              ] as const;
              const patch = {
                ...Object.fromEntries(
                  keys.flatMap((k) => (settings[k] === null || settings[k] === undefined ? [] : [[k, settings[k]]])),
                ),
                ...Object.fromEntries(tuning.map((k) => [k, settings[k] ?? null])),
              };
              void onSaveDefaults(patch).then((ok) => ok && setNote(t('fishStory.defaults_saved')));
            }}
          >
            {t('fishStory.save_defaults')}
          </Button>
          <p className="text-[11px] leading-snug text-muted-foreground">{t('fishStory.save_defaults_hint')}</p>
        </div>
        {note && (
          <p role="status" className="text-xs text-emerald-500">
            {note}
          </p>
        )}
      </Group>

      <Group title={t('fishStory.group_requests')}>
        <Field
          label={t('fishStory.multi_speaker')}
          hint={t(settings.request_mode === 'multi' ? 'fishStory.multi_speaker_on' : 'fishStory.multi_speaker_off')}
          inline
        >
          {(id) => (
            <Switch
              id={id}
              checked={settings.request_mode === 'multi'}
              onCheckedChange={(on) => onChange({ request_mode: on ? 'multi' : 'single' })}
            />
          )}
        </Field>
        <Field label={t('fishStory.minimal_request')} hint={t('fishStory.minimal_request_hint')} inline>
          {(id) => (
            <Switch
              id={id}
              checked={settings.minimal_request === true}
              onCheckedChange={(minimal_request) => onChange({ minimal_request })}
            />
          )}
        </Field>
        <Field label={t('fishStory.concurrency')} hint={t('fishStory.concurrency_hint')}>
          {(id) => (
            <Input
              id={id}
              type="number"
              min={1}
              max={8}
              step={1}
              placeholder={inherit(defaults?.max_concurrency)}
              value={inherit(settings.max_concurrency)}
              onChange={(e) => {
                const n = numberOrNull(e.target.value);
                onChange({ max_concurrency: n === null ? null : Math.max(1, Math.min(8, Math.round(n))) });
              }}
            />
          )}
        </Field>
      </Group>
    </div>
  );
}
