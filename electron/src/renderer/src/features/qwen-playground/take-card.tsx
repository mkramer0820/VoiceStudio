import { useEffect, useId, useState, type ReactNode } from 'react';
import { useTranslation } from 'react-i18next';
import {
  AlertTriangleIcon,
  ArrowDownIcon,
  ArrowUpIcon,
  ChevronDownIcon,
  ChevronRightIcon,
  CopyIcon,
  DicesIcon,
  Trash2Icon,
  UserPlusIcon,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import {
  KIND_ORDER,
  MAX_REPEAT,
  randomSeed,
  recipeOf,
  retarget,
  speakerGroups,
  summaryParts,
  validateTake,
  type Issue,
  type ModeCode,
  type QwenModel,
  modelOptionLabel,
  type QwenProfile,
  type TakeDraft,
  type TakeParams,
  type VoiceKind,
} from './qwen-model';

const selectClass =
  'h-8 min-w-0 rounded-md border border-input bg-input/20 px-2 text-sm outline-none focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/30 dark:bg-input/30';

function Field({ label, hint, children, className }: { label: string; hint?: string; children: (id: string) => ReactNode; className?: string }) {
  const id = useId();
  return (
    <div className={`space-y-1 ${className ?? ''}`}>
      <label htmlFor={id} className="block text-xs text-muted-foreground" title={hint}>
        {label}
      </label>
      {children(id)}
    </div>
  );
}

/** A number box that keeps what you type; an empty box means "engine default" (null). */
function NumberField({
  id,
  value,
  onChange,
  step,
  min,
  max,
  integer,
  placeholder,
}: {
  id: string;
  value: number | null;
  onChange: (value: number | null) => void;
  step?: number;
  min?: number;
  max?: number;
  integer?: boolean;
  placeholder?: string;
}) {
  const [text, setText] = useState(value == null ? '' : String(value));
  useEffect(() => {
    setText((cur) => (Number(cur) === value || (cur === '' && value == null) ? cur : value == null ? '' : String(value)));
  }, [value]);
  return (
    <Input
      id={id}
      type="number"
      inputMode="decimal"
      className="h-8"
      step={step}
      min={min}
      max={max}
      placeholder={placeholder}
      value={text}
      onChange={(e) => {
        setText(e.target.value);
        const n = e.target.value.trim() === '' ? null : Number(e.target.value);
        onChange(n == null || !Number.isFinite(n) ? null : integer ? Math.round(n) : n);
      }}
    />
  );
}

export function TakeCard({
  take,
  index,
  total,
  models,
  profiles,
  selected,
  onSelect,
  onChange,
  onDuplicate,
  onRemove,
  onMove,
  onAddVoice,
  children,
}: {
  take: TakeDraft;
  index: number;
  total: number;
  models: QwenModel[];
  profiles: QwenProfile[];
  selected: boolean;
  onSelect: (on: boolean) => void;
  onChange: (change: (take: TakeDraft) => TakeDraft) => void;
  onDuplicate: () => void;
  onRemove: () => void;
  onMove: (by: -1 | 1) => void;
  onAddVoice: () => void;
  children?: ReactNode;
}) {
  const { t } = useTranslation();
  const [advanced, setAdvanced] = useState(false);
  const model = models.find((m) => m.id === take.modelId);
  const issues = validateTake(take, model, profiles);
  const set = (patch: Partial<TakeDraft>) => onChange((cur) => ({ ...cur, ...patch }));
  const setParam = <K extends keyof TakeParams>(key: K, value: TakeParams[K]) =>
    onChange((cur) => ({ ...cur, params: { ...cur.params, [key]: value } }));
  const groups = speakerGroups(model);
  const profile = profiles.find((p) => p.id === take.profileId);

  const recipeLabel = {
    narrator: t('qwenPlayground.recipe_narrator'),
    clone: t('qwenPlayground.recipe_clone'),
    experiment: t('qwenPlayground.recipe_experiment'),
  }[recipeOf(take, model)];
  const kindLabel = (kind: VoiceKind) =>
    kind === 'custom_voice'
      ? t('qwenPlayground.kind_custom_voice')
      : kind === 'clone'
        ? t('qwenPlayground.kind_clone')
        : t('qwenPlayground.kind_clone_on_custom');
  const kindHint = (kind: VoiceKind) =>
    kind === 'custom_voice'
      ? t('qwenPlayground.kind_custom_voice_hint')
      : kind === 'clone'
        ? t('qwenPlayground.kind_clone_hint')
        : t('qwenPlayground.kind_clone_on_custom_hint');
  const modeLabel = (mode: ModeCode) => {
    switch (mode) {
      case 'baked_speaker':
        return t('qwenPlayground.mode_baked_speaker');
      case 'baked_speaker_instruct':
        return t('qwenPlayground.mode_baked_speaker_instruct');
      case 'borrowed_speaker':
        return t('qwenPlayground.mode_borrowed_speaker');
      case 'clip_and_text':
        return t('qwenPlayground.mode_clip_and_text');
      case 'embedding_only':
        return t('qwenPlayground.mode_embedding_only');
      case 'clone_on_custom':
        return t('qwenPlayground.mode_clone_on_custom');
    }
  };
  const issueText = (issue: Issue) => {
    const v = issue.values ?? {};
    switch (issue.code) {
      case 'no_model':
        return v.id ? t('modelLocations.model_not_found', { label: v.id, path: '' }) : t('qwenPlayground.issue_no_model');
      case 'no_text':
        return t('qwenPlayground.issue_no_text');
      case 'pair_mismatch':
        return t('qwenPlayground.issue_pair_mismatch', v);
      case 'pick_speaker':
        return t('qwenPlayground.issue_pick_speaker', v);
      case 'no_clip':
        return t('qwenPlayground.issue_no_clip');
      case 'no_ref_text':
        return t('qwenPlayground.issue_no_ref_text');
      case 'borrowed':
        return t('qwenPlayground.issue_borrowed', v);
    }
  };

  const parts = summaryParts(take, model, profiles);
  const summary = t('qwenPlayground.summary', {
    model: parts.modelLabel || t('qwenPlayground.none'),
    type: parts.modelType || t('qwenPlayground.none'),
    voice: parts.voice || t('qwenPlayground.none'),
    mode: modeLabel(parts.mode),
  });

  const pickProfile = (id: string) => {
    const next = profiles.find((p) => p.id === id);
    set({ profileId: id, refText: next?.ref_text ?? '' });
  };

  return (
    <article
      aria-label={t('qwenPlayground.take_n', { n: index + 1 })}
      className="space-y-2 rounded-xl border border-border/60 bg-card/30 p-3"
    >
      <header className="flex flex-wrap items-center gap-2">
        <input
          type="checkbox"
          aria-label={t('qwenPlayground.select_take')}
          checked={selected}
          onChange={(e) => onSelect(e.target.checked)}
        />
        <Button
          size="icon-xs"
          variant="ghost"
          aria-expanded={!take.collapsed}
          aria-label={take.collapsed ? t('qwenPlayground.expand') : t('qwenPlayground.collapse')}
          onClick={() => set({ collapsed: !take.collapsed })}
        >
          {take.collapsed ? <ChevronRightIcon /> : <ChevronDownIcon />}
        </Button>
        <span className="text-sm font-medium tabular-nums">{t('qwenPlayground.take_n', { n: index + 1 })}</span>
        <span className="rounded-full bg-primary/15 px-2 py-0.5 text-[11px]" data-testid="recipe">
          {recipeLabel}
        </span>
        <Input
          aria-label={t('qwenPlayground.label')}
          placeholder={t('qwenPlayground.label_placeholder')}
          className="h-7 w-48 max-w-[40vw]"
          value={take.label}
          maxLength={80}
          onChange={(e) => set({ label: e.target.value })}
        />
        <div className="ml-auto flex items-center gap-0.5">
          <Button size="icon-xs" variant="ghost" disabled={index === 0} aria-label={t('qwenPlayground.move_up')} onClick={() => onMove(-1)}>
            <ArrowUpIcon />
          </Button>
          <Button size="icon-xs" variant="ghost" disabled={index === total - 1} aria-label={t('qwenPlayground.move_down')} onClick={() => onMove(1)}>
            <ArrowDownIcon />
          </Button>
          <Button size="icon-xs" variant="ghost" aria-label={t('qwenPlayground.duplicate')} onClick={onDuplicate}>
            <CopyIcon />
          </Button>
          <Button size="icon-xs" variant="ghost" aria-label={t('qwenPlayground.delete')} onClick={onRemove}>
            <Trash2Icon />
          </Button>
        </div>
      </header>
      <p className="text-xs text-muted-foreground" data-testid="summary">
        {summary}
      </p>

      {!take.collapsed && (
        <div className="space-y-3">
          <Field label={t('qwenPlayground.prompt')}>
            {(id) => (
              <textarea
                id={id}
                rows={3}
                value={take.text}
                placeholder={t('qwenPlayground.prompt_placeholder')}
                className="w-full resize-y rounded-md border border-input bg-input/20 p-2 text-sm"
                onChange={(e) => set({ text: e.target.value })}
              />
            )}
          </Field>

          <div className="grid gap-3 sm:grid-cols-2">
            <Field label={t('qwenPlayground.model')}>
              {(id) => (
                <select
                  id={id}
                  className={`${selectClass} w-full`}
                  value={take.modelId}
                  onChange={(e) => onChange((cur) => retarget(cur, models, { modelId: e.target.value }))}
                >
                  {!model && <option value={take.modelId}>{take.modelId || t('qwenPlayground.pick_model')}</option>}
                  {models.map((m) => (
                    <option key={m.id} value={m.id}>
                      {modelOptionLabel(m, { huggingface: t('modelLocations.loc_hf'), environment: t('modelLocations.loc_env') })}
                    </option>
                  ))}
                </select>
              )}
            </Field>
            <Field label={t('qwenPlayground.voice_kind')}>
              {(id) => (
                <select
                  id={id}
                  className={`${selectClass} w-full`}
                  value={take.kind}
                  title={kindHint(take.kind)}
                  onChange={(e) => onChange((cur) => retarget(cur, models, { kind: e.target.value as VoiceKind }))}
                >
                  {KIND_ORDER.map((kind) => (
                    <option key={kind} value={kind} title={kindHint(kind)}>
                      {kindLabel(kind)}
                    </option>
                  ))}
                </select>
              )}
            </Field>
          </div>

          {take.kind !== 'clone' && (
            <div className="grid gap-3 sm:grid-cols-2">
              <Field label={t('qwenPlayground.speaker')}>
                {(id) => (
                  <select
                    id={id}
                    className={`${selectClass} w-full`}
                    value={take.speaker}
                    onChange={(e) => set({ speaker: e.target.value })}
                  >
                    {take.speaker === '' && <option value="">{t('qwenPlayground.speaker_choose')}</option>}
                    {take.speaker !== '' && ![...groups.own, ...groups.borrowed].includes(take.speaker) && (
                      <option value={take.speaker}>{take.speaker}</option>
                    )}
                    {groups.own.length > 0 && (
                      <optgroup label={t('qwenPlayground.speaker_own')}>
                        {groups.own.map((s) => (
                          <option key={s} value={s}>
                            {s}
                          </option>
                        ))}
                      </optgroup>
                    )}
                    {groups.borrowed.length > 0 && (
                      <optgroup label={t('qwenPlayground.speaker_borrowed')}>
                        {groups.borrowed.map((s) => (
                          <option key={s} value={s}>
                            {s}
                          </option>
                        ))}
                      </optgroup>
                    )}
                  </select>
                )}
              </Field>
              <Field label={t('qwenPlayground.instruct')} hint={t('qwenPlayground.instruct_hint')}>
                {(id) => (
                  <Input
                    id={id}
                    className="h-8"
                    value={take.instruct}
                    placeholder={t('qwenPlayground.instruct_placeholder')}
                    onChange={(e) => set({ instruct: e.target.value })}
                  />
                )}
              </Field>
            </div>
          )}

          {take.kind !== 'custom_voice' && (
            <div className="space-y-2 rounded-lg bg-background/40 p-2">
              {take.kind === 'clone_on_custom' && (
                <p className="text-xs text-amber-500">{t('qwenPlayground.experimental_clone_on_custom')}</p>
              )}
              <div className="grid gap-3 sm:grid-cols-2">
                <Field label={t('qwenPlayground.voice_profile')}>
                  {(id) => (
                    <div className="flex gap-1">
                      <select
                        id={id}
                        className={`${selectClass} min-w-0 flex-1`}
                        value={take.profileId}
                        onChange={(e) => pickProfile(e.target.value)}
                      >
                        <option value="">{t('qwenPlayground.profile_none')}</option>
                        {profiles.map((p) => (
                          <option key={p.id} value={p.id}>
                            {p.name}
                          </option>
                        ))}
                      </select>
                      <Button size="icon-sm" variant="outline" aria-label={t('qwenPlayground.add_voice')} title={t('qwenPlayground.add_voice')} onClick={onAddVoice}>
                        <UserPlusIcon />
                      </Button>
                    </div>
                  )}
                </Field>
                {!take.profileId && (
                  <Field label={t('qwenPlayground.ref_audio')}>
                    {(id) => (
                      <Input
                        id={id}
                        className="h-8 font-mono text-xs"
                        value={take.refAudio}
                        placeholder={t('qwenPlayground.ref_audio_placeholder')}
                        onChange={(e) => set({ refAudio: e.target.value })}
                      />
                    )}
                  </Field>
                )}
              </div>
              <Field label={t('qwenPlayground.ref_text')} hint={profile ? t('qwenPlayground.ref_text_profile_hint') : undefined}>
                {(id) => (
                  <textarea
                    id={id}
                    rows={2}
                    value={take.refText}
                    placeholder={t('qwenPlayground.ref_text_placeholder')}
                    className="w-full resize-y rounded-md border border-input bg-input/20 p-2 text-sm"
                    onChange={(e) => set({ refText: e.target.value })}
                  />
                )}
              </Field>
              {take.kind === 'clone' && (
                <label className="flex items-center gap-2 text-xs" title={t('qwenPlayground.xvec_hint')}>
                  <input type="checkbox" checked={take.xvecOnly} onChange={(e) => set({ xvecOnly: e.target.checked })} />
                  {t('qwenPlayground.xvec_only')}
                </label>
              )}
            </div>
          )}

          <div className="flex flex-wrap items-end gap-3">
            <Field label={t('qwenPlayground.repeat')} className="w-24">
              {(id) => (
                <Input
                  id={id}
                  type="number"
                  className="h-8"
                  min={1}
                  max={MAX_REPEAT}
                  value={take.repeat}
                  onChange={(e) => set({ repeat: Math.min(MAX_REPEAT, Math.max(1, Math.round(Number(e.target.value)) || 1)) })}
                />
              )}
            </Field>
            <label className="flex h-8 items-center gap-2 text-xs">
              <input type="checkbox" checked={take.randomSeed} onChange={(e) => set({ randomSeed: e.target.checked })} />
              {t('qwenPlayground.seed_random')}
            </label>
            {!take.randomSeed && (
              <Field label={t('qwenPlayground.seed')} hint={t('qwenPlayground.seed_hint')} className="w-40">
                {(id) => (
                  <div className="flex gap-1">
                    <Input
                      id={id}
                      type="number"
                      className="h-8"
                      min={0}
                      value={take.seed}
                      onChange={(e) => set({ seed: Math.max(0, Math.round(Number(e.target.value)) || 0) })}
                    />
                    <Button size="icon-sm" variant="outline" aria-label={t('qwenPlayground.seed_dice')} onClick={() => set({ seed: randomSeed() })}>
                      <DicesIcon />
                    </Button>
                  </div>
                )}
              </Field>
            )}
            <Button size="sm" variant="ghost" aria-expanded={advanced} onClick={() => setAdvanced((v) => !v)}>
              {advanced ? <ChevronDownIcon /> : <ChevronRightIcon />} {t('qwenPlayground.advanced')}
            </Button>
          </div>

          {advanced && (
            <div className="grid gap-3 rounded-lg bg-background/40 p-2 sm:grid-cols-3">
              <Field label={t('qwenPlayground.language')}>
                {(id) => (
                  <Input id={id} className="h-8" value={take.params.language} onChange={(e) => setParam('language', e.target.value)} />
                )}
              </Field>
              <Field label={t('qwenPlayground.temperature')}>
                {(id) => <NumberField id={id} step={0.05} min={0.05} max={2} value={take.params.temperature} onChange={(v) => setParam('temperature', v)} placeholder={t('qwenPlayground.engine_default')} />}
              </Field>
              <Field label={t('qwenPlayground.top_k')}>
                {(id) => <NumberField id={id} integer step={1} min={0} max={500} value={take.params.top_k} onChange={(v) => setParam('top_k', v)} placeholder={t('qwenPlayground.engine_default')} />}
              </Field>
              <Field label={t('qwenPlayground.top_p')}>
                {(id) => <NumberField id={id} step={0.05} min={0.05} max={1} value={take.params.top_p} onChange={(v) => setParam('top_p', v)} placeholder={t('qwenPlayground.engine_default')} />}
              </Field>
              <Field label={t('qwenPlayground.repetition_penalty')}>
                {(id) => <NumberField id={id} step={0.05} min={0.5} max={2} value={take.params.repetition_penalty} onChange={(v) => setParam('repetition_penalty', v)} placeholder={t('qwenPlayground.engine_default')} />}
              </Field>
              <Field label={t('qwenPlayground.pad_s')} hint={t('qwenPlayground.pad_s_hint')}>
                {(id) => <NumberField id={id} step={0.1} min={0} max={5} value={take.params.pad_s} onChange={(v) => setParam('pad_s', v ?? 0.3)} />}
              </Field>
              <Field label={t('qwenPlayground.chunk_chars')} hint={t('qwenPlayground.chunk_chars_hint')}>
                {(id) => <NumberField id={id} integer step={10} min={20} max={2000} value={take.params.chunk_chars} onChange={(v) => setParam('chunk_chars', v)} placeholder={t('qwenPlayground.engine_default')} />}
              </Field>
              <Field label={t('qwenPlayground.max_seq_len')} hint={t('qwenPlayground.max_seq_len_hint')}>
                {(id) => <NumberField id={id} integer step={128} min={512} max={4096} value={take.params.max_seq_len} onChange={(v) => setParam('max_seq_len', v)} placeholder={t('qwenPlayground.engine_default')} />}
              </Field>
            </div>
          )}
        </div>
      )}

      {issues.length > 0 && (
        <ul className="space-y-1">
          {issues.map((issue) => (
            <li key={issue.code} className="flex items-start gap-1 text-xs text-amber-500">
              <AlertTriangleIcon className="mt-0.5 size-3.5 shrink-0" />
              {issueText(issue)}
            </li>
          ))}
        </ul>
      )}
      {children}
    </article>
  );
}
