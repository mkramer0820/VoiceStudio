/** Pure helpers for the export dialog: the tag dictionary and the training settings. */

// ---- tags ------------------------------------------------------------------------------------

export type TagAction = { mode: 'remove' | 'keep' | 'convert'; to: string };
export type TagActions = Record<string, TagAction>;

const BARE = /^[\w\- ]{1,40}$/;

export const actionOf = (actions: TagActions, tag: string): TagAction =>
  actions[tag] ?? { mode: 'remove', to: '' };

/** What a Convert row writes: `sad` or `[sad]` mean the tag [sad]; anything else is taken literally. */
export function replacementOf(raw: string): string {
  let value = raw.trim();
  if (value.length > 2 && value.startsWith('[') && value.endsWith(']') && !value.slice(1, -1).includes('[')) {
    value = value.slice(1, -1).trim();
  }
  return BARE.test(value) ? value.toLowerCase() : value;
}

/** How a written value appears in the text: bare words in square brackets, anything else as typed. */
export const renderTag = (value: string) => (BARE.test(value) ? `[${value}]` : value);

/** What each kept tag will be written as, or null when the row is removed (an empty Convert is a Remove). */
export function writtenAs(tag: string, action: TagAction): string | null {
  if (action.mode === 'keep') return tag;
  if (action.mode === 'convert') return action.to.trim() ? replacementOf(action.to) : null;
  return null;
}

/** The request's `keep_tags`: one entry per tag that is not removed. */
export function keepTagsOf(tags: string[], actions: TagActions): string[] {
  const out: string[] = [];
  for (const tag of tags) {
    const action = actionOf(actions, tag);
    if (writtenAs(tag, action) === null) continue;
    out.push(action.mode === 'keep' ? tag : `[${tag}]:${action.to.trim()}`);
  }
  return out;
}

/** Fill the rows from a pasted `[frustrated]:(sad)` / `laughing=laughter` dictionary; unknown tags are ignored. */
export function parseDictionary(text: string, tags: string[]): TagActions {
  const known = new Set(tags);
  const out: TagActions = {};
  for (const part of text.split(/[,;\n]+/)) {
    const match = /^\[?\s*([^[\]:=]+?)\s*\]?\s*(?:[:=]\s*(.*))?$/.exec(part.trim());
    if (!match) continue;
    const source = match[1].trim().toLowerCase();
    if (!known.has(source)) continue;
    const target = (match[2] ?? '').trim();
    out[source] = target && replacementOf(target) !== source ? { mode: 'convert', to: target } : { mode: 'keep', to: '' };
  }
  return out;
}

/** Tags in the export that OmniVoice does not know (rendered the way the backend reports them). */
export function unknownTags(tags: string[], actions: TagActions, known: string[]): string[] {
  if (known.length === 0) return [];
  const out: string[] = [];
  for (const tag of tags) {
    const written = writtenAs(tag, actionOf(actions, tag));
    if (written !== null && !known.includes(written)) out.push(tag);
  }
  return out;
}

// ---- training --------------------------------------------------------------------------------

/** The backend accepts at most this many dictionary entries. */
export const MAX_KEPT_TAGS = 100;

export const FRAMES_PER_SECOND = 25;
export const TEXT_OVERHEAD = 1.15;
export const STEP_RANGE = [10, 200000] as const;

const tokensPerPass = (seconds: number) => Math.max(1, seconds * FRAMES_PER_SECOND * TEXT_OVERHEAD);

/** Same formula as the backend's `suggest_steps`. */
export function suggestSteps(seconds: number, batchTokens: number, passes: number): number {
  const raw = Math.ceil((passes * tokensPerPass(seconds)) / batchTokens);
  return Math.min(STEP_RANGE[1], Math.max(STEP_RANGE[0], raw));
}

/** Same formula as the backend's `estimate`: how often the training clips are seen. */
export function estimatePasses(seconds: number, steps: number, batchTokens: number): number {
  return Math.round(((steps * batchTokens) / tokensPerPass(seconds)) * 10) / 10;
}

export type TrainingMode = 'lora' | 'full';
export type TrainingKey =
  | 'steps' | 'learning_rate' | 'batch_tokens' | 'gradient_accumulation_steps' | 'lora_r'
  | 'lora_alpha' | 'lora_dropout' | 'warmup_ratio' | 'weight_decay' | 'seed' | 'num_workers'
  | 'save_every' | 'eval_every' | 'keep_last_n_checkpoints';
export type TrainingChoice = 'mixed_precision' | 'lr_scheduler_type';

/** Defaults and ranges mirror `TrainOptions` in backend/engines/fish_api/training.py. */
export const NUMBERS: Record<TrainingKey, { def: number; min: number; max: number; step: string; lora?: boolean; advanced?: boolean }> = {
  steps: { def: 2000, min: 10, max: 200000, step: '1' },
  learning_rate: { def: 1e-4, min: 1e-6, max: 1e-2, step: 'any' },
  batch_tokens: { def: 4096, min: 512, max: 65536, step: '1' },
  gradient_accumulation_steps: { def: 1, min: 1, max: 64, step: '1' },
  lora_r: { def: 16, min: 1, max: 256, step: '1', lora: true },
  lora_alpha: { def: 32, min: 1, max: 512, step: 'any', lora: true },
  lora_dropout: { def: 0.05, min: 0, max: 0.5, step: 'any', lora: true },
  warmup_ratio: { def: 0.05, min: 0, max: 0.5, step: 'any', advanced: true },
  weight_decay: { def: 0.01, min: 0, max: 1, step: 'any', advanced: true },
  seed: { def: 42, min: 0, max: 2 ** 31 - 1, step: '1', advanced: true },
  num_workers: { def: 2, min: 0, max: 32, step: '1', advanced: true },
  save_every: { def: 0, min: 0, max: 200000, step: '1', advanced: true },
  eval_every: { def: 0, min: 0, max: 200000, step: '1', advanced: true },
  keep_last_n_checkpoints: { def: 2, min: -1, max: 100, step: '1', advanced: true },
};
export const CHOICES: Record<TrainingChoice, { def: string; options: string[] }> = {
  mixed_precision: { def: 'bf16', options: ['bf16', 'fp16', 'no'] },
  lr_scheduler_type: { def: 'cosine', options: ['cosine', 'linear', 'constant'] },
};
export const DEFAULT_CHECKPOINT = 'k2-fsa/OmniVoice';
export const FULL_LEARNING_RATE = 5e-5;

export interface TrainingForm {
  mode: TrainingMode;
  heads: boolean;
  /** Fields the user typed into, as typed. An untouched field follows the default for the mode. */
  edits: Partial<Record<TrainingKey | TrainingChoice | 'base_checkpoint', string>>;
}
export const EMPTY_TRAINING: TrainingForm = { mode: 'lora', heads: false, edits: {} };

export function defaultFor(key: TrainingKey, mode: TrainingMode): number {
  return key === 'learning_rate' && mode === 'full' ? FULL_LEARNING_RATE : NUMBERS[key].def;
}

export const numberOf = (form: TrainingForm, key: TrainingKey): number => {
  const raw = form.edits[key];
  return raw === undefined ? defaultFor(key, form.mode) : raw.trim() === '' ? NaN : Number(raw);
};

export const visibleKeys = (form: TrainingForm) =>
  (Object.keys(NUMBERS) as TrainingKey[]).filter((key) => form.mode === 'lora' || !NUMBERS[key].lora);

export interface NumberSpec {
  def: number;
  min: number;
  max: number;
  step: string;
}
/** A value the backend would refuse. Shared by every Training setup field. */
export const outOfRange = (spec: NumberSpec, value: number) => !Number.isFinite(value) || value < spec.min || value > spec.max;

/** Fields with a value the backend would refuse. */
export function invalidKeys(form: TrainingForm): TrainingKey[] {
  return visibleKeys(form).filter((key) => outOfRange(NUMBERS[key], numberOf(form, key)));
}

/** Only what differs from the backend defaults, so the backend's own defaults apply to the rest. */
export function trainingBody(form: TrainingForm): Record<string, unknown> {
  const body: Record<string, unknown> = {};
  if (form.mode === 'full') body.mode = 'full';
  for (const key of visibleKeys(form)) {
    if (form.edits[key] === undefined) continue;
    const value = numberOf(form, key);
    if (value !== defaultFor(key, form.mode)) body[key] = value;
  }
  for (const key of Object.keys(CHOICES) as TrainingChoice[]) {
    const value = form.edits[key];
    if (value !== undefined && value !== CHOICES[key].def) body[key] = value;
  }
  const checkpoint = form.edits.base_checkpoint?.trim();
  if (checkpoint && checkpoint !== DEFAULT_CHECKPOINT) body.base_checkpoint = checkpoint;
  if (form.mode === 'lora' && form.heads) body.train_heads = true;
  return body;
}

// ---- Qwen3-TTS -------------------------------------------------------------------------------

export const QWEN_FORMAT = 'qwen3_tts';
export const QWEN_SPEAKER_RE = /^[A-Za-z0-9_-]{0,40}$/;
const MODEL_ID = /^[A-Za-z0-9][A-Za-z0-9._-]*\/[A-Za-z0-9][A-Za-z0-9._-]*$/;
const DRIVE_FOLDER = /^[A-Za-z0-9._-]+(\/[A-Za-z0-9._-]+)*$/;

/** Reference clip, ZIP and duration limits (the Training setup fields are below). */
export interface QwenForm {
  speaker: string;
  zip: boolean;
  minDuration: string;
  maxDuration: string;
}
export const EMPTY_QWEN: QwenForm = { speaker: '', zip: true, minDuration: '0.8', maxDuration: '30' };

export function qwenInvalid(form: QwenForm): Array<keyof QwenForm> {
  const out: Array<keyof QwenForm> = [];
  const lo = Number(form.minDuration);
  const hi = Number(form.maxDuration);
  if (!QWEN_SPEAKER_RE.test(form.speaker.trim())) out.push('speaker');
  if (form.minDuration.trim() === '' || !Number.isFinite(lo) || lo < 0.1) out.push('minDuration');
  if (form.maxDuration.trim() === '' || !Number.isFinite(hi) || hi > 120 || (Number.isFinite(lo) && hi <= lo)) {
    out.push('maxDuration');
  }
  return out;
}

/** The request's `qwen3` object; a null `referenceKey` lets the backend choose the reference clip. */
export function qwenBody(form: QwenForm, referenceKey: string | null): Record<string, unknown> {
  return {
    speaker_name: form.speaker.trim(),
    reference_key: referenceKey,
    make_zip: form.zip,
    min_duration: Number(form.minDuration),
    max_duration: Number(form.maxDuration),
  };
}

/** The numbers sft_12hz.py really takes; defaults are the script's own. Everything else in it is fixed. */
export type QwenNumberKey = 'epochs' | 'learning_rate' | 'batch_size';
export type QwenTextKey = 'base_model' | 'tokenizer_model' | 'drive_folder';
export const QWEN_NUMBERS: Record<QwenNumberKey, NumberSpec & { advanced?: boolean }> = {
  epochs: { def: 3, min: 1, max: 100, step: '1' },
  learning_rate: { def: 2e-6, min: 1e-7, max: 1e-3, step: 'any' },
  batch_size: { def: 2, min: 1, max: 256, step: '1' },
};
export const QWEN_TEXT: Record<QwenTextKey, { def: string; test: RegExp; advanced?: boolean }> = {
  drive_folder: { def: 'Qwen3TTS', test: DRIVE_FOLDER },
  base_model: { def: 'Qwen/Qwen3-TTS-12Hz-1.7B-Base', test: MODEL_ID },
  tokenizer_model: { def: 'Qwen/Qwen3-TTS-Tokenizer-12Hz', test: MODEL_ID, advanced: true },
};
/** sft_12hz.py hard-codes Accelerator(gradient_accumulation_steps=4). */
export const QWEN_ACCUMULATION = 4;

export interface QwenTrainForm {
  /** Fields the user typed into, as typed. An untouched field follows the script's default. */
  edits: Partial<Record<QwenNumberKey | QwenTextKey, string>>;
}
export const EMPTY_QWEN_TRAIN: QwenTrainForm = { edits: {} };

export const qwenNumber = (form: QwenTrainForm, key: QwenNumberKey): number => {
  const raw = form.edits[key];
  return raw === undefined ? QWEN_NUMBERS[key].def : raw.trim() === '' ? NaN : Number(raw);
};
export const qwenText = (form: QwenTrainForm, key: QwenTextKey): string => form.edits[key] ?? QWEN_TEXT[key].def;

export function qwenTrainInvalid(form: QwenTrainForm): Array<QwenNumberKey | QwenTextKey> {
  const bad: Array<QwenNumberKey | QwenTextKey> = [];
  for (const key of Object.keys(QWEN_NUMBERS) as QwenNumberKey[]) {
    const value = qwenNumber(form, key);
    const whole = key === 'learning_rate' || Number.isInteger(value);
    if (outOfRange(QWEN_NUMBERS[key], value) || !whole) bad.push(key);
  }
  for (const key of Object.keys(QWEN_TEXT) as QwenTextKey[]) {
    const value = qwenText(form, key).trim();
    if (!QWEN_TEXT[key].test.test(value) || value.split('/').some((part) => part === '.' || part === '..')) bad.push(key);
  }
  return bad;
}

/** Only what differs from the script's defaults, so the backend's defaults apply to the rest. */
export function qwenTrainBody(form: QwenTrainForm): Record<string, unknown> {
  const body: Record<string, unknown> = {};
  for (const key of Object.keys(QWEN_NUMBERS) as QwenNumberKey[]) {
    if (form.edits[key] !== undefined && qwenNumber(form, key) !== QWEN_NUMBERS[key].def) body[key] = qwenNumber(form, key);
  }
  for (const key of Object.keys(QWEN_TEXT) as QwenTextKey[]) {
    const value = form.edits[key]?.trim();
    if (value !== undefined && value !== QWEN_TEXT[key].def) body[key] = value;
  }
  return body;
}

/** What the script's own loop implies for `clips` training clips; an estimate, not a time or quality claim. */
export function qwenEstimate(clips: number, batchSize: number, epochs: number) {
  if (!(clips > 0) || !(batchSize >= 1) || !(epochs >= 1)) return null;
  const batches = Math.ceil(clips / batchSize);
  const updates = Math.ceil(batches / QWEN_ACCUMULATION);
  return { batches, updates, total: updates * epochs };
}

// ---- loudness --------------------------------------------------------------------------------

/** The level the training formats start at (dB, whole-clip RMS including pauses); -19 to -23 is the usual range. */
export const DEFAULT_LEVEL_DB = '-22';
export const LEVEL_PRESETS = [-19, -22, -23] as const;
/** The backend refuses anything outside this range. */
export const LEVEL_RANGE = [-60, -3] as const;
/** Clips within this many dB of the target count as already there. */
export const LEVEL_TOLERANCE_DB = 1;

export function levelInvalid(raw: string): boolean {
  const value = Number(raw);
  return raw.trim() === '' || !Number.isFinite(value) || value < LEVEL_RANGE[0] || value > LEVEL_RANGE[1];
}

/** How many measured clips sit below, within 1 dB of, or above `target`; `counts` is [level, clips] pairs. */
export function levelSplit(counts: Array<[number, number]>, target: number) {
  let below = 0;
  let within = 0;
  let above = 0;
  for (const [level, clips] of counts) {
    if (level < target - LEVEL_TOLERANCE_DB) below += clips;
    else if (level > target + LEVEL_TOLERANCE_DB) above += clips;
    else within += clips;
  }
  return { below, within, above };
}
