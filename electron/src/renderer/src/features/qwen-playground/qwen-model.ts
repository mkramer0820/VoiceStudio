/**
 * Pure logic for the Qwen playground: take defaults, the checks that produce
 * plain warnings, model-load counting (mirrors `plan`/`swaps` in
 * backend/engines/qwen3_studio/playground.py) and the run-request builder.
 * Nothing here touches React or the network.
 */

export type VoiceKind = 'custom_voice' | 'clone' | 'clone_on_custom';
export type ModelType = 'custom_voice' | 'base' | 'voice_design' | 'adapter';

export interface QwenModel {
  id: string;
  label: string;
  path: string;
  source: string;
  /** The model-location row id, or `huggingface_cache` / `environment`. */
  location?: string;
  /** The row's display name, when the model comes from a Settings model location. */
  location_name?: string;
  type: ModelType;
  size: string;
  speakers: string[];
  borrowable_speakers?: string[];
  bytes?: number;
}

export interface QwenProfile {
  id: string;
  name: string;
  ref_text: string;
  language: string;
  kind: string;
}

export interface QwenSettings {
  python: string;
  output_dir: string;
  warmup_runs: number;
  warmup_text: string;
}

export interface QwenState {
  settings: QwenSettings;
  output_dir_default: string;
  python_default?: string;
  engine: { available: boolean; reason: string };
  models: QwenModel[];
  stock_speakers: string[];
  profiles: QwenProfile[];
}

/** Every knob the backend's Params accepts; null means "leave it to the engine". */
export interface TakeParams {
  language: string;
  temperature: number | null;
  top_k: number | null;
  top_p: number | null;
  repetition_penalty: number | null;
  pad_s: number;
  chunk_chars: number | null;
  max_seq_len: number | null;
}

export interface TakeDraft {
  id: string;
  label: string;
  text: string;
  modelId: string;
  kind: VoiceKind;
  speaker: string;
  instruct: string;
  profileId: string;
  refAudio: string;
  refText: string;
  xvecOnly: boolean;
  /** true: no seed is sent, every render gets a new one. false: `seed` is used (+1 per repeat). */
  randomSeed: boolean;
  seed: number;
  params: TakeParams;
  repeat: number;
  /** LoRA adapter models only: 0 = the base model, 0.3 default, above 0.5 may turn noisy. Sent as lora_scale. */
  loraScale: number;
  collapsed: boolean;
}

export const DEFAULT_PARAMS: TakeParams = {
  language: 'Auto',
  temperature: null,
  top_k: null,
  top_p: null,
  repetition_penalty: null,
  pad_s: 0.3,
  chunk_chars: null,
  max_seq_len: null,
};

export const KIND_ORDER: VoiceKind[] = ['custom_voice', 'clone', 'clone_on_custom'];
export const MAX_REPEAT = 8;

/** The type a voice kind needs; a custom voice may also borrow a stock voice on Base. */
export function wantedType(kind: VoiceKind): ModelType {
  return kind === 'clone' ? 'base' : 'custom_voice';
}

/** A one-speaker model (the fine-tuned narrator) pre-fills its speaker; several or none stay blank. */
export function defaultSpeaker(model: QwenModel | undefined): string {
  return model && model.speakers.length === 1 ? model.speakers[0] : '';
}

export function speakerGroups(model: QwenModel | undefined): { own: string[]; borrowed: string[] } {
  if (!model) return { own: [], borrowed: [] };
  const own = model.speakers;
  const lower = new Set(own.map((s) => s.toLowerCase()));
  return { own, borrowed: (model.borrowable_speakers ?? []).filter((s) => !lower.has(s.toLowerCase())) };
}

export function isBorrowed(model: QwenModel | undefined, speaker: string): boolean {
  const name = speaker.trim().toLowerCase();
  if (!model || !name) return false;
  return !model.speakers.some((s) => s.toLowerCase() === name);
}

let counter = 0;
/** Matches the backend id pattern ^[A-Za-z0-9_-]{1,40}$. */
export function newTakeId(): string {
  counter += 1;
  return `t${Date.now().toString(36)}${counter.toString(36)}`;
}

export function randomSeed(): number {
  return Math.floor(Math.random() * 2 ** 31);
}

/** The first custom_voice model (a one-speaker one first), else the first model of any type. */
export function pickDefaultModel(models: QwenModel[], kind: VoiceKind = 'custom_voice'): QwenModel | undefined {
  const want = wantedType(kind);
  const ofType = models.filter((m) => m.type === want);
  return ofType.find((m) => m.speakers.length === 1) ?? ofType[0] ?? models[0];
}

export function newTake(models: QwenModel[], over: Partial<TakeDraft> = {}): TakeDraft {
  const kind = over.kind ?? 'custom_voice';
  const model = models.find((m) => m.id === over.modelId) ?? pickDefaultModel(models, kind);
  return {
    id: newTakeId(),
    label: '',
    text: '',
    modelId: model?.id ?? '',
    kind,
    speaker: kind === 'clone' ? '' : defaultSpeaker(model),
    instruct: '',
    profileId: '',
    refAudio: '',
    refText: '',
    xvecOnly: false,
    randomSeed: true,
    seed: 1234,
    params: { ...DEFAULT_PARAMS },
    repeat: 1,
    loraScale: 0.3,
    collapsed: false,
    ...over,
  };
}

export function duplicateTake(take: TakeDraft): TakeDraft {
  return { ...take, id: newTakeId(), params: { ...take.params }, label: take.label ? `${take.label} copy` : '' };
}

/** Switch model or kind and keep the speaker consistent with the new model. */
export function retarget(take: TakeDraft, models: QwenModel[], change: { modelId?: string; kind?: VoiceKind }): TakeDraft {
  const next = { ...take, ...(change.modelId !== undefined ? { modelId: change.modelId } : {}), ...(change.kind ? { kind: change.kind } : {}) };
  const model = models.find((m) => m.id === next.modelId);
  if (change.kind && change.modelId === undefined) {
    // A new kind moves to a model of the right type when the current one does not fit.
    if (model?.type !== wantedType(next.kind)) {
      const better = pickDefaultModel(models, next.kind);
      if (better && better.type === wantedType(next.kind)) next.modelId = better.id;
    }
  }
  const target = models.find((m) => m.id === next.modelId);
  if (next.kind === 'clone') return next;
  const name = next.speaker.trim().toLowerCase();
  const groups = speakerGroups(target);
  const known = name !== '' && (groups.own.some((x) => x.toLowerCase() === name) || groups.borrowed.includes(name));
  return known ? next : { ...next, speaker: defaultSpeaker(target) };
}

export type RecipeName = 'narrator' | 'clone' | 'experiment';

/** Narrator, Clone or Experiment: the three plain names a take is labelled with. */
export function recipeOf(take: TakeDraft, model: QwenModel | undefined): RecipeName {
  if (take.kind === 'clone') return take.xvecOnly || take.instruct.trim() ? 'experiment' : 'clone';
  if (take.kind === 'custom_voice' && model?.type === 'custom_voice' && !take.instruct.trim() && !isBorrowed(model, take.speaker)) {
    return 'narrator';
  }
  return 'experiment';
}

export type ModeCode =
  | 'baked_speaker'
  | 'baked_speaker_instruct'
  | 'borrowed_speaker'
  | 'clip_and_text'
  | 'embedding_only'
  | 'clone_on_custom';

export interface SummaryParts {
  modelLabel: string;
  modelType: string;
  /** The speaker name, the profile name or the clip file, whichever picks the voice. */
  voice: string;
  mode: ModeCode;
}

const baseName = (path: string) => path.split(/[\\/]/).filter(Boolean).pop() ?? path;

export function summaryParts(take: TakeDraft, model: QwenModel | undefined, profiles: QwenProfile[]): SummaryParts {
  const profile = profiles.find((p) => p.id === take.profileId);
  const clip = profile?.name || (take.refAudio.trim() ? baseName(take.refAudio.trim()) : '');
  let mode: ModeCode;
  let voice: string;
  if (take.kind === 'clone') {
    mode = take.xvecOnly ? 'embedding_only' : 'clip_and_text';
    voice = clip;
  } else if (take.kind === 'clone_on_custom') {
    mode = 'clone_on_custom';
    voice = clip;
  } else {
    mode = isBorrowed(model, take.speaker) && take.speaker.trim()
      ? 'borrowed_speaker'
      : take.instruct.trim()
        ? 'baked_speaker_instruct'
        : 'baked_speaker';
    voice = take.speaker.trim();
  }
  return { modelLabel: model?.label ?? take.modelId, modelType: model?.type ?? '', voice, mode };
}

export type IssueCode =
  | 'no_model'
  | 'no_text'
  | 'pair_mismatch'
  | 'pick_speaker'
  | 'no_clip'
  | 'no_ref_text'
  | 'borrowed';

export interface Issue {
  code: IssueCode;
  /** Blocking issues keep the take out of a run (it could not even be sent); the rest are warnings. */
  blocking: boolean;
  values?: Record<string, string>;
}

/**
 * Plain checks, in the spirit of the backend's: they warn, they do not stop the user trying a
 * combination (the backend answers 422 with its own reason, which the page shows). Only a
 * missing model or text blocks, because there is nothing to send.
 */
export function validateTake(take: TakeDraft, model: QwenModel | undefined, profiles: QwenProfile[]): Issue[] {
  const out: Issue[] = [];
  if (!model) out.push({ code: 'no_model', blocking: true, values: { id: take.modelId } });
  if (!take.text.trim()) out.push({ code: 'no_text', blocking: true });
  if (!model) return out;
  const speakerBorrowed = take.kind === 'custom_voice' && take.speaker.trim() !== '' && isBorrowed(model, take.speaker);
  const want = wantedType(take.kind);
  const runsAs = model.type === 'adapter' ? 'custom_voice' : model.type; // the playground merges an adapter into a custom voice model
  if (runsAs !== want && !(speakerBorrowed && model.type === 'base')) {
    out.push({ code: 'pair_mismatch', blocking: false, values: { kind: take.kind, want, type: model.type, model: model.label } });
  }
  if (take.kind !== 'clone') {
    if (!take.speaker.trim() && model.speakers.length !== 1) {
      out.push({ code: 'pick_speaker', blocking: false, values: { model: model.label } });
    }
    if (speakerBorrowed) out.push({ code: 'borrowed', blocking: false, values: { speaker: take.speaker.trim().toLowerCase() } });
  }
  if (take.kind !== 'custom_voice') {
    const profile = profiles.find((p) => p.id === take.profileId);
    if (!profile && !take.refAudio.trim()) out.push({ code: 'no_clip', blocking: false });
    const text = take.refText.trim() || (profile?.ref_text ?? '').trim();
    if (take.kind === 'clone' && !take.xvecOnly && !text && (profile || take.refAudio.trim())) {
      out.push({ code: 'no_ref_text', blocking: false });
    }
  }
  return out;
}

export const isRunnable = (take: TakeDraft, model: QwenModel | undefined, profiles: QwenProfile[]) =>
  !validateTake(take, model, profiles).some((i) => i.blocking);

// ── model-load counting (mirror of plan() / swaps() in playground.py) ────────

/** (take index, repetition 1-based) in run order. Grouping only reorders. */
export function planOrder(takes: TakeDraft[], group: boolean): Array<[number, number]> {
  const items: Array<[number, number]> = takes.flatMap((t, i) =>
    Array.from({ length: t.repeat }, (_, r): [number, number] => [i, r + 1]),
  );
  if (!group) return items;
  const modelKey = (t: TakeDraft) => `${t.modelId}\u0000${t.params.max_seq_len ?? ''}`;
  const voiceKey = (t: TakeDraft) =>
    [modelKey(t), t.kind, t.speaker, t.profileId, t.refAudio, t.xvecOnly].join('\u0000');
  const models = new Map<string, number>();
  const voices = new Map<string, number>();
  for (const [i] of items) {
    if (!models.has(modelKey(takes[i]))) models.set(modelKey(takes[i]), models.size);
    if (!voices.has(voiceKey(takes[i]))) voices.set(voiceKey(takes[i]), voices.size);
  }
  // Array.prototype.sort is stable, matching Python's sorted().
  return [...items].sort(
    (a, b) =>
      models.get(modelKey(takes[a[0]]))! - models.get(modelKey(takes[b[0]]))! ||
      voices.get(voiceKey(takes[a[0]]))! - voices.get(voiceKey(takes[b[0]]))!,
  );
}

/** How many times the resident model changes in this order (the first load counts). */
export function countLoads(takes: TakeDraft[], order: Array<[number, number]>): number {
  let last: string | null = null;
  let count = 0;
  for (const [i] of order) {
    const key = `${takes[i].modelId}\u0000${takes[i].params.max_seq_len ?? ''}`;
    if (key !== last) {
      count += 1;
      last = key;
    }
  }
  return count;
}

export const loadsFor = (takes: TakeDraft[], group: boolean) => countLoads(takes, planOrder(takes, group));

// ── request builder ─────────────────────────────────────────────────────────

export interface RunSetup {
  sessionName: string;
  groupByModel: boolean;
  warmupRuns: number;
  warmupText: string;
}

const clean = <T extends object>(o: T) =>
  Object.fromEntries(Object.entries(o).filter(([, v]) => v !== null && v !== undefined && v !== '')) as Partial<T>;

export function buildVoice(take: TakeDraft) {
  const instruct = take.instruct.trim();
  if (take.kind === 'custom_voice') {
    return clean({ kind: take.kind, speaker: take.speaker.trim(), instruct });
  }
  const clip = take.profileId
    ? { profile_id: take.profileId }
    : { ref_audio: take.refAudio.trim() };
  const ref_text = take.refText.trim();
  if (take.kind === 'clone') {
    return clean({ kind: take.kind, ...clip, ref_text, xvec_only: take.xvecOnly || undefined, instruct });
  }
  return clean({ kind: take.kind, speaker: take.speaker.trim(), ...clip, ref_text, instruct });
}

export function buildParams(take: TakeDraft) {
  const { language, pad_s, ...rest } = take.params;
  return {
    language: language.trim() || 'Auto',
    pad_s,
    ...clean(rest),
    ...(take.randomSeed ? {} : { seed: Math.max(0, Math.floor(take.seed)) }),
  };
}

export function buildRunRequest(takes: TakeDraft[], setup: RunSetup) {
  return {
    session_name: setup.sessionName.trim() || 'playground',
    group_by_model: setup.groupByModel,
    warmup_runs: setup.warmupRuns,
    warmup_text: setup.warmupText,
    takes: takes.map((t) => ({
      id: t.id,
      label: t.label.trim(),
      text: t.text,
      model_id: t.modelId,
      voice: buildVoice(t),
      params: buildParams(t),
      repeat: t.repeat,
      lora_scale: t.loraScale,
    })),
  };
}

// ── job views (what GET /playground/jobs/{id} and the manifest return) ───────

export interface RenderResult {
  rep: number;
  status: string;
  file: string | null;
  seconds?: number;
  gen_seconds?: number;
  realtime?: number | null;
  seed?: number;
  complete?: boolean | null;
  warnings?: string[];
  error?: string;
  warmed_up?: number;
  voice?: { borrowed?: boolean; warning?: string };
}

export interface TakeView {
  id: string;
  number: number;
  label: string;
  text: string;
  model_id: string;
  model?: { label: string; type: string; path?: string };
  repeat: number;
  results: RenderResult[];
  notes: string;
  starred: boolean;
}

export interface JobView {
  id: string;
  session: string;
  folder: string;
  status: 'running' | 'done' | 'cancelled' | 'error';
  message?: string;
  total: number;
  done: number;
  model_swaps: number;
  takes: TakeView[];
  log: Array<{ level?: string; message?: string } | string>;
}

export interface SessionRow {
  session: string;
  name: string;
  created: number | null;
  status: string | null;
  takes: number;
  renders: number;
  folder: string;
}

/** The model's label with where it lives, so two folders with the same checkpoint name stay apart. */
export function modelOptionLabel(m: QwenModel, where: { huggingface: string; environment: string }): string {
  const place =
    m.location_name ??
    (m.location === 'huggingface_cache' ? where.huggingface : m.location === 'environment' ? where.environment : '');
  return `${m.label}${place ? `, ${place}` : ''} (${m.type}, ${m.size})`;
}

/** True when a saved take points at a model the app can no longer find. Unknown until models are loaded. */
export function modelIsMissing(modelId: string, models: QwenModel[] | undefined): boolean {
  return !!models && models.length > 0 && !!modelId && !models.some((m) => m.id === modelId);
}

export const resultKey = (session: string, file: string) => `${session}/${file}`;

/** The warnings to show next to a render: its own list, or the borrowed-voice one. */
export function resultWarnings(r: RenderResult): string[] {
  if (r.warnings?.length) return r.warnings;
  return r.voice?.warning ? [r.voice.warning] : [];
}
