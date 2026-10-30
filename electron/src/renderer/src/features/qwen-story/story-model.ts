/**
 * Pure logic for the Qwen story page: profile forms, the draft sent to the backend, character mapping checks.
 * Mirrors backend/engines/qwen3_studio/{profiles,story,emotions}.py. Nothing here touches React or the network.
 */
import type { QwenModel } from '../qwen-playground/qwen-model';

export type ProfileKind = 'narrator' | 'clone' | 'stock';
export type Strength = 'normal' | 'strong' | 'very_strong';

export interface Emotion {
  preset: string;
  strength: Strength;
  custom_text?: string | null;
  /** Reserved: accepted by the backend and ignored ("not available yet"). */
  steer?: { vector_id?: string | null; weight?: number | null } | null;
}

export interface EmotionCatalog {
  presets: Array<{ name: string; texts: Record<Strength, string> }>;
  strengths: Strength[];
  unverified_on_narrator: string;
  steer_message: string;
}

export type EmbeddingState = 'none' | 'preparing' | 'ready' | 'failed';

export interface Profile {
  id: string;
  name: string;
  kind: ProfileKind;
  model_id: string;
  speaker: string | null;
  ref_audio: string | null;
  ref_text: string | null;
  xvec_only: boolean;
  default_instruct: string | null;
  emotion: Emotion;
  temperature: number | null;
  top_k: number | null;
  top_p: number | null;
  repetition_penalty: number | null;
  notes: string;
  has_embedding?: boolean;
  /** The speaker embedding is made automatically; the user never handles it by default. */
  embedding_state?: EmbeddingState;
  embedding_error?: string;
}

/** The profile form: numbers are strings while typing; empty = null = the engine's own default. */
export interface ProfileForm {
  id?: string;
  name: string;
  kind: ProfileKind;
  model_id: string;
  speaker: string;
  ref_audio: string;
  ref_text: string;
  xvec_only: boolean;
  default_instruct: string;
  emotion: Emotion;
  temperature: string;
  top_k: string;
  top_p: string;
  repetition_penalty: string;
  notes: string;
}

export const NEUTRAL: Emotion = { preset: 'neutral', strength: 'normal' };

export function blankForm(models: QwenModel[]): ProfileForm {
  const model = models.find((m) => m.type === 'custom_voice') ?? models[0];
  return {
    name: '',
    kind: 'narrator',
    model_id: model?.id ?? '',
    speaker: model?.speakers.length === 1 ? model.speakers[0] : '',
    ref_audio: '',
    ref_text: '',
    xvec_only: false,
    default_instruct: '',
    emotion: { ...NEUTRAL },
    temperature: '',
    top_k: '',
    top_p: '',
    repetition_penalty: '',
    notes: '',
  };
}

export function formFromProfile(p: Profile): ProfileForm {
  const num = (v: number | null) => (v == null ? '' : String(v));
  return {
    id: p.id,
    name: p.name,
    kind: p.kind,
    model_id: p.model_id,
    speaker: p.speaker ?? '',
    ref_audio: p.ref_audio ?? '',
    ref_text: p.ref_text ?? '',
    xvec_only: p.xvec_only,
    default_instruct: p.default_instruct ?? '',
    emotion: p.emotion ?? { ...NEUTRAL },
    temperature: num(p.temperature),
    top_k: num(p.top_k),
    top_p: num(p.top_p),
    repetition_penalty: num(p.repetition_penalty),
    notes: p.notes ?? '',
  };
}

const orNull = (s: string) => (s.trim() === '' ? null : s.trim());
const numOrNull = (s: string) => (s.trim() === '' || Number.isNaN(Number(s)) ? null : Number(s));

/** The request body: empty fields become null, never a made-up default. */
export function profileBody(f: ProfileForm) {
  const clone = f.kind === 'clone';
  return {
    name: f.name.trim(),
    kind: f.kind,
    model_id: f.model_id,
    speaker: clone ? null : orNull(f.speaker),
    ref_audio: clone ? orNull(f.ref_audio) : null,
    ref_text: clone ? orNull(f.ref_text) : null,
    xvec_only: clone && f.xvec_only,
    default_instruct: orNull(f.default_instruct),
    emotion: f.emotion,
    temperature: numOrNull(f.temperature),
    top_k: numOrNull(f.top_k),
    top_p: numOrNull(f.top_p),
    repetition_penalty: numOrNull(f.repetition_penalty),
    notes: f.notes,
  };
}

/** An i18n key naming what is missing, or null. */
export function profileProblem(f: ProfileForm, hasNewClip = false): string | null {
  if (!f.name.trim()) return 'qwenStory.problem_name';
  if (!f.model_id) return 'qwenStory.problem_model';
  if (f.kind === 'clone') {
    if (!f.ref_audio.trim() && !hasNewClip) return 'qwenStory.problem_clip';
    if (!f.xvec_only && !f.ref_text.trim()) return 'qwenStory.problem_text';
  }
  return null;
}

/** Models a profile of this kind can use: clones run on Base, speakers on custom_voice (stock voices may be borrowed). */
export function modelsForKind(kind: ProfileKind, models: QwenModel[]): QwenModel[] {
  const want = kind === 'clone' ? 'base' : 'custom_voice';
  return models.filter((m) => m.type === want);
}

// ── story ────────────────────────────────────────────────────────────────
export interface StoryChunk {
  id: string;
  text: string;
  speaker: string | null;
  emotion: Emotion | null;
}

export interface StorySettings {
  language: string;
  pad_s: number;
  seed: number | null;
  speaker_change_pause_ms: number;
  warmup: boolean;
}

export const DEFAULT_STORY_SETTINGS: StorySettings = {
  language: 'Auto',
  pad_s: 0.3,
  seed: null,
  speaker_change_pause_ms: 0,
  warmup: false,
};

export interface Draft {
  title: string;
  chunks: StoryChunk[];
  narrator_profile_id: string | null;
  cast: Record<string, string>;
  emotions: Record<string, Emotion>;
  settings: StorySettings;
}

export interface CharacterRow {
  /** null = narration (untagged text). */
  name: string | null;
  chunks: number;
  chars: number;
}

export function characterRows(chunks: StoryChunk[]): CharacterRow[] {
  const seen = new Map<string | null, CharacterRow>();
  for (const c of chunks) {
    if (!c.text.trim()) continue;
    const row = seen.get(c.speaker) ?? { name: c.speaker, chunks: 0, chars: 0 };
    row.chunks += 1;
    row.chars += c.text.length;
    seen.set(c.speaker, row);
  }
  return [...seen.values()].sort((a, b) => Number(a.name !== null) - Number(b.name !== null));
}

export interface MappingWarning {
  key: string;
  values: Record<string, string>;
}

/** Warnings only, never blocks (free mapping). A render with no voice for some chunk is refused by the backend. */
export function mappingWarnings(draft: Draft, profiles: Profile[], models: QwenModel[]): MappingWarning[] {
  const byId = new Map(profiles.map((p) => [p.id, p]));
  const out: MappingWarning[] = [];
  const names = characterRows(draft.chunks);
  for (const row of names) {
    const id = row.name === null ? draft.narrator_profile_id : (draft.cast[row.name] ?? null);
    const label = row.name ?? '';
    if (!id) {
      if (row.name === null) out.push({ key: 'qwenStory.warn_no_narrator', values: {} });
      else out.push({ key: 'qwenStory.warn_unmapped', values: { name: label } });
      continue;
    }
    const profile = byId.get(id);
    if (!profile) {
      out.push({ key: 'qwenStory.warn_profile_gone', values: { name: label || '-' } });
      continue;
    }
    if (!models.some((m) => m.id === profile.model_id)) {
      out.push({ key: 'qwenStory.warn_model_missing', values: { name: profile.name } });
    }
  }
  return out;
}

/** How many distinct models the render will load (profiles on one model never reload it). */
export function modelLoads(draft: Draft, profiles: Profile[]): number {
  const byId = new Map(profiles.map((p) => [p.id, p]));
  const models = new Set<string>();
  for (const row of characterRows(draft.chunks)) {
    const id = row.name === null ? draft.narrator_profile_id : (draft.cast[row.name] ?? draft.narrator_profile_id);
    const profile = id ? byId.get(id) : undefined;
    if (profile) models.add(profile.model_id);
  }
  return models.size;
}

export function unmappedCharacters(draft: Draft): string[] {
  return characterRows(draft.chunks)
    .filter((r): r is CharacterRow & { name: string } => r.name !== null && !draft.cast[r.name])
    .map((r) => r.name);
}

export function setEmotion(chunks: StoryChunk[], id: string, emotion: Emotion | null): StoryChunk[] {
  return chunks.map((c) => (c.id === id ? { ...c, emotion } : c));
}

export function setSpeaker(chunks: StoryChunk[], id: string, speaker: string | null): StoryChunk[] {
  return chunks.map((c) => (c.id === id ? { ...c, speaker } : c));
}

export function editText(chunks: StoryChunk[], id: string, text: string): StoryChunk[] {
  return chunks.map((c) => (c.id === id ? { ...c, text } : c));
}

export function removeChunk(chunks: StoryChunk[], id: string): StoryChunk[] {
  return chunks.filter((c) => c.id !== id);
}

// ── bulk editing (the same affordances as the Fish story board: select lines, apply to the selection, tags dialog) ──
/** Shift-click: everything between the anchor and the clicked row, inclusive. */
export function idsBetween(chunks: StoryChunk[], anchorId: string, targetId: string): string[] {
  const a = chunks.findIndex((c) => c.id === anchorId);
  const b = chunks.findIndex((c) => c.id === targetId);
  if (a < 0 || b < 0) return b < 0 ? [] : [targetId];
  return chunks.slice(Math.min(a, b), Math.max(a, b) + 1).map((c) => c.id);
}

/** All lines of one character (null = narration). */
export function idsOfSpeaker(chunks: StoryChunk[], speaker: string | null): string[] {
  return chunks.filter((c) => c.speaker === speaker).map((c) => c.id);
}

export function applyEmotion(chunks: StoryChunk[], ids: ReadonlySet<string>, emotion: Emotion | null): StoryChunk[] {
  return chunks.map((c) => (ids.has(c.id) ? { ...c, emotion } : c));
}

export function applySpeaker(chunks: StoryChunk[], ids: ReadonlySet<string>, speaker: string | null): StoryChunk[] {
  return chunks.map((c) => (ids.has(c.id) ? { ...c, speaker } : c));
}

/** Rename (or merge into another) character in the lines and in the mapping and emotion tables. */
export function renameCharacter(draft: Draft, from: string, to: string): Draft {
  const name = to.trim();
  if (!name || name === from) return draft;
  const cast = { ...draft.cast };
  const emotions = { ...draft.emotions };
  if (cast[from]) cast[name] = cast[name] ?? cast[from];
  if (emotions[from]) emotions[name] = emotions[name] ?? emotions[from];
  delete cast[from];
  delete emotions[from];
  return { ...draft, cast, emotions, chunks: draft.chunks.map((c) => (c.speaker === from ? { ...c, speaker: name } : c)) };
}

/** How many lines carry each inline emotion preset (a line with no tag is not counted). */
export function emotionCounts(chunks: StoryChunk[]): Array<{ preset: string; count: number }> {
  const counts = new Map<string, number>();
  for (const c of chunks) if (c.emotion) counts.set(c.emotion.preset, (counts.get(c.emotion.preset) ?? 0) + 1);
  return [...counts.entries()].map(([preset, count]) => ({ preset, count })).sort((a, b) => b.count - a.count);
}

export function clearEmotions(chunks: StoryChunk[], preset?: string): StoryChunk[] {
  return chunks.map((c) => (c.emotion && (!preset || c.emotion.preset === preset) ? { ...c, emotion: null } : c));
}

const escapeRegex = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
const finder = (find: string, matchCase: boolean) => (find ? new RegExp(escapeRegex(find), matchCase ? 'g' : 'gi') : null);

/** How many times `find` occurs, and on how many lines (plain text, not a pattern). */
export function countMatches(chunks: StoryChunk[], find: string, matchCase: boolean): { matches: number; lines: number } {
  const re = finder(find, matchCase);
  if (!re) return { matches: 0, lines: 0 };
  let matches = 0;
  let lines = 0;
  for (const c of chunks) {
    const n = c.text.match(re)?.length ?? 0;
    matches += n;
    if (n) lines += 1;
  }
  return { matches, lines };
}

export function findReplace(chunks: StoryChunk[], find: string, replace: string, matchCase: boolean): StoryChunk[] {
  const re = finder(find, matchCase);
  if (!re) return chunks;
  return chunks.map((c) => {
    const text = c.text.replace(re, () => replace);
    return text === c.text ? c : { ...c, text };
  });
}

// ── render job ───────────────────────────────────────────────────────────
export interface ChunkResult {
  id: string;
  index: number;
  status: 'queued' | 'running' | 'done' | 'error';
  file: string | null;
  history: string[];
  speaker: string | null;
  profile_name?: string;
  seconds?: number;
  realtime?: number | null;
  error?: string | null;
  warnings?: string[];
  complete?: boolean | null;
}

export interface StoryJob {
  id: string;
  story: string;
  run: string;
  folder: string;
  status: 'running' | 'done' | 'cancelled' | 'error';
  message: string;
  total: number;
  done: number;
  /** Names of the voices being prepared before the first chunk renders. */
  preparing?: string[];
  warnings: string[];
  chunks: ChunkResult[];
  joined?: string | null;
}

export function incompleteChunkIds(job: StoryJob | null): string[] {
  return (job?.chunks ?? []).filter((c) => c.status === 'error' || c.status === 'queued').map((c) => c.id);
}
