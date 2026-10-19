/** Pure logic for the Fish story board: chunks, selection, bulk voice assignment. */

export interface StoryChunk {
  id: string;
  text: string;
  /** Fish voice (reference) id; undefined/null = use the default voice. */
  voice?: string | null;
  /** Who speaks this line (from a `[Name]` tag); mapped to a voice through `cast`. */
  speaker?: string | null;
}

export interface StorySettings {
  model?: string | null;
  format?: 'wav' | 'pcm' | 'mp3' | 'opus' | null;
  sample_rate?: number | null;
  mp3_bitrate?: 64 | 128 | 192 | null;
  latency?: 'normal' | 'balanced' | null;
  normalize?: boolean | null;
  temperature?: number | null;
  top_p?: number | null;
  chunk_length?: number | null;
  opus_bitrate?: number | null;
  repetition_penalty?: number | null;
  max_new_tokens?: number | null;
  min_chunk_length?: number | null;
  condition_on_previous_chunks?: boolean | null;
  early_stop_threshold?: number | null;
  speed: number;
  volume?: number | null;
  normalize_loudness?: boolean | null;
  language?: string | null;
  max_concurrency?: number | null;
  default_voice?: string | null;
  pause_ms: number;
  join: boolean;
}

export interface StoryDraft {
  title: string;
  chunks: StoryChunk[];
  /** voice id -> display name, so the board reads well without contacting Fish. */
  voices: Record<string, string>;
  /** speaker name -> Fish voice id */
  cast: Record<string, string>;
  settings: StorySettings;
}

export type SplitMode = 'lines' | 'paragraphs' | 'sentences';

export const DEFAULT_SETTINGS: StorySettings = { speed: 1, pause_ms: 350, join: true };

let counter = 0;
export function newChunkId(): string {
  counter += 1;
  return `c${Date.now().toString(36)}${counter.toString(36)}`;
}

export function makeChunk(text: string, voice?: string | null): StoryChunk {
  return { id: newChunkId(), text, voice: voice ?? null };
}

const SENTENCE = /[^.!?…。！？]+[.!?…。！？]+["'”’)\]]*\s*|[^.!?…。！？]+$/g;

/** Split pasted text into chunks. Blank pieces are dropped; text is never rewritten. */
export function splitText(input: string, mode: SplitMode, maxChars = 0): StoryChunk[] {
  const text = input.replace(/\r\n?/g, '\n');
  let pieces: string[];
  if (mode === 'lines') pieces = text.split('\n');
  else if (mode === 'paragraphs') pieces = text.split(/\n\s*\n/);
  else pieces = text.split('\n').flatMap((line) => line.match(SENTENCE) ?? []);
  const out: string[] = [];
  for (const piece of pieces.map((p) => p.trim()).filter(Boolean)) {
    if (maxChars > 0 && piece.length > maxChars) out.push(...wrapAt(piece, maxChars));
    else out.push(piece);
  }
  return out.map((p) => makeChunk(p));
}

/** Break an over-long piece at the last space before `max`; hard-cut only when there is none. */
function wrapAt(piece: string, max: number): string[] {
  const parts: string[] = [];
  let rest = piece;
  while (rest.length > max) {
    let cut = rest.lastIndexOf(' ', max);
    if (cut < max * 0.4) cut = max;
    parts.push(rest.slice(0, cut).trim());
    rest = rest.slice(cut).trim();
  }
  if (rest) parts.push(rest);
  return parts;
}

/** 1-based inclusive range of lines, order-insensitive and clamped to the story. */
export function idsInRange(chunks: StoryChunk[], from: number, to: number): string[] {
  if (chunks.length === 0 || !Number.isFinite(from) || !Number.isFinite(to)) return [];
  const lo = Math.max(1, Math.min(from, to));
  const hi = Math.min(chunks.length, Math.max(from, to));
  return chunks.slice(lo - 1, hi).map((c) => c.id);
}

/** Shift-click: everything between the anchor and the clicked row, inclusive. */
export function idsBetween(chunks: StoryChunk[], anchorId: string, targetId: string): string[] {
  const a = chunks.findIndex((c) => c.id === anchorId);
  const b = chunks.findIndex((c) => c.id === targetId);
  if (a < 0 || b < 0) return b < 0 ? [] : [targetId];
  return chunks.slice(Math.min(a, b), Math.max(a, b) + 1).map((c) => c.id);
}

/**
 * The next block of the same size right after the current selection — "I just
 * gave these 15 lines a voice, now the next 15".
 */
export function nextBlock(chunks: StoryChunk[], selected: ReadonlySet<string>): string[] {
  const picked = chunks.map((c, i) => (selected.has(c.id) ? i : -1)).filter((i) => i >= 0);
  if (picked.length === 0) return [];
  const start = Math.max(...picked) + 1;
  return chunks.slice(start, start + picked.length).map((c) => c.id);
}

export function applyVoice(
  chunks: StoryChunk[],
  selected: ReadonlySet<string>,
  voice: string | null,
): StoryChunk[] {
  return chunks.map((c) => (selected.has(c.id) ? { ...c, voice } : c));
}

export function mergeWithNext(chunks: StoryChunk[], id: string): StoryChunk[] {
  const i = chunks.findIndex((c) => c.id === id);
  if (i < 0 || i >= chunks.length - 1) return chunks;
  const merged = { ...chunks[i], text: `${chunks[i].text.trimEnd()} ${chunks[i + 1].text.trimStart()}` };
  return [...chunks.slice(0, i), merged, ...chunks.slice(i + 2)];
}

export function insertAfter(chunks: StoryChunk[], id: string, chunk: StoryChunk): StoryChunk[] {
  const i = chunks.findIndex((c) => c.id === id);
  return i < 0 ? [...chunks, chunk] : [...chunks.slice(0, i + 1), chunk, ...chunks.slice(i + 1)];
}

export function removeChunks(chunks: StoryChunk[], ids: ReadonlySet<string>): StoryChunk[] {
  return chunks.filter((c) => !ids.has(c.id));
}

export interface StorySummary {
  chunks: number;
  chars: number;
  unassigned: number;
  perVoice: Array<{ voice: string | null; chunks: number; chars: number }>;
}

/** Local counts for the staging bar (the backend plan adds the effective settings). */
export function summarize(chunks: StoryChunk[]): StorySummary {
  const per = new Map<string | null, { chunks: number; chars: number }>();
  let count = 0;
  let chars = 0;
  let unassigned = 0;
  for (const c of chunks) {
    const text = c.text.trim();
    if (!text) continue;
    count += 1;
    chars += text.length;
    const key = c.voice ?? null;
    if (key === null) unassigned += 1;
    const slot = per.get(key) ?? { chunks: 0, chars: 0 };
    slot.chunks += 1;
    slot.chars += text.length;
    per.set(key, slot);
  }
  return {
    chunks: count,
    chars,
    unassigned,
    perVoice: [...per.entries()].map(([voice, v]) => ({ voice, ...v })),
  };
}

const HUES = [210, 28, 142, 330, 52, 265, 175, 8, 95, 300];

/** A stable colour for a voice so ranges are recognisable at a glance. */
export function voiceHue(voice: string | null | undefined, order: readonly string[]): number | null {
  if (!voice) return null;
  const index = order.indexOf(voice);
  if (index >= 0) return HUES[index % HUES.length];
  let hash = 0;
  for (const ch of voice) hash = (hash * 31 + ch.charCodeAt(0)) >>> 0;
  return HUES[hash % HUES.length];
}

/** Voices in order of first appearance, for colour assignment. */
export function voiceOrder(chunks: StoryChunk[]): string[] {
  const seen: string[] = [];
  for (const c of chunks) if (c.voice && !seen.includes(c.voice)) seen.push(c.voice);
  return seen;
}
