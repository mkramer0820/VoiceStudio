/**
 * Script-tag tools for AI-directed scripts such as
 * `[Shadow] [slow][soft] It feels strange, [pause] to become a [emphasis] human.`
 *
 * Two kinds of bracket tag:
 *  - a *speaker* (`[Shadow]`): who talks. It is recorded on the line and mapped to a voice.
 *  - a *direction* (`[slow]`, `[pause]`): delivery. Kept for Fish by default, or rewritten.
 */
import { makeChunk, type StoryChunk } from './story-model';

const TAG = /\[([^[\]\n]{1,40})\]/g;

/** Common delivery words, so `[Pause]` or `[Soft]` is never mistaken for a character. */
const DIRECTION_WORDS = new Set(
  (
    'slow slowly fast quick quickly soft softly loud loudly quiet quietly whisper whispers whispering ' +
    'shout shouts shouting yell scream emphasis emphasize stress pause beat break silence breath inhale ' +
    'exhale sigh sighs laugh laughs laughing chuckle giggle cry sob sobbing gasp cough clears throat ' +
    'sad happy angry excited calm nervous scared tired bored serious playful sarcastic gentle warm cold ' +
    'low high deep raspy hoarse trembling shaky whispered stage direction sfx sound music'
  ).split(' '),
);

export type TagKind = 'speaker' | 'direction';
export type TagKinds = Record<string, TagKind>;

export interface TagInfo {
  name: string;
  kind: TagKind;
  count: number;
}

function looksLikeSpeaker(name: string): boolean {
  const n = name.trim();
  return /^\p{Lu}[\p{L}\p{N} _'.-]{0,30}$/u.test(n) && !DIRECTION_WORDS.has(n.toLowerCase());
}

/** Tags that open a line (after other leading tags), in order: `[A] [b]x` → ['A','b']. */
function leadingTags(text: string): string[] {
  const out: string[] = [];
  let rest = text.trimStart();
  for (;;) {
    const m = /^\[([^[\]\n]{1,40})\]\s*/.exec(rest);
    if (!m) break;
    out.push(m[1].trim());
    rest = rest.slice(m[0].length);
  }
  return out;
}

/**
 * The set of speaker names: a capitalised, non-delivery tag that opens a line,
 * plus anything the user marked as a speaker, minus anything marked a direction.
 */
export function detectSpeakers(chunks: readonly StoryChunk[], kinds: TagKinds = {}): Set<string> {
  const found = new Set<string>();
  for (const chunk of chunks) {
    const lead = leadingTags(chunk.text)[0];
    if (lead && looksLikeSpeaker(lead)) found.add(lead);
  }
  for (const [name, kind] of Object.entries(kinds)) {
    if (kind === 'speaker') found.add(name);
    else found.delete(name);
  }
  return found;
}

/** Every tag in the story with its kind and how often it appears. */
export function scanTags(chunks: readonly StoryChunk[], kinds: TagKinds = {}): TagInfo[] {
  const speakers = detectSpeakers(chunks, kinds);
  const counts = new Map<string, number>();
  for (const chunk of chunks) {
    for (const m of chunk.text.matchAll(TAG)) {
      const name = m[1].trim();
      counts.set(name, (counts.get(name) ?? 0) + 1);
    }
  }
  return [...counts.entries()]
    .map(([name, count]) => ({ name, count, kind: (speakers.has(name) ? 'speaker' : 'direction') as TagKind }))
    .sort((a, b) => (a.kind === b.kind ? b.count - a.count : a.kind === 'speaker' ? -1 : 1));
}

const tidy = (s: string) => s.replace(/[ \t]{2,}/g, ' ').replace(/\s+([,.;:!?])/g, '$1').trim();

export interface ExtractOptions {
  /** Lines with no speaker tag keep talking as the previous speaker. */
  carry: boolean;
  /** Remove the speaker tags from the text that will be sent to Fish. */
  strip: boolean;
}

/**
 * Record who speaks each line (`chunk.speaker`) from the speaker tags. A tag in
 * the middle of a line starts a new line; a line that is only a speaker tag just
 * changes the current speaker. Direction tags are left alone.
 */
export function extractSpeakers(
  chunks: readonly StoryChunk[],
  speakers: ReadonlySet<string>,
  { carry, strip }: ExtractOptions,
): StoryChunk[] {
  const out: StoryChunk[] = [];
  let current: string | null = null;
  for (const chunk of chunks) {
    const matches = [...chunk.text.matchAll(TAG)].filter((m) => speakers.has(m[1].trim()));
    let speaker: string | null = chunk.speaker ?? (carry ? current : null);
    if (matches.length === 0) {
      out.push(speaker === (chunk.speaker ?? null) ? chunk : { ...chunk, speaker });
      current = speaker;
      continue;
    }
    const parts: Array<{ speaker: string | null; text: string }> = [];
    let cursor = 0;
    let buffer = '';
    for (const m of matches) {
      buffer += chunk.text.slice(cursor, m.index);
      cursor = (m.index ?? 0) + m[0].length;
      if (buffer.trim()) parts.push({ speaker, text: buffer });
      buffer = strip ? '' : m[0];
      speaker = m[1].trim();
    }
    buffer += chunk.text.slice(cursor);
    if (buffer.trim()) parts.push({ speaker, text: buffer });
    current = speaker;
    // A line that was only a speaker tag changes who is talking but adds no line.
    parts.forEach((part, i) => {
      const text = tidy(part.text);
      const base = i === 0 ? chunk : makeChunk(text, chunk.voice);
      out.push({ ...base, text, speaker: part.speaker });
    });
  }
  return out;
}

/** Give every line the voice its speaker is mapped to. Unmapped speakers are untouched. */
export function applyCast(
  chunks: readonly StoryChunk[],
  cast: Readonly<Record<string, string>>,
): StoryChunk[] {
  return chunks.map((c) => (c.speaker && cast[c.speaker] ? { ...c, voice: cast[c.speaker] } : c));
}

export type DirectionAction = { action: 'keep' } | { action: 'remove' } | { action: 'replace'; to: string };

/** Rewrite direction tags: remove them or swap them for other tags (`[slow]` → `[slowly]`). */
export function rewriteTags(
  chunks: readonly StoryChunk[],
  rules: Readonly<Record<string, DirectionAction>>,
): StoryChunk[] {
  return chunks.map((chunk) => {
    const text = chunk.text.replace(TAG, (whole, raw: string) => {
      const rule = rules[raw.trim()];
      if (!rule || rule.action === 'keep') return whole;
      if (rule.action === 'remove') return '';
      const to = rule.to.trim().replace(/^\[|\]$/g, '').trim();
      return to ? `[${to}]` : '';
    });
    return text === chunk.text ? chunk : { ...chunk, text: tidy(text) };
  });
}

const escapeRegex = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

function finder(find: string, matchCase: boolean): RegExp | null {
  return find ? new RegExp(escapeRegex(find), matchCase ? 'g' : 'gi') : null;
}

/** How many times `find` occurs, and on how many lines (plain text, not a pattern). */
export function countMatches(
  chunks: readonly StoryChunk[],
  find: string,
  matchCase: boolean,
): { matches: number; lines: number } {
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

/** Plain-text find and replace across the whole story. */
export function findReplace(
  chunks: readonly StoryChunk[],
  find: string,
  replace: string,
  matchCase: boolean,
): StoryChunk[] {
  const re = finder(find, matchCase);
  if (!re) return [...chunks];
  return chunks.map((c) => {
    const text = c.text.replace(re, () => replace);
    return text === c.text ? c : { ...c, text };
  });
}
