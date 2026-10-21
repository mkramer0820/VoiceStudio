import { describe, expect, it } from 'vitest';
import { makeChunk, type StoryChunk } from './story-model';
import {
  applyCast,
  countMatches,
  detectSpeakers,
  extractSpeakers,
  findReplace,
  findUnappliedSpeakerTags,
  rewriteTags,
  scanTags,
} from './tag-tools';

const lines = (...texts: string[]): StoryChunk[] => texts.map((t) => makeChunk(t));
const texts = (chunks: StoryChunk[]) => chunks.map((c) => c.text);
const speakers = (chunks: StoryChunk[]) => chunks.map((c) => c.speaker ?? null);

const SHADOW =
  '[Shadow] [slow][soft] It feels strange, [pause] to become a [emphasis] human once more.';

describe('detecting speakers vs directions', () => {
  it('treats a capitalised tag opening a line as a speaker and delivery words as directions', () => {
    const chunks = lines(SHADOW, '[Snuggle] Hi there.', 'Narration with a [Pause] in it.');
    expect([...detectSpeakers(chunks)].sort()).toEqual(['Shadow', 'Snuggle']);
  });

  it('never mistakes a capitalised delivery word for a character', () => {
    expect([...detectSpeakers(lines('[Pause] hold on', '[Soft] hush'))]).toEqual([]);
  });

  it('lets the user override either way', () => {
    const chunks = lines(SHADOW, '[Nephis] Hello.');
    expect([...detectSpeakers(chunks, { Shadow: 'direction' })]).toEqual(['Nephis']);
    expect([...detectSpeakers(chunks, { slow: 'speaker' })].sort()).toEqual(['Nephis', 'Shadow', 'slow']);
  });

  it('scans every tag with kind and count, speakers first', () => {
    const tags = scanTags(lines(SHADOW, '[Shadow] Again. [pause]'));
    expect(tags[0]).toEqual({ name: 'Shadow', kind: 'speaker', count: 2 });
    expect(tags.find((t) => t.name === 'pause')).toEqual({ name: 'pause', kind: 'direction', count: 2 });
    expect(tags.map((t) => t.kind)).toEqual(['speaker', 'direction', 'direction', 'direction', 'direction']);
  });
});

describe('extractSpeakers', () => {
  const opts = { carry: false, strip: true };

  it('records the speaker, strips the name and keeps the delivery tags', () => {
    const out = extractSpeakers(lines(SHADOW), new Set(['Shadow']), opts);
    expect(texts(out)).toEqual(['[slow][soft] It feels strange, [pause] to become a [emphasis] human once more.']);
    expect(speakers(out)).toEqual(['Shadow']);
  });

  it('leaves narration (no speaker tag) without a speaker, so it uses the default voice', () => {
    const out = extractSpeakers(
      lines('[Shadow] Hello.', 'He looked at his hands.', '[Snuggle] Hi.'),
      new Set(['Shadow', 'Snuggle']),
      opts,
    );
    expect(speakers(out)).toEqual(['Shadow', null, 'Snuggle']);
    expect(out[1].text).toBe('He looked at his hands.');
  });

  it('can carry a speaker over to the following untagged lines', () => {
    const out = extractSpeakers(lines('[Shadow] One.', 'Two.', '[Snuggle] Three.', 'Four.'), new Set(['Shadow', 'Snuggle']), {
      carry: true,
      strip: true,
    });
    expect(speakers(out)).toEqual(['Shadow', 'Shadow', 'Snuggle', 'Snuggle']);
  });

  it('splits a line where a different speaker starts', () => {
    const out = extractSpeakers(lines('[Shadow] Run. [Snuggle] Where to?'), new Set(['Shadow', 'Snuggle']), opts);
    expect(texts(out)).toEqual(['Run.', 'Where to?']);
    expect(speakers(out)).toEqual(['Shadow', 'Snuggle']);
    expect(new Set(out.map((c) => c.id)).size).toBe(2);
  });

  it('turns a speaker-only line into a speaker change without adding an empty line', () => {
    const out = extractSpeakers(lines('[Shadow]', 'Spoken.'), new Set(['Shadow']), { carry: true, strip: true });
    expect(texts(out)).toEqual(['Spoken.']);
    expect(speakers(out)).toEqual(['Shadow']);
  });

  it('keeps the tag in the text when stripping is off', () => {
    const out = extractSpeakers(lines('[Shadow] Hi.'), new Set(['Shadow']), { carry: false, strip: false });
    expect(texts(out)).toEqual(['[Shadow] Hi.']);
    expect(speakers(out)).toEqual(['Shadow']);
  });

  it('is idempotent and keeps already-extracted speakers', () => {
    const once = extractSpeakers(lines(SHADOW, 'Plain.'), new Set(['Shadow']), opts);
    const twice = extractSpeakers(once, new Set(['Shadow']), opts);
    expect(texts(twice)).toEqual(texts(once));
    expect(speakers(twice)).toEqual(['Shadow', null]);
  });

  it('keeps the voice already given to a line', () => {
    const [first] = lines('[Shadow] Hi.');
    const out = extractSpeakers([{ ...first, voice: 'vx' }], new Set(['Shadow']), opts);
    expect(out[0].voice).toBe('vx');
  });
});

describe('applyCast', () => {
  it('maps each speaker to a voice and leaves narration and unmapped speakers alone', () => {
    const chunks = extractSpeakers(
      lines('[Shadow] A.', 'Narration.', '[Nephis] B.'),
      new Set(['Shadow', 'Nephis']),
      { carry: false, strip: true },
    );
    const out = applyCast(chunks, { Shadow: 'v-shadow' });
    expect(out.map((c) => c.voice ?? null)).toEqual(['v-shadow', null, null]);
  });
});

describe('rewriteTags', () => {
  const chunks = lines('[slow][soft] It feels strange, [pause] to become [emphasis] human.');

  it('removes and replaces chosen direction tags and keeps the rest', () => {
    const out = rewriteTags(chunks, {
      slow: { action: 'replace', to: 'slowly' },
      pause: { action: 'remove' },
      soft: { action: 'keep' },
    });
    expect(out[0].text).toBe('[slowly][soft] It feels strange, to become [emphasis] human.');
  });

  it('accepts a replacement typed with brackets and treats an empty one as removal', () => {
    const out = rewriteTags(chunks, { soft: { action: 'replace', to: '[whisper]' }, slow: { action: 'replace', to: ' ' } });
    expect(out[0].text).toBe('[whisper] It feels strange, [pause] to become [emphasis] human.');
  });

  it('does not touch lines that have no matching tag', () => {
    const plain = lines('No tags here.');
    expect(rewriteTags(plain, { slow: { action: 'remove' } })[0]).toBe(plain[0]);
  });
});

describe('findReplace', () => {
  const chunks = lines('Shadow spoke. shadow left.', 'A (b) [c].');

  it('replaces plain text, case-insensitively by default', () => {
    expect(countMatches(chunks, 'shadow', false)).toEqual({ matches: 2, lines: 1 });
    expect(texts(findReplace(chunks, 'shadow', 'Nephis', false))[0]).toBe('Nephis spoke. Nephis left.');
  });

  it('can match case', () => {
    expect(texts(findReplace(chunks, 'Shadow', 'X', true))[0]).toBe('X spoke. shadow left.');
  });

  it('treats the search as literal text, not a pattern, and replaces a tag by its text', () => {
    expect(countMatches(chunks, '(b)', true).matches).toBe(1);
    expect(texts(findReplace(chunks, '[c]', '[$&]', true))[1]).toBe('A (b) [$&].');
    expect(countMatches(chunks, '', false)).toEqual({ matches: 0, lines: 0 });
  });
});

describe('findUnappliedSpeakerTags', () => {
  const line = (text: string, speaker?: string) => ({ ...makeChunk(text), speaker: speaker ?? null });

  it('flags lines that still open with a character tag, by story line number', () => {
    const chunks = [
      line('[Shadow] [slow] It feels strange.'),
      line('Plain narration.'),
      line('[Nephis][soft] Hello.'),
      line('[Pause] Wait.'),
      line('[Snuggle] Done.', 'Snuggle'),
    ];
    expect(findUnappliedSpeakerTags(chunks)).toEqual([
      { line: 1, tag: 'Shadow' },
      { line: 3, tag: 'Nephis' },
    ]);
  });

  it('ignores delivery tags and anything in the middle of a line', () => {
    const chunks = [line('[slow][soft] It is, [Pause] done [Shadow] ok.'), line('He said [Emphasis] no.')];
    expect(findUnappliedSpeakerTags(chunks)).toEqual([]);
  });
});
