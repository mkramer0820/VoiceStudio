import { describe, expect, it } from 'vitest';
import {
  applyVoice,
  idsBetween,
  idsInRange,
  insertAfter,
  makeChunk,
  mergeWithNext,
  nextBlock,
  removeChunks,
  splitText,
  summarize,
  voiceOrder,
} from './story-model';

const story = (n: number) => Array.from({ length: n }, (_, i) => makeChunk(`line ${i + 1}`));

describe('splitText', () => {
  it('splits lines, drops blanks and never rewrites the text', () => {
    const out = splitText('One.\r\n\r\n  Two  \nThree', 'lines');
    expect(out.map((c) => c.text)).toEqual(['One.', 'Two', 'Three']);
  });

  it('splits paragraphs on blank lines', () => {
    const out = splitText('A\nstill A\n\nB', 'paragraphs');
    expect(out.map((c) => c.text)).toEqual(['A\nstill A', 'B']);
  });

  it('splits sentences, keeping a closing quote with its sentence', () => {
    const out = splitText('He ran. "Why?" she asked! Then silence', 'sentences');
    expect(out.map((c) => c.text)).toEqual(['He ran.', '"Why?"', 'she asked!', 'Then silence']);
  });

  it('wraps over-long pieces at a space before the limit', () => {
    const out = splitText('aaaa bbbb cccc dddd', 'lines', 9);
    expect(out.map((c) => c.text)).toEqual(['aaaa bbbb', 'cccc dddd']);
    expect(out.every((c) => c.text.length <= 9)).toBe(true);
  });

  it('gives every chunk a unique id', () => {
    const ids = splitText('a\nb\nc\nd', 'lines').map((c) => c.id);
    expect(new Set(ids).size).toBe(4);
  });
});

describe('selection', () => {
  const chunks = story(30);

  it('selects a 1-based inclusive range, order-insensitive and clamped', () => {
    expect(idsInRange(chunks, 1, 15)).toHaveLength(15);
    expect(idsInRange(chunks, 15, 1)).toEqual(idsInRange(chunks, 1, 15));
    expect(idsInRange(chunks, 25, 99)).toHaveLength(6);
    expect(idsInRange(chunks, 0, 2)).toHaveLength(2);
    expect(idsInRange([], 1, 5)).toEqual([]);
    expect(idsInRange(chunks, Number.NaN, 5)).toEqual([]);
  });

  it('shift-click covers the rows between anchor and target in either direction', () => {
    expect(idsBetween(chunks, chunks[2].id, chunks[5].id)).toHaveLength(4);
    expect(idsBetween(chunks, chunks[5].id, chunks[2].id)).toHaveLength(4);
    expect(idsBetween(chunks, 'gone', chunks[1].id)).toEqual([chunks[1].id]);
  });

  it('nextBlock moves to the following lines with the same count', () => {
    const first = new Set(idsInRange(chunks, 1, 15));
    expect(nextBlock(chunks, first)).toEqual(idsInRange(chunks, 16, 30));
    const tail = new Set(idsInRange(chunks, 26, 30));
    expect(nextBlock(chunks, tail)).toEqual([]);
    expect(nextBlock(chunks, new Set())).toEqual([]);
  });
});

describe('editing', () => {
  it('applies a voice to exactly the selected chunks and can clear it', () => {
    const chunks = story(5);
    const picked = new Set(idsInRange(chunks, 2, 4));
    const withVoice = applyVoice(chunks, picked, 'sun_2');
    expect(withVoice.map((c) => c.voice ?? null)).toEqual([null, 'sun_2', 'sun_2', 'sun_2', null]);
    expect(applyVoice(withVoice, picked, null).every((c) => !c.voice)).toBe(true);
    expect(chunks.every((c) => !c.voice)).toBe(true); // input not mutated
  });

  it('merges with the next chunk, keeping the first one’s voice', () => {
    const chunks = applyVoice(story(3), new Set([story(1)[0].id]), 'x');
    const base = story(3).map((c, i) => ({ ...c, voice: i === 0 ? 'x' : null }));
    const merged = mergeWithNext(base, base[0].id);
    expect(merged).toHaveLength(2);
    expect(merged[0]).toMatchObject({ text: 'line 1 line 2', voice: 'x' });
    expect(mergeWithNext(base, base[2].id)).toBe(base);
    expect(chunks).toHaveLength(3);
  });

  it('inserts after and removes by id', () => {
    const chunks = story(3);
    const added = insertAfter(chunks, chunks[0].id, makeChunk('new'));
    expect(added.map((c) => c.text)).toEqual(['line 1', 'new', 'line 2', 'line 3']);
    expect(removeChunks(added, new Set([added[1].id])).map((c) => c.text)).toEqual([
      'line 1',
      'line 2',
      'line 3',
    ]);
  });
});

describe('summarize', () => {
  it('counts chunks, characters and unassigned lines, ignoring empty ones', () => {
    const chunks = story(4);
    chunks[3].text = '   ';
    const withVoices = applyVoice(chunks, new Set([chunks[0].id, chunks[1].id]), 'v1');
    const s = summarize(withVoices);
    expect(s.chunks).toBe(3);
    expect(s.chars).toBe(18);
    expect(s.unassigned).toBe(1);
    expect(s.perVoice).toEqual([
      { voice: 'v1', chunks: 2, chars: 12 },
      { voice: null, chunks: 1, chars: 6 },
    ]);
  });

  it('orders voices by first appearance', () => {
    const chunks = story(3).map((c, i) => ({ ...c, voice: ['b', 'a', 'b'][i] }));
    expect(voiceOrder(chunks)).toEqual(['b', 'a']);
  });
});
