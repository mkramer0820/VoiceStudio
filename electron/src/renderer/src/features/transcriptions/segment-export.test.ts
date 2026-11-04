import { describe, expect, it } from 'vitest';
import { formatSegmentsJson, formatSrt, formatTimedText, formatVtt, timedSegments } from '@shared/utils/transcriptionFormat';

const entry = {
  id: 1,
  text: 'Hello there. General Kenobi.',
  language: 'en',
  duration_s: 7.5,
  timestamp: '2026-11-03T10:00:00Z',
  segments: [
    { start: 0, end: 2.5, text: ' Hello there.' },
    { start: 3.25, end: null, text: 'General Kenobi.' },
    { start: null, end: null, text: 'no timing' },
  ],
};

describe('segment exports', () => {
  it('writes SRT with comma milliseconds and fills a missing end from the next cue or a reading time', () => {
    expect(formatSrt(entry)).toBe(
      '1\n00:00:00,000 --> 00:00:02,500\nHello there.\n\n2\n00:00:03,250 --> 00:00:04,250\nGeneral Kenobi.\n',
    );
  });

  it('writes WebVTT with a header and dot milliseconds', () => {
    const vtt = formatVtt(entry);
    expect(vtt.startsWith('WEBVTT\n\n00:00:00.000 --> 00:00:02.500\nHello there.')).toBe(true);
    expect(vtt).toContain('00:00:03.250 --> 00:00:04.250');
  });

  it('writes timed text and JSON that keeps untimed segments as null', () => {
    expect(formatTimedText(entry)).toBe('[00:00:00] Hello there.\n[00:00:03] General Kenobi.');
    const json = JSON.parse(formatSegmentsJson(entry));
    expect(json.segments).toHaveLength(3);
    expect(json.segments[1]).toEqual({ start: 3.25, end: null, text: 'General Kenobi.' });
    expect(json.segments[2].start).toBeNull();
    expect(json.language).toBe('en');
  });

  it('exports nothing timed for a transcript without timings instead of inventing them', () => {
    const plain = { ...entry, segments: [{ start: null, end: null, text: 'x' }] };
    expect(timedSegments(plain)).toEqual([]);
    expect(formatSrt(plain)).toBe('');
    expect(formatVtt(plain)).toBe('');
  });
});
