import { describe, expect, it } from 'vitest';
import type { QwenModel } from '../qwen-playground/qwen-model';
import {
  blankForm,
  applyEmotion,
  applySpeaker,
  characterRows,
  clearEmotions,
  countMatches,
  emotionCounts,
  findReplace,
  idsBetween,
  idsOfSpeaker,
  renameCharacter,
  DEFAULT_STORY_SETTINGS,
  incompleteChunkIds,
  mappingWarnings,
  modelLoads,
  modelsForKind,
  profileBody,
  profileProblem,
  unmappedCharacters,
  type Draft,
  type Profile,
  type StoryChunk,
  type StoryJob,
} from './story-model';

const models = [
  { id: 'ep09', label: 'EP09', path: 'p', source: 's', type: 'custom_voice', size: '1.7B', speakers: ['ss_narrator'] },
  { id: 'base', label: 'Base', path: 'p2', source: 's', type: 'base', size: '1.7B', speakers: [] },
] as QwenModel[];

const chunk = (id: string, speaker: string | null, text = 'Hello.'): StoryChunk => ({ id, text, speaker, emotion: null });
const profile = (id: string, model_id: string, over: Partial<Profile> = {}): Profile => ({
  id, name: id, kind: 'narrator', model_id, speaker: 'ss_narrator', ref_audio: null, ref_text: null, xvec_only: false,
  default_instruct: null, emotion: { preset: 'neutral', strength: 'normal' }, temperature: null, top_k: null, top_p: null,
  repetition_penalty: null, notes: '', ...over,
});
const draft = (over: Partial<Draft> = {}): Draft => ({
  title: 't', chunks: [chunk('a', null), chunk('b', 'Shadow'), chunk('c', 'Shadow')], narrator_profile_id: 'n', cast: {}, emotions: {},
  settings: DEFAULT_STORY_SETTINGS, ...over,
});

describe('profile form', () => {
  it('sends null, never an invented default, for empty sampling fields', () => {
    const body = profileBody({ ...blankForm(models), name: ' Narr ', top_p: '0.8' });
    expect(body).toMatchObject({ name: 'Narr', temperature: null, top_k: null, top_p: 0.8, repetition_penalty: null, ref_audio: null });
  });

  it('keeps clip fields only for clones', () => {
    const form = { ...blankForm(models), kind: 'clone' as const, ref_audio: 'a.wav', ref_text: 'x', speaker: 'ss_narrator' };
    expect(profileBody(form)).toMatchObject({ speaker: null, ref_audio: 'a.wav', ref_text: 'x' });
  });

  it('names what is missing', () => {
    const base = blankForm(models);
    expect(profileProblem(base)).toBe('qwenStory.problem_name');
    expect(profileProblem({ ...base, name: 'x' })).toBeNull();
    const clone = { ...base, name: 'x', kind: 'clone' as const };
    expect(profileProblem(clone)).toBe('qwenStory.problem_clip');
    expect(profileProblem({ ...clone, ref_audio: 'a.wav' })).toBe('qwenStory.problem_text');
    expect(profileProblem({ ...clone, ref_audio: 'a.wav', xvec_only: true })).toBeNull();
    expect(profileProblem(clone, true)).toBe('qwenStory.problem_text'); // a sample picked in the dialog counts as a clip
  });

  it('offers Base models to clones and custom voice models to speakers', () => {
    expect(modelsForKind('clone', models).map((m) => m.id)).toEqual(['base']);
    expect(modelsForKind('narrator', models).map((m) => m.id)).toEqual(['ep09']);
    expect(blankForm(models)).toMatchObject({ model_id: 'ep09', speaker: 'ss_narrator' });
  });
});

describe('characters and mapping', () => {
  it('lists narration first, then characters in order', () => {
    expect(characterRows(draft().chunks).map((r) => [r.name, r.chunks])).toEqual([[null, 1], ['Shadow', 2]]);
    expect(unmappedCharacters(draft())).toEqual(['Shadow']);
  });

  it('warns about an unmapped character and a missing narrator, never blocks', () => {
    const keys = mappingWarnings(draft({ narrator_profile_id: null }), [], models).map((w) => w.key);
    expect(keys).toEqual(['qwenStory.warn_no_narrator', 'qwenStory.warn_unmapped']);
  });

  it('warns about a deleted profile and a model that is gone', () => {
    const w = mappingWarnings(draft({ cast: { Shadow: 'gone' } }), [profile('n', 'old-model')], models);
    expect(w.map((x) => x.key)).toEqual(['qwenStory.warn_model_missing', 'qwenStory.warn_profile_gone']);
  });

  it('counts model loads: profiles on one model never add a load', () => {
    const profiles = [profile('n', 'ep09'), profile('s', 'ep09'), profile('c', 'base', { kind: 'clone' })];
    expect(modelLoads(draft({ cast: { Shadow: 's' } }), profiles)).toBe(1);
    expect(modelLoads(draft({ cast: { Shadow: 'c' } }), profiles)).toBe(2);
  });
});

describe('render job', () => {
  it('picks failed and queued chunks for the retry', () => {
    const job = { chunks: [{ id: 'a', status: 'done' }, { id: 'b', status: 'error' }, { id: 'c', status: 'queued' }] } as StoryJob;
    expect(incompleteChunkIds(job)).toEqual(['b', 'c']);
    expect(incompleteChunkIds(null)).toEqual([]);
  });
});

describe('bulk editing', () => {
  const chunks = [chunk('a', null, 'Night in Rome.'), chunk('b', 'Shadow', 'Hello.'), chunk('c', 'Shadow', 'Rome again.'), chunk('d', 'Mara', 'Hi.')];

  it('selects a range and all lines of a character', () => {
    expect(idsBetween(chunks, 'b', 'd')).toEqual(['b', 'c', 'd']);
    expect(idsBetween(chunks, 'd', 'b')).toEqual(['b', 'c', 'd']);
    expect(idsOfSpeaker(chunks, 'Shadow')).toEqual(['b', 'c']);
    expect(idsOfSpeaker(chunks, null)).toEqual(['a']);
  });

  it('applies an emotion or a speaker to the selection only, and clears it', () => {
    const sad = { preset: 'sad', strength: 'strong' as const };
    const out = applyEmotion(chunks, new Set(['b', 'c']), sad);
    expect(out.map((c) => c.emotion?.preset ?? null)).toEqual([null, 'sad', 'sad', null]);
    expect(emotionCounts(out)).toEqual([{ preset: 'sad', count: 2 }]);
    expect(clearEmotions(out, 'joy').filter((c) => c.emotion)).toHaveLength(2);
    expect(clearEmotions(out).filter((c) => c.emotion)).toHaveLength(0);
    expect(applySpeaker(chunks, new Set(['a']), 'Mara')[0].speaker).toBe('Mara');
  });

  it('renames or merges a character everywhere, keeping the existing mapping of the target', () => {
    const d = draft({ chunks, cast: { Shadow: 'p1', Mara: 'p2' }, emotions: { Shadow: { preset: 'sad', strength: 'normal' } } });
    const renamed = renameCharacter(d, 'Shadow', 'Ghost');
    expect(renamed.cast).toEqual({ Ghost: 'p1', Mara: 'p2' });
    expect(Object.keys(renamed.emotions)).toEqual(['Ghost']);
    expect(renamed.chunks.filter((c) => c.speaker === 'Ghost')).toHaveLength(2);
    expect(renameCharacter(d, 'Shadow', 'Mara').cast).toEqual({ Mara: 'p2' });
    expect(renameCharacter(d, 'Shadow', '  ')).toBe(d);
  });

  it('finds and replaces plain text, not patterns', () => {
    expect(countMatches(chunks, 'rome', false)).toEqual({ matches: 2, lines: 2 });
    expect(countMatches(chunks, 'rome', true)).toEqual({ matches: 0, lines: 0 });
    expect(findReplace(chunks, 'Rome', 'Paris', true).map((c) => c.text)).toContain('Night in Paris.');
    expect(countMatches(chunks, '.*', false).matches).toBe(0);
  });
});
