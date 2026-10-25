import { describe, expect, it } from 'vitest';
import {
  buildRunRequest,
  countLoads,
  defaultSpeaker,
  loadsFor,
  newTake,
  planOrder,
  recipeOf,
  retarget,
  speakerGroups,
  summaryParts,
  validateTake,
  type QwenModel,
  type QwenProfile,
  type RunSetup,
} from './qwen-model';

const STOCK = ['ryan', 'vivian'];
const EP09: QwenModel = {
  id: 'ep09',
  label: 'Run 2 EP09',
  path: 'p1',
  source: 'checkpoint',
  type: 'custom_voice',
  size: '1.7B',
  speakers: ['ss_narrator'],
  borrowable_speakers: STOCK,
};
const BASE: QwenModel = {
  id: 'base17',
  label: 'Qwen3-TTS 1.7B Base',
  path: 'p2',
  source: 'hf',
  type: 'base',
  size: '1.7B',
  speakers: [],
  borrowable_speakers: STOCK,
};
const STOCK_CV: QwenModel = {
  id: 'cv06',
  label: 'Qwen3-TTS 0.6B CustomVoice',
  path: 'p3',
  source: 'hf',
  type: 'custom_voice',
  size: '0.6B',
  speakers: ['ryan', 'vivian'],
  borrowable_speakers: [],
};
const MODELS = [STOCK_CV, EP09, BASE];
const PROFILES: QwenProfile[] = [{ id: 'p-shadow', name: 'Shadow', ref_text: 'Hello there.', language: 'English', kind: 'clone' }];
const SETUP: RunSetup = { sessionName: ' ', groupByModel: true, warmupRuns: 2, warmupText: 'Warm.' };

const take = (over = {}) => newTake(MODELS, { text: 'A line.', ...over });

describe('speaker rules', () => {
  it('pre-fills a one-speaker model and leaves a multi-speaker model blank', () => {
    expect(defaultSpeaker(EP09)).toBe('ss_narrator');
    expect(defaultSpeaker(STOCK_CV)).toBe('');
    expect(defaultSpeaker(BASE)).toBe('');
  });
  it('new takes prefer the one-speaker custom_voice model', () => {
    const t = newTake(MODELS);
    expect(t.modelId).toBe('ep09');
    expect(t.speaker).toBe('ss_narrator');
  });
  it('groups own speakers first and the borrowable ones apart', () => {
    expect(speakerGroups(EP09)).toEqual({ own: ['ss_narrator'], borrowed: ['ryan', 'vivian'] });
    expect(speakerGroups(STOCK_CV).borrowed).toEqual([]);
  });
  it('retargeting to a multi-speaker model blanks a speaker it does not know, keeps a borrowable one', () => {
    const t = retarget(take(), MODELS, { modelId: 'cv06' });
    expect(t.speaker).toBe('');
    expect(retarget({ ...take(), speaker: 'ryan' }, MODELS, { modelId: 'ep09' }).speaker).toBe('ryan');
    expect(retarget({ ...take(), speaker: 'ss_narrator' }, MODELS, { modelId: 'base17' }).speaker).toBe('');
  });
  it('switching to clone moves to a Base model', () => {
    expect(retarget(take(), MODELS, { kind: 'clone' }).modelId).toBe('base17');
  });
});

describe('recipe names and summary', () => {
  it('Narrator is the own speaker without instruct', () => {
    expect(recipeOf(take(), EP09)).toBe('narrator');
    expect(recipeOf(take({ instruct: 'whisper' }), EP09)).toBe('experiment');
    expect(recipeOf(take({ speaker: 'ryan' }), EP09)).toBe('experiment');
  });
  it('Clone is clip plus text on Base; embedding only and clone on custom are experiments', () => {
    const c = take({ kind: 'clone', modelId: 'base17', profileId: 'p-shadow' });
    expect(recipeOf(c, BASE)).toBe('clone');
    expect(recipeOf({ ...c, xvecOnly: true }, BASE)).toBe('experiment');
    expect(recipeOf({ ...c, kind: 'clone_on_custom' }, EP09)).toBe('experiment');
  });
  it('summarises model, voice and mode', () => {
    expect(summaryParts(take(), EP09, PROFILES)).toEqual({
      modelLabel: 'Run 2 EP09', modelType: 'custom_voice', voice: 'ss_narrator', mode: 'baked_speaker',
    });
    const c = take({ kind: 'clone', modelId: 'base17', profileId: 'p-shadow', xvecOnly: true });
    expect(summaryParts(c, BASE, PROFILES)).toMatchObject({ voice: 'Shadow', mode: 'embedding_only' });
    expect(summaryParts(take({ speaker: 'ryan' }), EP09, PROFILES).mode).toBe('borrowed_speaker');
    expect(summaryParts(take({ kind: 'clone', refAudio: 'C:\\clips\\a.wav' }), BASE, []).voice).toBe('a.wav');
  });
});

describe('validation', () => {
  const codes = (t: ReturnType<typeof take>, m: QwenModel | undefined) => validateTake(t, m, PROFILES).map((i) => i.code);
  it('a plain narrator take is clean', () => {
    expect(codes(take(), EP09)).toEqual([]);
  });
  it('only a missing model or text blocks', () => {
    const issues = validateTake(take({ text: ' ' }), EP09, PROFILES);
    expect(issues.filter((i) => i.blocking).map((i) => i.code)).toEqual(['no_text']);
    expect(codes(take(), undefined)).toEqual(['no_model', ]);
  });
  it('warns on a kind/type mismatch but not for a borrowed voice on Base', () => {
    expect(codes(take({ kind: 'clone' }), EP09)).toContain('pair_mismatch');
    expect(codes(take({ speaker: 'ryan' }), BASE)).toEqual(['borrowed']);
    expect(codes(take({ speaker: '' }), BASE)).toContain('pair_mismatch');
  });
  it('asks for a speaker on a multi-speaker model, not on a one-speaker one', () => {
    expect(codes(take({ modelId: 'cv06', speaker: '' }), STOCK_CV)).toEqual(['pick_speaker']);
    expect(codes(take({ speaker: '' }), EP09)).toEqual([]);
  });
  it('clone needs a clip, and its text unless embedding only', () => {
    const c = take({ kind: 'clone', modelId: 'base17' });
    expect(codes(c, BASE)).toEqual(['no_clip']);
    expect(codes({ ...c, refAudio: 'a.wav' }, BASE)).toEqual(['no_ref_text']);
    expect(codes({ ...c, refAudio: 'a.wav', xvecOnly: true }, BASE)).toEqual([]);
    expect(codes({ ...c, profileId: 'p-shadow' }, BASE)).toEqual([]);
    expect(codes({ ...c, profileId: 'p-shadow', refText: ' ' }, BASE)).toEqual([]);
  });
});

describe('model-load counting', () => {
  const a = take({ modelId: 'ep09' });
  const b = take({ modelId: 'base17', kind: 'clone', profileId: 'p-shadow' });
  const c = take({ modelId: 'ep09' });
  it('grouping merges the same model; the order of first appearance is kept', () => {
    const takes = [a, b, c];
    expect(planOrder(takes, false).map(([i]) => i)).toEqual([0, 1, 2]);
    expect(planOrder(takes, true).map(([i]) => i)).toEqual([0, 2, 1]);
    expect(loadsFor(takes, false)).toBe(3);
    expect(loadsFor(takes, true)).toBe(2);
  });
  it('repeats expand and stay together', () => {
    const takes = [{ ...a, repeat: 3 }, b];
    expect(planOrder(takes, true)).toEqual([[0, 1], [0, 2], [0, 3], [1, 1]]);
    expect(countLoads(takes, planOrder(takes, true))).toBe(2);
  });
  it('a different max_seq_len is a different load', () => {
    const d = { ...a, params: { ...a.params, max_seq_len: 2048 } };
    expect(loadsFor([a, d, a], true)).toBe(2);
    expect(loadsFor([a, d, a], false)).toBe(3);
  });
  it('no takes, no loads', () => {
    expect(loadsFor([], true)).toBe(0);
  });
});

describe('request builder', () => {
  it('narrator: speaker only, no instruct, no seed when random', () => {
    const req = buildRunRequest([take({ label: ' N ' })], SETUP);
    expect(req).toMatchObject({ session_name: 'playground', group_by_model: true, warmup_runs: 2, warmup_text: 'Warm.' });
    const t = req.takes[0];
    expect(t.voice).toEqual({ kind: 'custom_voice', speaker: 'ss_narrator' });
    expect(t.params).toEqual({ language: 'Auto', pad_s: 0.3 });
    expect(t.label).toBe('N');
    expect(t.id).toMatch(/^[A-Za-z0-9_-]{1,40}$/);
  });
  it('sends only the params that were set, and a fixed seed', () => {
    const t = take({ randomSeed: false, seed: 77, instruct: ' calm ', params: { ...take().params, temperature: 0.7, top_k: 40, pad_s: 0.5, max_seq_len: 2048 }, repeat: 3 });
    const out = buildRunRequest([t], SETUP).takes[0];
    expect(out.params).toEqual({ language: 'Auto', pad_s: 0.5, temperature: 0.7, top_k: 40, max_seq_len: 2048, seed: 77 });
    expect(out.voice).toEqual({ kind: 'custom_voice', speaker: 'ss_narrator', instruct: 'calm' });
    expect(out.repeat).toBe(3);
  });
  it('clone: profile id with an edited ref text, or a clip path; xvec_only only when set', () => {
    const c = take({ kind: 'clone', modelId: 'base17', profileId: 'p-shadow', refText: ' edited ' });
    expect(buildRunRequest([c], SETUP).takes[0].voice).toEqual({ kind: 'clone', profile_id: 'p-shadow', ref_text: 'edited' });
    const f = take({ kind: 'clone', modelId: 'base17', refAudio: ' C:\\a.wav ', refText: 'Hi', xvecOnly: true });
    expect(buildRunRequest([f], SETUP).takes[0].voice).toEqual({ kind: 'clone', ref_audio: 'C:\\a.wav', ref_text: 'Hi', xvec_only: true });
  });
  it('clone on custom voice keeps the speaker and has no xvec_only', () => {
    const c = take({ kind: 'clone_on_custom', profileId: 'p-shadow', xvecOnly: true });
    expect(buildRunRequest([c], SETUP).takes[0].voice).toEqual({ kind: 'clone_on_custom', speaker: 'ss_narrator', profile_id: 'p-shadow' });
  });
});
