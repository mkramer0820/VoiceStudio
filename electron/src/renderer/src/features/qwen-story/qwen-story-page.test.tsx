import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';

const mock = vi.hoisted(() => ({ api: vi.fn(), fetch: vi.fn() }));

vi.mock('@/lib/api/client', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/lib/api/client')>()),
  apiJson: mock.api,
  apiFetch: mock.fetch,
}));
vi.mock('@/components/app-shell/workspace-header', () => ({
  WorkspaceHeader: ({ children }: { children: React.ReactNode }) => <header>{children}</header>,
}));
vi.mock('@tanstack/react-router', () => ({
  Link: ({ children }: { children: React.ReactNode }) => <a href="#">{children}</a>,
}));
vi.mock('@/features/clone/reference-input', () => ({
  ReferenceSourcePicker: ({ onAccept }: { onAccept?: (file: File, seconds: number | null) => void }) => (
    <button type="button" onClick={() => onAccept?.(new File([new Uint8Array(2000)], 'sample.wav', { type: 'audio/wav' }), 8)}>
      pick-sample
    </button>
  ),
}));
vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, values?: Record<string, unknown>) => (values ? `${key} ${JSON.stringify(values)}` : key),
  }),
}));

import { QwenStoryPage } from './qwen-story-page';

const STATE = {
  settings: {},
  engine: { available: true, reason: '' },
  models: [
    { id: 'ep09', label: 'EP09', path: 'p1', source: 's', type: 'custom_voice', size: '1.7B', speakers: ['ss_narrator'], borrowable_speakers: [] },
    { id: 'base17', label: 'Stock Base', path: 'p2', source: 's', type: 'base', size: '1.7B', speakers: [], borrowable_speakers: [] },
  ],
};
const CATALOG = {
  presets: ['neutral', 'sad', 'joy', 'custom'].map((name) => ({ name, texts: { normal: '', strong: '', very_strong: '' } })),
  strengths: ['normal', 'strong', 'very_strong'],
  unverified_on_narrator: 'UNVERIFIED on the narrator',
  steer_message: 'not available yet',
};
const PROFILE = {
  id: 'aaaaaaaaaaaa', name: 'Narrator', kind: 'narrator', model_id: 'ep09', speaker: 'ss_narrator', ref_audio: null, ref_text: null,
  xvec_only: false, default_instruct: null, emotion: { preset: 'neutral', strength: 'normal' }, temperature: null, top_k: null,
  top_p: null, repetition_penalty: null, notes: '', has_embedding: false, embedding_state: 'none',
};
const CLONE = {
  ...PROFILE, id: 'dddddddddddd', name: 'Shadow', kind: 'clone', model_id: 'base17', speaker: null, ref_audio: 'C:/data/clips/x.wav',
  ref_text: 'Hello.', has_embedding: true, embedding_state: 'ready',
};
const CHUNKS = [
  { id: 'c0001', text: 'Night fell.', speaker: null, emotion: null },
  { id: 'c0002', text: 'Hello there.', speaker: 'Shadow', emotion: null },
];
const JOB = {
  id: 'bbbbbbbbbbbb', story: 'untitled-story', run: 'run-001_20261028-120000_bbbbbbbbbbbb', folder: 'C:/out/untitled-story/run-001', status: 'done',
  message: '', total: 2, done: 2, warnings: [], joined: 'story.wav',
  chunks: [
    { id: 'c0001', index: 1, status: 'done', file: '001_narration_Narrator.wav', history: ['001_narration_Narrator.wav'], speaker: null, seconds: 1.5 },
    { id: 'c0002', index: 2, status: 'error', file: null, history: [], speaker: 'Shadow', error: 'boom' },
  ],
};

let calls: Array<[string, string, unknown]> = [];

beforeEach(() => {
  calls = [];
  URL.createObjectURL = vi.fn(() => 'blob:audio');
  mock.fetch.mockResolvedValue({ blob: () => Promise.resolve(new Blob(['x'])), json: () => Promise.resolve({ path: 'C:/data/clips/new.wav' }) });
  mock.api.mockImplementation((path: string, init?: RequestInit) => {
    const body = init?.body && typeof init.body === 'string' ? JSON.parse(init.body) : init?.body;
    calls.push([path, init?.method ?? 'GET', body]);
    if (path === '/api/qwen/state') return Promise.resolve(STATE);
    if (path === '/api/qwen/emotions') return Promise.resolve(CATALOG);
    if (path === '/api/qwen/profiles') return Promise.resolve(init?.method === 'POST' ? PROFILE : { profiles: [PROFILE, CLONE] });
    if (path === '/api/qwen/story/drafts') return Promise.resolve(init?.method === 'POST' ? { id: 'cccccccccccc' } : { stories: [] });
    if (path.startsWith('/api/qwen/story/drafts/')) return Promise.resolve({ id: 'cccccccccccc' });
    if (path === '/transcribe') return Promise.resolve({ text: 'The words in the sample.' });
    if (path === '/api/qwen/story/parse') return Promise.resolve({ chunks: CHUNKS });
    if (path === '/api/qwen/story/render') return Promise.resolve(JOB);
    return Promise.resolve({});
  });
});
afterEach(() => cleanup());

async function split() {
  render(<QwenStoryPage />);
  await screen.findByText('qwenStory.engine_ready');
  fireEvent.change(screen.getByPlaceholderText('qwenStory.script_placeholder'), { target: { value: 'Night fell.\n[Shadow] Hello there.' } });
  fireEvent.click(screen.getByText('qwenStory.split'));
  await screen.findByLabelText('qwenStory.text_of {"n":1}');
}

it('splits a script into chunks and lists the characters with a warning for the unmapped one', async () => {
  await split();
  expect(calls.some(([p, m]) => p === '/api/qwen/story/parse' && m === 'POST')).toBe(true);
  expect(screen.getByDisplayValue('Hello there.')).toBeTruthy();
  expect(screen.getByLabelText('qwenStory.profile_for {"name":"Shadow"}')).toBeTruthy();
  expect(screen.getByText('qwenStory.warn_unmapped {"name":"Shadow"}')).toBeTruthy();
  expect(screen.getByText('qwenStory.warn_no_narrator {}')).toBeTruthy();
});

it('saves the story with the mapping, then renders it and shows per-chunk results', async () => {
  await split();
  fireEvent.change(screen.getByLabelText('qwenStory.profile_for {"name":"qwenStory.narration"}'), { target: { value: PROFILE.id } });
  fireEvent.change(screen.getByLabelText('qwenStory.profile_for {"name":"Shadow"}'), { target: { value: PROFILE.id } });
  fireEvent.click(screen.getByText('qwenStory.render_all'));
  await waitFor(() => expect(calls.some(([p]) => p === '/api/qwen/story/render')).toBe(true));
  const saved = calls.find(([p, m]) => p === '/api/qwen/story/drafts' && m === 'POST')?.[2] as { cast: Record<string, string>; narrator_profile_id: string };
  expect(saved.cast).toEqual({ Shadow: PROFILE.id });
  expect(saved.narrator_profile_id).toBe(PROFILE.id);
  const sent = calls.find(([p]) => p === '/api/qwen/story/render')?.[2] as { chunk_ids: unknown; story_id: string };
  expect(sent.chunk_ids).toBeNull();
  expect(sent.story_id).toBe('cccccccccccc');
  expect(await screen.findByText('qwenStory.job_done {"done":2,"total":2}')).toBeTruthy();
  expect(screen.getByText('qwenStory.chunk_failed {"error":"boom"}')).toBeTruthy();
});

it('re-renders one chunk into the same run folder', async () => {
  await split();
  fireEvent.change(screen.getByLabelText('qwenStory.profile_for {"name":"qwenStory.narration"}'), { target: { value: PROFILE.id } });
  fireEvent.click(screen.getByText('qwenStory.render_all'));
  await screen.findByText('qwenStory.job_done {"done":2,"total":2}');
  fireEvent.click(screen.getByLabelText('qwenStory.rerender_chunk {"n":2}'));
  await waitFor(() => expect(calls.filter(([p]) => p === '/api/qwen/story/render')).toHaveLength(2));
  const second = calls.filter(([p]) => p === '/api/qwen/story/render')[1][2] as { chunk_ids: string[]; run: string };
  expect(second.chunk_ids).toEqual(['c0002']);
  expect(second.run).toBe(JOB.run);
});

it('shows the profile library with embedding import and export, and the unverified-instruct warning', async () => {
  render(<QwenStoryPage />);
  expect(await screen.findByText('Narrator')).toBeTruthy();
  expect(screen.getByText('UNVERIFIED on the narrator')).toBeTruthy();
  // The .bin controls are not in the default UI: only under Advanced, and only for clone profiles.
  expect(screen.queryByLabelText('qwenStory.embedding_import_for {"name":"Shadow"}')).toBeNull();
  fireEvent.click(screen.getByText('qwenStory.advanced'));
  expect(screen.getByLabelText('qwenStory.embedding_export_for {"name":"Shadow"}')).toBeEnabled();
  const file = new File([new Uint8Array(8192)], 'v.bin');
  fireEvent.click(screen.getByLabelText('qwenStory.embedding_import_for {"name":"Shadow"}'));
  fireEvent.change(screen.getByTestId('embedding-file'), { target: { files: [file] } });
  await waitFor(() => expect(calls.some(([p, m]) => p === `/api/qwen/profiles/${CLONE.id}/embedding` && m === 'PUT')).toBe(true));
});

it('creates a profile with empty sampling fields sent as null', async () => {
  render(<QwenStoryPage />);
  await screen.findByText('Narrator');
  fireEvent.click(screen.getByText('qwenStory.profile_new'));
  fireEvent.change(await screen.findByLabelText('qwenStory.profile_name'), { target: { value: 'Mara' } });
  fireEvent.click(screen.getByText('qwenStory.profile_save'));
  await waitFor(() => expect(calls.some(([p, m]) => p === '/api/qwen/profiles' && m === 'POST')).toBe(true));
  const body = calls.find(([p, m]) => p === '/api/qwen/profiles' && m === 'POST')?.[2] as Record<string, unknown>;
  expect(body).toMatchObject({ name: 'Mara', kind: 'narrator', model_id: 'ep09', speaker: 'ss_narrator', temperature: null, top_k: null });
});

it('saving a clone voice: pick a sample, the words are filled in, one Save voice stores the clip and the profile', async () => {
  render(<QwenStoryPage />);
  await screen.findByText('Narrator');
  fireEvent.click(screen.getByText('qwenStory.profile_new'));
  fireEvent.change(await screen.findByLabelText('qwenStory.profile_name'), { target: { value: 'Mara' } });
  fireEvent.change(screen.getByLabelText('qwenStory.profile_kind'), { target: { value: 'clone' } });
  fireEvent.click(await screen.findByText('pick-sample'));
  expect(await screen.findByDisplayValue('The words in the sample.')).toBeTruthy();
  // nothing about embeddings in the default dialog
  expect(screen.queryByText('qwenStory.profile_xvec_only')).toBeNull();
  fireEvent.click(screen.getByText('qwenStory.save_voice'));
  await waitFor(() => expect(calls.some(([p, m]) => p === '/api/qwen/profiles' && m === 'POST')).toBe(true));
  expect(mock.fetch.mock.calls.some(([path]) => String(path).includes('/api/qwen/clips?name=sample.wav'))).toBe(true);
  const body = calls.find(([p, m]) => p === '/api/qwen/profiles' && m === 'POST')?.[2] as Record<string, unknown>;
  expect(body).toMatchObject({ name: 'Mara', kind: 'clone', model_id: 'base17', ref_audio: 'C:/data/clips/new.wav', ref_text: 'The words in the sample.', xvec_only: false });
});

it('shows a voice being prepared, then a failure with a retry, without the user touching a file', async () => {
  const state = { value: 'preparing' };
  mock.api.mockImplementation((path: string, init?: RequestInit) => {
    if (path === '/api/qwen/state') return Promise.resolve(STATE);
    if (path === '/api/qwen/emotions') return Promise.resolve(CATALOG);
    if (path === '/api/qwen/profiles' && !init?.method) return Promise.resolve({ profiles: [{ ...CLONE, has_embedding: false, embedding_state: state.value }] });
    if (path === '/api/qwen/story/drafts') return Promise.resolve({ stories: [] });
    return Promise.resolve({});
  });
  render(<QwenStoryPage />);
  expect(await screen.findByText('qwenStory.voice_preparing')).toBeTruthy();
  state.value = 'failed';
  expect(await screen.findByLabelText('qwenStory.voice_retry', {}, { timeout: 4000 })).toBeTruthy();
});

it('applies an emotion to the ticked chunks and selects all lines of a character', async () => {
  await split();
  fireEvent.click(screen.getByLabelText('qwenStory.select_lines_of {"name":"Shadow"}'));
  expect((screen.getByLabelText('qwenStory.select_chunk {"n":2}') as HTMLInputElement).checked).toBe(true);
  expect((screen.getByLabelText('qwenStory.select_chunk {"n":1}') as HTMLInputElement).checked).toBe(false);
  fireEvent.change(screen.getByLabelText('qwenStory.bulk_emotion'), { target: { value: 'sad' } });
  fireEvent.click(screen.getByText('qwenStory.bulk_apply_emotion'));
  expect((screen.getByLabelText('qwenStory.emotion_of {"n":1}') as HTMLSelectElement).value).toBe('');
  expect((screen.getByLabelText('qwenStory.emotion_of {"n":2}') as HTMLSelectElement).value).toBe('sad');
});

it('renames a character from the tags dialog in the lines and the table', async () => {
  await split();
  fireEvent.click(screen.getByText('qwenStory.tags_open'));
  fireEvent.change(await screen.findByLabelText('qwenStory.tags_rename {"name":"Shadow"}'), { target: { value: 'Mara' } });
  fireEvent.click(screen.getByText('qwenStory.tags_rename_apply'));
  await waitFor(() => expect(screen.getByLabelText('qwenStory.profile_for {"name":"Mara"}')).toBeTruthy());
  expect((screen.getByLabelText('qwenStory.speaker_of {"n":2}') as HTMLSelectElement).value).toBe('Mara');
});
