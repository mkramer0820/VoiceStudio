import { cleanup, fireEvent, render, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';

const mock = vi.hoisted(() => ({ api: vi.fn(), fetch: vi.fn() }));

vi.mock('@/lib/api/client', async (load) => {
  const actual = await load<typeof import('@/lib/api/client')>();
  return { ...actual, apiJson: mock.api, apiFetch: mock.fetch };
});
vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (key: string, o?: { defaultValue?: string }) => o?.defaultValue ?? key }),
}));

import { EffectsDialog } from './effects-dialog';
import { resetEffectsForTest } from './effects-model';

const BUILT_IN = {
  id: 'inner_monologue',
  label: 'Inner monologue',
  builtin: true,
  kind: 'graph',
  graph: '[0:a]...[out]',
  aliases: ['internal'],
  codec: 'pcm_s24le',
};

beforeEach(() => {
  resetEffectsForTest();
  URL.createObjectURL = vi.fn(() => 'blob:fake');
  URL.revokeObjectURL = vi.fn();
  mock.fetch.mockResolvedValue({ blob: () => Promise.resolve(new Blob(['RIFFxxxx'], { type: 'audio/wav' })) });
  mock.api.mockImplementation((path: string, init?: { method?: string }) => {
    if (path === '/api/qwen/effects' && !init?.method) return Promise.resolve({ effects: [BUILT_IN] });
    if (path === '/api/qwen/effects/test') return Promise.resolve({ ok: true, channels: 2, error: null });
    return Promise.resolve({ id: 'whisper_room' });
  });
});
afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

it('tests an unsaved effect and then saves it with its tags split into aliases', async () => {
  render(<EffectsDialog open onOpenChange={() => undefined} />);
  await waitFor(() => expect(document.body.textContent).toContain('{internal}'));
  // open the blank form with the New effect button
  fireEvent.click([...document.querySelectorAll('button')].find((b) => b.textContent === 'qwenStory.effects_new')!);
  const [name, aliases] = [...document.querySelectorAll<HTMLInputElement>('input')];
  fireEvent.change(name, { target: { value: 'Whisper room' } });
  fireEvent.change(aliases, { target: { value: 'whisper, quiet ' } });
  fireEvent.change(document.querySelector('textarea')!, { target: { value: 'volume=-6dB,aecho=0.8:0.5:30:0.2' } });
  fireEvent.click([...document.querySelectorAll('button')].find((b) => b.textContent === 'qwenStory.effects_test')!);
  await waitFor(() => expect(document.body.textContent).toContain('qwenStory.effects_test_ok'));
  fireEvent.click([...document.querySelectorAll('button')].find((b) => b.textContent === 'common.save')!);
  await waitFor(() =>
    expect(mock.api).toHaveBeenCalledWith('/api/qwen/effects', {
      method: 'POST',
      body: JSON.stringify({
        id: null,
        label: 'Whisper room',
        kind: 'chain',
        graph: 'volume=-6dB,aecho=0.8:0.5:30:0.2',
        codec: 'pcm_s16le',
        aliases: ['whisper', 'quiet'],
      }),
    }),
  );
});

it('shows ffmpeg\'s reason when a recipe does not work', async () => {
  mock.api.mockImplementation((path: string, init?: { method?: string }) => {
    if (path === '/api/qwen/effects' && !init?.method) return Promise.resolve({ effects: [BUILT_IN] });
    return Promise.resolve({ ok: false, channels: null, error: 'No such filter: nosuchfilter' });
  });
  render(<EffectsDialog open onOpenChange={() => undefined} />);
  fireEvent.click([...document.querySelectorAll('button')].find((b) => b.textContent === 'qwenStory.effects_new')!);
  fireEvent.change(document.querySelector('textarea')!, { target: { value: 'nosuchfilter' } });
  fireEvent.click([...document.querySelectorAll('button')].find((b) => b.textContent === 'qwenStory.effects_test')!);
  await waitFor(() => expect(document.body.textContent).toContain('No such filter: nosuchfilter'));
});

it('edits a built-in effect (stored under its own id) and offers a reset only once it was changed', async () => {
  render(<EffectsDialog open onOpenChange={() => undefined} />);
  await waitFor(() => expect(document.body.textContent).toContain('{internal}'));
  fireEvent.click([...document.querySelectorAll('button')].find((b) => b.textContent === 'qwenStory.effects_edit')!);
  expect([...document.querySelectorAll('button')].some((b) => b.textContent === 'qwenStory.effects_reset')).toBe(false); // untouched shipped recipe
  fireEvent.change(document.querySelector('textarea')!, { target: { value: '[0:a]volume=0.5[out]' } });
  fireEvent.click([...document.querySelectorAll('button')].find((b) => b.textContent === 'common.save')!);
  await waitFor(() => expect(mock.api).toHaveBeenCalledWith('/api/qwen/effects/inner_monologue', expect.objectContaining({ method: 'PUT' })));
});

it('plays the original and the processed audio side by side after Hear it', async () => {
  render(<EffectsDialog open onOpenChange={() => undefined} />);
  fireEvent.click([...document.querySelectorAll('button')].find((b) => b.textContent === 'qwenStory.effects_new')!);
  fireEvent.change(document.querySelector('textarea')!, { target: { value: 'volume=-6dB' } });
  const hearButton = [...document.querySelectorAll('button')].find((b) => b.textContent === 'qwenStory.effects_hear')!;
  expect(hearButton).toBeDisabled(); // needs a clip first
  const file = new File(['RIFFxxxx'], 'line.wav', { type: 'audio/wav' });
  fireEvent.change(document.querySelector('input[type="file"]')!, { target: { files: [file] } });
  await waitFor(() => expect(hearButton).not.toBeDisabled());
  fireEvent.click(hearButton);
  await waitFor(() => expect(mock.fetch).toHaveBeenCalled());
  console.log(document.body.textContent?.slice(-300), mock.fetch.mock.calls.length);
  await waitFor(() => expect(document.querySelectorAll('audio')).toHaveLength(2));
  const [path, init] = mock.fetch.mock.calls[0];
  expect(path).toBe('/api/qwen/effects/preview');
  const sent = (init as { body: FormData }).body;
  expect((sent.get('file') as File).name).toBe('line.wav');
  expect(JSON.parse(sent.get('definition') as string)).toMatchObject({ graph: 'volume=-6dB', kind: 'chain' }); // the unsaved text of the form
});

it('shows Reset for a built-in that was changed', async () => {
  mock.api.mockImplementation((path: string, init?: { method?: string }) => {
    if (path === '/api/qwen/effects' && !init?.method) return Promise.resolve({ effects: [{ ...BUILT_IN, modified: true, label: 'My inner voice' }] });
    return Promise.resolve({ ok: true });
  });
  render(<EffectsDialog open onOpenChange={() => undefined} />);
  await waitFor(() => expect(document.body.textContent).toContain('My inner voice'));
  fireEvent.click([...document.querySelectorAll('button')].find((b) => b.textContent === 'qwenStory.effects_edit')!);
  fireEvent.click([...document.querySelectorAll('button')].find((b) => b.textContent === 'qwenStory.effects_reset')!);
  await waitFor(() => expect(mock.api).toHaveBeenCalledWith('/api/qwen/effects/inner_monologue', { method: 'DELETE' }));
});

it('tells the user to restart when the backend does not know the preview request (Method Not Allowed)', async () => {
  const { ApiError } = await import('@/lib/api/client');
  mock.fetch.mockRejectedValue(new ApiError(405, 'Method Not Allowed'));
  render(<EffectsDialog open onOpenChange={() => undefined} />);
  fireEvent.click([...document.querySelectorAll('button')].find((b) => b.textContent === 'qwenStory.effects_new')!);
  fireEvent.change(document.querySelector('textarea')!, { target: { value: 'volume=-6dB' } });
  fireEvent.change(document.querySelector('input[type="file"]')!, { target: { files: [new File(['RIFFxxxx'], 'line.wav', { type: 'audio/wav' })] } });
  const hearButton = [...document.querySelectorAll('button')].find((b) => b.textContent === 'qwenStory.effects_hear')!;
  await waitFor(() => expect(hearButton).not.toBeDisabled());
  fireEvent.click(hearButton);
  await waitFor(() => expect(document.body.textContent).toContain('qwenStory.effects_backend_old'));
  expect(document.body.textContent).not.toContain('Method Not Allowed');
});
