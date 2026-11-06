import { cleanup, fireEvent, render, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';

const mock = vi.hoisted(() => ({ api: vi.fn() }));

vi.mock('@/lib/api/client', async (load) => {
  const actual = await load<typeof import('@/lib/api/client')>();
  return { ...actual, apiJson: mock.api };
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
