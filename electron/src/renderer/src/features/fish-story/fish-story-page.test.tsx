import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';

const mock = vi.hoisted(() => ({ api: vi.fn() }));

vi.mock('@/lib/api/client', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/lib/api/client')>()),
  apiJson: mock.api,
}));
vi.mock('@/components/app-shell/workspace-header', () => ({
  WorkspaceHeader: ({ children }: { children: React.ReactNode }) => <header>{children}</header>,
}));
vi.mock('@/components/workspace-sidebar', () => ({
  SecondarySidebar: ({ children }: { children: React.ReactNode }) => <aside>{children}</aside>,
}));
vi.mock('@tanstack/react-router', () => ({
  Link: ({ children }: { children: React.ReactNode }) => <a href="#">{children}</a>,
}));
vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, values?: Record<string, unknown>) =>
      values ? `${key} ${JSON.stringify(values)}` : key,
  }),
}));

import { FishStoryPage } from './fish-story-page';

const DEFAULTS = {
  model: 's2.1-pro',
  format: 'wav',
  sample_rate: 44100,
  mp3_bitrate: 192,
  latency: 'normal',
  normalize: true,
  temperature: null,
  top_p: null,
  chunk_length: null,
  max_concurrency: 2,
  volume: 0,
  normalize_loudness: true,
  language: null,
  reference_id: null,
};
const VOICES = {
  voices: [
    { id: 'v-sun', title: 'sun_2' },
    { id: 'v-narr', title: 'Narrator' },
  ],
};
const PLAN = {
  chunks: 20,
  empty: 0,
  chars: 123,
  bytes: 123,
  unassigned: 5,
  voices: { 'v-sun': { chunks: 15, chars: 90 } },
  warnings: [],
  model: 's2.1-pro',
  format: 'wav',
  sample_rate: 44100,
  max_concurrency: 2,
};

const calls = () => mock.api.mock.calls.map(([path, init]) => [path, init?.method ?? 'GET']);

beforeEach(() => {
  mock.api.mockImplementation((path: string, init?: RequestInit) => {
    if (path === '/api/fish/story/drafts' && !init?.method) return Promise.resolve({ drafts: [] });
    if (path === '/api/fish/story/defaults') return Promise.resolve(DEFAULTS);
    if (path === '/api/fish/story/drafts') return Promise.resolve({ id: 'd1' });
    if (path.startsWith('/api/fish/story/drafts/')) return Promise.resolve({ id: 'd1' });
    if (path === '/api/fish/voices?scope=mine') return Promise.resolve(VOICES);
    if (path === '/api/fish/story/plan') return Promise.resolve(PLAN);
    if (path === '/api/fish/story/render')
      return Promise.resolve({
        id: 'j1',
        status: 'running',
        message: '',
        out_dir: 'C:/out',
        joined: null,
        total: 20,
        done: 0,
        failed: 0,
        chunks: [],
      });
    return Promise.resolve({});
  });
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

const twentyLines = Array.from({ length: 20 }, (_, i) => `Line number ${i + 1}.`).join('\n');

async function loadStory() {
  render(<FishStoryPage />);
  fireEvent.change(await screen.findByLabelText('fishStory.paste'), { target: { value: twentyLines } });
  fireEvent.click(screen.getByRole('button', { name: 'fishStory.append' }));
  await screen.findAllByTestId('story-row');
}

it('selects lines 1-15, assigns a voice to just those, and sends nothing to Fish', async () => {
  await loadStory();
  expect(screen.getAllByTestId('story-row')).toHaveLength(20);

  fireEvent.click(screen.getByRole('button', { name: 'fishStory.load_voices' }));
  await waitFor(() => expect(calls()).toContainEqual(['/api/fish/voices?scope=mine', 'GET']));
  await screen.findAllByRole('option', { name: 'sun_2' });

  fireEvent.click(screen.getByRole('button', { name: 'fishStory.select' })); // defaults 1 to 15
  expect(screen.getAllByRole('checkbox').filter((c) => (c as HTMLInputElement).checked)).toHaveLength(15);

  fireEvent.change(screen.getByLabelText('fishStory.voice'), { target: { value: 'v-sun' } });
  fireEvent.click(screen.getByRole('button', { name: /fishStory\.apply \{/ }));

  const rows = screen.getAllByTestId('story-row');
  expect(within(rows[0]).getByText('sun_2')).toBeInTheDocument();
  expect(within(rows[14]).getByText('sun_2')).toBeInTheDocument();
  expect(within(rows[15]).queryByText('sun_2')).not.toBeInTheDocument();

  // Staging only: no render and no plan until the user opens the review dialog.
  expect(calls().some(([p]) => p === '/api/fish/story/render')).toBe(false);
  expect(calls().some(([p]) => p === '/api/fish/story/plan')).toBe(false);
});

it('"Apply & select next block" moves to the following lines with the same count', async () => {
  await loadStory();
  fireEvent.click(screen.getByRole('button', { name: 'fishStory.load_voices' }));
  await screen.findAllByRole('option', { name: 'Narrator' });
  fireEvent.change(screen.getByLabelText('fishStory.range_to'), { target: { value: '8' } });
  fireEvent.click(screen.getByRole('button', { name: 'fishStory.select' }));
  fireEvent.change(screen.getByLabelText('fishStory.voice'), { target: { value: 'v-narr' } });
  fireEvent.click(screen.getByRole('button', { name: 'fishStory.apply_next' }));

  const rows = screen.getAllByTestId('story-row');
  expect(within(rows[7]).getByText('Narrator')).toBeInTheDocument();
  expect(rows[8]).toHaveAttribute('aria-selected', 'true'); // line 9 starts the next block
  expect(rows[15]).toHaveAttribute('aria-selected', 'true');
  expect(rows[16]).toHaveAttribute('aria-selected', 'false');
});

it('shift-click selects a range between two rows', async () => {
  await loadStory();
  const rows = screen.getAllByTestId('story-row');
  fireEvent.click(rows[2]);
  fireEvent.click(rows[5], { shiftKey: true });
  expect(screen.getAllByTestId('story-row').filter((r) => r.getAttribute('aria-selected') === 'true')).toHaveLength(4);
});

it('submits only after the review dialog is confirmed, with confirm:true', async () => {
  await loadStory();
  fireEvent.click(screen.getByRole('button', { name: 'fishStory.review_submit' }));
  await waitFor(() => expect(calls()).toContainEqual(['/api/fish/story/plan', 'POST']));
  expect(calls().some(([p]) => p === '/api/fish/story/render')).toBe(false);

  fireEvent.click(await screen.findByRole('button', { name: 'fishStory.submit_to_fish' }));
  await waitFor(() => expect(calls()).toContainEqual(['/api/fish/story/render', 'POST']));
  const [, init] = mock.api.mock.calls.find(([p]) => p === '/api/fish/story/render')!;
  expect(JSON.parse(init.body)).toMatchObject({ confirm: true });
  expect(await screen.findByRole('progressbar')).toBeInTheDocument();
});

it('autosaves the draft locally without touching Fish', async () => {
  await loadStory();
  await waitFor(() => expect(calls()).toContainEqual(['/api/fish/story/drafts', 'POST']), { timeout: 3000 });
  expect(calls().some(([p]) => String(p).includes('/v1/') || p === '/api/fish/balance')).toBe(false);
});

it('checks the balance only when the button is pressed', async () => {
  mock.api.mockImplementation((path: string) =>
    path === '/api/fish/balance'
      ? Promise.resolve({ credit: '12.5', cumulative_top_up: '20', has_free_credit: false })
      : path === '/api/fish/story/drafts'
        ? Promise.resolve({ drafts: [] })
        : path === '/api/fish/story/defaults'
          ? Promise.resolve(DEFAULTS)
          : Promise.resolve({}),
  );
  render(<FishStoryPage />);
  await screen.findByLabelText('fishStory.paste');
  expect(calls().some(([p]) => p === '/api/fish/balance')).toBe(false);
  fireEvent.click(screen.getByRole('button', { name: 'fishStory.check_balance' }));
  expect(await screen.findByText(/12\.5/)).toBeInTheDocument();
});

const DIRECTED = [
  '[Shadow] [slow][soft] It feels strange, [pause] to become a [emphasis] human once more.',
  'He looked at his hands.',
  '[Snuggle] Hello!',
].join('\n');

it('maps [Name] tags to voices and leaves narration on the narrator voice', async () => {
  render(<FishStoryPage />);
  fireEvent.click(await screen.findByRole('button', { name: 'fishStory.load_voices' }));
  await screen.findAllByRole('option', { name: 'sun_2' });
  fireEvent.change(screen.getByLabelText('fishStory.paste'), { target: { value: DIRECTED } });
  fireEvent.click(screen.getByRole('button', { name: 'fishStory.append' }));

  // Speaker tags were found, so the Tags & cast dialog opens by itself.
  const shadow = await screen.findByLabelText(/fishStory\.speaker_voice.*Shadow/);
  expect(screen.getByLabelText(/fishStory\.speaker_voice.*Snuggle/)).toBeInTheDocument();
  fireEvent.change(shadow, { target: { value: 'v-sun' } });
  fireEvent.change(screen.getByLabelText('fishStory.narrator'), { target: { value: 'v-narr' } });
  fireEvent.click(screen.getByRole('button', { name: 'fishStory.apply_cast' }));
  fireEvent.click(screen.getByRole('button', { name: 'fishStory.done' }));

  await waitFor(() => expect(screen.getAllByTestId('story-row')).toHaveLength(3));
  const rows = screen.getAllByTestId('story-row');
  // Name stripped, delivery tags kept, Shadow's voice applied.
  expect(within(rows[0]).getByText(/^\[slow\]\[soft\] It feels strange, \[pause\]/)).toBeInTheDocument();
  expect(within(rows[0]).getByText('sun_2')).toBeInTheDocument();
  expect(within(rows[0]).getByText('Shadow')).toBeInTheDocument();
  // Narration has no speaker, so it shows the narrator (the default voice).
  expect(within(rows[1]).getByText('Narrator')).toBeInTheDocument();
  expect(within(rows[1]).queryByText('Shadow')).not.toBeInTheDocument();
  // Snuggle was never mapped: still the narrator, but the name is kept on the line.
  expect(within(rows[2]).getByText('Snuggle')).toBeInTheDocument();
  expect(calls().some(([p]) => p === '/api/fish/story/render')).toBe(false);
});
