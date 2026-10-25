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
vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, values?: Record<string, unknown>) =>
      values ? `${key} ${JSON.stringify(values)}` : key,
  }),
}));

import { QwenPlaygroundPage } from './qwen-playground-page';

const STATE = {
  settings: { python: 'py', output_dir: 'C:/out', warmup_runs: 1, warmup_text: 'Warm.' },
  output_dir_default: 'C:/default',
  engine: { available: true, reason: '' },
  models: [
    { id: 'ep09', label: 'Run 2 EP09', path: 'p1', source: 'loc1', location: 'loc1', location_name: 'Production', type: 'custom_voice', size: '1.7B', speakers: ['ss_narrator'], borrowable_speakers: ['ryan'] },
    { id: 'base17', label: 'Stock Base', path: 'p2', source: 'hf', location: 'huggingface_cache', type: 'base', size: '1.7B', speakers: [], borrowable_speakers: ['ryan'] },
  ],
  stock_speakers: ['ryan'],
  profiles: [{ id: 'p-shadow', name: 'Shadow', ref_text: 'Hello.', language: 'English', kind: 'clone' }],
};

const take = (over = {}) => ({
  id: 'x', number: 1, label: '', text: 'Hi', model_id: 'ep09', model: { label: 'Run 2 EP09', type: 'custom_voice' },
  repeat: 1, notes: '', starred: false, results: [], ...over,
});
const job = (status: string, results: unknown[], extra = {}) => ({
  id: 'j1', session: 'playground_20261025-120000', folder: 'C:/out/playground_20261025-120000', status,
  total: 1, done: status === 'running' ? 0 : 1, model_swaps: 1, log: [], takes: [take({ results })], ...extra,
});
const DONE_WARN = {
  rep: 1, status: 'done', file: 'take-001.wav', seconds: 3.2, gen_seconds: 1.6, realtime: 2, seed: 99, complete: false,
  warnings: ['ryan is borrowed from the stock voices'],
};

let runBody: { takes: Array<{ id: string }> } | null = null;
const calls = () => mock.api.mock.calls.map(([path, init]) => [path, init?.method ?? 'GET']);

beforeEach(() => {
  window.localStorage.clear();
  runBody = null;
  URL.createObjectURL = vi.fn(() => 'blob:audio');
  mock.fetch.mockResolvedValue({ blob: () => Promise.resolve(new Blob(['x'])) });
  mock.api.mockImplementation((path: string, init?: RequestInit) => {
    if (path === '/api/qwen/state') return Promise.resolve(STATE);
    if (path === '/api/qwen/playground/run') {
      runBody = JSON.parse(String(init?.body));
      return Promise.resolve({ ...job('running', []), takes: [take({ id: runBody!.takes[0].id })] });
    }
    if (path.startsWith('/api/qwen/playground/jobs/j1')) {
      return Promise.resolve({ ...job('done', [DONE_WARN]), takes: [take({ id: runBody!.takes[0].id, results: [DONE_WARN] })] });
    }
    return Promise.resolve({});
  });
});
afterEach(() => cleanup());

async function ready() {
  render(<QwenPlaygroundPage />);
  // The first take appears with the narrator pre-filled once the state has loaded.
  await waitFor(() => expect(screen.getByTestId('summary').textContent).toContain('ss_narrator'));
}

it('starts with one narrator take, summarised in plain words', async () => {
  await ready();
  expect(screen.getByTestId('recipe').textContent).toBe('qwenPlayground.recipe_narrator');
  expect(screen.getByTestId('summary').textContent).toContain('Run 2 EP09');
  expect(screen.getByTestId('loads').textContent).toContain('"count":1');
});

it('runs a take, shows progress, then the render with its warnings and incomplete flag', async () => {
  await ready();
  fireEvent.change(screen.getByLabelText('qwenPlayground.prompt'), { target: { value: 'Hello world.' } });
  fireEvent.click(screen.getByText('qwenPlayground.run_all'));

  await waitFor(() => expect(runBody).not.toBeNull());
  expect(runBody).toMatchObject({
    session_name: 'playground',
    group_by_model: true,
    takes: [{ text: 'Hello world.', model_id: 'ep09', voice: { kind: 'custom_voice', speaker: 'ss_narrator' } }],
  });
  // The saved output folder is already the one in the settings: no settings write.
  expect(calls()).not.toContainEqual(['/api/qwen/settings', 'PUT']);

  await waitFor(() => expect(screen.getByText(/qwenPlayground\.job_done/)).toBeTruthy(), { timeout: 4000 });
  expect(screen.getByText('ryan is borrowed from the stock voices')).toBeTruthy();
  expect(screen.getByText('qwenPlayground.incomplete')).toBeTruthy();
  expect(screen.getByText(/qwenPlayground\.realtime/)).toBeTruthy();
  await waitFor(() => expect(screen.getByLabelText('take-001.wav')).toBeTruthy());
}, 10000);

it('saves notes on leave and the star with PATCH', async () => {
  await ready();
  fireEvent.change(screen.getByLabelText('qwenPlayground.prompt'), { target: { value: 'Hello.' } });
  fireEvent.click(screen.getByText('qwenPlayground.run_all'));
  await waitFor(() => expect(screen.getByLabelText('qwenPlayground.notes')).toBeTruthy());

  const notes = screen.getByLabelText('qwenPlayground.notes');
  fireEvent.change(notes, { target: { value: 'too fast' } });
  fireEvent.blur(notes);
  await waitFor(() => {
    const patch = mock.api.mock.calls.find(([, init]) => init?.method === 'PATCH');
    expect(patch?.[0]).toBe(`/api/qwen/playground/sessions/playground_20261025-120000/takes/${runBody!.takes[0].id}`);
    expect(JSON.parse(patch![1].body)).toEqual({ notes: 'too fast' });
  });

  fireEvent.click(screen.getByLabelText('qwenPlayground.star'));
  await waitFor(() => {
    const bodies = mock.api.mock.calls.filter(([, init]) => init?.method === 'PATCH').map(([, init]) => JSON.parse(init.body));
    expect(bodies).toContainEqual({ starred: true });
  });
}, 10000);

it('shows the backend reason when a run is refused', async () => {
  mock.api.mockImplementation((path: string) => {
    if (path === '/api/qwen/state') return Promise.resolve(STATE);
    return Promise.reject(new Error('Pick a speaker for Stock CustomVoice: ryan, vivian.'));
  });
  await ready();
  fireEvent.change(screen.getByLabelText('qwenPlayground.prompt'), { target: { value: 'Hi.' } });
  fireEvent.click(screen.getByText('qwenPlayground.run_all'));
  await waitFor(() => expect(screen.getByText(/Pick a speaker for Stock CustomVoice/)).toBeTruthy());
});

it('warns about a mismatched model without blocking, and adds, duplicates and removes takes', async () => {
  await ready();
  fireEvent.change(screen.getByLabelText('qwenPlayground.model'), { target: { value: 'base17' } });
  expect(screen.getByText(/qwenPlayground\.issue_pair_mismatch/)).toBeTruthy();
  expect((screen.getByText('qwenPlayground.run_all').closest('button') as HTMLButtonElement).disabled).toBe(false);

  fireEvent.click(screen.getByText('qwenPlayground.add_take'));
  expect(screen.getAllByRole('article')).toHaveLength(2);
  fireEvent.click(screen.getAllByLabelText('qwenPlayground.duplicate')[0]);
  expect(screen.getAllByRole('article')).toHaveLength(3);
  fireEvent.click(screen.getAllByLabelText('qwenPlayground.delete')[0]);
  expect(screen.getAllByRole('article')).toHaveLength(2);
  // Left with a Base take and a narrator take: two models, two loads.
  expect(screen.getByTestId('loads').textContent).toContain('"count":2');
});

it('has no checkpoint-folder box, links to Settings and names where each model lives', async () => {
  await ready();
  expect(screen.queryByText('qwenPlayground.checkpoint_root')).toBeNull();
  expect(screen.getByText('modelLocations.manage_folders')).toBeTruthy();
  const labels = screen.getAllByRole('option').map((o) => o.textContent);
  expect(labels).toContain('Run 2 EP09, Production (custom_voice, 1.7B)');
  expect(labels.some((l) => l?.includes('Stock Base, modelLocations.loc_hf'))).toBe(true);
});

it('warns, without blocking, when a take points at a model that is no longer found', async () => {
  await ready();
  fireEvent.change(screen.getByLabelText('qwenPlayground.prompt'), { target: { value: 'Hello.' } });
  mock.api.mockImplementation((path: string, init?: RequestInit) => {
    if (path === '/api/qwen/state') return Promise.resolve(STATE);
    if (path === '/api/qwen/playground/run') {
      runBody = JSON.parse(String(init?.body));
      return Promise.resolve({
        ...job('done', [DONE_WARN]),
        takes: [
          take({
            id: runBody!.takes[0].id,
            model_id: 'gone',
            model: { label: 'Old tune', type: 'custom_voice', path: 'D:/old/ep3' },
            results: [DONE_WARN],
          }),
        ],
      });
    }
    return Promise.resolve({});
  });
  fireEvent.click(screen.getByText('qwenPlayground.run_all'));
  await waitFor(() =>
    expect(screen.getByText(/modelLocations\.model_not_found.*Old tune.*D:\/old\/ep3/)).toBeTruthy(),
  );
  await waitFor(() => expect(screen.getByLabelText('take-001.wav')).toBeTruthy());
});
