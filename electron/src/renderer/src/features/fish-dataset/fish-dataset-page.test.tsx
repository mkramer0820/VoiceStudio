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
vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, values?: Record<string, unknown>) =>
      values ? `${key} ${JSON.stringify(values)}` : key,
  }),
}));

import { FishDatasetPage } from './fish-dataset-page';

const OVERVIEW = {
  root: 'D:\\fish_tts\\input',
  exists: true,
  speakers: [
    {
      voice_id: 'v-narr',
      name: 'SS_Narrator',
      aliases: [],
      lines: 2152,
      duplicates: 228,
      seconds: 14364,
      avg_s: 6.7,
      chapters: 30,
      top_tags: [['slow', 2151]],
    },
    {
      voice_id: 'v-noct',
      name: 'Noctis',
      aliases: ['Night'],
      lines: 311,
      duplicates: 27,
      seconds: 2020,
      avg_s: 6.5,
      chapters: 12,
      top_tags: [['pause', 298]],
    },
  ],
  runs: [{ run_id: 'a', chapter: '3218', run: 'run-001', lines: 89, failed: 0, missing: 109 }],
  chapters: ['3203_x', '3204_y'],
  tags: [
    ['slow', 2663],
    ['laughing', 17],
    ['sighing', 33],
  ],
  totals: { lines: 2463, duplicates: 255, seconds: 16384, runs: 32 },
};
const LINES = {
  total: 2,
  seconds: 9,
  lines: [
    {
      key: 'r1:c1',
      chapter: '3203_x',
      run: 'run-001',
      index: 1,
      voice_id: 'v-narr',
      voice_name: 'SS_Narrator',
      text: '[slow] The road was long.',
      plain: 'The road was long.',
      tags: ['slow'],
      words: 5,
      duration_s: 3.2,
      duplicate: false,
    },
    {
      key: 'r1:c2',
      chapter: '3203_x',
      run: 'run-001',
      index: 2,
      voice_id: 'v-narr',
      voice_name: 'SS_Narrator',
      text: '[slow] Rain fell.',
      plain: 'Rain fell.',
      tags: ['slow'],
      words: 2,
      duration_s: 1.4,
      duplicate: true,
    },
  ],
};

const calls = () => mock.api.mock.calls.map(([path, init]) => [path, init?.method ?? 'GET']);
const lastBody = (match: string) => {
  const hit = [...mock.api.mock.calls].reverse().find(([p]) => String(p).startsWith(match));
  return JSON.parse(hit![1].body);
};

beforeEach(() => {
  mock.api.mockImplementation((path: string) => {
    if (path.startsWith('/api/fish/dataset/lines')) return Promise.resolve(LINES);
    if (path === '/api/fish/dataset/root') return Promise.resolve({ root: 'D:\\other' });
    if (path === '/api/fish/dataset/export')
      return Promise.resolve({ path: 'D:\\out\\set', clips: 1, hours: 0.001, speakers: {} });
    if (path.startsWith('/api/fish/dataset')) return Promise.resolve(OVERVIEW);
    return Promise.resolve({});
  });
});
afterEach(() => {
  cleanup();
  mock.api.mockReset();
});

it('shows speakers, totals and clips from the saved folder without contacting Fish', async () => {
  render(<FishDatasetPage />);
  expect(await screen.findAllByTestId('speaker-row')).toHaveLength(2);
  expect(screen.getByText(/fishDataset\.totals .*"clips":"2,463".*"hours":"4\.6"/)).toBeInTheDocument();
  expect(await screen.findAllByTestId('clip-row')).toHaveLength(2);
  expect(screen.getByText('The road was long.')).toBeInTheDocument();
  expect(screen.getByText('fishDataset.older_copy')).toBeInTheDocument();
  expect(screen.getByText(/fishDataset\.run_issues .*"missing":109/)).toBeInTheDocument();
  expect(calls().every(([p]) => String(p).startsWith('/api/fish/dataset'))).toBe(true);
});

it('limits the list to a clicked speaker and excludes a clip from the export', async () => {
  render(<FishDatasetPage />);
  const rows = await screen.findAllByTestId('speaker-row');
  fireEvent.click(rows[1]);
  expect(rows[1]).toHaveAttribute('aria-pressed', 'true');
  await waitFor(() => expect(lastBody('/api/fish/dataset/lines').voice_ids).toEqual(['v-noct']));

  fireEvent.click(await screen.findByLabelText('fishDataset.include {"index":1}'));
  await waitFor(() => expect(lastBody('/api/fish/dataset/lines').exclude_keys).toEqual(['r1:c1']));
  expect(screen.getByText(/fishDataset\.excluded \{"count":1\}/)).toBeInTheDocument();
});

it('saves a new dataset folder', async () => {
  render(<FishDatasetPage />);
  await screen.findAllByTestId('speaker-row');
  fireEvent.change(screen.getByLabelText('fishDataset.folder'), { target: { value: 'D:\\other' } });
  fireEvent.click(screen.getByRole('button', { name: 'fishDataset.folder_save' }));
  await waitFor(() => expect(calls()).toContainEqual(['/api/fish/dataset/root', 'PUT']));
  expect(JSON.parse(mock.api.mock.calls.find(([p]) => p === '/api/fish/dataset/root')![1].body)).toEqual({
    path: 'D:\\other',
  });
});

it('exports the filtered set only after the folder is given', async () => {
  render(<FishDatasetPage />);
  await screen.findAllByTestId('clip-row');
  fireEvent.click(screen.getByRole('button', { name: /fishDataset\.export_open/ }));
  const dialog = await screen.findByRole('dialog');
  const go = within(dialog).getByRole('button', { name: /fishDataset\.export_button/ });
  expect(go).toBeDisabled();
  expect(calls().some(([p]) => p === '/api/fish/dataset/export')).toBe(false);

  fireEvent.change(within(dialog).getByLabelText('fishDataset.export_dest'), { target: { value: 'D:\\out' } });
  fireEvent.change(within(dialog).getByLabelText('fishDataset.export_text'), { target: { value: 'strip' } });
  fireEvent.click(go);
  await within(dialog).findByText(/fishDataset\.export_done/);
  expect(lastBody('/api/fish/dataset/export')).toMatchObject({
    dest: 'D:\\out',
    formats: ['jsonl', 'fish_lab'],
    text_mode: 'strip',
    sample_rate: null,
    val_fraction: 0.05,
    keep_tags: [],
  });
});

it('asks for a folder first when none is saved', async () => {
  mock.api.mockImplementation(() =>
    Promise.resolve({ root: null, exists: false, speakers: [], runs: [], chapters: [], totals: null }),
  );
  render(<FishDatasetPage />);
  expect(await screen.findByText('fishDataset.empty')).toBeInTheDocument();
  expect(screen.getByRole('button', { name: /fishDataset\.export_open/ })).toBeDisabled();
});

it('strips all tags by default and keeps only the ones listed, optionally renamed', async () => {
  render(<FishDatasetPage />);
  await screen.findAllByTestId('clip-row');
  fireEvent.click(screen.getByRole('button', { name: /fishDataset\.export_open/ }));
  const dialog = await screen.findByRole('dialog');
  expect(within(dialog).queryByLabelText('fishDataset.keep_tags')).not.toBeInTheDocument(); // text is kept as-is
  fireEvent.change(within(dialog).getByLabelText('fishDataset.export_text'), { target: { value: 'strip' } });

  const keep = within(dialog).getByLabelText('fishDataset.keep_tags');
  expect(keep).toHaveValue('');
  fireEvent.click(within(dialog).getByRole('button', { name: /\[sighing\]/ }));
  expect(keep).toHaveValue('sighing');
  fireEvent.change(keep, { target: { value: 'sighing=sigh, laughing=laughter' } });
  expect(within(dialog).getByRole('button', { name: /\[laughing\]/ })).toHaveAttribute('aria-pressed', 'true');
  fireEvent.click(within(dialog).getByRole('button', { name: /\[sighing\]/ })); // un-keep
  expect(keep).toHaveValue('laughing=laughter');

  fireEvent.change(within(dialog).getByLabelText('fishDataset.export_dest'), { target: { value: 'D:\out' } });
  fireEvent.click(within(dialog).getByRole('button', { name: /fishDataset\.export_button/ }));
  await within(dialog).findByText(/fishDataset\.export_done/);
  expect(lastBody('/api/fish/dataset/export')).toMatchObject({ text_mode: 'strip', keep_tags: ['laughing=laughter'] });

  // Switching back to keeping every tag sends no keep list.
  fireEvent.change(within(dialog).getByLabelText('fishDataset.export_text'), { target: { value: 'keep' } });
  fireEvent.click(within(dialog).getByRole('button', { name: /fishDataset\.export_button/ }));
  await waitFor(() => expect(lastBody('/api/fish/dataset/export').text_mode).toBe('keep'));
  expect(lastBody('/api/fish/dataset/export').keep_tags).toEqual([]);
});
