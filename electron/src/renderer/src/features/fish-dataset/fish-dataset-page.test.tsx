import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const mock = vi.hoisted(() => ({ api: vi.fn(), fetch: vi.fn() }));

vi.mock('@/lib/api/client', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/lib/api/client')>()),
  apiJson: mock.api,
  apiFetch: mock.fetch,
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

const FACETS = {
  clips: 2,
  seconds: 9,
  duration: [
    { from: 0, to: 1, clips: 0 },
    { from: 1, to: 2, clips: 1 },
    { from: 2, to: 3, clips: 0 },
    { from: 3, to: 4, clips: 1 },
    { from: 4, to: null, clips: 0 },
  ],
  tags: [
    ['slow', 2],
    ['calm', 1],
  ],
  chapters: [
    { chapter: '3203_x', clips: 1, seconds: 4, rms_db: -26, peak: 0.3 },
    { chapter: '3195_loud', clips: 1, seconds: 5, rms_db: -14, peak: 0.95 },
  ],
  words: { min: 2, max: 5, avg: 3.5 },
  quality: { analyzed: 1, total: 2, clipping: 0, loud: 1, quiet: 0, choppy: 0, long_edges: 0 },
};
const IDLE = { running: false, done: 0, total: 0, failed: 0, cached: 0 };

const calls = () => mock.api.mock.calls.map(([path, init]) => [path, init?.method ?? 'GET']);
const lastBody = (match: string) => {
  const hit = [...mock.api.mock.calls].reverse().find(([p]) => String(p).startsWith(match));
  return JSON.parse(hit![1].body);
};

beforeEach(() => {
  mock.api.mockImplementation((path: string, init?: RequestInit) => {
    if (path.startsWith('/api/fish/dataset/lines')) {
      const spec = JSON.parse(String(init?.body ?? '{}'));
      const left = new Set<string>(spec.exclude_keys ?? []);
      const only: string[] | undefined = spec.only_keys ?? undefined;
      const shown = LINES.lines.filter((l) => !only || only.includes(l.key));
      const picked = shown.filter((l) => !left.has(l.key));
      return Promise.resolve({
        ...LINES,
        total: shown.length,
        lines: shown,
        selected: picked.length,
        selected_seconds: picked.reduce((n, l) => n + l.duration_s, 0),
      });
    }
    if (path === '/api/fish/dataset/reference-candidates')
      return Promise.resolve({
        speaker: { voice_id: 'v-narr', name: 'SS_Narrator' },
        considered: 2,
        unmeasured: 0,
        clips: 2,
        candidates: [
          {
            key: 'r1:c1', chapter: '3203_x', index: 1, text: 'The road was long.', duration_s: 3.2,
            stats: { peak: 0.4, rms_db: -23, clip: 0, lead_ms: 0, tail_ms: 0, pauses: 0, max_gap_ms: 0, noise_floor_db: -50 },
            score: -0.5, reasons: ['not_clipped'], cautions: [], relaxed: null,
          },
        ],
      });
    if (path === '/api/fish/dataset/facets')
      return Promise.resolve({ ...FACETS, levels: { measured: 1, total: 2, min: -26, median: -26, max: -26, counts: [[-26, 1]] } });
    if (path === '/api/fish/dataset/analyze')
      return Promise.resolve(init?.method === 'POST' ? { ...IDLE, running: true, total: 3 } : IDLE);
    if (path === '/api/fish/dataset/root') return Promise.resolve({ root: 'D:\\other' });
    if (path === '/api/fish/dataset/export')
      return Promise.resolve({
        path: 'D:\\out\\set',
        clips: 1,
        hours: 0.001,
        speakers: {},
        formats: JSON.parse(String(init?.body ?? '{}')).formats,
      });
    if (path === '/api/fish/training')
      return Promise.resolve({ enabled: false, defaults: {}, runs: [], limits: {} });
    if (path.startsWith('/api/fish/dataset')) return Promise.resolve(OVERVIEW);
    return Promise.resolve({});
  });
});
afterEach(() => {
  cleanup();
  mock.api.mockReset();
  mock.fetch.mockReset();
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
  expect(await screen.findByTestId('character-view')).toHaveTextContent('Noctis');
  await waitFor(() => expect(lastBody('/api/fish/dataset/lines').voice_ids).toEqual(['v-noct']));

  fireEvent.click(await screen.findByLabelText('fishDataset.include {"index":1}'));
  await waitFor(() => expect(lastBody('/api/fish/dataset/lines').exclude_keys).toEqual(['r1:c1']));
  expect(await screen.findByText(/fishDataset\.selected_of .*"selected":"1","total":"2"/)).toBeInTheDocument();
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

it('offers no Train step for an export without the OmniVoice format', async () => {
  render(<FishDatasetPage />);
  await screen.findAllByTestId('clip-row');
  fireEvent.click(screen.getByRole('button', { name: /fishDataset\.export_open/ }));
  const dialog = await screen.findByRole('dialog');
  fireEvent.change(within(dialog).getByLabelText('fishDataset.export_dest'), { target: { value: 'D:\\out' } });
  fireEvent.click(within(dialog).getByRole('button', { name: /fishDataset\.export_button/ }));
  await within(dialog).findByText(/fishDataset\.export_done/);
  expect(within(dialog).queryByRole('button', { name: /fishTraining\.train_open/ })).not.toBeInTheDocument();
  expect(calls().some(([p]) => String(p).startsWith('/api/fish/training'))).toBe(false);
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
    loudness_db: null,
  });
});

it('exports the OmniVoice format at 24 kHz with stripped text and the default -22 dB loudness level', async () => {
  render(<FishDatasetPage />);
  await screen.findAllByTestId('clip-row');
  fireEvent.click(screen.getByRole('button', { name: /fishDataset\.export_open/ }));
  const dialog = await screen.findByRole('dialog');
  fireEvent.click(within(dialog).getByLabelText(/fishDataset\.format_omnivoice(?!_)/));
  expect(within(dialog).getByLabelText('fishDataset.export_text')).toBeDisabled();
  expect(within(dialog).getByLabelText('fishDataset.export_rate')).toBeDisabled();
  // matching loudness is already ticked at -22 dB for this format: nothing to touch
  expect(within(dialog).getByLabelText('fishDataset.export_loudness')).toBeChecked();
  expect(within(dialog).getByLabelText('fishDataset.export_loudness_db')).toHaveValue(-22);
  fireEvent.change(within(dialog).getByLabelText('fishDataset.export_dest'), { target: { value: 'D:\\out' } });
  fireEvent.click(within(dialog).getByRole('button', { name: /fishDataset\.export_button/ }));
  await within(dialog).findByText(/fishDataset\.export_done/);
  expect(lastBody('/api/fish/dataset/export')).toMatchObject({
    formats: ['jsonl', 'fish_lab', 'omnivoice'],
    text_mode: 'strip',
    sample_rate: 24000,
    loudness_db: -22,
  });
  // the Train step is offered for this export, and starts out switched off
  fireEvent.click(within(dialog).getByRole('button', { name: /fishTraining\.train_open/ }));
  expect(await screen.findByTestId('train-off')).toBeInTheDocument();
  expect(calls().some(([p]) => p === '/api/fish/training')).toBe(true);
});

it('asks for a folder first when none is saved', async () => {
  mock.api.mockImplementation(() =>
    Promise.resolve({ root: null, exists: false, speakers: [], runs: [], chapters: [], totals: null }),
  );
  render(<FishDatasetPage />);
  expect(await screen.findByText('fishDataset.empty')).toBeInTheDocument();
  expect(screen.getByRole('button', { name: /fishDataset\.export_open/ })).toBeDisabled();
});

it('strips all tags by default and keeps, or rewrites, only the ones chosen in the tag table', async () => {
  render(<FishDatasetPage />);
  await screen.findAllByTestId('clip-row');
  fireEvent.click(screen.getByRole('button', { name: /fishDataset\.export_open/ }));
  const dialog = await screen.findByRole('dialog');
  expect(within(dialog).queryByTestId('tag-row')).not.toBeInTheDocument(); // text is kept as-is
  fireEvent.change(within(dialog).getByLabelText('fishDataset.export_text'), { target: { value: 'strip' } });

  expect(within(dialog).getAllByTestId('tag-row')).toHaveLength(3);
  fireEvent.change(within(dialog).getByLabelText(/tag_action .*sighing/), { target: { value: 'keep' } });
  fireEvent.change(within(dialog).getByLabelText(/tag_action .*laughing/), { target: { value: 'convert' } });
  fireEvent.change(within(dialog).getByLabelText(/tag_convert_to .*laughing/), { target: { value: 'laughter' } });

  fireEvent.change(within(dialog).getByLabelText('fishDataset.export_dest'), { target: { value: 'D:\\out' } });
  fireEvent.click(within(dialog).getByRole('button', { name: /fishDataset\.export_button/ }));
  await within(dialog).findByText(/fishDataset\.export_done/);
  expect(lastBody('/api/fish/dataset/export')).toMatchObject({
    text_mode: 'strip',
    keep_tags: ['[laughing]:laughter', 'sighing'],
  });

  // Switching back to keeping every tag sends no keep list.
  fireEvent.change(within(dialog).getByLabelText('fishDataset.export_text'), { target: { value: 'keep' } });
  fireEvent.click(within(dialog).getByRole('button', { name: /fishDataset\.export_button/ }));
  await waitFor(() => expect(lastBody('/api/fish/dataset/export').text_mode).toBe('keep'));
  expect(lastBody('/api/fish/dataset/export').keep_tags).toEqual([]);
});

const lastLines = () => lastBody('/api/fish/dataset/lines');

it('zooms into one character and back out', async () => {
  render(<FishDatasetPage />);
  fireEvent.click((await screen.findAllByRole('button', { name: /fishDataset\.zoom/ }))[0]);
  const view = await screen.findByTestId('character-view');
  expect(view).toHaveTextContent('SS_Narrator');
  expect(screen.queryAllByTestId('speaker-row')).toHaveLength(0);
  await waitFor(() => expect(lastLines().voice_ids).toEqual(['v-narr']));
  fireEvent.click(within(view).getByRole('button', { name: /fishDataset\.all_speakers/ }));
  expect(await screen.findAllByTestId('speaker-row')).toHaveLength(2);
  await waitFor(() => expect(lastLines().voice_ids).toEqual([]));
});

it('narrows by clicking the length chart, tags and chapters, with removable filter chips', async () => {
  render(<FishDatasetPage />);
  await screen.findAllByTestId('clip-row');
  const length = screen.getByRole('region', { name: 'fishDataset.chart_length' });
  fireEvent.click(within(length).getAllByRole('button')[3]); // the 3-4 s bar
  await waitFor(() => expect(lastLines()).toMatchObject({ min_dur: 3, max_dur: 4 }));
  expect(screen.getByRole('button', { name: /fishDataset\.remove_filter .*chip_dur/ })).toBeInTheDocument();

  const tags = screen.getByRole('region', { name: 'fishDataset.chart_tags' });
  fireEvent.click(within(tags).getByRole('button', { name: /tag_off .*calm/ }));
  await waitFor(() => expect(lastLines().tags_all).toEqual(['calm']));
  fireEvent.click(within(tags).getByRole('button', { name: /tag_all .*calm/ }));
  await waitFor(() => expect(lastLines()).toMatchObject({ tags_all: [], tags_none: ['calm'] }));
  fireEvent.click(within(tags).getByRole('button', { name: /tag_none .*calm/ }));
  await waitFor(() => expect(lastLines().tags_none).toEqual([]));

  const chapters = screen.getByRole('region', { name: 'fishDataset.chart_chapters' });
  fireEvent.click(within(chapters).getByRole('button', { name: /3203_x/ }));
  await waitFor(() => expect(lastLines().chapters).toEqual(['3203_x']));
  expect(within(chapters).getByRole('status')).toHaveTextContent('fishDataset.level_warning'); // -14 vs -26 dB

  fireEvent.click(screen.getByRole('button', { name: /fishDataset\.remove_filter .*chip_dur/ }));
  await waitFor(() => expect(lastLines()).toMatchObject({ min_dur: null, max_dur: null, chapters: ['3203_x'] }));
  fireEvent.click(screen.getByRole('button', { name: /fishDataset\.remove_filter .*3203_x/ }));
  await waitFor(() => expect(lastLines().chapters).toEqual([]));
});

it('measures the audio on request and sends quality limits', async () => {
  render(<FishDatasetPage />);
  await screen.findAllByTestId('clip-row');
  fireEvent.click(screen.getByRole('button', { name: 'fishDataset.analyze' }));
  await waitFor(() => expect(calls()).toContainEqual(['/api/fish/dataset/analyze', 'POST']));
  expect(await screen.findByText(/fishDataset\.analyzing .*"total":3/)).toBeInTheDocument();

  fireEvent.click(screen.getByRole('switch', { name: /hide_clipped/ }));
  fireEvent.change(screen.getByLabelText('fishDataset.max_peak'), { target: { value: '0.8' } });
  fireEvent.change(screen.getByLabelText('fishDataset.min_rms'), { target: { value: '-30' } });
  fireEvent.change(screen.getByLabelText('fishDataset.max_pauses'), { target: { value: '2' } });
  await waitFor(() =>
    expect(lastLines()).toMatchObject({ max_peak: 0.8, min_rms_db: -30, max_pauses: 2 }),
  );
});

it('shuffles a random sample and can show only the clips left out', async () => {
  render(<FishDatasetPage />);
  await screen.findAllByTestId('clip-row');
  fireEvent.click(screen.getByRole('button', { name: /fishDataset\.shuffle/ }));
  await waitFor(() => expect(lastLines().sort).toBe('random'));
  expect(lastLines().seed).toBeGreaterThan(0);

  fireEvent.click(screen.getByLabelText('fishDataset.include {"index":1}'));
  await waitFor(() => expect(lastLines().exclude_keys).toEqual(['r1:c1']));
  fireEvent.change(screen.getByLabelText('fishDataset.view'), { target: { value: 'excluded' } });
  await waitFor(() => expect(lastLines()).toMatchObject({ only_keys: ['r1:c1'], exclude_keys: [] }));
});

it('leaves out every matching clip, then puts them back', async () => {
  render(<FishDatasetPage />);
  await screen.findAllByTestId('clip-row');
  const selectNone = screen.getByRole('button', { name: /fishDataset\.select_none/ });
  const selectAll = screen.getByRole('button', { name: /fishDataset\.select_all/ });
  expect(selectAll).toBeDisabled(); // everything starts selected
  fireEvent.click(selectNone);
  await waitFor(() => expect(lastLines().exclude_keys).toEqual(['r1:c1', 'r1:c2']));
  expect(screen.getAllByTestId('clip-row')).toHaveLength(2); // still listed, just unticked
  fireEvent.click(selectAll);
  await waitFor(() => expect(lastLines().exclude_keys).toEqual([]));
});

const star = (index: number) =>
  screen.getByRole('button', { name: new RegExp(`fishDataset\\.shortlist_(add|remove) .*"index":${index}\\}`) });

describe('shortlist and reference', () => {
  it('stars any number of clips and unstars them again', async () => {
    render(<FishDatasetPage />);
    await screen.findAllByTestId('clip-row');
    fireEvent.click(star(1));
    fireEvent.click(star(2));
    expect(star(1)).toHaveAttribute('aria-pressed', 'true');
    expect(star(2)).toHaveAttribute('aria-pressed', 'true');
    fireEvent.click(star(1));
    expect(star(1)).toHaveAttribute('aria-pressed', 'false');
    expect(star(2)).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByRole('option', { name: /fishDataset\.view_shortlist .*"count":1/ })).toBeInTheDocument();
  });

  it('shows only the shortlist through only_keys, keeps it across filter changes and lets you unstar there', async () => {
    render(<FishDatasetPage />);
    await screen.findAllByTestId('clip-row');
    fireEvent.click(star(1));
    fireEvent.click(star(2));
    fireEvent.change(screen.getByLabelText('fishDataset.view'), { target: { value: 'shortlist' } });
    await waitFor(() => expect(lastLines()).toMatchObject({ only_keys: ['r1:c1', 'r1:c2'], exclude_keys: [] }));
    expect(await screen.findByTestId('filter-chip')).toHaveTextContent('fishDataset.chip_shortlist');

    fireEvent.change(screen.getByPlaceholderText('fishDataset.search'), { target: { value: 'road' } });
    await waitFor(() => expect(lastLines()).toMatchObject({ q: 'road', only_keys: ['r1:c1', 'r1:c2'] }));

    fireEvent.click(star(2));
    await waitFor(() => expect(lastLines().only_keys).toEqual(['r1:c1']));
    await waitFor(() => expect(screen.getAllByTestId('clip-row')).toHaveLength(1));
    fireEvent.change(screen.getByLabelText('fishDataset.view'), { target: { value: 'all' } });
    await waitFor(() => expect(lastLines().only_keys).toBeUndefined());
    expect(star(1)).toHaveAttribute('aria-pressed', 'true'); // still starred in the full view
  });

  it('plays only the shortlist in the shortlist view', async () => {
    mock.fetch.mockResolvedValue({ blob: () => Promise.resolve(new Blob(['x'])) });
    vi.stubGlobal('URL', Object.assign(URL, { createObjectURL: () => 'blob:x', revokeObjectURL: () => undefined }));
    vi.stubGlobal('Audio', class { onended = null; play = () => Promise.resolve(); pause = () => undefined; });
    render(<FishDatasetPage />);
    await screen.findAllByTestId('clip-row');
    fireEvent.click(star(2));
    fireEvent.change(screen.getByLabelText('fishDataset.view'), { target: { value: 'shortlist' } });
    await waitFor(() => expect(screen.getAllByTestId('clip-row')).toHaveLength(1));
    fireEvent.click(screen.getByRole('button', { name: 'fishDataset.play_all' }));
    await waitFor(() => expect(mock.fetch).toHaveBeenCalledTimes(1));
    expect(decodeURIComponent(String(mock.fetch.mock.calls[0][0]).split('key=')[1])).toBe('r1:c2');
    vi.unstubAllGlobals();
  });

  it('chooses one reference among the shortlist in the export dialog and sends exactly that key', async () => {
    render(<FishDatasetPage />);
    fireEvent.click((await screen.findAllByTestId('speaker-row'))[0]);
    await screen.findByTestId('character-view');
    await screen.findAllByTestId('clip-row');
    fireEvent.click(star(1));
    fireEvent.click(star(2));
    fireEvent.click(screen.getByRole('button', { name: /fishDataset\.export_open/ }));
    const dialog = await screen.findByRole('dialog');
    fireEvent.click(within(dialog).getByLabelText(/fishDataset\.format_qwen3_tts(?!_)/));
    expect(await within(dialog).findAllByTestId('reference-suggestion')).toHaveLength(1);
    expect(lastBody('/api/fish/dataset/reference-candidates')).toMatchObject({ voice_ids: ['v-narr'], limit: 5, min_duration: 0.8 });
    fireEvent.click(within(dialog).getByLabelText(/fishDataset\.ref_choose_clip .*3203_x #2/));
    expect(within(dialog).getByTestId('reference-current')).toHaveTextContent('Rain fell.');
    fireEvent.change(within(dialog).getByLabelText('fishDataset.export_dest'), { target: { value: 'D:\\out' } });
    fireEvent.click(within(dialog).getByRole('button', { name: /fishDataset\.export_button/ }));
    await within(dialog).findByText(/fishDataset\.export_done/);
    expect(lastBody('/api/fish/dataset/export')).toMatchObject({
      formats: ['qwen3_tts'],
      loudness_db: -22,
      qwen3: { reference_key: 'r1:c2' },
    });
  });
});

describe('playback', () => {
  const ended: Array<() => void> = [];
  beforeEach(() => {
    ended.length = 0;
    mock.fetch.mockResolvedValue({ blob: () => Promise.resolve(new Blob(['x'])) });
    vi.stubGlobal('URL', Object.assign(URL, { createObjectURL: () => 'blob:x', revokeObjectURL: () => undefined }));
    vi.stubGlobal(
      'Audio',
      class {
        onended: (() => void) | null = null;
        play = () => {
          if (this.onended) ended.push(this.onended);
          return Promise.resolve();
        };
        pause = () => undefined;
      },
    );
  });
  afterEach(() => vi.unstubAllGlobals());
  const fetched = () => mock.fetch.mock.calls.map(([path]) => decodeURIComponent(String(path).split('key=')[1]));

  it('plays from the top and moves on to the next clip by itself', async () => {
    render(<FishDatasetPage />);
    await screen.findAllByTestId('clip-row');
    fireEvent.click(screen.getByRole('button', { name: 'fishDataset.play_all' }));
    await waitFor(() => expect(fetched()).toEqual(['r1:c1']));
    ended.at(-1)!();
    await waitFor(() => expect(fetched()).toEqual(['r1:c1', 'r1:c2']));
    ended.at(-1)!(); // last clip of the only page: nothing further to play
    await new Promise((resolve) => setTimeout(resolve, 30));
    expect(fetched()).toHaveLength(2);
  });

  it('stays on one clip when auto-advance is off', async () => {
    render(<FishDatasetPage />);
    await screen.findAllByTestId('clip-row');
    fireEvent.click(screen.getByRole('switch', { name: /auto_advance/ }));
    fireEvent.click(screen.getByRole('button', { name: 'fishDataset.play_all' }));
    await waitFor(() => expect(fetched()).toEqual(['r1:c1']));
    ended.at(-1)!();
    await new Promise((resolve) => setTimeout(resolve, 30));
    expect(fetched()).toEqual(['r1:c1']);
  });

  it('moves with the arrow keys, plays with space and leaves a clip out with x', async () => {
    render(<FishDatasetPage />);
    await screen.findAllByTestId('clip-row');
    const list = screen.getByRole('listbox', { name: 'fishDataset.clips' });
    fireEvent.keyDown(list, { key: 'ArrowDown' });
    fireEvent.keyDown(list, { key: 'ArrowDown' });
    expect(within(list).getAllByRole('option')[1]).toHaveAttribute('aria-selected', 'true');
    fireEvent.keyDown(list, { key: 'ArrowUp' });
    fireEvent.keyDown(list, { key: ' ' });
    await waitFor(() => expect(fetched()).toEqual(['r1:c1']));
    fireEvent.keyDown(list, { key: 'x' });
    await waitFor(() => expect(lastLines().exclude_keys).toEqual(['r1:c1']));
  });
});
