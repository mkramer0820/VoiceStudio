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
  output_dir: null,
  output_dir_default: 'C:/data/fish_stories/renders',
  output_dir_effective: 'C:/data/fish_stories/renders',
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
  request_mode: 'single',
};

const calls = () => mock.api.mock.calls.map(([path, init]) => [path, init?.method ?? 'GET']);

beforeEach(() => {
  mock.api.mockImplementation((path: string, init?: RequestInit) => {
    if (path === '/api/fish/story/drafts' && !init?.method) return Promise.resolve({ drafts: [] });
    if (path === '/api/fish/story/defaults') return Promise.resolve(DEFAULTS);
    if (path.startsWith('/api/fish/story/jobs')) return Promise.resolve({ jobs: [] });
    if (path === '/api/fish/story/drafts') return Promise.resolve({ id: 'd1' });
    if (path.startsWith('/api/fish/story/drafts/')) return Promise.resolve({ id: 'd1' });
    if (path === '/api/fish/voices?scope=mine') return Promise.resolve(VOICES);
    if (path === '/api/fish/story/plan') return Promise.resolve(PLAN);
    if (path === '/api/fish/collection')
      return Promise.resolve({
        title: 'shadow sl',
        voices: [
          { id: '59aa49a62c9f4608926420443e673e5c', title: 'Nihilister' },
          { id: '0c67dc47f4704447a39d436ae677b606', title: 'Moon Goddess' },
        ],
      });
    if (path.startsWith('/api/fish/voices?scope=library'))
      return Promise.resolve({ voices: [{ id: 'pub1', title: 'Deep Narrator', author: 'someone', languages: ['en'] }] });
    if (path === '/api/fish/story/preview')
      return Promise.resolve({
        ok: true,
        total: 20,
        total_requests: 20,
        invalid: 0,
        shown: 1,
        endpoint: 'https://api.fish.audio/v1/tts',
        headers: { Authorization: 'Bearer <hidden>', model: 's2.1-pro' },
        folder: 'C:/out/story/run-NNN',
        curl: 'curl -X POST https://api.fish.audio/v1/tts',
        first_invalid: null,
        extra_fields: ['latency', 'normalize'],
        minimal: false,
        settings: [
          { name: 'temperature', value: 0.7, source: 'story', sent: true, note: null },
          { name: 'top_p', value: null, source: 'fish', sent: false, note: null },
          { name: 'normalize', value: false, source: 'saved default', sent: true, note: null },
        ],
        requests: [
          {
            index: 1,
            indices: [1],
            id: 'x',
            ids: ['x'],
            mode: 'single',
            speaker: null,
            speakers: ['narrator'],
            voice: 'pub1',
            voice_name: 'Deep Narrator',
            file: '001_narration_Deep.wav',
            chars: 14,
            body: { text: 'Line number 1.', format: 'wav' },
            issues: [],
            notes: [],
          },
        ],
      });
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
  const review = screen.getByRole('button', { name: 'fishStory.review_submit' });
  expect(review).toBeEnabled();
  fireEvent.click(review);
  await waitFor(() => expect(calls()).toContainEqual(['/api/fish/story/plan', 'POST']));
  expect(calls().some(([p]) => p === '/api/fish/story/render')).toBe(false);

  fireEvent.click(await screen.findByRole('button', { name: 'fishStory.submit_to_fish' }));
  await waitFor(() => expect(calls()).toContainEqual(['/api/fish/story/render', 'POST']));
  const [, init] = mock.api.mock.calls.find(([p]) => p === '/api/fish/story/render')!;
  expect(JSON.parse(init.body)).toMatchObject({
    confirm: true,
    draft: { settings: { request_mode: 'single' } },
  });
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

const JOB = {
  id: 'abc123abc123',
  title: 'The Long Night',
  status: 'partial',
  message: '',
  total: 10,
  done: 7,
  failed: 3,
  started_at: 1760000000,
  finished_at: 1760000090,
  out_dir: 'C:/data/fish_stories/renders/the-long-night-abc123/20261019-130000_abc123abc123',
  joined: 'story.wav',
  format: 'wav',
  model: 's2.1-pro',
  sample_rate: 44100,
  audio_seconds: 61.5,
  warnings: ['Fish returned 24000 Hz although 44100 Hz was requested.'],
};

it('lists past render jobs with progress, shows the log, and resumes only after confirmation', async () => {
  mock.api.mockImplementation((path: string, init?: RequestInit) => {
    if (path === '/api/fish/story/jobs' || path === '/api/fish/story/jobs?limit=20')
      return Promise.resolve({ jobs: [JOB] });
    if (path === `/api/fish/story/jobs/${JOB.id}/log`)
      return Promise.resolve({ log: '2026-10-19 13:00:01 ERROR #4 failed (HTTP 429): rate limited' });
    if (path === `/api/fish/story/jobs/${JOB.id}/resume` && init?.method === 'POST')
      return Promise.resolve({ ...JOB, status: 'running' });
    if (path === '/api/fish/story/drafts') return Promise.resolve({ drafts: [] });
    if (path === '/api/fish/story/defaults') return Promise.resolve(DEFAULTS);
    return Promise.resolve({});
  });
  render(<FishStoryPage />);
  fireEvent.click(await screen.findByRole('button', { name: /fishStory\.jobs_button/ }));

  const row = await screen.findByTestId('job-row');
  expect(within(row).getByText('The Long Night')).toBeInTheDocument();
  expect(within(row).getByText('fishStory.job_partial')).toBeInTheDocument();
  expect(within(row).getByText(/"done":7,"total":10,"failed":3/)).toBeInTheDocument();
  expect(within(row).getByText(/Fish returned 24000 Hz/)).toBeInTheDocument();
  expect(within(row).getByText(JOB.out_dir)).toBeInTheDocument();

  fireEvent.click(within(row).getByRole('button', { name: 'fishStory.jobs_log' }));
  expect(await within(row).findByText(/failed \(HTTP 429\)/)).toBeInTheDocument();

  // Resume needs a second, explicit click and only then calls the API (3 lines left).
  fireEvent.click(within(row).getByRole('button', { name: /fishStory\.jobs_resume \{"count":3\}/ }));
  expect(calls().some(([p]) => String(p).endsWith('/resume'))).toBe(false);
  expect(within(row).getByText(/fishStory\.jobs_resume_confirm/)).toBeInTheDocument();
  fireEvent.click(within(row).getByRole('button', { name: 'fishStory.jobs_resume_yes' }));
  await waitFor(() => expect(calls()).toContainEqual([`/api/fish/story/jobs/${JOB.id}/resume`, 'POST']));
  const [, init] = mock.api.mock.calls.find(([p]) => String(p).endsWith('/resume'))!;
  expect(JSON.parse(init.body)).toEqual({ confirm: true });
});

it('saves the output folder and render defaults to Settings (local calls only)', async () => {
  render(<FishStoryPage />);
  const folder = (await screen.findByPlaceholderText('C:/data/fish_stories/renders')) as HTMLInputElement;
  fireEvent.change(folder, { target: { value: 'D:\\Stories' } });
  fireEvent.click(screen.getByRole('button', { name: 'fishStory.output_folder_save' }));
  await waitFor(() =>
    expect(mock.api).toHaveBeenCalledWith('/api/fish/settings', {
      method: 'PUT',
      body: JSON.stringify({ output_dir: 'D:\\Stories' }),
    }),
  );

  fireEvent.change(screen.getByLabelText('fishStory.format'), { target: { value: 'mp3' } });
  fireEvent.change(screen.getByLabelText('fishStory.mp3_bitrate'), { target: { value: '192' } });
  fireEvent.click(screen.getByRole('button', { name: 'fishStory.save_defaults' }));
  await waitFor(() => {
    const saved = mock.api.mock.calls.filter(([p, i]) => p === '/api/fish/settings' && i?.method === 'PUT');
    expect(saved).toHaveLength(2);
    // Blank tuning fields are sent as null so a stored value (e.g. top_p 1.0) is cleared, not inherited.
    expect(JSON.parse(saved[1][1].body)).toMatchObject({
      format: 'mp3',
      mp3_bitrate: 192,
      top_p: null,
      temperature: null,
      repetition_penalty: null,
      condition_on_previous_chunks: null,
    });
  });
  expect(calls().some(([p]) => p === '/api/fish/story/render' || p === '/api/fish/balance')).toBe(false);
});

it('adds a voice by id or link, and from a public library search, without a render', async () => {
  await loadStory();
  fireEvent.click(screen.getByRole('button', { name: /fishStory\.more_voices/ }));
  const id = await screen.findByLabelText('fishStory.voice_id');
  fireEvent.change(id, { target: { value: 'not a valid id!' } });
  expect(screen.getByRole('alert')).toHaveTextContent('fishStory.voice_id_invalid');
  fireEvent.change(id, { target: { value: 'https://fish.audio/m/abc123def456/' } });
  fireEvent.change(screen.getByLabelText('fishStory.voice_name'), { target: { value: 'Friend' } });
  fireEvent.click(screen.getByRole('button', { name: 'fishStory.voice_add' }));

  fireEvent.change(screen.getByLabelText('fishStory.library_search'), { target: { value: 'narrator' } });
  fireEvent.click(screen.getByRole('button', { name: 'fishStory.library_go' }));
  await waitFor(() => expect(calls()).toContainEqual(['/api/fish/voices?scope=library&title=narrator', 'GET']));
  fireEvent.click(await screen.findByRole('button', { name: /voice_add_named/ }));

  fireEvent.click(screen.getByRole('button', { name: 'fishStory.done' }));
  expect(await screen.findAllByRole('option', { name: 'Friend' })).not.toHaveLength(0);
  expect(screen.getAllByRole('option', { name: 'Deep Narrator' }).length).toBeGreaterThan(0);
  expect(calls().some(([p]) => p === '/api/fish/story/render')).toBe(false);
});

it('previews the raw requests with a schema verdict and sends nothing to Fish', async () => {
  await loadStory();
  fireEvent.click(screen.getByRole('switch', { name: 'fishStory.multi_speaker' }));
  fireEvent.click(screen.getByRole('button', { name: 'fishStory.apply_settings' }));
  fireEvent.click(screen.getByRole('button', { name: /fishStory\.preview_button/ }));
  await screen.findByText(/fishStory\.preview_ok/);
  expect(screen.getByLabelText(/fishStory\.preview_body/)).toHaveTextContent('"format": "wav"');
  expect(screen.getAllByText(/POST https:\/\/api\.fish\.audio\/v1\/tts/).length).toBeGreaterThan(0);
  const [, init] = mock.api.mock.calls.find(([p]) => p === '/api/fish/story/preview')!;
  expect(JSON.parse(init.body).draft.chunks).toHaveLength(20);
  expect(JSON.parse(init.body).draft.settings.request_mode).toBe('multi');
  expect(calls().some(([p]) => p === '/api/fish/story/render')).toBe(false);
});

it('greys out the tuning fields while "send only the basics" is on, since they are not sent', async () => {
  render(<FishStoryPage />);
  const temperature = (await screen.findByLabelText('fishStory.temperature')) as HTMLInputElement;
  expect(temperature).toBeEnabled();
  fireEvent.click(screen.getByRole('switch', { name: 'fishStory.minimal_request' }));
  expect(temperature).toBeDisabled();
  expect(screen.getAllByText('fishStory.minimal_ignores').length).toBeGreaterThan(0);
  fireEvent.click(screen.getByRole('switch', { name: 'fishStory.minimal_request' }));
  expect(temperature).toBeEnabled();
});

it('shows every setting in effect with its source and whether it is sent, plus the curl command', async () => {
  await loadStory();
  fireEvent.click(screen.getByRole('button', { name: /fishStory\.preview_button/ }));
  const rows = await screen.findAllByTestId('setting-row');
  expect(rows).toHaveLength(3);
  expect(within(rows[0]).getByText('temperature')).toBeInTheDocument();
  expect(within(rows[0]).getByText('fishStory.eff_source_story')).toBeInTheDocument();
  expect(within(rows[1]).getByText('fishStory.eff_sent_no')).toBeInTheDocument();
  expect(within(rows[2]).getByText('fishStory.eff_source_saved_default')).toBeInTheDocument();
  expect(screen.getByText(/curl -X POST/)).toBeVisible();
});

it('imports a public collection by link and adds all of its voices to the story', async () => {
  await loadStory();
  fireEvent.click(screen.getByRole('button', { name: /fishStory\.more_voices/ }));
  fireEvent.change(await screen.findByLabelText('fishStory.collection_url'), {
    target: { value: 'https://fish.audio/collections/63bdf88542744737926c783b476c91b7' },
  });
  fireEvent.click(screen.getByRole('button', { name: 'fishStory.collection_import' }));
  await screen.findByTestId('collection-results');
  const [, init] = mock.api.mock.calls.find(([p]) => p === '/api/fish/collection')!;
  expect(JSON.parse(init.body)).toEqual({ url: 'https://fish.audio/collections/63bdf88542744737926c783b476c91b7' });
  fireEvent.click(screen.getByRole('button', { name: /fishStory\.collection_add_all/ }));
  fireEvent.click(screen.getByRole('button', { name: 'fishStory.done' }));
  expect(await screen.findAllByRole('option', { name: 'Moon Goddess' })).not.toHaveLength(0);
  expect(screen.getAllByRole('option', { name: 'Nihilister' }).length).toBeGreaterThan(0);
  expect(calls().some(([p]) => p === '/api/fish/story/render')).toBe(false);
});

it('holds setting edits until Apply, so previews only use what was applied', async () => {
  await loadStory();
  const apply = screen.getByRole('button', { name: 'fishStory.apply_settings' });
  expect(apply).toBeDisabled();
  expect(screen.getByText('fishStory.settings_in_effect')).toBeInTheDocument();

  fireEvent.click(screen.getByRole('switch', { name: 'fishStory.multi_speaker' }));
  expect(apply).toBeEnabled();
  expect(screen.getByText(/fishStory\.settings_unapplied \{"count":1\}/)).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: /fishStory\.preview_button/ }));
  await screen.findByText(/fishStory\.preview_ok/);
  const [, before] = mock.api.mock.calls.find(([p]) => p === '/api/fish/story/preview')!;
  expect(JSON.parse(before.body).draft.settings.request_mode).not.toBe('multi');
  fireEvent.click(screen.getByRole('button', { name: 'fishStory.done' }));

  fireEvent.click(apply);
  expect(apply).toBeDisabled();
  expect(screen.getByText('fishStory.settings_in_effect')).toBeInTheDocument();
});

it('discards unapplied setting edits', async () => {
  await loadStory();
  const toggle = screen.getByRole('switch', { name: 'fishStory.multi_speaker' });
  fireEvent.click(toggle);
  expect(toggle).toBeChecked();
  fireEvent.click(screen.getByRole('button', { name: 'fishStory.discard_settings' }));
  expect(toggle).not.toBeChecked();
});

it('loads a past run into the board with its settings, without contacting Fish', async () => {
  const base = mock.api.getMockImplementation()!;
  mock.api.mockImplementation((path: string, init?: RequestInit) => {
    if (path === '/api/fish/story/jobs' || path === '/api/fish/story/jobs?limit=20')
      return Promise.resolve({ jobs: [{ ...JOB, status: 'done', failed: 0, done: 10 }] });
    if (path === `/api/fish/story/jobs/${JOB.id}/load`)
      return Promise.resolve({
        draft: {
          title: 'Old run',
          chunks: [{ id: 'a', text: 'Hello again.', voice: null, speaker: null }],
          voices: {},
          cast: {},
          settings: { temperature: 0.4, request_mode: 'multi' },
        },
        lines: [],
      });
    return base(path, init);
  });
  render(<FishStoryPage />);
  fireEvent.click(await screen.findByRole('button', { name: /fishStory\.jobs_button/ }));
  const row = await screen.findByTestId('job-row');
  fireEvent.click(within(row).getByRole('button', { name: 'fishStory.jobs_load' }));

  expect(await screen.findByText(/fishStory\.jobs_loaded/)).toBeInTheDocument();
  expect(screen.getByText('Hello again.')).toBeInTheDocument();
  expect(screen.getByRole('switch', { name: 'fishStory.multi_speaker' })).toBeChecked();
  expect(calls().some(([p]) => p === '/api/fish/story/render')).toBe(false);
});
