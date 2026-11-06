import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';

const mock = vi.hoisted(() => ({ api: vi.fn() }));

vi.mock('@/lib/api/client', async (load) => {
  const actual = await load<typeof import('@/lib/api/client')>();
  return { ...actual, apiJson: mock.api };
});
vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (key: string, o?: Record<string, unknown>) => (o?.index ? `${key}:${o.index}` : key) }),
}));

import { RunLines } from './jobs-dialog';

const RUN = {
  draft: { chunks: [{ id: 'a', text: 'Hello.' }, { id: 'b', text: 'Wrong voice line.' }] },
  lines: [
    { id: 'a', index: 1, status: 'done', file: '001_narration_N.wav', voice_name: 'N', duration_s: 1.2 },
    { id: 'b', index: 2, status: 'done', file: '002_Shadow_sun_2.wav', voice_name: 'sun_2', duration_s: 2.4, effect: 'inner_monologue', replace_error: 'bad request' },
  ],
};

beforeEach(() => {
  mock.api.mockImplementation((path: string) => (path.endsWith('/load') ? Promise.resolve(RUN) : Promise.resolve({ id: 'job1' })));
});
afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

function setup() {
  const onChanged = vi.fn();
  const draft = { title: 'S', chunks: [{ id: 'b', text: 'Wrong voice line.' }], cast: { Shadow: 'right' } };
  render(<RunLines jobId="job1" status="done" onError={vi.fn()} getDraft={() => draft} onChanged={onChanged} />);
  return { onChanged, draft };
}

it('shows each line with its effect and the reason an earlier replacement kept the old audio', async () => {
  setup();
  await waitFor(() => expect(document.body.textContent).toContain('Wrong voice line.'));
  expect(document.body.textContent).toContain('{inner_monologue}');
  expect(document.body.textContent).toContain('fishStory.replace_kept');
});

it('replaces ticked lines only after a confirmation, with the board draft, in the same run', async () => {
  const { onChanged, draft } = setup();
  await waitFor(() => expect(screen.getByLabelText('fishStory.replace_pick:2')).toBeTruthy());
  expect(document.body.textContent).not.toContain('fishStory.replace_lines'); // nothing ticked yet
  fireEvent.click(screen.getByLabelText('fishStory.replace_pick:2'));
  fireEvent.click([...document.querySelectorAll('button')].find((b) => b.textContent === 'fishStory.replace_lines')!);
  expect(mock.api).not.toHaveBeenCalledWith('/api/fish/story/jobs/job1/replace', expect.anything()); // the confirm step comes first
  fireEvent.click([...document.querySelectorAll('button')].find((b) => b.textContent === 'fishStory.replace_go')!);
  await waitFor(() =>
    expect(mock.api).toHaveBeenCalledWith('/api/fish/story/jobs/job1/replace', {
      method: 'POST',
      body: JSON.stringify({ confirm: true, draft, chunk_ids: ['b'] }),
    }),
  );
  await waitFor(() => expect(onChanged).toHaveBeenCalled());
});

it('re-applies effects without a confirmation because Fish is not contacted', async () => {
  const { draft } = setup();
  await waitFor(() => expect(screen.getByLabelText('fishStory.replace_pick:1')).toBeTruthy());
  fireEvent.click(screen.getByLabelText('fishStory.replace_pick:1'));
  fireEvent.click([...document.querySelectorAll('button')].find((b) => b.textContent === 'fishStory.reapply_effects')!);
  await waitFor(() =>
    expect(mock.api).toHaveBeenCalledWith('/api/fish/story/jobs/job1/reapply-effects', {
      method: 'POST',
      body: JSON.stringify({ draft, chunk_ids: ['a'] }),
    }),
  );
});
