import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const mock = vi.hoisted(() => ({ api: vi.fn() }));

vi.mock('@/lib/api/client', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/lib/api/client')>()),
  apiJson: mock.api,
}));
vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, values?: Record<string, unknown>) => (values ? `${key} ${JSON.stringify(values)}` : key),
  }),
}));

import { TrainDialog } from './train-dialog';

const DEFAULTS = {
  steps: 2000,
  learning_rate: 1e-4,
  batch_tokens: 4096,
  gradient_accumulation_steps: 1,
  lora_r: 16,
  lora_alpha: 32,
  lora_dropout: 0.05,
  train_heads: false,
  seed: 42,
  base_checkpoint: 'k2-fsa/OmniVoice',
  tokenizer_path: 'eustlb/higgs-audio-v2-tokenizer',
  python: 'C:\\py\\python.exe',
  allow_download: false,
};
const RUN = {
  id: 'r1',
  name: 'narrator',
  status: 'running',
  phase: 'training',
  message: '',
  created_at: 1,
  started_at: 1,
  finished_at: null,
  step: 25,
  steps: 100,
  loss: 3.25,
  eval_loss: 3.1,
  dry_run: false,
  resumes: 0,
  adapter: null,
  run_dir: 'C:\\data\\fish_training\\run_x',
  export_dir: 'D:\\out\\ov',
  blockers: [],
  warnings: [],
  ship_ready: false,
  verdict: null,
};
let state: { enabled: boolean; runs: unknown[] };

const posted = (path: string) => mock.api.mock.calls.filter(([p]) => p === path);

beforeEach(() => {
  state = { enabled: false, runs: [] };
  mock.api.mockImplementation((path: string, init?: RequestInit) => {
    if (path === '/api/fish/training')
      return Promise.resolve({
        enabled: state.enabled,
        defaults: DEFAULTS,
        runs: state.runs,
        limits: { min_clips: 20, comfortable_clips: 300, comfortable_minutes: 30, vram_gb: 7.5 },
      });
    if (path === '/api/fish/training/enabled') {
      state.enabled = JSON.parse(String(init?.body)).enabled;
      return Promise.resolve({ enabled: state.enabled });
    }
    if (path === '/api/fish/training/dry-run')
      return Promise.resolve({
        ...RUN,
        status: 'dry_run',
        dry_run: true,
        warnings: ['small set'],
        report: {
          export: { train_clips: 30, val_clips: 5, seconds: 600, loudness_matched: true, warnings: [] },
          hardware: { ok: false, device: 'cpu', blockers: ['No GPU was found.'], warnings: [] },
          commands: {},
          heldout_clips: 5,
          ready: false,
        },
      });
    if (path === '/api/fish/training/runs') {
      state.runs = [RUN];
      return Promise.resolve(RUN);
    }
    if (path.endsWith('/log')) return Promise.resolve({ log: 'Step 25 | train/loss: 3.2500' });
    return Promise.resolve({});
  });
});
afterEach(() => {
  cleanup();
  mock.api.mockReset();
});

describe('TrainDialog', () => {
  it('stays off until the user opts in, and shows nothing to start before that', async () => {
    render(<TrainDialog open onOpenChange={() => undefined} />);
    expect(await screen.findByTestId('train-off')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /fishTraining\.start$/ })).not.toBeInTheDocument();
    expect(posted('/api/fish/training/runs')).toHaveLength(0);
    fireEvent.click(screen.getByRole('button', { name: /fishTraining\.off_enable/ }));
    expect(await screen.findByLabelText('fishTraining.export_dir')).toBeInTheDocument();
    expect(JSON.parse(String(posted('/api/fish/training/enabled')[0][1].body))).toEqual({ enabled: true });
  });

  it('does not touch the API while closed', () => {
    render(<TrainDialog open={false} onOpenChange={() => undefined} />);
    expect(mock.api).not.toHaveBeenCalled();
  });

  it('checks a dry run, shows why this machine cannot train, and starts only after the confirmation', async () => {
    state.enabled = true;
    render(<TrainDialog open onOpenChange={() => undefined} initialExportDir={'D:\\out\\ov'} />);
    const dir = await screen.findByLabelText('fishTraining.export_dir');
    expect(dir).toHaveValue('D:\\out\\ov');
    const start = screen.getByRole('button', { name: /fishTraining\.start$/ });
    expect(start).toBeDisabled();

    fireEvent.change(screen.getByLabelText('fishTraining.steps'), { target: { value: '100' } });
    fireEvent.click(screen.getByRole('button', { name: /fishTraining\.check/ }));
    const report = await screen.findByTestId('train-report');
    expect(within(report).getByText('fishTraining.report_blocked')).toBeInTheDocument();
    expect(within(report).getByText('No GPU was found.')).toBeInTheDocument();
    expect(within(report).getByText('small set')).toBeInTheDocument();
    expect(JSON.parse(String(posted('/api/fish/training/dry-run')[0][1].body))).toMatchObject({
      export_dir: 'D:\\out\\ov',
      options: { steps: 100, train_heads: false, allow_download: false, python: null },
    });
    expect(posted('/api/fish/training/runs')).toHaveLength(0);

    fireEvent.click(screen.getByLabelText('fishTraining.start_confirm'));
    expect(start).toBeEnabled();
    fireEvent.click(start);
    await waitFor(() => expect(posted('/api/fish/training/runs')).toHaveLength(1));
    expect(JSON.parse(String(posted('/api/fish/training/runs')[0][1].body))).toMatchObject({
      confirm: true,
      export_dir: 'D:\\out\\ov',
    });
  });

  it('lists runs with progress, a live log, cancel and resume', async () => {
    state.enabled = true;
    state.runs = [
      RUN,
      { ...RUN, id: 'r2', name: 'older', status: 'cancelled', verdict: 'incomplete', step: 10, message: 'Cancelled at step 10.' },
    ];
    render(<TrainDialog open onOpenChange={() => undefined} />);
    const rows = await screen.findAllByTestId('train-run');
    expect(rows).toHaveLength(2);
    expect(within(rows[0]).getByRole('progressbar')).toHaveAttribute('aria-valuenow', '25');
    expect(within(rows[0]).getByText(/fishTraining\.progress .*"step":25/)).toBeInTheDocument();

    fireEvent.click(within(rows[0]).getByRole('button', { name: /fishTraining\.log/ }));
    expect(await screen.findByTestId('train-log')).toHaveTextContent('Step 25 | train/loss: 3.2500');

    fireEvent.click(within(rows[0]).getByRole('button', { name: /fishTraining\.cancel/ }));
    await waitFor(() => expect(posted('/api/fish/training/runs/r1/cancel')).toHaveLength(1));

    expect(within(rows[1]).getByText('fishTraining.verdict_incomplete')).toBeInTheDocument();
    expect(within(rows[1]).queryByRole('button', { name: /fishTraining\.cancel/ })).not.toBeInTheDocument();
    fireEvent.click(within(rows[1]).getByRole('button', { name: /fishTraining\.resume/ }));
    await waitFor(() => expect(posted('/api/fish/training/runs/r2/resume')).toHaveLength(1));
  });

  it('explains an API error instead of failing silently', async () => {
    state.enabled = true;
    mock.api.mockImplementation((path: string) => {
      if (path === '/api/fish/training')
        return Promise.resolve({ enabled: true, defaults: DEFAULTS, runs: [], limits: { min_clips: 20, comfortable_clips: 300, comfortable_minutes: 30, vram_gb: 7.5 } });
      return Promise.reject(new Error('That export folder does not exist.'));
    });
    render(<TrainDialog open onOpenChange={() => undefined} />);
    fireEvent.change(await screen.findByLabelText('fishTraining.export_dir'), { target: { value: 'D:\\nope' } });
    fireEvent.click(screen.getByRole('button', { name: /fishTraining\.check/ }));
    expect(await screen.findByRole('alert')).toHaveTextContent('That export folder does not exist.');
  });
});
