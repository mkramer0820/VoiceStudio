import { useCallback, useEffect, useState } from 'react';
import { apiJson, describeError } from '@/lib/api/client';

export interface TrainOptions {
  steps: number;
  learning_rate: number;
  batch_tokens: number;
  gradient_accumulation_steps: number;
  lora_r: number;
  lora_alpha: number;
  lora_dropout: number;
  train_heads: boolean;
  seed: number;
  base_checkpoint: string;
  tokenizer_path: string;
  python: string | null;
  allow_download: boolean;
}

export type RunStatus = 'running' | 'done' | 'failed' | 'cancelled' | 'interrupted' | 'dry_run';

export interface TrainingRun {
  id: string;
  name: string;
  status: RunStatus;
  phase: string;
  message: string;
  created_at: number;
  started_at: number | null;
  finished_at: number | null;
  step: number;
  steps: number;
  loss: number | null;
  eval_loss: number | null;
  dry_run: boolean;
  resumes: number;
  adapter: string | null;
  run_dir: string;
  export_dir: string;
  blockers: string[];
  warnings: string[];
  ship_ready: boolean;
  verdict: 'passed' | 'incomplete' | 'failed' | null;
  ready?: boolean;
}

export interface DryRunReport {
  export: { train_clips: number; val_clips: number; seconds: number; loudness_matched: boolean | null; warnings: string[] };
  hardware: { ok: boolean; device: string; blockers: string[]; warnings: string[] };
  commands: Record<string, string[]>;
  heldout_clips: number;
  ready: boolean;
}

export interface TrainingOverview {
  enabled: boolean;
  defaults: TrainOptions;
  runs: TrainingRun[];
  limits: { min_clips: number; comfortable_clips: number; comfortable_minutes: number; vram_gb: number };
}

export type DryRunResult = TrainingRun & { report: DryRunReport | null };

const POLL_MS = 4000;
const POLL_BUSY_MS = 1500;

/**
 * Local LoRA training runs. Nothing is fetched until `active` is true (the dialog is open), and
 * nothing is started without an explicit call: enabling, checking (dry run), starting, resuming.
 */
export function useFishTraining(active: boolean) {
  const [overview, setOverview] = useState<TrainingOverview | null>(null);
  const [error, setError] = useState('');

  const refresh = useCallback(async () => {
    try {
      setOverview(await apiJson<TrainingOverview>('/api/fish/training'));
    } catch (reason) {
      setError(describeError(reason));
    }
  }, []);

  const busy = overview?.runs?.some((r) => r.status === 'running') ?? false;
  useEffect(() => {
    if (!active) return;
    void refresh();
    const timer = setInterval(() => void refresh(), busy ? POLL_BUSY_MS : POLL_MS);
    return () => clearInterval(timer);
  }, [active, refresh, busy]);

  const guard = useCallback(
    async <T,>(call: () => Promise<T>): Promise<T | null> => {
      setError('');
      try {
        const result = await call();
        await refresh();
        return result;
      } catch (reason) {
        setError(describeError(reason));
        return null;
      }
    },
    [refresh],
  );

  const post = (path: string, body: unknown) =>
    apiJson<TrainingRun>(path, { method: 'POST', body: JSON.stringify(body) });

  return {
    overview,
    error,
    clearError: () => setError(''),
    setError,
    refresh,
    setEnabled: (enabled: boolean) =>
      guard(() => apiJson('/api/fish/training/enabled', { method: 'PUT', body: JSON.stringify({ enabled }) })),
    dryRun: (exportDir: string, options: Partial<TrainOptions>) =>
      guard(() =>
        apiJson<DryRunResult>('/api/fish/training/dry-run', {
          method: 'POST',
          body: JSON.stringify({ export_dir: exportDir, options }),
        }),
      ),
    start: (exportDir: string, options: Partial<TrainOptions>) =>
      guard(() => post('/api/fish/training/runs', { confirm: true, export_dir: exportDir, options })),
    cancel: (id: string) => guard(() => post(`/api/fish/training/runs/${id}/cancel`, {})),
    resume: (id: string) => guard(() => post(`/api/fish/training/runs/${id}/resume`, { confirm: true })),
    loadLog: async (id: string): Promise<string> => {
      try {
        return (await apiJson<{ log: string }>(`/api/fish/training/runs/${id}/log`)).log;
      } catch (reason) {
        setError(describeError(reason));
        return '';
      }
    },
  };
}
