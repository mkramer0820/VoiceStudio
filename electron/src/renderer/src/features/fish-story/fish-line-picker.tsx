import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { apiFetch, apiJson, describeError } from '@/lib/api/client';
import { selectClass } from '../qwen-story/emotion-picker';

interface Line {
  jobId: string;
  file: string;
  label: string;
}

/**
 * Lines of the latest finished Fish render, as sample clips for the effect editor. The untouched Fish response is preferred, so an effect is
 * judged on what Fish really returned (a line that already carries an effect is skipped).
 */
export function FishLinePicker({ onPick }: { onPick: (blob: Blob, name: string) => void }) {
  const { t } = useTranslation();
  const [lines, setLines] = useState<Line[]>([]);
  const [error, setError] = useState('');

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const { jobs } = await apiJson<{ jobs: Array<{ id: string; status: string }> }>('/api/fish/story/jobs?limit=10');
        const job = jobs.find((j) => j.status === 'done' || j.status === 'partial');
        if (!job) return;
        const run = await apiJson<{
          draft: { chunks: Array<{ id: string; text: string }> };
          lines: Array<{ id: string; index: number; status: string; file: string | null; effect?: string | null }>;
        }>(`/api/fish/story/jobs/${job.id}/load`);
        const text = new Map(run.draft.chunks.map((c) => [c.id, c.text]));
        const rows = run.lines
          .filter((l) => l.status === 'done' && l.file && !l.effect)
          .slice(0, 40)
          .map((l) => ({ jobId: job.id, file: l.file as string, label: `#${l.index} ${(text.get(l.id) ?? '').slice(0, 48)}` }));
        if (alive) setLines(rows);
      } catch (cause) {
        if (alive) setError(describeError(cause));
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  if (error) return <span className="text-xs text-destructive">{error}</span>;
  if (lines.length === 0) return null;
  return (
    <select
      aria-label={t('fishStory.effect_sample_line')}
      className={`${selectClass} w-64`}
      value=""
      onChange={async (e) => {
        const line = lines[Number(e.target.value)];
        if (!line) return;
        try {
          const res = await apiFetch(`/api/fish/story/jobs/${line.jobId}/audio/${encodeURIComponent(line.file)}`);
          onPick(await res.blob(), line.file);
        } catch (cause) {
          setError(describeError(cause));
        }
      }}
    >
      <option value="">{t('fishStory.effect_sample_line')}</option>
      {lines.map((l, i) => (
        <option key={`${l.jobId}-${l.file}`} value={i}>
          {l.label}
        </option>
      ))}
    </select>
  );
}
