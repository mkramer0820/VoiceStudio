import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { apiJson, describeError } from '@/lib/api/client';

export interface DatasetSpeaker {
  voice_id: string;
  name: string;
  aliases: string[];
  lines: number;
  duplicates: number;
  seconds: number;
  avg_s: number;
  chapters: number;
  top_tags: Array<[string, number]>;
}

export interface DatasetRun {
  run_id: string;
  chapter: string;
  run: string;
  lines: number;
  failed: number;
  missing: number;
}

export interface DatasetOverview {
  root: string | null;
  exists: boolean;
  speakers: DatasetSpeaker[];
  runs: DatasetRun[];
  chapters: string[];
  /** Inline tags across the usable clips, most common first. */
  tags?: Array<[string, number]>;
  totals: { lines: number; duplicates: number; seconds: number; runs: number } | null;
}

export interface DatasetLine {
  key: string;
  chapter: string;
  run: string;
  index: number;
  voice_id: string;
  voice_name: string;
  text: string;
  plain: string;
  tags: string[];
  words: number;
  duration_s: number;
  duplicate: boolean;
}

export interface DatasetFilters {
  q: string;
  tag: string;
  chapter: string;
  minDur: string;
  maxDur: string;
  minWords: string;
  duplicates: boolean;
}

export const EMPTY_FILTERS: DatasetFilters = {
  q: '',
  tag: '',
  chapter: '',
  minDur: '',
  maxDur: '',
  minWords: '',
  duplicates: false,
};

export interface ExportOptions {
  dest: string;
  name: string;
  formats: string[];
  text_mode: 'keep' | 'strip';
  sample_rate: number | null;
  val_fraction: number;
  /** With text_mode strip: tags to leave in, each optionally `written=exported`. */
  keep_tags: string[];
}

export interface ExportResult {
  path: string;
  clips: number;
  hours: number;
  speakers: Record<string, { clips: number; seconds: number }>;
}

export const PAGE_SIZE = 50;
const num = (raw: string) => (raw.trim() === '' || !Number.isFinite(Number(raw)) ? null : Number(raw));

/**
 * Training-data browser state. Everything is a local call: it indexes the story renders already on
 * disk and never contacts Fish. Excluding a clip only affects the next export.
 */
export function useFishDataset() {
  const [overview, setOverview] = useState<DatasetOverview | null>(null);
  const [scanning, setScanning] = useState(false);
  const [error, setError] = useState('');
  const [voices, setVoices] = useState<ReadonlySet<string>>(new Set());
  const [filters, setFilters] = useState<DatasetFilters>(EMPTY_FILTERS);
  const [excluded, setExcluded] = useState<ReadonlySet<string>>(new Set());
  const [page, setPage] = useState(0);
  const [lines, setLines] = useState<DatasetLine[]>([]);
  const [total, setTotal] = useState(0);
  const [seconds, setSeconds] = useState(0);
  const [loading, setLoading] = useState(false);
  const request = useRef(0);

  const spec = useMemo(
    () => ({
      voice_ids: [...voices],
      chapters: filters.chapter ? [filters.chapter] : [],
      q: filters.q,
      tag: filters.tag,
      min_dur: num(filters.minDur),
      max_dur: num(filters.maxDur),
      min_words: num(filters.minWords),
      duplicates: filters.duplicates,
      exclude_keys: [...excluded],
    }),
    [voices, filters, excluded],
  );

  const refresh = useCallback(async (rescan = false) => {
    setScanning(true);
    try {
      setOverview(await apiJson<DatasetOverview>(`/api/fish/dataset${rescan ? '?refresh=true' : ''}`));
    } catch (reason) {
      setError(describeError(reason));
    } finally {
      setScanning(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  // Any filter change returns to the first page.
  useEffect(() => setPage(0), [voices, filters]);

  useEffect(() => {
    if (!overview?.exists) {
      setLines([]);
      setTotal(0);
      return;
    }
    const id = ++request.current;
    setLoading(true);
    const timer = setTimeout(() => {
      apiJson<{ total: number; seconds: number; lines: DatasetLine[] }>(
        `/api/fish/dataset/lines?offset=${page * PAGE_SIZE}&limit=${PAGE_SIZE}`,
        { method: 'POST', body: JSON.stringify(spec) },
      )
        .then((r) => {
          if (id !== request.current) return;
          setLines(r.lines);
          setTotal(r.total);
          setSeconds(r.seconds);
        })
        .catch((reason) => id === request.current && setError(describeError(reason)))
        .finally(() => id === request.current && setLoading(false));
    }, 250);
    return () => clearTimeout(timer);
  }, [overview?.exists, overview?.root, spec, page]);

  const saveRoot = useCallback(
    async (path: string | null): Promise<boolean> => {
      setError('');
      try {
        await apiJson('/api/fish/dataset/root', { method: 'PUT', body: JSON.stringify({ path }) });
        setVoices(new Set());
        setExcluded(new Set());
        await refresh(true);
        return true;
      } catch (reason) {
        setError(describeError(reason));
        return false;
      }
    },
    [refresh],
  );

  const toggleVoice = useCallback(
    (id: string) =>
      setVoices((current) => {
        const next = new Set(current);
        if (next.has(id)) next.delete(id);
        else next.add(id);
        return next;
      }),
    [],
  );

  const toggleExcluded = useCallback(
    (key: string) =>
      setExcluded((current) => {
        const next = new Set(current);
        if (next.has(key)) next.delete(key);
        else next.add(key);
        return next;
      }),
    [],
  );

  const runExport = useCallback(
    async (options: ExportOptions): Promise<ExportResult | null> => {
      setError('');
      try {
        return await apiJson<ExportResult>('/api/fish/dataset/export', {
          method: 'POST',
          body: JSON.stringify({ ...spec, ...options }),
        });
      } catch (reason) {
        setError(describeError(reason));
        return null;
      }
    },
    [spec],
  );

  return {
    overview,
    scanning,
    error,
    clearError: () => setError(''),
    setError,
    voices,
    toggleVoice,
    clearVoices: () => setVoices(new Set()),
    filters,
    setFilters,
    excluded,
    toggleExcluded,
    clearExcluded: () => setExcluded(new Set()),
    page,
    setPage,
    lines,
    total,
    seconds,
    loading,
    refresh,
    saveRoot,
    runExport,
  };
}
