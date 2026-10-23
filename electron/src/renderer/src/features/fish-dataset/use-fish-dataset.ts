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
  /** Non-verbal tags the OmniVoice export accepts. */
  omnivoice_tags?: string[];
  totals: { lines: number; duplicates: number; seconds: number; runs: number } | null;
}

export interface ClipStats {
  peak: number;
  rms_db: number;
  clip: number;
  lead_ms: number;
  tail_ms: number;
  pauses: number;
  max_gap_ms: number;
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
  stats?: ClipStats | null;
}

export interface DatasetFacets {
  clips: number;
  seconds: number;
  duration: Array<{ from: number; to: number | null; clips: number }>;
  tags: Array<[string, number]>;
  chapters: Array<{ chapter: string; clips: number; seconds: number; rms_db: number | null; peak: number | null }>;
  words: { min: number; max: number; avg: number } | null;
  quality: {
    analyzed: number;
    total: number;
    clipping: number;
    loud: number;
    quiet: number;
    choppy: number;
    long_edges: number;
  };
}

export interface AnalysisStatus {
  running: boolean;
  done: number;
  total: number;
  failed: number;
  cached: number;
}

export type SortKey = 'order' | 'duration_desc' | 'duration_asc' | 'words_desc' | 'words_asc' | 'random';

export interface DatasetFilters {
  q: string;
  chapters: string[];
  /** Clips must carry every one of these tags. */
  tagsAll: string[];
  /** Clips must carry none of these tags. */
  tagsNone: string[];
  minDur: string;
  maxDur: string;
  minWords: string;
  maxWords: string;
  duplicates: boolean;
  sort: SortKey;
  seed: number;
  /** Only the clips you unticked (to review or put back). */
  excludedOnly: boolean;
  hideClipped: boolean;
  maxPeak: string;
  minRms: string;
  maxRms: string;
  maxLead: string;
  maxTail: string;
  maxPauses: string;
  maxGap: string;
}

export const EMPTY_FILTERS: DatasetFilters = {
  q: '',
  chapters: [],
  tagsAll: [],
  tagsNone: [],
  minDur: '',
  maxDur: '',
  minWords: '',
  maxWords: '',
  duplicates: false,
  sort: 'order',
  seed: 0,
  excludedOnly: false,
  hideClipped: false,
  maxPeak: '',
  minRms: '',
  maxRms: '',
  maxLead: '',
  maxTail: '',
  maxPauses: '',
  maxGap: '',
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
  /** Bring every clip to this average level (dB RMS); null leaves levels alone. */
  loudness_db: number | null;
}

export interface ExportResult {
  path: string;
  clips: number;
  hours: number;
  speakers: Record<string, { clips: number; seconds: number }>;
}

export const PAGE_SIZE = 50;
const num = (raw: string) => (raw.trim() === '' || !Number.isFinite(Number(raw)) ? null : Number(raw));

/** The number of filters that differ from the defaults, for the Reset button and the sticky bar. */
export function activeFilterCount(f: DatasetFilters): number {
  const base = EMPTY_FILTERS;
  return (Object.keys(base) as Array<keyof DatasetFilters>).filter((key) => {
    if (key === 'seed') return false;
    return JSON.stringify(f[key]) !== JSON.stringify(base[key]);
  }).length;
}

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
  const [facets, setFacets] = useState<DatasetFacets | null>(null);
  const [total, setTotal] = useState(0);
  const [seconds, setSeconds] = useState(0);
  const [selected, setSelected] = useState(0);
  const [selectedSeconds, setSelectedSeconds] = useState(0);
  const [loading, setLoading] = useState(false);
  const [analysis, setAnalysis] = useState<AnalysisStatus | null>(null);
  const [tick, setTick] = useState(0);
  const request = useRef(0);

  const spec = useMemo(
    () => ({
      voice_ids: [...voices],
      chapters: filters.chapters,
      q: filters.q,
      tags_all: filters.tagsAll,
      tags_none: filters.tagsNone,
      min_dur: num(filters.minDur),
      max_dur: num(filters.maxDur),
      min_words: num(filters.minWords),
      max_words: num(filters.maxWords),
      duplicates: filters.duplicates,
      sort: filters.sort,
      seed: filters.seed,
      max_clip: filters.hideClipped ? 0.0005 : null,
      max_peak: num(filters.maxPeak),
      min_rms_db: num(filters.minRms),
      max_rms_db: num(filters.maxRms),
      max_lead_ms: num(filters.maxLead),
      max_tail_ms: num(filters.maxTail),
      max_pauses: num(filters.maxPauses),
      max_gap_ms: num(filters.maxGap),
      ...(filters.excludedOnly
        ? { only_keys: [...excluded], exclude_keys: [] as string[] }
        : { exclude_keys: [...excluded] }),
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
  const filterKey = JSON.stringify([...voices, filters, filters.excludedOnly ? [...excluded] : 0]);
  useEffect(() => setPage(0), [filterKey]);

  useEffect(() => {
    if (!overview?.exists) {
      setLines([]);
      setTotal(0);
      setSelected(0);
      setFacets(null);
      return;
    }
    const id = ++request.current;
    setLoading(true);
    const timer = setTimeout(() => {
      const body = JSON.stringify(spec);
      Promise.all([
        apiJson<{ total: number; seconds: number; selected?: number; selected_seconds?: number; lines: DatasetLine[] }>(
          `/api/fish/dataset/lines?offset=${page * PAGE_SIZE}&limit=${PAGE_SIZE}`,
          { method: 'POST', body },
        ),
        apiJson<DatasetFacets>('/api/fish/dataset/facets', { method: 'POST', body }),
      ])
        .then(([r, f]) => {
          if (id !== request.current) return;
          setLines(r.lines);
          setTotal(r.total);
          setSeconds(r.seconds);
          setSelected(r.selected ?? r.total);
          setSelectedSeconds(r.selected_seconds ?? r.seconds);
          setFacets(f?.duration ? f : null);
        })
        .catch((reason) => id === request.current && setError(describeError(reason)))
        .finally(() => id === request.current && setLoading(false));
    }, 250);
    return () => clearTimeout(timer);
  }, [overview?.exists, overview?.root, spec, page, tick]);

  // Audio measurements: read the saved status once, and follow a running analysis until it ends.
  const loadAnalysis = useCallback(async () => {
    try {
      const status = await apiJson<AnalysisStatus>('/api/fish/dataset/analyze');
      setAnalysis(status?.total !== undefined ? status : null);
      return status;
    } catch {
      return null;
    }
  }, []);

  useEffect(() => {
    if (overview?.exists) void loadAnalysis();
  }, [overview?.exists, loadAnalysis]);

  useEffect(() => {
    if (!analysis?.running) return;
    const timer = setInterval(() => {
      void loadAnalysis().then((s) => {
        if (s && !s.running) setTick((n) => n + 1); // numbers are in: refresh the list and charts
      });
    }, 1000);
    return () => clearInterval(timer);
  }, [analysis?.running, loadAnalysis]);

  const startAnalysis = useCallback(async () => {
    setError('');
    try {
      const status = await apiJson<AnalysisStatus>('/api/fish/dataset/analyze', { method: 'POST' });
      setAnalysis(status);
      if (!status.running) setTick((n) => n + 1);
    } catch (reason) {
      setError(describeError(reason));
    }
  }, []);

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

  /** Leave out (or put back) every clip the current filters match, not just the visible page. */
  const setMatchedExcluded = useCallback(
    async (exclude: boolean) => {
      setError('');
      try {
        const keys: string[] = [];
        for (let offset = 0; offset < total; offset += 500) {
          const r = await apiJson<{ lines: DatasetLine[] }>(
            `/api/fish/dataset/lines?offset=${offset}&limit=500`,
            { method: 'POST', body: JSON.stringify(spec) },
          );
          keys.push(...r.lines.map((l) => l.key));
        }
        setExcluded((current) => {
          const next = new Set(current);
          for (const key of keys) {
            if (exclude) next.add(key);
            else next.delete(key);
          }
          return next;
        });
      } catch (reason) {
        setError(describeError(reason));
      }
    },
    [spec, total],
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
    setOnlyVoice: (id: string | null) => setVoices(id ? new Set([id]) : new Set()),
    clearVoices: () => setVoices(new Set()),
    filters,
    setFilters,
    patchFilters: (patch: Partial<DatasetFilters>) => setFilters((current) => ({ ...current, ...patch })),
    excluded,
    toggleExcluded,
    clearExcluded: () => setExcluded(new Set()),
    setMatchedExcluded,
    page,
    setPage,
    lines,
    facets,
    total,
    seconds,
    selected,
    selectedSeconds,
    loading,
    analysis,
    startAnalysis,
    refresh,
    saveRoot,
    runExport,
  };
}
